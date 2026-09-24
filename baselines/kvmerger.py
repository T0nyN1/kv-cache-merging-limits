"""KVMerger (Wang et al., 2024, arXiv 2407.08454) — matched-protocol port.

Published operator (Section 4 of the paper):

  * merging-set identification: consecutive tokens whose key states have cosine
    similarity above delta (delta = 0.75 in the paper) form a set; tokens with
    the top aggregated attention scores and the most recent tokens are kept
    unmerged;
  * pivot: the member of the set with the largest aggregated attention score,
        k_p = argmax_{i in S} a_i;
  * Gaussian-kernel weighted merge of the whole set into the pivot, for keys
    AND values:
        g_pi = exp(-||k_p - k_i||^2 / (2 sigma^2)),  w_i = g_pi / sum_j g_pj,
        k_p <- w_p k_p + sum_{i != p} w_i k_i   (values identically).

The paper reaches its 50 % / 35 % budgets by merging (the retained cache is one
slot per set); it was never run at the 2-5 % budgets used here, where sets of
20-50 consecutive similar tokens do not exist at delta = 0.75. Under the matched
protocol the selection is therefore the base method's (accumulated attention
for the H2O-style variant, last-window attention for the SnapKV-style variant)
and KVMerger's operator acts on the EVICTED tokens: every evicted token that
belongs to a consecutive similar-key set containing a retained token is merged
into that set's highest-scoring retained member (the published pivot rule
restricted to slots that exist); evicted tokens in sets without a retained
member are evicted, as the budget requires. Keys and values are merged with
the kernel weights; the kernel bandwidth is the RMS distance of the set's
members to the pivot (the paper's sigma is defined through the weights it
produces, so a fixed choice is needed; sigma = RMS distance makes the kernel
scale-free and is the standard choice).

Set identification uses the head-averaged adjacent key cosine (the published
code keeps one token layout shared across heads); weights are per head.
Run with per_head=false. `merge_threshold` and `merge_keys` are exposed for
the sensitivity ablation.
"""

import json
import os

import torch
import torch.nn.functional as F

from baselines.snapkv import SnapKVCache


class KVMergerCache(SnapKVCache):
    """`merge_stats` counts sets, merged and evicted tokens over the cache's life. They are
    written to MERGE_STATS_PATH (one JSON line per call, `stage` = "prefill" after the
    prefill-end compression and "decode<n>" every 16 decode steps, cumulative) so a run's
    merge activity is recoverable: a run whose counters stay at zero is plain eviction."""

    def _emit_merge_stats(self, stage):
        path = os.environ.get("MERGE_STATS_PATH")
        stats = {k: float(v) for k, v in sorted(self.merge_stats.items())}
        if stage == "prefill":
            print(f"[KVMergerCache] " + " ".join(f"{k} {v:.4g}" for k, v in stats.items()), flush=True)
        if not path:
            return
        rec = {"cache": type(self).__name__, "method": os.environ.get("OTKV_METHOD", ""),
               "merge": True, "stage": stage, "stats": stats}
        try:
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    def on_prefill_end(self):
        super().on_prefill_end()
        self._emit_merge_stats("prefill")

    def on_decode_step(self, key_states, value_states, layer_idx, cache_kwargs):
        out = super().on_decode_step(key_states, value_states, layer_idx, cache_kwargs)
        if layer_idx == 0 and self._decode_step > 0 and self._decode_step % 16 == 0:
            self._emit_merge_stats(f"decode{self._decode_step}")
        return out

    def __init__(self, merge_threshold: float = 0.75, merge_keys: bool = True,
                 merge_values: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.merge_threshold = float(merge_threshold)
        self.merge_keys = bool(merge_keys)
        self.merge_values = bool(merge_values)
        self.merge_stats = {"sets": 0, "merged_tokens": 0, "evicted_tokens": 0}

    def _select_prune_middle(self, layer_idx: int, total_tokens: int):
        k, _ = self._get_existing_cache(layer_idx)
        if k is None:
            return super()._select_prune_middle(layer_idx, total_tokens)
        seq_len = k.shape[-2]
        ms = self.sink_size
        me = max(self.sink_size, seq_len - self.recent_size)
        if ms >= me:
            return super()._select_prune_middle(layer_idx, total_tokens)

        scores = self.attention_scores[layer_idx]
        middle_scores = scores[..., ms:me]
        budget = self.get_middle_budget(layer_idx, total_tokens)
        if budget >= middle_scores.shape[-1] or middle_scores.dim() != 1:
            # nothing to evict, or per-head bookkeeping (not supported by the
            # published operator, which shares one layout across heads)
            return super()._select_prune_middle(layer_idx, total_tokens)

        keep = self._middle_keep_indices(self._rank_scores(middle_scores), budget)
        Lm = middle_scores.shape[-1]
        kept = torch.zeros(Lm, dtype=torch.bool, device=middle_scores.device)
        if keep.numel():
            kept[keep] = True
        evicted = ~kept
        n_ev = int(evicted.sum())
        self.merge_stats["evicted_tokens"] += n_ev

        if n_ev and keep.numel() and Lm >= 2:
            middle_k, middle_v = self.get_middle_cache(layer_idx)     # (b, h, Lm, d)
            kf = middle_k.float()
            kn = F.normalize(kf, dim=-1)
            cos_adj = (kn[..., :-1, :] * kn[..., 1:, :]).sum(-1).mean(dim=(0, 1))   # (Lm-1,)
            link = cos_adj > self.merge_threshold
            # group id of each token: a new group starts wherever the link is broken
            gid = torch.zeros(Lm, dtype=torch.long, device=kf.device)
            gid[1:] = torch.cumsum((~link).long(), dim=0)
            n_groups = int(gid[-1]) + 1
            # pivot per group: the retained member with the largest score
            neg = torch.full((Lm,), float("-inf"), device=kf.device)
            sc_kept = torch.where(kept, middle_scores.float(), neg)
            best = torch.full((n_groups,), float("-inf"), device=kf.device)
            best = best.scatter_reduce(0, gid, sc_kept, reduce="amax", include_self=True)
            has_pivot = torch.isfinite(best)
            is_pivot = kept & (sc_kept == best[gid])
            # resolve ties deterministically: first pivot in the group
            pivot_of_group = torch.full((n_groups,), -1, dtype=torch.long, device=kf.device)
            idx = torch.nonzero(is_pivot).flatten()
            # later assignments overwrite earlier ones; reverse so the first wins
            pivot_of_group[gid[idx.flip(0)]] = idx.flip(0)
            # evicted tokens that can be merged: their group has a pivot and the
            # group has more than one member (otherwise it is a singleton)
            mergeable = evicted & has_pivot[gid]
            e_idx = torch.nonzero(mergeable).flatten()
            if e_idx.numel():
                p_idx = pivot_of_group[gid[e_idx]]
                # per-head Gaussian kernel around the pivot; bandwidth = RMS
                # distance of the group's members (pivot + merged) to the pivot
                d2 = ((kf[..., e_idx, :] - kf[..., p_idx, :]) ** 2).sum(-1)      # (b, h, n_e)
                cnt = torch.zeros(n_groups, device=kf.device).index_add_(
                    0, gid[e_idx], torch.ones_like(d2[0, 0]))
                b_, h_ = d2.shape[0], d2.shape[1]
                sum_d2 = torch.zeros(b_, h_, n_groups, device=kf.device).index_add_(2, gid[e_idx], d2)
                sig2 = (sum_d2 / (cnt + 1.0).clamp_min(1.0))                        # pivot contributes 0
                sig2 = sig2.clamp_min(1e-6)
                g = torch.exp(-d2 / (2.0 * sig2[..., gid[e_idx]]))                # (b, h, n_e)
                gsum = torch.zeros(b_, h_, n_groups, device=kf.device).index_add_(2, gid[e_idx], g)
                gsum = gsum + 1.0                                                  # pivot's own g = 1
                w_e = g / gsum[..., gid[e_idx]]                                    # (b, h, n_e)
                w_p = 1.0 / gsum                                                   # (b, h, n_groups)
                grp = gid[e_idx]
                if self.merge_keys:
                    acc = torch.zeros(b_, h_, n_groups, kf.shape[-1], device=kf.device)
                    acc.index_add_(2, grp, w_e.unsqueeze(-1) * kf[..., e_idx, :])
                    pv = pivot_of_group[has_pivot & (cnt > 0)]
                    gp = torch.nonzero(has_pivot & (cnt > 0)).flatten()
                    new_k = kf.clone()
                    new_k[..., pv, :] = w_p[..., gp].unsqueeze(-1) * kf[..., pv, :] + acc[..., gp, :]
                else:
                    new_k = kf
                if self.merge_values:
                    vf = middle_v.float()
                    accv = torch.zeros(b_, h_, n_groups, vf.shape[-1], device=kf.device)
                    accv.index_add_(2, grp, w_e.unsqueeze(-1) * vf[..., e_idx, :])
                    pv = pivot_of_group[has_pivot & (cnt > 0)]
                    gp = torch.nonzero(has_pivot & (cnt > 0)).flatten()
                    new_v = vf.clone()
                    new_v[..., pv, :] = w_p[..., gp].unsqueeze(-1) * vf[..., pv, :] + accv[..., gp, :]
                else:
                    new_v = middle_v.float()
                self.replace_middle_cache(layer_idx, new_k.to(middle_k.dtype), new_v.to(middle_v.dtype))
                self.merge_stats["merged_tokens"] += int(e_idx.numel())
                self.merge_stats["sets"] += int((cnt > 0).sum())

        # scores untouched -> the parent recomputes the identical keep set and
        # performs the prune + score bookkeeping
        super()._select_prune_middle(layer_idx, total_tokens)
