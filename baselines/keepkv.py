"""KeepKV (Tian et al., AAAI 2026, arXiv 2504.09936 v2) -- matched-protocol port.

No public code exists; this follows the paper's Algorithm 1 and Eqs. 3, 6-8.

Published operator
  * Electoral votes: every cache entry carries p (init 1). Attention at decode is
        o_t = sum_i p_i s_i^t v_i / sum_i p_i s_i^t,   s_i^t = exp(q_t . k_i / sqrt d)   (Eqs. 3, 6)
    i.e. a per-slot logit bias log p_i.
  * Candidates: U = cosineSimilarity(K_e, K_c) between to-be-evicted and retained keys;
    each k_e picks k_c = argmax_c U_{e,c}, merged only if U_{e,c} > T = 0.8.          (Alg. 1)
  * ZIP-Merging with (EMA-predicted) unnormalised scores s_hat:                       (Alg. 1 / Eq. 7)
        w_e = p_e s_e,  w_c = p_c s_c
        k_r = ln((w_e + w_c)/(p_e + p_c)) / (w_e ln s_e + w_c ln s_c) * (w_e k_e + w_c k_c)
        v_r = (w_e v_e + w_c v_c) / (w_e + w_c)
        p_r = p_e + p_c
    which makes p_r exp(q . k_r / sqrt d) = w_e + w_c exactly for the query the scores
    came from: zero perturbation at that query.
  * EMA prediction (Eq. 8), at the end of prefill t = L:
        s_hat^t = S^t / (1 - alpha^t),   S^t = sum_{k=t-w}^{t} (1-alpha) alpha^{t-k} s^k.

Port choices where the paper is silent (each recorded in docs/compensated_ports.md):
  * alpha and w are not stated: alpha = 0.9, w = 32 (the protocol's observation window).
  * GQA is not discussed: s_hat for a KV head is the mean of s over the query heads that
    share it, votes are per KV head, and the bias log p is repeated to those query heads.
  * The to-be-evicted set is the pooled-SnapKV eviction of the matched protocol (the paper
    uses PyramidInfer's allocation, which this framework does not implement); the retained
    set K_c is the whole retained cache, sink and recent window included.
  * Several evicted tokens choosing the same retained entry are merged in position order;
    after each merge the entry's state is (k_r, v_r, p_r, s_hat_r) with
    s_hat_r = (w_e + w_c)/p_r, the score that makes the merge perturbation-free.
  * An evicted token with no retained key above T is evicted (Algorithm 1 does not say).
  * `keys="fixed"` is a sensitivity, not the published operator: it applies the identical
    candidates, ZIP weights, value mixture and votes but leaves the retained key in place,
    which separates the operator's two channels exactly.
  * The key scale lam = ln((w_e+w_c)/(p_e+p_c)) / (pi_e ln s_e + pi_c ln s_c) (pi = w/(w_e+w_c))
    is applied as written, negative values included (Algorithm 1 does not restrict its sign and a
    negative scale still satisfies the mass identity). Two numerical guards skip a merge, and both
    are counted: a near-zero denominator |pi_e ln s_e + pi_c ln s_c| < 1e-6 (`skip_undefined`; this
    includes 0/0 but also tiny finite denominators), and a merged key that is not finite in the cache
    dtype (`skip_nonfinite`). `lam_policy` exposes two sensitivities that are NOT the published
    operator: "positive" (skip lam <= 0) and "cap:X" (skip |lam| > X).
  * All scores are handled in log space (ln s = q . k / sqrt d), since s itself overflows.
  * Merging happens once, at the end of prefill; decode maintenance is the protocol's.
  * `selector` (2026-09-19, reviewer request): which eviction the merge runs on. `"snapkv"`
    (default) is the port above, the protocol's pooled SnapKV. `"selkv"` keeps SelKV's own set
    (Eqs. 1-2 of baselines/selkv.py, via SelKVSelectorMixin: per-head top-m_s by window attention x
    value norm, average-pooled, union, trim by mean score, and SelKV's decode-time maintenance), so
    `keepkv;selector=selkv;merge=false` is bit-identical to `selkv;selector=selkv;merge=false`.
    KeepKV's own rules are unchanged on that base: the candidate set K_c is the whole shared
    retained cache (not SelKV's head-specific S_h), scores are the EMA of Eq. 8, threshold T = 0.8.
"""
import math

import torch
import torch.nn.functional as F

from baselines.compensated_base import BiasCompensatedCache
from baselines.selkv import SelKVSelectorMixin


class KeepKVCache(SelKVSelectorMixin, BiasCompensatedCache):

    def __init__(self, merge_threshold: float = 0.8, ema_alpha: float = 0.9, ema_window: int = 32,
                 lam_policy: str = "as_written", targets: str = "retained", votes_bias=True,
                 keys: str = "moved", selector: str = "snapkv", selkv_window: int = 32,
                 selkv_pool: int = 5, **kwargs):
        self.selector = str(selector)
        if self.selector not in ("snapkv", "selkv"):
            raise ValueError(f"selector must be 'snapkv' or 'selkv', got {selector!r}")
        if self.selector == "selkv":
            kwargs["pool_kernel"] = 0                        # SelKV's own average-pooled score (selkv.py)
            kwargs["observation_window"] = int(selkv_window)
        else:
            kwargs.setdefault("observation_window", 32)      # the protocol's pooled SnapKV base
            kwargs.setdefault("pool_kernel", 7)
        super().__init__(**kwargs)
        self._selkv_init(selkv_window, selkv_pool, bucket=32, need_bucket=False)
        self.merge_threshold = float(merge_threshold)
        self.ema_alpha = float(ema_alpha)
        self.ema_window = int(ema_window)
        self.lam_policy = str(lam_policy)
        if not (self.lam_policy in ("as_written", "positive") or self.lam_policy.startswith("cap:")):
            raise ValueError(f"lam_policy must be as_written, positive or cap:X, got {lam_policy!r}")
        # Eq. 6's electoral votes enter the merge weights AND the decode logits. Switching the
        # logit channel off (a sensitivity, not the published operator) separates the two: what
        # the moved keys cost, and what the mass compensation buys back.
        self.votes_bias = votes_bias if isinstance(votes_bias, bool) else str(votes_bias).lower() in ("1", "true", "yes")
        # "moved" is Algorithm 1; "fixed" keeps the retained key and applies only the value
        # mixture and the votes, which separates the two channels of the operator.
        self.keys = str(keys)
        if self.keys not in ("moved", "fixed"):
            raise ValueError(f"keys must be moved or fixed, got {keys!r}")
        self.targets = str(targets)                            # "retained" (paper) or "middle" (sensitivity)
        if self.targets not in ("retained", "middle"):
            raise ValueError(f"targets must be retained or middle, got {targets!r}")
        self._finalizing = False
        # evaluator pre-hook stores the last (w+1) post-RoPE prefill queries here
        self.wants_window_queries = self.ema_window + 1
        self.window_queries = {}

    # ------------------------------------------------------------------ selector plumbing
    # With selector="snapkv" the pending object is the window-query tensor, as before. With
    # selector="selkv" it is {"q": window queries, "sel": SelKV's prefill statistics}, or None at
    # decode time, so _keep_middle can run SelKV's Eqs. 1-2 and _merge_layer KeepKV's Alg. 1.
    def reduce_attention(self, attn_weights):
        scores = super().reduce_attention(attn_weights)
        if self.selector != "selkv":
            return scores
        return self._selkv_reduce(scores, attn_weights, strict=True)

    def _consume_attention_scores(self, layer_idx):
        if self.selector == "selkv":
            self._selkv_split_pending(layer_idx)
        return super()._consume_attention_scores(layer_idx)

    def _take_pending(self, layer_idx):
        q = self.window_queries.pop(layer_idx, None)
        if self.selector != "selkv":
            return q
        sel = self._pending.pop(layer_idx, None)
        if q is None and sel is None:
            return None
        return {"q": q, "sel": sel}

    @staticmethod
    def _unpack(extra):
        if isinstance(extra, dict):
            return extra.get("q"), extra.get("sel")
        return extra, None

    def _keep_middle(self, layer_idx, mid, budget, ms, me, extra):
        if self.selector != "selkv":
            return super()._keep_middle(layer_idx, mid, budget, ms, me, extra)
        _, sel = self._unpack(extra)
        return self._selkv_keep_middle(layer_idx, mid, budget, ms, me, sel)

    def _after_selection(self, layer_idx):
        self._head_sets.pop(layer_idx, None)

    def _require_pending(self, layer_idx, extra):
        if not self._finalizing:
            return
        q, sel = self._unpack(extra)
        if q is None:
            raise RuntimeError(f"KeepKV: layer {layer_idx} evicts at prefill but no prefill queries were "
                               f"captured -- the evaluator pre-hook did not run or position_embeddings were "
                               f"absent; refusing to silently fall back to plain eviction")
        if self.selector == "selkv" and sel is None:
            raise RuntimeError(f"KeepKV(selector=selkv): layer {layer_idx} evicts at prefill but no prefill "
                               f"attention statistics were captured; refusing to silently fall back")

    def on_prefill_end(self):
        self._finalizing = True
        try:
            super().on_prefill_end()
        finally:
            self._finalizing = False

    # ------------------------------------------------------------------ Eq. 8
    @staticmethod
    def log_ema_scores(q, k, alpha, seq_len):
        """ln s_hat for every cache entry, per KV head.
        q: (qh, W, d) queries of the last W prefill positions; k: (kv, L, d) keys.
        s^k = exp(q_k . k_i / sqrt d) averaged over the query heads of each KV head,
        causal (a query does not score keys after it), EMA weights (1-alpha) alpha^(L-1-pos)
        with the paper's bias correction 1/(1 - alpha^L). Returns (kv, L) float32."""
        qh, W, d = q.shape
        kv, L, _ = k.shape
        g = qh // kv
        qpos = torch.arange(L - W, L, device=k.device)
        back = (L - 1 - qpos).to(torch.float32)
        logw = math.log(1.0 - alpha) + back * math.log(alpha)                    # (W,)
        corr = -math.log1p(-(alpha ** seq_len)) if alpha ** seq_len < 1 else 0.0
        mask = torch.arange(L, device=k.device)[None, None, :] > qpos[None, :, None]
        out = torch.empty(kv, L, dtype=torch.float32, device=k.device)
        for h in range(kv):
            lg = q[h * g:(h + 1) * g].float() @ k[h].float().transpose(0, 1) / math.sqrt(d)   # (g, W, L)
            lg = lg.masked_fill(mask, float("-inf")) + logw[None, :, None] - math.log(g)
            out[h] = corr + torch.logsumexp(lg.reshape(-1, L), dim=0)
        return out

    # ------------------------------------------------------------------ Alg. 1
    @torch.no_grad()
    def _merge_layer(self, layer_idx, keep_mid, ms, me, extra):
        q, _ = self._unpack(extra)
        k, v = self._get_existing_cache(layer_idx)
        b, kv, L, d = k.shape
        if b != 1:
            raise NotImplementedError("KeepKVCache assumes batch size 1")
        dev = k.device
        qh = q.shape[1]
        self._gqa_groups = qh // kv
        keep = torch.zeros(L, dtype=torch.bool, device=dev)
        keep[:ms] = True
        keep[me:] = True
        keep[ms + keep_mid.to(dev)] = True
        ev = torch.nonzero(~keep).flatten()
        tgt = keep.clone()
        if self.targets == "middle":
            tgt[:ms] = False
            tgt[me:] = False
        rc = torch.nonzero(tgt).flatten()
        if ev.numel() == 0 or rc.numel() == 0:
            return
        kf, vf = k[0].float(), v[0].float()
        ln_s = self.log_ema_scores(q[0], kf, self.ema_alpha, L)                 # (kv, L)
        newk, newv = kf.clone(), vf.clone()
        votes = torch.ones(kv, L, dtype=torch.float32, device=dev)
        ln_s_cur = ln_s.clone()
        kn = F.normalize(kf, dim=-1)
        st = self.merge_stats
        # Counters are accumulated on the device and read back once per layer: a host
        # synchronisation inside the per-rank loop costs more than the merge itself
        # (it made the 8k LongBench run three times slower than its own base).
        acc = torch.zeros(9, dtype=torch.float32, device=dev)
        lam_seen, good_seen = [], []
        for h in range(kv):
            U = kn[h, ev] @ kn[h, rc].transpose(0, 1)                           # (E, R)
            best, arg = U.max(dim=1)
            ok = best > self.merge_threshold
            st["evicted"] += ev.numel()
            e_idx = ev[ok]
            st["candidates"] += e_idx.numel()
            if e_idx.numel() == 0:
                continue
            c_idx = rc[arg[ok]]
            order = torch.argsort(c_idx * L + e_idx)                             # group by target, then position
            e_idx, c_idx = e_idx[order], c_idx[order]
            first = torch.ones_like(c_idx, dtype=torch.bool)
            first[1:] = c_idx[1:] != c_idx[:-1]
            grp_start = torch.cumsum(first.long(), 0) - 1
            starts = torch.nonzero(first).flatten()
            rank = torch.arange(c_idx.numel(), device=dev) - starts[grp_start]
            # rank r is a slice of this permutation; the slice bounds are the only read-back
            perm = torch.argsort(rank, stable=True)
            counts = torch.bincount(rank).tolist()
            off = 0
            for cnt in counts:
                sl = perm[off:off + cnt]
                off += cnt
                e, c = e_idx[sl], c_idx[sl]
                p_c = votes[h, c]
                ln_we = ln_s[h, e]                                               # p_e = 1
                ln_wc = torch.log(p_c) + ln_s_cur[h, c]
                ln_W = torch.logaddexp(ln_we, ln_wc)
                pi_e = torch.exp(ln_we - ln_W)
                pi_c = 1.0 - pi_e
                ln_sr = ln_W - torch.log(p_c + 1.0)
                den = pi_e * ln_s[h, e] + pi_c * ln_s_cur[h, c]
                lam = ln_sr / torch.where(den.abs() < 1e-6, torch.ones_like(den), den)
                defined = (den.abs() >= 1e-6) & torch.isfinite(lam)
                if self.lam_policy == "positive":
                    pol = lam > 0
                elif self.lam_policy.startswith("cap:"):
                    pol = lam.abs() <= float(self.lam_policy.split(":", 1)[1])
                else:
                    pol = torch.ones_like(defined)
                kr = lam[:, None] * (pi_e[:, None] * newk[h, e] + pi_c[:, None] * newk[h, c])
                fin = torch.isfinite(kr.to(k.dtype)).all(dim=1)
                good = defined & pol & fin
                acc += torch.stack([(~defined).sum(), (defined & ~pol).sum(),
                                    (defined & pol & ~fin).sum(), good.sum(),
                                    (good & (lam < 0)).sum(), (good & (lam.abs() > 2)).sum(),
                                    (good & (lam.abs() > 10)).sum(),
                                    (good & (c < ms)).sum(), (good & (c >= me)).sum()]).float()
                lam_seen.append(lam)
                good_seen.append(good)
                gk = good[:, None]
                newv[h, c] = torch.where(gk, pi_e[:, None] * newv[h, e] + pi_c[:, None] * newv[h, c], newv[h, c])
                if self.keys == "moved":
                    newk[h, c] = torch.where(gk, kr, newk[h, c])
                votes[h, c] = p_c + good.float()
                ln_s_cur[h, c] = torch.where(good, ln_sr, ln_s_cur[h, c])
        for key, val in zip(("skip_undefined", "skip_policy", "skip_nonfinite", "merged",
                             "lam_negative", "lam_abs_gt2", "lam_abs_gt10",
                             "merged_into_sink", "merged_into_recent"), acc.tolist()):
            st[key] += val
        # where the votes pile up: a slot merged into n times carries a log(n+1) logit bias
        st["votes_max"] = max(st["votes_max"], float(votes.max()))
        st["votes_layers"] += 1
        st["votes_gt4_sum"] += float((votes > 4).sum())
        if lam_seen:
            lam_all = torch.cat(lam_seen)[torch.cat(good_seen)]
            if lam_all.numel():
                qs = torch.quantile(lam_all.float().cpu()[:200000], torch.tensor([0.01, 0.5, 0.99]))
                st["lam_p01_sum"] += float(qs[0]); st["lam_p50_sum"] += float(qs[1])
                st["lam_p99_sum"] += float(qs[2]); st["lam_layers"] += 1
                st["lam_absmax_seen"] = max(st["lam_absmax_seen"], float(lam_all.abs().max()))
        self._replace_existing_cache(layer_idx, newk.unsqueeze(0).to(k.dtype), newv.unsqueeze(0).to(v.dtype))
        self._set_bias(layer_idx, torch.log(votes).unsqueeze(0) if self.votes_bias
                       else torch.zeros_like(votes).unsqueeze(0))
