"""OT-KV v8 as a deployable cache: the fixed-key pair merge of core/otkv_v8.py
on top of the pooled-SnapKV retained set (baselines/snapkv.py), with the
per-slot logit bias injected at decode through the bias-aware attention
pre-hook (evaluation/models/wrapper.py, the path built for core/consolidate.py).

Protocol (docs/otkv_v8_contract.md, experiments/v8_offline.py): at prefill end,
per layer and KV head, the last `row_window` prefill queries' attention rows
(all query heads of the GQA group, position-major) are split by time into fit
rows (first `fit_rows` positions) and select rows (next `sel_rows`); each
evicted middle token may be merged into one of its `max_cand` nearest retained
middle neighbours within `max_dist` positions (keys unchanged), with (c, theta)
fitted jointly on the full attention output of the fit rows (core._fit_pairs
with joint_c), edges kept only where they lower the output error in every one
of `n_blocks` contiguous time blocks, receivers assigned by min-cost matching
with a drop option, and the plan applied at the gamma that most reduces the
error on the select rows if that reduction is >= min_gain with no query head
worse. Candidate positions are restricted to < L - row_window so that every
calibration row sees them. Everything is computed from the stored attention
rows (softmax over the whole prefill cache), so no query vectors are needed:
the baseline weights over A are the rows renormalised over A, and a bias b_j
multiplies slot j's weight by e^{b_j}.

Decode: the bias vector rides along the cache (pruned with the same keep set at
every maintenance step, padded by one for each new token) and is added to the
logits of every decode query through the pre-hook; new tokens are never merged.
"""
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from baselines.snapkv import SnapKVCache
from core.otkv_v8 import _fit_pairs, solve_partial_transport, EPS, KEY_ATOL


def candidate_pairs_fast(keep: np.ndarray, protect: np.ndarray, max_dist: int, max_cand: int
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised locality candidates: for every evicted, unprotected position i
    the up-to-`max_cand` nearest retained, unprotected positions j with
    |i - j| <= max_dist (ties broken towards the earlier position, as in
    core.otkv_v8.build_sparse_costs). Returns (i_idx, j_idx) int64 arrays."""
    recv = np.nonzero(keep & ~protect)[0]
    src_ok = (~keep) & (~protect)
    if recv.size == 0 or not src_ok.any():
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    offs = np.arange(-max_dist, max_dist + 1)
    offs = offs[offs != 0]
    cand_i = (recv[:, None] + offs[None, :]).reshape(-1)          # candidate sources per receiver
    cand_j = np.repeat(recv, offs.size)
    ok = (cand_i >= 0) & (cand_i < keep.size)
    cand_i, cand_j = cand_i[ok], cand_j[ok]
    ok = src_ok[cand_i]
    cand_i, cand_j = cand_i[ok], cand_j[ok]
    if cand_i.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    dist = np.abs(cand_i - cand_j)
    order = np.lexsort((cand_j, dist, cand_i))                    # by source, then distance, then position
    cand_i, cand_j = cand_i[order], cand_j[order]
    first = np.r_[0, np.nonzero(np.diff(cand_i))[0] + 1]
    rank = np.arange(cand_i.size) - np.repeat(first, np.diff(np.r_[first, cand_i.size]))
    sel = rank < max_cand
    return cand_i[sel].astype(np.int64), cand_j[sel].astype(np.int64)


class OTKVv8Cache(SnapKVCache):
    """Pooled-SnapKV selection + the v8 fixed-key compensated pair merge."""

    consolidate = True          # duck-typed flag read by the bias-aware pre-hook

    def __init__(self, row_window: int = 64, fit_rows: int = 32, sel_rows: int = 16,
                 max_dist: int = 8, max_cand: int = 4, n_blocks: int = 4, c_max: float = 4.0,
                 joint_c: int = 9, min_gain: float = 0.05, gammas=(0.25, 0.5, 1.0),
                 solver: str = "matching", merge: bool = True, force_theta=None, force_c=None, **kwargs):
        super().__init__(**kwargs)
        if self.per_head:
            raise NotImplementedError("OTKVv8Cache shares one token layout across heads: run with per_head=false")
        self.row_window = int(row_window)
        self.fit_rows = int(fit_rows)
        self.sel_rows = int(sel_rows)
        self.max_dist = int(max_dist)
        self.max_cand = int(max_cand)
        self.n_blocks = int(n_blocks)
        self.c_max = float(c_max)
        self.joint_c = int(joint_c)
        # force_theta=0 is the bias-only ablation: the receiver keeps its own
        # value and only its attention mass is re-fitted (docs/otkv_v8_contract.md).
        self.force_theta = None if force_theta in (None, "", "none") else float(force_theta)
        # force_c=1 is the no-bias ablation: the mixture is refitted but the slot
        # carries no logit bias, so only the value channel acts.
        self.force_c = None if force_c in (None, "", "none") else float(force_c)
        self.min_gain = float(min_gain)
        self.gammas = tuple(float(g) for g in (gammas if not isinstance(gammas, str)
                                                else gammas.split("/")))
        self.solver = str(solver)
        self.merge = bool(merge) if not isinstance(merge, str) else merge.lower() in ("1", "true", "yes")
        self._rows: Dict[int, torch.Tensor] = {}
        self.logit_bias: Dict[int, torch.Tensor] = {}
        self._has_bias: Dict[int, bool] = {}
        self._gqa_groups = 1
        self._prehook_seen = False
        self.decode_steps: Dict[int, int] = {}
        self.merge_stats = {"heads": 0, "heads_applied": 0, "pairs_proposed": 0, "pairs_applied": 0,
                            "gamma_sum": 0.0, "select_improvement_sum": 0.0, "candidates": 0, "edges_neg": 0,
                            "t_setup": 0.0, "t_cand": 0.0, "t_fit": 0.0, "t_cost": 0.0, "t_match": 0.0,
                            "t_gamma": 0.0, "t_write": 0.0, "t_layer": 0.0}

    # ------------------------------------------------------------------ rows
    def reduce_attention(self, attn_weights: torch.Tensor) -> torch.Tensor:
        a = attn_weights.detach()
        scores = super().reduce_attention(a)                       # (b, qh, L)
        R = self.row_window
        if not self.merge or a.dim() != 4 or a.shape[-2] < R or not torch.is_tensor(scores) or scores.dim() != 3:
            return scores
        rows = a[..., -R:, :].float()                             # (b, qh, R, L)
        return torch.cat([scores.float().unsqueeze(-2), rows], dim=-2)

    def _consume_attention_scores(self, layer_idx: int):
        x = self.current_attention_scores.get(layer_idx)
        R = self.row_window
        if x is not None and torch.is_tensor(x) and x.dim() == 4 and x.shape[-2] == 1 + R:
            self._rows[layer_idx] = x[..., 1:, :]
            self.current_attention_scores[layer_idx] = x[..., 0, :]
        return super()._consume_attention_scores(layer_idx)

    # ------------------------------------------------------------------ prefill-end merge
    def _keep_indices(self, layer_idx: int, total_tokens: int):
        k, _ = self._get_existing_cache(layer_idx)
        if k is None:
            return None, None, None
        seq_len = k.shape[-2]
        ms = self.sink_size
        me = max(self.sink_size, seq_len - self.recent_size)
        if ms >= me:
            return None, ms, me
        scores = self.attention_scores[layer_idx]
        middle_scores = scores[..., ms:me]
        budget = self.get_middle_budget(layer_idx, total_tokens)
        if budget >= middle_scores.shape[-1]:
            return None, ms, me
        keep = self._middle_keep_indices(self._rank_scores(middle_scores), budget)
        return keep, ms, me

    def _select_prune_middle(self, layer_idx: int, total_tokens: int):
        rows = self._rows.pop(layer_idx, None)
        keep, ms, me = self._keep_indices(layer_idx, total_tokens)
        if keep is None:
            return
        if rows is not None and self.merge:
            self._merge_layer(layer_idx, rows, keep, ms, me)
        super()._select_prune_middle(layer_idx, total_tokens)
        bias = self.logit_bias.get(layer_idx)
        if bias is not None:
            bm = bias[..., ms:me].index_select(-1, keep.to(bias.device))
            self.logit_bias[layer_idx] = torch.cat([bias[..., :ms], bm, bias[..., me:]], dim=-1)

    def _sync(self, device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    @torch.no_grad()
    def _merge_layer(self, layer_idx: int, rows: torch.Tensor, keep_mid: torch.Tensor, ms: int, me: int):
        t_layer = time.time()
        st = self.merge_stats
        k, v = self._get_existing_cache(layer_idx)
        b, kv, L, d = v.shape
        if b != 1:
            raise NotImplementedError("OTKVv8Cache assumes batch size 1")
        qh = rows.shape[1]
        g = qh // kv
        self._gqa_groups = g
        R = self.row_window
        if rows.shape[-1] != L or R > L or self.fit_rows + self.sel_rows > R:
            return
        device = v.device
        keep = torch.zeros(L, dtype=torch.bool, device=device)
        keep[:ms] = True
        keep[me:] = True
        keep[ms + keep_mid.to(device)] = True
        protect = torch.zeros(L, dtype=torch.bool, device=device)
        protect[:ms] = True
        protect[me:] = True
        protect[L - R:] = True                                       # every calibration row sees the candidates
        keep_np = keep.cpu().numpy()
        prot_np = protect.cpu().numpy()
        t0 = time.time()
        ii_np, jj_np = candidate_pairs_fast(keep_np, prot_np, self.max_dist, self.max_cand)
        st["t_cand"] += time.time() - t0
        st["candidates"] += int(ii_np.size) * kv
        if ii_np.size == 0:
            return
        ii = torch.as_tensor(ii_np, device=device)
        jj = torch.as_tensor(jj_np, device=device)
        keep_idx = torch.nonzero(keep).flatten()
        slot_of = torch.full((L,), -1, dtype=torch.long, device=device)
        slot_of[keep_idx] = torch.arange(keep_idx.numel(), device=device)
        n_fit, n_sel = self.fit_rows * g, self.sel_rows * g
        v_new = v[0].float().clone()                                 # (kv, L, d)
        bias_full = torch.zeros(1, kv, L, dtype=torch.float32, device=device)
        st["t_setup"] += time.time() - t_layer
        for h in range(kv):
            t0 = time.time()
            a = rows[0, h * g:(h + 1) * g].permute(1, 0, 2).reshape(R * g, L).float()   # position-major rows
            kh, vh = k[0, h].float(), v[0, h].float()
            a_fit, a_sel = a[:n_fit], a[n_fit:n_fit + n_sel]
            o_fit, o_sel = a_fit @ vh, a_sel @ vh
            aA = a_fit[:, keep_idx]
            p_fit = aA / aA.sum(dim=1, keepdim=True).clamp_min(1e-30)
            o0_fit = p_fit @ vh[keep_idx]
            same_k = ((kh[ii] - kh[jj]).abs() <= KEY_ATOL + 1e-5 * kh[jj].abs()).all(dim=1)
            c_all, th_all, err_all, acc_all = [], [], [], []
            for s in range(0, ii.numel(), 512):
                i_c, j_c = ii[s:s + 512], jj[s:s + 512]
                c_, th_, err_, acc_, _ = _fit_pairs(
                    a_fit[:, i_c].T, a_fit[:, j_c].T, p_fit[:, slot_of[j_c]].T, vh[i_c], vh[j_c],
                    o_fit, o0_fit, self.c_max, EPS, same_k[s:s + 512],
                    joint_c=(0 if self.force_c is not None else self.joint_c),
                    force_c=self.force_c, force_theta=self.force_theta)
                c_all.append(c_); th_all.append(th_); err_all.append(err_); acc_all.append(acc_)
            c_e, th_e, err_e, acc_e = torch.cat(c_all), torch.cat(th_all), torch.cat(err_all), torch.cat(acc_all)
            self._sync(device); st["t_fit"] += time.time() - t0; t0 = time.time()
            scale = float((o_fit * o_fit).sum(dim=1).mean()) + EPS
            nb = max(1, min(self.n_blocks, n_fit))
            cost = torch.stack([blk.mean(dim=1) for blk in torch.tensor_split(err_e, nb, dim=1)], dim=1).max(dim=1).values / scale
            ok = acc_e & (cost < 0)
            self.merge_stats["heads"] += 1
            idx = torch.nonzero(ok).flatten().cpu().tolist()          # one sync
            st["t_cost"] += time.time() - t0; t0 = time.time()
            st["edges_neg"] += len(idx)
            if not idx:
                continue
            cost_l = cost[ok].cpu().tolist(); c_l = c_e[ok].cpu().tolist(); th_l = th_e[ok].cpu().tolist()
            edges = [(int(ii_np[e]), int(jj_np[e]), cost_l[n], c_l[n], th_l[n]) for n, e in enumerate(idx)]
            plan = solve_partial_transport(edges, 0, None, method=self.solver)
            st["t_match"] += time.time() - t0; t0 = time.time()
            if not plan:
                continue
            self.merge_stats["pairs_proposed"] += len(plan)
            # gamma on the select rows (rows-based joint evaluation of the whole plan)
            pi = torch.tensor([p[0] for p in plan], device=device)
            pj = torch.tensor([p[1] for p in plan], device=device)
            pc = torch.tensor([p[2] for p in plan], device=device, dtype=torch.float32)
            pt = torch.tensor([p[3] for p in plan], device=device, dtype=torch.float32)
            aS = a_sel[:, keep_idx]
            base = ((aS / aS.sum(1, keepdim=True).clamp_min(1e-30)) @ vh[keep_idx] - o_sel).pow(2).sum(1)
            head_of = torch.arange(n_sel, device=device) % g
            base_mean = base.mean()
            base_head = torch.stack([base[head_of == q].mean() for q in range(g)])
            stats = []
            for gm in self.gammas:
                vg = vh.clone()
                vg[pj] = vh[pj] + gm * pt[:, None] * (vh[pi] - vh[pj])
                bg = torch.zeros(L, device=device)
                bg[pj] = gm * torch.log(pc)
                w = aS * torch.exp(bg[keep_idx])[None, :]
                og = (w / w.sum(1, keepdim=True).clamp_min(1e-30)) @ vg[keep_idx]
                e = (og - o_sel).pow(2).sum(1)
                e_head = torch.stack([e[head_of == q].mean() for q in range(g)])
                stats.append(torch.cat([(1.0 - e.mean() / base_mean.clamp_min(1e-30)).reshape(1),
                                        1.0 - e_head / base_head.clamp_min(1e-30)]))
            stats = torch.stack(stats).cpu().tolist()                 # one sync for all gammas
            st["t_gamma"] += time.time() - t0; t0 = time.time()
            best_g, best_imp = 0.0, 0.0
            for gm, row in zip(self.gammas, stats):
                imp, per = row[0], row[1:]
                if imp >= self.min_gain and min(per) >= -1e-6 and imp > best_imp:
                    best_g, best_imp = gm, imp
            if best_g <= 0.0:
                continue
            v_new[h, pj] = vh[pj] + best_g * pt[:, None] * (vh[pi] - vh[pj])
            bias_full[0, h, pj] = best_g * torch.log(pc)
            self.merge_stats["heads_applied"] += 1
            self.merge_stats["pairs_applied"] += len(plan)
            self.merge_stats["gamma_sum"] += best_g
            self.merge_stats["select_improvement_sum"] += best_imp
        t0 = time.time()
        self._replace_existing_cache(layer_idx, k, v_new.unsqueeze(0).to(v.dtype))
        self.logit_bias[layer_idx] = bias_full
        self._has_bias[layer_idx] = bool((bias_full != 0).any())
        self._sync(device); st["t_write"] += time.time() - t0
        st["t_layer"] += time.time() - t_layer

    # ------------------------------------------------------------------ decode-time bias
    def pre_attention_step(self, layer_idx: int):
        self._prehook_seen = True
        if self.finalize_prefill() and layer_idx == 0:
            st = self.merge_stats
            line = (f"[otkv8] prefill merge: heads {st['heads']} applied {st['heads_applied']} "
                    f"pairs proposed {st['pairs_proposed']} applied {st['pairs_applied']} "
                    f"mean gamma {st['gamma_sum'] / max(1, st['heads_applied']):.2f} "
                    f"mean select gain {st['select_improvement_sum'] / max(1, st['heads_applied']):.3f} | "
                    f"candidates {st['candidates']} edges_neg {st['edges_neg']} | seconds: layer {st['t_layer']:.2f} "
                    f"setup {st['t_setup']:.2f} cand {st['t_cand']:.2f} fit {st['t_fit']:.2f} cost {st['t_cost']:.2f} "
                    f"match {st['t_match']:.2f} gamma {st['t_gamma']:.2f} write {st['t_write']:.2f}")
            print(line, flush=True)
            path = os.environ.get("OTKV8_STATS_PATH")
            if path:
                try:
                    with open(path, "a") as f:
                        f.write(line + "\n")
                except OSError:
                    pass
        self._consume_attention_scores(layer_idx)
        total = self.get_seq_length()
        self._update_budget(total + 1, is_prefill_end=False)
        self._select_prune_middle(layer_idx, total + 1)

    def get_decode_bias(self, layer_idx: int) -> Optional[torch.Tensor]:
        if not self._has_bias.get(layer_idx):
            return None
        bias = self.logit_bias.get(layer_idx)
        if bias is None:
            return None
        x = F.pad(bias, (0, 1))                                      # incoming token: bias 0
        if self._gqa_groups > 1:
            x = x.repeat_interleave(self._gqa_groups, dim=1)
        return x.unsqueeze(2)

    def on_decode_step(self, key_states, value_states, layer_idx, cache_kwargs):
        q_len = key_states.shape[-2]
        self._consume_attention_scores(layer_idx)
        total_after = self.get_seq_length() + q_len
        self._update_budget(total_after, is_prefill_end=False)
        if not self._prehook_seen:
            self._select_prune_middle(layer_idx, total_after)
        self._grow_scores(layer_idx, key_states)
        if layer_idx in self.logit_bias:
            self.logit_bias[layer_idx] = F.pad(self.logit_bias[layer_idx], (0, q_len))
        return key_states, value_states
