"""SelKV (Bouyahiaoui et al., arXiv 2607.16213 v1) -- matched-protocol port.

No public code exists; this follows the paper's Algorithm 1 and Eqs. 1-6
(RoPE repositioning, Eq. 7, is disabled by default in the paper and here).

Published operator
  * Contribution score (Eq. 1): c_{h,j} = sum_{q in W} A_{h,q,j} * ||v_{h,j}||, W = last 32
    positions, average-pooling smoothing with kernel K = 5.
  * Selection (Eq. 2): a recent window is always kept; each KV head keeps its top-m_s tokens;
    on GQA the kept set is the UNION across KV heads, "if the union set exceeds the layer
    budget m, we trim by the mean aggregate score across heads".
  * Routing (Eq. 3): pi_h(j) = argmax_{i in S_h cap B(j)} A_{h,j,i}, B = 32 -- each evicted token
    goes to the kept token of its bucket that it attended to most during prefill.
  * Gate (Eq. 4): g_{h,j} = max(cos(v_{h,j}, v_{h,pi(j)}), 0).
  * Merge (Eq. 5): v~_{h,i} = (a_{h,i} v_{h,i} + sum_{j->i} g_{h,j} a_{h,j} v_{h,j})
                              / (a_{h,i} + sum_{j->i} g_{h,j} a_{h,j}),  a_{h,i} = sum_{q in W} A_{h,q,i}.
    Keys are not merged.
  * Compensation (Eq. 6): R_{h,i} = (a_{h,i} + sum_{j->i} g_{h,j} a_{h,j}) / a_{h,i}; on GQA models R is
    averaged across KV heads and alpha * log R_i (alpha = 0.5) is added to the decode logits.
  * Compression once, after prefill.

Port choices where the paper is silent (each recorded in docs/compensated_ports.md):
  * `selector="selkv"` is SelKV's own rule (Eqs. 1-2); `selector="snapkv"` keeps the pooled-SnapKV
    set of the matched protocol, so the merge can be compared on the same base as the other ports.
  * Budget accounting, the attention sink and the recent window are the protocol's (sink 4,
    recent 10 % of the budget) for every method in the table; SelKV's own m_r = 16 is not used.
  * Buckets are fixed blocks of B positions, B(j) = floor(j / B). An evicted token whose bucket has
    no kept token it attended to (all A = 0, e.g. only later kept tokens) is dropped (counted).
  * Routing honours Eq. 3's head-specific S_h: with `selector="selkv"` head h routes only to
    positions that survived the union-and-trim AND were in head h's own top-m_s, plus the
    protected sink and recent window, which every head keeps. With `selector="snapkv"` there are
    no per-head sets and S_h is the shared retained set.
  * m_s = m - m_r in the paper; with the protocol's protected regions this is the middle budget.
  * On GQA, A for a KV head is summed over its query heads (a constant factor that cancels in the
    argmax, in Eq. 5 and in R).
  * Average pooling uses zero padding ('same' length), applied to the score along the token axis
    before top-k; decode-time maintenance ranks with the same score using head-summed attention.
  * R is not clipped. Eq. 6 is singular when a routed-to kept token has zero window attention
    (possible after bf16 underflow); a_i is floored at 1e-8 for R, and every routed-to target with
    a_i < 1e-8 (zero or small positive) is counted as `floored_attention_targets`.
  * Decode-time maintenance (the protocol keeps the budget during decoding; SelKV itself compresses
    once) ranks with the head-summed score times the head-mean value norm, average-pooled; the
    merge-off pairing of the own selector uses the identical maintenance.
  * Chunked prefill is not supported: a prefill call whose query length differs from the key
    length raises instead of silently disabling the merge.
  * `fallback` (2026-09-19, reviewer request): what happens to an evicted token whose bucket
    S_h cap B(j) holds no kept token it attended to. `"drop"` (default, the port above) evicts it.
    `"global"` applies Eq. 3 over the whole kept set, pi_h(j) = argmax_{i in S_h} A_{h,j,i}, with
    the same gate (Eq. 4), merge (Eq. 5) and compensation (Eq. 6). The full prefill attention
    cannot be held for 8k tokens, so reduce_attention keeps, per KV head and token, the
    `fallback_topk` (default 128) most-attended positions of the head-summed A; the global argmax
    over S_h is exact whenever one of them lies in S_h (the protected sink is in every S_h and is
    the top-attended position of nearly every query), and a token none of whose candidates is
    kept is dropped and counted as `fallback_miss`. Counters: `routed_bucket`, `fallback_routed`,
    `fallback_miss`; `routed` is the total.

The selector (Eqs. 1-2, the prefill statistics and the decode-time maintenance rule) lives in
`SelKVSelectorMixin` so KeepKVCache(selector="selkv") runs the identical selection.
"""
import torch
import torch.nn.functional as F

from baselines.compensated_base import BiasCompensatedCache


def avgpool_tokens(x, kernel):
    """'same'-length average pooling along the last axis (zero padding counted)."""
    if kernel is None or int(kernel) <= 1:
        return x
    kernel = int(kernel)
    shp = x.shape
    flat = x.reshape(-1, 1, shp[-1]).float()
    out = F.avg_pool1d(flat, kernel_size=kernel, stride=1, padding=kernel // 2, count_include_pad=True)
    return out[..., :shp[-1]].reshape(shp).to(x.dtype)


class SelKVSelectorMixin:
    """SelKV's own selection (Eqs. 1-2), its prefill statistics and its decode-time maintenance
    rule, as plain helpers. SelKVCache and KeepKVCache(selector="selkv") call the same functions,
    which is what makes `keepkv;selector=selkv;merge=false` bit-identical to
    `selkv;selector=selkv;merge=false` (checked by experiments/test_compensated_ports.py)."""

    def _selkv_init(self, selkv_window, selkv_pool, bucket, need_bucket):
        self.selkv_window = int(selkv_window)
        self.selkv_pool = int(selkv_pool)
        self.bucket = int(bucket)
        self._selkv_need_bucket = bool(need_bucket)
        self._head_sets = {}                              # layer -> (kv, M) bool, head-specific top-m_s

    def _selkv_kv_heads(self):
        """KV-head count from the cache itself (reduce_attention has no layer index); the hook runs
        after this layer's update(), so layer 0 always holds keys by then."""
        k0, _ = self._get_existing_cache(0)
        if k0 is None or k0.dim() != 4 or k0.numel() == 0:
            raise RuntimeError("SelKV: cannot infer the KV head count before layer 0 is cached")
        return int(k0.shape[1])

    def _selkv_reduce(self, scores, attn_weights, strict, global_topk=0):
        a = attn_weights.detach()
        if a.dim() != 4 or a.shape[-2] <= 1:
            return scores
        b, qh, n, L = a.shape
        if n != L:
            if strict:
                raise RuntimeError(f"SelKV: chunked prefill (query length {n} != key length {L}) is not "
                                   f"supported; refusing to silently disable the merge")
            return scores                                  # snapkv selector, merge off: plain SnapKV
        W = min(self.selkv_window, n)
        win = a[..., -W:, :].float().sum(dim=-2)           # (b, qh, L)
        extra = {"win_q": win}
        if self._selkv_need_bucket:
            B = self.bucket
            pos = torch.arange(L, device=a.device)
            cols = (pos // B * B)[:, None] + torch.arange(B, device=a.device)[None, :]     # (L, B)
            valid = cols < L
            idx = cols.clamp(max=L - 1).view(1, 1, L, B).expand(b, qh, L, B)
            extra["bucket_q"] = torch.gather(a, -1, idx).float() * valid.view(1, 1, L, B)  # A[h, j, bucket(j)]
        if global_topk > 0:
            # per KV head: the top-K attended positions of the head-summed row A_{h,j,.}; one head at a
            # time so the transient float32 (b, L, L) block stays at L^2 floats
            kv = self._selkv_kv_heads()
            g = qh // kv
            K = min(int(global_topk), L)
            vals, idxs = [], []
            for h in range(kv):
                ag = a[:, h * g:(h + 1) * g].float().sum(dim=1)                          # (b, L, L)
                v, i = ag.topk(K, dim=-1)                                                 # sorted desc
                vals.append(v); idxs.append(i.to(torch.int32))
            extra["glob_val"] = torch.stack(vals, dim=1)                                  # (b, kv, L, K)
            extra["glob_idx"] = torch.stack(idxs, dim=1)
        return (scores, extra)

    def _selkv_split_pending(self, layer_idx):
        """reduce_attention returns (scores, stats); move the stats to _pending before the base
        class consumes the scores."""
        x = self.current_attention_scores.get(layer_idx)
        if isinstance(x, tuple):
            self._pending[layer_idx] = x[1]
            self.current_attention_scores[layer_idx] = x[0]

    def _selkv_group(self, t, kv):
        """(b, qh, ...) -> (b, kv, ...) summed over the query heads of each KV head."""
        b, qh = t.shape[:2]
        g = qh // kv
        return t.reshape(b, kv, g, *t.shape[2:]).sum(dim=2)

    # ------------------------------------------------------------------ Eqs. 1-2
    def _selkv_keep_middle(self, layer_idx, mid, budget, ms, me, extra):
        _, v = self._get_existing_cache(layer_idx)
        vn = v[0].float().norm(dim=-1)[:, ms:me]                                     # (kv, M)
        if extra is None:                                   # decode-time maintenance
            score = avgpool_tokens(mid.float() * vn.mean(dim=0), self.selkv_pool)
            return self._middle_keep_indices(score, budget)
        kv = v.shape[1]
        a = self._selkv_group(extra["win_q"], kv)[0, :, ms:me]                         # (kv, M)
        c = avgpool_tokens(a * vn, self.selkv_pool)                                    # Eq. 1 + pooling
        M = c.shape[-1]
        top = torch.topk(c, k=min(budget, M), dim=-1).indices                          # per head, m_s = budget
        head_sets = torch.zeros(kv, M, dtype=torch.bool, device=c.device)
        head_sets.scatter_(1, top, True)
        self._head_sets[layer_idx] = head_sets
        union = head_sets.any(dim=0)
        self.merge_stats["union_size"] += int(union.sum())
        if int(union.sum()) > budget:                                                  # trim by mean score
            mean = c.mean(dim=0).masked_fill(~union, float("-inf"))
            keep = torch.topk(mean, k=budget).indices
        else:
            keep = torch.nonzero(union).flatten()
        return keep.sort().values


class SelKVCache(SelKVSelectorMixin, BiasCompensatedCache):

    def __init__(self, selector: str = "selkv", compensation_alpha: float = 0.5, bucket: int = 32,
                 selkv_window: int = 32, selkv_pool: int = 5, fallback: str = "drop",
                 fallback_topk: int = 128, **kwargs):
        if str(selector) == "selkv":
            kwargs["pool_kernel"] = 0                     # SelKV pools with its own average rule
            kwargs["observation_window"] = int(selkv_window)
        else:
            kwargs.setdefault("observation_window", 32)  # the protocol's pooled SnapKV base
            kwargs.setdefault("pool_kernel", 7)
        super().__init__(**kwargs)
        self.selector = str(selector)
        if self.selector not in ("selkv", "snapkv"):
            raise ValueError(f"selector must be 'selkv' or 'snapkv', got {selector!r}")
        self.alpha = float(compensation_alpha)
        self.fallback = str(fallback)
        if self.fallback not in ("drop", "global"):
            raise ValueError(f"fallback must be 'drop' or 'global', got {fallback!r}")
        self.fallback_topk = int(fallback_topk)
        self._selkv_init(selkv_window, selkv_pool, bucket, need_bucket=True)
        self._finalizing = False

    # ------------------------------------------------------------------ prefill statistics
    def reduce_attention(self, attn_weights):
        scores = super().reduce_attention(attn_weights)
        return self._selkv_reduce(scores, attn_weights, strict=(self.merge or self.selector == "selkv"),
                                  global_topk=self.fallback_topk if (self.merge and self.fallback == "global") else 0)

    def _consume_attention_scores(self, layer_idx):
        self._selkv_split_pending(layer_idx)
        return super()._consume_attention_scores(layer_idx)

    def _require_pending(self, layer_idx, extra):
        if extra is None and self._finalizing:
            raise RuntimeError(f"SelKV: layer {layer_idx} evicts at prefill but no prefill attention "
                               f"statistics were captured; refusing to silently fall back to eviction")

    def _after_selection(self, layer_idx):
        self._head_sets.pop(layer_idx, None)

    def on_prefill_end(self):
        self._finalizing = True
        try:
            super().on_prefill_end()
        finally:
            self._finalizing = False

    def _group(self, t, kv):
        return self._selkv_group(t, kv)

    # ------------------------------------------------------------------ Eqs. 1-2
    def _keep_middle(self, layer_idx, mid, budget, ms, me, extra):
        if self.selector == "snapkv":
            return super()._keep_middle(layer_idx, mid, budget, ms, me, extra)
        return self._selkv_keep_middle(layer_idx, mid, budget, ms, me, extra)

    # ------------------------------------------------------------------ Eqs. 3-6
    @torch.no_grad()
    def _merge_layer(self, layer_idx, keep_mid, ms, me, extra):
        k, v = self._get_existing_cache(layer_idx)
        b, kv, L, d = k.shape
        if b != 1:
            raise NotImplementedError("SelKVCache assumes batch size 1")
        dev = v.device
        B = self.bucket
        win = self._group(extra["win_q"].to(dev), kv)[0]                               # (kv, L)  a_{h,i}
        buck = self._group(extra["bucket_q"].to(dev), kv)[0]                           # (kv, L, B)
        self._gqa_groups = extra["win_q"].shape[1] // kv
        keep = torch.zeros(L, dtype=torch.bool, device=dev)
        keep[:ms] = True
        keep[me:] = True
        keep[ms + keep_mid.to(dev)] = True
        ev = torch.nonzero(~keep).flatten()
        vf = v[0].float()
        newv = vf.clone()
        R = torch.ones(kv, L, dtype=torch.float32, device=dev)
        st = self.merge_stats
        head_sets = self._head_sets.pop(layer_idx, None)
        use_global = self.fallback == "global"
        if use_global:
            if "glob_idx" not in extra:
                raise RuntimeError("SelKV: fallback=global but no global candidates were captured at prefill")
            glob_idx = extra["glob_idx"][0].to(dev)                                    # (kv, L, K) int32, desc
            glob_val = extra["glob_val"][0].to(dev)
            st["fallback_routed"] += 0                                                 # keys present even when unused
            st["fallback_miss"] += 0
        if ev.numel():
            cols = (ev // B * B)[:, None] + torch.arange(B, device=dev)[None, :]      # (E, B)
            in_range = cols < L
            cc = cols.clamp(max=L - 1)
            for h in range(kv):
                s_h = keep.clone()                                                     # S_h (Eq. 3)
                if head_sets is not None:
                    s_h[ms:me] = keep[ms:me] & head_sets[h].to(dev)
                valid = in_range & s_h[cc]
                att = buck[h, ev].masked_fill(~valid, -1.0)                            # Eq. 3
                best, arg = att.max(dim=1)
                ok = best > 0
                st["evicted"] += ev.numel()
                st["routed_bucket"] += int(ok.sum())
                j = ev[ok]
                i = cols[ok, arg[ok]]
                if use_global and not bool(ok.all()):
                    # Eq. 3 over the whole S_h for the tokens whose bucket is empty: the candidates are
                    # sorted by descending A_{h,j,.}, so the first one inside S_h is the global argmax
                    rest = ev[~ok]
                    gi = glob_idx[h, rest].long()                                      # (E', K)
                    gv = glob_val[h, rest]
                    cand = s_h[gi] & (gv > 0)
                    has = cand.any(dim=1)
                    first = cand.float().argmax(dim=1)                                 # first True per row
                    st["fallback_routed"] += int(has.sum())
                    st["fallback_miss"] += int((~has).sum())
                    if bool(has.any()):
                        j = torch.cat([j, rest[has]])
                        i = torch.cat([i, gi[has, first[has]]])
                st["routed"] += j.numel()
                st["dropped_empty_bucket"] += ev.numel() - j.numel()
                if j.numel() == 0:
                    continue
                g = F.cosine_similarity(vf[h, j], vf[h, i], dim=-1).clamp(min=0.0)     # Eq. 4
                st["gate_sum"] += float(g.sum())
                st["gate_zero"] += int((g == 0).sum())
                w = g * win[h, j]
                num = torch.zeros(L, d, dtype=torch.float32, device=dev).index_add_(0, i, w[:, None] * vf[h, j])
                den = torch.zeros(L, dtype=torch.float32, device=dev).index_add_(0, i, w)
                a_i = win[h]
                touched = den > 0
                st["floored_attention_targets"] += int((touched & (a_i < 1e-8)).sum())
                a_f = a_i.clamp_min(1e-8)
                tot = a_i + den
                newv[h] = torch.where(touched[:, None], (a_i[:, None] * vf[h] + num)
                                      / tot.clamp_min(1e-30)[:, None], vf[h])          # Eq. 5
                R[h] = torch.where(touched, (a_f + den) / a_f, R[h])                   # Eq. 6
        R_bar = R.mean(dim=0)                                                          # GQA: average over KV heads
        st["logR_max"] = max(st["logR_max"], float(torch.log(R_bar).max()))
        bias = (self.alpha * torch.log(R_bar)).view(1, 1, L).expand(1, kv, L).contiguous()
        self._replace_existing_cache(layer_idx, k, newv.unsqueeze(0).to(v.dtype))
        self._set_bias(layer_idx, bias)
