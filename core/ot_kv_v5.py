"""OT-KV v5 — transport in logit-response space.

`docs/ot_kv_theory.md` derives that the fraction of squared error a
query-independent merge can remove is ``e^{-sigma^2}`` with

    sigma_ij^2 = (k_i - k_j)^T Sigma_q (k_i - k_j) / d

i.e. a Mahalanobis distance under the second moment of the queries that will
read the cache. Measured on Qwen3-1.7B, the cosine distance used by v1-v4
correlates with that quantity at only **0.273**, and routing by it leaves
sigma = 1.50 (ceiling 21 % of the error removable) where routing by the whitened
distance leaves sigma = 1.21 (ceiling 29 %). Every version so far has been
solving the wrong problem accurately.

v5 estimates sigma directly, with no model-specific code and no covariance
matrix, from attention weights the framework already captures. For the last `W`
queries of the prefill,

    log A[w, i] - log A[w, j] = q_w . (k_i - k_j) / sqrt(d)

because the per-query softmax normaliser cancels in the difference. Writing
`z_t` for token `t`'s centred log-attention column divided by sqrt(W), and
`mu_t` for its mean:

    cost_ij  = || z_i - z_j ||^2          (this *is* sigma_ij^2)
    c*_ij    = exp( mu_i - mu_j + cost_ij / 2 )

The cost is an ordinary squared Euclidean distance, so the top-`r` candidate
search and the sparse Sinkhorn carry over unchanged -- only the space changes.
The coefficient is no longer the historical mass ratio `s_i/s_j` used by v3/v4
but the expectation of the *actual* attention ratio under the observed queries,
which is what the derivation asks for.

Merging is additive here, not convex: the merged entry keeps key `k_j`, so it
receives weight `a_j`, and reproducing `a_j v_j + sum_i a_i v_i` requires
`v'_j = v_j + sum_i c_ij v_i`. `coeff_cap` bounds the absorbed coefficient mass
so a badly-estimated ratio cannot blow a value up.

The logit window is captured once, at the end of prefill, and then dropped;
decode-time compressions fall back to v3's key-space cost. That is deliberate:
prefill is where almost all of the compression happens, and keeping a
`W x L` attention block alive through decoding would cost more memory than the
cache it is compressing.
"""

from typing import Optional, Tuple

import torch

from core.ot_kv_v3 import (EPS, OTKVv3Cache, _scatter_weighted_sum, prepare_scores,
                           select_anchors, sparse_sinkhorn)

LOG_FLOOR = 1e-12


def logit_space_embedding(window_attn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(B, H, Nq, L) attention weights -> centred log-columns and their means.

    Returns z of shape (B, H, L, Nq) with ``||z_i - z_j||^2 = Var_w(log A_i - log A_j)``,
    and mu of shape (B, H, L).
    """
    x = window_attn.clamp_min(LOG_FLOOR).log()          # (B, H, Nq, L)
    mu = x.mean(dim=-2)                                 # (B, H, L)
    n_q = x.shape[-2]
    z = (x - mu.unsqueeze(-2)) / (n_q ** 0.5)
    return z.transpose(-1, -2).contiguous(), mu         # (B, H, L, Nq), (B, H, L)


def _topr_logit(z_e: torch.Tensor, z_a: torch.Tensor, top_r: int, chunk: int = 1024):
    """Per evicted token, the `top_r` anchors with smallest ||z_i - z_j||^2."""
    n, r = z_e.shape[-2], min(top_r, z_a.shape[-2])
    a_sq = z_a.pow(2).sum(-1).unsqueeze(-2)             # (B, H, 1, m)
    costs, idxs = [], []
    for s in range(0, n, chunk):
        blk = z_e[..., s:s + chunk, :]
        d2 = blk.pow(2).sum(-1, keepdim=True) + a_sq - 2 * torch.matmul(blk, z_a.transpose(-1, -2))
        c, i = torch.topk(d2.clamp_min(0), r, dim=-1, largest=False)
        costs.append(c)
        idxs.append(i)
        del d2, blk
    return torch.cat(costs, dim=-2), torch.cat(idxs, dim=-2)


def otkv_v5_compress(key_states, value_states, budget, scores, window_attn,
                     *,
                     merge: bool = True,
                     merge_strength: float = 1.0,
                     top_r: int = 4,
                     epsilon: float = 0.02,
                     sinkhorn_iters: int = 5,
                     sigma_max: float = 2.0,
                     coeff_cap: float = 2.0,
                     select_mode: str = "topk",
                     select_scope: str = "global",
                     chunk: int = 1024):
    """Compress with the transport cost and coefficient of docs/ot_kv_theory.md.

    sigma_max gates the merge: an evicted token whose best anchor still leaves
    sigma above this keeps almost none of the attainable reduction
    (e^{-sigma^2} < 2 % at sigma = 2), so folding it in only injects noise.
    """
    batch, heads, seq_len, head_dim = key_states.shape
    s_all = prepare_scores(key_states, scores, "attn")

    budget = min(max(int(budget), 1), seq_len)
    if seq_len <= budget:
        return key_states, value_states, s_all, torch.arange(seq_len, device=key_states.device
                                                             ).view(1, 1, -1).expand(batch, heads, -1)

    anchor_idx = select_anchors(key_states, s_all, budget, select_mode, scope=select_scope)
    g4 = anchor_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    k_anchor = torch.gather(key_states, 2, g4)
    v_anchor = torch.gather(value_states, 2, g4)
    s_anchor = torch.gather(s_all, 2, anchor_idx)

    n_evict = seq_len - budget
    if not merge or n_evict <= 0 or window_attn is None:
        return k_anchor, v_anchor, s_anchor, anchor_idx

    keep = torch.zeros(batch, heads, seq_len, device=key_states.device, dtype=torch.bool)
    keep.scatter_(-1, anchor_idx, True)
    evict_idx = torch.argsort(keep.to(torch.uint8), dim=-1, stable=True)[..., :n_evict].sort(dim=-1).values

    z, mu = logit_space_embedding(window_attn)                      # (B,H,L,Nq), (B,H,L)
    nq = z.shape[-1]
    z_a = torch.gather(z, 2, anchor_idx.unsqueeze(-1).expand(-1, -1, -1, nq))
    z_e = torch.gather(z, 2, evict_idx.unsqueeze(-1).expand(-1, -1, -1, nq))
    mu_a = torch.gather(mu, 2, anchor_idx)
    mu_e = torch.gather(mu, 2, evict_idx)

    sigma2, cand_idx = _topr_logit(z_e, z_a, top_r, chunk=chunk)     # (B,H,n,r)

    # log of the expected attention ratio, E[a_i / a_j] for a log-normal ratio
    log_c = (mu_e.unsqueeze(-1) - torch.gather(mu_a.unsqueeze(-2).expand(-1, -1, n_evict, -1), -1, cand_idx)
             + 0.5 * sigma2)
    coeff = log_c.exp()

    # gate: beyond sigma_max the attainable reduction e^{-sigma^2} is negligible
    coeff = coeff * (sigma2 <= sigma_max ** 2).to(coeff.dtype)

    # transport plan over the candidate support; the capacity constraint keeps a
    # popular anchor from absorbing an unbounded number of tokens
    ev_mass = s_all.gather(-1, evict_idx).clamp_min(EPS)
    log_mu_row = (ev_mass / ev_mass.sum(-1, keepdim=True).clamp_min(EPS)).log()
    nu = (s_anchor / s_anchor.sum(-1, keepdim=True).clamp_min(EPS)).clamp_min(EPS)
    plan = sparse_sinkhorn(sigma2, cand_idx, log_mu_row, nu.log(), epsilon, sinkhorn_iters)
    plan = plan / plan.sum(-1, keepdim=True).clamp_min(EPS)          # row-stochastic assignment

    absorbed = plan * coeff * merge_strength                          # (B,H,n,r)

    # cap the coefficient mass any single anchor takes on
    flat_idx = cand_idx.reshape(batch, heads, -1)
    load = torch.zeros_like(s_anchor)
    load.scatter_add_(-1, flat_idx, absorbed.reshape(batch, heads, -1))
    scale = (coeff_cap / load.clamp_min(EPS)).clamp(max=1.0)
    absorbed = absorbed * torch.gather(scale.unsqueeze(-2).expand(-1, -1, n_evict, -1), -1, cand_idx)

    v_evict = torch.gather(value_states, 2, evict_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim))
    contrib = _scatter_weighted_sum(v_evict.float(), absorbed, cand_idx, v_anchor.shape[-2], head_dim)
    v_merged = v_anchor.float() + contrib

    load = torch.zeros_like(s_anchor)
    load.scatter_add_(-1, flat_idx, absorbed.reshape(batch, heads, -1))
    new_scores = s_anchor * (1.0 + load)

    return k_anchor, v_merged.to(value_states.dtype), new_scores, anchor_idx


class OTKVv5Cache(OTKVv3Cache):
    """v4's scaffolding with the logit-space transport of the theory note.

    The prefill compression uses the measured cost and coefficient; decode-time
    compressions fall back to the inherited key-space path, which is what v4
    already does well and costs nothing extra to keep.
    """

    def __init__(self, **kwargs):
        defaults = dict(epsilon=0.02, top_r=4, compress_interval=4, merge=True,
                        select_mode="topk", select_scope="global", observation_window=32,
                        per_head=True)
        defaults.update(kwargs)
        super().__init__(**defaults)
        self.logit_window = int(defaults.get("logit_window", 32))
        self.logit_query_mode = str(defaults.get("logit_query_mode", "recent"))
        self.sigma_max = float(defaults.get("sigma_max", 2.0))
        self.coeff_cap = float(defaults.get("coeff_cap", 2.0))
        self.v5_merge_strength = float(defaults.get("v5_merge_strength", 1.0))
        self._logattn = {}

    # -- capture ---------------------------------------------------------
    def reduce_attention(self, attn_weights: torch.Tensor) -> torch.Tensor:
        a = attn_weights.detach()
        n_q = a.shape[-2]
        if n_q <= self.logit_window:                 # decode step: nothing to capture
            return super().reduce_attention(a)
        score = super().reduce_attention(a)          # (B, qh, L)
        if self.logit_query_mode == "spread":
            # The embedding is only as informative as the queries it is built
            # from. A recent window is right when a question at the tail decides
            # relevance; for open-ended continuation the queries that matter are
            # spread over the whole prompt, and taking only the last W makes the
            # geometry describe the wrong thing.
            idx = torch.linspace(0, n_q - 1, self.logit_window, device=a.device).long()
            win = a.index_select(-2, idx).float()
        else:
            win = a[..., -self.logit_window:, :].float()
        # one tensor carries both so the existing hook plumbing is untouched;
        # `_consume_attention_scores` knows the layer and splits them apart
        return torch.cat([score.unsqueeze(-2), win], dim=-2)

    def _consume_attention_scores(self, layer_idx: int):
        x = self.current_attention_scores.get(layer_idx)
        if x is not None and x.dim() == 4 and x.shape[-2] == self.logit_window + 1:
            score, win = x[..., 0, :], x[..., 1:, :]
            b, qh, w, L = win.shape
            kv = self.attention_scores[layer_idx].shape[-2] if layer_idx in self.attention_scores else qh
            if qh % kv == 0:                          # GQA: group queries are separate queries
                win = win.reshape(b, kv, (qh // kv) * w, L)
            self._logattn[layer_idx] = win
            self.current_attention_scores[layer_idx] = score
        return super()._consume_attention_scores(layer_idx)

    # -- compression -----------------------------------------------------
    def _compress_layer(self, layer_idx: int, total_tokens: int, reserve_tokens: int = 0,
                        allow_merge: bool = True, allow_select: bool = True):
        win = self._logattn.pop(layer_idx, None)
        if win is None or not allow_merge:
            return super()._compress_layer(layer_idx, total_tokens, reserve_tokens,
                                           allow_merge, allow_select)

        middle_k, middle_v = self.get_middle_cache(layer_idx)
        if middle_k is None or middle_k.shape[-2] == 0:
            return
        budget = max(0, int(self.get_middle_budget(layer_idx, total_tokens)) - reserve_tokens)
        if budget >= middle_k.shape[-2]:
            return

        seq_len = self._get_existing_cache(layer_idx)[0].shape[-2]
        ms, me = self.sink_size, max(self.sink_size, seq_len - self.recent_size)
        scores = self.attention_scores[layer_idx]
        if win.shape[-1] != seq_len:
            return super()._compress_layer(layer_idx, total_tokens, reserve_tokens,
                                           allow_merge, allow_select)

        new_k, new_v, new_scores, _ = otkv_v5_compress(
            middle_k, middle_v, budget, scores[..., ms:me], win[..., ms:me],
            merge=self.merge, merge_strength=self.v5_merge_strength, top_r=self.top_r,
            epsilon=self.epsilon, sinkhorn_iters=self.sinkhorn_iters,
            sigma_max=self.sigma_max, coeff_cap=self.coeff_cap,
            select_mode=self.select_mode if allow_select else "topk",
            select_scope=self.select_scope, chunk=self.chunk,
        )
        self.replace_middle_cache(layer_idx, new_k, new_v)
        self.attention_scores[layer_idx] = torch.cat(
            [scores[..., :ms], new_scores, scores[..., me:]], dim=-1)
