"""OT-KV v3: mass-conserving optimal-transport KV merging.

What changed relative to v1/v2 and why
-------------------------------------
v1 merged with ``V_A += T^T V_E`` where ``T`` is a doubly-stochastic plan of
total mass 1.  Each anchor column of ``T`` therefore sums to ``1/m``, so the
update perturbed the anchor values by ~0.03% of their norm: the "optimal
transport merge" was numerically a no-op and v1 behaved as plain top-k
eviction.  v2 tried to fix the magnitude by rescaling with ``n_evict``, which
overshoots in the other direction -- it adds roughly one full-magnitude value
vector to every anchor, inflating ``||v||`` and biasing every attention output.

v3 fixes the scale by deriving it instead of choosing it.

The derivation
--------------
Write ``s_t`` for the attention mass token ``t`` accumulated during prefill,
``A`` for the retained anchors and ``E`` for the evicted tokens, and let
``S_A = sum_{j in A} s_j``, ``S_E = sum_{i in E} s_i``.

Transport the evicted mass with a plan ``m_ij >= 0`` whose marginals are

    sum_j m_ij = s_i                (every evicted token ships its own mass)
    sum_i m_ij = c_j = s_j * S_E/S_A  (anchors absorb in proportion to weight)

and merge as a *convex combination*

    v'_j = (s_j v_j + sum_i m_ij v_i) / (s_j + c_j).

Consistency: at decode time softmax renormalises over the retained set only, so
the anchor's weight becomes ``s_j / S_A`` (assuming future attention is
proportional to accumulated attention).  The true share of the cluster the
anchor now stands for is ``(s_j + c_j) / (S_A + S_E) = s_j / S_A`` -- exactly
equal.  So with this particular capacity the merged entry receives precisely the
attention mass of the token set it represents, and carries that set's
mass-weighted centroid as its value.  No free scale constant is left over.

Two consequences worth noting: the merge is deliberately *mild* when eviction
removes little mass (as at 50% budget) and grows automatically as the budget
tightens, and the value stays inside the convex hull of the originals so the
cache cannot blow up numerically.

Cost and cost of computing it
-----------------------------
Keys are stored post-RoPE, and that is the right space to compare them in: a
future query ``q`` scores token ``t`` by ``q . k_t``, so ``k_i ~= k_j`` is
precisely the condition under which two tokens are interchangeable for *every*
query and can share one slot.

A dense ``n x m`` Sinkhorn is memory-bound and hopeless at long context, so v3
restricts transport to each evicted token's ``top_r`` nearest anchors and runs
log-domain Sinkhorn over that sparse structure with scatter reductions.  Cost
drops from ``O(n*m)`` per iteration to ``O(n*r)``; the single dense similarity
matmul used to pick the candidates is chunked and never materialised in full.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from evaluation.models.base_cache import BaseCompressCache

EPS = 1e-9
NEG_INF = -1e30


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def _exposure_correction(seq_len: int, device, dtype) -> torch.Tensor:
    """1 / (number of queries a token has been visible to).

    Accumulated attention is biased towards early tokens purely because they sat
    in the context for longer. Dividing by the exposure count removes that.
    """
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    return 1.0 / (seq_len - pos).clamp_min(1.0)


def prepare_scores(key_states: torch.Tensor, scores: Optional[torch.Tensor],
                   score_mode: str = "attn") -> torch.Tensor:
    """Return a strictly positive (batch, heads, seq) importance tensor."""
    batch, heads, seq_len, _ = key_states.shape
    device = key_states.device

    if scores is None:
        s = key_states.float().norm(dim=-1)
        s = s * _exposure_correction(seq_len, device, s.dtype).view(1, 1, -1)
    else:
        s = scores.detach().to(device=device, dtype=torch.float32)
        s = torch.where(torch.isfinite(s), s, torch.zeros_like(s)).clamp_min(0.0)
        if s.dim() == 1:
            s = s.view(1, 1, -1).expand(batch, heads, -1)
        elif s.dim() == 2:
            s = s.unsqueeze(0).expand(batch, -1, -1)

        if score_mode == "attn_debiased":
            s = s * _exposure_correction(seq_len, device, s.dtype).view(1, 1, -1)
        elif score_mode == "keynorm":
            s = key_states.float().norm(dim=-1) * _exposure_correction(seq_len, device, torch.float32).view(1, 1, -1)

        dead = s.sum(dim=-1, keepdim=True) <= EPS
        if dead.any():
            fb = key_states.float().norm(dim=-1)
            s = torch.where(dead, fb, s)

    return s.clamp_min(EPS).contiguous()


# ---------------------------------------------------------------------------
# sparse log-domain Sinkhorn
# ---------------------------------------------------------------------------
def _scatter_logsumexp(vals: torch.Tensor, idx: torch.Tensor, num_cols: int) -> torch.Tensor:
    """logsumexp of `vals` grouped by `idx` along the last axis.

    vals/idx: (B, H, n, r) -> returns (B, H, num_cols). Empty groups map to
    NEG_INF rather than NaN.
    """
    lead = vals.shape[:2]
    flat_v = vals.reshape(*lead, -1)
    flat_i = idx.reshape(*lead, -1)

    amax = torch.full((*lead, num_cols), NEG_INF, device=vals.device, dtype=vals.dtype)
    amax.scatter_reduce_(-1, flat_i, flat_v, reduce="amax", include_self=True)

    shifted = torch.exp(flat_v - amax.gather(-1, flat_i))
    sums = torch.zeros((*lead, num_cols), device=vals.device, dtype=vals.dtype)
    sums.scatter_add_(-1, flat_i, shifted)

    out = amax + torch.log(sums.clamp_min(1e-30))
    return torch.where(sums > 0, out, torch.full_like(out, NEG_INF))


def sparse_sinkhorn(cost: torch.Tensor, idx: torch.Tensor, log_mu: torch.Tensor,
                    log_nu: torch.Tensor, epsilon: float, iters: int) -> torch.Tensor:
    """Entropy-regularised OT restricted to the (i, idx[i, :]) support.

    cost   : (B, H, n, r) transport cost of the candidate pairs
    idx    : (B, H, n, r) anchor index of each candidate
    log_mu : (B, H, n)    log row marginal (sums to 1 in probability space)
    log_nu : (B, H, m)    log column marginal
    Returns the plan restricted to the support, shape (B, H, n, r), summing to 1.
    """
    c_eps = cost / epsilon
    m = log_nu.shape[-1]

    f = torch.zeros_like(log_mu)
    g = torch.zeros_like(log_nu)

    n, r = cost.shape[-2], cost.shape[-1]
    flat_idx = idx.reshape(*idx.shape[:2], -1)

    def gather_g(gv):
        return gv.gather(-1, flat_idx).view_as(cost)

    for _ in range(max(1, int(iters))):
        # row scaling: enforce sum_j T_ij = mu_i (dense over the r candidates)
        f = log_mu - torch.logsumexp(gather_g(g) - c_eps, dim=-1)
        # column scaling: enforce sum_i T_ij = nu_j (sparse scatter over anchors)
        g = log_nu - _scatter_logsumexp(f.unsqueeze(-1) - c_eps, idx, m)
        g = torch.where(torch.isfinite(g), g, torch.zeros_like(g))

    return torch.exp(f.unsqueeze(-1) + gather_g(g) - c_eps)


def _scatter_weighted_sum(vectors: torch.Tensor, weights: torch.Tensor, idx: torch.Tensor,
                          num_anchors: int, head_dim: int, group: int = 4) -> torch.Tensor:
    """sum_i weights[i, j] * vectors[i], accumulated onto anchor idx[i, j].

    Done `group` candidate columns at a time: one scatter per group instead of
    one per column keeps the kernel-launch count low without materialising the
    full (n, r, head_dim) product.
    """
    b, h, n, r = weights.shape
    out = torch.zeros((b, h, num_anchors, head_dim), device=vectors.device, dtype=torch.float32)
    for start in range(0, r, group):
        w = weights[..., start:start + group]
        i = idx[..., start:start + group]
        g = w.shape[-1]
        src = (vectors.unsqueeze(-2) * w.unsqueeze(-1)).reshape(b, h, n * g, head_dim)
        out.scatter_add_(2, i.reshape(b, h, n * g, 1).expand(-1, -1, -1, head_dim), src)
    return out


def select_anchors(keys: torch.Tensor, scores: torch.Tensor, budget: int,
                   select_mode: str = "topk", kmeans_iters: int = 4,
                   scope: str = "head") -> torch.Tensor:
    """Choose which tokens keep a slot.

    "topk" keeps the highest accumulated attention -- maximises retained mass but
    ignores whether the survivors *cover* the key cloud, which is what decides
    whether the evicted tokens have a usable stand-in.

    "kmeans:P" reserves P% of the budget for heavy hitters and spends the rest on
    a mass-weighted k-means quantisation of the keys, taking the real token
    nearest each centroid. Better coverage, less retained mass.
    """
    if scope == "global":
        # Pool the evidence across heads before ranking, then give every head the
        # same token set. Scores are still *accumulated* per head, so this is not
        # the same as global-mode bookkeeping: it keeps per-head transport while
        # estimating importance from all heads at once. Short observation windows
        # need this -- 32 query rows split eight ways is too little evidence to
        # rank on (see docs/experiments.md section 1).
        pooled = scores.sum(dim=1, keepdim=True)
        scores = pooled.expand_as(scores)

    if not (select_mode.startswith("kmeans") or select_mode.startswith("adaptive")):
        return torch.topk(scores, budget, dim=-1).indices.sort(dim=-1).values

    b, h, L, d = keys.shape
    adaptive = select_mode.startswith("adaptive")
    param = float(select_mode.split(":")[1]) if ":" in select_mode else (1.0 if adaptive else 50)
    if adaptive:
        n_heavy, n_cl = 0, budget
    else:
        n_heavy = min(budget, int(budget * param / 100))
        n_cl = max(1, budget - n_heavy)

    kf = keys.float()
    seed_pos = torch.linspace(0, L - 1, n_cl, device=keys.device).long()
    centroids = kf[:, :, seed_pos, :].clone()
    w = scores.unsqueeze(-1)
    for _ in range(max(1, kmeans_iters)):
        assign = torch.cdist(kf, centroids).argmin(-1)
        oh = F.one_hot(assign, n_cl).to(kf.dtype) * w
        centroids = torch.einsum("bhlc,bhld->bhcd", oh, kf) / oh.sum(-2).unsqueeze(-1).clamp_min(1e-6)
    medoids = torch.cdist(centroids, kf).argmin(-1)

    picked = torch.zeros_like(scores, dtype=torch.bool)
    picked.scatter_(-1, medoids, True)

    if adaptive:
        # How much coverage a head can afford depends on how concentrated it is.
        # A head whose top-`budget` tokens already hold nearly all of its
        # attention is a sharp/retrieval head: spending slots on being
        # representative there evicts exactly the rare token the head exists to
        # find. A diffuse head loses little by trading peak for coverage.
        top = torch.topk(scores, budget, dim=-1).values
        conc = (top.sum(-1, keepdim=True) / scores.sum(-1, keepdim=True).clamp_min(EPS)).clamp(0.0, 1.0)
        beta = param * (1.0 - conc)                                   # (b, h, 1)

        # Score by rank, not magnitude: attention mass is heavy-tailed, so
        # dividing by the max leaves nearly every token at ~0 and any positive
        # bonus would elect every medoid. In rank units the knob means something
        # concrete -- a medoid may displace a kept token if it sits within
        # beta * budget ranks of the cut-off.
        order = torch.argsort(scores, dim=-1, descending=True)
        rank = torch.empty_like(order)
        rank.scatter_(-1, order, torch.arange(L, device=scores.device).expand_as(order))
        ranked = beta * picked.to(scores.dtype) - rank.to(scores.dtype) / max(1, budget)
    else:
        if n_heavy:
            picked.scatter_(-1, torch.topk(scores, n_heavy, dim=-1).indices, True)
        # keep everything picked, then top up with the heaviest leftovers
        ranked = scores + picked.to(scores.dtype) * (scores.max() + 1.0)
    return torch.topk(ranked, budget, dim=-1).indices.sort(dim=-1).values


# ---------------------------------------------------------------------------
# candidate selection
# ---------------------------------------------------------------------------
def _top_candidates(k_evict: torch.Tensor, k_anchor: torch.Tensor, top_r: int,
                    chunk: int = 1024) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per evicted token, the `top_r` most cosine-similar anchors.

    The (n x m) similarity is computed in row chunks so it is never held in full.
    Returns (similarity, anchor index), each (B, H, n, top_r).
    """
    # The (n x m) similarity is only used to *rank* candidates, so half precision
    # is plenty and halves what is by far the dominant cost of a compression.
    mm_dtype = torch.float16 if k_evict.device.type in ("cuda", "mps") else torch.float32
    ke = F.normalize(k_evict.to(mm_dtype), dim=-1)
    ka = F.normalize(k_anchor.to(mm_dtype), dim=-1).transpose(-1, -2)

    n = ke.shape[-2]
    r = min(top_r, k_anchor.shape[-2])

    sims, idxs = [], []
    for start in range(0, n, chunk):
        block = torch.matmul(ke[..., start:start + chunk, :], ka)
        v, i = torch.topk(block, r, dim=-1)
        sims.append(v.float())
        idxs.append(i)
        del block
    return torch.cat(sims, dim=-2), torch.cat(idxs, dim=-2)


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def otkv_v3_compress(key_states: torch.Tensor, value_states: torch.Tensor, budget: int,
                     scores: Optional[torch.Tensor] = None,
                     *,
                     merge: bool = True,
                     top_r: int = 16,
                     epsilon: float = 0.05,
                     sinkhorn_iters: int = 5,
                     capacity_beta: float = 1.0,
                     sim_threshold: float = 0.0,
                     merge_strength: float = 1.0,
                     score_mode: str = "attn",
                     chunk: int = 1024,
                     key_merge: float = 0.0,
                     score_writeback: str = "absorbed",
                     select_mode: str = "topk",
                     select_scope: str = "head",
                     gate_sigma: float = 0.0,
                     cost_alpha: float = 0.8,
                     lambda_pos: float = 0.0,
                     frozen_mass: Optional[torch.Tensor] = None,
                     aux_scores: Optional[torch.Tensor] = None,
                     select_scores: Optional[torch.Tensor] = None,
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compress (batch, heads, seq, dim) K/V down to `budget` tokens per head.

    capacity_beta interpolates the anchor capacity between uniform (0.0) and the
    mass-consistent choice c_j proportional to s_j (1.0); see module docstring.
    sim_threshold drops evicted tokens whose best anchor is less similar than the
    threshold -- there is nothing to merge them into, so folding them in would
    only add noise. merge_strength scales the absorbed capacity for ablations.

    frozen_mass is the attention mass held by retained-but-untouchable tokens
    (the sink and the recent window). Decode-time softmax renormalises over
    those too, so they silently absorb their share ``s_t * S_E / S_R`` of the
    evicted mass; only the remaining ``S_A / S_R`` fraction may be transported
    into the anchors. Ignoring this over-merges by the ratio S_R / S_A, which is
    close to 2x once attention sinks are counted.
    """
    batch, heads, seq_len, head_dim = key_states.shape
    s_all = prepare_scores(key_states, scores, score_mode)

    budget = min(max(int(budget), 1), seq_len)
    if seq_len <= budget:
        return (key_states, value_states, s_all) if aux_scores is None else \
            (key_states, value_states, s_all, aux_scores)

    # Selection and merging want different evidence. Selection must be recent
    # (only the tail of the prompt knows the answer matters); the merge needs a
    # reliable estimate of future attention mass, because it uses it as the
    # transport capacity -- feeding it the noisy short-window score makes the
    # merge actively harmful. `select_scores` lets the caller separate them.
    sel = s_all if select_scores is None else prepare_scores(key_states, select_scores, "attn")
    anchor_idx = select_anchors(key_states, sel, budget, select_mode, scope=select_scope)

    gather4 = anchor_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    k_anchor = torch.gather(key_states, 2, gather4)
    v_anchor = torch.gather(value_states, 2, gather4)
    s_anchor = torch.gather(s_all, 2, anchor_idx)

    def _gather_aux():
        # (C, b, h, L) -> (C, b, h, m), following the anchors
        return None if aux_scores is None else torch.gather(
            aux_scores, -1, anchor_idx.unsqueeze(0).expand(aux_scores.shape[0], -1, -1, -1))

    if not merge:
        return (k_anchor, v_anchor, s_anchor) if aux_scores is None else \
            (k_anchor, v_anchor, s_anchor, _gather_aux())

    n_evict = seq_len - budget
    if n_evict <= 0:
        return (k_anchor, v_anchor, s_anchor) if aux_scores is None else \
            (k_anchor, v_anchor, s_anchor, _gather_aux())

    # Complement by index, not by boolean mask. Boolean indexing flattens across
    # heads and silently assumes every head kept exactly `budget` distinct
    # positions; a selection rule that returns a repeated index (ties, degenerate
    # scores) then yields a different count per head and the reshape explodes.
    # A stable argsort puts non-anchors first in each head, so taking the first
    # n_evict is exact per head regardless.
    keep = torch.zeros(batch, heads, seq_len, device=key_states.device, dtype=torch.bool)
    keep.scatter_(-1, anchor_idx, True)
    evict_idx = torch.argsort(keep.to(torch.uint8), dim=-1, stable=True)[..., :n_evict]
    evict_idx = evict_idx.sort(dim=-1).values

    ev4 = evict_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    k_evict = torch.gather(key_states, 2, ev4)
    v_evict = torch.gather(value_states, 2, ev4)
    s_evict = torch.gather(s_all, 2, evict_idx)

    sim, cand_idx = _top_candidates(k_evict, k_anchor, top_r, chunk=chunk)
    cost = 1.0 - sim

    # --- optional joint-cost terms, ported from core/ot_kv_joint.py ---------
    # That variant blends value-space distance into the key-space cost and adds a
    # penalty for merging across long positional distances. Neither term is
    # implied by the derivation (which only needs key proximity), so they are off
    # by default and exist to be measured.
    if cost_alpha < 1.0:
        ve = F.normalize(v_evict.float(), dim=-1)
        va = F.normalize(v_anchor.float(), dim=-1)
        cand_v = torch.gather(
            va.unsqueeze(-3).expand(-1, -1, n_evict, -1, -1), -2,
            cand_idx.unsqueeze(-1).expand(-1, -1, -1, -1, head_dim))
        cost_v = 1.0 - (ve.unsqueeze(-2) * cand_v).sum(-1)
        cost = cost_alpha * cost + (1.0 - cost_alpha) * cost_v

    if lambda_pos > 0.0:
        pos_a = anchor_idx
        pos_e = evict_idx
        cand_pos = torch.gather(pos_a.unsqueeze(-2).expand(-1, -1, n_evict, -1), -1, cand_idx)
        cost = cost + lambda_pos * torch.log1p((cand_pos - pos_e.unsqueeze(-1)).abs().float())

    if sim_threshold > 0.0:
        keep = (sim[..., 0] >= sim_threshold).float()
        s_evict = s_evict * keep

    # --- marginals ---------------------------------------------------------
    S_E = s_evict.sum(-1, keepdim=True).clamp_min(EPS)
    S_A = s_anchor.sum(-1, keepdim=True).clamp_min(EPS)
    if frozen_mass is None:
        S_R = S_A
    else:
        # global mode carries one shared score vector, so frozen_mass is a
        # scalar there and a per-head column in per-head mode
        fm = frozen_mass.to(S_A.device, S_A.dtype)
        fm = fm.reshape(1, 1, 1) if fm.numel() == 1 else fm.reshape(*S_A.shape)
        S_R = (S_A + fm).clamp_min(EPS)
    # total mass the anchors are entitled to absorb (the rest is claimed by the
    # sink / recent window through softmax renormalisation and is dropped)
    transportable = S_E * (S_A / S_R)
    log_mu = torch.log(s_evict.clamp_min(EPS) / S_E)

    if capacity_beta <= 0.0:
        nu = torch.full_like(s_anchor, 1.0 / s_anchor.shape[-1])
    else:
        nu = (s_anchor / S_A).clamp_min(EPS).pow(capacity_beta)
        nu = nu / nu.sum(-1, keepdim=True).clamp_min(EPS)
    log_nu = torch.log(nu.clamp_min(EPS))

    plan = sparse_sinkhorn(cost, cand_idx, log_mu, log_nu, epsilon, sinkhorn_iters)

    if gate_sigma > 0.0:
        # The merge applies each evicted value with coefficient s_i / s_j, i.e. the
        # historical attention ratio. Measured against held-out queries that ratio
        # is off by ~1 nat, and the error grows with key distance -- so damp the
        # transported mass by how close the pair actually is. Mass that is damped
        # away is simply dropped, which is what eviction would have done anyway.
        plan = plan * torch.exp(-cost / gate_sigma)

    # --- mass-weighted merge ----------------------------------------------
    absorbed = plan * transportable.unsqueeze(-1) * merge_strength   # (B,H,n,r) real mass
    lead = (batch, heads)

    flat_idx = cand_idx.reshape(*lead, -1)
    c_anchor = torch.zeros_like(s_anchor)
    c_anchor.scatter_add_(-1, flat_idx, absorbed.reshape(*lead, -1))

    # sum_i m_ij v_i, accumulated per anchor
    contrib = _scatter_weighted_sum(v_evict.float(), absorbed, cand_idx, v_anchor.shape[-2], head_dim)

    denom = (s_anchor + c_anchor).clamp_min(EPS).unsqueeze(-1)
    v_merged = (v_anchor.float() * s_anchor.unsqueeze(-1) + contrib) / denom

    if key_merge > 0.0:
        # Optionally pull the anchor key toward the centroid of what it absorbed,
        # so queries matching the absorbed tokens also retrieve the anchor. The
        # value derivation assumes k_j is unchanged, so this trades exactness for
        # retrieval coverage; key_merge scales how far we go.
        k_contrib = _scatter_weighted_sum(k_evict.float(), absorbed, cand_idx, k_anchor.shape[-2], head_dim)
        k_full = (k_anchor.float() * s_anchor.unsqueeze(-1) + k_contrib) / denom
        k_anchor = (k_anchor.float() + key_merge * (k_full - k_anchor.float())).to(key_states.dtype)

    new_scores = s_anchor if score_writeback == "own" else s_anchor + c_anchor
    if aux_scores is None:
        return k_anchor, v_merged.to(value_states.dtype), new_scores

    # Any per-token quantity transports with the *fraction* of a token's mass
    # that moved, f_ij = m_ij / s_i. Applied to the score itself this reduces to
    # sum_i m_ij = c_j, so every channel stays consistent with the main one.
    frac = absorbed / s_evict.clamp_min(EPS).unsqueeze(-1)
    aux_new = _gather_aux()
    for c in range(aux_scores.shape[0]):
        aux_e = torch.gather(aux_scores[c], -1, evict_idx)
        moved = torch.zeros_like(aux_new[c])
        moved.scatter_add_(-1, flat_idx, (frac * aux_e.unsqueeze(-1)).reshape(*lead, -1))
        aux_new[c] = aux_new[c] + moved
    return k_anchor, v_merged.to(value_states.dtype), new_scores, aux_new


# ---------------------------------------------------------------------------
class OTKVv3Cache(BaseCompressCache):
    """Drop-in cache applying mass-conserving OT merging at prefill end and
    every `compress_interval` decode steps."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.merge = bool(kwargs.get("merge", True))
        self.top_r = int(kwargs.get("top_r", 16))
        self.epsilon = float(kwargs.get("epsilon", 0.05))
        self.sinkhorn_iters = int(kwargs.get("sinkhorn_iters", 5))
        self.capacity_beta = float(kwargs.get("capacity_beta", 1.0))
        self.sim_threshold = float(kwargs.get("sim_threshold", 0.0))
        self.merge_strength = float(kwargs.get("merge_strength", 0.5))
        self.score_mode = str(kwargs.get("score_mode", "attn"))
        self.frozen_correction = bool(kwargs.get("frozen_correction", True))
        self.key_merge = float(kwargs.get("key_merge", 0.0))
        # Optional depth-dependent budget (PyramidKV's observation that early
        # layers attend broadly and late layers sharply). Orthogonal to the
        # selection and transport changes, so it is a separate knob.
        self.layer_budget = str(kwargs.get("layer_budget", "uniform"))
        self.pyramid_low_scale = float(kwargs.get("pyramid_low_scale", 1.5))
        self.pyramid_high_scale = float(kwargs.get("pyramid_high_scale", 0.5))
        self.num_layers = kwargs.get("num_layers", None)
        # Merging is a one-shot summarisation of the prefill. Re-merging every
        # `compress_interval` decode steps re-averages already-averaged values,
        # and that blur compounds; decode maintenance defaults to plain eviction.
        self.merge_decode = bool(kwargs.get("merge_decode", False))
        # Write the absorbed mass back into the anchor score. Correct in
        # principle (the anchor now stands for more tokens) but it compounds
        # across repeated compressions into a rich-get-richer drift.
        self.score_writeback = str(kwargs.get("score_writeback", "absorbed"))
        self.select_mode = str(kwargs.get("select_mode", "topk"))
        # Coverage selection is a one-shot quantisation of the prefill. Re-running
        # it every compress_interval would re-pick the whole anchor set from
        # scratch and churn the cache, so decode maintenance stays on top-k.
        self.select_decode = bool(kwargs.get("select_decode", False))
        self.gate_sigma = float(kwargs.get("gate_sigma", 0.0))
        self.cost_alpha = float(kwargs.get("cost_alpha", 0.8))
        self.lambda_pos = float(kwargs.get("lambda_pos", 0.0))
        self.compress_interval = int(kwargs.get("compress_interval", 32))
        self.chunk = int(kwargs.get("chunk", 1024))
        # Two scoring regimes matter and they disagree. Summing attention over
        # every query (H2O) predicts open-ended continuation best; summing only
        # the last W queries (SnapKV) is the only thing that works when the
        # prompt ends in a question, because nothing earlier marks the answer as
        # important. score_blend mixes them: 0 = all-query, 1 = window-only.
        self.observation_window = kwargs.get("observation_window", None)
        self.score_blend = float(kwargs.get("score_blend", 1.0))
        # Keep the two streams apart instead of blending them into one number,
        # so selection can use the window and the merge can use all-query mass.
        self.split_scores = bool(kwargs.get("split_scores", False))
        self.select_scope = str(kwargs.get("select_scope", "head"))
        # Pick the scoring regime per layer from the data instead of per task by
        # hand. See `_regime` for the statistic.
        self.auto_regime = bool(kwargs.get("auto_regime", False))
        self.auto_threshold = float(kwargs.get("auto_threshold", 0.73))
        self._regime_votes = {"qa": 0, "continuation": 0}

        # Transport is naturally per-head: keys, and therefore what is
        # interchangeable with what, differ head by head. Global mode is still
        # honoured -- scores are then shared, so every head selects the same
        # tokens and only the value merge stays per-head. Silently forcing
        # per_head=True here (as v1/v2 do) would compare a per-head OT-KV
        # against global-mode baselines, which is not a fair table.
        self.per_head = bool(kwargs.get("per_head", False))
        self.decode_steps = {}

    def get_middle_budget(self, layer_idx: int, total_tokens: int) -> int:
        if self.layer_budget != "pyramid":
            return super().get_middle_budget(layer_idx, total_tokens)
        """Depth-dependent budget.

        Scales the *base* middle budget the parent computed, rather than
        recomputing a layer budget from `total_tokens`. Recomputing is wrong once
        compression has happened: during decode `total_tokens` is the length of
        the already-compressed cache, so each pass shrinks the budget again and
        the cache collapses (measured: [90,70,50,30] -> [11,11,11,11] over 30
        decode steps in mode="prefill", where the budget must stay fixed).
        Scaling the parent's value inherits the correct prefill/entire semantics.
        """
        import math
        self.num_layers = max(self.num_layers or 0, layer_idx + 1)
        n = max(1, self.num_layers)
        depth = 0.0 if n == 1 else layer_idx / float(n - 1)
        scale = self.pyramid_low_scale + depth * (self.pyramid_high_scale - self.pyramid_low_scale)
        mean_scale = 0.5 * (self.pyramid_low_scale + self.pyramid_high_scale) or 1.0

        base_middle = super().get_middle_budget(layer_idx, total_tokens)
        return max(0, math.floor(base_middle * scale / mean_scale))

    @property
    def _dual(self) -> bool:
        """Keep both score streams instead of one blended stream.

        Only meaningful per-head: in global mode there is a single shared score
        vector and the channel axis would be mistaken for a head axis. Global
        mode falls back to the blended single stream.
        """
        return bool(self.split_scores and self.observation_window and self.per_head)

    def reduce_attention(self, attn_weights: torch.Tensor) -> torch.Tensor:
        a = attn_weights.detach()
        full = a.sum(dim=-2)

        w = self.observation_window
        n_q = a.shape[-2]
        if w is None or int(w) <= 0 or n_q <= int(w):
            return torch.stack([full, full]) if self._dual else full
        w = int(w)
        win = a[..., -w:, :].sum(dim=-2)

        if self._dual:
            # channel 0: all-query mass (reliable, used for transport capacity)
            # channel 1: window mass (recent, used to choose what to keep)
            return torch.stack([full, win * (n_q / w)])

        if self.score_blend <= 0.0:
            return full
        # put both on a per-query scale before mixing, then restore the
        # magnitude so accumulation across steps stays comparable
        blended = (1.0 - self.score_blend) * (full / n_q) + self.score_blend * (win / w)
        return blended * n_q

    def _grow_scores(self, layer_idx: int, key_states: torch.Tensor):
        if not self._dual:
            return super()._grow_scores(layer_idx, key_states)
        b, h, q_len, _ = key_states.shape
        shape = (2, b, h, q_len) if self.per_head else (2, q_len)
        new = torch.zeros(shape, device=key_states.device, dtype=torch.float32)
        cur = self.attention_scores.get(layer_idx)
        self.attention_scores[layer_idx] = new if cur is None else torch.cat([cur, new], dim=-1)

    def _attention_to_token_scores(self, attn_weights, target_shape=None):
        if attn_weights.dim() >= 3 and attn_weights.shape[0] == 2 and self._dual:
            inner = tuple(target_shape[1:]) if target_shape is not None else None
            return torch.stack([
                BaseCompressCache._attention_to_token_scores(self, attn_weights[c], inner)
                for c in range(2)])
        return BaseCompressCache._attention_to_token_scores(self, attn_weights, target_shape)

    def on_prefill(self, key_states, value_states, layer_idx, cache_kwargs):
        self._grow_scores(layer_idx, key_states)
        return key_states, value_states

    def on_prefill_end(self):
        total = self.get_seq_length()
        self._update_budget(total, is_prefill_end=True)
        for layer_idx in list(self.attention_scores.keys()):
            self._consume_attention_scores(layer_idx)
            # Reserve one interval's worth of slots. Compression runs only every
            # `compress_interval` decode steps, so without this the cache would
            # grow to budget + interval between compressions and OT-KV would
            # quietly hold more KV than the every-step baselines it is compared
            # against. With the reserve the *peak* equals the budget and the
            # mean sits below it, which errs against OT-KV rather than for it.
            self._compress_layer(layer_idx, total, reserve_tokens=self._reserve(layer_idx, total))

    def on_decode_step(self, key_states, value_states, layer_idx, cache_kwargs):
        q_len = key_states.shape[-2]
        self._consume_attention_scores(layer_idx)

        total_after = self.get_seq_length() + q_len
        self._update_budget(total_after, is_prefill_end=False)

        step = self.decode_steps.get(layer_idx, 0)
        if step > 0 and step % self.compress_interval == 0:
            self._compress_layer(layer_idx, total_after,
                                 reserve_tokens=max(q_len, self.compress_interval),
                                 allow_merge=self.merge_decode,
                                 allow_select=self.select_decode)
        self.decode_steps[layer_idx] = step + q_len

        self._grow_scores(layer_idx, key_states)
        return key_states, value_states

    @staticmethod
    def _norm_entropy(p: torch.Tensor) -> torch.Tensor:
        """Entropy of each row of `p` (unnormalised, non-negative), in [0, 1]."""
        q = p.clamp_min(EPS)
        q = q / q.sum(-1, keepdim=True)
        h = -(q * q.log()).sum(-1)
        return h / math.log(max(2, p.shape[-1]))

    def _regime(self, full: torch.Tensor, win: torch.Tensor) -> str:
        """Decide whether the prompt ends in a question or continues a document.

        A question at the tail makes the last queries attend sharply to the few
        tokens that answer it, so the window's attention is far more peaked than
        the all-query distribution. Open-ended continuation leaves the two with
        similar shape. The *ratio* of normalised entropies is the signal -- the
        raw difference is confounded, because a 32-query window is always more
        peaked than a sum over thousands of queries whatever the task, and every
        prompt then looks like a question.

        Measured on Qwen3-1.7B: 0.770-0.776 for wikitext continuation against
        0.660-0.692 for needle questions, so the default threshold sits between
        the two clusters. It costs one reduction over scores already held.
        """
        ratio = self._norm_entropy(win).mean() / self._norm_entropy(full).mean().clamp_min(EPS)
        return "qa" if float(ratio) < self.auto_threshold else "continuation"

    def _reserve(self, layer_idx: int, total_tokens: int) -> int:
        """Slots held back so the cache never exceeds the budget between
        compressions. Capped at a quarter of the middle budget: with a tiny
        budget an uncapped reserve would wipe the middle region outright.
        """
        budget = int(self.get_middle_budget(layer_idx, total_tokens))
        return int(min(self.compress_interval, max(0, budget // 4)))

    def _compress_layer(self, layer_idx: int, total_tokens: int, reserve_tokens: int = 0,
                        allow_merge: bool = True, allow_select: bool = True):
        middle_k, middle_v = self.get_middle_cache(layer_idx)
        if middle_k is None or middle_k.shape[-2] == 0:
            return

        budget = max(0, int(self.get_middle_budget(layer_idx, total_tokens)) - reserve_tokens)
        if budget >= middle_k.shape[-2]:
            return

        seq_len = self._get_existing_cache(layer_idx)[0].shape[-2]
        middle_start = self.sink_size
        middle_end = max(self.sink_size, seq_len - self.recent_size)

        scores = self.attention_scores[layer_idx]
        middle_scores = scores[..., middle_start:middle_end]

        aux, select_scores = None, None
        select_mode, select_scope = self.select_mode, self.select_scope
        if self._dual and scores.dim() >= 3 and scores.shape[0] == 2:
            aux = middle_scores                      # (2, ...) transported together
            middle_scores = aux[0]                   # all-query mass -> merge
            select_scores = aux[1]                   # window mass    -> selection
            if self.auto_regime:
                regime = self._regime(aux[0], aux[1])
                self._regime_votes[regime] += 1
                if regime == "continuation":
                    # nothing in the tail singles out an answer: rank on the
                    # reliable all-query mass, per head, and buy coverage
                    select_scores, select_mode, select_scope = None, "kmeans:50", "head"
                else:
                    select_mode, select_scope = "topk", "global"
        fz = scores[0] if aux is not None else scores
        frozen = (fz[..., :middle_start].sum(-1, keepdim=True)
                  + fz[..., middle_end:].sum(-1, keepdim=True))

        out = otkv_v3_compress(
            middle_k, middle_v, budget, middle_scores,
            merge=self.merge and allow_merge, top_r=self.top_r, epsilon=self.epsilon,
            sinkhorn_iters=self.sinkhorn_iters, capacity_beta=self.capacity_beta,
            sim_threshold=self.sim_threshold, merge_strength=self.merge_strength,
            score_mode=self.score_mode, chunk=self.chunk, key_merge=self.key_merge,
            aux_scores=aux, select_scores=select_scores,
            score_writeback=self.score_writeback,
            select_mode=select_mode if allow_select else "topk",
            gate_sigma=self.gate_sigma, cost_alpha=self.cost_alpha, lambda_pos=self.lambda_pos,
            select_scope=select_scope,
            frozen_mass=frozen if self.frozen_correction else None,
        )

        if aux is None:
            new_k, new_v, new_scores = out
        else:
            new_k, new_v, _, new_scores = out        # (2, ...) both channels

        self.replace_middle_cache(layer_idx, new_k, new_v)

        if new_scores.dim() != scores.dim():
            # global mode keeps a single (k_len,) score vector: fold the
            # per-head merged mass back the same way global scores accumulate
            new_scores = new_scores.sum(dim=tuple(range(new_scores.dim() - 1)))
        self.attention_scores[layer_idx] = torch.cat(
            [scores[..., :middle_start], new_scores, scores[..., middle_end:]], dim=-1)
