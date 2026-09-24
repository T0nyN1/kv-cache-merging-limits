"""OT-KV v8 -- rejectable local transport with full-output costs.

Pure functions implementing docs/otkv_v8_contract.md (from
research-wiki/otkv_v8_design.md). Everything is per KV head of one layer,
with real post-RoPE keys/queries and float32 (or float64, if given) maths.

Notation.  Cache positions 0..L-1; retained set A (bool keep_mask, |A| = m);
evicted set E = ~A minus the protected regions; fit queries (W, d); dense
attention output o(q); baseline output o0(q) = softmax over A only; p_j(q) the
baseline attention weight of retained slot j.

A merge of evicted i into retained j keeps k_j, replaces
    v'_j = (1 - theta) v_j + theta v_i            (0 <= theta <= 1)
and adds a per-slot logit bias log(c) (c >= 1).  With
    D(q)    = 1 + (c - 1) p_j(q)
    u(q)    = [o0(q) + (c - 1) p_j(q) v_j] / D(q)
    dvec(q) = c p_j(q) (v_i - v_j) / D(q)
the output after the single merge is exactly o_pair(q) = u(q) + theta dvec(q).

The routing between many evicted tokens and few receiving slots is a partial
transport problem with a zero-cost drop node (unit supply per evicted token,
capacity one per retained slot); its integral optimum is a maximum-weight
bipartite matching, solved exactly with scipy's linear_sum_assignment.

Conventions: everything torch, device-agnostic; no in-place edits of caller
tensors; identical-key detection by torch.allclose(k_i, k_j, atol=1e-6).
Causal handling: calibration queries are the last W prefill positions and every
candidate pair sits before the earliest calibration query, so the pairs are
visible to all calibration rows; the dense rows a(q) still use the causal mask
for the calibration block itself (pass `causal_positions`, the absolute
position of each query row; cache index == absolute position for a prefill
cache).

Additions beyond the contract's function list (kept optional / trailing):
  * `causal_positions=None` on baseline_outputs / validate_plan /
    evaluate_outputs, so the baseline over A can apply the same causal mask
    as the dense rows for the calibration block.
  * `v0=None` on evaluate_outputs: the un-merged values, needed to compute
    |o0 - o|^2 when v_new is a merged cache.
  * `time_split_rows`: the design's GQA rule (split the calibration window by
    TIME first, then group the query heads sharing a KV head into rows).
"""
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

EPS = 1e-8            # denominators / scale guard
KEY_ATOL = 1e-6       # identical-key (and identical-value) detection
GROUP_TOL = 1e-6      # "no query-head group worse" tolerance (relative)
_FIT_CHUNK = 512      # edges fitted per batch inside build_sparse_costs


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _work_dtype(*tensors: torch.Tensor) -> torch.dtype:
    dt = torch.float32
    for t in tensors:
        if t is not None and t.is_floating_point():
            dt = torch.promote_types(dt, t.dtype)
    return dt


def _as_bool(mask, n: int, device) -> torch.Tensor:
    m = torch.as_tensor(mask, device=device)
    if m.dtype != torch.bool:
        m = m != 0
    if m.numel() != n:
        raise ValueError(f"mask has {m.numel()} entries, expected {n}")
    return m.reshape(n)


def _softmax_rows(logits: torch.Tensor) -> torch.Tensor:
    """Row softmax; rows that are entirely -inf (nothing visible) become 0."""
    rows = torch.softmax(logits, dim=-1)
    return torch.nan_to_num(rows, nan=0.0)


def _causal_mask(logits: torch.Tensor, key_index: torch.Tensor,
                 causal_positions) -> torch.Tensor:
    cp = torch.as_tensor(causal_positions, device=logits.device).reshape(-1, 1)
    if cp.shape[0] != logits.shape[0]:
        raise ValueError("causal_positions must have one entry per query row")
    future = key_index.reshape(1, -1) > cp
    return logits.masked_fill(future, float("-inf"))


def _id_set(ids) -> set:
    if isinstance(ids, torch.Tensor):
        return set(int(x) for x in ids.detach().cpu().reshape(-1).tolist())
    return set(int(x) for x in ids)


def _plan_items(plan) -> List[Tuple[int, int, float, float]]:
    out = []
    for e in plan:
        i, j, c, theta = e[0], e[1], e[2], e[3]
        out.append((int(i), int(j), float(c), float(theta)))
    return out


# --------------------------------------------------------------------------- #
# attention primitives
# --------------------------------------------------------------------------- #
def attention_rows(q: torch.Tensor, k: torch.Tensor, causal_positions=None
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dense attention of queries q (W, d) over the whole cache k (L, d).

    Returns (logits (W, L), softmax rows (W, L)).  If `causal_positions` (W,)
    is given, cache entries with index > the query's absolute position are
    masked (cache index == absolute position for a prefill cache).
    """
    dt = _work_dtype(q, k)
    q = q.to(dt)
    k = k.to(dt)
    d = q.shape[-1]
    logits = q @ k.transpose(0, 1) / math.sqrt(d)
    if causal_positions is not None:
        logits = _causal_mask(logits, torch.arange(k.shape[0], device=q.device),
                              causal_positions)
    return logits, _softmax_rows(logits)


def baseline_outputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, keep_mask,
                     bias: Optional[torch.Tensor] = None, causal_positions=None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention restricted to the retained set A (+ optional per-slot logit bias).

    Returns o0 (W, d) and p (W, m): the softmax over A only (renormalised), so
    p sums to one per row.  `bias` may be (L,) indexed by cache position (the
    apply_plan convention; non-retained entries are ignored) or (m,) over A.
    """
    dt = _work_dtype(q, k, v)
    q = q.to(dt)
    k = k.to(dt)
    v = v.to(dt)
    L, d = k.shape
    keep = _as_bool(keep_mask, L, q.device)
    idx = torch.nonzero(keep).flatten()
    logits = q @ k[idx].transpose(0, 1) / math.sqrt(d)
    if bias is not None:
        b = torch.as_tensor(bias, device=q.device).to(dt).reshape(-1)
        if b.numel() == L:
            b = b[idx]
        elif b.numel() != idx.numel():
            raise ValueError(f"bias must have L={L} or m={idx.numel()} entries, got {b.numel()}")
        logits = logits + b.reshape(1, -1)
    if causal_positions is not None:
        logits = _causal_mask(logits, idx, causal_positions)
    p = _softmax_rows(logits)
    o0 = p @ v[idx]
    return o0, p


# --------------------------------------------------------------------------- #
# per-pair fit (batched core + the contract's single-edge wrapper)
# --------------------------------------------------------------------------- #
def _fit_pairs(a_i: torch.Tensor, a_j: torch.Tensor, p_j: torch.Tensor,
               v_i: torch.Tensor, v_j: torch.Tensor, o: torch.Tensor, o0: torch.Tensor,
               c_max: float, eps: float, identical_keys: torch.Tensor,
               force_c: Optional[float] = None, joint_c: int = 0,
               _fixed_c: Optional[torch.Tensor] = None, force_theta: Optional[float] = None):
    """Vectorised over E edges.

    a_i, a_j, p_j : (E, W)   v_i, v_j : (E, d)   o, o0 : (W, d)   identical_keys : (E,) bool
    force_c : None = fit c (with the exact branch for identical keys); a float = use that c
              for every edge, fit theta only, never reject (the no-bias ablation, c = 1).
              With force_c the identical-key theta = 1/2 override does NOT apply: exactness
              needs c = 2 AND theta = 1/2 together.
    force_theta : if given, theta is fixed at this value for every edge (bias-only
              ablation with force_theta = 0: the receiver keeps its own value and
              only its mass is re-fitted).
    joint_c : if > 0, after the mass fit a one-dimensional search over c on a
              log grid of `joint_c` points in [c*/2, 2c*] (clipped to [1, c_max])
              with the conditional optimum theta(c) picks the c with the
              smallest total fit error of the FULL output (joint (c, theta)
              fit); identical-key edges keep the exact branch.
    Returns c (E,), theta (E,), err (E, W), accepted (E,) bool, reason: list[str].
    """
    E, W = a_j.shape
    dt = o.dtype
    forced = force_c is not None
    if joint_c and not forced:
        base = _fit_pairs(a_i, a_j, p_j, v_i, v_j, o, o0, c_max, eps, identical_keys,
                          force_theta=force_theta)
        c0, th0, err0, acc0, reason0 = base
        G = int(joint_c)
        # one vectorised fit over the whole grid: edge e, grid point g -> row e*G + g
        fac = torch.logspace(-1.0, 1.0, G, base=2.0).to(device=c0.device, dtype=c0.dtype)   # (no MPS kernel for logspace)
        cg = torch.clamp(c0.unsqueeze(1) * fac.unsqueeze(0), 1.0, float(c_max)).reshape(-1)   # (E*G,)
        rep = lambda t: t.repeat_interleave(G, dim=0)
        _, thg, errg, _, _ = _fit_pairs(rep(a_i), rep(a_j), rep(p_j), rep(v_i), rep(v_j), o, o0, c_max, eps,
                                        torch.zeros(E * G, dtype=torch.bool, device=c0.device), force_c=None,
                                        joint_c=0, _fixed_c=cg, force_theta=force_theta)
        tot = errg.sum(dim=1).reshape(E, G)
        base_tot = err0.sum(dim=1)
        # first grid point (in grid order) that beats the mass-fit total, ties -> the earlier one
        best_g = torch.argmin(tot, dim=1)
        tot_best = tot.gather(1, best_g.unsqueeze(1)).squeeze(1)
        better = (tot_best < base_tot) & acc0 & ~identical_keys
        sel = torch.arange(E, device=c0.device) * G + best_g
        best_c = torch.where(better, cg[sel], c0)
        best_th = torch.where(better, thg[sel], th0)
        best_err = torch.where(better.unsqueeze(1), errg[sel], err0)
        return best_c, best_th, best_err, acc0, reason0
    if forced:
        identical_keys = torch.zeros_like(identical_keys)
    # -- 4.1 mass fit: c* = sum(a_j m)/sum(a_j^2) = 1 + sum(a_i a_j)/sum(a_j^2) (identical
    #    algebra; the second form keeps c >= 1 in floating point since a_i >= 0)
    den_c = (a_j * a_j).sum(dim=1)
    bad_den = (den_c < eps) & (not forced)
    c = 1.0 + (a_i * a_j).sum(dim=1) / torch.where(den_c < eps, torch.ones_like(den_c), den_c)
    c = torch.where(den_c < eps, torch.ones_like(c), c)
    if forced:
        c = torch.full_like(c, float(force_c))
    if _fixed_c is not None:                      # internal: per-edge c from the joint search
        c = _fixed_c.to(c.dtype)
        bad_den = torch.zeros_like(bad_den)
    over = (c > c_max) & ~identical_keys & (not forced) & (_fixed_c is None)
    c = torch.where(identical_keys, torch.full_like(c, 2.0), c)        # exact branch

    # -- 4.2 value fit with the full softmax.
    #
    # The design writes the merged output of one edge as
    #     o_pair(q) = u(q) + theta * dvec(q),
    #     D = 1 + (c-1) p_j,  u = (o0 + (c-1) p_j v_j)/D,  dvec = c p_j (v_i - v_j)/D,
    # which as written needs the (E, W, d) tensors u and dvec.  Everything the
    # fit and the error need are inner products over d, so we never form them:
    # with A(q) = o0(q) - o(q), B(e,q) = v_j(e) - o(q), w(e) = v_i(e) - v_j(e),
    #     u + theta dvec - o = [A + alpha B + theta gamma w] / D,
    #     alpha = (c-1) p_j,  gamma = c p_j,
    # so the squared error expands into terms built from four (E, d) x (d, W)
    # products and a few (E,) / (W,) norms -- O(EW) memory instead of O(EWd)
    # (d = 128 in the models measured, and the joint-c grid multiplies E by G).
    # The differenced form A + alpha B + theta gamma w is used rather than the
    # raw squares so the cancellation at alpha -> 0 stays exact in float32.
    cm1 = (c - 1.0).unsqueeze(1)                                        # (E,1)
    alpha = cm1 * p_j                                                   # (E,W)
    D = 1.0 + alpha                                                     # (E,W)
    gamma = c.unsqueeze(1) * p_j                                        # (E,W)
    w = v_i - v_j                                                       # (E,d)

    A2 = ((o0 - o) ** 2).sum(dim=1)                                     # (W,)
    o2 = (o * o).sum(dim=1)                                             # (W,)
    o0o = (o0 * o).sum(dim=1)                                           # (W,)
    o0v = v_j @ o0.transpose(0, 1)                                      # (E,W)  o0(q).v_j(e)
    ov = v_j @ o.transpose(0, 1)                                        # (E,W)  o(q).v_j(e)
    o0w = w @ o0.transpose(0, 1)                                        # (E,W)  o0(q).w(e)
    ow = w @ o.transpose(0, 1)                                          # (E,W)  o(q).w(e)
    v2 = (v_j * v_j).sum(dim=1).unsqueeze(1)                            # (E,1)
    w2 = (w * w).sum(dim=1).unsqueeze(1)                                # (E,1)
    vw = (v_j * w).sum(dim=1).unsqueeze(1)                              # (E,1)

    AB = o0v - o0o.unsqueeze(0) - ov + o2.unsqueeze(0)                  # A(q).B(e,q)
    Aw = o0w - ow                                                       # A(q).w(e)
    Bw = vw - ow                                                        # B(e,q).w(e)
    B2 = v2 - 2.0 * ov + o2.unsqueeze(0)                                # |B(e,q)|^2

    # theta* = argmin_theta sum_q |A + alpha B + theta gamma w|^2 / D^2
    g_over_d2 = gamma / (D * D)
    num_t = -(g_over_d2 * (Aw + alpha * Bw)).sum(dim=1)                 # = sum dvec.(o-u)
    den_t = ((gamma / D) ** 2 * w2).sum(dim=1)                          # = sum |dvec|^2
    same_v = (w.abs() <= KEY_ATOL + 1e-5 * v_j.abs()).all(dim=1)
    half = den_t < eps
    theta = torch.clamp(num_t / torch.where(half, torch.ones_like(den_t), den_t), 0.0, 1.0)
    theta = torch.where(half | same_v | identical_keys, torch.full_like(theta, 0.5), theta)
    theta = torch.where(bad_den, torch.zeros_like(theta), theta)       # degenerate: no change
    if force_theta is not None:
        theta = torch.full_like(theta, float(force_theta))

    th = theta.unsqueeze(1)                                             # (E,1)
    tg = th * gamma                                                     # theta * gamma
    num = (A2.unsqueeze(0) + alpha * alpha * B2 + tg * tg * w2
           + 2.0 * alpha * AB + 2.0 * tg * Aw + 2.0 * alpha * tg * Bw)
    err = num / (D * D) - A2.unsqueeze(0)
    err = torch.where(bad_den.unsqueeze(1), torch.zeros_like(err), err)

    accepted = ~bad_den & ~over
    # reasons from three host transfers, never one per edge (a per-edge
    # bool(tensor[e]) is a device sync each; that made the deployed cache
    # ~80 s per 8k-token sample on an H200)
    bad_l, id_l, over_l = bad_den.cpu().tolist(), identical_keys.cpu().tolist(), over.cpu().tolist()
    reason = ["mass_denominator" if b else ("exact_duplicate_key" if i else ("c_max" if o else "ok"))
              for b, i, o in zip(bad_l, id_l, over_l)]
    return c.to(dt), theta.to(dt), err, accepted, reason


def fit_pair(a_i: torch.Tensor, a_j: torch.Tensor, p_j: torch.Tensor,
             v_i: torch.Tensor, v_j: torch.Tensor, o: torch.Tensor, o0: torch.Tensor,
             c_max: float = 4.0, eps: float = 1e-8, identical_keys: bool = False) -> Dict:
    """Fit one merge edge (evicted i -> retained j) on the fit queries.

    a_i, a_j : dense attention of i / j over the fit queries (W,)
    p_j      : baseline (over A) weight of j (W,)
    v_i, v_j : (d,);  o, o0 : (W, d) dense / baseline outputs.
    Returns dict(c, theta, err (W,), accepted: bool, reason: str).

    c*    = sum(a_j m)/sum(a_j^2), m = a_i + a_j; reject if sum(a_j^2) < eps or c* > c_max.
    D     = 1 + (c-1) p_j;  u = (o0 + (c-1) p_j v_j)/D;  dvec = c p_j (v_i - v_j)/D
    theta*= clip(sum(dvec.(o-u)) / sum(|dvec|^2), 0, 1); theta = 1/2 if v_i == v_j or the
            denominator < eps.  identical_keys=True -> c = 2, theta = 1/2 (exact branch).
    err(q)= |u + theta dvec - o|^2 - |o0 - o|^2   (negative = the merge helps on that query).
    """
    dt = _work_dtype(a_i, a_j, p_j, v_i, v_j, o, o0)
    o = o.to(dt)
    o0 = o0.to(dt)
    c, theta, err, acc, reason = _fit_pairs(
        a_i.to(dt).reshape(1, -1), a_j.to(dt).reshape(1, -1), p_j.to(dt).reshape(1, -1),
        v_i.to(dt).reshape(1, -1), v_j.to(dt).reshape(1, -1), o, o0,
        float(c_max), float(eps),
        torch.tensor([bool(identical_keys)], device=o.device))
    return dict(c=float(c[0]), theta=float(theta[0]), err=err[0],
                accepted=bool(acc[0]), reason=reason[0])


# --------------------------------------------------------------------------- #
# candidate edges and their transport costs
# --------------------------------------------------------------------------- #
def build_sparse_costs(k: torch.Tensor, v: torch.Tensor, q_fit: torch.Tensor, pos,
                       keep_mask, protect_mask, o: torch.Tensor, o0: torch.Tensor,
                       p: torch.Tensor, a_dense: torch.Tensor,
                       max_dist: int = 8, max_cand: int = 4, n_blocks: int = 4,
                       c_max: float = 4.0, sep_ids=None, joint_c: int = 0,
                       force_theta: Optional[float] = None
                       ) -> List[Tuple[int, int, float, float, float]]:
    """Candidate edges with their transport costs, for one KV head.

    k, v (L, d); q_fit (W, d) fit queries (rows time-ordered, position-major for
    GQA groups); pos (L,) absolute positions of the cache entries; keep_mask /
    protect_mask (L,) bool; o, o0 (W, d) dense / baseline outputs on the fit
    rows; p (W, m) baseline weights over A; a_dense (W, L) dense attention rows.

    For every evicted i (~keep & ~protect): up to `max_cand` nearest retained,
    unprotected j with |pos_i - pos_j| <= max_dist and no separator position
    strictly between pos_i and pos_j (`sep_ids`: separator positions, in the
    units of `pos`).  Per edge: fit_pair (exact branch when
    allclose(k_i, k_j, atol=1e-6)); cost
        C_ij = max over n_blocks contiguous time blocks of mean_block(err) / scale,
        scale = mean_fit |o|^2 + eps           (one scale per head).
    Only accepted edges with C_ij < 0 are returned, as (i, j, C_ij, c, theta).
    """
    dt = _work_dtype(k, v, q_fit, o, o0, p, a_dense)
    device = k.device
    k = k.to(dt)
    v = v.to(dt)
    o = o.to(dt)
    o0 = o0.to(dt)
    p = p.to(dt)
    a_dense = a_dense.to(dt)
    L = k.shape[0]
    W = o.shape[0]
    keep = _as_bool(keep_mask, L, device)
    prot = _as_bool(protect_mask, L, device)
    pos_t = torch.as_tensor(pos, device=device).reshape(-1)
    if pos_t.numel() != L:
        raise ValueError("pos must have one entry per cache slot")
    if a_dense.shape != (W, L) or o0.shape != (W, k.shape[1]):
        raise ValueError("a_dense must be (W, L) and o, o0 (W, d) on the fit rows")
    slot_of = torch.cumsum(keep.long(), dim=0) - 1        # cache index -> slot in A
    if p.shape != (W, int(keep.sum())):
        raise ValueError(f"p must be (W, m) with m={int(keep.sum())}, got {tuple(p.shape)}")

    # candidate search on the CPU (positions only; avoids one device sync per source)
    pos_l = pos_t.detach().cpu().tolist()
    keep_l = keep.detach().cpu().tolist()
    prot_l = prot.detach().cpu().tolist()
    seps = sorted(set(float(s) for s in torch.as_tensor(sep_ids).detach().cpu().reshape(-1).tolist())) \
        if sep_ids is not None else []
    receivers = [j for j in range(L) if keep_l[j] and not prot_l[j]]
    sources = [i for i in range(L) if (not keep_l[i]) and (not prot_l[i])]
    if not receivers or not sources:
        return []

    def crosses(pa, pb) -> bool:
        lo, hi = (pa, pb) if pa <= pb else (pb, pa)
        return any(lo < s < hi for s in seps)

    recv_pos = torch.tensor([pos_l[j] for j in receivers], dtype=torch.float64)
    recv_pos_l = recv_pos.tolist()
    pairs: List[Tuple[int, int]] = []
    for i in sources:
        near = torch.nonzero((recv_pos - pos_l[i]).abs() <= max_dist).flatten().tolist()
        cands = []
        for r in near:
            j = receivers[r]
            if crosses(pos_l[i], recv_pos_l[r]):
                continue
            cands.append((abs(recv_pos_l[r] - pos_l[i]), recv_pos_l[r], j))
        cands.sort()
        for _, _, j in cands[:max_cand]:
            pairs.append((i, j))
    if not pairs:
        return []

    scale = float((o * o).sum(dim=1).mean()) + EPS
    nb = max(1, min(int(n_blocks), W))
    edges: List[Tuple[int, int, float, float, float]] = []
    for start in range(0, len(pairs), _FIT_CHUNK):
        chunk = pairs[start:start + _FIT_CHUNK]
        ii = torch.tensor([i for i, _ in chunk], device=device)
        jj = torch.tensor([j for _, j in chunk], device=device)
        same_k = (k[ii] - k[jj]).abs() <= KEY_ATOL + 1e-5 * k[jj].abs()      # allclose(atol=1e-6)
        same_k = same_k.all(dim=1)
        c, theta, err, acc, _ = _fit_pairs(
            a_dense[:, ii].transpose(0, 1), a_dense[:, jj].transpose(0, 1),
            p[:, slot_of[jj]].transpose(0, 1), v[ii], v[jj], o, o0,
            float(c_max), EPS, same_k, joint_c=int(joint_c), force_theta=force_theta)
        block_means = torch.stack([b.mean(dim=1) for b in torch.tensor_split(err, nb, dim=1)],
                                  dim=1)                                          # (E, nb)
        cost = block_means.max(dim=1).values / scale
        keep_edge = (acc & (cost < 0)).detach().cpu().tolist()          # one transfer per chunk
        cost_l, c_l, th_l = cost.detach().cpu().tolist(), c.detach().cpu().tolist(), theta.detach().cpu().tolist()
        for e, ok in enumerate(keep_edge):
            if ok:
                edges.append((chunk[e][0], chunk[e][1], cost_l[e], c_l[e], th_l[e]))
    return edges


# --------------------------------------------------------------------------- #
# partial transport with a drop node
# --------------------------------------------------------------------------- #
def solve_partial_transport(edges: Sequence, n_evicted_ids, retained_ids,
                            method: str = "matching") -> List[Tuple[int, int, float, float]]:
    """Route evicted tokens to retained slots or to the zero-cost drop node.

        minimise  sum C_ij x_ij
        s.t.      sum_j x_ij + x_i,drop = 1   (unit supply per evicted i)
                  sum_i x_ij <= 1            (capacity one per retained j)

    edges: (i, j, C_ij, c, theta) from build_sparse_costs (C_ij < 0).
    n_evicted_ids: the evicted cache ids (iterable); an int is accepted and then
    the sources are taken from the edges.  retained_ids: the retained cache ids.
    method="matching": exact -- maximum-weight bipartite matching (weights
    -C_ij > 0) via scipy.optimize.linear_sum_assignment on a dense matrix with
    one zero-cost drop column per source.  method="greedy": most-negative cost
    first, respecting both capacities (the comparator for hypothesis 3).
    Returns the chosen (i, j, c, theta), sorted by i.
    """
    if method not in ("matching", "greedy"):
        raise ValueError(f"unknown method {method!r}")
    ev = None if isinstance(n_evicted_ids, int) else _id_set(n_evicted_ids)
    re_ = None if retained_ids is None else _id_set(retained_ids)

    best: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
    for e in edges:
        i, j, cost, c, theta = int(e[0]), int(e[1]), float(e[2]), float(e[3]), float(e[4])
        if ev is not None and i not in ev:
            raise ValueError(f"edge source {i} is not an evicted id")
        if re_ is not None and j not in re_:
            raise ValueError(f"edge target {j} is not a retained id")
        if cost >= 0:
            continue                                     # never better than dropping
        if (i, j) not in best or cost < best[(i, j)][0]:
            best[(i, j)] = (cost, c, theta)
    if not best:
        return []

    chosen: List[Tuple[int, int, float, float]] = []
    if method == "greedy":
        used_i, used_j = set(), set()
        for (i, j), (cost, c, theta) in sorted(best.items(), key=lambda kv: (kv[1][0], kv[0])):
            if i in used_i or j in used_j:
                continue
            used_i.add(i)
            used_j.add(j)
            chosen.append((i, j, c, theta))
    else:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        srcs = sorted(set(i for i, _ in best))
        tgts = sorted(set(j for _, j in best))
        si = {i: r for r, i in enumerate(srcs)}
        tj = {j: col for col, j in enumerate(tgts)}
        n_s, n_t = len(srcs), len(tgts)
        # Rectangular formulation, equivalent to explicit drop columns: a
        # non-edge cell costs 0, exactly what dropping the source (or leaving
        # the receiver unused) costs, so the optimum of the (n_s x n_t)
        # assignment equals the optimum of the partial transport; rows or
        # columns left unassigned are drops.  (The (n_s x (n_t + n_s)) matrix
        # with explicit drop columns made the deployed cache O(n_s^2) per head.)
        cost_m = np.zeros((n_s, n_t), dtype=np.float64)
        for (i, j), (cost, _, _) in best.items():
            cost_m[si[i], tj[j]] = cost
        rows, cols = linear_sum_assignment(cost_m)
        for r, col in zip(rows.tolist(), cols.tolist()):
            if cost_m[r, col] < 0:
                i, j = srcs[r], tgts[col]
                _, c, theta = best[(i, j)]
                chosen.append((i, j, c, theta))
    chosen.sort()
    return chosen


# --------------------------------------------------------------------------- #
# applying, validating and evaluating a plan
# --------------------------------------------------------------------------- #
def apply_plan(v: torch.Tensor, keep_mask, plan: Sequence, gamma: float
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Write a plan into the values: v_j <- v_j + gamma*theta*(v_i - v_j) and a
    per-slot logit bias (L,) = gamma*log(c) on merged j, 0 elsewhere.

    Returns (v_new (L, d), bias (L,)); the caller's tensors are not modified.
    Raises on overlapping pairs or on a pair whose endpoints contradict keep_mask.
    """
    dt = _work_dtype(v)
    v = v.to(dt)
    L = v.shape[0]
    keep = _as_bool(keep_mask, L, v.device)
    v_new = v.clone()
    bias = torch.zeros(L, dtype=dt, device=v.device)
    gamma = float(gamma)
    items = _plan_items(plan)
    if not items:
        return v_new, bias
    keep_l = keep.detach().cpu().tolist()
    used = set()
    for i, j, c, theta in items:
        if not (0 <= i < L and 0 <= j < L):
            raise ValueError(f"pair ({i}, {j}) outside the cache")
        if not keep_l[j] or keep_l[i]:
            raise ValueError(f"pair ({i}, {j}) must route an evicted token into a retained slot")
        if i in used or j in used:
            raise ValueError(f"pair ({i}, {j}) overlaps another pair (cluster size must be <= 2)")
        if not (0.0 <= theta <= 1.0) or c <= 0:
            raise ValueError(f"pair ({i}, {j}) has theta={theta}, c={c}")
        used.add(i)
        used.add(j)
    # pairs are disjoint, so the vectorised writes never collide
    ii = torch.tensor([it[0] for it in items], device=v.device)
    jj = torch.tensor([it[1] for it in items], device=v.device)
    th = torch.tensor([gamma * it[3] for it in items], dtype=dt, device=v.device)
    lb = torch.tensor([gamma * math.log(it[2]) for it in items], dtype=dt, device=v.device)
    v_new[jj] = v[jj] + th.unsqueeze(1) * (v[ii] - v[jj])
    bias[jj] = lb
    return v_new, bias


def validate_plan(q_sel: torch.Tensor, k: torch.Tensor, v: torch.Tensor, keep_mask,
                  plan: Sequence, o_sel: torch.Tensor,
                  gammas: Sequence[float] = (0.0, 0.25, 0.5, 1.0), min_gain: float = 0.05,
                  per_head_rows: Optional[Sequence[torch.Tensor]] = None,
                  causal_positions=None) -> Tuple[float, float, List[float]]:
    """Joint re-check of the whole plan on the SELECT queries and the gamma choice.

    For each gamma the plan is applied jointly (values and bias) and the full
    attention output over A is recomputed on q_sel.  improvement(gamma) =
    1 - mean|o_pair - o|^2 / mean|o0 - o|^2; among the gammas whose improvement
    is >= min_gain AND for which no query-head group (per_head_rows: list of
    row-index tensors, one per query head sharing this KV head; default one
    group) is worse than baseline, the one with the largest improvement is
    chosen (best feasible gamma); if none qualifies, gamma = 0.
    Returns (gamma, improvement, per-group improvements) for the chosen gamma.
    """
    dt = _work_dtype(q_sel, k, v, o_sel)
    q = q_sel.to(dt)
    o_sel = o_sel.to(dt)
    W = q.shape[0]
    if per_head_rows is None:
        groups = [torch.arange(W, device=q.device)]
    else:
        groups = [torch.as_tensor(r, device=q.device).reshape(-1) for r in per_head_rows]
    zeros = [0.0] * len(groups)
    plan = _plan_items(plan)
    o0, _ = baseline_outputs(q, k, v, keep_mask, None, causal_positions)
    e0 = ((o0 - o_sel) ** 2).sum(dim=1)
    base = float(e0.mean())
    if not plan or base <= 0.0:
        return 0.0, 0.0, zeros

    def rel(e_new: torch.Tensor, rows: torch.Tensor) -> float:
        b = float(e0[rows].mean()) if rows.numel() else 0.0
        return 0.0 if b <= 0.0 else 1.0 - float(e_new[rows].mean()) / b

    # policy: among the gammas that satisfy BOTH constraints (mean improvement
    # >= min_gain and no query-head group worse than baseline), take the one
    # with the largest mean improvement; if none qualifies, gamma = 0.
    best_gamma, best_imp, best_groups = 0.0, 0.0, zeros
    for g in gammas:
        g = float(g)
        if g <= 0.0:
            continue
        v_g, b_g = apply_plan(v, keep_mask, plan, g)
        o_g, _ = baseline_outputs(q, k, v_g, keep_mask, b_g, causal_positions)
        e_g = ((o_g - o_sel) ** 2).sum(dim=1)
        imp = 1.0 - float(e_g.mean()) / base
        grp = [rel(e_g, r) for r in groups]
        if imp >= min_gain and min(grp) >= -GROUP_TOL and imp > best_imp:
            best_gamma, best_imp, best_groups = g, imp, grp
    return best_gamma, best_imp, best_groups


def evaluate_outputs(q: torch.Tensor, k: torch.Tensor, v_new: torch.Tensor, keep_mask,
                     bias: Optional[torch.Tensor], o: torch.Tensor,
                     v0: Optional[torch.Tensor] = None, causal_positions=None
                     ) -> Tuple[float, float]:
    """mean |o' - o|^2 (merged cache: v_new + bias over A) and mean |o0 - o|^2
    (baseline eviction) on any query set.  `v0` = the un-merged values; if
    omitted, v_new is used for the baseline too (correct only when v_new is
    the un-merged cache).
    """
    dt = _work_dtype(q, k, v_new, o)
    o = o.to(dt)
    o_new, _ = baseline_outputs(q, k, v_new, keep_mask, bias, causal_positions)
    o_base, _ = baseline_outputs(q, k, v_new if v0 is None else v0, keep_mask, None,
                                 causal_positions)
    err_new = float(((o_new - o) ** 2).sum(dim=1).mean())
    err_base = float(((o_base - o) ** 2).sum(dim=1).mean())
    return err_new, err_base


# --------------------------------------------------------------------------- #
# GQA calibration split
# --------------------------------------------------------------------------- #
def time_split_rows(n_pos: int, n_heads: int = 1,
                    fractions: Sequence[float] = (0.5, 0.25, 0.25)) -> Dict[str, object]:
    """Split the calibration window by TIME, then group the query heads.

    The W = n_pos * n_heads calibration rows are assumed position-major
    (row = t * n_heads + h, i.e. q.reshape(n_pos * n_heads, d) for q of shape
    (n_pos, n_heads, d)); the first, second and third time segments (default
    1/2, 1/4, 1/4 of the positions) become fit / select / audit.  Returns
    {'fit','select','audit'}: row tensors into the window rows (position-major,
    so contiguous chunks are time blocks); '<split>_heads': list of per-head
    row tensors LOCAL to that split (indices into q_rows[split['<split>']], as
    validate_plan's per_head_rows expects); '<split>_pos': window position
    indices of the split.
    """
    if n_pos <= 0 or n_heads <= 0:
        raise ValueError("n_pos and n_heads must be positive")
    n_fit = int(round(n_pos * fractions[0]))
    n_sel = int(round(n_pos * fractions[1]))
    n_fit = max(0, min(n_fit, n_pos))
    n_sel = max(0, min(n_sel, n_pos - n_fit))
    bounds = {"fit": (0, n_fit), "select": (n_fit, n_fit + n_sel), "audit": (n_fit + n_sel, n_pos)}
    out: Dict[str, object] = {}
    for name, (a, b) in bounds.items():
        t = torch.arange(a, b)
        rows = (t.unsqueeze(1) * n_heads + torch.arange(n_heads).unsqueeze(0)).reshape(-1)
        out[name] = rows
        out[name + "_pos"] = t
        t_local = torch.arange(b - a)
        out[name + "_heads"] = [t_local * n_heads + h for h in range(n_heads)]
    return out


__all__ = ["attention_rows", "baseline_outputs", "fit_pair", "build_sparse_costs",
           "solve_partial_transport", "apply_plan", "validate_plan", "evaluate_outputs",
           "time_split_rows"]
