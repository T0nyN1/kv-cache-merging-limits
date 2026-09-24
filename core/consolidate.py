"""Compensated pair consolidation — the one merge the theory says can pay.

`docs/ot_kv_theory.md` closes *value-coefficient* merging: replacing a token's
query-dependent weight R(q)=e^{q·Δ/√d} with a constant leaves 1−e^{−σ²} of the
error, and σ ≈ 1.2 everywhere. That theorem assumes the merged slot can only
carry a value. The 2025-26 merging literature (KeepKV's electoral votes,
SelKV's attention-ratio bias) adds a second channel the theorem does not
cover: a per-slot **logit bias** b, so a slot represents e^{q·k+b}·v.

With that channel, merging a *pair* (1,2) into (k̄, v̄, b) where

    k̄ = (m₁k₁ + m₂k₂)/(m₁+m₂)          (logits are linear in k, so the merged
                                          logit is the mass-weighted mean)
    v̄ = (m₁v₁ + m₂v₂)/(m₁+m₂)
    b  = E_w[ log(a₁w+a₂w) − l̄w ]        (fits the group's exp-mass exactly on
                                          the observed queries; softmax Z cancels)

leaves the per-query log-error  ε(q) = logcosh-type curvature of the pair's
logit gap — **second order** in the gap where value-only merging is first
order. Measured directly, per pair, from the stored window rows:

    ε_w = log(a₁w + a₂w) − l̄w − b,       err = std over queries and heads.

Pairs with err below a break-even threshold are safe to consolidate; the freed
slots admit additional tokens at the same physical budget. Selecting which
non-overlapping adjacent pairs to fuse is a matching problem — the honest
descendant of the project's optimal-transport framing.

Everything here is measured in the same space the theory note prescribes
(log-attention response), not key cosine — the criterion the prior merging
literature gates on, which correlates with the deciding quantity at 0.273.
"""

from typing import Optional, Tuple

import torch

LOG_FLOOR = 1e-30


def pair_statistics(log_rows: torch.Tensor, left: torch.Tensor, right: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-pair merge statistics from stored window log-attention rows.

    log_rows : (B, H, L, W)  log attention of each token under W window queries
    left/right: (B, H, P)    token indices forming candidate pairs

    Returns (err, bias, w_left) each (B, H, P):
      err    — std over queries of the residual the merged slot cannot express
      bias   — the logit bias the merged slot must carry
      w_left — the left token's share of the pair mass (for k/v weighting)
    """
    W = log_rows.shape[-1]
    idx_l = left.unsqueeze(-1).expand(-1, -1, -1, W)
    idx_r = right.unsqueeze(-1).expand(-1, -1, -1, W)
    xl = torch.gather(log_rows, 2, idx_l)               # (B,H,P,W)
    xr = torch.gather(log_rows, 2, idx_r)

    # mass weights: mean attention over the window queries
    ml = xl.exp().mean(dim=-1)                          # (B,H,P)
    mr = xr.exp().mean(dim=-1)
    w_left = ml / (ml + mr).clamp_min(LOG_FLOOR)

    # merged-key logit is the mass-weighted mean of the member logits
    l_bar = w_left.unsqueeze(-1) * xl + (1.0 - w_left).unsqueeze(-1) * xr
    group = torch.logaddexp(xl, xr)                     # log(a1+a2), Z cancels below
    resid = group - l_bar                               # (B,H,P,W)

    bias = resid.mean(dim=-1)
    err = resid.std(dim=-1, unbiased=False)
    return err, bias, w_left


def select_pairs(err_pooled: torch.Tensor, eligible: torch.Tensor, tau: float,
                 max_pairs: int) -> torch.Tensor:
    """Greedy non-overlapping selection of adjacent pairs, lowest error first.

    err_pooled : (P,) pair error pooled over batch/heads (rms)
    eligible   : (P,) bool — positional adjacency etc.
    Returns a bool mask (P,) of chosen pairs. Adjacent pairs share tokens
    (pair p couples admitted slots p and p+1), so chosen pairs may not touch.
    """
    P = err_pooled.shape[0]
    chosen = torch.zeros(P, dtype=torch.bool)
    if max_pairs <= 0:
        return chosen
    order = torch.argsort(err_pooled)
    used = torch.zeros(P + 1, dtype=torch.bool)         # admitted-slot usage
    taken = 0
    for p in order.tolist():
        if taken >= max_pairs:
            break
        if not bool(eligible[p]) or float(err_pooled[p]) > tau:
            continue
        if used[p] or used[p + 1]:
            continue
        chosen[p] = True
        used[p] = used[p + 1] = True
        taken += 1
    return chosen


def consolidate_pairs(k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor,
                      log_rows: torch.Tensor, scores: torch.Tensor,
                      aux: Optional[torch.Tensor],
                      target_slots: int, tau: float, gap: int = 1,
                      gate: str = "sigma",
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                 Optional[torch.Tensor], torch.Tensor, int]:
    """Fuse qualifying adjacent pairs among admitted tokens down to target_slots.

    k, v      : (B, H, N, D) admitted middle tokens (N = target + over-admit)
    positions : (B, H, N)    original middle positions (sorted ascending)
    log_rows  : (B, H, N, W) log window attention of the admitted tokens
    scores    : (C?, B, H, N) score channels riding along (dual streams)
    aux       : optional extra channel stack shaped like scores
    Returns (k', v', scores', aux', bias', n_merged) with second dim N' and a
    per-slot logit bias (B, H, N'). If fewer pairs qualify than needed, the
    lowest-score surplus tokens are dropped instead, so N' == target_slots
    whenever N >= target_slots.
    """
    B, H, N, D = k.shape
    n_over = max(0, N - target_slots)
    if N < 2 or n_over == 0:
        bias = k.new_zeros(B, H, N, dtype=torch.float32)
        return k, v, scores, aux, bias, 0

    left = torch.arange(N - 1, device=k.device).view(1, 1, -1).expand(B, H, -1)
    right = left + 1
    err, bias_pair, w_left = pair_statistics(log_rows.float(), left, right)

    pos_gap = (positions[..., 1:] - positions[..., :-1])
    eligible = (pos_gap <= gap)
    if gate == "cos":
        # ablation: gate on key cosine (the criterion the merging literature
        # uses), everything else identical
        kn = torch.nn.functional.normalize(k.float(), dim=-1)
        cos = (kn[..., :-1, :] * kn[..., 1:, :]).sum(-1)
        err_gate = 1.0 - cos
    elif gate == "none":
        err_gate = torch.zeros_like(err)
    else:
        err_gate = err

    # one pair structure for every head (the cache is rectangular per layer):
    # pool the gate over batch and heads, rms so a single bad head vetoes
    err_pooled = err_gate.pow(2).mean(dim=(0, 1)).sqrt().cpu()
    eligible_pooled = (eligible.all(dim=0).all(dim=0) if eligible.dim() == 3 else eligible).cpu()

    chosen = select_pairs(err_pooled, eligible_pooled, tau, n_over)
    n_merged = int(chosen.sum())

    keep_mask = torch.ones(N, dtype=torch.bool)
    merge_left = torch.zeros(N, dtype=torch.bool)
    for p in torch.nonzero(chosen).flatten().tolist():
        merge_left[p] = True
        keep_mask[p + 1] = False                        # right member folds into left

    # not enough safe pairs: drop the weakest surplus tokens instead
    deficit = n_over - n_merged
    if deficit > 0:
        sel_strength = scores.reshape(-1, B, H, N)[-1].sum(dim=(0, 1)).cpu() \
            if scores.dim() >= 3 else scores.cpu()
        protected = merge_left.clone()
        protected[1:] |= merge_left[:-1]                # both members of chosen pairs
        cand = torch.nonzero(keep_mask & ~protected).flatten()
        order = cand[torch.argsort(sel_strength[cand])]
        keep_mask[order[:deficit]] = False

    device = k.device
    merge_left_d = merge_left.to(device)
    wl = w_left.to(k.dtype)

    k2, v2 = k.clone(), v.clone()
    bias = k.new_zeros(B, H, N, dtype=torch.float32)
    if n_merged:
        li = torch.nonzero(merge_left_d).flatten()
        ri = li + 1
        w = wl[..., li].unsqueeze(-1)                   # (B,H,P,1)
        k2[..., li, :] = w * k[..., li, :] + (1 - w) * k[..., ri, :]
        v2[..., li, :] = w * v[..., li, :] + (1 - w) * v[..., ri, :]
        bias[..., li] = bias_pair[..., li].to(bias.dtype)
        # score channels: the merged slot stands for both members
        if scores.dim() == 3:
            scores = scores.clone()
            scores[..., li] = scores[..., li] + scores[..., ri]
        else:
            scores = scores.clone()
            scores[..., li] = scores[..., li] + scores[..., ri]
        if aux is not None:
            aux = aux.clone()
            aux[..., li] = aux[..., li] + aux[..., ri]

    keep_idx = torch.nonzero(keep_mask.to(device)).flatten()
    k2 = k2.index_select(2, keep_idx)
    v2 = v2.index_select(2, keep_idx)
    bias = bias.index_select(-1, keep_idx)
    scores = scores.index_select(-1, keep_idx)
    aux = aux.index_select(-1, keep_idx) if aux is not None else None
    return k2, v2, scores, aux, bias, n_merged
