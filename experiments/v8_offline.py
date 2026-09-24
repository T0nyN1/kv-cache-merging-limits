"""OT-KV v8 offline harness (docs/otkv_v8_contract.md, "Harness" section).

Per prompt, per `--layer-stride` layer, per KV head: build the pooled-SnapKV
retained set A (shared across the layer's KV heads), split the last-W prefill
positions by TIME into fit/select/audit rows (all query heads of the GQA
group, rows grouped after the time split), fit the v8 pair merges on the fit
rows with the pure functions of core/otkv_v8.py, solve the partial transport,
choose gamma on the select rows, and report full-output errors on the audit
rows and on the real decode queries (64 greedy steps, repeated-trigram
blocking) for six conditions:

    baseline    eviction only (o0)
    ot          v8: matching solver, gamma chosen on select
    greedy      v8 with the greedy comparator (hypothesis 3)
    nobias      the ot pairs with c forced to 1 (no logit bias), theta refit
    random      random pairing among the same locality candidates, same count
                of merges as ot, same c/theta fitting and gamma rule
    oracle      NOT deployable: costs, c, theta fitted on the decode rows
                themselves, matching solver, gamma = 1

Causality. The calibration queries sit inside the retained window, so a
query at position t must not see window keys later than t. The module's
`attention_rows` takes `causal_positions`; the other module functions take a
flat (q, k, v, keep_mask) and are used verbatim on the decode rows, where
every cache slot is visible. On the calibration rows the harness evaluates
the compressed cache with its own vectorised causal softmax (identical
maths, per-row key mask) unless the module function accepts a
`causal_positions` keyword, in which case the module is used there too.
Decode rows attend to the compressed prefill cache PLUS the generated tokens'
keys/values (kept verbatim, protected, causal), as at deployment; the
prefill-only variant is reported as a diagnostic (`rel_decode_prefill_only`,
or `--decode-prefill-only` for the whole run).

Metrics per head: mean |o'-o|^2 / mean |o0-o|^2 on audit and decode rows,
merged pairs, merged fraction of evicted attention mass (decode rows),
solver wall time. Summary: means, medians, paired CI over heads, tails,
fraction of heads where ot beats baseline / greedy.

    python experiments/v8_offline.py --selftest
    python experiments/v8_offline.py --model Qwen/Qwen3-1.7B --device mps --budget 0.05 0.10
"""
import argparse
import inspect
import json
import math
import os
import random
import re
import statistics as st
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GAMMAS = (0.0, 0.25, 0.5, 1.0)
CONDITIONS = ("baseline", "ot", "greedy", "nobias", "biasonly", "random", "massc", "online", "oracle")
EPS = 1e-8


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def _f(x):
    return float(x.item()) if torch.is_tensor(x) else float(x)


def _as_plan(plan):
    """Coerce the solver output to a list of (i, j, c, theta) python scalars."""
    out = []
    for e in plan or []:
        i, j, c, th = e[0], e[1], e[2], e[3]
        out.append((int(i), int(j), float(c), float(th)))
    return out


def _as_edges(edges):
    """Coerce build_sparse_costs output to (i, j, C, c, theta) python scalars."""
    out = []
    for e in edges or []:
        out.append((int(e[0]), int(e[1]), float(e[2]), float(e[3]), float(e[4])))
    return out


def _accepts(fn, name):
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def maxpool_tokens(scores, kernel):
    """Token-axis max pool with 'same' length (core.ot_kv_v7.maxpool_tokens)."""
    if kernel is None or int(kernel) <= 1:
        return scores
    kernel = int(kernel)
    length = scores.shape[-1]
    if length == 0:
        return scores
    flat = scores.reshape(-1, 1, length).float()
    pooled = F.max_pool1d(flat, kernel_size=min(kernel, length), stride=1,
                          padding=min(kernel, length) // 2)
    return pooled[..., :length].reshape(scores.shape).to(scores.dtype)


def separator_positions(tok, ids, mode="sentence"):
    """Positions of cheap separator tokens: newline tokens (paragraphs) and,
    in `sentence` mode, tokens ending a sentence (. ! ?). Not entity detection."""
    if mode == "none":
        return []
    out = []
    for p, t in enumerate(ids):
        s = tok.decode([int(t)])
        if "\n" in s:
            out.append(p)
        elif mode == "sentence" and re.search(r"[.!?]\s*$", s):
            out.append(p)
    return out


# --------------------------------------------------------------------------- #
# cache-side maths kept local to the harness (causal, vectorised)
# --------------------------------------------------------------------------- #
def cache_outputs(q, k, v, keep_mask, bias=None, qpos=None):
    """Attention output over the retained slots with an optional per-slot
    logit bias and an optional causal mask (slot position > query position).
    q (W,d) -> o (W,d), p (W,m) in slot order."""
    kk, vv = k[keep_mask], v[keep_mask]
    d = k.shape[-1]
    logits = (q @ kk.T) * d ** -0.5
    if bias is not None:
        logits = logits + bias[keep_mask][None, :]
    if qpos is not None:
        slot_pos = torch.nonzero(keep_mask).flatten()
        logits = logits.masked_fill(slot_pos[None, :] > qpos[:, None], float("-inf"))
    p = torch.softmax(logits, dim=-1)
    return p @ vv, p


def row_errors(q, k, v_new, keep_mask, bias, o, qpos=None):
    """Per-row |o'-o|^2 (W,) of the compressed cache against the dense target."""
    o1, _ = cache_outputs(q, k, v_new, keep_mask, bias, qpos)
    return ((o1 - o) ** 2).sum(-1)


def fit_edges(i_idx, j_idx, k, v, a_dense, p, slot_of, o, o0, c_max=4.0, eps=EPS,
              force_c=None, chunk=512, joint_c=0):
    """Vectorised pair fit for E edges, delegating to core.otkv_v8._fit_pairs so
    that EVERY condition uses one fitter (`force_c` = the no-bias ablation:
    c fixed, theta refitted, no identical-key theta override).

    i_idx, j_idx : (E,) long cache indices (i evicted, j retained)
    a_dense      : (W, L) dense attention rows;  p : (W, m) baseline weights over A
    slot_of      : (L,) slot index of each retained position (-1 elsewhere)
    o, o0        : (W, d) dense / baseline outputs
    Returns dict of (E,) tensors: c, theta, accepted, identical  and err (W, E).
    """
    from core.otkv_v8 import _fit_pairs, KEY_ATOL
    E = int(i_idx.numel())
    W, d = o.shape
    if E == 0:
        z = torch.zeros(0, dtype=o.dtype)
        return dict(c=z, theta=z, accepted=torch.zeros(0, dtype=torch.bool),
                    identical=torch.zeros(0, dtype=torch.bool), err=torch.zeros(W, 0, dtype=o.dtype))
    outs = {key: [] for key in ("c", "theta", "accepted", "identical", "err")}
    for st_ in range(0, E, chunk):
        ii, jj = i_idx[st_:st_ + chunk], j_idx[st_:st_ + chunk]
        identical = ((k[ii] - k[jj]).abs() <= KEY_ATOL + 1e-5 * k[jj].abs()).all(1)
        c, th, err, acc, _ = _fit_pairs(a_dense[:, ii].T, a_dense[:, jj].T, p[:, slot_of[jj]].T,
                                        v[ii], v[jj], o, o0, float(c_max), float(eps), identical,
                                        force_c=force_c, joint_c=int(joint_c))
        outs["c"].append(c); outs["theta"].append(th); outs["accepted"].append(acc)
        outs["identical"].append(identical); outs["err"].append(err.T)
    return {key: torch.cat(val, 1 if key == "err" else 0) for key, val in outs.items()}


def edge_costs(err, n_blocks, scale):
    """C = max over n_blocks contiguous row blocks of the block-mean err / scale.
    err (W, E); rows are time-major (position-major, GQA heads interleaved), so
    contiguous row blocks are contiguous time blocks (module convention)."""
    if err.shape[1] == 0:
        return torch.zeros(0, dtype=err.dtype)
    nb = max(1, min(int(n_blocks), err.shape[0]))
    means = torch.stack([b.mean(0) for b in torch.tensor_split(err, nb, dim=0)], 0)
    return means.max(0).values / scale


def candidate_pairs(keep_mask, protect_mask, sep_ids, max_dist, max_cand):
    """Locality candidates: for each evicted, unprotected i, the nearest
    <= max_cand retained, unprotected j with |i-j| <= max_dist and no separator
    strictly between them. Returns (E, 2) long tensor of (i, j)."""
    keep = keep_mask.cpu().numpy()
    prot = protect_mask.cpu().numpy()
    ret = np.nonzero(keep & ~prot)[0]
    ev = np.nonzero(~keep & ~prot)[0]
    seps = np.array(sorted(set(int(s) for s in sep_ids)), dtype=np.int64) if len(sep_ids) else np.zeros(0, np.int64)
    out = []
    for i in ev:
        lo = np.searchsorted(ret, i - max_dist, side="left")
        hi = np.searchsorted(ret, i + max_dist, side="right")
        js = ret[lo:hi]
        if js.size == 0:
            continue
        if seps.size:
            a = np.minimum(js, i) + 1
            b = np.maximum(js, i)
            n_between = np.searchsorted(seps, b, side="left") - np.searchsorted(seps, a, side="left")
            js = js[n_between == 0]
            if js.size == 0:
                continue
        order = np.argsort(np.abs(js - i), kind="stable")[:max_cand]
        for j in js[order]:
            out.append((int(i), int(j)))
    if not out:
        return torch.zeros(0, 2, dtype=torch.long)
    return torch.tensor(out, dtype=torch.long)


def random_plan(cands, n_pairs, fit_fn, rng, tries=20):
    """A random matching of `n_pairs` locality candidates with an ACCEPTED fit
    (each i once, each j once), fitted with the same c/theta rule as ot.
    All candidates are fitted first, so rejected fits never consume an
    endpoint; up to `tries` shuffles, the largest matching (capped at n_pairs)
    is returned, so the random comparator has ot's cardinality whenever a
    matching of that size exists among the accepted candidates."""
    if n_pairs <= 0 or cands.shape[0] == 0:
        return []
    fit = fit_fn(cands[:, 0], cands[:, 1])
    ok = torch.nonzero(fit["accepted"]).flatten().tolist()
    if not ok:
        return []
    best = []
    for _ in range(tries):
        order = ok[:]
        rng.shuffle(order)
        used_i, used_j, plan = set(), set(), []
        for e in order:
            i, j = int(cands[e, 0]), int(cands[e, 1])
            if i in used_i or j in used_j:
                continue
            used_i.add(i); used_j.add(j)
            plan.append((i, j, float(fit["c"][e]), float(fit["theta"][e])))
            if len(plan) >= n_pairs:
                break
        if len(plan) > len(best):
            best = plan
        if len(best) >= n_pairs:
            break
    return best


def local_validate(V8, q_sel, k, v, keep_mask, plan, o_sel, gammas, min_gain, per_head_rows, qpos=None):
    """The contract's validate_plan rule with a causal row mask (harness copy)."""
    base = row_errors(q_sel, k, v, keep_mask, None, o_sel, qpos)
    best = (0.0, 0.0, [0.0] * len(per_head_rows))
    if not plan:
        return best
    for g in gammas:
        if g == 0:
            continue
        v_new, bias = V8.apply_plan(v, keep_mask, plan, g)
        e = row_errors(q_sel, k, v_new, keep_mask, bias, o_sel, qpos)
        imp = 1.0 - _f(e.mean()) / max(_f(base.mean()), 1e-30)
        per = [1.0 - _f(e[r].mean()) / max(_f(base[r].mean()), 1e-30) for r in per_head_rows]
        if imp >= min_gain and all(x >= -1e-9 for x in per) and imp > best[1]:
            best = (float(g), imp, per)
    return best


# --------------------------------------------------------------------------- #
# retained set and calibration rows
# --------------------------------------------------------------------------- #
def retained_set(Q, K, budget, window, sink=4, obs_window=32, pool_kernel=7, sep_ids=()):
    """Pooled-SnapKV retained set shared by the KV heads of one layer.

    Q (Hq, L, d) post-RoPE queries of ALL positions; K (Hk, L, d) post-RoPE keys.
    A = sink [0, sink) + window [L-window, L) kept verbatim + top-k middle by the
    attention of the last `obs_window` queries (causal), summed over query heads
    (hence over KV heads), max-pooled with `pool_kernel` on the token axis, to
    reach round(budget*L) slots. Protected = sink + window + separator +-1.
    Returns keep_mask (L,), protect_mask (L,), n_middle."""
    Hq, L, d = Q.shape
    Hk = K.shape[0]
    g = Hq // Hk
    m = int(round(budget * L))
    n_mid = m - sink - window
    mid_lo, mid_hi = sink, L - window
    if n_mid <= 0 or mid_hi <= mid_lo:
        raise ValueError(f"budget {budget} x L {L} = {m} slots leaves no middle slot "
                         f"(sink {sink} + window {window})")
    qpos = torch.arange(L - obs_window, L, device=Q.device)
    score = torch.zeros(L, dtype=torch.float32, device=Q.device)
    for h in range(Hq):
        lg = (Q[h, -obs_window:] @ K[h // g].T) * d ** -0.5                 # (obs, L)
        lg = lg.masked_fill(torch.arange(L, device=Q.device)[None, :] > qpos[:, None], float("-inf"))
        score += torch.softmax(lg, -1).sum(0)
    mid = maxpool_tokens(score[mid_lo:mid_hi], pool_kernel)
    top = torch.topk(mid, min(n_mid, mid.numel())).indices + mid_lo
    keep = torch.zeros(L, dtype=torch.bool, device=Q.device)
    keep[:sink] = True
    keep[mid_hi:] = True
    keep[top] = True
    protect = torch.zeros(L, dtype=torch.bool, device=Q.device)
    protect[:sink] = True
    protect[mid_hi:] = True
    for s in sep_ids:
        protect[max(0, int(s) - 1):min(L, int(s) + 2)] = True
    return keep, protect, int(keep[mid_lo:mid_hi].sum())


def group_rows(Qh, pos_start):
    """Time-major rows for one GQA group. Qh (g, P, d) -> q (P*g, d), positions
    (P*g,), head_of_row (P*g,), per_head_rows [g tensors of row indices]."""
    g, P, d = Qh.shape
    q = Qh.transpose(0, 1).reshape(P * g, d)
    pos = (pos_start + torch.arange(P)).repeat_interleave(g)
    head = torch.arange(g).repeat(P)
    per_head = [torch.nonzero(head == h).flatten() for h in range(g)]
    return q, pos, head, per_head


def time_split(pos, head, L, window):
    """fit/select/audit = first 1/2, next 1/4, last 1/4 of the window positions;
    all query heads of the group; rows grouped after the time split."""
    t1 = L - window + window // 2
    t2 = L - window + window // 2 + window // 4
    out = {}
    for name, rows in (("fit", pos < t1), ("select", (pos >= t1) & (pos < t2)), ("audit", pos >= t2)):
        idx = torch.nonzero(rows).flatten()
        g = int(head.max()) + 1
        per_head = [torch.nonzero(head[idx] == h).flatten() for h in range(g)]   # rows relative to the subset
        out[name] = (idx, per_head)
    return out


# --------------------------------------------------------------------------- #
# one KV head
# --------------------------------------------------------------------------- #
def run_head(V8, k, v, q_cal, pos_cal, head_cal, q_dec, pos_dec, per_head_dec, keep, protect, sep_ids,
             args, rng, L, k_gen=None, v_gen=None):
    """All six conditions for one KV head. Returns the per-head record.

    k, v (L, d): the prefill cache. k_gen, v_gen (T, d): post-RoPE keys/values
    of the generated tokens; they are appended to the cache seen by the decode
    rows (kept verbatim, protected, never merge sources or receivers) and the
    decode rows attend causally (row at position L+t sees generated slots
    <= t). The prefill-only variant of the decode metric is kept as a
    diagnostic (`rel_decode_prefill_only`).
    """
    t_head = time.time()
    d = k.shape[1]
    m = int(keep.sum())
    slot_of = torch.full((L,), -1, dtype=torch.long)
    slot_of[keep] = torch.arange(m)
    evicted = torch.nonzero(~keep).flatten()
    src_ok = ~keep & ~protect
    rec_ok = keep & ~protect
    cache_pos = torch.arange(L)
    mod_causal = _accepts(V8.baseline_outputs, "causal_positions")

    def base_out(q, kk, vv, kp, bias, qpos):
        if mod_causal:
            return V8.baseline_outputs(q, kk, vv, kp, bias, causal_positions=qpos)
        return cache_outputs(q, kk, vv, kp, bias, qpos)

    # decode-side cache: prefill + generated slots (unchanged, protected, causal)
    if k_gen is not None and k_gen.shape[0] > 0:
        T = int(k_gen.shape[0])
        k_x, v_x = torch.cat([k, k_gen], 0), torch.cat([v, v_gen], 0)
        keep_x = torch.cat([keep, torch.ones(T, dtype=torch.bool)])
        prot_x = torch.cat([protect, torch.ones(T, dtype=torch.bool)])
        pos_x = torch.arange(L + T)
        qpos_dec = pos_dec
    else:
        T = 0
        k_x, v_x, keep_x, prot_x, pos_x, qpos_dec = k, v, keep, protect, cache_pos, None

    # dense rows (module; causal on the calibration block and on the decode rows) and targets
    _, a_cal = V8.attention_rows(q_cal, k, causal_positions=pos_cal)
    o_cal = a_cal @ v
    _, a_dec = V8.attention_rows(q_dec, k_x, causal_positions=qpos_dec)
    o_dec = a_dec @ v_x
    _, a_dec_pre = V8.attention_rows(q_dec, k, causal_positions=None)          # prefill-only diagnostic
    o_dec_pre = a_dec_pre @ v

    o0_cal, p_cal = base_out(q_cal, k, v, keep, None, pos_cal)
    o0_dec, p_dec = base_out(q_dec, k_x, v_x, keep_x, None, qpos_dec)

    split = time_split(pos_cal, head_cal, L, args.window)
    fit_rows, _ = split["fit"]
    sel_rows, sel_per_head = split["select"]
    aud_rows, aud_per_head = split["audit"]

    def sub(rows):
        return (q_cal[rows], pos_cal[rows], o_cal[rows], o0_cal[rows], p_cal[rows], a_cal[rows])

    q_fit, pos_fit, o_fit, o0_fit, p_fit, a_fit = sub(fit_rows)
    q_sel, pos_sel, o_sel, _, _, _ = sub(sel_rows)
    q_aud, pos_aud, o_aud, _, _, _ = sub(aud_rows)

    # -- v8 costs on the fit rows, matching and greedy solutions ------------- #
    ev_ids = torch.nonzero(src_ok).flatten().tolist()
    ret_ids = torch.nonzero(rec_ok).flatten().tolist()
    t0 = time.time()
    edges = _as_edges(V8.build_sparse_costs(k, v, q_fit, cache_pos, keep, protect, o_fit, o0_fit, p_fit,
                                            a_fit, max_dist=args.max_dist, max_cand=args.max_cand,
                                            n_blocks=args.n_blocks, c_max=args.c_max, sep_ids=sep_ids,
                                            joint_c=args.joint_c))
    t_cost = time.time() - t0
    n_edges_raw = len(edges)
    edges = [e for e in edges if bool(src_ok[e[0]]) and bool(rec_ok[e[1]])]   # defensive: protected slots never merge
    t0 = time.time()
    plan_ot = _as_plan(V8.solve_partial_transport(edges, ev_ids, ret_ids, method="matching"))
    t_match = time.time() - t0
    t0 = time.time()
    plan_gr = _as_plan(V8.solve_partial_transport(edges, ev_ids, ret_ids, method="greedy"))
    t_greedy = time.time() - t0

    def fit_on(rows_o, rows_o0, rows_p, rows_a, force_c=None):
        def fn(ii, jj):
            return fit_edges(ii, jj, k, v, rows_a, rows_p, slot_of, rows_o, rows_o0,
                             c_max=args.c_max, force_c=force_c,
                             joint_c=(0 if force_c is not None else args.joint_c))
        return fn

    # -- (4) same pairs, c forced to 1 (no bias), theta refit on the fit rows -- #
    plan_nb = []
    if plan_ot:
        ii = torch.tensor([e[0] for e in plan_ot]); jj = torch.tensor([e[1] for e in plan_ot])
        fnb = fit_on(o_fit, o0_fit, p_fit, a_fit, force_c=1.0)(ii, jj)
        plan_nb = [(plan_ot[e][0], plan_ot[e][1], 1.0, float(fnb["theta"][e])) for e in range(len(plan_ot))]

    # -- (5) random pairing among the same locality candidates, same count ---- #
    cands = candidate_pairs(keep, protect, sep_ids, args.max_dist, args.max_cand)
    plan_rd = random_plan(cands, len(plan_ot), fit_on(o_fit, o0_fit, p_fit, a_fit), rng)

    # -- (4b) bias-only: theta forced to 0 (receiver keeps its value), joint c, matching -- #
    edges_b = _as_edges(V8.build_sparse_costs(k, v, q_fit, cache_pos, keep, protect, o_fit, o0_fit, p_fit,
                                              a_fit, max_dist=args.max_dist, max_cand=args.max_cand,
                                              n_blocks=args.n_blocks, c_max=args.c_max, sep_ids=sep_ids,
                                              joint_c=args.joint_c, force_theta=0.0))
    edges_b = [e for e in edges_b if bool(src_ok[e[0]]) and bool(rec_ok[e[1]])]
    plan_b = _as_plan(V8.solve_partial_transport(edges_b, ev_ids, ret_ids, method="matching"))

    # -- (6a) ablation: c from the mass fit only (no joint search), matching -- #
    t0 = time.time()
    edges_j = []
    if args.joint_c > 0:
        edges_j = _as_edges(V8.build_sparse_costs(k, v, q_fit, cache_pos, keep, protect, o_fit, o0_fit, p_fit,
                                                  a_fit, max_dist=args.max_dist, max_cand=args.max_cand,
                                                  n_blocks=args.n_blocks, c_max=args.c_max, sep_ids=sep_ids,
                                                  joint_c=0))
        edges_j = [e for e in edges_j if bool(src_ok[e[0]]) and bool(rec_ok[e[1]])]
    plan_j = _as_plan(V8.solve_partial_transport(edges_j, ev_ids, ret_ids, method="matching")) if edges_j else []
    t_joint = time.time() - t0

    # -- (6b) online re-calibration: fitted on the first `online_fit` decode
    #    steps, gamma chosen on the next `online_sel` steps with the standard
    #    gate, judged on the remaining (late) steps ------------------------ #
    T_dec = int(q_dec.shape[0] // max(1, len(per_head_dec)))
    head_dec = torch.zeros(q_dec.shape[0], dtype=torch.long)
    for hh, rr in enumerate(per_head_dec):
        head_dec[rr] = hh
    step = (pos_dec - L) if qpos_dec is not None else torch.arange(q_dec.shape[0]) // max(1, len(per_head_dec))
    on_fit = torch.nonzero(step < args.online_fit).flatten()
    on_sel = torch.nonzero((step >= args.online_fit) & (step < args.online_fit + args.online_sel)).flatten()
    late = torch.nonzero(step >= args.online_fit + args.online_sel).flatten()
    late_per_head = [torch.nonzero(head_dec[late] == hh).flatten() for hh in range(len(per_head_dec))]
    sel_on_per_head = [torch.nonzero(head_dec[on_sel] == hh).flatten() for hh in range(len(per_head_dec))]
    t0 = time.time()
    plan_on = []
    if on_fit.numel() and on_sel.numel() and late.numel():
        qp = qpos_dec[on_fit] if qpos_dec is not None else None
        edges_on = _as_edges(V8.build_sparse_costs(k_x, v_x, q_dec[on_fit], pos_x, keep_x, prot_x, o_dec[on_fit],
                                                   o0_dec[on_fit], p_dec[on_fit], a_dec[on_fit],
                                                   max_dist=args.max_dist, max_cand=args.max_cand,
                                                   n_blocks=args.n_blocks, c_max=args.c_max, sep_ids=sep_ids,
                                                   joint_c=args.joint_c))
        edges_on = [e for e in edges_on if e[0] < L and e[1] < L and bool(src_ok[e[0]]) and bool(rec_ok[e[1]])]
        plan_on = _as_plan(V8.solve_partial_transport(edges_on, ev_ids, ret_ids, method="matching"))
    t_online = time.time() - t0

    # -- (6) oracle (NOT deployable): costs, c, theta fitted on the decode rows
    #    themselves (extended cache), matching solver, gamma chosen on the same
    #    rows among the allowed gammas with min_gain = 0 ------------------------ #
    t0 = time.time()
    edges_or = _as_edges(V8.build_sparse_costs(k_x, v_x, q_dec, pos_x, keep_x, prot_x, o_dec, o0_dec, p_dec,
                                               a_dec, max_dist=args.max_dist, max_cand=args.max_cand,
                                               n_blocks=args.n_blocks, c_max=args.c_max, sep_ids=sep_ids,
                                               joint_c=args.joint_c))
    edges_or = [e for e in edges_or if e[0] < L and e[1] < L and bool(src_ok[e[0]]) and bool(rec_ok[e[1]])]
    plan_or = _as_plan(V8.solve_partial_transport(edges_or, ev_ids, ret_ids, method="matching"))
    t_oracle = time.time() - t0

    # -- evaluation -------------------------------------------------------- #
    base_aud = row_errors(q_aud, k, v, keep, None, o_aud, pos_aud)
    base_dec = row_errors(q_dec, k_x, v_x, keep_x, None, o_dec, qpos_dec)
    base_dec_pre = row_errors(q_dec, k, v, keep, None, o_dec_pre, None)
    ev_mass = a_dec[:, evicted].mean(0)                                # mean decode attention of each evicted token
    tot_mass = max(_f(ev_mass.sum()), 1e-30)
    mass_of = dict(zip(evicted.tolist(), ev_mass.tolist()))
    use_module_validate = _accepts(V8.validate_plan, "causal_positions")
    use_module_eval = _accepts(V8.evaluate_outputs, "causal_positions")

    def select_gamma(plan, q, kk, vv, kp, o, qpos, per_head, min_gain):
        if not plan:
            return 0.0, 0.0, [0.0] * len(per_head)
        if use_module_validate:
            gm, imp, per = V8.validate_plan(q, kk, vv, kp, plan, o, gammas=tuple(args.gammas),
                                            min_gain=min_gain, per_head_rows=per_head, causal_positions=qpos)
            return float(gm), _f(imp), [_f(x) for x in per]
        return local_validate(V8, q, kk, vv, kp, plan, o, args.gammas, min_gain, per_head, qpos)

    def evaluate(plan, gamma):
        v_new, bias = V8.apply_plan(v, keep, plan, gamma)                # prefill cache (audit rows)
        v_new_x, bias_x = V8.apply_plan(v_x, keep_x, plan, gamma)        # decode-side cache
        e_aud = row_errors(q_aud, k, v_new, keep, bias, o_aud, pos_aud)
        e_dec = row_errors(q_dec, k_x, v_new_x, keep_x, bias_x, o_dec, qpos_dec)
        e_dec_pre = row_errors(q_dec, k, v_new, keep, bias, o_dec_pre, None)
        gap = 0.0
        if use_module_eval:
            mod_dec, _ = V8.evaluate_outputs(q_dec, k_x, v_new_x, keep_x, bias_x, o_dec, causal_positions=qpos_dec)
            gap = abs(_f(mod_dec) - _f(e_dec.mean())) / max(_f(e_dec.mean()), 1e-30)
        return e_aud, e_dec, e_dec_pre, gap

    def rel(e, base, rows=None):
        if rows is not None:
            e, base = e[rows], base[rows]
        return _f(e.mean()) / max(_f(base.mean()), 1e-30)

    conds = {"baseline": ([], "zero"), "ot": (plan_ot, "select"), "greedy": (plan_gr, "select"),
             "nobias": (plan_nb, "select"), "biasonly": (plan_b, "select"), "random": (plan_rd, "select"),
             "massc": (plan_j, "select"),
             "online": (plan_on, "online"), "oracle": (plan_or, "decode")}
    out = {}
    eval_gap = 0.0
    invalid = False
    for name, (plan, mode) in conds.items():
        if mode == "select":
            gamma, imp, per = select_gamma(plan, q_sel, k, v, keep, o_sel, pos_sel, sel_per_head, args.min_gain)
        elif mode == "decode":
            gamma, imp, per = select_gamma(plan, q_dec, k_x, v_x, keep_x, o_dec, qpos_dec, per_head_dec, 0.0)
        elif mode == "online":
            gamma, imp, per = select_gamma(plan, q_dec[on_sel], k_x, v_x, keep_x, o_dec[on_sel],
                                           (qpos_dec[on_sel] if qpos_dec is not None else None),
                                           sel_on_per_head, args.min_gain)
        else:
            gamma, imp, per = 0.0, 0.0, []
        e_aud, e_dec, e_dec_pre, gap = evaluate(plan, gamma)
        eval_gap = max(eval_gap, gap)
        applied = plan if gamma > 0 else []
        r = {"n_proposed": len(plan), "n_merged": len(applied), "gamma": gamma, "select_improvement": imp,
             "select_improvement_per_qhead": per,
             "rel_audit": rel(e_aud, base_aud), "rel_decode": rel(e_dec, base_dec),
             "rel_decode_late": (rel(e_dec, base_dec, late) if late.numel() else float("nan")),
             "rel_decode_late_per_qhead": ([rel(e_dec, base_dec, late[rr]) for rr in late_per_head]
                                           if late.numel() else []),
             "rel_decode_prefill_only": rel(e_dec_pre, base_dec_pre),
             "rel_decode_per_qhead": [rel(e_dec, base_dec, rr) for rr in per_head_dec],
             "rel_audit_per_qhead": [rel(e_aud, base_aud, rr) for rr in aud_per_head],
             "merged_mass_frac": sum(mass_of.get(pp[0], 0.0) for pp in applied) / tot_mass,
             "proposed_mass_frac": sum(mass_of.get(pp[0], 0.0) for pp in plan) / tot_mass,
             "mean_c": (st.fmean(pp[2] for pp in plan) if plan else float("nan")),
             "mean_theta": (st.fmean(pp[3] for pp in plan) if plan else float("nan"))}
        if not all(math.isfinite(x) for x in (r["rel_audit"], r["rel_decode"], r["rel_decode_prefill_only"])):
            invalid = True
        if mode == "select" and plan:
            # what the same plan would do at every gamma on select / decode (diagnostic)
            by_gamma = {}
            for gm in args.gammas:
                if gm == 0:
                    continue
                v_g, b_g = V8.apply_plan(v, keep, plan, gm)
                v_gx, b_gx = V8.apply_plan(v_x, keep_x, plan, gm)
                by_gamma[str(gm)] = {"select": rel(row_errors(q_sel, k, v_g, keep, b_g, o_sel, pos_sel),
                                                   row_errors(q_sel, k, v, keep, None, o_sel, pos_sel)),
                                     "decode": rel(row_errors(q_dec, k_x, v_gx, keep_x, b_gx, o_dec, qpos_dec), base_dec)}
            r["rel_by_gamma"] = by_gamma
        out[name] = r
    return {"L": L, "m": m, "n_generated": T, "n_evicted": int(evicted.numel()), "n_sources": len(ev_ids),
            "n_receivers": len(ret_ids), "n_candidates": int(cands.shape[0]),
            "n_edges_neg": n_edges_raw, "n_edges_used": len(edges), "n_edges_oracle": len(edges_or),
            "n_edges_massc": len(edges_j), "n_late_rows": int(late.numel()), "joint_c": int(args.joint_c),
            "base_err_audit": _f(base_aud.mean()), "base_err_decode": _f(base_dec.mean()),
            "base_err_decode_prefill_only": _f(base_dec_pre.mean()),
            "dense_norm_decode": _f((o_dec ** 2).sum(-1).mean()),
            "evaluate_outputs_gap": eval_gap, "invalid": invalid,
            "time": {"cost": t_cost, "match": t_match, "greedy": t_greedy, "oracle": t_oracle,
                     "joint": t_joint, "online": t_online, "head_total": time.time() - t_head},
            "conditions": out}


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def _finite(xs):
    return [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]


def _ci(xs):
    xs = _finite(xs)
    n = len(xs)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    mu = st.fmean(xs)
    sd = st.stdev(xs) if n > 1 else 0.0
    h = 1.96 * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"mean": mu, "lo": mu - h, "hi": mu + h, "n": n}


def _pct(xs, q):
    xs = sorted(_finite(xs))
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def _cluster_bootstrap(values, clusters, B=2000, seed=0):
    """Mean with a percentile CI from resampling whole clusters (prompts) with
    replacement: the per-head records of one prompt share its text, retained
    set and decode, so they are not independent samples."""
    vals = np.asarray(values, dtype=float)
    cl = np.asarray(clusters)
    ok = np.isfinite(vals)
    vals, cl = vals[ok], cl[ok]
    if vals.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_clusters": 0, "per_cluster_mean": {}}
    names = sorted(set(cl.tolist()))
    groups = [vals[cl == c] for c in names]
    rng_ = np.random.default_rng(seed)
    means = []
    for _ in range(B):
        pick = rng_.integers(0, len(groups), len(groups))
        means.append(float(np.concatenate([groups[i] for i in pick]).mean()))
    return {"mean": float(vals.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "n_clusters": len(groups),
            "per_cluster_mean": {str(c): float(g.mean()) for c, g in zip(names, groups)}}


def summarise(records):
    """Means / medians per condition, head-level (descriptive) and
    prompt-clustered (gate) CIs, tails, invalid-record count, gates."""
    S = {"n_heads": len(records), "conditions": {}}
    S["n_invalid"] = sum(int(bool(r.get("invalid", False))) for r in records)
    cl = [r.get("prompt", "p0") for r in records]
    for c in CONDITIONS:
        rd = [r["conditions"][c]["rel_decode"] for r in records]
        rp = [r["conditions"][c]["rel_decode_prefill_only"] for r in records]
        rl = [r["conditions"][c].get("rel_decode_late", float("nan")) for r in records]
        ra = [r["conditions"][c]["rel_audit"] for r in records]
        S["conditions"][c] = {
            "rel_decode_mean": st.fmean(_finite(rd)) if _finite(rd) else float("nan"),
            "rel_decode_median": st.median(_finite(rd)) if _finite(rd) else float("nan"),
            "rel_decode_p10": _pct(rd, 0.10), "rel_decode_p90": _pct(rd, 0.90),
            "rel_decode_prefill_only_mean": st.fmean(_finite(rp)) if _finite(rp) else float("nan"),
            "rel_decode_late_mean": st.fmean(_finite(rl)) if _finite(rl) else float("nan"),
            "rel_decode_late_median": st.median(_finite(rl)) if _finite(rl) else float("nan"),
            "gain_late_cluster": _cluster_bootstrap([1.0 - x for x in rl], cl),
            "rel_audit_mean": st.fmean(_finite(ra)) if _finite(ra) else float("nan"),
            "rel_audit_median": st.median(_finite(ra)) if _finite(ra) else float("nan"),
            "n_proposed_mean": st.fmean(r["conditions"][c]["n_proposed"] for r in records),
            "n_merged_mean": st.fmean(r["conditions"][c]["n_merged"] for r in records),
            "merged_mass_mean": st.fmean(r["conditions"][c]["merged_mass_frac"] for r in records),
            "proposed_mass_mean": st.fmean(r["conditions"][c]["proposed_mass_frac"] for r in records),
            "frac_gamma_positive": st.fmean(float(r["conditions"][c]["gamma"] > 0) for r in records),
            "frac_heads_better_than_baseline": st.fmean(float(x < 1.0 - 1e-12) for x in rd),
            "frac_heads_worse_than_baseline_5pct": st.fmean(float(x > 1.05) for x in rd),
            "n_nonfinite": sum(int(not (isinstance(x, float) and math.isfinite(x))) for x in rd),
        }
    ot = [r["conditions"]["ot"]["rel_decode"] for r in records]
    gr = [r["conditions"]["greedy"]["rel_decode"] for r in records]
    orc = [r["conditions"]["oracle"]["rel_decode"] for r in records]
    gains = [1.0 - x for x in ot]
    srt = sorted(_finite(gains))
    trimmed = srt[: max(1, int(len(srt) * 0.9))] if srt else []
    S["ot_gain_decode"] = {"paired_ci_heads": _ci(gains), "cluster_bootstrap": _cluster_bootstrap(gains, cl),
                           "median": st.median(_finite(gains)) if _finite(gains) else float("nan"),
                           "mean_without_top10pct_heads": st.fmean(trimmed) if trimmed else float("nan"),
                           "worst_head_rel": max(_finite(ot)) if _finite(ot) else float("nan")}
    dg = [g - o for g, o in zip(gr, ot)]
    S["ot_vs_greedy_decode"] = {"paired_ci_heads_greedy_minus_ot": _ci(dg),
                                "cluster_bootstrap_greedy_minus_ot": _cluster_bootstrap(dg, cl),
                                "frac_heads_ot_better": st.fmean(float(o < g - 1e-12) for o, g in zip(ot, gr)),
                                "frac_heads_tied": st.fmean(float(abs(o - g) <= 1e-12) for o, g in zip(ot, gr))}
    S["ot_gain_audit"] = _ci([1.0 - r["conditions"]["ot"]["rel_audit"] for r in records])
    S["ot_gain_decode_prefill_only"] = _ci([1.0 - r["conditions"]["ot"]["rel_decode_prefill_only"] for r in records])
    og = [1.0 - x for x in orc]
    S["oracle_gain_decode"] = {"paired_ci_heads": _ci(og), "cluster_bootstrap": _cluster_bootstrap(og, cl),
                               "median": st.median(_finite(og)) if _finite(og) else float("nan")}
    S["massc_gain_decode"] = _cluster_bootstrap([1.0 - r["conditions"]["massc"]["rel_decode"] for r in records], cl)
    S["biasonly_gain_decode"] = _cluster_bootstrap([1.0 - r["conditions"]["biasonly"]["rel_decode"] for r in records], cl)
    S["online_gain_decode_late"] = _cluster_bootstrap([1.0 - r["conditions"]["online"]["rel_decode_late"] for r in records], cl)
    S["ot_gain_decode_late"] = _cluster_bootstrap([1.0 - r["conditions"]["ot"]["rel_decode_late"] for r in records], cl)
    S["oracle_gain_decode_late"] = _cluster_bootstrap([1.0 - r["conditions"]["oracle"]["rel_decode_late"] for r in records], cl)
    S["time"] = {key: sum(r["time"].get(key, 0.0) for r in records)
                 for key in ("cost", "match", "greedy", "oracle", "joint", "online", "head_total")}
    S["evaluate_outputs_gap_max"] = max(r["evaluate_outputs_gap"] for r in records) if records else 0.0
    g = S["ot_gain_decode"]["cluster_bootstrap"]
    gg = S["ot_vs_greedy_decode"]["cluster_bootstrap_greedy_minus_ot"]
    oc = S["oracle_gain_decode"]["cluster_bootstrap"]
    fin = lambda x: isinstance(x, float) and math.isfinite(x)
    gates = {
        "no_invalid_records": S["n_invalid"] == 0,
        "decode_gain_ge_5pct": fin(g["mean"]) and g["mean"] >= 0.05,
        "decode_gain_cluster_ci_excludes_zero": fin(g["lo"]) and g["lo"] > 0,
        "decode_gain_not_concentrated": fin(S["ot_gain_decode"]["mean_without_top10pct_heads"])
        and S["ot_gain_decode"]["mean_without_top10pct_heads"] >= 0.025,
        "merges_applied_not_near_zero": (S["conditions"]["ot"]["n_merged_mean"] >= 1.0 and
                                         S["conditions"]["ot"]["merged_mass_mean"] > 0.01),
        "ot_beats_greedy_cluster_ci": fin(gg["lo"]) and gg["lo"] > 0,
        "oracle_gain_positive": fin(oc["lo"]) and oc["lo"] > 0,
        "oracle_gain_ge_5pct": fin(oc["mean"]) and oc["mean"] >= 0.05,
    }
    S["gates"] = {k_: bool(v_) for k_, v_ in gates.items()}
    S["gate_H1_deployable_merge"] = all(S["gates"][k_] for k_ in (
        "no_invalid_records", "decode_gain_ge_5pct", "decode_gain_cluster_ci_excludes_zero",
        "decode_gain_not_concentrated", "merges_applied_not_near_zero"))
    S["gate_H3_ot_beats_greedy"] = S["gates"]["no_invalid_records"] and S["gates"]["ot_beats_greedy_cluster_ci"]
    return S


def print_summary(S, label):
    print(f"\n== v8 offline: {label}   ({S['n_heads']} heads, {S['n_invalid']} invalid) ==")
    print(f"{'condition':<10}{'dec mean':>10}{'dec med':>9}{'dec p90':>9}{'late':>8}{'dec pre':>9}{'aud mean':>10}"
          f"{'prop':>7}{'appl':>7}{'mass':>7}{'g>0':>6}{'<base':>7}{'>1.05':>7}")
    for c in CONDITIONS:
        r = S["conditions"][c]
        print(f"{c:<10}{r['rel_decode_mean']:>10.4f}{r['rel_decode_median']:>9.4f}{r['rel_decode_p90']:>9.4f}"
              f"{r['rel_decode_late_mean']:>8.4f}"
              f"{r['rel_decode_prefill_only_mean']:>9.4f}{r['rel_audit_mean']:>10.4f}"
              f"{r['n_proposed_mean']:>7.1f}{r['n_merged_mean']:>7.1f}"
              f"{r['merged_mass_mean']:>7.3f}{r['frac_gamma_positive']:>6.2f}"
              f"{r['frac_heads_better_than_baseline']:>7.2f}{r['frac_heads_worse_than_baseline_5pct']:>7.2f}")
    g = S["ot_gain_decode"]
    ci, cb = g["paired_ci_heads"], g["cluster_bootstrap"]
    print(f"ot decode gain (1-rel): mean {cb['mean']:+.4f}  heads-CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}]  "
          f"prompt-cluster-CI [{cb['lo']:+.4f}, {cb['hi']:+.4f}] ({cb['n_clusters']} prompts)")
    print(f"                        median {g['median']:+.4f} without-top-10% {g['mean_without_top10pct_heads']:+.4f} "
          f"worst head rel {g['worst_head_rel']:.4f}")
    print("                        per prompt: " + ", ".join(f"{k_} {v_:+.3f}" for k_, v_ in cb["per_cluster_mean"].items()))
    ga = S["ot_gain_audit"]; gp = S["ot_gain_decode_prefill_only"]
    print(f"ot audit gain:          mean {ga['mean']:+.4f} [{ga['lo']:+.4f}, {ga['hi']:+.4f}]   "
          f"ot decode gain, prefill-only cache: {gp['mean']:+.4f} [{gp['lo']:+.4f}, {gp['hi']:+.4f}]")
    og = S["ot_vs_greedy_decode"]
    ci, cb = og["paired_ci_heads_greedy_minus_ot"], og["cluster_bootstrap_greedy_minus_ot"]
    print(f"greedy - ot (decode):   mean {cb['mean']:+.4f}  heads-CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}]  "
          f"cluster-CI [{cb['lo']:+.4f}, {cb['hi']:+.4f}]  ot better in {og['frac_heads_ot_better']:.2f} of heads, "
          f"tied {og['frac_heads_tied']:.2f}")
    oc = S["oracle_gain_decode"]
    print(f"oracle decode gain:     mean {oc['cluster_bootstrap']['mean']:+.4f}  cluster-CI "
          f"[{oc['cluster_bootstrap']['lo']:+.4f}, {oc['cluster_bootstrap']['hi']:+.4f}]  median {oc['median']:+.4f}")
    j = S["massc_gain_decode"]
    print(f"massc decode gain:      mean {j['mean']:+.4f}  cluster-CI [{j['lo']:+.4f}, {j['hi']:+.4f}]  (c from the mass fit only)")
    bo = S["biasonly_gain_decode"]
    print(f"biasonly decode gain:   mean {bo['mean']:+.4f}  cluster-CI [{bo['lo']:+.4f}, {bo['hi']:+.4f}]  (theta = 0)")
    for name, key in (("ot", "ot_gain_decode_late"), ("online", "online_gain_decode_late"), ("oracle", "oracle_gain_decode_late")):
        x = S[key]
        print(f"{name:<8} LATE-row gain:  mean {x['mean']:+.4f}  cluster-CI [{x['lo']:+.4f}, {x['hi']:+.4f}]")
    t = S["time"]
    print(f"time: costs {t['cost']:.1f}s match {t['match']:.1f}s greedy {t['greedy']:.1f}s "
          f"oracle {t['oracle']:.1f}s heads total {t['head_total']:.1f}s; "
          f"evaluate_outputs vs harness max rel gap {S['evaluate_outputs_gap_max']:.2e}")
    print("gates: " + ", ".join(f"{k_}={'PASS' if v_ else 'FAIL'}" for k_, v_ in S["gates"].items()))
    print(f"H1 (deployable merge beats eviction): {'PASS' if S['gate_H1_deployable_merge'] else 'FAIL'}   "
          f"H3 (matching beats greedy): {'PASS' if S['gate_H3_ot_beats_greedy'] else 'FAIL'}")


# --------------------------------------------------------------------------- #
# model plumbing
# --------------------------------------------------------------------------- #
def load_model(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="eager").to(args.device).eval()
    return tok, model


def build_prompts(tok, args):
    """N wikitext continuation prompts + N NIAH retrieval prompts (local data)."""
    from experiments.data import load_wikitext_docs, build_niah_samples
    n = args.prompts
    docs = load_wikitext_docs(min_tokens=args.context + 200, max_docs=n + 4, tokenizer=tok)
    prompts = []
    for i, d in enumerate(docs[:n]):
        ids = tok.encode(d, add_special_tokens=False)[:args.context]
        prompts.append({"kind": "continuation", "idx": i, "ids": ids, "answer": None})
    depths = (0.35, 0.7, 0.5, 0.2, 0.85, 0.6, 0.4, 0.75)[:n]
    for i, s in enumerate(build_niah_samples(docs[n:] or docs, tok, context_tokens=args.context,
                                             depths=depths, seed=args.seed)):
        ids = tok.encode(s["prompt"], add_special_tokens=False)[-(args.context + 200):]
        prompts.append({"kind": "niah", "idx": i, "ids": ids, "answer": s["answer"]})
    return prompts


def arch_of(model):
    """'llama' = a rotary decoder with model.model.layers[i].self_attn and
    separate q/k/v projections (Llama, Qwen, Mistral); 'gpt2' = GPT-2's
    absolute-position blocks, transformer.h[i].attn with a fused c_attn and no
    rotation. The two differ only in where the keys/queries come from, so the
    whole merge protocol below is identical for both."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return "gpt2"
    return "llama"


def model_layers(model):
    return model.transformer.h if arch_of(model) == "gpt2" else model.model.layers


def _gpt2_qkv(module, hs):
    """(q, k, v) each (b, heads, T, d) from GPT-2's fused projection; no RoPE,
    so the keys are already the ones attention uses."""
    q, k, v = module.c_attn(hs).split(module.split_size, dim=2)
    shape = (*hs.shape[:-1], -1, module.head_dim)
    return (q.view(shape).transpose(1, 2), k.view(shape).transpose(1, 2),
            v.view(shape).transpose(1, 2))


def capture_layers_gpt2(model, ids, layers):
    """capture_layers for GPT-2: absolute positions, no rotation applied."""
    store = {}

    def pre_hook(li):
        def hook(module, a, kw):
            hs = a[0] if a else kw.get("hidden_states")
            if hs is None:
                return a, kw
            with torch.no_grad():
                q, k, v = _gpt2_qkv(module, hs)
                store[li] = (k[0].detach().float().cpu(), q[0].detach().float().cpu(),
                             v[0].detach().float().cpu())
            return a, kw
        return hook

    handles = [model.transformer.h[li].attn.register_forward_pre_hook(pre_hook(li), with_kwargs=True)
               for li in layers]
    try:
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return store


def decode_capture_gpt2(model, ids, n_new, layers, no_repeat_ngram=3):
    """decode_capture for GPT-2 (same greedy rule and returns)."""
    from transformers.cache_utils import DynamicCache
    qs, ks, vs = {}, {}, {}

    def pre_hook(li):
        def hook(module, a, kw):
            hs = a[0] if a else kw.get("hidden_states")
            if hs is None or hs.shape[1] != 1:
                return a, kw
            with torch.no_grad():
                q, k, v = _gpt2_qkv(module, hs)
                qs.setdefault(li, []).append(q.detach().float().cpu()[0, :, 0, :])
                ks.setdefault(li, []).append(k.detach().float().cpu()[0, :, 0, :])
                vs.setdefault(li, []).append(v.detach().float().cpu()[0, :, 0, :])
            return a, kw
        return hook

    handles = [model.transformer.h[li].attn.register_forward_pre_hook(pre_hook(li), with_kwargs=True)
               for li in layers]
    gen = []

    def pick(logits):
        lg = logits.float().clone()
        if no_repeat_ngram > 0 and len(gen) >= no_repeat_ngram - 1:
            n = no_repeat_ngram
            hist = ids[0].tolist() + gen
            prefix = tuple(hist[-(n - 1):]) if n > 1 else tuple()
            for i in range(len(hist) - n + 1):
                if tuple(hist[i:i + n - 1]) == prefix:
                    lg[hist[i + n - 1]] = float("-inf")
        return lg.argmax().view(1, 1)

    try:
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True, past_key_values=DynamicCache())
            nxt = pick(out.logits[0, -1])
            for _ in range(n_new):
                out = model(input_ids=nxt, use_cache=True, past_key_values=out.past_key_values)
                gen.append(int(nxt))
                nxt = pick(out.logits[0, -1])
    finally:
        for h in handles:
            h.remove()

    def stack(dct):
        return {li: torch.stack(x, 0) for li, x in dct.items()}
    return stack(qs), stack(ks), stack(vs), gen


def capture_layers(model, ids, layers):
    """Post-RoPE K (Hk,L,d), Q (Hq,L,d) of ALL positions and V (Hk,L,d) for the
    given layers, moved to the CPU as float32 as soon as they are produced.
    Same projection/norm/RoPE path as experiments.classa_oracle.capture_kqv."""
    from transformers.cache_utils import DynamicCache
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    store = {}

    def pre_hook(li):
        def hook(module, a, kw):
            hs = a[0] if a else kw.get("hidden_states")
            pos = kw.get("position_embeddings")
            if hs is None or pos is None:
                return a, kw
            with torch.no_grad():
                head_dim = getattr(module, "head_dim", None) or \
                    module.q_proj.out_features // module.config.num_attention_heads
                shape = (*hs.shape[:-1], -1, head_dim)
                q = module.q_proj(hs).view(shape)
                k = module.k_proj(hs).view(shape)
                v = module.v_proj(hs).view(shape)
                if hasattr(module, "q_norm"):
                    q = module.q_norm(q)
                if hasattr(module, "k_norm"):
                    k = module.k_norm(k)
                q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                cos, sin = pos
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                store[li] = (k[0].detach().float().cpu(), q[0].detach().float().cpu(),
                             v[0].detach().float().cpu())
            return a, kw
        return hook

    handles = [model.model.layers[li].self_attn.register_forward_pre_hook(pre_hook(li), with_kwargs=True)
               for li in layers]
    try:
        with torch.no_grad():
            model(input_ids=ids, use_cache=True, past_key_values=DynamicCache())
    finally:
        for h in handles:
            h.remove()
    return store


def decode_capture(model, ids, n_new, layers, no_repeat_ngram=3):
    """Greedy decode of n_new tokens with repeated-n-gram blocking (the rule of
    experiments.sigma_decode.decode_queries), capturing for the given layers
    the post-RoPE decode queries (T, Hq, d) AND the post-RoPE keys / values of
    the generated tokens (T, Hk, d). Returns (q, k, v, generated ids)."""
    from transformers.cache_utils import DynamicCache
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    qs, ks, vs = {}, {}, {}

    def pre_hook(li):
        def hook(module, a, kw):
            hs = a[0] if a else kw.get("hidden_states")
            pos = kw.get("position_embeddings")
            if hs is None or pos is None or hs.shape[1] != 1:
                return a, kw
            with torch.no_grad():
                head_dim = getattr(module, "head_dim", None) or \
                    module.q_proj.out_features // module.config.num_attention_heads
                shape = (*hs.shape[:-1], -1, head_dim)
                q = module.q_proj(hs).view(shape)
                k = module.k_proj(hs).view(shape)
                v = module.v_proj(hs).view(shape)
                if hasattr(module, "q_norm"):
                    q = module.q_norm(q)
                if hasattr(module, "k_norm"):
                    k = module.k_norm(k)
                q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                cos, sin = pos
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                qs.setdefault(li, []).append(q.detach().float().cpu()[0, :, 0, :])
                ks.setdefault(li, []).append(k.detach().float().cpu()[0, :, 0, :])
                vs.setdefault(li, []).append(v.detach().float().cpu()[0, :, 0, :])
            return a, kw
        return hook

    handles = [model.model.layers[li].self_attn.register_forward_pre_hook(pre_hook(li), with_kwargs=True)
               for li in layers]
    gen = []

    def pick(logits):
        lg = logits.float().clone()
        if no_repeat_ngram > 0 and len(gen) >= no_repeat_ngram - 1:
            n = no_repeat_ngram
            hist = ids[0].tolist() + gen
            prefix = tuple(hist[-(n - 1):]) if n > 1 else tuple()
            for i in range(len(hist) - n + 1):
                if tuple(hist[i:i + n - 1]) == prefix:
                    lg[hist[i + n - 1]] = float("-inf")
        return lg.argmax().view(1, 1)

    try:
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True, past_key_values=DynamicCache())
            nxt = pick(out.logits[0, -1])
            for _ in range(n_new):
                out = model(input_ids=nxt, use_cache=True, past_key_values=out.past_key_values)
                gen.append(int(nxt))
                nxt = pick(out.logits[0, -1])
    finally:
        for h in handles:
            h.remove()

    def stack(dct):
        return {li: torch.stack(x, 0) for li, x in dct.items()}
    return stack(qs), stack(ks), stack(vs), gen


def process_prompt(V8, store, qdec, kgen, vgen, prompt, sep_ids, layers, budgets, args, rng):
    """All layers x KV heads x budgets of one prompt -> {budget: [records]}."""
    out = {b: [] for b in budgets}
    L = len(prompt["ids"])
    for li in layers:
        K, Q, V = store[li]
        Hk, Hq = K.shape[0], Q.shape[0]
        g = Hq // Hk
        Qd = qdec[li].float().cpu()                                         # (T, Hq, d)
        Kg = kgen[li].float().cpu() if (kgen is not None and not args.decode_prefill_only) else None
        Vg = vgen[li].float().cpu() if Kg is not None else None
        for b in budgets:
            keep, protect, n_mid = retained_set(Q, K, b, args.window, args.sink, args.obs_window,
                                                args.pool_kernel, sep_ids)
            if args.max_sources > 0:
                cand_src = torch.nonzero(~keep & ~protect).flatten().tolist()
                if len(cand_src) > args.max_sources:
                    drop = rng.sample(cand_src, len(cand_src) - args.max_sources)
                    protect = protect.clone()
                    protect[torch.tensor(drop)] = True
            for h in range(Hk if args.max_heads <= 0 else min(Hk, args.max_heads)):
                k, v = K[h], V[h]
                q_cal, pos_cal, head_cal, _ = group_rows(Q[h * g:(h + 1) * g, L - args.window:], L - args.window)
                q_dec, pos_dec, _, per_head_dec = group_rows(Qd[:, h * g:(h + 1) * g].transpose(0, 1), L)
                rec = run_head(V8, k, v, q_cal, pos_cal, head_cal, q_dec, pos_dec, per_head_dec, keep, protect,
                               sep_ids, args, rng, L,
                               k_gen=(Kg[:, h] if Kg is not None else None),
                               v_gen=(Vg[:, h] if Vg is not None else None))
                rec.update({"prompt": f"{prompt['kind']}{prompt['idx']}", "kind": prompt["kind"],
                            "layer": li, "kv_head": h, "group": g, "n_middle": n_mid, "budget": b})
                out[b].append(rec)
    return out


def write_outputs(records_by_budget, meta, args):
    os.makedirs(args.out_dir, exist_ok=True)
    tag = os.path.basename(args.model.rstrip("/")).lower() + (f"_{args.tag}" if args.tag else "")
    paths = []
    for b, recs in records_by_budget.items():
        S = summarise(recs)
        print_summary(S, f"{tag} budget {b:g}")
        path = os.path.join(args.out_dir, f"v8_offline_{tag}_b{b:g}.json")
        with open(path, "w") as f:
            json.dump({"meta": dict(meta, budget=b), "summary": S, "heads": recs}, f, indent=1)
        print(f"wrote {path}")
        paths.append(path)
    return paths


def main_real(args):
    import core.otkv_v8 as V8
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    t_start = time.time()
    tok, model = load_model(args)
    arch = arch_of(model)
    n_layers = len(model_layers(model))
    layers = args.layers or list(range(0, n_layers, args.layer_stride))
    prompts = build_prompts(tok, args)
    records = {b: [] for b in args.budget}
    prompt_meta = []
    for p in prompts:
        t_p = time.time()
        ids = torch.tensor([p["ids"]], device=args.device)
        store = (capture_layers_gpt2 if arch == "gpt2" else capture_layers)(model, ids, layers)
        t_cap = time.time() - t_p
        qdec, kgen, vgen, gen = (decode_capture_gpt2 if arch == "gpt2" else decode_capture)(
            model, ids, args.new, layers, no_repeat_ngram=3)
        missing = [li for li in layers if li not in qdec]
        if missing:
            raise RuntimeError(f"decode_capture captured no queries for layers {missing}")
        t_dec = time.time() - t_p - t_cap
        sep_ids = separator_positions(tok, p["ids"], args.sep_mode)
        t0 = time.time()
        res = process_prompt(V8, store, qdec, kgen, vgen, p, sep_ids, layers, args.budget, args, rng)
        for b in args.budget:
            records[b].extend(res[b])
        text = tok.decode(gen)
        prompt_meta.append({"prompt": f"{p['kind']}{p['idx']}", "kind": p["kind"], "L": len(p["ids"]),
                            "n_separators": len(sep_ids), "generated": text[:160],
                            "answer": p["answer"], "answer_in_generation": (p["answer"] in text) if p["answer"] else None,
                            "time_capture": t_cap, "time_decode": t_dec, "time_heads": time.time() - t0})
        print(f"[{p['kind']}{p['idx']}] L={len(p['ids'])} seps={len(sep_ids)} capture {t_cap:.1f}s "
              f"decode {t_dec:.1f}s heads {time.time() - t0:.1f}s  gen={text[:60]!r}"
              + (f"  answer_ok={p['answer'] in text}" if p["answer"] else ""), flush=True)
        del store, qdec, kgen, vgen
    meta = {"model": args.model, "arch": arch, "dtype": args.dtype, "device": args.device, "layers": layers,
            "context": args.context, "new_tokens": args.new, "window": args.window, "sink": args.sink,
            "obs_window": args.obs_window, "pool_kernel": args.pool_kernel, "max_dist": args.max_dist,
            "max_cand": args.max_cand, "n_blocks": args.n_blocks, "c_max": args.c_max,
            "min_gain": args.min_gain, "gammas": list(args.gammas), "sep_mode": args.sep_mode,
            "max_sources": args.max_sources, "seed": args.seed, "prompts": prompt_meta,
            "module_validate_plan_causal": _accepts(V8.validate_plan, "causal_positions"),
            "module_baseline_outputs_causal": _accepts(V8.baseline_outputs, "causal_positions"),
            "decode_prefill_only": bool(args.decode_prefill_only),
            "protocol": ("decode rows attend to the prefill cache only" if args.decode_prefill_only else
                         "decode rows attend to the compressed prefill cache plus the generated tokens "
                         "(kept verbatim, protected), causally") +
                        "; calibration rows causal; split fit/select/audit = W/2, W/4, W/4 by time, GQA rows "
                        "grouped after the split; the SnapKV observation window (last 32 positions) overlaps the "
                        "select+audit rows, so 'audit' is in-sample w.r.t. the retained-set choice (decode is not)",
            "total_time_s": time.time() - t_start}
    write_outputs(records, meta, args)
    print(f"total {time.time() - t_start:.0f}s")


# --------------------------------------------------------------------------- #
# selftest: a tiny stub with the contract's signatures, random tensors
# --------------------------------------------------------------------------- #
class _StubV8:
    """Reference implementation of docs/otkv_v8_contract.md (per-edge loops,
    no causal handling beyond attention_rows) used only by --selftest."""

    @staticmethod
    def attention_rows(q, k, causal_positions=None):
        d = k.shape[-1]
        logits = (q @ k.T) * d ** -0.5
        if causal_positions is not None:
            logits = logits.masked_fill(torch.arange(k.shape[0])[None, :] > causal_positions[:, None], float("-inf"))
        return logits, torch.softmax(logits, -1)

    @staticmethod
    def baseline_outputs(q, k, v, keep_mask, bias=None):
        kk, vv = k[keep_mask], v[keep_mask]
        lg = (q @ kk.T) * k.shape[-1] ** -0.5
        if bias is not None:
            lg = lg + bias[keep_mask][None, :]
        p = torch.softmax(lg, -1)
        return p @ vv, p

    @staticmethod
    def fit_pair(a_i, a_j, p_j, v_i, v_j, o, o0, c_max=4.0, eps=1e-8, identical_keys=False, force_theta=None):
        m = a_i + a_j
        den = _f((a_j * a_j).sum())
        if identical_keys:
            c = 2.0
        else:
            if den < eps:
                return dict(c=float("nan"), theta=float("nan"), err=None, accepted=False, reason="denominator")
            c = _f((a_j * m).sum()) / den
            if c > c_max:
                return dict(c=c, theta=float("nan"), err=None, accepted=False, reason="c_max")
        D = 1.0 + (c - 1.0) * p_j
        u = (o0 + ((c - 1.0) * p_j)[:, None] * v_j[None]) / D[:, None]
        dvec = (c * p_j / D)[:, None] * (v_i - v_j)[None]
        dd = _f((dvec * dvec).sum())
        if identical_keys or torch.equal(v_i, v_j) or dd < eps:
            theta = 0.5
        else:
            theta = min(1.0, max(0.0, _f((dvec * (o - u)).sum()) / dd))
        if force_theta is not None:
            theta = float(force_theta)
        err = ((u + theta * dvec - o) ** 2).sum(-1) - ((o0 - o) ** 2).sum(-1)
        return dict(c=c, theta=theta, err=err, accepted=True, reason="ok")

    @classmethod
    def build_sparse_costs(cls, k, v, q_fit, pos, keep_mask, protect_mask, o, o0, p, a_dense,
                           max_dist=8, max_cand=4, n_blocks=4, c_max=4.0, sep_ids=None, joint_c=0, force_theta=None):
        L = k.shape[0]
        slot_of = torch.full((L,), -1, dtype=torch.long)
        slot_of[keep_mask] = torch.arange(int(keep_mask.sum()))
        cands = candidate_pairs(keep_mask, protect_mask, sep_ids or [], max_dist, max_cand)
        scale = _f((o ** 2).sum(-1).mean()) + 1e-8
        edges = []
        for i, j in cands.tolist():
            ident = bool(torch.allclose(k[i], k[j], atol=1e-6))
            r = cls.fit_pair(a_dense[:, i], a_dense[:, j], p[:, slot_of[j]], v[i], v[j], o, o0,
                             c_max=c_max, identical_keys=ident, force_theta=force_theta)
            if not r["accepted"]:
                continue
            C = _f(edge_costs(r["err"][:, None], n_blocks, scale)[0])
            if C < 0:
                edges.append((i, j, C, r["c"], r["theta"]))
        return edges

    @staticmethod
    def solve_partial_transport(edges, n_evicted_ids, retained_ids, method="matching"):
        if not edges:
            return []
        if method == "greedy":
            used_i, used_j, plan = set(), set(), []
            for i, j, C, c, th in sorted(edges, key=lambda e: e[2]):
                if i in used_i or j in used_j:
                    continue
                plan.append((i, j, c, th)); used_i.add(i); used_j.add(j)
            return plan
        from scipy.optimize import linear_sum_assignment
        rows = sorted({e[0] for e in edges}); cols = sorted({e[1] for e in edges})
        ri = {i: a for a, i in enumerate(rows)}; cj = {j: b for b, j in enumerate(cols)}
        n, mm = len(rows), len(cols)
        cost = np.zeros((n, mm + n)); cost[:, :mm] = np.inf
        best = {}
        for e in edges:
            a, b = ri[e[0]], cj[e[1]]
            if e[2] < cost[a, b]:
                cost[a, b] = e[2]; best[(a, b)] = e
        r, c = linear_sum_assignment(cost)
        return [(best[(a, b)][0], best[(a, b)][1], best[(a, b)][3], best[(a, b)][4])
                for a, b in zip(r, c) if b < mm]

    @staticmethod
    def apply_plan(v, keep_mask, plan, gamma):
        v_new = v.clone()
        bias = torch.zeros(v.shape[0], dtype=v.dtype)
        for i, j, c, th in plan:
            v_new[j] = v[j] + gamma * th * (v[i] - v[j])
            bias[j] = gamma * math.log(c)
        return v_new, bias

    @classmethod
    def validate_plan(cls, q_sel, k, v, keep_mask, plan, o_sel, gammas=GAMMAS, min_gain=0.05, per_head_rows=None):
        per_head_rows = per_head_rows or [torch.arange(q_sel.shape[0])]
        o0, _ = cls.baseline_outputs(q_sel, k, v, keep_mask)
        base = ((o0 - o_sel) ** 2).sum(-1)
        best = (0.0, 0.0, [0.0] * len(per_head_rows))
        for g in gammas:
            if g == 0:
                continue
            v_new, bias = cls.apply_plan(v, keep_mask, plan, g)
            o1, _ = cls.baseline_outputs(q_sel, k, v_new, keep_mask, bias)
            e = ((o1 - o_sel) ** 2).sum(-1)
            imp = 1.0 - _f(e.mean()) / _f(base.mean())
            per = [1.0 - _f(e[r].mean()) / _f(base[r].mean()) for r in per_head_rows]
            if imp >= min_gain and all(x >= 0 for x in per) and imp > best[1]:
                best = (g, imp, per)
        return best

    @classmethod
    def evaluate_outputs(cls, q, k, v_new, keep_mask, bias, o):
        o1, _ = cls.baseline_outputs(q, k, v_new, keep_mask, bias)
        o2, _ = cls.baseline_outputs(q, k, v_new, keep_mask, None)
        return ((o1 - o) ** 2).sum(-1).mean(), ((o2 - o) ** 2).sum(-1).mean()


class _StubV8Causal(_StubV8):
    """Same stub, but the flat functions accept `causal_positions` (exercises
    the harness path that hands causality to the module)."""

    @staticmethod
    def baseline_outputs(q, k, v, keep_mask, bias=None, causal_positions=None):
        return cache_outputs(q, k, v, keep_mask, bias, causal_positions)

    @classmethod
    def validate_plan(cls, q_sel, k, v, keep_mask, plan, o_sel, gammas=GAMMAS, min_gain=0.05,
                      per_head_rows=None, causal_positions=None):
        return local_validate(cls, q_sel, k, v, keep_mask, plan, o_sel, gammas, min_gain,
                              per_head_rows or [torch.arange(q_sel.shape[0])], causal_positions)


# --------------------------------------------------------------------------- #
# selftest driver and entry point
# --------------------------------------------------------------------------- #
def _random_head(rng_seed, L=96, d=16, g=2, window=32, new=56, budget=0.30):
    """Random post-'RoPE'-like tensors for one KV head with a GQA group of g
    query heads: k, v (L,d); calibration queries of the last `window`
    positions (g heads); `new` decode queries per head."""
    gen = torch.Generator().manual_seed(rng_seed)
    k = torch.randn(L, d, generator=gen)
    # make some keys near-duplicates so the exact branch and the locality
    # candidates have something to bite on
    for t in range(8, L - window, 9):
        k[t] = k[t - 1] + 1e-7 * torch.randn(d, generator=gen)
    v = torch.randn(L, d, generator=gen)
    Q = torch.randn(g, window, d, generator=gen) + 0.3 * k[L - window:][None]
    Qd = torch.randn(new, g, d, generator=gen) + 0.3 * k[L - 1][None, None]
    k_gen = torch.randn(new, d, generator=gen)
    v_gen = torch.randn(new, d, generator=gen)
    return k, v, Q, Qd, k_gen, v_gen


def run_selftest(args):
    """Run run_head with (a) the reference stub, (b) the stub whose flat
    functions take causal_positions, (c) core.otkv_v8; the three must agree on
    every reported number up to float tolerance, and the diagnostic invariants
    (gamma=0 -> baseline; protected slots never merge; oracle >= 0 merges) hold."""
    import core.otkv_v8 as REAL
    ns = types.SimpleNamespace(**vars(args))
    ns.window = 32
    ns.joint_c = 0                       # stub == module on every condition without the joint search
    ns.gammas = list(GAMMAS)
    rng = random.Random(0)
    fails = 0
    all_module = []
    for seed in range(args.selftest_seeds):
        L, g = 96, 2
        k, v, Q, Qd, k_gen, v_gen = _random_head(seed, L=L, g=g, window=ns.window)
        if ns.decode_prefill_only:
            k_gen = v_gen = None
        keep, protect, _ = retained_set(Q.new_zeros(g, L, k.shape[1]).copy_(
            torch.cat([torch.randn(g, L - ns.window, k.shape[1]), Q], 1)),
            k[None], 0.50, ns.window, sink=4, obs_window=8, pool_kernel=3, sep_ids=[40])
        q_cal, pos_cal, head_cal, _ = group_rows(Q, L - ns.window)
        q_dec, pos_dec, _, per_head_dec = group_rows(Qd.transpose(0, 1), L)
        outs = {}
        for name, mod in (("stub", _StubV8), ("stub_causal", _StubV8Causal), ("module", REAL)):
            outs[name] = run_head(mod, k, v, q_cal, pos_cal, head_cal, q_dec, pos_dec, per_head_dec, keep, protect,
                                  [40], ns, random.Random(seed), L, k_gen=k_gen, v_gen=v_gen)
            outs[name]["prompt"] = f"s{seed}"
        ref = outs["stub"]
        for name in ("stub_causal", "module"):
            r = outs[name]
            for key in ("n_edges_neg", "n_edges_used", "n_edges_oracle"):
                if r[key] != ref[key]:
                    print(f"  seed {seed}: {name} {key} {r[key]} != stub {ref[key]}"); fails += 1
            for c in CONDITIONS:
                if c in ("ot", "greedy", "nobias", "biasonly", "random", "online", "oracle") and ns.joint_c > 0:
                    continue          # the stub has no joint (c, theta) search; compare massc only
                for key in ("rel_audit", "rel_decode", "rel_decode_late", "gamma", "n_merged", "merged_mass_frac"):
                    a, b = r["conditions"][c][key], ref["conditions"][c][key]
                    if abs(a - b) > 1e-5 * max(1.0, abs(b)):
                        print(f"  seed {seed}: {name} {c}.{key} {a:.6g} != stub {b:.6g}"); fails += 1
        # invariants
        b0 = ref["conditions"]["baseline"]
        if abs(b0["rel_audit"] - 1.0) > 1e-9 or abs(b0["rel_decode"] - 1.0) > 1e-9:
            print(f"  seed {seed}: baseline rel != 1"); fails += 1
        for c in CONDITIONS:
            r = ref["conditions"][c]
            if r["gamma"] == 0 and (abs(r["rel_audit"] - 1) > 1e-9 or abs(r["rel_decode"] - 1) > 1e-9):
                print(f"  seed {seed}: {c} gamma=0 but rel != 1"); fails += 1
        for c in CONDITIONS:
            r = ref["conditions"][c]
            if r["gamma"] == 0 and r["n_merged"] != 0:
                print(f"  seed {seed}: {c} gamma=0 but n_merged {r['n_merged']}"); fails += 1
        if len(ref["conditions"]["random"]) and ref["conditions"]["random"]["n_proposed"] < ref["conditions"]["ot"]["n_proposed"]:
            print(f"  seed {seed}: random proposed {ref['conditions']['random']['n_proposed']} < ot {ref['conditions']['ot']['n_proposed']} (note)")
        if ref["evaluate_outputs_gap"] > 1e-5 or outs["module"]["evaluate_outputs_gap"] > 1e-5:
            print(f"  seed {seed}: evaluate_outputs gap {ref['evaluate_outputs_gap']:.2e} / "
                  f"{outs['module']['evaluate_outputs_gap']:.2e}"); fails += 1
        all_module.append(outs["module"])
        print(f"seed {seed}: edges {ref['n_edges_used']} ot pairs {ref['conditions']['ot']['n_proposed']} "
              f"gamma {ref['conditions']['ot']['gamma']} rel_dec ot {ref['conditions']['ot']['rel_decode']:.4f} "
              f"greedy {ref['conditions']['greedy']['rel_decode']:.4f} oracle {ref['conditions']['oracle']['rel_decode']:.4f} "
              f"(module: ot {outs['module']['conditions']['ot']['rel_decode']:.4f})")
    # the aggregation must run on the records too
    S = summarise(all_module)
    print_summary(S, f"selftest ({len(all_module)} heads)")
    print(f"\nselftest: {'OK' if fails == 0 else f'{fails} mismatches'}")
    return fails


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--dtype", default="float32", help="model dtype (float32 on MPS for Qwen3; bfloat16 for Llama)")
    p.add_argument("--prompts", type=int, default=3, help="N continuation + N retrieval prompts")
    p.add_argument("--context", type=int, default=2500)
    p.add_argument("--new", type=int, default=64, help="decode steps (decode rows)")
    p.add_argument("--layer-stride", type=int, default=4)
    p.add_argument("--layers", type=int, nargs="*", default=None, help="explicit layer list (subset of the stride)")
    p.add_argument("--budget", type=float, nargs="+", default=[0.05, 0.10])
    p.add_argument("--window", type=int, default=64, help="calibration window W (kept verbatim)")
    p.add_argument("--sink", type=int, default=4)
    p.add_argument("--obs-window", type=int, default=32)
    p.add_argument("--pool-kernel", type=int, default=7)
    p.add_argument("--max-dist", type=int, default=8)
    p.add_argument("--max-cand", type=int, default=4)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--c-max", type=float, default=4.0)
    p.add_argument("--min-gain", type=float, default=0.05)
    p.add_argument("--gammas", type=float, nargs="+", default=list(GAMMAS))
    p.add_argument("--sep-mode", default="sentence", choices=["none", "newline", "sentence"])
    p.add_argument("--max-sources", type=int, default=0, help="subsample evicted sources per layer (0 = all)")
    p.add_argument("--max-heads", type=int, default=0, help="limit KV heads per layer (0 = all)")
    p.add_argument("--joint-c", type=int, default=9, help="grid points of the 1-D joint (c, theta) search (0 = off)")
    p.add_argument("--online-fit", type=int, default=32, help="decode steps used to fit the online re-calibration")
    p.add_argument("--online-sel", type=int, default=16, help="decode steps used to choose gamma online")
    p.add_argument("--decode-prefill-only", action="store_true",
                   help="evaluate decode rows against the prefill cache only (old protocol; diagnostic)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs/local_repro")
    p.add_argument("--tag", default="")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--selftest-seeds", type=int, default=3)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.selftest:
        return 1 if run_selftest(args) else 0
    main_real(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
