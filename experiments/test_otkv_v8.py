"""Math/code checks for OT-KV v8 (core/otkv_v8.py) -- design section 10.A, no model download.

1. Identical keys, different values, with background entries: the merged full
   output equals the dense output exactly (exact branch c=2, theta=1/2), also
   for several disjoint duplicate pairs and through the whole pipeline.
2. Identical values: theta leaves the value unchanged and the mass compensation
   is correct (duplicate keys -> c = 2; distinct keys -> c* is the least-squares
   mass factor and >= 1).
3. Large logit gap with opposite values: the full-output err exposes the damage
   (err > 0 on the deviating queries, block-max cost > 0, edge rejected) even
   though the mass fit is good; a constant gap of the same size is exactly
   representable, so the damage is the query-dependence of the gap.
4. gamma = 0 reproduces the baseline element-wise.
5. GQA time split then head grouping, non-overlapping pairs, mask/bias in sync,
   causal mask, and an empty candidate set runs end to end.
6. The matching solver equals brute-force enumeration on small graphs; the
   design's 2x2 toy (OT -18.5 vs greedy -10); an infeasible local support
   routes everything to the drop node.
7. The single-edge closed form o_pair = u + theta*dvec equals direct softmax
   recomputation to 1e-6 on random data (design section 13), and theta* is no
   worse than a 1001-point grid search.

Run: ~/anaconda3/envs/ot-kv/bin/python experiments/test_otkv_v8.py
"""
import itertools
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.otkv_v8 import (apply_plan, attention_rows, baseline_outputs,  # noqa: E402
                          build_sparse_costs, evaluate_outputs, fit_pair,
                          solve_partial_transport, time_split_rows, validate_plan)

PASS, FAIL = [], []
F64 = torch.float64


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))


def slot_of(keep, j):
    return int(keep[:j].sum())


def dense(q, k, v, causal_positions=None):
    _, a = attention_rows(q, k, causal_positions)
    return a, a @ v.to(a.dtype)


def edge_cost(err, n_blocks, scale):
    return max(float(b.mean()) for b in torch.tensor_split(err, n_blocks)) / scale


# --------------------------------------------------------------------------- #
def test_identical_keys_exact():
    torch.manual_seed(1)
    L, d, W = 7, 8, 24
    k = torch.randn(L, d, dtype=F64)
    v = torch.randn(L, d, dtype=F64)
    q = torch.randn(W, d, dtype=F64) * 1.5
    i, j = 4, 5
    k[i] = k[j]                                   # identical keys, different values
    keep = torch.ones(L, dtype=torch.bool)
    keep[i] = False
    a, o = dense(q, k, v)
    o0, p = baseline_outputs(q, k, v, keep)
    base = float(((o0 - o) ** 2).sum(1).mean())
    check("baseline eviction of the duplicate is not already exact", base > 1e-6, str(base))

    res = fit_pair(a[:, i], a[:, j], p[:, slot_of(keep, j)], v[i], v[j], o, o0, identical_keys=True)
    check("exact branch gives c=2, theta=1/2", res["c"] == 2.0 and res["theta"] == 0.5 and res["accepted"])
    check("exact branch err = -|o0-o|^2 on every fit query",
          torch.allclose(res["err"], -((o0 - o) ** 2).sum(1), atol=1e-12))
    v_new, bias = apply_plan(v, keep, [(i, j, res["c"], res["theta"])], 1.0)
    e_new, e_base = evaluate_outputs(q, k, v_new, keep, bias, o, v0=v)
    check("merged full output equals the dense output (mean sq err < 1e-24)", e_new < 1e-24, str(e_new))
    check("evaluate_outputs baseline term matches", abs(e_base - base) < 1e-12)
    check("merged value is the midpoint, bias log 2",
          torch.allclose(v_new[j], 0.5 * (v[i] + v[j])) and abs(float(bias[j]) - math.log(2)) < 1e-12)

    # whole pipeline: candidate search -> exact branch -> matching -> apply
    pos = torch.arange(L)
    prot = torch.zeros(L, dtype=torch.bool)
    edges = build_sparse_costs(k, v, q, pos, keep, prot, o, o0, p, a)
    dup = [e for e in edges if e[0] == i and e[1] == j]
    check("build_sparse_costs finds the duplicate edge via the exact branch",
          len(dup) == 1 and dup[0][3] == 2.0 and dup[0][4] == 0.5 and dup[0][2] < 0, str(edges))
    plan = solve_partial_transport(edges, [i], torch.nonzero(keep).flatten(), "matching")
    check("matching selects the exact pair", plan == [(i, j, 2.0, 0.5)], str(plan))
    g, imp, groups = validate_plan(q, k, v, keep, plan, o)
    check("validate_plan accepts it with gamma=1 and ~100% improvement",
          g == 1.0 and imp > 1 - 1e-9 and min(groups) > 1 - 1e-9, f"{g} {imp} {groups}")

    # several disjoint duplicate pairs, applied jointly, with background
    torch.manual_seed(2)
    L = 10
    k = torch.randn(L, d, dtype=F64)
    v = torch.randn(L, d, dtype=F64)
    k[2] = k[3]
    k[6] = k[8]
    keep = torch.ones(L, dtype=torch.bool)
    keep[2] = False
    keep[6] = False
    a, o = dense(q, k, v)
    o0, p = baseline_outputs(q, k, v, keep)
    plan = [(2, 3, 2.0, 0.5), (6, 8, 2.0, 0.5)]
    v_new, bias = apply_plan(v, keep, plan, 1.0)
    e_new, e_base = evaluate_outputs(q, k, v_new, keep, bias, o, v0=v)
    check("two disjoint duplicate pairs jointly exact", e_new < 1e-24 and e_base > 1e-6, f"{e_new} {e_base}")


# --------------------------------------------------------------------------- #
def test_identical_values():
    torch.manual_seed(3)
    L, d, W = 6, 8, 32
    k = torch.randn(L, d, dtype=F64)
    v = torch.randn(L, d, dtype=F64)
    q = torch.randn(W, d, dtype=F64) * 1.5
    i, j = 3, 4
    keep = torch.ones(L, dtype=torch.bool)
    keep[i] = False

    # (a) duplicate key AND value: the mass fit alone must find c = 2
    k[i] = k[j]
    v[i] = v[j]
    a, o = dense(q, k, v)
    o0, p = baseline_outputs(q, k, v, keep)
    res = fit_pair(a[:, i], a[:, j], p[:, slot_of(keep, j)], v[i], v[j], o, o0)   # no exact flag
    check("duplicate K,V: fitted mass factor c = 2 exactly", res["c"] == 2.0 and res["accepted"], str(res["c"]))
    check("duplicate K,V: theta = 1/2", res["theta"] == 0.5)
    v_new, bias = apply_plan(v, keep, [(i, j, res["c"], res["theta"])], 1.0)
    check("identical values: value unchanged element-wise", torch.equal(v_new[j], v[j]))
    check("identical values: bias = log 2", abs(float(bias[j]) - math.log(2.0)) < 1e-12)
    e_new, e_base = evaluate_outputs(q, k, v_new, keep, bias, o, v0=v)
    check("duplicate K,V: mass compensation reproduces the dense output", e_new < 1e-24 and e_base > 1e-6)

    # (b) distinct keys, identical values: theta = 1/2 (no change), c* = least squares
    torch.manual_seed(4)
    k = torch.randn(L, d, dtype=F64)
    k[i] = k[j] + 0.3 * torch.randn(d, dtype=F64)
    a, o = dense(q, k, v)
    o0, p = baseline_outputs(q, k, v, keep)
    res = fit_pair(a[:, i], a[:, j], p[:, slot_of(keep, j)], v[i], v[j], o, o0)
    m = a[:, i] + a[:, j]
    c_ls = float((a[:, j] * m).sum() / (a[:, j] ** 2).sum())
    check("distinct K, same V: c* = sum(a_j m)/sum(a_j^2)", abs(res["c"] - c_ls) < 1e-12, f"{res['c']} vs {c_ls}")
    grid = torch.linspace(1.0, 4.0, 3001, dtype=F64)
    mass_err = ((grid[:, None] * a[:, j][None, :] - m[None, :]) ** 2).sum(1)
    at_c = float(((res["c"] * a[:, j] - m) ** 2).sum())
    check("c* minimises the mass MSE (vs 3001-point grid)", at_c <= float(mass_err.min()) + 1e-12)
    check("c* >= 1", res["c"] >= 1.0)
    check("distinct K, same V: theta = 1/2 leaves the value unchanged", res["theta"] == 0.5
          and torch.equal(apply_plan(v, keep, [(i, j, res["c"], 0.5)], 1.0)[0][j], v[j]))
    # with the value unchanged, the merge is a pure mass compensation: o_pair = u
    v_new, bias = apply_plan(v, keep, [(i, j, res["c"], res["theta"])], 1.0)
    o_new, _ = baseline_outputs(q, k, v_new, keep, bias)
    pj = p[:, slot_of(keep, j)]
    D = 1 + (res["c"] - 1) * pj
    u = (o0 + ((res["c"] - 1) * pj)[:, None] * v[j]) / D[:, None]
    check("same V: merged output equals the closed-form u", float((o_new - u).abs().max()) < 1e-12)


# --------------------------------------------------------------------------- #
def test_logit_gap_opposite_values():
    torch.manual_seed(0)
    d, W, L = 8, 40, 6
    G = 1.0
    i, j = 4, 5
    k = torch.randn(L, d, dtype=F64)
    delta = torch.randn(d, dtype=F64)
    delta = delta / delta.norm()
    k[i] = k[j] + delta
    r = torch.randn(W, d, dtype=F64)
    r = r - (r @ delta)[:, None] * delta          # query part orthogonal to the key gap
    v = torch.randn(L, d, dtype=F64)
    v[i] = -v[j]                                  # opposite values
    keep = torch.ones(L, dtype=torch.bool)
    keep[i] = False
    pos = torch.tensor([0, 20, 40, 60, 100, 101])  # only j is within max_dist of i
    prot = torch.zeros(L, dtype=torch.bool)

    def run(gaps):
        q = r + (gaps * math.sqrt(d))[:, None] * delta
        logits, a = attention_rows(q, k)
        assert torch.allclose(logits[:, i] - logits[:, j], gaps)
        o = a @ v
        o0, p = baseline_outputs(q, k, v, keep)
        res = fit_pair(a[:, i], a[:, j], p[:, slot_of(keep, j)], v[i], v[j], o, o0)
        m = a[:, i] + a[:, j]
        mass_res = float(((res["c"] * a[:, j] - m) ** 2).sum() / (m ** 2).sum())
        scale = float((o ** 2).sum(1).mean()) + 1e-8
        edges = build_sparse_costs(k, v, q, pos, keep, prot, o, o0, p, a)
        return res, mass_res, edge_cost(res["err"], 4, scale), edges

    # time-varying gap: +G on the first 30 fit rows, -G on the last 10 (the last time block)
    gaps = torch.full((W,), G, dtype=F64)
    gaps[30:] = -G
    res, mass_res, cost, edges = run(gaps)
    check("gap +-1: the mass fit is good and accepted (c <= 4, relative residual < 5%)",
          res["accepted"] and res["c"] <= 4.0 and mass_res < 0.05, f"c={res['c']:.3f} res={mass_res:.4f}")
    check("gap +-1: full-output err > 0 on every deviating query", bool((res["err"][30:] > 0).all()),
          str(res["err"][30:]))
    check("gap +-1: mean err over all fit rows is negative (a mean would accept it)",
          float(res["err"].mean()) < 0)
    check("gap +-1: block-max cost > 0, so the edge is rejected", cost > 0
          and not any(e[0] == i and e[1] == j for e in edges), f"cost={cost} edges={edges}")

    # constant gap of the same size: exactly representable (c = 1 + e^G, theta = e^G/(1+e^G))
    res_c, mass_res_c, cost_c, edges_c = run(torch.full((W,), G, dtype=F64))
    th = math.exp(G) / (1 + math.exp(G))
    check("constant gap: c = 1 + e^G and theta = e^G/(1+e^G)",
          abs(res_c["c"] - (1 + math.exp(G))) < 1e-9 and abs(res_c["theta"] - th) < 1e-9,
          f"{res_c['c']} {res_c['theta']}")
    check("constant gap: err <= 0 everywhere and edge accepted",
          bool((res_c["err"] <= 1e-12).all()) and cost_c < 0 and len(edges_c) == 1 and edges_c[0][:2] == (i, j))


# --------------------------------------------------------------------------- #
def test_gamma_zero_is_baseline():
    torch.manual_seed(5)
    L, d, W = 12, 8, 20
    k = torch.randn(L, d)
    v = torch.randn(L, d)
    q = torch.randn(W, d)
    keep = torch.ones(L, dtype=torch.bool)
    keep[[2, 5, 9]] = False
    plan = [(2, 3, 1.7, 0.4), (5, 6, 2.5, 0.9), (9, 10, 1.1, 0.0)]
    v_new, bias = apply_plan(v, keep, plan, 0.0)
    check("gamma=0: values identical element-wise", torch.equal(v_new, v))
    check("gamma=0: bias all zero", bool((bias == 0).all()) and bias.shape == (L,))
    a, o = dense(q, k, v)
    o_new, _ = baseline_outputs(q, k, v_new, keep, bias)
    o0, _ = baseline_outputs(q, k, v, keep)
    check("gamma=0: outputs identical element-wise", torch.equal(o_new, o0))
    e_new, e_base = evaluate_outputs(q, k, v_new, keep, bias, o, v0=v)
    check("gamma=0: evaluate_outputs terms identical", e_new == e_base)
    g, imp, groups = validate_plan(q, k, v, keep, plan, o, gammas=(0.0,))
    check("validate_plan with gammas=(0,) returns gamma 0 / no improvement", g == 0.0 and imp == 0.0)
    check("caller tensors untouched", v.shape == (L, d) and torch.equal(apply_plan(v, keep, plan, 1.0)[0][3],
                                                                         v[3] + 0.4 * (v[2] - v[3])))


# --------------------------------------------------------------------------- #
def test_gqa_split_pairs_empty():
    torch.manual_seed(6)
    L, d, H, Wp = 48, 8, 2, 16         # cache length, head dim, query heads per KV head, window positions
    k = torch.randn(L, d)
    v = torch.randn(L, d)
    # duplicate keys among the middle tokens so that exact-branch merges exist
    for src, dst in [(5, 6), (12, 13), (20, 21)]:
        k[src] = k[dst]
    keep = torch.zeros(L, dtype=torch.bool)
    keep[:2] = True                                     # sink
    keep[L - Wp:] = True                                # calibration window, verbatim
    keep[[6, 9, 13, 17, 21, 25, 28]] = True             # some middle tokens
    prot = torch.zeros(L, dtype=torch.bool)
    prot[:2] = True
    prot[L - Wp:] = True
    pos = torch.arange(L)

    # window queries: (Wp, H, d) -> position-major rows (Wp*H, d); causal position per row
    q_win = torch.randn(Wp, H, d)
    q_rows = q_win.reshape(Wp * H, d)
    cpos = torch.arange(L - Wp, L).repeat_interleave(H)
    split = time_split_rows(Wp, H)
    check("time split: 8/4/4 positions x 2 heads = 16/8/8 rows",
          [len(split[s]) for s in ("fit", "select", "audit")] == [16, 8, 8])
    fit_pos = cpos[split["fit"]]
    check("time split: fit rows contain both heads at every fit position",
          torch.equal(split["fit"], torch.arange(16)) and int(fit_pos.max()) < int(cpos[split["select"]].min())
          < int(cpos[split["audit"]].min()) and all(len(r) == 8 for r in split["fit_heads"]))
    check("time split: per-head rows are local to the split and interleave the heads",
          torch.equal(split["select_heads"][1], torch.arange(1, 8, 2)) and int(split["audit_heads"][0].max()) < 8)
    check("time split: a head-major half split would NOT be a time split",
          set(split["fit"].tolist()) != set(range(0, Wp * H, 2)))

    a_all, o_all = dense(q_rows, k, v, cpos)
    future = torch.arange(L)[None, :] > cpos[:, None]                  # (rows, L)
    check("causal mask: no attention to future window tokens",
          bool((a_all[future] == 0).all()) and bool(future.any()))
    o0_all, p_all = baseline_outputs(q_rows, k, v, keep, causal_positions=cpos)
    check("baseline over A renormalised (rows sum to 1)", torch.allclose(p_all.sum(1), torch.ones(Wp * H)))
    check("baseline over A causal", bool((p_all[0, -Wp + 1:] == 0).all()))

    fr = split["fit"]
    edges = build_sparse_costs(k, v, q_rows[fr], pos, keep, prot, o_all[fr], o0_all[fr], p_all[fr], a_all[fr])
    check("GQA fit rows produce candidate edges", len(edges) > 0, str(edges))
    check("all edges: evicted, unprotected source -> retained, unprotected target, |dpos|<=8",
          all((not keep[e[0]]) and (not prot[e[0]]) and keep[e[1]] and (not prot[e[1]]) and abs(e[0] - e[1]) <= 8
              for e in edges))
    check("all edges have negative cost", all(e[2] < 0 for e in edges))
    check("duplicate edges use the exact branch",
          all(e[3] == 2.0 and e[4] == 0.5 for e in edges if (e[0], e[1]) in [(5, 6), (12, 13), (20, 21)]))
    # per-edge fit_pair agrees with the batched fit inside build_sparse_costs
    e = next(e for e in edges if (e[0], e[1]) not in [(5, 6), (12, 13), (20, 21)]) if \
        any((e[0], e[1]) not in [(5, 6), (12, 13), (20, 21)] for e in edges) else edges[0]
    res = fit_pair(a_all[fr][:, e[0]], a_all[fr][:, e[1]], p_all[fr][:, slot_of(keep, e[1])], v[e[0]], v[e[1]],
                   o_all[fr], o0_all[fr], identical_keys=bool(torch.allclose(k[e[0]], k[e[1]], atol=1e-6)))
    check("batched fit equals the single-edge fit_pair", abs(res["c"] - e[3]) < 1e-5 and abs(res["theta"] - e[4]) < 1e-5,
          f"{res['c']},{res['theta']} vs {e[3]},{e[4]}")

    evicted = torch.nonzero(~keep & ~prot).flatten()
    retained = torch.nonzero(keep).flatten()
    for method in ("matching", "greedy"):
        plan = solve_partial_transport(edges, evicted, retained, method)
        srcs = [pp[0] for pp in plan]
        tgts = [pp[1] for pp in plan]
        check(f"{method}: pairs do not overlap (each token in at most one pair)",
              len(set(srcs)) == len(srcs) and len(set(tgts)) == len(tgts) and not set(srcs) & set(tgts) and len(plan) > 0)
    plan = solve_partial_transport(edges, evicted, retained, "matching")
    plan_g = solve_partial_transport(edges, evicted, retained, "greedy")
    dup_plan = [e for e in edges if (e[0], e[1]) in [(5, 6), (12, 13), (20, 21)]]
    check("matching cost <= greedy cost and <= the feasible duplicates-only plan",
          plan_cost(plan, edges) <= plan_cost(plan_g, edges) + 1e-9
          and plan_cost(plan, edges) <= sum(e[2] for e in dup_plan) + 1e-9, f"{plan} vs {plan_g}")

    v_new, bias = apply_plan(v, keep, plan, 1.0)
    tg = torch.tensor(sorted(pp[1] for pp in plan))
    nz = torch.nonzero(bias != 0).flatten()
    check("mask and bias in sync: bias only on retained merged slots", torch.equal(nz, tg) and bool(keep[nz].all()))
    check("bias on non-retained slots is zero", bool((bias[~keep] == 0).all()))
    check("values of unmerged slots untouched", torch.equal(v_new[~torch.isin(torch.arange(L), tg)],
                                                            v[~torch.isin(torch.arange(L), tg)]))
    o_L, _ = baseline_outputs(q_rows, k, v_new, keep, bias, causal_positions=cpos)
    o_m, _ = baseline_outputs(q_rows, k, v_new, keep, bias[keep], causal_positions=cpos)
    check("bias accepted as (L,) or (m,) with identical result", torch.equal(o_L, o_m))

    sr = split["select"]
    g, imp, groups = validate_plan(q_rows[sr], k, v, keep, plan, o_all[sr], per_head_rows=split["select_heads"],
                                   causal_positions=cpos[sr])
    check("validate_plan returns one improvement per query head", len(groups) == H)
    check("validate_plan gamma in the allowed set", g in (0.0, 0.25, 0.5, 1.0))
    if g > 0:
        check("chosen gamma: gain >= 5% and no head worse", imp >= 0.05 and min(groups) >= -1e-6, f"{imp} {groups}")
    ar = split["audit"]
    v_g, b_g = apply_plan(v, keep, plan, g)
    e_new, e_base = evaluate_outputs(q_rows[ar], k, v_g, keep, b_g, o_all[ar], v0=v, causal_positions=cpos[ar])
    check("audit evaluation runs and is finite", math.isfinite(e_new) and math.isfinite(e_base) and e_base > 0)

    # overlapping plan is refused
    try:
        apply_plan(v, keep, [(5, 6, 2.0, 0.5), (7, 6, 1.5, 0.5)], 1.0)
        check("apply_plan rejects overlapping pairs", False)
    except ValueError:
        check("apply_plan rejects overlapping pairs", True)

    # empty candidate set: everything protected
    prot_all = torch.ones(L, dtype=torch.bool)
    edges0 = build_sparse_costs(k, v, q_rows[fr], pos, keep, prot_all, o_all[fr], o0_all[fr], p_all[fr], a_all[fr])
    plan0 = solve_partial_transport(edges0, evicted, retained, "matching")
    g0, imp0, groups0 = validate_plan(q_rows[sr], k, v, keep, plan0, o_all[sr], per_head_rows=split["select_heads"],
                                      causal_positions=cpos[sr])
    v0_, b0_ = apply_plan(v, keep, plan0, 1.0)
    check("empty candidate set: no edges, no plan, gamma 0, cache unchanged",
          edges0 == [] and plan0 == [] and (g0, imp0, groups0) == (0.0, 0.0, [0.0, 0.0])
          and torch.equal(v0_, v) and bool((b0_ == 0).all()))
    # nothing evicted at all
    keep_all = torch.ones(L, dtype=torch.bool)
    o0_k, p_k = baseline_outputs(q_rows[fr], k, v, keep_all, causal_positions=cpos[fr])
    check("nothing evicted: build_sparse_costs returns []",
          build_sparse_costs(k, v, q_rows[fr], pos, keep_all, prot, o_all[fr], o0_k, p_k, a_all[fr]) == [])
    # a separator strictly between i and j blocks the edge; one at an endpoint does not
    k2 = k.clone()
    k2[15] = k2[13]                                                   # 15 evicted, 13 retained, 14 between
    args = (k2, v, q_rows[fr], pos, keep, prot, *dense_and_base(q_rows[fr], k2, v, keep, cpos[fr]))
    has = lambda edges_, i_, j_: any(e[:2] == (i_, j_) for e in edges_)   # noqa: E731
    check("separator strictly between i and j blocks the edge (endpoint separator does not)",
          has(build_sparse_costs(*args), 15, 13) and not has(build_sparse_costs(*args, sep_ids=[14]), 15, 13)
          and has(build_sparse_costs(*args, sep_ids=[13, 15]), 15, 13))


def dense_and_base(q, k, v, keep, cpos):
    a, o = dense(q, k, v, cpos)
    o0, p = baseline_outputs(q, k, v, keep, causal_positions=cpos)
    return o, o0, p, a


# --------------------------------------------------------------------------- #
def brute_force(edges, sources):
    """Enumerate every routing (candidate target or drop) that respects capacity."""
    by_src = {i: [] for i in sources}
    for (i, j, cost, *_rest) in edges:
        by_src[i].append((j, cost))
    best = 0.0
    for choice in itertools.product(*[[None] + by_src[i] for i in sources]):
        tg = [c[0] for c in choice if c is not None]
        if len(set(tg)) != len(tg):
            continue
        best = min(best, sum(c[1] for c in choice if c is not None))
    return best


def plan_cost(plan, edges):
    lookup = {(e[0], e[1]): e[2] for e in edges}
    return sum(lookup[(i, j)] for i, j, _, _ in plan)


def test_solver_vs_brute_force():
    # the design's toy: greedy takes -10 first and gets -10; the optimum is -9.5 + -9 = -18.5
    toy = [(0, 10, -10.0, 1.5, 0.5), (0, 11, -9.5, 1.5, 0.5), (1, 10, -9.0, 1.5, 0.5), (1, 11, 0.0, 1.5, 0.5)]
    m = solve_partial_transport(toy, [0, 1], [10, 11], "matching")
    g = solve_partial_transport(toy, [0, 1], [10, 11], "greedy")
    check("toy 2x2: matching optimum -18.5", abs(plan_cost(m, toy) + 18.5) < 1e-12 and len(m) == 2, str(m))
    check("toy 2x2: greedy gets -10", abs(plan_cost(g, toy) + 10.0) < 1e-12, str(g))

    ok_opt, ok_greedy, n_tie = True, True, 0
    for seed in range(60):
        gen = torch.Generator().manual_seed(seed)
        n_s, n_t = 4, 4
        sources = list(range(n_s))
        targets = list(range(100, 100 + n_t))
        edges = []
        for i in sources:
            for j in targets:
                if torch.rand(1, generator=gen).item() < 0.6:
                    edges.append((i, j, -float(torch.rand(1, generator=gen).item()), 2.0, 0.5))
        plan = solve_partial_transport(edges, sources, targets, "matching")
        srcs = [p[0] for p in plan]
        tgts = [p[1] for p in plan]
        feasible = len(set(srcs)) == len(srcs) and len(set(tgts)) == len(tgts)
        bf = brute_force(edges, sources)
        ok_opt &= feasible and abs(plan_cost(plan, edges) - bf) < 1e-9
        gp = solve_partial_transport(edges, sources, targets, "greedy")
        ok_greedy &= plan_cost(gp, edges) >= bf - 1e-9
        n_tie += abs(plan_cost(gp, edges) - bf) < 1e-9
    check("matching == brute force on 60 random 4x4 sparse graphs", ok_opt)
    check("greedy never beats the optimum", ok_greedy)
    check("greedy is strictly worse on some graphs", n_tie < 60, str(n_tie))

    # infeasible local support: three sources share one target; one source has no edge; a positive edge
    edges = [(0, 50, -1.0, 2.0, 0.5), (1, 50, -3.0, 2.0, 0.5), (2, 50, -2.0, 2.0, 0.5), (3, 51, 0.5, 2.0, 0.5)]
    plan = solve_partial_transport(edges, [0, 1, 2, 3, 4], [50, 51], "matching")
    check("shared single target: only the best source is merged, the rest go to drop", plan == [(1, 50, 2.0, 0.5)], str(plan))
    check("no edges at all -> everything dropped", solve_partial_transport([], [0, 1], [50], "matching") == [])
    check("non-negative edges are never taken",
          solve_partial_transport([(0, 50, 0.0, 2.0, 0.5), (1, 51, 0.3, 2.0, 0.5)], [0, 1], [50, 51], "matching") == [])
    plan_g = solve_partial_transport(edges, [0, 1, 2, 3, 4], [50, 51], "greedy")
    check("greedy on the same support agrees", plan_g == [(1, 50, 2.0, 0.5)], str(plan_g))
    try:
        solve_partial_transport([(0, 7, -1.0, 2.0, 0.5)], [0], [50], "matching")
        check("edge to a non-retained slot is refused", False)
    except ValueError:
        check("edge to a non-retained slot is refused", True)


# --------------------------------------------------------------------------- #
def test_closed_form_and_theta():
    worst = {F64: 0.0, torch.float32: 0.0}
    theta_ok = True
    for dt in (F64, torch.float32):
        for seed in range(20):
            torch.manual_seed(seed)
            d, W, L = 16, 80, 5
            q = torch.randn(W, d, dtype=dt) * 2
            k = torch.randn(L, d, dtype=dt)
            v = torch.randn(L, d, dtype=dt)
            i, j = 3, 1
            keep = torch.ones(L, dtype=torch.bool)
            keep[i] = False
            a, o = dense(q, k, v)
            o0, p = baseline_outputs(q, k, v, keep)
            pj = p[:, slot_of(keep, j)]
            res = fit_pair(a[:, i], a[:, j], pj, v[i], v[j], o, o0, c_max=1e9)
            c, th = res["c"], res["theta"]
            D = 1 + (c - 1) * pj
            u = (o0 + ((c - 1) * pj)[:, None] * v[j]) / D[:, None]
            dvec = (c * pj)[:, None] * (v[i] - v[j]) / D[:, None]
            closed = u + th * dvec
            v_new, bias = apply_plan(v, keep, [(i, j, c, th)], 1.0)
            direct, _ = baseline_outputs(q, k, v_new, keep, bias)
            worst[dt] = max(worst[dt], float((closed - direct).abs().max()))
            if dt is F64:
                # theta* vs a 1001-point grid on [0, 1]
                grid = torch.linspace(0, 1, 1001, dtype=dt)
                sse = ((u[None] + grid[:, None, None] * dvec[None] - o[None]) ** 2).sum((1, 2))
                at_star = float(((closed - o) ** 2).sum())
                theta_ok &= at_star <= float(sse.min()) + 1e-12
                # err consistency with the closed form
                theta_ok &= torch.allclose(res["err"], ((closed - o) ** 2).sum(1) - ((o0 - o) ** 2).sum(1), atol=1e-12)
    check("closed form o_pair = u + theta*dvec equals direct softmax recomputation (fp64, < 1e-6)",
          worst[F64] < 1e-6, str(worst[F64]))
    check("closed form vs direct recomputation (fp32, < 1e-5)", worst[torch.float32] < 1e-5, str(worst[torch.float32]))
    check("theta* is no worse than a 1001-point grid search, err consistent", theta_ok)
    print(f"        max |closed - direct|: fp64 {worst[F64]:.2e}, fp32 {worst[torch.float32]:.2e}")

    # rejection rules
    torch.manual_seed(9)
    W, d = 10, 4
    o = torch.randn(W, d, dtype=F64)
    o0 = torch.randn(W, d, dtype=F64)
    res = fit_pair(torch.rand(W, dtype=F64), torch.zeros(W, dtype=F64), torch.rand(W, dtype=F64) * 0.1,
                   torch.randn(d, dtype=F64), torch.randn(d, dtype=F64), o, o0)
    check("degenerate mass denominator is rejected with no change", not res["accepted"]
          and res["reason"] == "mass_denominator" and res["c"] == 1.0 and res["theta"] == 0.0 and bool((res["err"] == 0).all()))
    aj = torch.full((W,), 0.1, dtype=F64)
    res = fit_pair(aj * 5, aj, aj, torch.randn(d, dtype=F64), torch.randn(d, dtype=F64), o, o0, c_max=4.0)
    check("c* > c_max is rejected (c*=6)", not res["accepted"] and res["reason"] == "c_max" and abs(res["c"] - 6) < 1e-12)
    res = fit_pair(aj * 5, aj, aj, torch.randn(d, dtype=F64), torch.randn(d, dtype=F64), o, o0, c_max=4.0,
                   identical_keys=True)
    check("exact branch overrides the c_max cap", res["accepted"] and res["c"] == 2.0 and res["theta"] == 0.5)


# --------------------------------------------------------------------------- #

def test_joint_c_matches_bruteforce():
    """The vectorised joint (c, theta) search must equal an explicit grid search
    that refits theta at each grid point with the per-edge formulas, written
    here from the design text rather than from core.otkv_v8."""
    from core.otkv_v8 import _fit_pairs
    torch.manual_seed(7)
    E, W, d, G = 11, 23, 8, 9
    a_i = torch.rand(E, W).double() * 0.05
    a_j = torch.rand(E, W).double() * 0.05
    p_j = torch.rand(E, W).double() * 0.3 + 0.01
    v_i, v_j = torch.randn(E, d).double(), torch.randn(E, d).double()
    o, o0 = torch.randn(W, d).double(), torch.randn(W, d).double()
    ident = torch.zeros(E, dtype=torch.bool)
    c_fit, th_fit, err_fit, acc, _ = _fit_pairs(a_i, a_j, p_j, v_i, v_j, o, o0, 4.0, 1e-8, ident,
                                                joint_c=G)

    def per_edge(e, c):
        """u, dvec, the clipped theta* and the total error of one edge at a given c."""
        D = 1.0 + (c - 1.0) * p_j[e]
        u = (o0 + ((c - 1.0) * p_j[e])[:, None] * v_j[e][None]) / D[:, None]
        dvec = (c * p_j[e] / D)[:, None] * (v_i[e] - v_j[e])[None]
        dd = float((dvec * dvec).sum())
        th = 0.5 if dd < 1e-8 else min(1.0, max(0.0, float((dvec * (o - u)).sum()) / dd))
        err = ((u + th * dvec - o) ** 2).sum(-1) - ((o0 - o) ** 2).sum(-1)
        return th, float(err.sum())

    mism = 0
    for e in range(E):
        c0 = 1.0 + float((a_i[e] * a_j[e]).sum() / (a_j[e] * a_j[e]).sum())
        base_th, base_tot = per_edge(e, c0)
        best = (c0, base_th, base_tot)
        for g in range(G):
            c = min(4.0, max(1.0, c0 * (2.0 ** (-1.0 + 2.0 * g / (G - 1)))))
            th, tot = per_edge(e, c)
            if tot < best[2]:
                best = (c, th, tot)
        if abs(float(c_fit[e]) - best[0]) > 1e-9 or abs(float(th_fit[e]) - best[1]) > 1e-9:
            mism += 1
    check("joint (c, theta) search equals an independent per-edge grid search", mism == 0,
          f"{mism} of {E} edges differ")
    # and the joint fit is never worse than the mass fit it starts from
    _, _, err0, _, _ = _fit_pairs(a_i, a_j, p_j, v_i, v_j, o, o0, 4.0, 1e-8, ident, joint_c=0)
    check("joint fit never increases the total fit error",
          bool((err_fit.sum(1) <= err0.sum(1) + 1e-12).all()))

if __name__ == "__main__":
    tests = [
        ("1 identical keys, different values, background -> exact", test_identical_keys_exact),
        ("2 identical values -> theta no-op, mass compensation", test_identical_values),
        ("3 large logit gap, opposite values -> damage exposed", test_logit_gap_opposite_values),
        ("4 gamma=0 reproduces the baseline", test_gamma_zero_is_baseline),
        ("4b joint (c,theta) search vs brute force", test_joint_c_matches_bruteforce),
        ("5 GQA split, non-overlap, mask/bias sync, empty candidates", test_gqa_split_pairs_empty),
        ("6 matching == brute force; drop node", test_solver_vs_brute_force),
        ("7 closed form vs direct softmax; theta* vs grid", test_closed_form_and_theta),
    ]
    for name, fn in tests:
        print(name)
        try:
            fn()
        except Exception as exc:                        # noqa: BLE001
            FAIL.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"  FAIL  {name} raised {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)
