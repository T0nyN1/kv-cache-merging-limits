#!/usr/bin/env python
"""Independent float64 numpy check of OT-KV v8 design section 4 (research-wiki/otkv_v8_design.md).

Written from the design text only (no project imports, no implementation read).
Every quantity is recomputed from raw softmax attention; the closed forms are compared against it.

Checks
  (i)   o_pair(q;theta) = u + theta*dvec  vs direct recomputation (bias log c on slot j, value mix, softmax
        renormalised over the retained visible set), 80 rows (GQA: 4 query heads x 20 positions),
        cache sizes 5/16/64/256, full and causal visibility, random c in [1,4], theta in [0,1].
  (ii)  closed-form clipped theta* vs a 1001-point grid on [0,1].
  (iii) opposite values (+1,-1) and a query-dependent logit gap +-x: residual after optimal c*, theta*
        is first order in x; mass channel (identical values) under the v8 fixed key is ALSO first order;
        a key-averaged slot (paper's construction, not v8) is second order.
  (iv)  two disjoint identical-key pairs, exact branch c=2, theta=1/2: reproduces the dense output exactly.
  (v)   joint vs additive gains for edges on DIFFERENT retained slots: exact identity and sign of the cross term.
"""
import numpy as np

rng = np.random.default_rng(20260910)


# ----------------------------------------------------------------------------------------------
# raw attention (the only "ground truth" used anywhere)
# ----------------------------------------------------------------------------------------------
def softmax_masked(logits, mask):
    x = np.where(mask, logits, -np.inf)
    x = x - x.max(axis=1, keepdims=True)
    w = np.where(mask, np.exp(x), 0.0)
    return w / w.sum(axis=1, keepdims=True)


def attention(Q, K, V, mask, bias=None):
    """Q (R,d) rows, K,V (L,d), mask (R,L) bool visibility, bias (L,) per-slot logit bias."""
    logits = Q @ K.T / np.sqrt(K.shape[1])
    if bias is not None:
        logits = logits + bias[None, :]
    w = softmax_masked(logits, mask)
    return w @ V, w


def attention_logits(logits, V, mask, bias=None):
    if bias is not None:
        logits = logits + bias[None, :]
    w = softmax_masked(logits, mask)
    return w @ V, w


# ----------------------------------------------------------------------------------------------
# design section 4 closed forms (transcribed from the design, not from code)
# ----------------------------------------------------------------------------------------------
def pair_closed_form(o0, p_j, v_i, v_j, c):
    D = 1.0 + (c - 1.0) * p_j                                  # (R,)
    u = (o0 + (c - 1.0) * p_j[:, None] * v_j[None, :]) / D[:, None]
    dvec = c * p_j[:, None] * (v_i - v_j)[None, :] / D[:, None]
    return D, u, dvec


def c_star(a_i, a_j):
    m = a_i + a_j
    return np.sum(a_j * m) / np.sum(a_j ** 2)


def theta_star(dvec, o, u):
    den = np.sum(dvec ** 2)
    if den == 0.0:
        return 0.5, np.nan
    th_u = np.sum(dvec * (o - u)) / den
    return float(np.clip(th_u, 0.0, 1.0)), float(th_u)


def make_rows(H, W, d):
    return rng.standard_normal((H * W, d))   # GQA: H query heads sharing one KV head, W positions


# ----------------------------------------------------------------------------------------------
# (i) + (ii)
# ----------------------------------------------------------------------------------------------
def check_i_ii():
    H, W, d = 4, 20, 8
    R = H * W
    grid = np.linspace(0.0, 1.0, 1001)
    worst_i = 0.0
    worst_grid_margin = -np.inf   # max over trials of J(theta*) - min_grid J  (must be <= ~0)
    best_grid_margin = np.inf
    n_clip = 0
    n_trials = 0
    for L in (5, 16, 64, 256):
        for causal in (False, True):
            for trial in range(25):
                Q = make_rows(H, W, d) * 1.5
                K = rng.standard_normal((L, d)) * 1.5
                V = rng.standard_normal((L, d))
                if causal:
                    # queries sit at the last W positions; candidates must precede the earliest query (design s3)
                    Wq = min(W, L - 2)
                    tpos = np.arange(L - Wq, L)
                    tpos_rows = np.tile(tpos, H)[:R] if Wq == W else np.repeat(tpos, int(np.ceil(R / Wq)))[:R]
                    vis = np.arange(L)[None, :] <= tpos_rows[:, None]
                    cand = np.arange(0, L - Wq)
                else:
                    vis = np.ones((R, L), bool)
                    cand = np.arange(L)
                if len(cand) < 2:
                    continue
                i, j = rng.choice(cand, size=2, replace=False)
                keep = np.ones(L, bool)
                # evict i and a few random others (not j) so that o0 != o
                n_extra = min(max(L // 4, 0), max(len(cand) - 2, 0))
                others = [t for t in cand if t not in (i, j)]
                if n_extra > 0 and others:
                    keep[rng.choice(others, size=min(n_extra, len(others)), replace=False)] = False
                keep[i] = False
                keep[j] = True
                o, a = attention(Q, K, V, vis)
                o0, p = attention(Q, K, V, vis & keep[None, :])
                c = rng.uniform(1.0, 4.0)
                theta = rng.uniform(0.0, 1.0)
                D, u, dvec = pair_closed_form(o0, p[:, j], V[i], V[j], c)
                o_pair = u + theta * dvec
                V2 = V.copy()
                V2[j] = (1 - theta) * V[j] + theta * V[i]
                bias = np.zeros(L)
                bias[j] = np.log(c)
                o_dir, _ = attention(Q, K, V2, vis & keep[None, :], bias)
                worst_i = max(worst_i, np.abs(o_pair - o_dir).max())
                # (ii) fitted c*, theta* vs grid
                cs = c_star(a[:, i], a[:, j])
                assert cs >= 1.0 - 1e-15
                D, u, dvec = pair_closed_form(o0, p[:, j], V[i], V[j], cs)
                th, th_u = theta_star(dvec, o, u)
                if th_u < 0 or th_u > 1:
                    n_clip += 1
                J = lambda t: np.sum((u + t * dvec - o) ** 2)
                Jgrid = np.array([J(t) for t in grid])
                margin = (J(th) - Jgrid.min()) / max(Jgrid.min(), 1e-300)
                worst_grid_margin = max(worst_grid_margin, margin)
                best_grid_margin = min(best_grid_margin, margin)
                n_trials += 1
    print(f"(i)  closed form vs direct softmax: max |o_pair - o_direct| = {worst_i:.3e}  over {n_trials} trials "
          f"(L in 5/16/64/256, full+causal, 80 GQA rows)")
    print(f"(ii) theta*: relative J(theta*) - min_grid(1001): max {worst_grid_margin:.3e} (0 only when clipped to a grid point), "
          f"min {best_grid_margin:.3e}; unconstrained theta outside [0,1] (clipped) in {n_clip}/{n_trials} trials")
    assert worst_i < 1e-12
    assert worst_grid_margin <= 1e-12


# ----------------------------------------------------------------------------------------------
# (iii) first-order value channel; v8 fixed-key mass channel; key-averaged mass channel
# ----------------------------------------------------------------------------------------------
def check_iii():
    """Gap pattern Delta(q) = Delta0 + x*s_q with s standardised (mean 0, RMS 1, several levels), background fixed.
    Schemes: v8 (key of j unchanged, bias log c*, mix theta*) and, for comparison only, the paper's weight-matched
    averaged key l_bar = l_j + w*Delta with w = sigmoid(Delta0) (bias fitted by the same LS rule, theta* fitted)."""
    d = 4
    nB = 6
    R = 80
    lB = rng.standard_normal(nB)                 # background logits, query independent
    VB = rng.standard_normal((nB, d))
    s = rng.standard_normal(R); s = (s - s.mean()); s = s / np.sqrt(np.mean(s ** 2))
    e1 = np.zeros(d); e1[0] = 1.0
    xs = np.logspace(-3, 0, 7)
    L = nB + 2
    i, j = nB + 1, nB

    def run(x, Delta0, v_i, v_j):
        logits = np.zeros((R, L)); logits[:, :nB] = lB[None, :]; logits[:, j] = 0.0; logits[:, i] = Delta0 + x * s
        V = np.vstack([VB, v_j[None, :], v_i[None, :]])
        full = np.ones((R, L), bool); keep = full.copy(); keep[:, i] = False
        o, a = attention_logits(logits, V, full); o0, p = attention_logits(logits, V, keep)
        out = {}
        cs = c_star(a[:, i], a[:, j]); D, u, dvec = pair_closed_form(o0, p[:, j], V[i], V[j], cs); th, _ = theta_star(dvec, o, u)
        out["v8"] = np.sqrt(np.mean(np.sum((u + th * dvec - o) ** 2, axis=1))); out["c"] = cs; out["theta"] = th
        # first-order prediction for v8: residual(q) ~ (Delta(q)-Delta0) * a_i(q) * ||v_i - o(q)||  (softmax derivative)
        out["pred"] = np.sqrt(np.mean((x * s * a[:, i] * np.linalg.norm(V[i][None, :] - o, axis=1)) ** 2))
        # paper-style weight-matched averaged key (not v8)
        w = 1.0 / (1.0 + np.exp(-Delta0))
        la = logits.copy(); la[:, j] = logits[:, j] + w * (logits[:, i] - logits[:, j])
        o0a, pa = attention_logits(la, V, keep)
        abar = np.exp(la[:, j]) / np.exp(logits).sum(axis=1)
        ca = np.sum(abar * (a[:, i] + a[:, j])) / np.sum(abar ** 2)
        Da, ua, dva = pair_closed_form(o0a, pa[:, j], V[i], V[j], ca); tha, _ = theta_star(dva, o, ua)
        out["avg"] = np.sqrt(np.mean(np.sum((ua + tha * dva - o) ** 2, axis=1)))
        return out

    print("(iii) residual RMS_q ||o_pair - o|| vs gap fluctuation x  (Delta(q) = Delta0 + x s_q, s standardised, background fixed)")
    for Delta0 in (0.0, -1.2):
        print(f"      Delta0 = {Delta0:+.1f}:   x     | v8 opposite  v8 identical | avg-key opposite  avg-key identical |  c*      theta*   | v8/first-order-pred (opposite)")
        tab = {}
        for x in xs:
            ro = run(x, Delta0, -e1, +e1); ri = run(x, Delta0, +e1, +e1)
            tab[x] = (ro, ri)
            print(f"                    {x:8.4f} | {ro['v8']:.3e}   {ri['v8']:.3e}  | {ro['avg']:.3e}        {ri['avg']:.3e}        | "
                  f"{ro['c']:.4f}  {ro['theta']:.4f}  | {ro['v8'] / ro['pred']:.5f}")
        def slope(sel):
            lx = np.log(xs[:4]); ly = np.log([sel(tab[x]) for x in xs[:4]]); return np.polyfit(lx, ly, 1)[0]
        print(f"      Delta0 = {Delta0:+.1f}: log-log slopes on x in [1e-3, 3e-2]:  v8 opposite {slope(lambda t: t[0]['v8']):.3f} | "
              f"v8 identical {slope(lambda t: t[1]['v8']):.3f} | avg-key opposite {slope(lambda t: t[0]['avg']):.3f} | "
              f"avg-key identical {slope(lambda t: t[1]['avg']):.3f}")
    # general random values, Delta0 = -1.2: coefficient check
    for x in (1e-3, 1e-2):
        vi, vj = rng.standard_normal(d), rng.standard_normal(d)
        r = run(x, -1.2, vi, vj)
        print(f"      random values, Delta0=-1.2, x={x:g}: v8 residual / first-order prediction = {r['v8'] / r['pred']:.5f}")
    # constant gap across rows -> exact for ANY x and ANY values (single evicted token)
    worst = 0.0
    for x in (0.3, 1.0, 2.5):
        logits = np.zeros((R, L)); logits[:, :nB] = lB[None, :]; logits[:, i] = x
        V = np.vstack([VB, rng.standard_normal((2, d))])
        full = np.ones((R, L), bool); keep = full.copy(); keep[:, i] = False
        o, a = attention_logits(logits, V, full); o0, p = attention_logits(logits, V, keep)
        cs = c_star(a[:, i], a[:, j]); D, u, dvec = pair_closed_form(o0, p[:, j], V[i], V[j], cs)
        th, _ = theta_star(dvec, o, u)
        worst = max(worst, np.abs(u + th * dvec - o).max())
        assert abs(cs - (1 + np.exp(x))) < 1e-12 and abs(th - 1 / (1 + np.exp(-x))) < 1e-12
    print(f"      constant gap x across rows (random unequal values): max |o_pair - o| = {worst:.3e}  (c* = 1+e^x, theta* = sigmoid(x) exactly)")


# ----------------------------------------------------------------------------------------------
# (iv) two disjoint identical-key pairs, exact branch
# ----------------------------------------------------------------------------------------------
def check_iv():
    H, W, d = 4, 20, 16
    R = H * W
    L = 14
    worst = 0.0
    worst_extra = 0.0
    for trial in range(50):
        Q = make_rows(H, W, d) * 2.0
        K = rng.standard_normal((L, d)) * 2.0
        V = rng.standard_normal((L, d))
        i1, j1, i2, j2 = 3, 1, 9, 6
        K[i1] = K[j1]; K[i2] = K[j2]          # identical keys, different values
        vis = np.ones((R, L), bool)
        keep = np.ones(L, bool); keep[[i1, i2]] = False
        V2 = V.copy(); V2[j1] = 0.5 * (V[j1] + V[i1]); V2[j2] = 0.5 * (V[j2] + V[i2])
        bias = np.zeros(L); bias[[j1, j2]] = np.log(2.0)
        o, _ = attention(Q, K, V, vis)
        o_joint, _ = attention(Q, K, V2, vis & keep[None, :], bias)
        worst = max(worst, np.abs(o_joint - o).max())
        # with one more evicted token (not merged): output equals attention over F \ {i3} exactly
        i3 = 11
        keep3 = keep.copy(); keep3[i3] = False
        o_joint3, _ = attention(Q, K, V2, vis & keep3[None, :], bias)
        keep_ref = np.ones(L, bool); keep_ref[i3] = False
        o_ref, _ = attention(Q, K, V, vis & keep_ref[None, :])
        worst_extra = max(worst_extra, np.abs(o_joint3 - o_ref).max())
    print(f"(iv) two disjoint identical-key pairs, c=2 theta=1/2 jointly: max |o_joint - o_dense| = {worst:.3e}")
    print(f"     + one extra unmerged eviction: max |o_joint - o(F minus i3)| = {worst_extra:.3e}  (residual = that eviction only)")
    assert worst < 1e-12 and worst_extra < 1e-12


# ----------------------------------------------------------------------------------------------
# (v) joint vs additive gains, edges on distinct retained slots
# ----------------------------------------------------------------------------------------------
def check_v():
    H, W, d = 4, 20, 8
    R = H * W
    L = 24
    nE = 3
    worst_delta = 0.0
    worst_gain = 0.0
    n_pos = n_neg = 0
    n_pos_ben = n_neg_ben = 0
    lead_ratio = []
    rel = []
    n_ben_joint_harm = 0
    for trial in range(300):
        Q = make_rows(H, W, d) * 1.5
        K = rng.standard_normal((L, d)) * 1.5
        V = rng.standard_normal((L, d))
        vis = np.ones((R, L), bool)
        perm = rng.permutation(L)
        I = perm[:nE]; Jn = perm[nE:2 * nE]           # distinct evicted i_e, distinct retained j_e
        keep = np.ones(L, bool); keep[I] = False
        keep[perm[2 * nE:2 * nE + 3]] = False           # a few other evictions so o0 != o
        o, a = attention(Q, K, V, vis)
        o0, p = attention(Q, K, V, vis & keep[None, :])
        r0 = o0 - o
        # fitted edges (design rule) so that edges are typically beneficial
        cs, ths = [], []
        for e in range(nE):
            c = min(c_star(a[:, I[e]], a[:, Jn[e]]), 4.0)
            D, u, dvec = pair_closed_form(o0, p[:, Jn[e]], V[I[e]], V[Jn[e]], c)
            th, _ = theta_star(dvec, o, u)
            cs.append(c); ths.append(th)
        for scale in (1.0, 1e-1, 1e-2, 1e-3):
            deltas, dl, gains = [], [], []
            V2 = V.copy(); bias = np.zeros(L)
            for e in range(nE):
                c = 1.0 + scale * (cs[e] - 1.0)
                th = scale * ths[e]
                D, u, dvec = pair_closed_form(o0, p[:, Jn[e]], V[I[e]], V[Jn[e]], c)
                Delta_e = u + th * dvec - o0          # single-edge output change
                deltas.append(Delta_e); dl.append(D - 1.0)
                gains.append(np.sum(r0 ** 2, axis=1) - np.sum((r0 + Delta_e) ** 2, axis=1))
                V2[Jn[e]] = (1 - th) * V[Jn[e]] + th * V[I[e]]; bias[Jn[e]] = np.log(c)
            o_joint, _ = attention(Q, K, V2, vis & keep[None, :], bias)
            Delta_E = o_joint - o0
            sdl = sum(dl)
            lam = [(1.0 + dl[e]) / (1.0 + sdl) for e in range(nE)]
            Delta_pred = sum(lam[e][:, None] * deltas[e] for e in range(nE))
            worst_delta = max(worst_delta, np.abs(Delta_E - Delta_pred).max())
            G_E = np.sum(r0 ** 2, axis=1) - np.sum((r0 + Delta_E) ** 2, axis=1)
            G_sum = sum(gains)
            # exact identity: G_E - sum G_e = -sum (1-lam_e) G_e + sum lam_e(1-lam_e)|Delta_e|^2 - 2 sum_{e<f} lam_e lam_f Delta_e.Delta_f
            cross = np.zeros(R)
            for e in range(nE):
                cross += -(1 - lam[e]) * gains[e] + lam[e] * (1 - lam[e]) * np.sum(deltas[e] ** 2, axis=1)
                for f in range(e + 1, nE):
                    cross += -2 * lam[e] * lam[f] * np.sum(deltas[e] * deltas[f], axis=1)
            worst_gain = max(worst_gain, np.abs((G_E - G_sum) - cross).max())
            if scale == 1.0:
                diff = np.sum(G_E - G_sum)
                if diff > 0: n_pos += 1
                else: n_neg += 1
                if all(np.sum(g) > 0 for g in gains):
                    if diff > 0: n_pos_ben += 1
                    else: n_neg_ben += 1
                    rel.append(diff / np.sum(G_sum))
                    if np.sum(G_E) < 0: n_ben_joint_harm += 1
            # leading second-order term: -2 sum_{e<f} Delta_e.Delta_f + 2 sum_e (r0.Delta_e) sum_{f!=e} delta_f
            lead = np.zeros(R)
            for e in range(nE):
                lead += 2 * np.sum(r0 * deltas[e], axis=1) * (sdl - dl[e])
                for f in range(e + 1, nE):
                    lead += -2 * np.sum(deltas[e] * deltas[f], axis=1)
            if scale < 1.0:
                lead_ratio.append((scale, np.sum(G_E - G_sum) / np.sum(lead)))
    print(f"(v)  joint change identity Delta_E = sum_e lambda_e Delta_e, lambda_e=(1+delta_e)/(1+sum delta): max err = {worst_delta:.3e}")
    print(f"     exact gain identity G_E - sum G_e = -sum(1-lam_e)G_e + sum lam_e(1-lam_e)|Delta_e|^2 - 2 sum_(e<f) lam_e lam_f Delta_e.Delta_f: max err = {worst_gain:.3e}")
    print(f"     sign of (joint - additive) over 300 random 3-edge sets (fitted c*,theta*): joint>sum in {n_pos}, joint<sum in {n_neg}; "
          f"restricted to sets whose 3 edges are each beneficial on fit: joint>sum in {n_pos_ben}, joint<sum in {n_neg_ben}")
    rel = np.array(rel)
    print(f"     among beneficial sets: (joint - sum)/sum  median {np.median(rel):+.4f}, 10th pct {np.percentile(rel, 10):+.4f}, "
          f"90th pct {np.percentile(rel, 90):+.4f}, min {rel.min():+.4f}, max {rel.max():+.4f}; "
          f"joint gain negative although every edge is beneficial: {n_ben_joint_harm}")
    for sc in (1e-1, 1e-2, 1e-3):
        rs = [r for s_, r in lead_ratio if s_ == sc]
        print(f"     leading-order cross term check, perturbation scale {sc:g}: (joint-additive)/(second-order formula) "
              f"mean {np.mean(rs):.5f}, min {np.min(rs):.5f}, max {np.max(rs):.5f}")
    assert worst_delta < 1e-12 and worst_gain < 1e-12


# ----------------------------------------------------------------------------------------------
# small sanity items from design s10.A: identical V, gamma=0
# ----------------------------------------------------------------------------------------------
def check_misc():
    H, W, d = 4, 20, 8
    R = H * W; L = 12
    Q = make_rows(H, W, d); K = rng.standard_normal((L, d)); V = rng.standard_normal((L, d))
    i, j = 4, 2
    V[i] = V[j]                         # identical values, different keys
    vis = np.ones((R, L), bool); keep = np.ones(L, bool); keep[i] = False
    o, a = attention(Q, K, V, vis); o0, p = attention(Q, K, V, vis & keep[None, :])
    cs = c_star(a[:, i], a[:, j])
    D, u, dvec = pair_closed_form(o0, p[:, j], V[i], V[j], cs)
    assert np.abs(dvec).max() == 0.0
    th, th_u = theta_star(dvec, o, u)
    # gamma = 0 restores the baseline bit for bit
    bias = np.zeros(L); V2 = V.copy(); V2[j] = V[j] + 0.0 * th * (V[i] - V[j])
    og, _ = attention(Q, K, V2, vis & keep[None, :], bias)
    print(f"(misc) identical values: dvec == 0 -> theta = {th} (no effect); c* = {cs:.4f}; gamma=0 output == baseline: "
          f"{np.array_equal(og, o0)}")


if __name__ == "__main__":
    np.set_printoptions(precision=4, linewidth=140)
    check_i_ii()
    check_iii()
    check_iv()
    check_v()
    check_misc()
    print("ALL CHECKS PASSED")
