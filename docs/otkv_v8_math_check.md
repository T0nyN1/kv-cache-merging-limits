# OT-KV v8 — independent check of the §4 pair formulas

Status: mathematical verification of `research-wiki/otkv_v8_design.md` §4 (single-edge output, `c*`, `theta*`, exact
branch) plus the joint-vs-additive question for §5/§6. Derived from the design text only; no implementation was read.
Numerical companion: `docs/otkv_v8_math_check.py` (numpy float64, no project imports).

Check command (from the worktree root):

    ~/anaconda3/envs/ot-kv/bin/python docs/otkv_v8_math_check.py

Its full output is reproduced in §7. Seed fixed (`20260910`); results are deterministic.

Verdict in one paragraph: every formula in §4 is exact as written (single-edge output, `D`, `u`, `dvec`, `c*`,
the clipped `theta*`, the identical-key branch and its disjoint-pair extension). Nothing in §4–§8 is wrong. Three
statements deserve sharpening, none of which is a formula error: (a) the "mass channel is second order" property
that the project's paper proves for a *key-averaged* compensated slot does **not** hold for v8's fixed-key slot — in
v8 both the mass and the value channel are first order in the query-to-query fluctuation of the logit gap (§4 below,
verified numerically: slope 1.00 for identical values, versus 2.00 for the averaged key); (b) the identical-key exact
branch is a special case of a stronger true statement: a single v8 edge is exact on a query set iff the gap
`l_i(q) − l_j(q)` is constant on that set, with `c = 1+e^gap`, `theta = sigmoid(gap)`; (c) the additive OT surrogate
`Σ C_ij` is neither an upper nor a lower bound on the joint gain even for edges on distinct slots — the exact
cross term (§5) has a dilution part that is negative for beneficial edges and an interference part of indefinite
sign; empirically the sum over-estimates the joint gain in ~90% of random 3-edge sets (median −4.4%).

---

## 1. Setting and notation

One KV head, head dimension `d`, cache slots `k = 1..L` with keys `k_k`, values `v_k`. For a query row `q`
(a (position, query-head) pair; under GQA all `H` query heads of the group are rows over the same K/V), let
`l_k(q) = q·k_k/√d`, and let `F(q)` be the set of slots visible to `q` (causal mask). Retained set `A ⊂ F`
(the same for every row), evicted token `i ∉ A`, anchor `j ∈ A`. Per row:

    Z_F = Σ_{k∈F} e^{l_k}          a_k = e^{l_k}/Z_F         o  = Σ_{k∈F} a_k v_k       (dense)
    Z_A = Σ_{k∈A} e^{l_k}          p_k = e^{l_k}/Z_A         o0 = Σ_{k∈A} p_k v_k       (baseline eviction)

`m = a_i + a_j` is the pair's dense mass. Pre-existing per-slot biases (the contract's optional `bias`) enter only
through `l_k ← l_k + b_k`; nothing below changes. All fitting sums (`Σ_fit`) run over rows, so GQA rows are
handled by construction (the check uses 4 query heads × 20 positions = 80 rows).

## 2. Single-edge output — derivation of `D`, `u`, `dvec`

Apply one edge `(i → j)`: keys unchanged, slot `j` gets logit bias `log c` and value
`v'_j = (1−θ) v_j + θ v_i`, every other retained slot unchanged, softmax renormalised over `A`.

New normaliser:

    Z' = Σ_{k∈A, k≠j} e^{l_k} + c e^{l_j} = Z_A − e^{l_j} + c e^{l_j} = Z_A [1 + (c−1) p_j] =: Z_A · D

so `p'_k = p_k / D` for `k ≠ j` and `p'_j = c p_j / D`. Output:

    o_pair = Σ_{k≠j} (p_k/D) v_k + (c p_j/D) [(1−θ) v_j + θ v_i]
           = [o0 − p_j v_j + c p_j v_j]/D + θ · c p_j (v_i − v_j)/D
           = [o0 + (c−1) p_j v_j]/D  +  θ · c p_j (v_i − v_j)/D
           = u + θ · dvec                                                    (design §4.2, exact)

with `D = 1 + (c−1) p_j`, `u = [o0 + (c−1) p_j v_j]/D`, `dvec = c p_j (v_i − v_j)/D`. No approximation; the
denominator is carried exactly. Numerically (check (i)): max |closed form − direct softmax| = 4.0e-15 over 200
random trials, `L ∈ {5,16,64,256}`, full and causal visibility, `c ∈ [1,4]`, `θ ∈ [0,1]`, 80 GQA rows.

Sanity limits: `c=1, θ=0` gives `o0` (γ=0 in §7 restores the baseline bit for bit — checked); `p_j=0` on a row gives
`o_pair = o0` on that row (the edge is invisible where `j` has no mass).

## 3. Closed forms `c*` and `theta*`

**Mass channel (§4.1).** Objective `Σ_fit (c a_j − m)²` is a scalar least squares in `c`:

    c* = Σ_fit a_j m / Σ_fit a_j²  = 1 + Σ a_i a_j / Σ a_j²  ≥ 1      (since m ≥ a_j)

exact, and `c* ≥ 1` holds in exact arithmetic exactly as the design states. The per-row ideal is
`c(q) = m/a_j = 1 + e^{Δ(q)}`, `Δ = l_i − l_j` (normaliser cancels); `c*` is its `a_j²`-weighted mean.

**Value channel (§4.2).** With `c` and `k_j` fixed, `J(θ) = Σ_fit ‖u + θ dvec − o‖²` is a convex quadratic in `θ`
(leading coefficient `Σ‖dvec‖² ≥ 0`), so the constrained minimiser over `[0,1]` is the clipped unconstrained one:

    theta* = clip_[0,1]( Σ_fit dvec·(o − u) / Σ_fit ‖dvec‖² )

exact. If `v_i = v_j` then `dvec ≡ 0` and `θ` has no effect anywhere (checked: `dvec == 0`, `θ = 1/2` returned).
Check (ii): `J(theta*) − min over a 1001-point grid ≤ 0` in all 200 trials (equality only when `theta*` is clipped
to 0 or 1, both grid points; 44/200 trials clipped), best relative improvement over the grid −2.0e-6, i.e. the
`O(h²)` you expect from `h = 10⁻³`.

Two remarks, neither a formula error:

* "分母退化时用 θ=1/2；此时 θ 不改变输出" is literally true only when `dvec ≡ 0` on the *fit* rows. If the
  denominator is tiny because `p_j ≈ 0` on the fit rows while `v_i ≠ v_j`, `θ` still changes the stored `v'_j`,
  which a future query can read. In practice that edge is already rejected by the `Σ a_j² < eps` guard of §4.1,
  so the statement is harmless, but the contract should reject on either denominator rather than return `θ = 1/2`.
* `(c*, theta*)` is a conditional pair, not the joint minimiser of the output MSE — exactly as §4.2 says. For the
  record, the joint problem is one-dimensional: for each `c` the optimal `θ(c)` is the closed form above, so a
  1-D search over `c` is cheap if ever wanted.

## 4. Exactness, and the first-order residual (check (iii))

**When is a single v8 edge exact?** Take `c = 1 + e^{Δ(q)}`. Then `Z' = Z_A + e^{l_i} = Z_F` (if `i` is the only
evicted token), `p'_k = a_k` for `k ≠ j`, `p'_j = a_i + a_j`, and `o_pair = o` iff `θ = a_i/(a_i+a_j) = sigmoid(Δ(q))`.
Both are constants only if `Δ(q)` is the same on every row. Hence

> a single v8 edge reproduces the dense pair contribution exactly on a query set **iff** the logit gap
> `Δ(q) = (k_i − k_j)·q/√d` is constant on that set; then `c* = 1 + e^Δ`, `theta* = sigmoid(Δ)` exactly.

Checked: constant gap `x ∈ {0.3, 1, 2.5}`, random unequal values — max |o_pair − o| = 6.7e-16 and the fitted
`(c*, theta*)` equal `(1+e^x, sigmoid(x))` to 1e-12. The design's identical-key branch (`Δ ≡ 0`, `c=2`, `θ=1/2`) is
the special case, and its disjoint-pair extension (§8.2) follows from `Z' = Z_A + Σ_e e^{l_{i_e}} = Z_F` with each
merged numerator `2e^{l_j}(v_i+v_j)/2 = e^{l_i}v_i + e^{l_j}v_j`. Check (iv): two disjoint identical-key pairs with
different values plus 10 background slots, applied jointly with `c=2, θ=1/2`: max |o_joint − o_dense| = 1.8e-15; with
one further unmerged eviction `i₃`, `o_joint` equals attention over `F∖{i₃}` to 1.8e-15 (only that eviction's
error remains, as §8.2 claims).

**First-order residual.** Write `Δ(q) = Δ̄ + δ(q)` with `δ` the query-to-query fluctuation. `o_pair` does not depend
on `l_i(q)` at all (the token is gone; only `k_j`, `c`, `θ` remain), while the dense output does, with the softmax
derivative `∂o/∂l_i = a_i (v_i − o)`. Since the edge is exact at `δ = 0` (previous paragraph) and the fitted
`(c*, theta*)` differ from `(1+e^{Δ̄}, sigmoid(Δ̄))` only at second order when `δ` is centred,

    o(q) − o_pair(q) = δ(q) · a_i(q) · (v_i − o(q)) + O(δ²)
                     = δ a_i (v_i − v_j)      [value channel: the pair's internal mixture moves]
                     + δ a_i (v_j − o)        [mass channel: the pair's total mass moves]

Both terms are first order in `δ`. For opposite values `v_i = −v_j = ±e` and a balanced pair (`Δ̄ = 0`,
`a_i = a_j = m/2`) the value term is `δ (m/2)(v_i − v_j) = δ m e·(−1)`, i.e. the paper's "`tanh(x/2) ≈ x/2`"
per unit pair mass; the mass term adds `δ (m/2)(v_j − o)` on top.

Numerical result (gap pattern `Δ(q) = Δ̄ + x s_q`, `s` standardised with several levels, background fixed;
residual = RMS over rows of `‖o_pair − o‖` after fitting `c*`, `theta*`):

| `Δ̄` | scheme | values | log-log slope in `x` (x ∈ [1e-3, 3e-2]) | residual at x = 1e-3 |
|---|---|---|---|---|
| 0 | v8 (key fixed) | opposite (+1, −1) | **1.001** | 1.33e-4 |
| 0 | v8 (key fixed) | identical | **1.001** | 7.46e-5 |
| 0 | weight-matched averaged key + bias (paper §"Compensated merging", not v8) | opposite | 1.001 | 9.49e-5 |
| 0 | weight-matched averaged key + bias | identical | **2.002** | 2.68e-8 |
| −1.2 | v8 (key fixed) | opposite / identical | 1.001 / 1.001 | 4.60e-5 / 2.58e-5 |
| −1.2 | averaged key + bias | opposite / identical | 1.001 / 2.003 | 4.71e-5 / 1.42e-8 |

The measured residual divided by the first-order prediction `x · RMS_q[s_q a_i ‖v_i − o‖]` is 0.9999 at
`x = 10⁻³` and 0.999 at `10⁻²` (also for random values at `Δ̄ = −1.2`); the fitted `c*`, `theta*` stay at
`(2.0000, 0.5000)` / `(1.3012, 0.2315) = (1+e^{−1.2}, sigmoid(−1.2))` up to `O(x²)`, as derived. The full table is in §7.

**Interpretation — what "second order" does and does not apply to.** The paper's second-order statement is for a
merged slot whose *key is the weight-matched average* `k̄ = w k_i + (1−w) k_j`, `w = sigmoid(Δ̄)`: its logit tracks
`l_j + wΔ(q)` per query, so the ideal log-bias `log(1+e^Δ) − wΔ` is stationary at `Δ̄` and the mass residual is
`O(δ²)`, while the value channel stays first order (`δ a_i (1−w)(v_i − v_j)`, coefficient `m/4` for a balanced pair —
the paper's `σ/4`). v8 keeps `k_j`, so its merged slot's logit does not track the pair; the mass term
`δ a_i (v_j − o)` survives and the **v8 mass channel is first order too** (slope 1.00 above, versus 2.00 for the
averaged key). So the requested confirmation holds — the value channel is first order — but the premise "the mass
channel is second order" is a property of the averaged-key construction, not of v8. The design text never claims
second order for v8 (§4.1 only claims the LS optimum; §8.5 correctly says the full output is evaluated), and its
§5 cost `e_ij` measures the full output error, so both first-order terms are already inside `C_ij`. The practical
consequence is worth stating in the design: a v8 pair is accurate to the extent that `(k_i − k_j)` is nearly
orthogonal to the calibration queries (constant gap), independently of value agreement; value agreement removes
the first term only. The design's own "do not assume key cosine implies mergeable" stance is therefore right for
the wrong reason: key similarity in the *query subspace* is necessary for v8, not sufficient.

## 5. Several edges on different slots — is `Σ` of single-edge gains a bound?

Edges `e = (i_e → j_e)` with distinct `j_e` (capacity 1) and distinct `i_e`. Let `δ_e := (c_e − 1) p_{j_e}` be the
extra mass at `j_e`, `s := Σ_e δ_e`, and `Δ_e := o_e − o0` the single-edge output change (so `(1+δ_e) Δ_e =
N_e − δ_e o0`, where `N_e = (c_e−1)p_{j_e}v_{j_e} + θ_e c_e p_{j_e}(v_{i_e} − v_{j_e})` is edge `e`'s numerator
increment). Applying all edges jointly, the normaliser is `Z_A(1+s)` and

    o_E − o0 = (Σ_e N_e − s o0)/(1+s) = Σ_e λ_e Δ_e,       λ_e := (1+δ_e)/(1+s) ∈ (0, 1]      (exact)

each edge's change is diluted by the mass the *other* edges add. Verified to 3.3e-15 (check (v)).

Gains (positive = improvement; `r0 = o0 − o`): `G_e = ‖r0‖² − ‖r0 + Δ_e‖² = −2 r0·Δ_e − ‖Δ_e‖²`, and
`G_E = −2 r0·(Σλ_eΔ_e) − ‖Σλ_eΔ_e‖²`. Substituting `−2 r0·Δ_e = G_e + ‖Δ_e‖²` gives the **exact identity**

    G_E − Σ_e G_e = − Σ_e (1−λ_e) G_e                     (dilution: ≤ 0 whenever G_e ≥ 0)
                    + Σ_e λ_e (1−λ_e) ‖Δ_e‖²               (dilution also shrinks the quadratic penalty: ≥ 0, third order)
                    − 2 Σ_{e<f} λ_e λ_f Δ_e·Δ_f              (interference: sign of −Δ_e·Δ_f)

verified per row to 2.4e-14. With `1 − λ_e = Σ_{f≠e} δ_f /(1+s)` and `G_e = −2 r0·Δ_e + O(Δ²)`, the **exact
second-order cross term** (in the edge perturbations `δ`, `Δ`) is

    G_E − Σ_e G_e  =  −2 Σ_{e<f} Δ_e·Δ_f  +  2 Σ_e (r0·Δ_e) Σ_{f≠e} δ_f  +  O(3rd order)

(ratio to the exact discrepancy 0.9995 ± 0.03 at perturbation scale 10⁻³, 0.995 at 10⁻², see §7).

Answer: **neither bound holds in general.** For a beneficial edge `r0·Δ_e < 0`, so the second (dilution) term is
negative, and if the two edges push the output the same way (`Δ_e·Δ_f > 0`, both correcting the same residual
component) the interference term is negative too — then the additive surrogate over-estimates the joint gain
(sum is an upper bound to second order). But if the edges' changes are opposed in some coordinates (`Δ_e·Δ_f < 0`,
their individual over-shoots cancel), the interference term is positive and the joint gain can exceed the sum;
a 2-D example: `r0 = (1,0)`, `Δ_1 = (−0.4, 0.6)`, `Δ_2 = (−0.4, −0.6)` gives `G_1 = G_2 = 0.28` but `G_E = 0.96`.
Even the dilution term is not sign-definite once third-order terms are kept: shrinking an edge whose `Δ_e`
over-shoots along its own direction (optimal scale `−r0·Δ_e/‖Δ_e‖² < 1`) *helps*.

Empirically (300 random 3-edge sets on 24 slots, edges fitted with the §4 rules, 80 rows): joint < sum in 266/300;
among the 273 sets whose three edges are each beneficial, joint < sum in 247 and joint > sum in 26; relative
discrepancy `(joint − sum)/sum` median −4.4%, 10th percentile −11%, min −27%, max +10%; in no set did a
per-edge-beneficial plan have a negative joint gain (this is not a theorem — it is what 3 edges on 24 slots do).
The dilution grows with the total added mass `s = Σ_e (c_e − 1) p_{j_e}` of the *whole* plan, so with many merges per
head the surrogate becomes systematically optimistic by roughly the factor `1/(1+s)`; the §7 γ-selection on the
select rows is the design's guard against this and should stay.

## 6. Verdict on the design's formulas

| Design statement | Status |
|---|---|
| §4.2 `D`, `u`, `dvec`, `o_pair = u + θ dvec` | exact (4.0e-15 vs direct softmax, incl. causal and GQA rows) |
| §4.1 `c* = Σ a_j m / Σ a_j²`, `c* ≥ 1` | exact scalar LS solution; `c* = 1 + Σa_i a_j/Σa_j² ≥ 1` |
| §4.2 clipped `theta*` = constrained optimum | exact (convex quadratic); never worse than a 1001-point grid |
| §4.2 exact branch `c=2, θ=1/2` for `k_i = k_j` | exact; special case of "constant gap ⇒ `c = 1+e^Δ`, `θ = sigmoid(Δ)`" |
| §8.2 disjoint identical-key pairs exact, other evictions' error unchanged | exact (1.8e-15) |
| §8.1 `‖v'_j‖ ≤ max(‖v_i‖,‖v_j‖)` | convexity of the norm |
| §8.3 γ = 0 restores baseline | bit-exact |
| §6 "OT optimum is for the additive surrogate only; joint softmax couples edges" | correct; cross term now explicit (§5) — not a bound in either direction |
| §4.2 "θ = 1/2 when the denominator degenerates; θ then does not change the output" | true on fit rows only; stored `v'_j` still changes if `v_i ≠ v_j` — reject instead |
| (implicit in the task, and in the paper for averaged keys) "mass channel is second order" | **not true for v8's fixed key**: first order with coefficient `a_i (v_j − o)`; second order only for the weight-matched averaged key |

No formula in §4 is wrong. Two wording/design notes worth adding to the design: (1) the exactness condition
"constant gap on the query set" and the first-order residual `δ a_i (v_i − o)`, making explicit that v8 trades the
averaged-key's second-order mass channel for key stability; (2) the sign structure of the joint cross term, so
the OT surrogate is described as "optimistic on average, not a bound".

## 7. Numerical output (verbatim, `docs/otkv_v8_math_check.py`)

```
(i)  closed form vs direct softmax: max |o_pair - o_direct| = 3.997e-15  over 200 trials (L in 5/16/64/256, full+causal, 80 GQA rows)
(ii) theta*: relative J(theta*) - min_grid(1001): max 0.000e+00 (0 only when clipped to a grid point), min -2.016e-06; unconstrained theta outside [0,1] (clipped) in 44/200 trials
(iii) residual RMS_q ||o_pair - o|| vs gap fluctuation x  (Delta(q) = Delta0 + x s_q, s standardised, background fixed)
      Delta0 = +0.0:   x     | v8 opposite  v8 identical | avg-key opposite  avg-key identical |  c*      theta*   | v8/first-order-pred (opposite)
                      0.0010 | 1.330e-04   7.457e-05  | 9.491e-05        2.680e-08        | 2.0000  0.5000  | 0.99991
                      0.0032 | 4.207e-04   2.359e-04  | 3.002e-04        2.682e-07        | 2.0000  0.5000  | 0.99972
                      0.0100 | 1.331e-03   7.463e-04  | 9.500e-04        2.687e-06        | 2.0000  0.5000  | 0.99902
                      0.0316 | 4.219e-03   2.365e-03  | 3.011e-03        2.703e-05        | 2.0003  0.5001  | 0.99601
                      0.1000 | 1.347e-02   7.552e-03  | 9.613e-03        2.757e-04        | 2.0031  0.5011  | 0.97902
                      0.3162 | 4.491e-02   2.519e-02  | 3.182e-02        2.937e-03        | 2.0302  0.5109  | 0.87824
                      1.0000 | 1.809e-01   1.022e-01  | 1.180e-01        2.957e-02        | 2.2124  0.6015  | 0.72911
      Delta0 = +0.0: log-log slopes on x in [1e-3, 3e-2]:  v8 opposite 1.001 | v8 identical 1.001 | avg-key opposite 1.001 | avg-key identical 2.002
      Delta0 = -1.2:   x     | v8 opposite  v8 identical | avg-key opposite  avg-key identical |  c*      theta*   | v8/first-order-pred (opposite)
                      0.0010 | 4.595e-05   2.576e-05  | 4.706e-05        1.423e-08        | 1.3012  0.2315  | 0.99990
                      0.0032 | 1.454e-04   8.149e-05  | 1.488e-04        1.424e-07        | 1.3012  0.2315  | 0.99967
                      0.0100 | 4.600e-04   2.579e-04  | 4.710e-04        1.428e-06        | 1.3012  0.2315  | 0.99882
                      0.0316 | 1.459e-03   8.178e-04  | 1.493e-03        1.438e-05        | 1.3013  0.2316  | 0.99495
                      0.1000 | 4.671e-03   2.619e-03  | 4.771e-03        1.478e-04        | 1.3025  0.2323  | 0.97158
                      0.3162 | 1.598e-02   8.961e-03  | 1.607e-02        1.651e-03        | 1.3147  0.2400  | 0.82626
                      1.0000 | 8.466e-02   4.753e-02  | 7.845e-02        2.482e-02        | 1.4409  0.3213  | 0.50533
      Delta0 = -1.2: log-log slopes on x in [1e-3, 3e-2]:  v8 opposite 1.001 | v8 identical 1.001 | avg-key opposite 1.001 | avg-key identical 2.003
      random values, Delta0=-1.2, x=0.001: v8 residual / first-order prediction = 0.99990
      random values, Delta0=-1.2, x=0.01: v8 residual / first-order prediction = 0.99882
      constant gap x across rows (random unequal values): max |o_pair - o| = 6.661e-16  (c* = 1+e^x, theta* = sigmoid(x) exactly)
(iv) two disjoint identical-key pairs, c=2 theta=1/2 jointly: max |o_joint - o_dense| = 1.776e-15
     + one extra unmerged eviction: max |o_joint - o(F minus i3)| = 1.776e-15  (residual = that eviction only)
(v)  joint change identity Delta_E = sum_e lambda_e Delta_e, lambda_e=(1+delta_e)/(1+sum delta): max err = 3.333e-15
     exact gain identity G_E - sum G_e = -sum(1-lam_e)G_e + sum lam_e(1-lam_e)|Delta_e|^2 - 2 sum_(e<f) lam_e lam_f Delta_e.Delta_f: max err = 2.358e-14
     sign of (joint - additive) over 300 random 3-edge sets (fitted c*,theta*): joint>sum in 34, joint<sum in 266; restricted to sets whose 3 edges are each beneficial on fit: joint>sum in 26, joint<sum in 247
     among beneficial sets: (joint - sum)/sum  median -0.0440, 10th pct -0.1110, 90th pct -0.0010, min -0.2694, max +0.1034; joint gain negative although every edge is beneficial: 0
     leading-order cross term check, perturbation scale 0.1: (joint-additive)/(second-order formula) mean 0.96369, min 0.52419, max 1.82774
     leading-order cross term check, perturbation scale 0.01: (joint-additive)/(second-order formula) mean 0.99538, min 0.79507, max 1.02475
     leading-order cross term check, perturbation scale 0.001: (joint-additive)/(second-order formula) mean 0.99949, min 0.96788, max 1.00231
(misc) identical values: dvec == 0 -> theta = 0.5 (no effect); c* = 1.6148; gamma=0 output == baseline: True
ALL CHECKS PASSED
```

Construction notes for §7: (i)/(ii) use random N(0,1)-scaled Q, K, V with 4 query heads × 20 positions (80 GQA rows
over one KV head), L ∈ {5,16,64,256}, half the trials with a causal mask where queries sit at the last positions and
the candidate pair precedes the earliest query (design §3), a random extra eviction set so `o0 ≠ o`, random
`c ∈ [1,4]`, `θ ∈ [0,1]` for (i) and fitted `c*, theta*` for (ii). (iii) builds logits directly (any logit pattern is
realisable with keys; the formulas depend on logits and values only) with 6 background slots, `l_j = 0`,
`l_i = Δ̄ + x s_q`, values `v_j = +e₁`, `v_i = −e₁` (or equal). (iv) uses L = 14, d = 16, two identical-key pairs with
different values, 50 trials. (v) uses L = 24, 3 edges on distinct slots, 3 further plain evictions, 300 trials, with
the perturbation scaled by `c_e ← 1 + ε(c_e−1)`, `θ_e ← ε θ_e` for the leading-order check.
