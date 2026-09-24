# OT-KV v7 — budget transport

The design note for `core/ot_kv_v7.py`. Prior context: `docs/SUMMARY.md`
(the v1–v6 record), `docs/ot_kv_theory.md` (why value merging is closed).

## 1. Where v7 starts from

The v1–v6 investigation ended with three facts.

1. **Merging is closed.** A query-independent value merge removes at most
   `e^{-σ²}` of the eviction error and breaks even only below σ ≤ 0.285;
   post-RoPE keys sit at σ ≈ 1.2 under oracle routing, at every budget and
   context length, even for verbatim-repeated text. Nothing profitable can be
   transported *into* anchors.
2. **Every measured gain came from selection** — which tokens keep a slot
   (pooled-evidence ranking +0.91, the qa/continuation regime switch +0.64),
   and, weakly, from how the budget is laid out across layers (pyramid +0.09).
3. **The best previous configuration (v4 auto + pyramid, 23.98) leads
   PyramidKV (23.67) by a margin 400 samples cannot resolve.** To claim a win
   the method needs mechanisms worth ≥ +1 point combined, and a test split
   with more samples.

So v7 does not touch the merge. It asks Lemma 1 for the rest of its budget:

    o_evict(q) − o(q) = ρ(q) · [ v̄_A(q) − v̄_E(q) ]

The squared error contributed by evicting token `i` under query `q` is, to
first order in its attention weight,

    ‖ a_i(q) · ( v_i − v̄_A(q) ) ‖²  ≈  a_i(q)² · ‖ v_i − v̄ ‖²

Three selection-side mechanisms follow directly.

## 2. The three mechanisms

**(a) Value-magnitude weighting** (`vnorm_tau`, default 1.0).
The error weight of a token is attention × value displacement, not attention
alone. v7 ranks on `ŝ_i · ‖v_i‖^τ` with the uncentred norm as the cheap proxy
for `‖v_i − v̄‖` (value centroids concentrate near zero relative to individual
values, measured in the theory note: `|v̄_A − v̄_E|/‖v‖ = 0.291`). The same
weighting appears in the literature as VATP (Guo et al., 2024) with reported
gains over SnapKV/H2O on most LongBench tasks; here it is the Lemma 1 factor.

**(b) Local max-pooling of the selection score** (`pool_kernel`, default 7).
The window score is a ~32-query estimate of where *future* queries will look.
Future queries hit the same semantic unit but not the same position, so the
per-position estimate has positional jitter; a token-axis max over a small
neighbourhood keeps the unit a spike belongs to rather than one token of it.
Published SnapKV ships exactly this pooling and this repository's
re-implementation had dropped it — so the same option is restored to the
baseline (`snapkv;pool_kernel=7`) and the final table reports the baseline in
its stronger form. A claimed win must survive that.

**(c) Cross-layer water-filling** (`layer_alloc="waterfill"`).
Uniform budgets equalise token counts per layer; the objective wants to
equalise the *marginal evicted mass*. Let `w_l(i)` be layer `l`'s middle-region
(attention × value-norm) score and `T_l` the layer's total mass including the
frozen sink/recent regions. The allocation problem is

    max_{B_1..B_L}  Σ_l Σ_{i ≤ B_l} g_l(i)     s.t.  Σ_l B_l = L · B̄,
    g_l(i) = i-th largest of w_l / T_l          floor ≤ B_l ≤ cap

Because the objective is separable and concave in each `B_l` (sorted gains are
non-increasing), the greedy — pool all layers' per-slot marginal gains and keep
the globally largest `L·B̄` — solves the LP exactly. This is the transportation
problem "move a fixed stock of slots to layers"; it is what remains of the
"optimal transport" in OT-KV, applied where the evidence says transport
matters: allocation, not value movement. A floor (0.25× uniform) and cap
(2.5× uniform) bound the damage if the score stream mis-predicts the future.

The same greedy applied across *heads* is Ada-KV (Feng et al., 2024); across
layers it is the data-driven version of PyramidKV's hand-tuned ramp and of
CAKE's dispersion heuristic. Heads share one rectangular tensor per layer in
this framework (no varlen kernel), so v7 water-fills across layers only —
per-layer tensors are independent, so ragged budgets are free there.

Allocation uses the **unpooled** weighted scores: pooling is a robustness prior
on *which token in a unit survives*, not extra error mass, so it must not
inflate a layer's claim on the budget.

## 3. What is inherited unchanged

From the v3/v4 engine: dual score streams (window mass selects, all-query mass
is bookkept), the per-layer qa/continuation regime detector (`auto_regime`),
pooled cross-head evidence for window ranking (`select_scope="global"`),
k-means coverage in the continuation regime, per-head transport bookkeeping,
`compress_interval=4` with the reserve that caps the peak cache *at* budget.
Merging stays available behind `merge=true` for ablation only.

Budget parity is preserved by construction: Σ_l B_l equals the uniform total,
and `experiments/test_v7.py` asserts it (29 checks, plus the original 44
multi-head checks still passing).

## 4. Evaluation protocol

The +0.31 lesson from v4 (19 configurations, expected best-of-n under the null
+1.17) dictates the protocol:

- **dev split** = LongBench samples 0–50 of the 8 English tasks — the split all
  cached baselines were run on; used to pick the v7 configuration.
- **test split** = samples 50–150 (`--longbench_offset 50`), touched once, for
  the final table: v7, its three single-mechanism ablations, v4-auto+pyramid,
  PyramidKV (global), SnapKV (global), SnapKV (global, pooled), H2O, dense.
- paired bootstrap + sign test throughout (`experiments/paired_stats.py`).
- NIAH and wikitext perplexity reported from the same harness; local KL is
  *not* used for ranking (`docs/SUMMARY.md` §8).
