# OT-KV v8 — implementation contract (from research-wiki/otkv_v8_design.md)

Everything below is per KV head of one layer, with real post-RoPE keys/queries and float32 maths.
Notation: cache positions 0..L-1; retained set A (bool mask, |A| = m); evicted set E = ~A minus the
protected regions; fit queries Q_fit (W_fit, d) with absolute positions; dense attention output o(q);
baseline output o0(q) = softmax over A only; p_j(q) = baseline attention weight of retained slot j.

## Module `core/otkv_v8.py` (pure functions, no cache class in this round)

```python
def attention_rows(q, k, causal_positions=None):      # (W,d),(L,d) -> logits (W,L), softmax rows (W,L)
def baseline_outputs(q, k, v, keep_mask, bias=None):  # -> o0 (W,d), p (W,m) renormalised over keep_mask (+ optional per-slot logit bias)
def fit_pair(a_i, a_j, p_j, v_i, v_j, o, o0, c_max=4.0, eps=1e-8, identical_keys=False):
    """a_i,a_j: dense attention of evicted i / retained j over fit queries (W,); p_j: baseline weight of j (W,);
    v_i,v_j: (d,); o,o0: (W,d). Returns dict(c, theta, err (W,), accepted: bool, reason).
    c*  = sum(a_j*m)/sum(a_j^2), m=a_i+a_j; reject if sum(a_j^2)<eps or c*>c_max.
    D   = 1+(c-1)p_j;  u = (o0+(c-1)p_j v_j)/D;  dvec = c p_j (v_i-v_j)/D
    theta* = clip(sum(dvec.(o-u))/sum(|dvec|^2), 0, 1); theta=1/2 if v_i==v_j or denominator<eps.
    identical_keys=True -> c=2, theta=1/2 (exact branch).
    err(q) = |u+theta dvec-o|^2 - |o0-o|^2   (negative = merge helps on that query)."""
def build_sparse_costs(k, v, q_fit, pos, keep_mask, protect_mask, o, o0, p, a_dense,
                       max_dist=8, max_cand=4, n_blocks=4, c_max=4.0, sep_ids=None):
    """For every evicted i (not protected): up to max_cand nearest retained j with |pos_i-pos_j|<=max_dist,
    not crossing a separator position (sep_ids optional). Per edge: fit_pair; cost C_ij = max over n_blocks
    contiguous time blocks of the block-mean err / scale, scale = mean_fit |o|^2 + eps (one scale per head).
    Keep only accepted edges with C_ij < 0. Returns edges: list of (i, j, C_ij, c, theta)."""
def solve_partial_transport(edges, n_evicted_ids, retained_ids, method="matching"):
    """Unit supply per evicted i, capacity 1 per retained j, drop node at cost 0. Exact solution by
    maximum-weight bipartite matching (weights -C_ij>0) via scipy.optimize.linear_sum_assignment on a sparse->dense
    matrix with drop columns; method="greedy" = most-negative-cost-first respecting both capacities (the
    comparator for hypothesis 3). Returns list of chosen (i, j, c, theta)."""
def apply_plan(v, keep_mask, plan, gamma):            # -> v_new (L,d) with v_j <- v_j + gamma*theta*(v_i-v_j); bias (L,) = gamma*log c on merged j, 0 elsewhere
def validate_plan(q_sel, k, v, keep_mask, plan, o_sel, gammas=(0,0.25,0.5,1.0), min_gain=0.05, per_head_rows=None):
    """Apply all chosen pairs jointly, recompute full attention output over A with the bias on the SELECT
    queries, per gamma; choose the gamma with the largest mean relative improvement of |o_pair-o|^2 over
    |o0-o|^2, require >= min_gain and no query-head group worse (per_head_rows: list of row index tensors,
    one per query head sharing this KV head); else gamma=0. Returns gamma, improvement, per-group improvements."""
def evaluate_outputs(q, k, v_new, keep_mask, bias, o):  # -> mean |o'-o|^2 and mean |o0-o|^2 on any query set
```

Conventions: everything torch, device-agnostic; no in-place edits of caller tensors; identical-key detection
by torch.allclose(k_i, k_j, atol=1e-6). Causal handling: fit/select/audit queries are the last W prefill
positions; all candidate pairs are at positions < the earliest calibration query, so they are visible to all
calibration rows; the dense rows a(q) still use the causal mask for the calibration block itself.

## Harness `experiments/v8_offline.py`

Model Qwen3-1.7B (default; `--model` for Llama-3.1-8B at bf16), device mps/cuda, eager attention.
Prompts: N continuation prompts (wikitext, 2500 tokens) and N retrieval prompts (build_niah_samples), `--prompts 3`.
Per prompt: capture post-RoPE K, Q (all positions), V for every `--layer-stride` layer (same projection/norm/RoPE path as
experiments.classa_oracle.capture_kqv); greedy decode `--new 64` steps with repeated-trigram blocking (the rule of
experiments.sigma_decode.decode_queries) capturing the decode queries AND the post-RoPE keys/values of the generated tokens.
Retained set (shared across heads of a layer, per the design): sink 4 + calibration window W=64 (last 64 prefill positions, kept verbatim) + top-k middle by pooled last-32-query attention (SnapKV rule, kernel 7, summed over KV heads) to reach budget fraction `--budget` (0.05 and 0.10) of L. Protected: sink, window, and the separator neighbourhood.
Splits: calibration window rows split by TIME into fit/select/audit = 32/16/16 positions (all query heads of the group, rows grouped after the time split); decode rows = the 64 generated steps.
Caveat (code review, 2026-09-10): the SnapKV observation window (last 32 positions) coincides with the select+audit
positions, so the retained set is chosen with the audit queries in view — "audit" is an in-sample diagnostic with respect
to selection; the decode rows are the held-out test.
Decode-row cache: the compressed prefill cache PLUS the generated tokens' keys/values (kept verbatim, protected — never a
merge source or receiver), attended causally (row at position L+t sees generated slots <= t), as at deployment. The
prefill-only variant (old protocol) is reported per head as `rel_decode_prefill_only` (`--decode-prefill-only` switches the whole run).
Conditions per head (same A, same candidate edges, same fit): (1) baseline eviction; (2) v8-OT (matching, gamma chosen on select); (3) v8-greedy; (4) v8-OT without bias (c forced to 1, theta refit with the same fitter, no identical-key theta override); (5) random pairing among the same locality candidates with an accepted fit, same number of merges as ot (all candidates fitted first; up to 20 shuffles to reach ot's cardinality), same gamma rule; (6) non-deployable oracle: costs, c, theta fitted on the decode rows themselves (extended cache), matching solver, gamma chosen on the same decode rows among the allowed gammas with min_gain = 0.
Fitter (2026-09-10, after the first review round): `_fit_pairs(..., joint_c=9)` — c* from the mass fit, then a
one-dimensional search over c on a 9-point log grid in [c*/2, 2c*] (clipped to [1, c_max]) with the conditional optimum
theta(c), keeping the c with the smallest total fit error of the FULL output (joint (c, theta) fit). Ablations:
`massc` (c from the mass fit only — the original design), `nobias` (c = 1, theta refit), `biasonly` (joint c, theta
forced to 0: the receiver keeps its own value and only its mass is re-fitted). Further conditions: `online` (costs, c,
theta fitted on decode steps 1-32, gamma chosen with the standard gate on steps 33-48, judged on steps 49-64 — the
"late" rows, reported for every condition as `rel_decode_late`), `oracle` (fitted on all 64 decode rows, gamma chosen on
the same rows with min_gain 0, joint fit).
All conditions use ONE fitter (core.otkv_v8._fit_pairs, with its `force_c` / `force_theta` / `joint_c` paths) and ONE gamma policy (core.otkv_v8.validate_plan: best feasible gamma — largest mean improvement among the gammas with improvement >= min_gain and no query-head group worse).
Metrics: mean |o'-o|^2 / mean |o0-o|^2 (relative full-output error) on audit rows and on decode rows, per head; proposed and APPLIED (gamma > 0) merge counts and the applied fraction of evicted attention mass (under decode rows); solver wall time; fraction of heads where v8-OT is better than baseline on decode rows; per-head `invalid` flag if any metric is non-finite.
Outputs: JSON `runs/local_repro/v8_offline_<model>_b<budget>.json` with per-head records and a summary; a printed table. Runs end-to-end in ~2.5 min on the laptop for 3+3 prompts at layer stride 4 (Qwen3-1.7B, MPS).

## Gates (design §10 C; research decisions, not theorems; as implemented in `summarise()`)
Head-level CIs are descriptive only (the heads of one prompt share its text, retained set and decode); the gates use a
prompt-clustered bootstrap (2000 resamples of whole prompts).
- H1 (deployable merge beats eviction) = ALL of: no invalid records; decode-row relative error improvement of v8-OT vs
  baseline >= 5% on average; prompt-clustered bootstrap CI of that gain excludes 0; not driven by a few heads (mean gain
  with the top 10% of heads removed >= 2.5%); applied merges >= 1 per head on average and applied merged mass > 1% of the
  evicted decode attention mass.
- H3 (matching is a contribution) = no invalid records and the prompt-clustered CI of (greedy - ot) on decode rows > 0.
- Oracle: `oracle_gain_positive` (clustered CI > 0) and `oracle_gain_ge_5pct` are reported separately; the design's stop
  rule ("if the decode-fitted oracle shows no gain, stop the fixed-key scheme") refers to the first.

## Deployable cache `core/otkv_v8_cache.py` (`OTKVv8Cache(SnapKVCache)`, method string `otkv8;...`)
Pooled-SnapKV selection (observation_window 32, pool_kernel 7, per_head false) + the v8 merge at prefill end, per layer
and KV head, from the stored attention rows of the last `row_window`=64 prefill queries (fit = first 32 positions, select =
next 16; candidates restricted to positions < L-64 so every calibration row sees them; no separator rule); joint (c, theta)
fit, matching, gamma on the select rows with the standard gate; values of receivers rewritten before the prune, per-slot
logit bias injected at decode through the bias-aware pre-hook (evaluation/models/wrapper.py, `consolidate=True`
duck-typed flag), pruned with the same keep set at every maintenance step and padded by one per new token.
`experiments/test_otkv_v8_cache.py` checks the cache reproduces the harness plan (values and bias) on random tensors and
that the bias stays aligned with the cache over decode steps. Registered in main.py as `otkv8` (extra keys: row_window,
fit_rows, sel_rows, max_dist, max_cand, n_blocks, c_max, joint_c, min_gain, gammas, solver, merge).

## End-task ablation of the deployable cache (2026-09-11)

Llama-3.1-8B, 5 % budget, LongBench dev, 400 paired samples each, all three sharing ONE base (pooled SnapKV, method
string `snapkv;observation_window=32;per_head=false;pool_kernel=7`), so each row isolates what the operator does to an
identical selection:

| construction | method string | LongBench delta |
|---|---|---|
| keys fixed, mass + mixture | `otkv8;observation_window=32;per_head=false;pool_kernel=7` | +0.70 [+0.14, +1.37], perm p = 0.017 |
| keys fixed, mass only (theta = 0) | `...;force_theta=0` | +0.08 [-0.47, +0.75], p = 0.81 |
| keys fixed, mixture only (c = 1) | `...;force_c=1` | +0.20 [-0.18, +0.63], p = 0.35 |
| keys averaged | `kvmerger;observation_window=32;per_head=false;pool_kernel=7` | -0.58 [-1.23, +0.05] |

Only the joint row resolves; it leads each ablation by +0.62 and +0.50 under the task bootstrap, but neither contrast
survives exact randomisation over the eight task means (p = 0.13, 0.07) and the interaction is unresolved everywhere.
The mechanism-level ablation (attention-output error on decode rows, `experiments/v8_offline.py`) orders the channels
the OTHER way round: mass-only keeps most of the output-error recovery (6-18 %) and no-bias keeps almost none
(<= 1.6 %). Reducing the squared attention-output error is therefore not the same as improving the task, which is
worth stating explicitly because output-error surrogates are what several recent selectors optimise.
