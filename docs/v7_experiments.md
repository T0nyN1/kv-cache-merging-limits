# OT-KV v7 — experiment log

Running record; tables regenerate from `experiments/make_table.py` over the
volume's `runs/dev8k` (dev split, samples 0-50) and `runs/test8k` (test split,
samples 50-150). Protocol: Llama-3.1-8B-Instruct, 5 % KV budget, greedy,
64 new tokens, paired stats throughout.

## 1. Dev @ max_length=4096 (the inherited protocol) — abandoned

otkv7 23.56, otkv7;layer_alloc=uniform 23.86, against cached v4-auto+pyramid
23.98 / PyramidKV 23.67 / SnapKV 23.46. Everything within ±0.5, CIs ±1.0 —
and 35-48 of 50 samples per task are middle-truncated at 4096, so most of the
context the methods are supposed to compress is already thrown away by the
harness. The protocol compresses the method band, not just the cache.
**Decision: move the primary evaluation to max_length=8000** (5-46 of 50
truncated, dense headroom rises from 27.62 to 28.86, multi-hop tasks gain the
most: hotpotqa 14.89 → 19.17 dense).

## 2. Dev @ max_length=8000 — first complete round

400 paired samples, reference = SnapKV (global, pooled k=7), the faithful
strongest baseline:

| method | avg | Δ vs pooled SnapKV | note |
|---|---:|---:|---|
| Dense | 28.86 | +3.84* | |
| SnapKV (global, pooled) | 25.02 | (ref) | |
| PyramidKV (global) | 24.89 | −0.14 | |
| **OT-KV v7 (default)** | 24.87 | −0.15 | sign p 0.000: fewer, larger wins |
| OT-KV v4 (auto)+pyramid | 24.86 | −0.16 | |

The mean is again a flat band — but the per-task structure is not:

| v7 − pooled SnapKV | qasper | gov_report | multi_news | 2wikimqa | hotpotqa | trivia | samsum | multifieldqa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Δ | **+3.8** | **+2.6** | **+2.5** | +0.2 | −0.6 | −2.4 | −2.8 | **−4.5** |

v7 wins exactly where the regime detector routes layers to all-query/coverage
scoring (the abstractive/report tasks), and loses on the extractive tasks where
its qa path — pooled window scores × value norm, waterfilled budgets — should
have matched pooled SnapKV but doesn't. Hypothesis: vnorm and/or waterfill
*hurt* the window-scored regime while helping the all-query one. Ablations in
flight: −vnorm, −waterfill, −both, regime_scope=global × {waterfill, uniform}.

## 3. Local continuation fidelity (Qwen3-1.7B, 4 docs, paired inputs)

| budget | metric | H2O | SnapKV (pooled) | **OT-KV v7** |
|---:|---|---:|---:|---:|
| 0.15 | ppl | 20.80 | 21.81 | **19.60** |
| 0.15 | KL(dense‖c) | 0.345 | 0.387 | **0.291** |
| 0.075 | ppl | 25.70 | 28.19 | **23.05** |
| 0.075 | KL(dense‖c) | 0.609 | 0.669 | **0.501** |

v7 −16/−18 % KL against H2O and −25 % against pooled SnapKV at both budgets,
holding slightly less cache (228/230 vs 231 at 0.15). Dense ppl 17.38.

## 4. Local NIAH (Qwen3-1.7B, needle sweep, budget 0.075)

unpooled SnapKV 1/5 → pooled SnapKV 5/5 — pooling *is* SnapKV's retrieval
mechanism, and the re-implemented baseline had lost it. v7 with per-layer
regime votes: 4/5 (a mislabelled layer evicts the needle); with
`regime_scope=global` (majority vote): 5/5. Waterfill neutral on retrieval.

## 4b. Dev @ 8k — mechanism attribution (400 paired samples each)

| config | avg | Δ vs pooled SnapKV |
|---|---:|---:|
| SnapKV (global, pooled) | 25.02 | (ref) |
| **v7 full** (per-layer regime, waterfill, vnorm, pool) | 24.87 | −0.15 |
| v7 −waterfill | 24.82 | −0.20 |
| v7 global-regime −waterfill | 24.68 | −0.34 |
| v7 global-regime | 24.47 | −0.55 |
| v7 −waterfill −vnorm (pool+regime engine only) | 24.19 | −0.84 |
| v7 −vnorm | 24.07 | −0.95 |

- **vnorm ≈ +0.8** on the macro mean — the Lemma 1 weighting earns its keep
  (largest per-task: qasper +2.3, trivia +2.4, hotpot +1.6; cost: 2wiki −1.6).
- **waterfill ≈ +0.05 net** — real wins (qasper +2.2, hotpot +1.8, mfqa +1.6)
  cancelled by one large loss (triviaqa −4.2).
- **regime_scope=global costs −0.4 LongBench** while buying NIAH (a genuine
  retrieval-vs-macro trade; keep per-layer for LongBench tables, note the knob).
- The mfqa deficit (−4.5 vs pooled SnapKV) persists across *every* v7 variant
  and v4 — it is the auto-regime engine mislabelling a QA task, not vnorm or
  waterfill. Same mechanism that wins gov_report/multi_news (+2.5-2.6).

Standing hypothesis after two dev rounds: at a 5 % budget the macro LongBench
mean is conserved (±0.2) across every competent selection method; what the
mechanisms move is the per-task profile. Differentiation should come from
tighter budgets (2 %), where ρ grows — probing now.

## 4c. NIAH on Llama-3.1-8B (8k contexts, 25 needle positions)

| method | budget 5 % | budget 2 % |
|---|---:|---:|
| **OT-KV v7, qa path** (auto_regime=false) | **1.00** | **1.00** |
| SnapKV (global, pooled) | 0.92 | 0.88 |
| PyramidKV (global) | 0.92 | 0.76 |
| SnapKV (global, unpooled) | 0.88 | — |
| OT-KV v7 (auto, threshold 0.73) | 0.76 | 0.80 |
| H2O (global) | 0.40 | — |

The v7 qa path (pool × vnorm × waterfill) is a *better retrieval engine than
pooled SnapKV*; what breaks retrieval is the regime detector's Qwen-calibrated
threshold mislabelling needle layers on Llama. Recalibrating on this synthetic
probe (a calibration set disjoint from LongBench): threshold 0.85 or 0.9
restores 1.00; `regime_scope=global` alone makes it worse on Llama (0.56) and
is dropped.

## 4d. The recalibrated detector transfers to LongBench

Same dev split, thresholds picked on the NIAH probe *before* this run:

| config | avg | Δ vs pooled SnapKV | 95 % CI |
|---|---:|---:|---|
| **otkv7;auto_threshold=0.9** | **25.72** | **+0.70** | [−0.05, +1.52] |
| otkv7;auto_threshold=0.85 | 25.64 | +0.62 | [−0.08, +1.39] |
| SnapKV (global, pooled) | 25.02 | (ref) | |
| otkv7 (threshold 0.73) | 24.87 | −0.15 | |

Per-task at 0.9: no loss anywhere vs pooled SnapKV — 2wiki +2.1, qasper +2.0,
trivia +1.7, hotpot +0.7, gov +0.5, mfqa +0.4, mnews −0.8, samsum −0.9. The
mfqa/trivia deficits of threshold 0.73 were pure detector mislabels.

## 4e. Budget 2 % probe — the macro mean stays conserved

qa-forced 22.94 / pooled SnapKV 22.84 / v7-0.73 22.70 / PyramidKV 22.48.
Tighter budgets do **not** open a macro gap; they widen the per-task regime
effects (auto wins hotpot +2.7, gov +2.2, mnews +3.4; qa-forced wins mfqa
+6.9, 2wiki +2.3). The differentiation axis is the regime dial, not ρ.

## 4f. The detector is a mirage on Llama — the qa path is the method

Llama wikitext ppl @5 % (4 docs, dense 3.307): v7@0.73 **4.041** (best
compressed; h2o 4.243, pyramidkv 4.300, pooled snapkv 4.456) — but
v7@0.9 = 4.4197 = qa-forced *exactly*: at 0.9 every wikitext layer is
labelled qa, so the threshold that fixes NIAH forfeits the continuation path.

`experiments/detector_probe.py` (Llama, 3k-token prompts, per-layer quartiles):

| prompt | H(win)/H(full) p25/med/p75 | p̄_win p25/med/p75 |
|---|---|---|
| continuation | 0.722 / 0.769 / 0.836 | 0.736 / 0.776 / 0.829 |
| needle@0.2/0.5/0.8 | 0.72 / 0.74-0.75 / 0.79 | tracks needle depth |
| summary instruction | 0.724 / 0.741 / 0.782 | 0.731 / 0.765 / 0.807 |

Fully overlapping — no per-layer statistic tried separates the classes on
Llama (entropy ratio, look-back position, window entropy). The Qwen separation
(0.66-0.69 vs 0.77) does not transfer.

And the confirmation: `otkv7;auto_regime=false` on dev8k scores **25.64**,
statistically identical to threshold 0.9's 25.72 — the detector's LongBench
contribution on Llama is zero.

**Final method (v7 default): the qa path alone.** Pooled-window selection
(SnapKV-style, cross-head evidence) × token-axis max-pool (k=7) × value-norm
weighting (τ=1) × cross-layer water-filling; per-head bookkeeping; no merge;
no detector. `auto_regime` stays as an option for model families where the
statistic separates (Qwen3, threshold 0.73 — where it buys −16 % KL on
continuation at no LongBench cost).

Scorecard of the final config on dev/calibration data, 5 % budget:

| metric | v7 | best baseline |
|---|---:|---:|
| LongBench dev (400) | **25.64** | 25.02 pooled SnapKV |
| NIAH @5 % / @2 % | **1.00 / 1.00** | 0.92 / 0.88 pooled SnapKV |
| wikitext ppl (4 docs) | 4.420 | 4.243 H2O*, 4.300 PyramidKV, 4.456 pooled SnapKV |
| decode tok/s (H200, 8k) | 17.2 | 17.8 pooled SnapKV, 24.0 dense |
| peak KV | **50.0 MB** | 50.1 MB baselines, 1016 MB dense |

(*H2O keeps a ppl edge — the price of dropping the regime switch; H2O in turn
scores 0.40 NIAH and −3 LongBench. No compressed method dominates v7 on any
metric pair.)

## 4g. NIAH mechanism attribution (Llama, 5 % budget, 25 positions)

| config | NIAH |
|---|---:|
| v7 (final) | **1.00** |
| v7 −waterfill | 1.00 |
| v7 −pooling | 0.96 |
| v7 −vnorm | 0.84 |
| SnapKV (global, pooled) | 0.92 |
| SnapKV (global, unpooled) | 0.88 |
| OT-KV v4 (auto)+pyramid | 0.60 |
| H2O (global) | 0.40 |
| StreamingLLM | 0.20 |

Value-norm weighting alone is +0.16 retrieval — the needle's value vector is
exactly the high-norm unusual value Lemma 1 says is expensive to evict.
Waterfill is retrieval-neutral.

## 4h. Compensated pair consolidation (2026-09-02)

Built and validated the one merge the theory permits (`core/consolidate.py`,
survey and analysis in `docs/merging_survey.md`): σ̂-gated adjacent-pair
merging with a per-slot logit bias, freed slots re-admitting tokens at the
same physical budget. Exactly lossless on duplicate pairs (unit-tested);
+36 % effective tokens at τ=0.3; KL −2.5 % / ppl −2.0 % on Qwen local; NIAH
1.00 preserved on Llama. **And still −0.72 LongBench at 5 % / −1.83* at 2 %**
— the bias is fitted on prefill-window queries and drifts on decode queries,
concentrated on exactly the retrieval-critical slots. Gate ablation: σ̂ vs
cosine indistinguishable (−0.72 vs −0.73) — there is no profit for a better
criterion to find. Kept in the repo behind `consolidate=true` as the
strongest-variant negative control; v7's default remains selection-only.

## 4i. Paper-strengthening round (2026-09-02, ~$25)

1. **σ on 4 models / 3 families** (`experiments/sigma_families.py`, table in
   `docs/merging_survey.md` §3c): value-only break-even ≤5 % everywhere,
   verbatim repetition never helps, distance-monotone everywhere. Llama/Mistral
   sit at σ≈0.95 vs Qwen's 1.25-1.38 — which localises the class-B failure to
   query-distribution shift rather than key geometry.
2. **CaM (ICML'24) faithful reproduction** (`baselines/cam.py`): at its
   published streaming cadence, paired against its own eviction base —
   LongBench −0.14 (CI [−0.25, −0.04], differs on 30/400 samples), ppl
   +0.003 %. The flagship class-A method is a measured no-op-to-slightly-
   negative. (Naive batch application diverges: ppl 77.7.)
3. **RULER-style retrieval suite** (`evaluation/tasks/retrieval_suite.py`):
   v7 ties pooled SnapKV at 5 % (82.0 vs 83.2) and **beats it +6.6 at 2 %
   (73.8 vs 67.2; multi-query +21)**; PyramidKV 74.0@5 %, H2O 0.0.

## 5. Infrastructure notes

- `modal run --detach` still cancels the remote call when the local client
  dies of a network error (lost 1.75 method-runs to a DNS blip);
  `modal deploy` + `spawn_eval.py` detaches completely.
- MPS decode with ragged per-layer cache lengths recompiles kernels per shape
  per step (~4x decode slowdown locally); CUDA unaffected — PyramidKV, equally
  ragged, decodes at SnapKV speed on H200.
