# OT-KV experiment record

Every number here is reproducible from `experiments/`. Two machines are used:

- **local** — Qwen3-1.7B, fp16, MPS. Long-document wikitext-2, measuring
  KL(dense‖compressed) over held-out decode positions and perplexity. Cheap
  enough to iterate on, and far more sensitive than any end-task metric.
- **Modal H200** — Llama-3.1-8B-Instruct, LongBench. The metric a reviewer will
  actually look at, and much noisier per sample.

Statistical policy: every method is run on identical inputs, so all comparisons
are **paired**, with a 20 000-resample bootstrap CI on the paired difference and
a two-sided sign test. Unpaired means over a few hundred LongBench samples
cannot separate methods a point apart and are not reported as differences.
The unit of analysis is the document (local) or the sample (LongBench); decode
positions inside one document are far too autocorrelated to count as
independent.

---

## 0. Multi-head compression is correct (`experiments/test_per_head.py`)

44 assertions, all passing, over H2O / SnapKV / PyramidKV / EchoKV / OT-KV v2 /
OT-KV v3:

- each KV head genuinely selects its own token subset, and keeps its own
  high-score band while the sink and recent windows survive verbatim in every
  head;
- surviving K and V rows stay paired, positions stay in causal order, and rows
  are traceable to the tokens the head actually selected (tokens are tagged with
  their position for this);
- `score[h, i]` still describes cache row `i` of head `h` after compression, and
  stays aligned across 20 further decode-time compressions;
- GQA query heads fold onto the KV head `repeat_kv` pairs them with;
- `per_head=True` and `per_head=False` keep the **same** total cache length, so
  the budget means the same thing in both modes.

One fairness bug found and fixed: `core/ot_kv.py` and `core/ot_kv_joint.py`
silently overrode `per_head=False` to `True`, while every baseline honoured the
flag. Any run with `--per_head false` was therefore comparing global-mode
baselines against a per-head OT-KV — an unfair table, tilted towards OT-KV.
v3 now implements global mode properly; v2 and joint emit a `RuntimeWarning`
instead of silently switching. (All Modal runs in this record used
`per_head=True` throughout, so none of them were affected.)

## 1. Per-head is a large win — but not for every method

Local, 6 documents, paired, `runs/perhead_{true,false}.json`. Negative = per-head
better.

| method @ budget | per-head | global | Δ | Δ % | docs won |
|---|---:|---:|---:|---:|---:|
| H2O @ 0.15 | 0.21570 | 0.34876 | −0.13306 * | **−38.2 %** | 6/6 |
| OT-KV v3 @ 0.15 | 0.18197 | 0.30976 | −0.12779 * | **−41.3 %** | 6/6 |
| SnapKV @ 0.15 | 0.25511 | 0.22133 | +0.03379 * | **+15.3 %** | 2/6 |
| H2O @ 0.075 | 0.36028 | 0.52305 | −0.16277 * | −31.1 % | 6/6 |
| OT-KV v3 @ 0.075 | 0.28437 | 0.41895 | −0.13458 * | −32.1 % | 6/6 |
| SnapKV @ 0.075 | 0.33737 | 0.31193 | +0.02544 | +8.2 % | 2/6 |

(* = bootstrap 95 % CI excludes zero.)

Per-head compression pays for **H2O and OT-KV and costs SnapKV accuracy**. The
mechanism is evidence volume, not head specificity: with GQA the 16 query heads
fold onto 8 KV heads, so a per-head score is estimated from a quarter of the
attention rows. H2O accumulates over every query in the prefill and can afford
that; SnapKV sees only its 32-query observation window, and splitting those 32
rows eight ways is too noisy to beat a pooled ranking.

**Consequence for fair comparison:** "run everything per-head" is *not*
automatically the fair protocol — it handicaps short-window scorers. Each method
should be reported in its better mode, or both modes should be shown.


---

## 2. Two baseline bugs found while building the suite

Both would have biased the paper **in OT-KV's favour**, so both are fixed before
any headline number is quoted.

### 2a. PyramidKV's budget collapses during decode

`PyramidKVCache.get_middle_budget` recomputed a per-layer budget from
`total_tokens`. During decode `total_tokens` is the length of the *already
compressed* cache, so every pass shrank the budget again. Measured on a 4-layer
toy cache with `mode="prefill"`, where the budget must stay fixed:

```
after prefill      : [90, 70, 50, 30]
after 30 decode    : [11, 11, 11, 11]      <- runaway
```

That is why PyramidKV scored 14.60 on the LongBench table below — far under
StreamingLLM, and nothing like its published standing. Fixed by scaling the
*parent's* (correctly fixed) middle budget by the depth factor instead of
recomputing: `[90,70,50,30] -> [86,69,52,36]` after 30 steps.

### 2b. OT-KV held more cache than the methods it was compared with

OT-KV compresses every `compress_interval` (32) decode steps rather than every
step, so between compressions its cache grew past the budget: peak
`budget + interval`, roughly +15 % at a 5 % budget. The baselines prune every
step and sit exactly at budget.

Fixed by reserving one interval's worth of slots at prefill end (capped at a
quarter of the middle budget, or a small budget would be wiped outright). Now:

| | budget | peak | mean |
|---|---:|---:|---:|
| H2O | 285 | 286 | 286.0 |
| OT-KV v3 | 285 | **285** | **267.7** |

OT-KV now holds *less* KV than its baselines on average, so the comparison errs
against it. `experiments/bench.py` reports mean/peak cache so this stays
auditable.

---

### 2c. OT-KV never received the `per_head` flag

`main.py`'s `otkv3` config block listed every other kwarg but not `per_head`, so
`OTKVv3Cache` fell back to the class default while every baseline got the flag
from the CLI. This was invisible while v3 forced `per_head=True` internally
(§0); the moment v3 started honouring the flag, every Modal run silently
switched OT-KV to global mode and crashed on the dual-score layout. Fixed by
passing the flag, and by making split scoring fall back to a single stream in
global mode where there is no head axis to split.

### 2d. Boolean-mask indexing assumed every head evicts the same count

`otkv_v3_compress` built the evicted set with `key_states[evict_mask]`, which
flattens across heads and only reshapes correctly if every head kept exactly
`budget` *distinct* positions. A selection rule that returns a repeated index —
ties, degenerate scores, a clustering that maps two centroids to one medoid —
produces a different count per head and the reshape fails with
`shape '[1, 8, 3728, 128]' is invalid for input of size 3924224` (note 3924224
is not divisible by 8·128). Replaced with a stable-argsort complement that is
exact per head, and covered by a tie-heavy regression case.

## 3. LongBench, 8 tasks (Llama-3.1-8B, 5 % budget)

50 samples per task, 400 pooled paired samples, `max_length=4096`, all methods
per-head. Reference = SnapKV. **This table predates the two fixes above** — it
is kept because it is what the fixes were found from; the corrected run is §4.

| method | mean | Δ vs SnapKV | 95 % CI | win/loss | p |
|---|---:|---:|---|---:|---:|
| dense | 27.62 | +4.75 | [+3.43, +6.16] * | 203/84 | <1e-4 |
| **OT-KV v3** (top-k) | **23.38** | +0.51 | [−0.54, +1.58] | 132/135 | 0.90 |
| OT-KV v3 (kmeans:50) | 23.16 | +0.29 | [−0.78, +1.40] | 149/131 | 0.31 |
| SnapKV (w=32) | 22.87 | (ref) | | | |
| H2O | 22.11 | −0.76 | [−2.27, +0.77] | 159/143 | 0.39 |
| StreamingLLM | 20.39 | −2.48 | [−4.04, −0.92] * | 118/183 | 2e-4 |
| PyramidKV (buggy) | 14.60 | −8.27 | [−10.04, −6.54] * | 60/258 | <1e-4 |

Per task, OT-KV v3 beats SnapKV on qasper, hotpotqa, samsum and triviaqa, and
loses on 2wikimqa, multifieldqa_en, gov_report and multi_news.

**OT-KV v3 is the best compressed method by point estimate, and the margin is
not statistically resolvable.** 400 paired samples, win/loss 132/135 — v3 wins
fewer samples but by larger margins. Honest reading: *matches or slightly
exceeds SnapKV on LongBench*, while being significantly better on the local
KL/perplexity measurements.

---

## 4. The observation window is a hard trade-off

Local, 8 documents, budget 0.075, per-head. `runs/window_sweep.json`.

| scoring | KL(dense‖c) | NIAH |
|---|---:|---:|
| H2O (all queries, no window) | 0.34859 | 0.00 |
| SnapKV (window 32) | 0.35490 | **0.70** |
| OT-KV v3, window 32 | 0.39353 | 0.30 |
| OT-KV v3, window 128 | 0.30356 | 0.00 |
| OT-KV v3, window 512 | **0.24794** | 0.00 |
| OT-KV v3, all queries | 0.29031 | 0.00 |

A short window is the *only* thing that finds the needle — nothing earlier in
the prompt marks the answer as important, so all-query accumulation cannot know
it matters. A long window is the only thing that models open-ended continuation
well. The two objectives want opposite scorers, and no single window serves
both. This also explains §1: per-head splitting hurts SnapKV because a 32-row
window divided over 8 KV heads is too little evidence, while all-query
accumulation has plenty.

`score_blend` mixes the two streams on a per-query scale (`0` = all-query,
`1` = window-only). **Blending does not resolve the trade-off** — at `0.5` the
needle is already gone (NIAH 0.00, partial 0.50):

| config | KL | NIAH |
|---|---:|---:|
| blend 1.0, top-k | 0.43536 | 0.60 |
| blend 0.5, top-k | 0.31959 | 0.00 |
| blend 0.5, kmeans:50 | 0.29949 | 0.00 |
| blend 0.25, kmeans:50 | **0.27955** | 0.00 |

What *does* help is keeping the streams **separate and giving them different
jobs**: the window score chooses which tokens to keep, the all-query score
supplies the transport capacity. Feeding the noisy window score to the merge is
what made OT-KV worse than SnapKV on KL under window scoring. With
`split_scores=true` retrieval returns to the SnapKV level at no KL cost:

| config | KL | NIAH | partial |
|---|---:|---:|---:|
| SnapKV (w=32) | 0.35490 | 0.70 | 0.950 |
| v3 window-for-both, top-k | 0.43526 | 0.60 | 0.867 |
| **v3 split (window selects, all-query merges)** | 0.43210 | **0.70** | 0.917 |
| v3 all-query + kmeans:50 | **0.29112** | 0.00 | 0.150 |

There is no single configuration that is best at both. OT-KV has two operating
points, and which one to use is decided by the task, not by tuning:
`split_scores=true, select_mode=topk` for question-answering, and all-query
scoring with `kmeans:50` coverage for open-ended continuation.

---

## 5. Compression-ratio sweep (local, 8 documents, paired)

The all-query/coverage operating point against the matched baselines, over five
budgets. `runs/budget_sweep.json`, `experiments/paired_kl.py`.

| budget | H2O | SnapKV | **OT-KV v3** | Δ vs H2O | CI excl. 0 | docs won |
|---|---:|---:|---:|---:|:--:|---:|
| 0.30 | 0.11784 | 0.16729 | **0.08685** | −26.3 % | yes | 8/8 |
| 0.15 | 0.21255 | 0.27295 | **0.17440** | −17.9 % | yes | 7/8 |
| 0.075 | 0.34859 | 0.35490 | **0.28983** | −16.9 % | yes | 8/8 |
| 0.05 | 0.45845 | 0.40730 | **0.37146** | −19.0 % | yes | 7/8 |
| 0.025 | 0.80802 | 0.55839 | **0.51422** | −36.4 % | yes | 8/8 |

Perplexity follows: 17.87 / 18.65 / 20.29 / 21.58 / 24.97 for v3 against H2O's
18.15 / 18.97 / 20.91 / 23.39 / 32.47. Against SnapKV the gap is significant at
budgets 0.075, 0.15 and 0.30 (−18 %, −36 %, −48 %) and not at 0.05 / 0.025.

OT-KV wins these while holding **less** cache (mean 1139 vs 1156 at budget 0.30,
82 vs 97 at 0.025) — see §2b.

Cost: prefill is unchanged (3.4-3.6 s either way; the OT pass is amortised into
one compression), decode is 1.4-2.7x slower per token than H2O on MPS
(85-159 ms vs 56-64 ms), the ratio growing as the budget shrinks because the
per-compression fixed cost stops being amortised.


---

## 6. LongBench with every fix applied — OT-KV does **not** win here

Llama-3.1-8B, 8 tasks x 50 samples, 5 % budget, `max_length=4096`, per-head,
`compress_interval=4` so OT-KV's mean cache is 98-99 % of budget rather than
92 % (§2b). Baselines are the corrected ones (§2a).

| method | mean | vs PyramidKV | 95 % CI |
|---|---:|---:|---|
| dense | 27.62 | +4.02 | [+2.84, +5.27] * |
| **PyramidKV** (fixed) | **23.60** | (ref) | |
| SnapKV, global mode | 23.46 | −0.14 | [−1.24, +0.96] |
| OT-KV v3, kmeans:50 + pyramid | 22.88 | −0.72 | [−1.82, +0.29] |
| OT-KV v3, top-k + pyramid | 22.93 | | |
| OT-KV v3, top-k + split + pyramid | 22.81 | | |
| OT-KV v3, top-k | 22.56 | | |
| SnapKV, per-head | 22.87 | −0.73 | [−1.72, +0.15] |
| H2O | 22.11 | −1.48 | [−2.82, −0.17] * |

**The earlier +0.51 "win" over SnapKV was a budget-accounting artefact.** OT-KV
scored 23.38 while holding peak `budget + compress_interval`; once the peak is
capped at budget the same method lands at 22.6-22.9, i.e. the ~0.6 point the fix
costs is exactly the size of the margin being claimed. Every OT-KV configuration
tried — top-k or coverage selection, with or without split scoring, with or
without the pyramid layer budget — sits below the fixed PyramidKV and below
global-mode SnapKV, and none of the gaps is individually significant.

Honest statement of the LongBench result:

> On LongBench at a 5 % budget with matched cache and corrected baselines,
> OT-KV v3 is **competitive with but not better than** SnapKV and PyramidKV.

This does not contradict §5: on open-ended continuation OT-KV is significantly
better at every budget, while holding less cache. The two results together say
the method's advantage is real but regime-specific — it comes from choosing
survivors that quantise the key distribution, which pays when future queries
resemble past ones, and not when a question at the end of the prompt decides
what matters.

### What would have to change for a LongBench win

The diagnosis in `docs/ot_kv_v3.md` §3b is that a heavy-hitter anchor is a poor
stand-in (median log-attention gap 1.72 nats, a 5.6x weight error), so there is
little for any merge to recover. Coverage selection shrinks that gap but costs
retrieval, and on question-answering retrieval is what the score is made of.
The open direction is a selection rule that covers the key cloud *within* the
set the question is likely to query, rather than globally — i.e. conditioning
coverage on the observation window instead of trading against it.


---

## 7. v5 and v6 — following the theory

`docs/ot_kv_theory.md` derives that mergeability is decided by
`σ² = Δᵀ Σ_q Δ / d`, and that cosine distance in key space — the cost every
version from v1 to v4 minimises — correlates with it at only **0.273**.

### 7a. v5: the theory's prescription for merging, and why it fails

`core/ot_kv_v5.py` replaces the cosine cost with the exact quantity, estimated
from attention weights alone (`log A[w,i] − log A[w,j] = q_w·Δ/√d`, the softmax
normaliser cancelling), and replaces the historical mass ratio with the measured
log-normal mean `exp(μ_ij + σ²_ij/2)`. Budget 0.075, 4 documents:

| | KL(dense‖c) |
|---|---:|
| no merge | 0.4664 |
| v4 merge (key space) | 0.4612 |
| **v5 merge strength 0.2** | 0.4719 |
| **v5 merge strength 0.5** | 0.5124 |
| **v5 merge strength 1.0** | 0.5761 |

Monotonically worse. That negative result is what identified the missing term in
the derivation — the coefficient errors are driven by the *shared* query and so
do not average out — which yields the break-even criterion and the oracle
measurement in §6 of the theory note. **The merging direction is closed**: under
0.4 % of evicted tokens have a usable stand-in anywhere in the cache, at any
budget from 5 % to 95 %.

### 7b. v6: transport as anchor placement

`core/ot_kv_v6.py` keeps the space and drops the merging. Lemma 1 says the
eviction error is `ρ(q)·[v̄_A(q) − v̄_E(q)]`, a *product*: heavy-hitter selection
minimises the first factor, coverage the second. v6 minimises a surrogate for
the product — the attention-mass-weighted quantisation error
`Σ_t s_t · d(z_t, A)²` in logit-response space — which is a discrete
Wasserstein-2 quantisation problem with attention mass as the measure.

Budget 0.075, 6 documents, per-head, window scoring:

| method | KL(dense‖c) | NIAH |
|---|---:|---:|
| SnapKV | 0.3374 | 0.70 |
| OT-KV v4 (key space + merge) | 0.4609 | 0.90 |
| **OT-KV v6 (logit space, no merge)** | **0.4270** | **0.90** |
| v6 ablation: key-space quantisation | 0.4514 | 0.90 |
| v6 ablation: uniform weights (pure coverage) | **0.4075** | **0.10** |

Both claims hold:

- **the space matters** — quantising in logit-response space beats quantising in
  key space by 5.4 % at identical retrieval, which is the theory's prediction
  made operational;
- **mass weighting is the ρ-vs-coverage dial** — removing it gives the best KL of
  any configuration and destroys retrieval (0.90 → 0.10). This is the principled
  replacement for the ad-hoc `kmeans:P` split of v3/v4.

And v6 beats v4 by 7.4 % KL at equal retrieval **while doing no merging at all**,
so the cache stays a subset of real tokens.

### 7b-2. The v6 gain is significant, and the ablation localises it

Paired over documents, budget 0.075, reference = v4 (key space + merge):

| method | KL | Δ vs v4 | 95 % CI | docs won | sign p |
|---|---:|---:|---|---:|---:|
| **v6 (logit space)** | 0.42698 | **−7.3 %** | [−0.060, −0.011] * | **6/6** | 0.031 |
| v6, heavy_frac 0.25 | 0.43589 | −5.4 % | [−0.047, −0.008] * | 5/6 | 0.219 |
| v6, **key-space** ablation | 0.45137 | −2.1 % | [−0.026, +0.003] | 3/6 | 1.000 |
| v6, **uniform-weight** ablation | 0.40750 | −11.6 % | [−0.233, +0.064] | 3/6 | 1.000 |

The full-strength result wins every document and its CI excludes zero; the
key-space ablation does not. **The space is what makes it work** — which is the
theory's prediction, tested and confirmed. (The uniform-weight row has the best
mean but is unusable: NIAH 0.10, and its CI is wide because it is erratic
across documents.)

This is the first mechanism in the project that is derived from the OT
formulation rather than borrowed from a baseline, and that reaches significance
on its own.

### 7c. Where v6 does not (yet) win

In continuation mode (all-query scoring, budget 0.15 / 0.075, 8 documents) v6
scores KL 0.1927 / unstable against v4-continuation's 0.1811 / 0.2814. The
diagnosis follows from the construction: the logit embedding is only as
informative as the queries used to build it, and a *recent* 32-query window
describes the wrong geometry when the task is open-ended continuation.
`logit_query_mode="spread"` samples the window uniformly across the prefill
instead; a pure mass-weighted quantiser also collapsed on one document
(KL 1.90, ppl 109) in a degenerate Lloyd configuration, so `heavy_frac` now
defaults to 0.25 as a floor. Both fixes help and neither closes the gap:

| budget | v4-continuation | v6 spread | v6 recent | Δ (spread vs v4) |
|---:|---:|---:|---:|---|
| 0.15 | **0.18139** | 0.18646 | 0.18955 | +2.8 % [−0.008, +0.015] |
| 0.075 | **0.28093** | 0.29408 | 0.29819 | +4.7 % [+0.004, +0.022] * |

So the picture is regime-split and the mechanism explains it: the logit
embedding is informative exactly when the queries it is built from are the ones
that matter. A question at the tail makes the recent window the right sample,
and v6 wins there by 7.3 %; open-ended continuation has no such privileged
queries, and v4's key-space selection is 2.8-4.7 % better.


---

## 8. v6 on LongBench, and a metric that disagrees with itself

### 8a. v6 does not transfer

8 tasks x 50 samples, Llama-3.1-8B, 5 % budget:

| method | LongBench avg |
|---|---:|
| OT-KV v4 (auto + pyramid) | 23.98 |
| PyramidKV (global) | 23.67 |
| PyramidKV (per-head) | 23.60 |
| **OT-KV v6 + pyramid** | **23.49** |
| SnapKV (global) | 23.46 |
| OT-KV v6, key-space ablation | 23.36 |
| OT-KV v6, heavy_frac 0 | 23.01 |
| OT-KV v6 (default) | 22.81 |
| SnapKV (per-head) | 22.87 |
| H2O | 22.11 |

v6 does not beat the baselines, and — more tellingly — **the ablation that
carries its whole claim shows almost nothing here**: logit-space quantisation
scores 23.49 against key-space 23.36, a 0.13 gap, where the same ablation on
local KL was −7.3 % and won 6 of 6 documents with the CI excluding zero.

### 8b. The two metrics rank methods in opposite directions

Nine methods measured on both, same budget:

| method | local KL | LongBench | KL rank | LB rank |
|---|---:|---:|---:|---:|
| OT-KV v4 continuation | **0.2807** | 22.81 | 1 | 3 |
| OT-KV v4 auto | 0.2944 | 23.20 | 2 | 5 |
| H2O | 0.3486 | 22.11 | 3 | 2 |
| SnapKV | 0.3549 | 22.87 | 4 | 4 |
| PyramidKV | 0.3583 | **23.60** | 5 | 9 |
| StreamingLLM | 0.3885 | 20.39 | 6 | 1 |
| OT-KV v6 + pyramid | 0.4359 | 23.49 | 7 | 8 |
| OT-KV v6 key-space | 0.4514 | 23.36 | 8 | 7 |
| OT-KV v4 qa | 0.4777 | 23.26 | 9 | 6 |

```
Spearman( KL rank , LongBench rank ) = +0.450
```

Agreement would give −1. The best method by KL (v4 continuation) is 3rd of 9 on
LongBench; the best on LongBench (PyramidKV) is 5th of 9 by KL. **The two
metrics are mildly anti-correlated.**

This explains the whole arc of the project. Coverage selection, the split score
streams, the logit-space geometry — each was validated on local KL, each
reached significance there, and none transferred. It also has a consequence
beyond this project: perplexity-style metrics are routinely reported alongside
LongBench in the KV-compression literature, and on this evidence they are not a
weaker version of the same signal, they are a different and partly opposing one.

**Methodological rule adopted from here on:** a mechanism is only credible if it
is validated on the end task, with configurations chosen on a held-out split.
Local KL is useful for debugging and for measuring *quantities* (sigma, centroid
gaps, transport mass), not for ranking methods.
