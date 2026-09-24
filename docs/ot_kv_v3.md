# OT-KV v3 — what went wrong in v2, and what the measurements actually say

Three things are recorded here:

1. why the v2 rework lost accuracy (two independent bugs),
2. what the v1 code was *actually* computing (not what the README claims),
3. the derivation and the measurements behind `core/ot_kv_v3.py`, including the
   diagnostic that explains why merging on its own barely helps — and what
   unlocks it.

All numbers come from Qwen3-1.7B on wikitext-2 documents, `experiments/`.

---

## 1. What v1 and v2 were really computing

Both build a transport plan `T` with `sinkhorn_matrix_space` under uniform
marginals `mu_i = 1/n`, `nu_j = 1/m`, so `T` has total mass 1 and each anchor
column sums to `1/m`.

**v1** merges with `V_A <- V_A + T^T V_E`. Column `j` sums to `1/m`, so the
update adds a vector of norm ~`mean(||v||)/m`. Measured on a real cost matrix
(512x512, 128-dim keys): `||delta|| = 0.0036` against `||v_anchor|| = 11.29` —
**0.03 %**.

> v1's optimal-transport merge was numerically a no-op. v1 was top-k eviction
> with a per-head accumulated-attention score, i.e. per-head H2O.

The benchmark agrees, and so does the probe:

| | ppl | KL(dense‖compressed) |
|---|---:|---:|
| H2O | 19.69 | 0.0985 |
| OT-KV v1 | 19.53 | 0.0913 |

and offline, at budget 0.3, `otkv_v1` scores KL 0.06592 against plain eviction's
0.06598 — identical to three digits. Nothing in v1 validates the OT idea; it
also does not refute it, because the OT term never fired.

**v2** noticed the magnitude problem and rescaled by `n_evict`:
`V_A <- V_A + (T^T V_E) * n_evict`. `n_evict` is not a scale that appears
anywhere in the transport problem. With near-hard assignment (what a small
`epsilon` produces) column `j` then receives `n/m ~ 1` *full-magnitude* value
vectors on top of its own, displacing every value in the cache by about its own
norm. That is the regression:

| budget | evict | otkv_v1 | **otkv_v2** |
|---|---:|---:|---:|
| 0.30 | 0.0660 | 0.0659 | **0.5094** |
| 0.15 | 0.1104 | 0.1101 | **1.7268** |
| 0.075 | 0.1570 | 0.1568 | **5.1777** |

(KL(dense‖compressed), `experiments/probe.py`.)

### 1b. A second, independent bug: the Sinkhorn underflows

`sinkhorn_matrix_space` forms `K = exp(-C/epsilon)` in float32 on a cost matrix
normalised to mean 1. At the shipped default `epsilon = 0.01` that is
`exp(-100 C)`:

| epsilon | zero entries in `K` | recovered transport mass (should be 1.0) |
|---------|--------------------:|-----------------------------------------:|
| 0.01    | 27.3 %              | **0.38** |
| 0.05    |  1.3 %              | 0.87 |
| 0.2     |  0.0 %              | 0.98 |

At the default, the plan loses 62 % of its mass and violates both marginals — it
is not a transport plan. `sinkhorn_log_space` is present in the file, is correct
(mass 1.0000, marginal error < 1e-7), **and is never called by either version**.
The README describes the log-space algorithm; the code runs the matrix-space
one. Reproduce with `experiments/sinkhorn_diag.py`.

---

## 2. v3: deriving the merge instead of choosing it

For one layer and KV head, the middle region holds tokens `t = 1..L` with
post-RoPE keys `k_t`, values `v_t`, accumulated attention mass `s_t`. Keep `m`
anchors `A`, evict `n = L - m` tokens `E`; `S_A`, `S_E` are the corresponding
mass sums, `S = S_A + S_E`.

Find `m_ij >= 0` on `E x A` with

```
sum_j m_ij = s_i                        each evicted token ships its own mass
sum_i m_ij = c_j = s_j * S_E / S_A      anchors absorb in proportion to weight
```

and merge as a **convex combination**

```
v'_j = ( s_j v_j + sum_i m_ij v_i ) / ( s_j + c_j )
```

**Why that capacity.** Assume future queries attend like past ones,
`a_t ~ s_t / S`. After compression softmax renormalises over the retained set,
so anchor `j` gets weight `s_j / S_A`. The token set it now represents holds
true mass `(s_j + c_j)/S`, which with this `c_j` equals `s_j / S_A` — the same
number. Substituting:

```
o' = sum_j (s_j/S_A) (S_A/(s_j S)) ( s_j v_j + sum_i m_ij v_i )
   = (1/S) [ sum_j s_j v_j + sum_i s_i v_i ] = o
```

**Proposition.** Under `a_t ∝ s_t`, v3 reproduces the dense attention output
exactly, for *any* plan satisfying the two marginals. Eviction does not:
`o'_evict = (1/S_A) sum_j s_j v_j`, biased whenever the evicted values' centroid
differs from the retained one.

**So what is the plan for?** The marginals fix the query-*averaged* behaviour;
the plan fixes the first-order term — which anchor carries which value when the
actual query deviates from average. A query that attends hard to anchor `j`
should receive the values of the evicted tokens it *would* have attended to,
i.e. those with keys near `k_j`. Hence a key-space cost, and hence OT (whose
capacity constraint spreads the load) rather than greedy nearest-anchor
assignment.

**Frozen regions.** Sink and recent tokens are retained but never modified, and
softmax renormalises over them too. Redoing the argument with `S_R` = mass of
the whole retained cache gives `c_t = s_t S_E / S_R` for every retained token,
so only `S_E * S_A / S_R` may be transported into the anchors; the rest is
claimed by sink/recent whether we like it or not, and is dropped. Attention
sinks carry a large share, so `S_R/S_A` is close to 2 — ignoring this term
over-merges by roughly a factor of two.

---

## 3. The measurement that reframes the problem

With the maths fixed, merging still bought almost nothing: at budget 0.15,
KL 0.1107 for the merge versus 0.1104 for plain eviction. So we measured whether
the premise holds.

### 3a. The cache is redundant — in cosine

`experiments/redundancy.py`, prefill 3584, budget 0.15, averaged over layers:

| quantity | mean | p50 | frac > 0.7 | frac > 0.9 |
|---|---:|---:|---:|---:|
| evicted key → nearest anchor key | 0.850 | 0.866 | 0.879 | 0.416 |
| evicted value → its anchor's value | 0.561 | 0.536 | 0.219 | 0.047 |

Keys are highly redundant, values are not — exactly the configuration in which
merging should pay: the anchor is a plausible stand-in *and* the evicted token
carried different content that eviction discards.

### 3b. …but not in attention

Cosine is the wrong yardstick. Attention weights are exponentials of dot
products: with `||k|| ~ 11` and `d = 128`, a cosine gap of 0.1 is worth about
`||q|| ||k|| sqrt(2-2cos)/sqrt(d) ~ 5` nats, a 100x difference in weight. The
quantity that decides interchangeability is the log-ratio of the weights two
tokens receive from a real query. Measured against held-out queries
(`experiments/mergeability.py`, budget 0.15):

| anchor rule | median gap (nats) | frac gap<1 | mass gap<1 | attn mass kept | residual after historical ratio |
|---|---:|---:|---:|---:|---:|
| top-k (H2O/v1/v2) | 1.72 | 0.21 | 0.29 | 0.390 | 1.08 |
| k-medoids coverage | 1.34 | 0.33 | 0.42 | 0.229 | 0.82 |
| 50/50 hybrid | 1.47 | 0.28 | 0.36 | 0.354 | 0.90 |

Read this as: under heavy-hitter selection, an evicted token and its nearest
anchor typically receive attention weights differing by a factor of `e^1.72 ≈
5.6`. The merge applies the evicted value with coefficient `s_i/s_j`, the
historical ratio; that estimate removes about a third of the error (1.72 → 1.08
nats) but leaves a ~3x coefficient error. So the merge is a real but noisy
correction — which is precisely why the empirically best `merge_strength` is
~0.5 rather than the derived 1.0: for log-normal coefficient noise of `sigma ≈
1`, the shrinkage that minimises squared error is `e^{-sigma^2/2} ≈ 0.6`.

### 3c. The bottleneck is selection, not transport

The table above also says the gap is a property of the *anchor set*.
Heavy-hitter selection maximises retained mass and ignores whether the survivors
cover the key cloud, so it leaves poor stand-ins. Replacing it with a
mass-weighted k-medoids quantisation of the keys (`select_mode="kmeans:50"`:
half the budget to heavy hitters, half to coverage) drops the gap and — the
point — makes the merge start to pay:

`experiments/probe.py`, 3 docs, prefill 3584, KL(dense‖compressed):

| budget | evict (H2O-style) | coverage, no merge | coverage + merge | coverage + merge + gate |
|---|---:|---:|---:|---:|
| 0.30 | 0.06598 | 0.05392 (−18 %) | 0.05875 | **0.04928 (−25 %)** |
| 0.15 | 0.11038 | 0.10312 (−6.6 %) | **0.10215 (−7.5 %)** | 0.10581 |
| 0.075 | 0.15701 | 0.15512 (−1.2 %) | **0.15351 (−2.2 %)** | 0.15650 |

with top-k selection the same merge is worth 0 to −2 %, and at budget 0.3 it is
slightly *negative*. The conclusion is not "OT merging works" or "OT merging
doesn't" — it is:

> **Merging needs stand-ins, and heavy-hitter selection does not provide them.
> Once anchors are chosen to quantise the key distribution, transport of the
> evicted value mass onto them becomes worth having.**

### 3d. End-to-end, the same ordering holds

`experiments/bench.py`, Qwen3-1.7B, 3 wikitext docs, 4096 tokens (3850 prefill /
246 decoded), per-head, KL(dense‖compressed):

| budget | H2O | OT-KV v1 | v3 top-k | v3 coverage | v3 coverage+merge | v3 coverage+merge+gate |
|---|---:|---:|---:|---:|---:|---:|
| 0.30 | 0.0834 | 0.0872 | 0.0872 | 0.0649 | 0.0632 | **0.0621 (−25.6 %)** |
| 0.15 | 0.1709 | 0.1643 | 0.1626 | 0.1357 | **0.1334 (−21.9 %)** | 0.1337 |
| 0.075 | 0.2911 | 0.2710 | 0.2737 | 0.2402 | 0.2480 | **0.2384 (−18.1 %)** |

Perplexity moves the same way at tight budgets (21.97 vs H2O's 22.96 at 0.075,
20.21 vs 20.50 at 0.15). At budget 0.3 every compressed method scores *below*
dense perplexity (19.7-19.9 vs 21.2) — compression acts as a regulariser on
open-ended wikitext continuation, which is exactly why perplexity is a poor
metric for this work and KL against the dense model is the honest one.

### 3e. The catch: coverage trades average fidelity for peak fidelity

Needle-in-a-haystack, 10 needle depths, ~3200-token contexts
(`bench.py --tasks niah`):

| budget | dense | SnapKV | v3 top-k (+window) | v3 kmeans:85 | v3 kmeans:75 | v3 kmeans:50 |
|---|---:|---:|---:|---:|---:|---:|
| 0.075 | 1.00 | 0.70 | 0.70 | 0.60 | 0.60 | 0.50 |
| 0.05 | 1.00 | 0.40 | 0.40 | — | — | 0.10 |

Coverage selection spends budget on being *representative*, and a needle is the
opposite of representative: it is one sharp, rare token. So the same mechanism
that cuts average KL by a fifth costs 10-30 points of retrieval accuracy. Use
`select_mode="kmeans:50"` when distributional fidelity is the goal and
`select_mode="topk"` when the task is retrieval. Merging itself is
retrieval-neutral (v3 top-k with merge matches SnapKV exactly at 0.70/0.950).

### 3f. Scoring dominates retrieval, and the two regimes do not mix

The needle is invisible to prefill attention statistics because the question
comes after it. With all-query accumulation every method scores 0.00 on the
needle; restricting scoring to the last 32 queries takes the same methods to
0.83-1.00. Blending the two does not give the best of both — at
`score_blend=0.5` needle accuracy is already back to 0.00, because the window
signal has to dominate to survive. Meanwhile windowed scoring is much worse for
open-ended continuation (KL 0.119 vs 0.068 at budget 0.3).

There is no single setting: use `observation_window=32` for anything whose
prompt ends in a question (LongBench, NIAH) and leave it unset for perplexity.
This is the H2O-vs-SnapKV split, and it will dominate LongBench numbers far more
than anything in the transport step — worth fixing before reading too much into
LongBench comparisons.

### 3g. Per-head adaptive coverage — a strictly better selection rule

`kmeans:50` reserves half the budget for coverage in *every* head, which is what
costs retrieval: in a sharp head the rare token is exactly what gets displaced.
But heads differ. Define a head's concentration as the share of its attention
mass held by its top-`budget` tokens, and let the coverage bonus scale with
`1 - concentration`:

```
beta_h   = B * (1 - conc_h)
keep_h,t = beta_h * [t is a medoid] - rank_h(t) / budget
```

Ranks, not magnitudes: attention mass is heavy-tailed, so normalising by the max
leaves nearly every token at ~0 and any positive bonus elects every medoid — the
first version of this made the tight-budget result *worse than eviction*. In
rank units the knob is legible: a medoid may displace a kept token if it sits
within `beta_h * budget` ranks of the cut-off, and `B = 0` reproduces top-k
exactly.

Sharp heads then keep 99-100 % of their top-k selection while diffuse heads take
the coverage. `B = 4` dominates `kmeans:50` on **both** axes:

| budget | evict | kmeans:50 | **adaptive:4** |
|---|---:|---:|---:|
| 0.30 | 0.06598 | 0.05910 | **0.05222 (−20.9 %)** |
| 0.15 | 0.11038 | 0.10525 | **0.09113 (−17.5 %)** |
| 0.075 | 0.15701 | 0.15544 | **0.14684 (−6.5 %)** |

and needle retrieval at budget 0.075 goes 0.50 (kmeans:50) → 0.60
(adaptive:4) → 0.70 (top-k). `B` is the dial between distributional fidelity and
retrieval precision; `B = 2` and `B = 8` are both worse than 4 on KL.

**But end-to-end the ordering reverses.** Head to head, 4 docs, identical
conditions, KL(dense‖compressed):

| budget | H2O | kmeans:50 | adaptive:4 |
|---|---:|---:|---:|
| 0.30 | 0.09734 | **0.07877 (−19.1 %)** | 0.08616 (−11.5 %) |
| 0.15 | 0.18775 | **0.14134 (−24.7 %)** | 0.14691 (−21.8 %) |
| 0.075 | 0.31620 | **0.25475 (−19.4 %)** | 0.26902 (−14.9 %) |

The probe compresses once and stops; the benchmark also recompresses every 32
decode steps, and decode maintenance runs plain top-k. Coverage anchors carry
little accumulated attention of their own, so top-k evicts them first. The
plausible reading is that `adaptive` concentrates its coverage in a few diffuse
heads, and losing those anchors removes the benefit for those heads outright,
whereas `kmeans:50` spreads coverage thinly over every head so the erosion is
partial everywhere. `kmeans:50` is therefore the recommended setting, and
`adaptive:B` is kept because it is strictly better on retrieval (0.60 vs 0.50)
and better whenever compression happens once — a single long prefill followed by
a short answer, which is what most long-context benchmarks actually do.

Re-running the coverage selection during decode (`select_decode=True`) confirms
the erosion and partly repairs it — for `kmeans:50`, KL 0.13264 → 0.13048 at
budget 0.15 and 0.23782 → **0.22249** at 0.075, for ~0.7 s of extra prefill and
~1 ms/token. It does nothing for `adaptive` (0.13546 → 0.13743), consistent with
the reading above: adaptive's coverage is concentrated in a few heads and does
not come back once those anchors are gone.

**Final configuration and result.** `select_mode="kmeans:50"`,
`select_decode=True`, `merge_strength=0.5`, `gate_sigma=0.3`, `cost_alpha=0.8`,
against H2O on the same documents (the H2O column reproduced bit-identically
across three separate runs, so cross-run comparison here is safe):

| budget | H2O | **v3** | Δ |
|---|---:|---:|---:|
| 0.30 | 0.08336 | 0.06206 | **−25.5 %** |
| 0.15 | 0.17091 | 0.13048 | **−23.7 %** |
| 0.075 | 0.29114 | 0.22249 | **−23.6 %** |

with perplexity also lower at the budgets where compression actually bites
(22.21 vs 22.96 at 0.075, 20.18 vs 20.50 at 0.15).

This is also the clearest evidence that the transport step earns its keep: with
adaptive selection, merging beats not-merging end-to-end at every budget
(−3.3 %, −4.0 %, −2.2 %), partly because the absorbed mass written back into the
anchor score is what stops decode-time top-k from evicting the coverage
anchors.

### 3h. What the joint variant contributes

`core/ot_kv_joint.py` differs from `core/ot_kv.py` only in the cost: it blends
value-space cosine distance into the key-space cost (`alpha`) and adds
`lambda_pos * log1p(|pos_i - pos_j|)`. Both inherit the same two bugs, so
neither had ever been measured on a working transport plan. Ported into v3 and
measured:

| term | budget 0.30 | 0.15 | 0.075 |
|---|---:|---:|---:|
| key cost only (`cost_alpha=1.0`) | 0.05983 | 0.10687 | 0.15648 |
| + 20 % value distance (`0.8`) | **0.05910** | **0.10525** | **0.15536** |
| + 35 % value distance (`0.65`) | 0.05901 | 0.10512 | 0.15558 |
| + position penalty `0.05` | 0.05934 | 0.10670 | 0.15711 |
| + position penalty `0.2` | 0.05893 | 0.10625 | 0.15746 |

**The value term is worth keeping** — a consistent 0.7-1.5 % across budgets, and
it is free (one extra chunked matmul). It is not implied by the derivation,
which only needs key proximity; the plausible reading is that among anchors with
equally similar keys, the one whose value is already close is distorted least by
absorbing. **The position penalty is not worth keeping** — it helps slightly at
loose budgets and hurts at tight ones. `cost_alpha=0.8` is now the default;
`lambda_pos` stays at 0.

### 3i. What does *not* work: merging the keys

The most literal reading of "represent two tokens by a linear combination" is to
move the anchor's key to the centroid of what it absorbed, which would shrink
the log-attention gap of §3b by construction. It makes things clearly worse:

| budget | `key_merge=0` | `0.5` | `1.0` |
|---|---:|---:|---:|
| 0.30 | 0.05910 | 0.06110 | 0.06358 |
| 0.15 | 0.10525 | 0.11141 | 0.11876 |
| 0.075 | 0.15544 | 0.16276 | 0.17302 |

A centroid key has a smaller *mean* distance to the cluster but is no longer any
real token's key, so it stops retrieving the anchor's own content reliably —
and the attention weight of a merged entry should be the log-sum-exp of its
members' logits, which the mean under-represents. Keys must stay real; only
values are linearly combined.

That makes selection and transport one problem — choosing the anchor set *is*
the quantisation step of the transport, which is the natural OT formulation
(a Wasserstein quantisation of the mass-weighted key cloud, then a transport
plan onto it) rather than the two-stage "score, then patch" pipeline.

---

## 3j. LongBench on the real model — what survives

Llama-3.1-8B-Instruct, LongBench **qasper**, 150 samples, 10 % budget,
`max_length=8000`, H200 via Modal. Every method sees identical inputs, so
differences are paired (`experiments/paired_stats.py`); CI is a 10 000-resample
bootstrap on the paired difference.

| method | qa_F1 | Δ vs SnapKV | 95 % CI | sign test |
|---|---:|---:|---|---:|
| dense | 20.35 | +3.78 | [+2.45, +5.18] * | 0.007 |
| **otkv3** kmeans:50 + merge + gate | **17.10** | +0.53 | [−0.42, +1.49] | 0.85 |
| otkv3 top-k, no merge | 17.04 | +0.47 | [−0.10, +1.10] | 0.34 |
| otkv3 top-k + merge | 16.68 | +0.11 | [−0.45, +0.63] | 0.73 |
| SnapKV (window 32) | 16.57 | (ref) | | |
| H2O | 15.64 | −0.93 | [−2.47, +0.56] | 1.00 |
| **OT-KV v2** | **3.75** | **−12.82** | **[−15.07, −10.59] \*** | **<1e-4** |

Two things to take from this.

**The v2 regression is real, large and unambiguous.** 3.75 F1 against SnapKV's
16.57, losing 117 of the 129 samples where the two differ. Whatever else is
uncertain, this is not.

**The v3 improvement is directionally consistent but not statistically
significant on one task.** All three v3 variants beat both baselines, they close
13-14 % of the dense↔SnapKV gap, and the ordering matches the KL results — but
every CI includes zero. The paired standard deviations say what it would take:

| comparison | Δ | sd(Δ) | n needed for 80 % power |
|---|---:|---:|---:|
| v3 (kmeans:50) − SnapKV | +0.53 | 5.93 | ~990 |
| v3 (top-k, no merge) − SnapKV | +0.47 | 3.77 | ~500 |
| SnapKV − H2O | +0.93 | 9.58 | ~840 |

qasper has 200 samples in total, so **no single LongBench task can resolve a
half-point F1 difference** — note this also applies to the SnapKV-vs-H2O gap
that the literature reports. Settling it needs several tasks pooled (6-8 tasks
at 100 samples gets to ~600-800 paired samples). Until then the defensible
claims are the KL/perplexity results of §3d-3i and the v2 regression above; the
LongBench number should be reported as "matches or slightly exceeds SnapKV",
not as a win.

Reproduce:

```bash
modal run misc/modal/evaluate.py \
  --methods "baseline,h2o,snapkv;observation_window=32,otkv3;observation_window=32;select_mode=kmeans:50;select_decode=true;gate_sigma=0.3" \
  --limit 150 --max-length 8000 --compression-size 0.1 --longbench-tasks qasper
python experiments/paired_stats.py <dir with persample_*.json> --ref "snapkv;observation_window=32"
```

---

## 3k. Does v3 actually beat H2O? A paired answer

The tables above were 3-4 documents with no error bars. Re-run over **12
documents** with per-document logging, so the comparison can be paired on the
document (positions inside a document are far too autocorrelated to be treated
as independent samples). `experiments/paired_kl.py`, 20 000-resample bootstrap:

**KL(dense‖compressed), reference = H2O**

| budget | method | Δ vs H2O | 95 % CI | docs won | sign p |
|---|---|---:|---|---:|---:|
| 0.075 | v3 (selection + merge) | **−20.3 %** | [−0.111, −0.046] * | **12/12** | 0.0005 |
| 0.075 | v3 (selection only) | −18.8 % | [−0.101, −0.043] * | 12/12 | 0.0005 |
| 0.075 | SnapKV | −4.7 % | [−0.079, +0.044] | 6/12 | 1.00 |
| 0.15 | v3 (selection + merge) | **−16.7 %** | [−0.052, −0.019] * | 9/12 | 0.146 |
| 0.15 | v3 (selection only) | −15.6 % | [−0.050, −0.018] * | 11/12 | 0.0063 |
| 0.15 | SnapKV | +20.5 % | [+0.007, +0.084] * | 4/12 | 0.388 |

**Perplexity, same runs**: −1.4 % at 0.075 (CI [−0.071, −0.010], 8/12 docs) and
−0.7 % at 0.15 (CI includes zero). Much weaker than the KL gap — v3 moves the
model's whole output distribution closer to dense without changing next-token
likelihood much.

**The increment from the transport step itself**, v3-with-merge against
v3-selection-only on the same 12 documents:

| budget | Δ | 95 % CI | docs won | sign p |
|---|---:|---|---:|---:|
| 0.075 | −1.8 % | [−0.0131, +0.0009] | 8/12 | 0.39 |
| 0.15 | −1.3 % | [−0.0059, +0.0009] | 6/12 | 1.00 |

So, stated precisely:

> **v3 beats H2O, and the reason is the anchor selection, not the optimal
> transport.** The selection change alone is significant at both budgets
> (11-12 of 12 documents). The merge adds a further 1-2 % that is consistently
> in the right direction and never once reaches significance — on KL, on
> perplexity, on needle retrieval, or on LongBench.

A paper claiming OT-based *merging* is therefore not yet supported by this
evidence. A paper claiming that KV compression should choose survivors to
quantise the key distribution rather than to maximise retained attention mass
*is* supported, and §3b explains why: heavy-hitter survivors are not usable
stand-ins, so there is nothing for any merging rule to transport into.

Caveats on all of the above: one model (Qwen3-1.7B), one corpus (wikitext-2),
12 documents, and KL is not a metric any reviewer will accept on its own.

---

## 4. Making it fast

A dense `n x m` Sinkhorn is memory-bound: at `n = m = 4000`, 50 iterations move
~6 GB per head per layer, ~1.4 TB for a 28-layer 8-KV-head model. Unusable.

v3 restricts transport to each evicted token's `top_r` most similar anchors and
runs log-domain Sinkhorn over that sparse support with `scatter_reduce`:

- candidates: one chunked similarity matmul, never materialised in full
- per iteration `O(n r)` instead of `O(n m)`, `r = 16`
- row update: dense logsumexp over `r`; column update: scatter-logsumexp

On the sparse support the column marginal converges to < 5e-9 after one
iteration and the row marginal to 1.8e-3 after five, so `sinkhorn_iters=5`
suffices where the dense version asked for 50.

Compression cost per layer, M1 Max, 8 KV heads (`experiments/timing.py`):

| | L=3300 → 660 | L=7000 → 1400 |
|---|---:|---:|
| eviction (top-k) | 2.2 ms | 0.8 ms |
| v3 top-k + merge | 65 ms | 142 ms |
| v3 coverage, no merge | 21 ms | 76 ms |
| v3 coverage + merge | 88 ms | 216 ms |
| v2 (dense Sinkhorn, 50 iters) | 43 ms | 139 ms |

so v3 with coverage and merging costs ~6 s of one-off prefill-end compression on
a 28-layer model at 7k context — on a laptop GPU. The dominant term is the
`n x m` candidate matmul (now in fp16, which halved it) and the k-means
`cdist`; both are matmul-shaped and far cheaper on a datacenter GPU.

Per *decode step* v3 is no slower than H2O — it recompresses every
`compress_interval` steps where H2O prunes every step. The elevated
`dec ms/tok` in the benchmark tables is the one-off prefill-end compression
amortised over 246 decoded tokens (6 s / 246 ≈ 24 ms), not per-step overhead:
v3-coverage-without-merge shows 67.6 ms/tok against H2O's 65.5.

Two settings exist because both merging and coverage selection are one-shot
summarisations of the prefill: `merge_decode=False` and `select_decode=False`
(both default) keep decode-time maintenance on plain top-k eviction. Re-merging
every `compress_interval` steps re-averages already-averaged values and the blur
compounds — measured as a consistent ~5 % KL regression end-to-end.

---

## 5. Knobs

| kwarg | default | meaning |
|---|---|---|
| `merge` | `True` | `False` = anchors only; the control for every merge claim |
| `select_mode` | `"topk"` | `"adaptive:B"` (recommended, B≈4) scales the coverage bonus per head by `1 - concentration`; `"kmeans:P"` gives P % of budget to heavy hitters and the rest to coverage in every head |
| `merge_strength` | 0.5 | shrinkage on the transported mass; 0.5 matches the measured coefficient noise |
| `gate_sigma` | 0.0 | soft reliability gate `exp(-cost/sigma)`; helps at loose budgets, hurts at tight ones |
| `top_r` | 16 | transport candidates per evicted token |
| `epsilon` | 0.05 | entropic regularisation (cost lives in `[0, 2]`) |
| `sinkhorn_iters` | 5 | sparse Sinkhorn iterations |
| `capacity_beta` | 1.0 | 1.0 = mass-consistent capacity, 0.0 = uniform (what v1/v2 used) |
| `frozen_correction` | `True` | account for sink/recent absorbing their share |
| `merge_decode` | `False` | re-merge during decode too; measured to compound blur, keep off |
| `select_decode` | `False` | re-run coverage selection during decode; **turn on with `kmeans:P`** (−6 % KL at tight budgets) |
| `key_merge` | 0.0 | pull anchor keys toward what they absorbed — measured to be harmful, keep at 0 |
| `cost_alpha` | 0.8 | key/value blend in the transport cost (1.0 = keys only); from `ot_kv_joint` |
| `lambda_pos` | 0.0 | positional penalty in the cost; from `ot_kv_joint`, measured to be not worth it |
| `observation_window` | `None` | score with the last W queries (SnapKV-style) instead of all |
| `score_writeback` | `"absorbed"` | write absorbed mass into the anchor score |

### Recommended settings

```bash
# distributional fidelity (perplexity / KL / open-ended generation)
python main.py --methods otkv3 --compression_size 0.15 \
  --v3_select_mode kmeans:50 --v3_merge_strength 0.5 --v3_gate_sigma 0.3 --v3_select_decode true

# retrieval / QA, i.e. anything whose prompt ends in a question
python main.py --methods otkv3 --compression_size 0.15 \
  --v3_select_mode topk --v3_merge_strength 0.5 --observation_window 32
```

---

## 6. Tooling

| script | what it does | cost |
|---|---|---|
| `experiments/bench.py` | end-to-end: perplexity, KL vs dense, cache size, prefill/decode timing, needle-in-a-haystack | ~30 s per (method, doc) |
| `experiments/probe.py` | compression quality only: compress a dense prefill offline, score a held-out block in one forward | ~1 s per config |
| `experiments/validate_probe.py` | proves the probe's single-forward path matches a plain forward (KL 2e-5, i.e. fp16 noise) | seconds |
| `experiments/redundancy.py` | key/value redundancy of a real cache | ~1 min |
| `experiments/mergeability.py` | the log-attention-gap diagnostic of §3b | ~2 min |
| `experiments/sinkhorn_diag.py` | the underflow table of §1b | instant |
| `experiments/timing.py` | per-layer compression cost of each rule | seconds |

The probe is the one to iterate with; `bench.py` is the arbiter. They can
disagree, and when they do the difference is decode-time recompression — which
is how `merge_decode` was found.
