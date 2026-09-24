# OT-KV v7 — final results

Model: Llama-3.1-8B-Instruct, 5 % KV budget, `max_length=8000`, greedy, 64 new
tokens. All comparisons are paired (20 000-resample bootstrap CI + two-sided
sign test). Dev/test discipline: every configuration choice was made on
LongBench samples 0–50 per task ("dev") or on synthetic probes (NIAH,
`experiments/detector_probe.py`); samples 50–150 ("test", 800 paired samples)
were run exactly once, after the method was frozen. Raw per-sample scores:
`ot_kv_data` volume, `runs/test8k` / `runs/dev8k` / `runs/niah8k` /
`runs/ppl8k` / `runs/profile8k`.

## 1. The method

**OT-KV v7** (`core/ot_kv_v7.py`, defaults): keep sink + recent verbatim;
score middle tokens by SnapKV-style last-32-query attention, evidence pooled
across KV heads, then

1. token-axis max-pool, kernel 7 (keep the semantic unit a spike belongs to);
2. multiply by ‖v‖ (Lemma 1: eviction error is attention × value displacement);
3. water-fill the total slot budget across layers by normalised marginal gains
   (floor 0.25×, cap 2.5× uniform) — the transportation LP, solved greedily;
4. keep top-k per layer; per-head bookkeeping; compress every 4 decode steps
   with the peak-capping reserve. No value merging (closed by
   `docs/ot_kv_theory.md`); no regime detector (closed for Llama by
   `experiments/detector_probe.py`).

## 2. LongBench test split — 800 paired samples, run once

| method | avg | Δ vs pooled SnapKV | 95 % CI |
|---|---:|---:|---|
| Dense (no compression) | 28.77 | +2.77* | [+1.98, +3.57] |
| SnapKV (global, unpooled) | 26.13 | +0.13 | [−0.44, +0.72] |
| SnapKV (global, pooled k=7) | 26.00 | (ref) | |
| v7 −waterfill | 25.96 | −0.04 | [−0.54, +0.47] |
| v7 −pooling | 25.90 | −0.10 | [−0.67, +0.48] |
| PyramidKV (global) | 25.89 | −0.10 | [−0.67, +0.46] |
| **OT-KV v7** | **25.71** | −0.29 | [−0.79, +0.19] |
| v7 −vnorm | 25.36 | −0.64* | [−1.15, −0.14] |
| v7 with auto-regime (τ=0.73) | 24.97 | −1.03* | [−1.93, −0.17] |
| OT-KV v4 (auto)+pyramid | 24.51 | −1.48* | [−2.45, −0.55] |
| H2O (per-head) | 23.38 | −2.62* | [−3.62, −1.66] |
| StreamingLLM | 22.64 | −3.35* | [−4.36, −2.38] |

Reading this honestly:

- **v7 is statistically indistinguishable from SnapKV and PyramidKV** (every
  pairwise CI in the top block includes zero) and significantly better than
  v4-auto (+1.2), H2O (+2.3), StreamingLLM (+3.1), and its own auto-regime
  variant (+0.7).
- The dev-split margin (+0.62, CI [−0.16, +1.46]) did not survive the test
  split (−0.29). Combining both splits, the true difference between v7 and
  pooled SnapKV on LongBench is ≈ +0.2 ± 0.4 — **not resolvable, and probably
  near zero**.
- Of the three mechanisms, only **value weighting survives on the test split**
  (+0.35, same direction as dev's +0.8); waterfill and pooling read −0.25 and
  −0.19 here after reading +0.05 and positive on dev — i.e. within noise, no
  test-split evidence they help LongBench.

### The conservation finding

Across two disjoint sample splits (400 + 800 paired samples), two budgets
(5 % and 2 %) and twelve configurations, every competent window-scored
selection method — SnapKV pooled or not, PyramidKV, every v7 variant — lands
within ±0.4 of the same LongBench mean, while the *per-task* profile swings by
±3–7 points per configuration. LongBench macro at these budgets measures the
scoring family, not the mechanism details; resolving ±0.3 would need ≥10 000
paired samples (the full English suite has 3 350). This extends
`docs/SUMMARY.md` §8.2 with far more evidence.

## 3. Needle-in-a-haystack — where the mechanisms do resolve

Llama-3.1-8B, contexts 1k–8k, 25 needle positions:

| method | 5 % budget | 2 % budget |
|---|---:|---:|
| **OT-KV v7** | **1.00** | **1.00** |
| v7 −waterfill | 1.00 | — |
| v7 −pooling | 0.96 | — |
| SnapKV (global, pooled) | 0.92 | 0.88 |
| PyramidKV (global) | 0.92 | 0.76 |
| SnapKV (global, unpooled) | 0.88 | — |
| v7 −vnorm | 0.84 | — |
| v7 with auto-regime (τ=0.73) | 0.76 | 0.80 |
| OT-KV v4 (auto)+pyramid | 0.60 | — |
| H2O (global) | 0.40 | — |
| StreamingLLM | 0.20 | — |

Perfect retrieval at both budgets, and the ablations attribute it: value
weighting is +0.16 on its own (the needle's value vector is exactly the
high-norm unusual value Lemma 1 prices highest), pooling +0.04 on top of it,
waterfill neutral. Qwen3-1.7B local sweeps agree (v7 5/5, pooled SnapKV 5/5,
unpooled 1/5).

## 3b. RULER-style retrieval suite (synthetic, seeded; 25 samples/task, 8k)

Harder multi-target retrieval, `evaluation/tasks/retrieval_suite.py`
(the public llamastack RULER mirror holds 1–12 rows per task, so the four
core recipes are regenerated from seeds):

| method @ 5 % | single | multikey | multiquery | vt | avg |
|---|---:|---:|---:|---:|---:|
| Dense | 100 | 96 | 99 | 55 | 87.5 |
| SnapKV (global, pooled) | 96 | 96 | 94 | 47 | 83.2 |
| **OT-KV v7** | 96 | 96 | 91 | 45 | 82.0 |
| PyramidKV (global) | 96 | 76 | 88 | 36 | 74.0 |
| H2O (global) | 0 | 0 | 0 | 0 | 0.0 |

| method @ 2 % | single | multikey | multiquery | vt | avg |
|---|---:|---:|---:|---:|---:|
| **OT-KV v7** | **100** | **100** | **63** | 32 | **73.8** |
| SnapKV (global, pooled) | 100 | 96 | 42 | 31 | 67.2 |

At 5 % v7 and pooled SnapKV tie (both above PyramidKV's multikey collapse and
H2O's total failure). At 2 %, where every slot counts, **v7 leads by +6.6
overall and +21 on multi-query retrieval** — keeping several needles
simultaneously is exactly what value-weighted, water-filled selection buys
when slots are scarce.

## 4. Wikitext perplexity — 6 documents, prefill 50 %, 5 % budget

| method | word ppl |
|---|---:|
| Dense | 3.32 |
| StreamingLLM | 3.90 |
| H2O (per-head) | 4.30 |
| SnapKV (global, unpooled) | 4.31 |
| PyramidKV (global) | 4.32 |
| **OT-KV v7** | **4.43** |
| v7 −waterfill | 4.47 |
| SnapKV (global, pooled) | 4.49 |
| v7 −vnorm | 4.52 |

Perplexity rewards recency- and peak-attention-weighted allocations —
StreamingLLM "wins" it while scoring 0.20 NIAH and last-but-one on LongBench;
the metric is mildly anti-correlated with LongBench (`docs/SUMMARY.md` §8.1).
Pooling itself costs ppl (unpooled SnapKV 4.31 vs pooled 4.49): smearing the
selection over a spike's neighbourhood trades next-token fidelity for
retrieval. v7 sits between the two SnapKV variants on ppl while beating both
on NIAH; H2O/PyramidKV/unpooled-SnapKV keep a 2–3 % ppl edge, the price of
v7's retrieval-oriented scoring. On Qwen3-1.7B, where the regime detector's
statistic separates, `auto_regime=true` recovers the continuation path and v7
is best outright: KL −16 % vs H2O, −25 % vs pooled SnapKV at both 0.15 and
0.075 budgets, holding less cache.

## 5. Efficiency — H200, 8 001-token prompt, 128 new tokens, 5 % budget

| method | TTFT (s) | decode tok/s | peak KV (MB) |
|---|---:|---:|---:|
| Dense | 1.40 | 23.96 | 1016.0 |
| PyramidKV | 1.39 | 18.11 | 50.07 |
| H2O | 1.43 | 18.02 | 50.13 |
| SnapKV (pooled) | 1.40 | 17.82 | 50.13 |
| **OT-KV v7** | 1.44 | 17.22 | **50.00** |
| OT-KV v4 (auto)+pyramid | 1.44 | 16.17 | 49.94 |

3.4 % slower decode than pooled SnapKV, smallest peak KV of any method
(20.3× below dense), TTFT parity.

## 6. What can and cannot be claimed

**Supported.**
- v7 is the only method in the top statistical tier of all three metrics
  simultaneously: LongBench tied with SnapKV/PyramidKV, NIAH strictly best
  (1.00 at both budgets), perplexity mid-pack (beats pooled SnapKV; within
  3 % of every ppl-stronger method, each of which loses NIAH by ≥ 0.08 and/or
  LongBench significantly).
- Against each named baseline: **H2O** — beaten by +2.3 LongBench (CI excludes
  0) and +0.60 NIAH, loses ppl by 3 %; **SnapKV** — tied on LongBench, beaten
  on NIAH by both variants (1.00 vs 0.92 pooled / 0.88 unpooled), ppl sits
  between the variants (beats pooled 4.49, trails unpooled 4.31);
  **PyramidKV** — tied on LongBench, beaten on NIAH (decisively at 2 %:
  1.00 vs 0.76), loses ppl by 2.5 %.
- Value weighting (attention × ‖v‖) is the one new mechanism that helps on
  every metric: +0.35 LongBench (test), +0.16 NIAH, +0.09 ppl.
- The conservation finding (§2) and the detector non-transfer
  (`detector_probe.py`) — both negative results with more evidential power
  than the headline comparisons.

**Not supported.**
- Any claim that v7 beats SnapKV or PyramidKV on the LongBench mean.
- Test-split evidence that waterfill or pooling improve LongBench (they are
  motivated by NIAH +0.04 / ppl +0.04, and by dev trends only).
- The regime detector on anything but Qwen-family models.
