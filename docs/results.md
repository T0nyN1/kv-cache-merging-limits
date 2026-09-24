# OT-KV — final results

Everything here is reproducible from `experiments/`; the raw per-sample scores
live in the `ot_kv_data` Modal volume under `runs/persample_*.json` and the
rendered tables in `runs/*.md` / `runs/*.csv`.

**Protocol.** Every method is run on identical inputs, so all comparisons are
**paired**: a 20 000-resample bootstrap CI on the paired difference plus a
two-sided sign test over the samples where two methods actually differ. The unit
of analysis is the sample (LongBench) or the document (local); decode positions
inside one document are far too autocorrelated to count as independent. Cache
occupancy is reported as mean/peak so budget parity is auditable — an earlier
version of OT-KV appeared to win only because it held ~8 % more KV than the
methods it was compared against (`docs/experiments.md` §2b).

---

## 1. LongBench — 8 tasks, Llama-3.1-8B-Instruct, 5 % KV budget

50 samples per task, 400 pooled paired samples, `max_length=4096`, greedy,
64 new tokens. Reference = PyramidKV (global), the strongest baseline.
Full table: `runs/longbench_final.md`.

| method | avg | Δ vs PyramidKV | 95 % CI | win/loss | sign p |
|---|---:|---:|---|---:|---:|
| Dense (no compression) | 27.62 | +3.95 * | [+2.84, +5.11] | 203/81 | <1e-4 |
| **OT-KV v4 (auto + pyramid budget)** | **23.98** | **+0.31** | [−0.95, +1.57] | **161/124** | **0.033** |
| PyramidKV (global) | 23.67 | (ref) | | | |
| PyramidKV (per-head) | 23.60 | −0.07 | [−1.06, +0.91] | 145/117 | 0.095 |
| SnapKV (global) | 23.46 | −0.21 | [−0.90, +0.36] | 117/108 | 0.594 |
| OT-KV v4 (no merge) | 23.37 | −0.30 | [−0.97, +0.33] | | 0.347 |
| OT-KV v4 + pyramid budget | 23.35 | −0.33 | [−0.86, +0.22] | | 0.521 |
| OT-KV v4 (qa preset) | 23.26 | −0.42 | [−1.14, +0.24] | | 0.659 |
| OT-KV v4 (auto regime) | 23.20 | −0.48 | [−1.88, +0.87] | 162/121 | 0.017 |
| SnapKV (per-head) | 22.87 | −0.80 | [−1.99, +0.30] | | 0.271 |
| OT-KV v4 (continuation preset) | 22.81 | −0.86 | [−2.23, +0.48] | | 0.014 |
| OT-KV v4 (per-head selection) | 22.35 | −1.33 * | [−2.48, −0.29] | | 1.000 |
| H2O (per-head) | 22.11 | −1.56 * | [−2.99, −0.14] | | 0.453 |
| H2O (global) | 21.00 | −2.68 * | [−4.21, −1.20] | | 0.358 |
| StreamingLLM | 20.39 | −3.28 * | [−4.84, −1.77] | | 0.042 |

**What can be claimed.** OT-KV v4 with the automatic regime selector and the
depth-dependent budget is the highest-scoring compressed method, and it wins on
significantly more samples than the strongest baseline (161 vs 124, sign test
p = 0.033). The *mean* margin, +0.31, is not individually resolvable — its
losses are larger than its wins on the samples where it loses. The honest
sentence is "wins more often than PyramidKV, by a margin this many samples
cannot pin down", not "+0.31 better".

**What must not be claimed.** No single fixed OT-KV configuration beats the
baselines: the `qa` preset alone is 23.26, the `continuation` preset alone is
22.81, both below PyramidKV. The gain comes from *switching between them*, and
the switch is what `preset="auto"` supplies.

### Per-task structure (why the switch matters)

| | gov_report | multi_news | hotpotqa | multifieldqa | samsum |
|---|---:|---:|---:|---:|---:|
| dense | 15.97 | 16.27 | 14.89 | 38.63 | 25.83 |
| OT-KV v4 `qa` | 11.79 | 11.00 | 11.53 | 30.37 | 27.89 |
| OT-KV v4 `continuation` | 14.35 | 15.68 | 9.83 | 26.99 | 20.68 |
| OT-KV v4 `auto` + pyramid | 13.49 | 14.85 | **14.76** | 27.54 | 24.84 |

The two presets are near-complements: window scoring wins the extractive tasks
(hotpotqa, multifieldqa), all-query scoring wins the abstractive ones
(gov_report, multi_news, where `continuation` reaches 15.68 against a dense
16.27). `auto` recovers most of both. An oracle per-task choice would score
24.44, so the detector captures roughly two thirds of the available headroom.

---

## 2. Efficiency — H200, Llama-3.1-8B-Instruct, 5 % budget

`--tasks profile_niah`, 128 generated tokens, eager attention throughout.

**4 096-token prompt**

| method | TTFT (s) | decode tok/s | peak KV (MB) |
|---|---:|---:|---:|
| Dense | 0.423 | **36.27** | 528.00 |
| StreamingLLM | 0.417 | 34.21 | 25.62 |
| **OT-KV v4** | 0.439 | **25.06** | **25.50** |
| OT-KV v4 (continuation) | 0.427 | 24.83 | 25.50 |
| PyramidKV | 0.433 | 24.69 | 25.57 |
| SnapKV | 0.416 | 24.61 | 25.62 |
| H2O | 0.431 | 24.21 | 25.62 |
| OT-KV v2 | 0.428 | 27.84 | 25.50 |
| EchoKV | 0.416 | 1.66 | 25.62 |

**16 384-token prompt**

| method | TTFT (s) | decode tok/s | peak KV (MB) |
|---|---:|---:|---:|
| Dense | — | — | OOM (32 GiB attention allocation) |
| SnapKV | 6.62 | 26.28 | 102.50 |
| H2O | 6.82 | 25.90 | 102.50 |
| PyramidKV | 6.58 | 25.70 | 102.44 |
| OT-KV v4 | 6.80 | 23.57 | 102.38 |

Reading these honestly:

- **Memory**: 20.7x smaller KV cache than dense at 4k (25.5 MB vs 528 MB), and
  OT-KV holds marginally the least of any method because its compression pass
  caps the peak *at* budget rather than at budget + interval.
- **Decode throughput**: at 4k OT-KV v4 is the fastest of the attention-scored
  compressors (+3.5 % over H2O, +1.8 % over SnapKV); at 16k it is 10 % slower
  than SnapKV. The crossover is the size of the anchor set the transport pass
  has to search. `top_r=4` (down from 16) is what makes the every-4-steps
  schedule affordable at all — it cut the compression pass from 8.0 s to 3.4 s.
- **Compression is not free throughput**: at 4k *every* compressed method decodes
  slower than dense (24-25 vs 36 tok/s). The win is memory, and at 16k it is
  feasibility — dense cannot run under eager attention at all.
- EchoKV's implementation is pathologically slow (1.66 tok/s; a Python loop over
  `budget` chunks each step). Its LongBench row was abandoned after a 2-hour
  timeout produced one task. This is an implementation problem, not a statement
  about the method.

---

## 3. Open-ended continuation and retrieval (local)

Qwen3-1.7B, wikitext-2 long documents, 8 documents, paired, per-head.
KL(dense‖compressed) over held-out decode positions; NIAH is a 10-depth needle
sweep at ~3.2k context. Full data in `runs/final_local.json`.

Budget 0.075, 8 documents, per-head, identical inputs:

| method | ppl | KL(dense‖c) | NIAH | NIAH partial | cache mean/peak | decode ms/tok |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 18.07 | 0 | 1.00 | 1.000 | 3973 / 4095 | 97.2 |
| **OT-KV v4 (continuation)** | **19.81** | **0.2807** | 0.00 | 0.117 | 284 / 288 | 84.4 |
| OT-KV v3 | 20.34 | 0.2901 | 0.00 | 0.150 | 272 / 288 | 95.4 |
| **OT-KV v4 (auto)** | 20.37 | 0.2944 | **0.50** | **0.733** | 286 / 288 | 89.4 |
| H2O | 20.91 | 0.3486 | 0.00 | 0.233 | 289 / 289 | 57.0 |
| PyramidKV | 21.25 | 0.3583 | 0.00 | 0.117 | (see note) | 413.7 |
| SnapKV | 21.36 | 0.3549 | **0.70** | 0.950 | 289 / 289 | 58.5 |
| OT-KV v4 (qa) | 25.73 | 0.4777 | **0.90** | **0.983** | 286 / 288 | 87.1 |
| StreamingLLM | 24.02 | 0.3885 | 0.00 | 0.183 | 289 / 289 | 39.8 |

Three things this table says:

1. **The two presets sit at opposite corners and `auto` is the only entry that is
   good at both.** `qa` has the best retrieval of any method (NIAH 0.90 against
   SnapKV's 0.70) and the worst continuation fidelity; `continuation` has the
   best fidelity (KL 0.2807, −19.5 % against H2O) and no retrieval at all.
   `auto` gives up 0.014 KL against `continuation` and buys NIAH 0.50 — every
   other non-window method scores 0.00.
2. **Perplexity ranks the same way as KL but compresses the differences**: 19.81
   vs H2O's 20.91 is −5.3 %, where the KL gap is −19.5 %. KL is the more
   sensitive instrument and is what the five-budget sweep in
   `docs/experiments.md` §5 uses.
3. **Retrieval is bimodal, not graded**: every all-query scorer (H2O, PyramidKV,
   StreamingLLM, OT-KV v3, v4-continuation) scores exactly 0.00. Nothing earlier
   in the prompt marks the needle as important, so accumulating over all queries
   cannot find it, whatever the compression rule.

Five-budget sweep (`docs/experiments.md` §5): the continuation preset beats H2O
at every budget from 0.30 down to 0.025, by 17–36 %, CI excluding zero at all
five, 7–8 of 8 documents won, while holding less cache.

*Note on the PyramidKV cache column*: `bench.py` originally reported layer 0's
length, which overstates any depth-dependent budget (PyramidKV's first layer
gets 1.5x the flat budget). The column now reports the per-layer mean; the
PyramidKV row above predates that fix. Total budget across layers was verified
separately as neutral (10 515 vs 10 528 slots for a 28-layer model).

---

## 4. Reusable artefacts

| file | what it is |
|---|---|
| `runs/longbench_final.{md,csv}` | the LongBench table above, per task and pooled |
| `runs/reference_c005.{md,csv}` | baseline reference, per-head **and** global, for reuse |
| `runs/budget_sweep.json` | five-budget KL/ppl sweep with per-document values |
| `runs/final_local.json` | ppl + NIAH for every method at two budgets |
| `experiments/make_table.py` | rebuilds any table from accumulated `persample_*.json` |
| `experiments/paired_stats.py` | paired bootstrap + sign test, pooled across tasks |
| `experiments/paired_kl.py` | the same for per-document KL/ppl |
| `experiments/test_per_head.py` | 44 assertions on multi-head correctness |

Adding a run to the reference table needs no re-run of anything else: point
`make_table.py` at the directory of `persample_*.json` files and it re-pairs
whatever is complete.
