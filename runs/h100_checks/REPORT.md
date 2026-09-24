# H100 follow-up checks: paired statistics

Generated 2026-09-24 11:52 by experiments/h100_checks_report.py. Llama-3.1-8B-Instruct, LongBench 8 English tasks, max_length 8000, sink 4, recent 10 %, 64 new tokens, H100-80GB bf16. Δ = mean per-sample difference ×100 (method − base). CI = 20,000-resample paired bootstrap over samples. perm-s = paired permutation p over samples; perm-t = exact sign-flip p over the task means. changed = samples whose score moved at all.

## Step 5: reproduction gate (pooled SnapKV, dev split, H100 vs Modal H200)

| run | task | n | samples differing | max \|Δ\| ×100 |
|---|---|---|---|---|
| repro_dev | qasper | 50 | 9 | 11.56 |
| repro_dev | samsum | 50 | 9 | 8.06 |
| base_dev | 2wikimqa | 50 | 5 | 5.19 |
| base_dev | gov_report | 50 | 16 | 3.18 |
| base_dev | hotpotqa | 50 | 5 | 4.88 |
| base_dev | multi_news | 50 | 25 | 5.32 |
| base_dev | multifieldqa_en | 50 | 10 | 28.57 |
| base_dev | qasper | 50 | 9 | 11.56 |
| base_dev | samsum | 50 | 9 | 8.06 |
| base_dev | triviaqa | 50 | 0 | 0.00 |
| base_test | 2wikimqa | 100 | 16 | 9.30 |
| base_test | gov_report | 100 | 33 | 4.23 |
| base_test | hotpotqa | 100 | 9 | 12.24 |
| base_test | multi_news | 100 | 51 | 8.84 |
| base_test | multifieldqa_en | 100 | 19 | 29.03 |
| base_test | qasper | 100 | 26 | 17.54 |
| base_test | samsum | 100 | 30 | 28.82 |
| base_test | triviaqa | 100 | 3 | 80.00 |

Pooled-SnapKV base used below: dev = `runs/h100_checks/base_dev`, test = `runs/h100_checks/base_test` (H100 rerun, because the H100 run differed from the Modal files).

## Step 7 gate: `keepkv;selector=selkv;merge=false` vs H100 `selkv;selector=selkv;merge=false` (must be identical)

| task | n | samples differing | max \|Δ\| ×100 |
|---|---|---|---|
| 2wikimqa | 50 | 0 | 0.00 |
| gov_report | 50 | 0 | 0.00 |
| hotpotqa | 50 | 0 | 0.00 |
| multi_news | 50 | 0 | 0.00 |
| multifieldqa_en | 50 | 0 | 0.00 |
| qasper | 50 | 0 | 0.00 |
| samsum | 50 | 0 | 0.00 |
| triviaqa | 50 | 0 | 0.00 |

## Step 7 gate: `keepkv;selector=selkv;merge=false` vs Modal H200 `selkv;selector=selkv;merge=false` (hardware drift expected)

| task | n | samples differing | max \|Δ\| ×100 |
|---|---|---|---|
| 2wikimqa | 50 | 7 | 19.84 |
| gov_report | 50 | 17 | 4.48 |
| hotpotqa | 50 | 5 | 24.76 |
| multi_news | 50 | 24 | 4.70 |
| multifieldqa_en | 50 | 8 | 48.15 |
| qasper | 50 | 14 | 13.31 |
| samsum | 50 | 13 | 34.62 |
| triviaqa | 50 | 1 | 60.00 |

## Step 5b: H100 vs Modal H200 for `selkv;selector=selkv;merge=false` (own-selector base)

| task | n | samples differing | max \|Δ\| ×100 |
|---|---|---|---|
| 2wikimqa | 50 | 7 | 19.84 |
| gov_report | 50 | 17 | 4.48 |
| hotpotqa | 50 | 5 | 24.76 |
| multi_news | 50 | 24 | 4.70 |
| multifieldqa_en | 50 | 8 | 48.15 |
| qasper | 50 | 14 | 13.31 |
| samsum | 50 | 13 | 34.62 |
| triviaqa | 50 | 1 | 60.00 |

Own-selector merge-off base used below: `runs/h100_checks/base_selkv_off`.

## Paired statistics

| step | comparison | n | Δ ×100 | 95 % CI | sign test (win/loss, p) | perm-s p | perm-t p | tasks > 0 | changed | resolved |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 sanity | otkv8 dev (Modal dev8k_v8) | 400 | +0.70 | [+0.14, +1.37] | 120/96, 0.117 | 0.017 | 0.023 | 7/8 | 216 | yes |
| 6 / Run A | otkv8, test split (800) | 800 | +0.18 | [-0.27, +0.62] | 226/238, 0.610 | 0.455 | 0.242 | 6/8 | 464 | no |
| 7 / Run B | keepkv;selector=selkv (keys moved) vs merge-off | 400 | -10.40 | [-12.71, -8.24] | 83/252, 0.000 | 0.000 | 0.008 | 0/8 | 335 | yes |
| 7 / Run B | keepkv;selector=selkv;keys=fixed vs merge-off | 400 | +0.10 | [-0.81, +0.98] | 131/123, 0.661 | 0.823 | 0.875 | 3/8 | 254 | no |
| 7 / Run B | keys fixed vs keys moved (direct contrast) | 400 | +10.50 | [+8.32, +12.79] | 252/84, 0.000 | 0.000 | 0.008 | 8/8 | 336 | yes |
| 8 / Run C | selkv;selector=snapkv;fallback=global vs pooled SnapKV | 400 | -0.37 | [-0.75, -0.04] | 63/99, 0.006 | 0.035 | 0.078 | 1/8 | 162 | no |
| 8 / Run C | selkv;selector=selkv;fallback=global vs merge-off | 400 | -0.23 | [-0.83, +0.26] | 77/76, 1.000 | 0.481 | 0.484 | 5/8 | 153 | no |
| 9 / Run D | kvmerger @35 % vs pooled SnapKV @35 % | 400 | -0.48 | [-0.95, -0.04] | 87/107, 0.172 | 0.038 | 0.047 | 2/8 | 194 | yes |
| 9 / Run D | kvmerger @35 % rerun (stats job) vs pooled SnapKV @35 % | 400 | -0.40 | [-0.85, +0.01] | 88/100, 0.422 | 0.067 | 0.070 | 2/8 | 188 | no |
| 9 / Run D | kvmerger rerun vs kvmerger original (same config, same GPU type: run-to-run noise) | 400 | +0.08 | [-0.11, +0.29] | 31/19, 0.119 | 0.443 | 0.375 | 4/8 | 50 | no |

## Method and base means (LongBench ×100, mean of the 8 task means over the paired samples)

| comparison | method | base |
|---|---|---|
| otkv8 dev (Modal dev8k_v8) | 25.73 | 25.02 |
| otkv8, test split (800) | 26.12 | 25.94 |
| keepkv;selector=selkv (keys moved) vs merge-off | 14.72 | 25.12 |
| keepkv;selector=selkv;keys=fixed vs merge-off | 25.22 | 25.12 |
| keys fixed vs keys moved (direct contrast) | 25.22 | 14.72 |
| selkv;selector=snapkv;fallback=global vs pooled SnapKV | 24.75 | 25.12 |
| selkv;selector=selkv;fallback=global vs merge-off | 24.89 | 25.12 |
| kvmerger @35 % vs pooled SnapKV @35 % | 27.47 | 27.95 |
| kvmerger @35 % rerun (stats job) vs pooled SnapKV @35 % | 27.55 | 27.95 |
| kvmerger rerun vs kvmerger original (same config, same GPU type: run-to-run noise) | 27.55 | 27.47 |

## Per-task Δ ×100

| comparison | 2wikimqa | gov_report | hotpotqa | multi_news | multifieldqa_en | qasper | samsum | triviaqa |
|---|---|---|---|---|---|---|---|---|
| otkv8 dev (Modal dev8k_v8) | +0.68 | +0.51 | +0.23 | -0.40 | +0.58 | +0.76 | +1.55 | +1.73 |
| otkv8, test split (800) | -0.35 | +0.36 | +0.15 | +0.22 | -0.42 | +0.35 | +0.70 | +0.40 |
| keepkv;selector=selkv (keys moved) vs merge-off | -6.79 | -3.39 | -10.13 | -3.04 | -11.83 | -6.44 | -3.65 | -37.90 |
| keepkv;selector=selkv;keys=fixed vs merge-off | +1.53 | +0.89 | -1.23 | -0.16 | +3.13 | -0.96 | -2.22 | -0.15 |
| keys fixed vs keys moved (direct contrast) | +8.32 | +4.28 | +8.90 | +2.88 | +14.96 | +5.48 | +1.43 | +37.75 |
| selkv;selector=snapkv;fallback=global vs pooled SnapKV | -0.24 | -0.06 | -0.28 | -0.34 | -1.78 | +0.29 | -0.55 | +0.00 |
| selkv;selector=selkv;fallback=global vs merge-off | -0.19 | +0.11 | +0.36 | +0.23 | +0.00 | -0.88 | +0.28 | -1.73 |
| kvmerger @35 % vs pooled SnapKV @35 % | -1.50 | -0.24 | -0.83 | +0.25 | -0.33 | -0.60 | -0.59 | +0.01 |
| kvmerger @35 % rerun (stats job) vs pooled SnapKV @35 % | -1.00 | -0.08 | -0.71 | +0.44 | -0.36 | -0.85 | -0.64 | +0.01 |
| kvmerger rerun vs kvmerger original (same config, same GPU type: run-to-run noise) | +0.50 | +0.16 | +0.12 | +0.19 | -0.03 | -0.25 | -0.05 | +0.00 |

## Merge statistics (summed over samples; a run with zero merges/routings is plain eviction)

- `keepkv;selector=selkv` over 400 sample records: merged 4.472e+07, evicted 5.887e+08, candidates 4.472e+07; merged / evicted = 7.6 %
- `keepkv;selector=selkv;keys=fixed` over 400 sample records: merged 4.472e+07, evicted 5.887e+08, candidates 4.472e+07; merged / evicted = 7.6 %
- `selkv;selector=snapkv;fallback=global` over 400 sample records: routed 5.887e+08, routed_bucket 3.181e+07, fallback_routed 5.569e+08, fallback_miss 5047, evicted 5.887e+08, dropped_empty_bucket 5047; routed / evicted = 100.0 %
- `selkv;selector=selkv;fallback=global` over 400 sample records: routed 5.887e+08, routed_bucket 2.239e+07, fallback_routed 5.663e+08, fallback_miss 9729, evicted 5.887e+08, dropped_empty_bucket 9729; routed / evicted = 100.0 %
- `kvmerger;observation_window=32;per_head=false;pool_kernel=7 [prefill]` over 400 sample records: sets 1.705e+06, merged_tokens 3.363e+07, evicted_tokens 5.035e+07; merged / evicted = 66.8 %
- `kvmerger;observation_window=32;per_head=false;pool_kernel=7 [decode16]` over 333 sample records: sets 1.564e+06, merged_tokens 2.767e+07, evicted_tokens 4.123e+07; merged / evicted = 67.1 %
- `kvmerger;observation_window=32;per_head=false;pool_kernel=7 [decode32]` over 282 sample records: sets 1.436e+06, merged_tokens 2.307e+07, evicted_tokens 3.446e+07; merged / evicted = 66.9 %
- `kvmerger;observation_window=32;per_head=false;pool_kernel=7 [decode48]` over 242 sample records: sets 1.319e+06, merged_tokens 1.916e+07, evicted_tokens 2.882e+07; merged / evicted = 66.5 %

## Notes

- Hardware: every base and every method row above was produced on H100-80GB (bf16, eager attention). The Modal H200 files differ on about a fifth of the samples (tables at the top) while task means agree to within 0.1, so no delta in this report pairs an H100 run with an H200 run.
- Determinism: eviction-only runs (pooled SnapKV, SelKV/KeepKV with merge off) reproduce bit for bit across H100 jobs. The KVMerger and SelKV merges accumulate with CUDA `index_add_` (float atomics, order-dependent), so their per-sample scores carry run-to-run noise; the `kvmerger rerun vs kvmerger original` row measures it directly for Run D. KeepKV and otkv8 do not use atomic accumulation.
- `resolved` follows the paper's rule: bootstrap CI excludes zero and both permutation p < 0.05.
- The wikitext smoke test (step 4) was not completed: the lm-eval task needs `EleutherAI/wikitext_document_level` (now cached) and evaluates all 62 documents without `--wiki_docs`; the GPU path was validated by the LongBench runs.
