# Results

Per-sample score files (`persample_<task>_<method>.json`, one score per sample), summary CSVs, merge/routing
counters (`merge_stats.jsonl`) and reports for every run the paper reports. File names carry the method string
with `;` and `=` replaced by `_`; LongBench files ending in `@50` are the test split (samples 50-150).

Protocol unless stated: LongBench, eight English tasks, 5 % KV budget, 4 sink tokens, recency window 10 % of the
budget, `max_length 8000`, greedy decoding, 64 new tokens, eager attention, bf16. Dev split = samples 0-50 (400
paired samples), test split = samples 50-150 (800 paired).

## `modal/` — Llama-3.1-8B-Instruct, H200

| directory | contents | paper |
|---|---|---|
| `dev8k` | dev split, every selector and merging operator at 5 %, including the pooled-SnapKV base of Table 2 | Tables 2, 5, Fig. 2 |
| `dev8k_fresh` | the pooled-SnapKV base re-run from scratch; identical sample for sample | Reproducibility statement |
| `dev8k_c02` | dev split at a 2 % budget (selectors and the key-averaged consolidation) | Table 2, App. B.6 |
| `dev8k_v8` | fixed-key construction (`otkv8`), dev split | Tables 2, 5 |
| `dev8k_bias`, `dev8k_nobias` | its channel ablations (`force_theta=0`, `force_c=1`) | Tables 2, 5 |
| `test8k` | test split, selectors and ablations | Tables 3, 4 |
| `ports_a` … `ports_h`, `ports_d25`, `ports_g25`, `ports_diag` | KeepKV and SelKV ports and their sensitivities (see `docs/compensated_ports.md`) | Table 2, App. C |
| `niah8k` | needle-in-a-haystack, 25 positions, 5 % / 2 %, VWS component ablations | Table 3 notes |
| `niah8k_100` | needle-in-a-haystack, 100 positions | Table 3, Sec. 8 |
| `niah8k_kvm`, `niah8k_v8` | needles for KVMerger and the fixed-key construction | Sec. 7, Sec. 6 |
| `ret8k`, `ret8k_c02` | RULER-style retrieval suite at 5 % and 2 % | Table 3, Sec. 8 |
| `ppl8k`, `ppl8k_kvm`, `ppl8k_v8` | wikitext perplexity | Tables 3, 7 |
| `profile8k` | time to first token, decode throughput, peak KV memory (8k prompt) | Table 3 |
| `probe_v8`, `probe_v8b` | merge statistics of the deployed fixed-key cache | App. B.2 |
| `sigma`, `review` | oracle-routing sigma and per-head / de-rotated sigma for Llama, Mistral, Qwen2.5 | Table 1, Fig. 1 |

## `local_repro/` — mechanism measurements (single GPU / laptop)

Oracle-routing and same-token sigma, the RoPE-removed analysis, the GPT-2 control, centroid gaps and break-even
constants, the class-A ridge oracle (including the anti-causal and selector-conditional checks), decode drift and
its within-prefill nulls, and the offline fixed-key intervention (`v8_offline_*`, Table 6). `run_*.sh` and
`nohup_*.out` are the launchers and their console output. Paper: Tables 1, 6, Fig. 1, Sec. 4-6, App. A-B.

## `h100_checks/` — follow-up checks, Llama-3.1-8B-Instruct, H100

Held-out split for the fixed-key construction, KeepKV on SelKV's selector, the SelKV global fallback, KVMerger at
a 35 % budget and its rerun, and the H100 re-runs of every base they are paired with. Report:
`REPORT.md` (`experiments/h100_checks_report.py`). Paper: App. B.7, Table 8.

## `qwen_checks/` — second model, Qwen2.5-7B-Instruct, RTX PRO 6000

Dev split for pooled SnapKV, dense, PyramidKV, VWS, the fixed-key construction, KeepKV (keys rewritten / fixed /
merge off), SelKV and KVMerger, plus 100-position needles at 5 % and 2 %. Report: `REPORT.md`
(`experiments/qwen_checks_report.py`). Paper: App. B.8, Table 9.

Every comparison pairs a method against the identical selection without the operator on the same hardware. A run
whose merge counters show zero merges is plain eviction and is never reported as a merge.
