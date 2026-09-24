# On the Limits of KV Cache Merging

Code, per-sample results and analysis scripts for the paper *On the Limits of KV Cache Merging*
(under review).

Merging evicted key–value entries into retained ones is an appealing alternative to evicting them.
This repository contains everything behind the paper's answer to how much that can pay on frozen
rotary-position LLMs:

- **Theory checks and mechanism measurements**: the key gap σ under oracle routing across model
  families, the RoPE-removed same-token analysis, the unrestricted class-A (value-only) oracle, and
  the prefill-to-decode drift of the logit gap.
- **Matched-protocol ports** of published merging operators (CaM, KVMerger, KeepKV, SelKV), each
  paired against its own eviction base, with the underspecified choices documented.
- **The fixed-key compensated construction** that fits a value mixture and a per-slot mass
  correction jointly against the attention output.
- **Value-weighted selection (VWS)**: window attention × value norm, token-axis pooling and
  cross-layer water-filling.
- **The evaluation framework** (LongBench, needle-in-a-haystack, a RULER-style retrieval suite,
  wikitext perplexity, efficiency profiling) with paired bootstrap / permutation statistics.

## Layout

| path | contents |
|---|---|
| `main.py` | evaluation entry point (`--tasks`, `--methods "name;key=value;..."`) |
| `evaluation/` | Hugging Face wrapper with attention hooks (`models/`), task evaluators (`tasks/`) |
| `baselines/` | StreamingLLM, H2O, SnapKV, PyramidKV, and the CaM, KVMerger, KeepKV and SelKV ports |
| `core/` | method code; `ot_kv_v7.py` is VWS, `otkv_v8.py` / `otkv_v8_cache.py` the fixed-key construction, `consolidate.py` the key-averaging compensated merge |
| `experiments/` | mechanism measurements, oracles, statistics and report generators, unit tests |
| `paper/figures/` | scripts that regenerate the paper's figures and tables from `runs/` |
| `runs/` | per-sample score files, summaries and merge counters of the runs reported in the paper |
| `docs/` | method notes, port documentation and the experiment record |
| `misc/modal/`, `scripts/` | launchers (Modal and plain GPU boxes) |

Module and method names keep the project's original `otkv` prefix: `otkv7` with
`auto_regime=false` is VWS (no transport is involved), `otkv8` is the fixed-key construction, and
`otkv4` is the earlier transport merge.

## Setup

Python 3.12, `torch==2.11.0`, `transformers==5.7.0` (the attention hooks rely on the eager
attention path of this transformers version):

```bash
pip install torch==2.11.0 transformers==5.7.0 accelerate lm-eval wonderwords nltk datasets tiktoken rouge jieba scipy numpy tqdm sentencepiece protobuf
```

Models are public checkpoints (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3,
Qwen3-1.7B, GPT-2 medium). LongBench is downloaded on first use; the needle haystack expects the
Paul Graham essays under `datasets/niah/PaulGrahamEssays/`.

## Reproducing the main comparisons

The protocol of every LongBench row in the paper: 5 % KV budget, 4 sink tokens, recency window 10 %
of the budget, `max_length 8000` (middle truncation), greedy decoding, 64 new tokens, eager
attention. Development split = samples 0–50 of the eight English tasks, test split = samples 50–150.

```bash
python main.py --model_id <path>/Llama-3.1-8B-Instruct --tasks longbench \
  --longbench_tasks qasper,multifieldqa_en,hotpotqa,2wikimqa,gov_report,multi_news,triviaqa,samsum \
  --methods "snapkv;observation_window=32;per_head=false;pool_kernel=7" "otkv7;auto_regime=false" \
  --compression_size 0.05 --recent_size 0.1 --sink_size 4 --mode prefill --per_head true \
  --max_length 8000 --limit 50 --longbench_offset 0 --save_dir runs/example
```

Needle retrieval over 100 positions: `--tasks niah --context_intervals 10 --depth_intervals 10`.
`scripts/run_qwen_phase2.sh` runs the complete second-model block on a single GPU.

## Statistics and reports

Every comparison is paired against the identical selection without the operator.

- `experiments/paired_stats.py`, `experiments/ports_report.py`: bootstrap CI, sign test, permutation
  tests over samples and over task means.
- `experiments/h100_checks_report.py` → `runs/h100_checks/REPORT.md` (held-out split, second
  selector, SelKV fallback, KVMerger at 35 %).
- `experiments/qwen_checks_report.py` → `runs/qwen_checks/REPORT.md` (second model).
- `experiments/pooled_devtest.py`, `experiments/table9_llama_dev.py`: pooled and dev-split rows.
- `experiments/verify_paper_numbers.py`: recomputes the numbers quoted in the paper from `runs/`.

A run whose merge counters (`merge_stats.jsonl`) show zero merges is plain eviction and is never
reported as a merge.

## License

MIT (see `LICENSE`).
