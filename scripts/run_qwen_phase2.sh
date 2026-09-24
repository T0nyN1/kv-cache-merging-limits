#!/usr/bin/env bash
# Phase-2 runner for a plain Linux GPU box (no SLURM): Qwen2.5-7B-Instruct, LongBench dev + 100-position NIAH.
# Second-model runs of paper Appendix B.8 (Table 9). Usage, from the repo root:
#   bash scripts/run_qwen_phase2.sh setup      # venv + model download + sanity (10 samples)
#   bash scripts/run_qwen_phase2.sh longbench  # the 400-sample dev block (8 methods + bit-check)
#   bash scripts/run_qwen_phase2.sh niah       # 100-position needle runs at 5 % and 2 %
#   bash scripts/run_qwen_phase2.sh all        # everything, in order
# Run it under nohup or tmux; every python call logs to $OUT/logs/. Re-running skips finished runs
# (a run is finished when its summary CSV exists).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
DATA=${DATA:-$ROOT/..}                      # data disk root: model + venv + uv cache live here
MODEL=${MODEL:-$DATA/models/Qwen2.5-7B-Instruct}
VENV=${VENV:-$DATA/ot-kv-env}
OUT=${OUT:-$ROOT/runs/qwen_checks}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$DATA/.uv-cache}
export HF_HOME=${HF_HOME:-$DATA/hf}
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
mkdir -p "$OUT/logs" "$DATA/models" "$HF_HOME"
TASKS8=qasper,multifieldqa_en,hotpotqa,2wikimqa,gov_report,multi_news,triviaqa,samsum
PROTO=(--compression_size 0.05 --recent_size 0.1 --sink_size 4 --mode prefill --per_head true --max_length 8000)
SNAP="snapkv;observation_window=32;per_head=false;pool_kernel=7"

setup() {
  command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }
  export PATH=$HOME/.local/bin:$PATH
  if [ ! -x "$VENV/bin/python" ]; then
    uv python install 3.12
    uv venv "$VENV" --python 3.12
    uv pip install --python "$VENV/bin/python" torch==2.11.0 transformers==5.7.0 accelerate lm-eval wonderwords nltk \
      datasets tiktoken hf_transfer rouge jieba scipy numpy tqdm sentencepiece protobuf "huggingface_hub[cli]"
    uv cache clean || true
  fi
  "$VENV/bin/python" -c "import torch,transformers;print('torch',torch.__version__,'cuda',torch.version.cuda,'transformers',transformers.__version__);print('gpu',torch.cuda.get_device_name(0),'bf16',torch.cuda.is_bf16_supported())" | tee "$OUT/logs/env.txt"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee -a "$OUT/logs/env.txt"
  if [ ! -f "$MODEL/model.safetensors.index.json" ]; then
    "$VENV/bin/huggingface-cli" download Qwen/Qwen2.5-7B-Instruct --local-dir "$MODEL" --exclude "*.gguf"
  fi
  [ -d datasets/LongBench_dataset/data ] || echo "LongBench will be downloaded by the evaluator on first use"
  [ -d datasets/niah/PaulGrahamEssays ] || { echo "datasets/niah/PaulGrahamEssays missing"; exit 1; }
  run sanity longbench --longbench_tasks qasper,samsum --limit 10 --longbench_offset 0 \
      --methods baseline "$SNAP"
  echo "sanity done; check $OUT/sanity/*.csv (dense qasper/samsum should be sane) and peak memory in the log"
}

# run <name> <task> <extra args...>   (skips if the summary CSV already exists)
run() {
  local name=$1 task=$2; shift 2
  if ls "$OUT/$name"/*.csv >/dev/null 2>&1; then echo "[skip] $name"; return; fi
  echo "[run ] $name  ($(date))"
  "$VENV/bin/python" main.py --model_id "$MODEL" --tasks "$task" "${PROTO[@]}" \
      --save_dir "$OUT/$name" --filename "$name" "$@" 2>&1 | tee "$OUT/logs/$name.log"
}

longbench() {
  local LB=(longbench --longbench_tasks $TASKS8 --limit 50 --longbench_offset 0)
  run base_dev        "${LB[@]}" --methods "$SNAP"
  run keepkv_off      "${LB[@]}" --methods "keepkv;merge=false"
  MERGE_STATS_PATH="$OUT/otkv8/merge_stats.jsonl"  run otkv8        "${LB[@]}" --methods "otkv8;observation_window=32;per_head=false;pool_kernel=7"
  MERGE_STATS_PATH="$OUT/keepkv/merge_stats.jsonl" run keepkv       "${LB[@]}" --methods "keepkv"
  MERGE_STATS_PATH="$OUT/keepkv_fixed/merge_stats.jsonl" run keepkv_fixed "${LB[@]}" --methods "keepkv;keys=fixed"
  run vws             "${LB[@]}" --methods "otkv7;auto_regime=false"
  run pyramidkv       "${LB[@]}" --methods "pyramidkv;observation_window=32;per_head=false"
  run dense           "${LB[@]}" --methods baseline
  MERGE_STATS_PATH="$OUT/selkv/merge_stats.jsonl"    run selkv    "${LB[@]}" --methods "selkv;selector=snapkv"
  MERGE_STATS_PATH="$OUT/kvmerger/merge_stats.jsonl" run kvmerger "${LB[@]}" --methods "kvmerger;observation_window=32;per_head=false;pool_kernel=7"
  echo "bit-check: keepkv;merge=false must equal the base sample for sample"
  "$VENV/bin/python" experiments/test_compensated_ports.py --bitcheck "$OUT/keepkv_off" _keepkv_merge_false.json \
      "$OUT/base_dev" _snapkv_observation_window_32_per_head_false_pool_kernel_7.json 2>&1 | tail -3 || echo "bitcheck helper failed; compare the persample files by hand"
}

niah() {
  local N=(niah --context_intervals 10 --depth_intervals 10)
  run niah100_c05 "${N[@]}" --methods "$SNAP" "pyramidkv;observation_window=32;per_head=false" "otkv7;auto_regime=false"
  run niah100_c05_dense "${N[@]}" --methods baseline
  PROTO=(--compression_size 0.02 --recent_size 0.1 --sink_size 4 --mode prefill --per_head true --max_length 8000)
  run niah100_c02 "${N[@]}" --methods "$SNAP" "pyramidkv;observation_window=32;per_head=false" "otkv7;auto_regime=false"
}

case "${1:-all}" in
  setup) setup ;;
  longbench) longbench ;;
  niah) niah ;;
  all) setup; longbench; niah ;;
  *) echo "usage: $0 {setup|longbench|niah|all}"; exit 1 ;;
esac
echo "DONE $(date)"
