#!/bin/bash
PY=~/anaconda3/envs/ot-kv/bin/python
cd "."
export PYTORCH_ENABLE_MPS_FALLBACK=1
S=runs/local_repro/STATUS
echo "ROUND6 START $(date)" >> $S
run() { name=$1; shift; echo "[$(date +%H:%M:%S)] start $name" >> $S; "$@" > runs/local_repro/$name.log 2>&1; rc=$?; echo "[$(date +%H:%M:%S)] done $name exit=$rc" >> $S; return $rc; }
$PY -m py_compile experiments/centroid_gap.py experiments/classa_oracle_decode.py || { echo "COMPILE FAIL" >> $S; exit 1; }
run centroid_gap_qwen3 $PY experiments/centroid_gap.py --model Qwen/Qwen3-1.7B --device mps --dtype float32 --out runs/local_repro/centroid_gap_qwen3.json
run classa_oracle_decode_qwen3 $PY experiments/classa_oracle_decode.py --device mps --out runs/local_repro/classa_oracle_decode_qwen3.json
RAM=$(( $(sysctl -n hw.memsize) / 1073741824 ))
if [ "$RAM" -ge 32 ]; then
  run centroid_gap_llama $PY experiments/centroid_gap.py --model models/Llama-3.1-8B-Instruct --device mps --dtype bfloat16 --docs 2 --prefill 2500 --out runs/local_repro/centroid_gap_llama.json
else
  echo "SKIP llama: RAM ${RAM}GB" >> $S
fi
echo "ROUND6 DONE $(date)" >> $S
