#!/bin/bash
PY=~/anaconda3/envs/ot-kv/bin/python
cd "."
export PYTORCH_ENABLE_MPS_FALLBACK=1
S=runs/local_repro/STATUS
until grep -q '"--thresholds"' experiments/sigma_families.py && grep -q 'summary\[f"oracle_frac_le_' experiments/sigma_families.py; do sleep 2; done
sleep 2
echo "ROUND7 START $(date)" >> $S
run() { name=$1; shift; echo "[$(date +%H:%M:%S)] start $name" >> $S; "$@" > runs/local_repro/$name.log 2>&1; rc=$?; echo "[$(date +%H:%M:%S)] done $name exit=$rc" >> $S; return $rc; }
$PY -m py_compile experiments/sigma_families.py || { echo "COMPILE FAIL" >> $S; exit 1; }
run sigma_thr_llama $PY experiments/sigma_families.py --model models/Llama-3.1-8B-Instruct --device mps --dtype bfloat16 --thresholds 0.25 0.285 0.30 0.316 0.9 --out runs/local_repro/sigma_thr_llama.json
run sigma_thr_qwen3 $PY experiments/sigma_families.py --model Qwen/Qwen3-1.7B --device mps --dtype float32 --thresholds 0.25 0.285 0.30 0.316 0.9 --out runs/local_repro/sigma_thr_qwen3.json
echo "ROUND7 DONE $(date)" >> $S
