#!/bin/bash
PY=~/anaconda3/envs/ot-kv/bin/python
cd "."
export PYTORCH_ENABLE_MPS_FALLBACK=1
echo "START $(date)" > runs/local_repro/STATUS
run() { name=$1; shift; echo "[$(date +%H:%M:%S)] start $name" >> runs/local_repro/STATUS; "$@" > runs/local_repro/$name.log 2>&1; echo "[$(date +%H:%M:%S)] done $name exit=$?" >> runs/local_repro/STATUS; }
run classa_oracle_qwen3      $PY experiments/classa_oracle.py --device mps --out runs/local_repro/classa_oracle_qwen3.json
run cancel_qwen3             $PY experiments/cancel.py
run closed_qwen3             $PY experiments/closed.py
run merge_theory_qwen3       $PY experiments/merge_theory.py --device mps
run sigma_families_qwen3     $PY experiments/sigma_families.py --model Qwen/Qwen3-1.7B --device mps --dtype float32 --out runs/local_repro/sigma_qwen3.json
run classa_oracle_qwen3_p1024 $PY experiments/classa_oracle.py --device mps --probe 1024 --out runs/local_repro/classa_oracle_qwen3_probe1024.json
echo "ALL DONE $(date)" >> runs/local_repro/STATUS
