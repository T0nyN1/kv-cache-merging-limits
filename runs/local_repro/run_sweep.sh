#!/bin/bash
PY=~/anaconda3/envs/ot-kv/bin/python
cd "."
export PYTORCH_ENABLE_MPS_FALLBACK=1
echo "SWEEP START $(date)" >> runs/local_repro/STATUS
run() { name=$1; shift; echo "[$(date +%H:%M:%S)] start $name" >> runs/local_repro/STATUS; "$@" > runs/local_repro/$name.log 2>&1; echo "[$(date +%H:%M:%S)] done $name exit=$?" >> runs/local_repro/STATUS; }
for P in 128 512 2048; do
  run classa_oracle_qwen3_p$P $PY experiments/classa_oracle.py --device mps --prefill 2048 --probe $P --out runs/local_repro/classa_oracle_qwen3_probe$P.json
done
run classa_oracle_qwen3_pre3072_p2048 $PY experiments/classa_oracle.py --device mps --prefill 3072 --probe 2048 --out runs/local_repro/classa_oracle_qwen3_prefill3072_probe2048.json
for R in 1 2 3; do run merge_theory_qwen3_rep$R $PY experiments/merge_theory.py --device mps; done
echo "SWEEP DONE $(date)" >> runs/local_repro/STATUS
