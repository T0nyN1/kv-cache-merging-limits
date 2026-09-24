#!/bin/bash
PY=~/anaconda3/envs/ot-kv/bin/python
cd "."
export PYTORCH_ENABLE_MPS_FALLBACK=1
S=runs/local_repro/STATUS
echo "ROUND2 START $(date)" >> $S
run() { name=$1; shift; echo "[$(date +%H:%M:%S)] start $name" >> $S; "$@" > runs/local_repro/$name.log 2>&1; rc=$?; echo "[$(date +%H:%M:%S)] done $name exit=$rc" >> $S; return $rc; }
$PY -m py_compile experiments/classa_oracle.py experiments/sigma_decode.py || { echo "COMPILE FAIL" >> $S; exit 1; }
run smoke_oracle $PY experiments/classa_oracle.py --device mps --docs 1 --prefill 512 --probe 64 --split temporal --causal || { echo "SMOKE FAIL" >> $S; exit 1; }
run sigma_decode_qwen3 $PY experiments/sigma_decode.py --device mps --out runs/local_repro/sigma_decode_qwen3.json
# isolate the two corrections at the paper's default size
run oracle_p256_temporal        $PY experiments/classa_oracle.py --device mps --probe 256 --split temporal --out runs/local_repro/oracle_p256_temporal.json
run oracle_p256_causal          $PY experiments/classa_oracle.py --device mps --probe 256 --causal --out runs/local_repro/oracle_p256_causal.json
# corrected sweep: temporal split + causal mask
for P in 256 512 1024 2048; do
  run oracle_p${P}_temporal_causal $PY experiments/classa_oracle.py --device mps --prefill 2048 --probe $P --split temporal --causal --out runs/local_repro/oracle_p${P}_temporal_causal.json
done
run oracle_pre3072_p2048_temporal_causal $PY experiments/classa_oracle.py --device mps --prefill 3072 --probe 2048 --split temporal --causal --out runs/local_repro/oracle_pre3072_p2048_temporal_causal.json
echo "ROUND2 DONE $(date)" >> $S
