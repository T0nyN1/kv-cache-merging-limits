#!/bin/bash
PY=~/anaconda3/envs/ot-kv/bin/python
cd "."
export PYTORCH_ENABLE_MPS_FALLBACK=1
S=runs/local_repro/STATUS
echo "ROUND3 START $(date)" >> $S
run() { name=$1; shift; echo "[$(date +%H:%M:%S)] start $name" >> $S; "$@" > runs/local_repro/$name.log 2>&1; rc=$?; echo "[$(date +%H:%M:%S)] done $name exit=$rc" >> $S; return $rc; }
$PY -m py_compile experiments/classa_oracle.py || { echo "COMPILE FAIL" >> $S; exit 1; }
run smoke_oracle3 $PY experiments/classa_oracle.py --device mps --docs 1 --prefill 512 --probe 64 --split temporal --causal --prefix-cache || { echo "SMOKE FAIL" >> $S; exit 1; }
for P in 256 512 1024 2048; do
  run oracle_p${P}_prefixcache $PY experiments/classa_oracle.py --device mps --prefill 2048 --probe $P --split temporal --causal --prefix-cache --out runs/local_repro/oracle_p${P}_prefixcache.json
done
run oracle_pre3072_p2048_prefixcache $PY experiments/classa_oracle.py --device mps --prefill 3072 --probe 2048 --split temporal --causal --prefix-cache --out runs/local_repro/oracle_pre3072_p2048_prefixcache.json
run oracle_pre4096_p3072_prefixcache $PY experiments/classa_oracle.py --device mps --prefill 4096 --probe 3072 --split temporal --causal --prefix-cache --out runs/local_repro/oracle_pre4096_p3072_prefixcache.json
echo "ROUND3 DONE $(date)" >> $S
