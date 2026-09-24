#!/bin/bash
PY=~/anaconda3/envs/ot-kv/bin/python
cd "."
export PYTORCH_ENABLE_MPS_FALLBACK=1
S=runs/local_repro/STATUS
echo "ROUND4 START $(date)" >> $S
run() { name=$1; shift; echo "[$(date +%H:%M:%S)] start $name" >> $S; "$@" > runs/local_repro/$name.log 2>&1; rc=$?; echo "[$(date +%H:%M:%S)] done $name exit=$rc" >> $S; return $rc; }
$PY -m py_compile experiments/classa_oracle.py || { echo "COMPILE FAIL" >> $S; exit 1; }
run smoke_oracle4 $PY experiments/classa_oracle.py --device mps --docs 1 --prefill 512 --probe 192 --split temporal --causal --prefix-cache --test-len 64 || { echo "SMOKE FAIL" >> $S; exit 1; }
# fixed 128-position horizon, prefill 2560, fit positions = probe-128
for P in 256 384 640 1152 2176; do
  run oracle_h128_p${P} $PY experiments/classa_oracle.py --device mps --prefill 2560 --probe $P --split temporal --causal --prefix-cache --test-len 128 --out runs/local_repro/oracle_h128_p${P}.json
done
# horizon dependence at fixed fit size (512 positions)
run oracle_h64_p576   $PY experiments/classa_oracle.py --device mps --prefill 2560 --probe 576 --split temporal --causal --prefix-cache --test-len 64  --out runs/local_repro/oracle_h64_p576.json
run oracle_h256_p768  $PY experiments/classa_oracle.py --device mps --prefill 2560 --probe 768 --split temporal --causal --prefix-cache --test-len 256 --out runs/local_repro/oracle_h256_p768.json
run oracle_h512_p1024 $PY experiments/classa_oracle.py --device mps --prefill 2560 --probe 1024 --split temporal --causal --prefix-cache --test-len 512 --out runs/local_repro/oracle_h512_p1024.json
# longer context, same horizon
run oracle_h128_pre4096_p2176 $PY experiments/classa_oracle.py --device mps --prefill 4096 --probe 2176 --split temporal --causal --prefix-cache --test-len 128 --out runs/local_repro/oracle_h128_pre4096_p2176.json
echo "ROUND4 DONE $(date)" >> $S
