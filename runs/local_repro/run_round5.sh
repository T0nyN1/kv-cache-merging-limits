#!/bin/bash
PY=~/anaconda3/envs/ot-kv/bin/python
cd "."
export PYTORCH_ENABLE_MPS_FALLBACK=1
S=runs/local_repro/STATUS
echo "ROUND5 START $(date)" >> $S
run() { name=$1; shift; echo "[$(date +%H:%M:%S)] start $name" >> $S; "$@" > runs/local_repro/$name.log 2>&1; rc=$?; echo "[$(date +%H:%M:%S)] done $name exit=$rc" >> $S; return $rc; }
$PY -m py_compile experiments/classa_oracle.py || { echo "COMPILE FAIL" >> $S; exit 1; }
run smoke_oracle5 $PY experiments/classa_oracle.py --device mps --docs 1 --prefill 512 --probe 192 --split temporal --causal --prefix-cache --test-len 64 --anchor-window 32 || { echo "SMOKE FAIL" >> $S; exit 1; }
# original protocol re-run under the nan-robust aggregation (should reproduce 0.164/0.188/0.122)
run oracle_orig_p256_robust $PY experiments/classa_oracle.py --device mps --probe 256 --out runs/local_repro/oracle_orig_p256_robust.json
# decoupled deployment protocol: anchors from the last 128 fit positions; test = 128 positions after the cache; fit = W positions
for W in 128 256 512 1024 2048; do
  P=$((W+128))
  run oracle_dec_W${W} $PY experiments/classa_oracle.py --device mps --prefill 2560 --probe $P --split temporal --causal --prefix-cache --test-len 128 --anchor-window 128 --out runs/local_repro/oracle_dec_W${W}.json
done
run oracle_dec_W2048_pre4096 $PY experiments/classa_oracle.py --device mps --prefill 4096 --probe 2176 --split temporal --causal --prefix-cache --test-len 128 --anchor-window 128 --out runs/local_repro/oracle_dec_W2048_pre4096.json
# horizon dependence at W=512
run oracle_dec_W512_h64  $PY experiments/classa_oracle.py --device mps --prefill 2560 --probe 576  --split temporal --causal --prefix-cache --test-len 64  --anchor-window 128 --out runs/local_repro/oracle_dec_W512_h64.json
run oracle_dec_W512_h256 $PY experiments/classa_oracle.py --device mps --prefill 2560 --probe 768  --split temporal --causal --prefix-cache --test-len 256 --anchor-window 128 --out runs/local_repro/oracle_dec_W512_h256.json
echo "ROUND5 DONE $(date)" >> $S
