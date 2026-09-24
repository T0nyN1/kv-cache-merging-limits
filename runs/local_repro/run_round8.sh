#!/bin/bash
PY=~/anaconda3/envs/ot-kv/bin/python
cd "."
export PYTORCH_ENABLE_MPS_FALLBACK=1
S=runs/local_repro/STATUS
LL=models/Llama-3.1-8B-Instruct
echo "ROUND8 START $(date)" >> $S
run() { name=$1; shift; echo "[$(date +%H:%M:%S)] start $name" >> $S; "$@" > runs/local_repro/$name.log 2>&1; rc=$?; echo "[$(date +%H:%M:%S)] done $name exit=$rc" >> $S; return $rc; }
$PY -m py_compile experiments/sigma_decode.py experiments/classa_oracle_decode.py experiments/sigma_perhead_derot.py experiments/sigma_nonrope.py experiments/oracle_selector_abs.py experiments/sigma_drift_null.py || { echo "COMPILE FAIL" >> $S; exit 1; }
run perhead_llama  $PY experiments/sigma_perhead_derot.py --model $LL --dtype bfloat16 --layer-stride 4 --out runs/local_repro/perhead_derot_llama.json
run perhead_qwen3  $PY experiments/sigma_perhead_derot.py --model Qwen/Qwen3-1.7B --dtype float32 --layer-stride 2 --out runs/local_repro/perhead_derot_qwen3.json
run nonrope_gpt2m  $PY experiments/sigma_nonrope.py --model gpt2-medium --out runs/local_repro/sigma_nonrope_gpt2medium.json
run drift_null_qwen3 $PY experiments/sigma_drift_null.py runs/local_repro/sigma_drift_null_qwen3.json
run sigma_decode_v2 $PY experiments/sigma_decode.py --device mps --prompts 6 --no-repeat-ngram 3 --out runs/local_repro/sigma_decode_qwen3_v2.json
run oracle_decode_b005 $PY experiments/classa_oracle_decode.py --device mps --budget 0.05 --new 256 --no-repeat-ngram 3 --out runs/local_repro/classa_oracle_decode_qwen3_b005.json
run oracle_decode_b015 $PY experiments/classa_oracle_decode.py --device mps --budget 0.15 --new 256 --no-repeat-ngram 3 --out runs/local_repro/classa_oracle_decode_qwen3_b015.json
run oracle_selector_abs $PY experiments/oracle_selector_abs.py runs/local_repro/oracle_selector_abs_qwen3.json
echo "ROUND8 DONE $(date)" >> $S
