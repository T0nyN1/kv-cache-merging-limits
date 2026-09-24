"""Llama-3.1-8B-Instruct dev-split rows of Table 9 (tab:qwen): dense, VWS and PyramidKV against pooled SnapKV,
from runs/modal/dev8k (first 50 samples of each of the 8 English tasks, 400 pairs). Seed 0, NumPy, 200,000 bootstrap
draws over the concatenated pairs (the Table 2 convention), exact sign flip over the 8 task means.
VWS is the `otkv7;auto_regime=false` artifact, not the historical bare `otkv7`.
Usage (repo root): python experiments/table9_llama_dev.py"""
import glob, itertools, json, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "_snapkv_observation_window_32_per_head_false_pool_kernel_7.json"
ROWS = [("dense", "_baseline.json"), ("VWS", "_otkv7_auto_regime_false.json"),
        ("PyramidKV", "_pyramidkv_observation_window_32_per_head_false.json")]
rng = np.random.default_rng(0)
def load(suf):
    out = {}
    for f in glob.glob(os.path.join(ROOT, "runs/modal/dev8k", "persample_*" + suf)):
        J = json.load(open(f)); out[J["task"]] = np.array(J["scores"], float)[:50]
    return out
base = load(BASE); tasks = sorted(base); assert len(tasks) == 8
for label, suf in ROWS:
    m = load(suf); assert sorted(m) == tasks, label
    d = {t: (m[t] - base[t]) * 100 for t in tasks}
    tm = np.array([d[t].mean() for t in tasks]); allD = np.concatenate([d[t] for t in tasks])
    bs = allD[rng.integers(0, len(allD), (200_000, len(allD)))].mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
    pt = sum(abs(np.mean(tm * np.array(s))) >= abs(tm.mean()) - 1e-12 for s in itertools.product([-1, 1], repeat=8)) / 256
    print(f"{label}: {tm.mean():+.2f} [{lo:+.2f},{hi:+.2f}]  perm_t={pt:.3f}  (n={len(allD)})")
