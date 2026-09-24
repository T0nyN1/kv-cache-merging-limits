"""Pooled dev (Modal H200, 400 pairs) + held-out test (H100 cluster, 800 pairs) statistics for the
fixed-key construction (otkv8) against pooled SnapKV. Deterministic: seed 0, 200,000 bootstrap
resamples of the concatenated paired differences (the sample bootstrap of Table 2; with equal task sizes the
mean over samples equals the mean of task means), 200,000 sign-flip permutations, exact sign flip over the
eight task means.
Usage: python experiments/pooled_devtest.py   (from the repo root)"""
import glob, itertools, json, os
from math import comb
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP = "_snapkv_observation_window_32_per_head_false_pool_kernel_7"
OTKV8 = "_otkv8_observation_window_32_per_head_false_pool_kernel_7"
B = P = 200_000
rng = np.random.default_rng(0)

def load(d, suffix):
    out = {}
    for f in glob.glob(os.path.join(ROOT, d, "persample_*" + suffix)):
        J = json.load(open(f)); out[J["task"]] = np.array(J["scores"], dtype=float)
    return out

def stats(pairs, label):
    tasks = sorted(pairs)
    d = {t: (pairs[t][0] - pairs[t][1]) * 100 for t in tasks}
    tm = np.array([d[t].mean() for t in tasks]); delta = tm.mean()
    allD = np.concatenate([d[t] for t in tasks])
    idx = rng.integers(0, len(allD), (B, len(allD)))
    bs = allD[idx].mean(axis=1)                      # sample bootstrap over the concatenated pairs
    lo, hi = np.percentile(bs, [2.5, 97.5])
    cnt = 0
    for _ in range(P):
        cnt += abs(np.mean([(d[t] * rng.choice([-1, 1], len(d[t]))).mean() for t in tasks])) >= abs(delta) - 1e-12
    perm_s = (cnt + 1) / (P + 1)
    perm_t = sum(abs(np.mean(tm * np.array(sg))) >= abs(delta) - 1e-12
                 for sg in itertools.product([-1, 1], repeat=len(tasks))) / 2 ** len(tasks)
    w = int((allD > 0).sum()); l = int((allD < 0).sum())
    m = w + l; k = min(w, l); sign_p = min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / 2 ** m)
    print(f"{label}: n={len(allD)} delta={delta:+.3f} CI95=[{lo:+.3f},{hi:+.3f}] wins/losses={w}/{l} "
          f"sign_p={sign_p:.3f} perm_samples={perm_s:.4f} perm_tasks={perm_t:.4f} "
          f"tasks>0={int((tm>0).sum())}/{len(tasks)} changed={int((allD!=0).sum())}")

dev_m = load("runs/modal/dev8k_v8", OTKV8 + ".json"); dev_b = load("runs/modal/dev8k", SNAP + ".json")
dev = {t: (dev_m[t][:50], dev_b[t][:50]) for t in dev_m}
test_m = load("runs/h100_checks/test8k_v8", OTKV8 + "@50.json"); test_b = load("runs/h100_checks/base_test", SNAP + "@50.json")
test = {t: (test_m[t], test_b[t]) for t in test_m}
assert sorted(dev) == sorted(test) and len(dev) == 8
stats(dev, "dev (Modal H200, 400 pairs)")
stats(test, "test (H100, 800 pairs)")
stats({t: (np.concatenate([dev[t][0], test[t][0]]), np.concatenate([dev[t][1], test[t][1]])) for t in dev},
      "pooled dev+test (1,200 pairs; dev pairs H200/H200, test pairs H100/H100)")
