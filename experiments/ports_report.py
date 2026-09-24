"""Every inference the matched-protocol table quotes, for one (method, base) pair.

The KeepKV and SelKV rows are paired against different bases -- the two operator
rows against the existing pooled-SnapKV run, the own-selector row against its own
merge-off run -- so the reference is given explicitly rather than assumed.

    python experiments/ports_report.py \
        --dir runs/modal/ports_a --suffix _keepkv.json \
        --ref-dir runs/modal/dev8k \
        --ref-suffix _snapkv_observation_window_32_per_head_false_pool_kernel_7.json

Prints the paired mean, the per-sample bootstrap CI, the task-clustered bootstrap
CI, a sample-level permutation p, the exact sign-flip p over the task means, the
sign test over samples, and the per-task table. Also summarises
`merge_stats.jsonl` when it sits next to the per-sample files.
"""
import argparse
import glob
import itertools
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats as _s

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_pairs(d, suffix, rd, ref_suffix):
    per, names = [], []
    for f in sorted(glob.glob(os.path.join(d, "persample_*" + suffix))):
        J = json.load(open(f))
        t = J["task"]
        rf = os.path.join(rd, f"persample_{t}{ref_suffix}")
        if not os.path.exists(rf):
            print(f"  [skip] {t}: no reference {os.path.basename(rf)}")
            continue
        R = json.load(open(rf))
        x, r = np.array(J["scores"]), np.array(R["scores"])
        n = min(len(x), len(r))
        if len(x) != len(r):
            print(f"  [warn] {t}: {len(x)} vs {len(r)} samples, pairing the first {n}")
        per.append(100 * (x[:n] - r[:n]))
        names.append(t)
    return per, names


def report(per, names, draws=200000, seed=0):
    dd = np.concatenate(per)
    means = np.array([x.mean() for x in per])
    rng = np.random.default_rng(seed)
    sb = np.array([rng.choice(dd, len(dd)).mean() for _ in range(20000)])
    cb = np.array([means[rng.integers(0, len(means), len(means))].mean() for _ in range(draws)])
    perm = np.array([(dd * rng.choice([-1, 1], len(dd))).mean() for _ in range(draws)])
    flips = np.array([np.array(f) for f in itertools.product([-1, 1], repeat=len(means))])
    task_p = float((np.abs(flips @ means) / len(means) >= abs(means.mean()) - 1e-12).mean())
    nz = dd[dd != 0]
    wins, losses = int((nz > 0).sum()), int((nz < 0).sum())
    return {
        "mean": dd.mean(), "n": len(dd), "n_tasks": len(names),
        "lo": np.percentile(sb, 2.5), "hi": np.percentile(sb, 97.5),
        "task_lo": np.percentile(cb, 2.5), "task_hi": np.percentile(cb, 97.5),
        "perm_p": float((np.abs(perm) >= abs(dd.mean())).mean()), "task_p": task_p,
        "wins": wins, "losses": losses,
        "sign_p": float(_s.binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0,
        "n_pos": int((means > 0).sum()), "task_means": dict(zip(names, means)),
    }


def merge_stats(directory, method=None):
    path = os.path.join(directory, "merge_stats.jsonl")
    if not os.path.exists(path):
        return {}
    agg = defaultdict(lambda: defaultdict(float))
    count = defaultdict(int)
    for line in open(path):
        rec = json.loads(line)
        if method is not None and rec["method"] != method:
            continue
        count[rec["method"]] += 1
        for k, v in rec["stats"].items():
            if k in ("lam_absmax_seen", "votes_max", "logR_max"):
                agg[rec["method"]][k] = max(agg[rec["method"]][k], v)
            else:
                agg[rec["method"]][k] += v
    return {m: (dict(a), count[m]) for m, a in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--ref-dir", dest="ref_dir", required=True)
    ap.add_argument("--ref-suffix", dest="ref_suffix", required=True)
    ap.add_argument("--method", default=None, help="method string for the merge-statistics summary")
    a = ap.parse_args()
    d = a.dir if os.path.isabs(a.dir) else os.path.join(ROOT, a.dir)
    rd = a.ref_dir if os.path.isabs(a.ref_dir) else os.path.join(ROOT, a.ref_dir)

    per, names = load_pairs(d, a.suffix, rd, a.ref_suffix)
    if not per:
        raise SystemExit("no paired tasks found")
    r = report(per, names)
    print(f"\n{a.suffix}  vs  {a.ref_suffix}")
    print(f"  paired mean      {r['mean']:+.2f}  over {r['n']} samples, {r['n_tasks']} tasks")
    print(f"  sample bootstrap [{r['lo']:+.2f},{r['hi']:+.2f}]   task-clustered [{r['task_lo']:+.2f},{r['task_hi']:+.2f}]")
    print(f"  permutation p    samples {r['perm_p']:.3f}   exact sign-flip over task means {r['task_p']:.3f}")
    print(f"  sign test        {r['wins']}/{r['losses']}  p={r['sign_p']:.3f}   tasks positive {r['n_pos']}/{r['n_tasks']}")
    print("  per task         " + "  ".join(f"{t} {m:+.2f}" for t, m in r["task_means"].items()))
    for m, (st, n) in merge_stats(d, a.method).items():
        line = " ".join(f"{k} {v:.4g}" for k, v in sorted(st.items()))
        print(f"  stats [{m}] over {n} samples: {line}")


if __name__ == "__main__":
    main()
