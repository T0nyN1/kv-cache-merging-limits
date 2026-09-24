"""Paired comparison of LongBench methods from per-sample score dumps.

LongBench scores vary enormously across documents -- the between-sample standard
deviation is comparable to the whole gap between dense and a compressed cache.
Comparing unpaired means therefore cannot separate methods a point or two apart,
and reporting such a gap as an improvement would be wrong. Every method is run
on identical inputs, so per-sample differences can be paired, which removes the
document-difficulty variance and is the only way to get useful power out of a
few hundred samples.

Reads the `persample_<task>_<method>.json` files written by
`evaluation/tasks/longbench.py` and reports, per task and pooled across tasks:
the paired mean difference against a reference method, a bootstrap 95 % CI, and
a two-sided sign test over the samples where the two methods differ.
"""
import argparse
import glob
import json
import math
import os
import random
import statistics


def load(directory):
    """-> {method: {task: [scores]}}"""
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, "persample_*.json"))):
        with open(path) as f:
            d = json.load(f)
        out.setdefault(d["method"], {})[d["task"]] = d["scores"]
    return out


def pooled(per_task, tasks):
    vals = []
    for t in tasks:
        vals.extend(per_task[t])
    return vals


def bootstrap_ci(diffs, n=20000, alpha=0.05, seed=0):
    rng = random.Random(seed)
    k = len(diffs)
    means = sorted(sum(diffs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return means[int(alpha / 2 * n)], means[int((1 - alpha / 2) * n)]


def sign_test(diffs):
    pos = sum(1 for d in diffs if d > 1e-12)
    neg = sum(1 for d in diffs if d < -1e-12)
    n = pos + neg
    if n == 0:
        return 1.0, 0, 0
    k = min(pos, neg)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n), pos, neg


def short(method, width=44):
    return method if len(method) <= width else method[:width - 1] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="directory holding persample_*.json")
    ap.add_argument("--ref", default=None, help="reference method (default: a snapkv entry, else h2o)")
    ap.add_argument("--per_task", action="store_true", help="also print a per-task score matrix")
    args = ap.parse_args()

    data = load(args.dir)
    if not data:
        raise SystemExit(f"no persample_*.json found in {args.dir}")

    # only tasks every method actually ran, so pooling stays paired
    tasks = sorted(set.intersection(*(set(v) for v in data.values())))
    methods = [m for m in data if all(t in data[m] for t in tasks)]
    lengths = {t: {len(data[m][t]) for m in methods} for t in tasks}
    bad = {t: l for t, l in lengths.items() if len(l) > 1}
    if bad:
        raise SystemExit(f"sample counts differ across methods for {bad}; cannot pair")

    ref = args.ref
    if ref is None:
        ref = next((m for m in methods if m.startswith("snapkv")), None) or \
              next((m for m in methods if m == "h2o"), methods[0])
    if ref not in methods:
        raise SystemExit(f"reference {ref!r} not among {methods}")

    n = sum(len(data[ref][t]) for t in tasks)
    print(f"tasks  : {', '.join(tasks)}")
    print(f"pooled : {n} paired samples   reference = {ref}\n")

    if args.per_task:
        print(f"{'method':<46} " + " ".join(f"{t[:11]:>11}" for t in tasks))
        print("-" * (46 + 12 * len(tasks)))
        for m in sorted(methods, key=lambda m: -statistics.mean(pooled(data[m], tasks))):
            print(f"{short(method=m):<46} " +
                  " ".join(f"{100 * statistics.mean(data[m][t]):>11.2f}" for t in tasks))
        print()

    ref_vals = pooled(data[ref], tasks)
    print(f"{'method':<46} {'mean':>7} {'Δ vs ref':>9} {'95% CI':>19} {'win/loss':>10} {'p':>8}")
    print("-" * 104)
    rows = []
    for m in methods:
        vals = pooled(data[m], tasks)
        mean = statistics.mean(vals)
        if m == ref:
            rows.append((mean, m, None, None, None, None))
            continue
        diffs = [a - b for a, b in zip(vals, ref_vals)]
        lo, hi = bootstrap_ci(diffs)
        p, pos, neg = sign_test(diffs)
        rows.append((mean, m, statistics.mean(diffs), (lo, hi), (pos, neg), p))

    for mean, m, d, ci, wl, p in sorted(rows, key=lambda r: -r[0]):
        if ci is None:
            print(f"{short(m):<46} {mean * 100:>7.2f} {'(ref)':>9}")
        else:
            star = "*" if (ci[0] > 0 or ci[1] < 0) else " "
            print(f"{short(m):<46} {mean * 100:>7.2f} {d * 100:>+9.2f} "
                  f"[{ci[0] * 100:>+6.2f},{ci[1] * 100:>+6.2f}]{star} {wl[0]:>4}/{wl[1]:<5} {p:>8.4f}")
    print("\n* = bootstrap 95 % CI on the paired difference excludes zero")


if __name__ == "__main__":
    main()
