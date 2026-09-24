"""Render a reusable LongBench results table from per-sample score dumps.

`evaluation/tasks/longbench.py` writes one `persample_<task>_<method>.json` per
(task, method). This collects them into a paper-ready table -- per-task scores,
the macro average, and paired statistics against a reference method -- and
writes both Markdown and CSV so results accumulated over several runs can be
combined without re-running anything.

    python experiments/make_table.py runs/reference --ref h2o --out docs/table

Methods are matched on the exact string recorded inside each JSON, so a config
run twice with different kwargs stays distinguishable. Only tasks that every
listed method actually ran are used, keeping the comparison paired.
"""
import argparse
import csv
import glob
import json
import math
import os
import random
import statistics

# short, stable display names for the configurations used in the paper
# Display names follow the paper. VWS (value-weighted selection) is `otkv7;auto_regime=false`.
# Bare `otkv7` in runs made before commit 4735f18 (runs/modal/dev8k, dev8k_c02) carries the
# regime detector, so it is NOT VWS. The `otkv4` family is the earlier transport-merge configuration.
ALIASES = [
    ("baseline", "Dense (no compression)"),
    ("streamingllm", "StreamingLLM"),
    ("h2o;per_head=false", "H2O (global)"),
    ("h2o", "H2O (per-head)"),
    ("snapkv;observation_window=32;per_head=false", "SnapKV (global)"),
    ("snapkv;observation_window=32", "SnapKV (per-head)"),
    ("pyramidkv;observation_window=32;per_head=false", "PyramidKV (global)"),
    ("pyramidkv;observation_window=32", "PyramidKV (per-head)"),
    ("echokv;per_head=false", "EchoKV (global)"),
    ("echokv", "EchoKV (per-head)"),
    ("otkv7;auto_regime=false;layer_alloc=uniform", "VWS - water-filling"),
    ("otkv7;auto_regime=false;pool_kernel=0", "VWS - pooling"),
    ("otkv7;auto_regime=false;vnorm_tau=0", "VWS - value weighting"),
    ("otkv7;auto_regime=false", "VWS"),
    ("otkv7;auto_regime=true;auto_threshold=0.73", "VWS + regime detector"),
    ("otkv7;auto_regime=true", "VWS + regime detector"),
    ("otkv7;consolidate=true", "our consolidation (keys averaged) on VWS"),
    ("otkv7;regime_scope=global;layer_alloc=uniform", "VWS + regime detector (global) - water-filling"),
    ("otkv7;regime_scope=global;pool_kernel=0", "VWS + regime detector (global) - pooling"),
    ("otkv7;regime_scope=global;vnorm_tau=0", "VWS + regime detector (global) - value weighting"),
    ("otkv7;regime_scope=global;auto_regime=false", "VWS (qa forced, global scope)"),
    ("otkv7;regime_scope=global", "VWS + regime detector (global)"),
    ("otkv7;layer_alloc=uniform;pool_kernel=0;vnorm_tau=0", "transport-merge engine, selection only (+ detector)"),
    ("otkv7;layer_alloc=uniform", "VWS + regime detector - water-filling (pre-4735f18)"),
    ("otkv7;pool_kernel=0", "VWS + regime detector - pooling (pre-4735f18)"),
    ("otkv7;vnorm_tau=0", "VWS + regime detector - value weighting (pre-4735f18)"),
    ("otkv7", "VWS + regime detector (bare otkv7, pre-4735f18 default)"),
    ("snapkv;observation_window=32;per_head=false;pool_kernel=7", "SnapKV (global, pooled)"),
    ("otkv4;preset=auto;layer_budget=pyramid", "transport merge + regime detector + pyramid"),
    ("otkv4;preset=auto", "transport merge (auto regime)"),
    ("otkv4;preset=continuation", "transport merge (continuation preset)"),
    ("otkv4;select_scope=head", "transport merge (per-head selection)"),
    ("otkv4;merge=false", "transport-merge selector, merge off"),
    ("otkv4;layer_budget=pyramid", "transport merge + pyramid budget"),
    ("otkv4", "transport merge"),
    ("otkv", "transport merge, first version (broken)"),
]


def display(method):
    for key, name in ALIASES:          # longest/most specific patterns first
        if method == key:
            return name
    for key, name in ALIASES:
        if method.startswith(key):
            return name + (" *" if method != key else "")
    return method


def load(directory):
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, "persample_*.json"))):
        with open(path) as f:
            d = json.load(f)
        out.setdefault(d["method"], {})[d["task"]] = d["scores"]
    return out


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
        return 1.0
    k = min(pos, neg)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--ref", default=None, help="reference method for paired stats")
    ap.add_argument("--out", default=None, help="path prefix for .md and .csv output")
    ap.add_argument("--title", default="LongBench")
    args = ap.parse_args()

    data = load(args.dir)
    if not data:
        raise SystemExit(f"no persample_*.json in {args.dir}")

    tasks = sorted(set.intersection(*(set(v) for v in data.values())))
    methods = list(data)
    counts = {t: {len(data[m][t]) for m in methods} for t in tasks}
    bad = {t: c for t, c in counts.items() if len(c) > 1}
    if bad:
        raise SystemExit(f"sample counts differ across methods for {bad}; cannot pair")
    n_samples = sum(len(data[methods[0]][t]) for t in tasks)

    def pooled(m):
        return [x for t in tasks for x in data[m][t]]

    ref = args.ref
    if ref is None:
        ref = next((m for m in methods if m.startswith("snapkv")), methods[0])

    rows = []
    ref_vals = pooled(ref) if ref in data else None
    for m in methods:
        vals = pooled(m)
        row = {"method": display(m), "raw": m,
               "avg": 100 * statistics.mean(vals)}
        for t in tasks:
            row[t] = 100 * statistics.mean(data[m][t])
        if ref_vals is not None and m != ref:
            diffs = [a - b for a, b in zip(vals, ref_vals)]
            lo, hi = bootstrap_ci(diffs)
            row["delta"] = 100 * statistics.mean(diffs)
            row["ci"] = f"[{100 * lo:+.2f}, {100 * hi:+.2f}]"
            row["sig"] = "*" if (lo > 0 or hi < 0) else ""
            row["p"] = sign_test(diffs)
        rows.append(row)
    rows.sort(key=lambda r: -r["avg"])

    head = ["method"] + tasks + ["avg", "delta", "ci", "p"]
    md = [f"### {args.title} — {len(tasks)} tasks, {n_samples} paired samples "
          f"(reference: {display(ref)})", "",
          "| " + " | ".join(["method"] + [t[:12] for t in tasks] + ["**avg**", "Δ vs ref", "95% CI", "p"]) + " |",
          "|" + "---|" * (len(tasks) + 5)]
    for r in rows:
        cells = [r["method"]] + [f"{r[t]:.2f}" for t in tasks] + [f"**{r['avg']:.2f}**"]
        cells += [f"{r['delta']:+.2f}{r['sig']}" if "delta" in r else "(ref)",
                  r.get("ci", ""), f"{r['p']:.3f}" if "p" in r else ""]
        md.append("| " + " | ".join(cells) + " |")
    md += ["", "`*` = bootstrap 95 % CI on the paired difference excludes zero; "
               "`p` = two-sided sign test over samples where the methods differ."]
    text = "\n".join(md)
    print(text)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out + ".md", "w") as f:
            f.write(text + "\n")
        with open(args.out + ".csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["method", "raw"] + tasks + ["avg", "delta", "ci", "p"],
                               extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nwrote {args.out}.md and {args.out}.csv")


if __name__ == "__main__":
    main()
