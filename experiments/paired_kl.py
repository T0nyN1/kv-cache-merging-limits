"""Paired comparison of methods on the per-document KL/NLL from bench.py.

`bench.py` runs every method over identical documents, so the comparison should
be paired on the document. Positions within one document are strongly
autocorrelated, so the document -- not the token position -- is the unit of
analysis; treating positions as independent would inflate significance by more
than an order of magnitude.

Reports the paired mean difference, a bootstrap 95 % CI over documents, and how
many documents each method wins.
"""
import argparse
import json
import math
import random
import statistics


def bootstrap_ci(diffs, n=20000, alpha=0.05, seed=0):
    rng = random.Random(seed)
    k = len(diffs)
    means = sorted(sum(diffs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return means[int(alpha / 2 * n)], means[int((1 - alpha / 2) * n)]


def sign_p(diffs):
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0, 0, 0
    k = min(pos, neg)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n), pos, neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--ref", default="h2o")
    ap.add_argument("--metric", default="per_doc_kl", choices=["per_doc_kl", "per_doc_nll"])
    args = ap.parse_args()

    rows = json.load(open(args.json_path))["rows"]
    budgets = sorted({r["budget"] for r in rows if r["budget"] is not None})

    for budget in budgets:
        at = {r["method"]: r for r in rows if r["budget"] == budget}
        if args.ref not in at:
            print(f"budget {budget}: reference {args.ref!r} missing, have {list(at)}")
            continue
        ref = at[args.ref][args.metric]
        n = len(ref)
        print(f"\n=== budget {budget}   metric {args.metric}   n = {n} documents "
              f"(lower is better; reference {args.ref} = {statistics.mean(ref):.5f}) ===")
        print(f"{'method':<18} {'mean':>9} {'Δ vs ref':>10} {'Δ %':>7} {'95% CI on Δ':>22} {'wins':>7} {'p':>7}")
        print("-" * 90)
        for method, r in sorted(at.items(), key=lambda kv: statistics.mean(kv[1][args.metric])):
            vals = r[args.metric]
            mean = statistics.mean(vals)
            if method == args.ref:
                print(f"{method:<18} {mean:>9.5f} {'(ref)':>10}")
                continue
            diffs = [a - b for a, b in zip(vals, ref)]
            d = statistics.mean(diffs)
            lo, hi = bootstrap_ci(diffs)
            p, pos, neg = sign_p(diffs)
            star = "*" if (lo > 0 or hi < 0) else " "
            print(f"{method:<18} {mean:>9.5f} {d:>+10.5f} {100 * d / statistics.mean(ref):>+6.1f}% "
                  f"[{lo:>+9.5f},{hi:>+9.5f}]{star} {neg:>3}/{n:<3} {p:>7.4f}")
    print("\n* = bootstrap 95 % CI excludes zero;  wins = documents where the method beats the reference")


if __name__ == "__main__":
    main()
