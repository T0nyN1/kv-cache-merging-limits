"""Unit checks for the KVMerger port (baselines/kvmerger.py).

1. Duplicate-run merge: if an evicted token's key/value equal its retained
   neighbour's, the pivot is unchanged (convex combination of equal vectors).
2. Distinct keys below the threshold are never merged (pure eviction).
3. A merged pivot equals the Gaussian-kernel convex combination computed by hand.
4. Cache/score channel lengths stay aligned through prefill + 10 decode prunes.

Run: python experiments/test_kvmerger.py
"""
import builtins
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_real_print = builtins.print


def _quiet(*a, **kw):
    if a and isinstance(a[0], str) and "[KV Monitor]" in a[0]:
        return
    _real_print(*a, **kw)


builtins.print = _quiet

from baselines.kvmerger import KVMergerCache  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))


def make(seq, heads=2, dim=8, budget=0.5, thr=0.75, **kw):
    torch.manual_seed(0)
    c = KVMergerCache(compression_size=budget, recent_size=2, sink_size=1, mode="prefill",
                      merge_threshold=thr, per_head=False, **kw)
    k = torch.randn(1, heads, seq, dim)
    v = torch.randn(1, heads, seq, dim)
    return c, k, v


def prefill(c, k, v, scores):
    c.update(k.clone(), v.clone(), 0, {})
    c.attention_scores[0] = scores.clone()
    c.finalize_prefill()


def test_duplicates_lossless():
    c, k, v = make(seq=12, budget=0.5)
    # middle = positions 1..9; make tokens 3 and 4 identical, keep 4 (high score), evict 3
    k[:, :, 3] = k[:, :, 4]
    v[:, :, 3] = v[:, :, 4]
    scores = torch.zeros(12)
    # middle = positions 1..9, middle budget = int(12*0.5) - 1 - 2 = 3: keep 4, 2, 6
    scores[4], scores[2], scores[6], scores[8] = 10.0, 9.0, 8.0, 1.0
    scores[3] = 2.0
    ref = v[:, :, 4].clone()
    prefill(c, k, v, scores)
    kk, vv = c._get_existing_cache(0)
    # find the row that came from position 4: its key equals k[:, :, 4]
    match = (kk - k[:, :, 4:5]).abs().sum(-1).sum(1)[0] < 1e-5
    check("identical evicted neighbour merged into pivot leaves the pivot unchanged",
          bool(match.any()) and torch.allclose(vv[0, :, match.nonzero()[0, 0]], ref[0], atol=1e-5))
    check("merge counted", c.merge_stats["merged_tokens"] >= 1, str(c.merge_stats))


def test_dissimilar_not_merged():
    c, k, v = make(seq=12, budget=0.5, thr=0.999)
    scores = torch.zeros(12)
    scores[4], scores[2], scores[6], scores[8] = 10.0, 9.0, 8.0, 1.0
    v0 = v.clone()
    prefill(c, k, v, scores)
    kk, vv = c._get_existing_cache(0)
    check("no merges above an unreachable threshold", c.merge_stats["merged_tokens"] == 0, str(c.merge_stats))
    # kept rows are untouched: sink 0, middle 2/4/6, recent 10/11
    kept_rows = [0, 2, 4, 6, 10, 11]
    check("kept rows untouched", torch.allclose(vv[0], v0[0, :, kept_rows], atol=1e-6))


def test_kernel_formula():
    c, k, v = make(seq=8, budget=0.625, thr=-1.0)  # everything linked; middle budget = 5 - 1 - 2 = 2
    scores = torch.zeros(8)
    scores[2], scores[4] = 10.0, 9.0               # middle = 1..5; kept 2 and 4 -> one group [1..5], pivot = 2
    prefill(c, k, v, scores)
    kk, vv = c._get_existing_cache(0)
    # hand computation for head 0: group members = pivot 2 plus evicted 1, 3, 5
    kf = k[0, 0]
    e = torch.tensor([1, 3, 5])
    d2 = ((kf[e] - kf[2]) ** 2).sum(-1)
    sig2 = d2.sum() / (len(e) + 1)
    g = torch.exp(-d2 / (2 * sig2))
    w_e = g / (1 + g.sum())
    w_p = 1 / (1 + g.sum())
    v_expect = w_p * v[0, 0, 2] + (w_e[:, None] * v[0, 0, e]).sum(0)
    # merged pivot row is the first middle row of the pruned cache (sink=1 row before it)
    check("pivot value equals the hand-computed Gaussian-kernel combination",
          torch.allclose(vv[0, 0, 1], v_expect, atol=1e-5), f"{vv[0,0,1][:3]} vs {v_expect[:3]}")


def test_alignment_through_decode():
    c, k, v = make(seq=60, budget=0.3, thr=0.0)
    scores = torch.rand(60)
    prefill(c, k, v, scores)
    for step in range(10):
        L = c._get_existing_cache(0)[0].shape[-2]
        c.current_attention_scores[0] = torch.rand(1, 2, 1, L + 1)
        kk = torch.randn(1, 2, 1, 8)
        c.update(kk, kk.clone(), 0, {})
    L = c._get_existing_cache(0)[0].shape[-2]
    check("scores aligned with cache after 10 decode prunes", c.attention_scores[0].shape[-1] == L)
    check("cache at budget", L <= int(60 * 0.3) + 1, str(L))


if __name__ == "__main__":
    print("test_duplicates_lossless");        test_duplicates_lossless()
    print("test_dissimilar_not_merged");      test_dissimilar_not_merged()
    print("test_kernel_formula");             test_kernel_formula()
    print("test_alignment_through_decode");   test_alignment_through_decode()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)
