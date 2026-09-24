"""Correctness tests for per-head (multi-head) KV compression.

Per-head compression is what makes the comparison between methods fair -- every
method must keep the same number of tokens per KV head -- so the mechanism needs
to be verified rather than assumed. The tests check, for every registered cache:

1. each KV head really selects its own token subset (not silently shared);
2. the surviving K/V rows for head h are exactly the rows head h selected
   (tokens are tagged with their position so this is checkable);
3. the score tensor stays aligned with the cache after every compression, so
   later decode-time compressions score the right tokens;
4. GQA query heads are folded onto the KV head they actually share, matching
   `repeat_kv`;
5. per_head=True and per_head=False keep the same total cache length, i.e. the
   budget means the same thing in both modes;
6. every method runs in both modes at all.

Run: python experiments/test_per_head.py
"""
import os
import sys

import builtins

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_real_print = builtins.print


def _quiet(*a, **kw):
    if a and isinstance(a[0], str) and "[KV Monitor]" in a[0]:
        return
    _real_print(*a, **kw)


builtins.print = _quiet

from evaluation.models.base_cache import BaseCompressCache

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))


def make_cache(cls, per_head, **kw):
    kwargs = dict(compression_size=0.5, recent_size=4, sink_size=2, mode="prefill", per_head=per_head)
    kwargs.update(kw)
    return cls(**kwargs)


def tagged_kv(batch, heads, seq, dim):
    """K/V whose first channel is the token's original position, so surviving
    rows can be traced back to where they came from."""
    k = torch.randn(batch, heads, seq, dim) * 0.01
    v = torch.randn(batch, heads, seq, dim) * 0.01
    pos = torch.arange(seq, dtype=torch.float32).view(1, 1, -1)
    k[..., 0] = pos
    v[..., 0] = pos
    return k, v


def drive_prefill(cache, k, v, scores, n_layers=1):
    """Run one prefill through the cache with a chosen per-head score tensor."""
    for li in range(n_layers):
        cache.update(k.clone(), v.clone(), li, {})
        cache.attention_scores[li] = scores.clone() if cache.per_head else scores[0, 0].clone()
    cache.on_prefill_end()
    return cache


# ---------------------------------------------------------------------------
def test_selection_is_per_head():
    from baselines.h2o import H2OCache
    from core.ot_kv_v3 import OTKVv3Cache

    batch, heads, seq, dim = 1, 4, 40, 8
    k, v = tagged_kv(batch, heads, seq, dim)

    # each head prefers a disjoint band of the middle region
    scores = torch.rand(batch, heads, seq) * 0.01
    bands = [(2, 12), (12, 22), (22, 32), (26, 36)]
    for h, (lo, hi) in enumerate(bands):
        scores[0, h, lo:hi] += 10.0

    for cls, name, extra in ((H2OCache, "H2O", {}),
                             (OTKVv3Cache, "OT-KV v3", {"merge": False})):
        cache = make_cache(cls, True, **extra)
        drive_prefill(cache, k, v, scores)

        kept_k = cache.layers[0].keys
        got = [set(kept_k[0, h, :, 0].round().long().tolist()) for h in range(heads)]

        distinct = len({frozenset(g) for g in got}) == heads
        check(f"{name}: every KV head keeps a different token subset", distinct,
              f"subsets={[sorted(g)[:6] for g in got]}")

        # The middle budget (14) is wider than a band (10), so a head must keep
        # its whole band plus a few low-score extras -- assert containment, not
        # equality.
        for h, (lo, hi) in enumerate(bands):
            band = set(range(lo, hi))
            check(f"{name}: head {h} kept its entire high-score band", band <= got[h],
                  f"missing={sorted(band - got[h])}")

        # sink and recent must survive untouched in every head
        for h in range(heads):
            has_sink = all(t in got[h] for t in range(cache.sink_size))
            has_recent = all(t in got[h] for t in range(seq - cache.recent_size, seq))
            check(f"{name}: head {h} kept sink+recent verbatim", has_sink and has_recent)


def test_kv_rows_match_selection():
    """The surviving K row and V row for a head must come from the same token."""
    from baselines.h2o import H2OCache
    from baselines.snapkv import SnapKVCache
    from baselines.pyramidkv import PyramidKVCache
    from baselines.echokv import EchoKVCache

    batch, heads, seq, dim = 1, 4, 40, 8
    k, v = tagged_kv(batch, heads, seq, dim)
    scores = torch.rand(batch, heads, seq)

    for cls, name in ((H2OCache, "H2O"), (SnapKVCache, "SnapKV"),
                      (PyramidKVCache, "PyramidKV"), (EchoKVCache, "EchoKV")):
        cache = make_cache(cls, True)
        drive_prefill(cache, k, v, scores)
        kk, vv = cache.layers[0].keys, cache.layers[0].values
        aligned = torch.allclose(kk[..., 0], vv[..., 0])
        check(f"{name}: K and V rows stay paired per head", aligned)

        # tags must be a subset of the original positions, strictly increasing
        tags = kk[0, :, :, 0].round().long()
        monotone = bool((tags[:, 1:] > tags[:, :-1]).all())
        in_range = bool(((tags >= 0) & (tags < seq)).all())
        check(f"{name}: kept positions are original and in causal order", monotone and in_range)


def test_scores_stay_aligned():
    """After compression, score[h, i] must describe cache row i of head h."""
    from baselines.h2o import H2OCache
    from core.ot_kv_v3 import OTKVv3Cache

    batch, heads, seq, dim = 1, 4, 48, 8
    k, v = tagged_kv(batch, heads, seq, dim)
    scores = torch.rand(batch, heads, seq) * 5

    for cls, name, extra in ((H2OCache, "H2O", {}),
                             (OTKVv3Cache, "OT-KV v3", {"merge": False})):
        cache = make_cache(cls, True, **extra)
        drive_prefill(cache, k, v, scores)

        kept_tags = cache.layers[0].keys[0, :, :, 0].round().long()
        kept_scores = cache.attention_scores[0][0]
        expected = torch.stack([scores[0, h, kept_tags[h]] for h in range(heads)])

        same_len = kept_scores.shape[-1] == kept_tags.shape[-1]
        check(f"{name}: score tensor length matches cache length", same_len,
              f"{kept_scores.shape[-1]} vs {kept_tags.shape[-1]}")
        if same_len:
            check(f"{name}: score[h,i] belongs to cache row i of head h",
                  torch.allclose(kept_scores, expected, atol=1e-4),
                  f"max err {(kept_scores - expected).abs().max():.4f}")


def test_gqa_head_mapping():
    """Query heads must fold onto the KV head repeat_kv would pair them with."""
    n_kv, n_rep, k_len = 4, 3, 7
    n_q = n_kv * n_rep

    # attention where query head q puts all its mass on a value identifying q
    scores = torch.zeros(1, n_q, k_len)
    for q in range(n_q):
        scores[0, q, 0] = q + 1.0

    cache = make_cache(__import__("baselines.h2o", fromlist=["H2OCache"]).H2OCache, True)
    folded = cache._attention_to_token_scores(scores, (1, n_kv, k_len))

    # repeat_kv expands kv head j into query heads [j*n_rep, (j+1)*n_rep)
    expected = torch.tensor([[sum(j * n_rep + r + 1.0 for r in range(n_rep)) for j in range(n_kv)]])
    check("GQA: query heads fold onto the KV head repeat_kv pairs them with",
          torch.allclose(folded[0, :, 0], expected[0]),
          f"got {folded[0, :, 0].tolist()} want {expected[0].tolist()}")


def test_budget_parity_and_both_modes():
    """per_head True/False must keep the same cache length, and both must run."""
    from baselines.h2o import H2OCache
    from baselines.snapkv import SnapKVCache
    from baselines.pyramidkv import PyramidKVCache
    from baselines.echokv import EchoKVCache
    from core.ot_kv_v3 import OTKVv3Cache
    from core.ot_kv import OTKVCache

    batch, heads, seq, dim = 1, 4, 40, 8
    k, v = tagged_kv(batch, heads, seq, dim)
    scores = torch.rand(batch, heads, seq)

    cases = [(H2OCache, "H2O", {}), (SnapKVCache, "SnapKV", {}),
             (PyramidKVCache, "PyramidKV", {}), (EchoKVCache, "EchoKV", {}),
             (OTKVv3Cache, "OT-KV v3", {}), (OTKVCache, "OT-KV v2", {})]

    for cls, name, extra in cases:
        lengths = {}
        for per_head in (True, False):
            try:
                cache = make_cache(cls, per_head, **extra)
                drive_prefill(cache, k, v, scores)
                lengths[per_head] = cache.layers[0].keys.shape[-2]
            except Exception as exc:
                lengths[per_head] = f"{type(exc).__name__}: {exc}"
        ok_both = all(isinstance(x, int) for x in lengths.values())
        check(f"{name}: runs in both per_head modes", ok_both, f"{lengths}")
        if ok_both:
            check(f"{name}: same cache length in both modes (budget is comparable)",
                  lengths[True] == lengths[False], f"{lengths}")


def test_decode_compression_keeps_alignment():
    """Several decode-time compressions must not desynchronise scores and cache."""
    from core.ot_kv_v3 import OTKVv3Cache

    batch, heads, seq, dim = 1, 4, 40, 8
    k, v = tagged_kv(batch, heads, seq, dim)
    scores = torch.rand(batch, heads, seq)

    cache = make_cache(OTKVv3Cache, True, compress_interval=4, merge=True)
    drive_prefill(cache, k, v, scores)

    ok = True
    for step in range(20):
        nk = torch.randn(batch, heads, 1, dim) * 0.01
        nv = torch.randn(batch, heads, 1, dim) * 0.01
        nk[..., 0] = seq + step
        nv[..., 0] = seq + step
        cache.update(nk, nv, 0, {})
        cache.current_attention_scores[0] = torch.rand(batch, heads, cache.layers[0].keys.shape[-2])
        if cache.attention_scores[0].shape[-1] != cache.layers[0].keys.shape[-2]:
            ok = False
            break
    check("OT-KV v3: scores stay length-aligned across 20 decode steps", ok,
          f"scores={cache.attention_scores[0].shape[-1]} cache={cache.layers[0].keys.shape[-2]}")


if __name__ == "__main__":
    torch.manual_seed(0)
    print("\n[1] per-head selection is genuinely per head")
    test_selection_is_per_head()
    print("\n[2] surviving K/V rows match the selection")
    test_kv_rows_match_selection()
    print("\n[3] scores stay aligned with the cache")
    test_scores_stay_aligned()
    print("\n[4] GQA query-head -> KV-head folding")
    test_gqa_head_mapping()
    print("\n[5] budget parity between per_head modes")
    test_budget_parity_and_both_modes()
    print("\n[6] decode-time compression alignment")
    test_decode_compression_keeps_alignment()

    print(f"\n{'=' * 70}\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failures:")
        for f in FAIL:
            print(f"  - {f}")
    sys.exit(1 if FAIL else 0)
