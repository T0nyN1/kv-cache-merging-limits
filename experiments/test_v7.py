"""Correctness tests for OT-KV v7 (pooled/value-weighted selection + cross-layer
water-filling).

What has to be true before any GPU money is spent:

1.  `maxpool_tokens` is an exact 'same'-length token-axis max pool.
2.  The water-filling allocation respects floors and caps, hands out exactly
    `n_layers * uniform_budget` slots, and moves budget from concentrated
    layers to diffuse ones.
3.  Budget parity: the total cache held across layers equals what the uniform
    baselines hold, so a v7 win can never be a budget-accounting artefact.
4.  Value weighting changes selection the way Lemma 1 says: a token with
    moderate attention but a large value beats one with slightly more attention
    and a tiny value; with vnorm_tau=0 the order flips back.
5.  Pooling keeps the neighbourhood of an isolated spike; without pooling the
    neighbours are dropped.
6.  The full prefill + decode lifecycle runs in per-head and global modes,
    scores stay aligned with the cache, and the cache never exceeds budget.
7.  SnapKV's pool_kernel ranks on pooled scores but writes back untransformed
    accumulations.

Run: python experiments/test_v7.py
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

from core.ot_kv_v7 import OTKVv7Cache, maxpool_tokens

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))


torch.manual_seed(0)


# ---------------------------------------------------------------------------
def test_maxpool():
    x = torch.tensor([0., 0., 5., 0., 0., 0., 1.])
    p = maxpool_tokens(x, 3)
    check("maxpool spreads a spike to its neighbours",
          torch.equal(p, torch.tensor([0., 5., 5., 5., 0., 1., 1.])), str(p))
    check("maxpool keeps length for even kernels",
          maxpool_tokens(torch.rand(2, 3, 11), 4).shape == (2, 3, 11))
    check("maxpool kernel<=1 is identity", maxpool_tokens(x, 1) is x)


# ---------------------------------------------------------------------------
def make_cache(**kw):
    kwargs = dict(compression_size=0.25, recent_size=8, sink_size=2, mode="prefill")
    kwargs.update(kw)
    return OTKVv7Cache(**kwargs)


def drive_prefill(cache, n_layers, seq, heads=2, dim=8, batch=1,
                  score_fn=None, value_fn=None):
    """Prefill `n_layers` with tagged K/V and controllable dual scores."""
    ks, vs = [], []
    for li in range(n_layers):
        k = torch.randn(batch, heads, seq, dim) * 0.01
        v = torch.randn(batch, heads, seq, dim) * 0.01
        pos = torch.arange(seq, dtype=torch.float32).view(1, 1, -1)
        k[..., 0] = pos
        v[..., 0] = pos
        if value_fn is not None:
            v = value_fn(li, v)
        ks.append(k)
        vs.append(v)
        cache.update(k.clone(), v.clone(), li, {})
        full = torch.rand(batch, heads, seq) + 0.1
        win = torch.rand(batch, heads, seq) + 0.1
        if score_fn is not None:
            full, win = score_fn(li, full, win)
        if cache._dual:
            cache.attention_scores[li] = torch.stack([full, win])
        elif cache.per_head:
            cache.attention_scores[li] = full
        else:
            cache.attention_scores[li] = full[0].sum(0)
    cache.finalize_prefill()
    return ks, vs


def cache_len(cache, li):
    return cache._get_existing_cache(li)[0].shape[-2]


# ---------------------------------------------------------------------------
def test_allocation_and_parity():
    n_layers, seq = 4, 256
    torch.manual_seed(1)

    def score_fn(li, full, win):
        # layer 0: diffuse window scores; layer 3: all mass on 4 tokens
        if li == 0:
            win = torch.ones_like(win)
        if li == 3:
            win = torch.full_like(win, 1e-4)
            win[..., 60:64] = 50.0
        return full, win

    cache = make_cache(auto_regime=False)     # force qa regime: window selects
    drive_prefill(cache, n_layers, seq, score_fn=score_fn)

    alloc = cache._alloc
    uniform = cache.middle_budget
    check("allocation exists", alloc is not None and len(alloc) == n_layers)
    if alloc:
        total = sum(alloc.values())
        check("allocation hands out exactly n_layers * uniform slots",
              total == uniform * n_layers, f"{total} vs {uniform * n_layers}")
        floor = int(uniform * cache.alloc_floor)
        cap = int(torch.tensor(uniform * cache.alloc_cap).ceil())
        check("floors respected", all(a >= floor for a in alloc.values()), str(alloc))
        check("caps respected", all(a <= cap for a in alloc.values()), str(alloc))
        check("diffuse layer draws more budget than concentrated layer",
              alloc[0] > alloc[3], str(alloc))

    # parity: total physical cache across layers == what uniform baselines hold,
    # minus the reserve v7 deliberately errs against itself with
    total_len = sum(cache_len(cache, li) for li in range(n_layers))
    reserve = sum(cache._reserve(li, seq) for li in range(n_layers))
    expected = n_layers * (cache.sink_size + cache.recent_size + uniform) - reserve
    check("total cache equals uniform-budget parity minus reserve",
          total_len == expected, f"{total_len} vs {expected}")


def test_uniform_and_pyramid_fallbacks():
    for mode, name in (("uniform", "uniform"), ("pyramid", "pyramid")):
        cache = make_cache(layer_alloc=mode, auto_regime=False)
        drive_prefill(cache, 3, 200)
        check(f"layer_alloc={name} leaves _alloc unset", cache._alloc is None)
        lengths = [cache_len(cache, li) for li in range(3)]
        if mode == "uniform":
            check("uniform mode: every layer same length", len(set(lengths)) == 1, str(lengths))
        else:
            check("pyramid mode: early layer holds more than late", lengths[0] > lengths[2], str(lengths))


# ---------------------------------------------------------------------------
def test_value_weighting():
    seq = 128
    torch.manual_seed(2)

    def score_fn(li, full, win):
        win = torch.full_like(win, 1e-3)
        win[..., 40] = 1.00        # slightly higher attention, tiny value
        win[..., 80] = 0.90        # slightly lower attention, huge value
        return full, win

    def value_fn(li, v):
        v = v * 0.01
        v[..., 40, :] = 0.001
        v[..., 80, :] = 5.0
        return v

    picks = {}
    for tau in (0.0, 1.0):
        cache = make_cache(compression_size=0.1, auto_regime=False, pool_kernel=0,
                           vnorm_tau=tau, layer_alloc="uniform", recent_size=8, sink_size=2)
        drive_prefill(cache, 1, seq, score_fn=score_fn, value_fn=value_fn)
        k = cache._get_existing_cache(0)[0]
        kept = set(k[0, 0, :, 0].long().tolist())
        picks[tau] = kept
    check("vnorm_tau=1 keeps the large-value token", 80 in picks[1.0], str(sorted(picks[1.0])))
    check("vnorm_tau=0 ranks by attention alone", 40 in picks[0.0], str(sorted(picks[0.0])))


def test_pooling_keeps_neighbourhood():
    seq = 128
    torch.manual_seed(3)

    def score_fn(li, full, win):
        win = torch.full_like(win, 1e-3)
        win[..., 64] = 100.0       # one isolated spike
        return full, win

    kept = {}
    for kernel in (0, 7):
        cache = make_cache(compression_size=0.15, auto_regime=False, pool_kernel=kernel,
                           vnorm_tau=0.0, layer_alloc="uniform")
        drive_prefill(cache, 1, seq, score_fn=score_fn)
        k = cache._get_existing_cache(0)[0]
        kept[kernel] = set(k[0, 0, :, 0].long().tolist())
    neigh = {62, 63, 64, 65, 66}
    check("pooling keeps the spike's neighbourhood",
          neigh.issubset(kept[7]), str(sorted(kept[7])[:20]))
    check("no pooling keeps the spike but not the neighbourhood",
          64 in kept[0] and not neigh.issubset(kept[0]), str(sorted(kept[0])[:20]))


# ---------------------------------------------------------------------------
def drive_decode_steps(cache, n_layers, steps, heads=2, dim=8, batch=1):
    seq_next = cache.get_seq_length()
    for _ in range(steps):
        for li in range(n_layers):
            L = cache_len(cache, li)
            full = torch.rand(batch, heads, L + 1)
            win = torch.rand(batch, heads, L + 1)
            if cache._dual:
                cache.current_attention_scores[li] = torch.stack([full, win])
            elif cache.per_head:
                cache.current_attention_scores[li] = full
            else:
                cache.current_attention_scores[li] = full[0].sum(0)
            k = torch.randn(batch, heads, 1, dim) * 0.01
            v = torch.randn(batch, heads, 1, dim) * 0.01
            k[..., 0] = float(seq_next)
            v[..., 0] = float(seq_next)
            cache.update(k, v, li, {})
        seq_next += 1


def test_lifecycle(per_head):
    n_layers, seq, steps = 3, 200, 24
    cache = make_cache(per_head=per_head)
    drive_prefill(cache, n_layers, seq)
    label = "per-head" if per_head else "global"

    ok_align, ok_budget = True, True
    drive_decode_steps(cache, n_layers, steps)
    for li in range(n_layers):
        L = cache_len(cache, li)
        s = cache.attention_scores[li]
        if s.shape[-1] != L:
            ok_align = False
        budget_l = cache.get_middle_budget(li, seq) + cache.sink_size + cache.recent_size
        if L > budget_l + cache.compress_interval:
            ok_budget = False
    check(f"[{label}] scores stay aligned with cache through decode", ok_align)
    check(f"[{label}] cache never exceeds budget + interval", ok_budget)
    check(f"[{label}] regimes recorded for every layer",
          len(cache._layer_regime) == n_layers, str(cache._layer_regime))


def test_lifecycle_merge_ablation():
    cache = make_cache(merge=True)             # the ablation config must still run
    drive_prefill(cache, 2, 160)
    drive_decode_steps(cache, 2, 10)
    check("merge=true ablation runs through decode",
          all(cache.attention_scores[li].shape[-1] == cache_len(cache, li) for li in range(2)))


def test_continuation_regime_path():
    cache = make_cache(observation_window=None, split_scores=False, auto_regime=False)
    drive_prefill(cache, 2, 160)
    drive_decode_steps(cache, 2, 10)
    check("continuation path (no window) runs and prunes",
          all(cache_len(cache, li) < 160 for li in range(2)))
    check("continuation regime recorded",
          all(r == "continuation" for r in cache._layer_regime.values()), str(cache._layer_regime))


# ---------------------------------------------------------------------------
def test_snapkv_pooling():
    from baselines.snapkv import SnapKVCache
    seq = 128
    scores = torch.full((seq,), 1e-3)
    scores[64] = 100.0

    kept = {}
    for kernel in (0, 7):
        cache = SnapKVCache(compression_size=0.15, recent_size=8, sink_size=2,
                            mode="prefill", per_head=False, observation_window=32,
                            pool_kernel=kernel)
        k = torch.randn(1, 2, seq, 8) * 0.01
        v = torch.randn(1, 2, seq, 8) * 0.01
        pos = torch.arange(seq, dtype=torch.float32).view(1, 1, -1)
        k[..., 0] = pos
        cache.update(k, v, 0, {})
        cache.attention_scores[0] = scores.clone()
        cache.finalize_prefill()
        kk = cache._get_existing_cache(0)[0]
        kept[kernel] = set(kk[0, 0, :, 0].long().tolist())
        if kernel == 7:
            # written-back scores are the untransformed accumulation
            s = cache.attention_scores[0]
            check("snapkv writeback keeps raw scores (max is the spike, not the pool)",
                  int((s == 100.0).sum()) == 1, str(s.topk(5).values))
    check("snapkv pool_kernel=7 keeps the spike's neighbours",
          {62, 63, 64, 65, 66}.issubset(kept[7]), str(sorted(kept[7])[:20]))
    check("snapkv pool_kernel=0 does not", not {62, 66}.issubset(kept[0]))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("test_maxpool");                     test_maxpool()
    print("test_allocation_and_parity");       test_allocation_and_parity()
    print("test_uniform_and_pyramid_fallbacks"); test_uniform_and_pyramid_fallbacks()
    print("test_value_weighting");             test_value_weighting()
    print("test_pooling_keeps_neighbourhood"); test_pooling_keeps_neighbourhood()
    print("test_lifecycle per-head");          test_lifecycle(True)
    print("test_lifecycle global");            test_lifecycle(False)
    print("test_lifecycle_merge_ablation");    test_lifecycle_merge_ablation()
    print("test_continuation_regime_path");    test_continuation_regime_path()
    print("test_snapkv_pooling");              test_snapkv_pooling()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)
