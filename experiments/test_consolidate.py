"""Correctness tests for compensated pair consolidation (core/consolidate.py).

The claims that must hold before any GPU money:

1. Merging an *identical* adjacent pair (same k, same v, same attention rows)
   is exactly lossless for every query once the merged slot carries bias
   log 2 — the softmax attention output over the consolidated cache equals
   the output over the original cache to float tolerance.
2. Dissimilar pairs are refused by the sigma gate; the slot target is then
   met by dropping surplus tokens, never by unsafe merges.
3. The full OTKVv7Cache(consolidate=True) lifecycle keeps every channel
   aligned (cache rows, both score streams, logit bias) through prefill
   consolidation and decode-time evictions, and get_decode_bias anticipates
   the incoming token.

Run: python experiments/test_consolidate.py
"""
import builtins
import math
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

from core.consolidate import consolidate_pairs, pair_statistics
from core.ot_kv_v7 import OTKVv7Cache

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))


torch.manual_seed(0)


def attn_out(q, k, v, bias=None):
    """Reference softmax attention for one query, per head."""
    logits = (q.unsqueeze(-2) @ k.transpose(-1, -2)).squeeze(-2) / math.sqrt(k.shape[-1])
    if bias is not None:
        logits = logits + bias
    w = torch.softmax(logits.float(), dim=-1).to(v.dtype)
    return (w.unsqueeze(-2) @ v).squeeze(-2)


# ---------------------------------------------------------------------------
def test_identical_pair_lossless():
    B, H, D, W = 1, 2, 16, 8
    base_k = torch.randn(B, H, 3, D)
    base_v = torch.randn(B, H, 3, D)
    # tokens: [a, a, b] — the identical adjacent pair (0,1) should merge
    k = torch.cat([base_k[:, :, :1], base_k[:, :, :1], base_k[:, :, 1:3]], dim=2)[:, :, :4]
    v = torch.cat([base_v[:, :, :1], base_v[:, :, :1], base_v[:, :, 1:3]], dim=2)[:, :, :4]
    N = 4
    pos = torch.arange(N).view(1, 1, N).expand(B, H, N)
    # attention rows consistent with identical logits for the pair
    q_obs = torch.randn(B, H, W, D)
    logits = q_obs @ k.transpose(-1, -2) / math.sqrt(D)
    rows = torch.softmax(logits.float(), dim=-1)                       # (B,H,W,N)
    log_rows = rows.clamp_min(1e-30).log().transpose(-2, -1)           # (B,H,N,W)
    scores = torch.rand(2, B, H, N) + 0.5

    k2, v2, s2, _, bias, n_merged = consolidate_pairs(
        k, v, pos, log_rows, scores, None, target_slots=3, tau=0.05, gap=1)

    check("identical pair is merged", n_merged == 1 and k2.shape[2] == 3,
          f"n_merged={n_merged}, slots={k2.shape[2]}")
    check("merged slot carries bias log2",
          torch.allclose(bias[..., 0], torch.full_like(bias[..., 0], math.log(2.0)), atol=1e-3),
          str(bias[0, 0]))

    q = torch.randn(B, H, D)
    o_orig = attn_out(q, k, v)
    o_merged = attn_out(q, k2, v2, bias)
    err = (o_orig - o_merged).norm() / o_orig.norm()
    check("attention output exactly preserved for an unseen query",
          err < 1e-5, f"rel err {err:.2e}")


def test_gate_refuses_dissimilar():
    B, H, D, W, N = 1, 2, 16, 8, 6
    k = torch.randn(B, H, N, D) * 2.0
    v = torch.randn(B, H, N, D)
    pos = torch.arange(N).view(1, 1, N).expand(B, H, N)
    q_obs = torch.randn(B, H, W, D)
    rows = torch.softmax((q_obs @ k.transpose(-1, -2) / math.sqrt(D)).float(), dim=-1)
    log_rows = rows.clamp_min(1e-30).log().transpose(-2, -1)
    scores = torch.rand(2, B, H, N) + 0.5

    k2, v2, s2, _, bias, n_merged = consolidate_pairs(
        k, v, pos, log_rows, scores, None, target_slots=4, tau=0.05, gap=1)
    check("dissimilar pairs are not merged", n_merged == 0, f"n_merged={n_merged}")
    check("slot target met by dropping surplus", k2.shape[2] == 4, str(k2.shape))
    check("bias stays zero without merges", float(bias.abs().max()) == 0.0)

    err, b, w = pair_statistics(log_rows, pos[..., :-1], pos[..., 1:])
    check("pair error is large for random keys", float(err.median()) > 0.3,
          f"median {float(err.median()):.3f}")


# ---------------------------------------------------------------------------
def make_cache(**kw):
    kwargs = dict(compression_size=0.25, recent_size=8, sink_size=2, mode="prefill",
                  consolidate=True, observation_window=4, layer_alloc="uniform")
    kwargs.update(kw)
    return OTKVv7Cache(**kwargs)


def drive_prefill_consolidate(cache, n_layers, seq, heads=2, dim=8, W=4,
                              duplicate_middle=True):
    torch.manual_seed(3)
    for li in range(n_layers):
        k = torch.randn(1, heads, seq, dim) * 0.05
        v = torch.randn(1, heads, seq, dim) * 0.05
        if duplicate_middle:
            k[:, :, 1::2] = k[:, :, 0::2]       # adjacent duplicates everywhere
            v[:, :, 1::2] = v[:, :, 0::2]
        pos = torch.arange(seq, dtype=torch.float32).view(1, 1, -1)
        k[..., 0] += pos * 1e-4
        cache.update(k.clone(), v.clone(), li, {})
        full = torch.rand(1, heads, seq) + 0.1
        win = torch.rand(1, heads, seq) + 0.1
        if duplicate_middle:
            win[:, :, 1::2] = win[:, :, 0::2]
            full[:, :, 1::2] = full[:, :, 0::2]
        cache.attention_scores[li] = torch.stack([full, win])
        rows = win.unsqueeze(-2).expand(1, heads, W, seq).clone()
        rows = rows / rows.sum(-1, keepdim=True)
        cache._logattn[li] = rows                # (b, kv, W, L)
    cache.finalize_prefill()
    return cache


def lengths_aligned(cache, li):
    L = cache._get_existing_cache(li)[0].shape[-2]
    s_ok = cache.attention_scores[li].shape[-1] == L
    b = cache.logit_bias.get(li)
    b_ok = b is None or b.shape[-1] == L
    return L, s_ok and b_ok


def test_lifecycle():
    n_layers, seq = 2, 160
    cache = make_cache()
    drive_prefill_consolidate(cache, n_layers, seq)

    merged = cache._merged_slots["total"]
    check("consolidation merged pairs on duplicate-heavy input", merged > 0, str(merged))
    ok = all(lengths_aligned(cache, li)[1] for li in range(n_layers))
    check("channels aligned after prefill consolidation", ok)

    L0 = lengths_aligned(cache, 0)[0]
    bias = cache.get_decode_bias(0)
    check("decode bias anticipates the incoming token",
          bias is not None and bias.shape == (1, 2, 1, L0 + 1), str(None if bias is None else bias.shape))
    check("some merged slot carries positive bias",
          bias is not None and float(bias.max()) > 0.5, str(None if bias is None else float(bias.max())))

    # decode with the pre-hook contract: pre_attention_step before each update
    heads, dim = 2, 8
    for step in range(20):
        for li in range(n_layers):
            cache.pre_attention_step(li)
            b = cache.get_decode_bias(li)
            L = cache._get_existing_cache(li)[0].shape[-2]
            if b is not None and b.shape[-1] != L + 1:
                check("bias mask length matches cache before update", False,
                      f"step {step} layer {li}: {b.shape[-1]} vs {L + 1}")
                return
            full = torch.rand(1, heads, L + 1)
            cache.current_attention_scores[li] = torch.stack([full, full])
            kk = torch.randn(1, heads, 1, dim) * 0.05
            cache.update(kk, kk.clone(), li, {})
    check("bias mask length matches cache before update", True)
    ok = all(lengths_aligned(cache, li)[1] for li in range(n_layers))
    check("channels aligned through 20 decode steps with evictions", ok)
    budget0 = cache.get_middle_budget(0, seq) + cache.sink_size + cache.recent_size
    check("cache stays near budget", lengths_aligned(cache, 0)[0] <= budget0 + cache.compress_interval,
          f"{lengths_aligned(cache, 0)[0]} vs {budget0}")


def test_no_consolidation_paths_unchanged():
    cache = make_cache(consolidate=False)
    for li in range(2):
        k = torch.randn(1, 2, 120, 8) * 0.05
        cache.update(k.clone(), k.clone(), li, {})
        full = torch.rand(1, 2, 120) + 0.1
        cache.attention_scores[li] = torch.stack([full, full])
    cache.finalize_prefill()
    check("consolidate=False leaves no bias", not cache.logit_bias and cache.get_decode_bias(0) is None)


if __name__ == "__main__":
    print("test_identical_pair_lossless");     test_identical_pair_lossless()
    print("test_gate_refuses_dissimilar");     test_gate_refuses_dissimilar()
    print("test_lifecycle");                   test_lifecycle()
    print("test_no_consolidation_paths_unchanged"); test_no_consolidation_paths_unchanged()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)
