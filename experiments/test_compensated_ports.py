"""Checks for the KeepKV and SelKV matched-protocol ports (baselines/keepkv.py, baselines/selkv.py).

  1. Fairness: with merge=False both ports are bit-identical to SnapKVCache through prefill
     and decode maintenance (same keys, values, positions kept, no bias), so every paired
     comparison against the existing pooled-SnapKV runs isolates the merge.
  2. KeepKV: Eq. 8 in log space equals a direct float64 evaluation; ZIP-Merging with the
     scores of one query is exactly perturbation-free for that query, including when several
     evicted tokens merge into the same retained entry in sequence; the cosine threshold and
     the key-scale guard behave as documented.
  3. SelKV: routing (Eq. 3), gate (Eq. 4), merge (Eq. 5) and compensation (Eq. 6) equal an
     explicit per-token loop; keys are untouched; the own selector returns exactly the budget
     and trims the union by the mean score.
  4. Decode: the logit bias stays aligned with the cache over maintenance steps.
  5. 2026-09-19 knobs: KeepKV(selector="selkv") with merge off is bit-identical to
     SelKV(selector="selkv") with merge off; SelKV(fallback="global") equals an explicit loop
     that applies Eq. 3 over the whole S_h when the bucket is empty.

    python experiments/test_compensated_ports.py
    python experiments/test_compensated_ports.py --bitcheck RUN_DIR _keepkv_selector_selkv_merge_false.json \
        REF_DIR _selkv_selector_selkv_merge_false.json      # per-sample files of two GPU runs must agree
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.keepkv import KeepKVCache                     # noqa: E402
from baselines.selkv import SelKVCache, avgpool_tokens        # noqa: E402
from baselines.snapkv import SnapKVCache                      # noqa: E402

PASS, FAIL = 0, 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


def causal_attn(q, k):
    """q (b,qh,n,d), k (b,kv,L,d) -> softmax attention (b,qh,n,L), causal with query at L-n+t."""
    b, qh, n, d = q.shape
    kv, L = k.shape[1], k.shape[2]
    g = qh // kv
    kk = k.repeat_interleave(g, dim=1)
    lg = q @ kk.transpose(-1, -2) / math.sqrt(d)
    qpos = torch.arange(L - n, L)
    mask = torch.arange(L)[None, :] > qpos[:, None]
    return torch.softmax(lg.masked_fill(mask, float("-inf")), dim=-1)


def common(**kw):
    base = dict(compression_size=0.25, recent_size=8, sink_size=4, observation_window=16,
                pool_kernel=7, per_head=False, mode="prefill")
    base.update(kw)
    return base


# ============================================================================ 1. fairness
def drive(cache, K, V, Q, steps, seed, prehook):
    """Prefill with K,V,Q then `steps` decode steps, the evaluator's call order."""
    gen = torch.Generator().manual_seed(seed)
    b, kv, L, d = K.shape
    qh = Q.shape[1]
    cache.update(K, V, 0)
    cache.current_attention_scores[0] = cache.reduce_attention(causal_attn(Q, K))
    for _ in range(steps):
        kn = torch.randn(b, kv, 1, d, generator=gen)
        vn = torch.randn(b, kv, 1, d, generator=gen)
        qn = torch.randn(b, qh, 1, d, generator=gen)
        if prehook:
            cache.pre_attention_step(0)
        cache.update(kn, vn, 0)
        kc, _ = cache._get_existing_cache(0)
        cache.current_attention_scores[0] = cache.reduce_attention(causal_attn(qn, kc))
    return cache


def test_fairness():
    print("\n[1] merge=False is the SnapKV cache")
    torch.manual_seed(0)
    b, kv, g, L, d = 1, 2, 2, 96, 8
    K, V, Q = torch.randn(b, kv, L, d), torch.randn(b, kv, L, d), torch.randn(b, kv * g, L, d)
    ref = drive(SnapKVCache(**common()), K, V, Q, steps=6, seed=1, prehook=False)
    rk, rv = ref._get_existing_cache(0)
    for name, cache in (("KeepKV", KeepKVCache(merge=False, **common())),
                        ("SelKV(snapkv)", SelKVCache(selector="snapkv", merge=False, **common()))):
        c = drive(cache, K, V, Q, steps=6, seed=1, prehook=True)
        ck, cv = c._get_existing_cache(0)
        check(f"{name}: identical keys and values to SnapKV after prefill + 6 decode steps",
              ck.shape == rk.shape and torch.equal(ck, rk) and torch.equal(cv, rv),
              f"{tuple(ck.shape)} vs {tuple(rk.shape)}")
        check(f"{name}: identical accumulated scores", torch.equal(c.attention_scores[0], ref.attention_scores[0]))
        check(f"{name}: no bias when merge is off", c.get_decode_bias(0) is None)


# ============================================================================ 2. KeepKV
def test_keepkv_ema():
    print("\n[2a] KeepKV Eq. 8 in log space")
    torch.manual_seed(1)
    kv, g, W, L, d, alpha = 2, 3, 5, 12, 6, 0.9
    q, k = torch.randn(kv * g, W, d), torch.randn(kv, L, d)
    got = KeepKVCache.log_ema_scores(q, k, alpha, L)
    exp = torch.zeros(kv, L, dtype=torch.float64)
    for h in range(kv):
        for i in range(L):
            tot = 0.0
            for t in range(W):
                pos = L - W + t
                if i > pos:
                    continue
                s = sum(math.exp(float(q[h * g + gg, t].double() @ k[h, i].double()) / math.sqrt(d))
                        for gg in range(g)) / g
                tot += (1 - alpha) * alpha ** (L - 1 - pos) * s
            exp[h, i] = math.log(tot / (1 - alpha ** L))
    check("log EMA scores equal the direct float64 formula (causal, head-averaged, bias-corrected)",
          torch.allclose(got.double(), exp, atol=1e-4), f"max err {float((got.double() - exp).abs().max()):.2e}")


def keepkv_setup(seed, n_merge_into_one):
    """A single-head, single-query cache in which `n_merge_into_one` evicted keys are near
    copies of one retained key (cosine > 0.8) and must merge into it."""
    torch.manual_seed(seed)
    L, d = 40, 16
    k = torch.randn(1, 1, L, d) * 0.6
    v = torch.randn(1, 1, L, d)
    target = 10
    evicted = list(range(20, 20 + n_merge_into_one))
    for j, e in enumerate(evicted):
        k[0, 0, e] = k[0, 0, target] + 0.15 * torch.randn(d)
    q = torch.randn(1, 1, 1, d)
    return k, v, q, target, evicted


def keepkv_merge(k, v, q, evicted, alpha=1e-7, lam_policy="as_written"):
    L = k.shape[2]
    cache = KeepKVCache(merge_threshold=0.8, ema_alpha=alpha, ema_window=0, lam_policy=lam_policy,
                        compression_size=L, recent_size=0, sink_size=0, observation_window=1,
                        pool_kernel=0, per_head=False, mode="prefill")
    cache.update(k.clone(), v.clone(), 0)
    ms, me = 0, L
    keep = torch.tensor([i for i in range(L) if i not in evicted])
    cache._merge_layer(0, keep, ms, me, q)
    nk, nv = cache._get_existing_cache(0)
    return cache, nk, nv, keep


def output(q, k, v, bias, idx):
    lg = (q[0, 0, 0] @ k[0, 0, idx].T) / math.sqrt(k.shape[-1]) + bias[idx]
    return torch.softmax(lg.double(), dim=-1) @ v[0, 0, idx].double()


def test_keepkv_zip():
    print("\n[2b] KeepKV ZIP-Merging is perturbation-free for the scoring query")
    for n in (1, 2, 4):
        k, v, q, target, evicted = keepkv_setup(3 + n, n)
        L = k.shape[2]
        cache, nk, nv, keep = keepkv_merge(k, v, q, evicted)
        dense = output(q, k, v, torch.zeros(L), torch.arange(L))
        bias = cache.logit_bias[0][0, 0]
        merged = output(q, nk, nv, bias, keep)
        st = cache.merge_stats
        check(f"{n} evicted -> one target: merged {int(st['merged'])}/{n}, output error "
              f"{float((merged - dense).norm() / dense.norm()):.1e}",
              int(st["merged"]) == n and torch.allclose(merged, dense, rtol=1e-4, atol=1e-5))
        check(f"{n} evicted -> votes on the target = {n + 1}, log-bias recorded",
              abs(float(bias[target]) - math.log(n + 1)) < 1e-6)


def test_keepkv_threshold_and_guard():
    print("\n[2c] KeepKV threshold, key-scale policy and fail-fast")
    torch.manual_seed(7)
    L, d = 30, 8
    k = torch.randn(1, 1, L, d)
    v = torch.randn(1, 1, L, d)
    k[0, 0, 20] = -k[0, 0, 5]                                  # cosine -1 to its partner: never merges
    q = torch.randn(1, 1, 1, d)
    cache, nk, nv, keep = keepkv_merge(k, v, q, evicted=[20])
    check("an evicted key with no retained key above T=0.8 is not merged",
          cache.merge_stats["merged"] == 0 and torch.equal(nk[0, 0, keep], k[0, 0, keep]))

    # lam = ln s_r / (pi_e l_e + pi_c l_c). Since ln s_r - den = H(pi) - ln 2 <= 0, lam < 0 exactly when
    # den > 0 > ln s_r: logits +0.2 (retained) and -0.3 (evicted) give lam = -1.7. Algorithm 1 does not
    # restrict the sign, and the mass identity still holds, so as written the merge is applied.
    d = 4
    qv = torch.tensor([0.0, 2.0, 0.0, 0.0]).view(1, 1, 1, d)   # logit = q.k/sqrt(4) = 2nd component
    L = 12
    k = torch.zeros(1, 1, L, d)
    k[0, 0, :, 2] = 1.0                                          # other keys orthogonal to the pair
    k[0, 0, :, 3] = 0.01 * torch.randn(L)
    v = torch.randn(1, 1, L, d)
    k[0, 0, 3] = torch.tensor([1.0, 0.2, 0.0, 0.0])             # retained, logit +0.2
    k[0, 0, 8] = torch.tensor([1.0, -0.3, 0.0, 0.0])            # evicted, logit -0.3, cosine 0.88
    cache, nk, nv, keep = keepkv_merge(k, v, qv, evicted=[8])
    st = cache.merge_stats
    dense = output(qv, k, v, torch.zeros(L), torch.arange(L))
    merged = output(qv, nk, nv, cache.logit_bias[0][0, 0], keep)
    check(f"as written: a negative key scale is applied (negative {int(st['lam_negative'])}) and is still "
          f"perturbation-free for the scoring query (err {float((merged - dense).norm() / dense.norm()):.1e})",
          st["merged"] == 1 and st["lam_negative"] == 1 and torch.allclose(merged, dense, rtol=1e-4, atol=1e-5))
    cache, nk, nv, keep = keepkv_merge(k, v, qv, evicted=[8], lam_policy="positive")
    check("sensitivity lam_policy=positive skips it and leaves the target untouched",
          cache.merge_stats["skip_policy"] == 1 and torch.equal(nk[0, 0, 3], k[0, 0, 3]))
    cache, nk, nv, keep = keepkv_merge(k, v, qv, evicted=[8], lam_policy="cap:1.0")
    check("sensitivity lam_policy=cap:1 skips |lam| = 1.7", cache.merge_stats["skip_policy"] == 1)

    # both logits exactly 0: ln s_r = 0 and den = 0, the scale is 0/0 -> undefined, skipped and counted
    k0 = torch.zeros(1, 1, L, d)
    k0[0, 0, :, 2] = 1.0
    k0[0, 0, 3] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    k0[0, 0, 8] = torch.tensor([1.0, 0.0, 0.1, 0.0])
    q0 = torch.tensor([0.0, 0.0, 0.0, 5.0]).view(1, 1, 1, d)     # orthogonal to both: logits 0
    cache, nk, nv, keep = keepkv_merge(k0, v, q0, evicted=[8])
    check("an undefined (0/0) key scale is skipped and counted",
          cache.merge_stats["skip_undefined"] == 1 and cache.merge_stats["merged"] == 0)

    c = KeepKVCache(**common())
    K, V = torch.randn(1, 2, 96, 8), torch.randn(1, 2, 96, 8)
    c.update(K, V, 0)
    c.current_attention_scores[0] = c.reduce_attention(causal_attn(torch.randn(1, 4, 96, 8), K))
    raised = False
    try:
        c.finalize_prefill()                                    # no window queries were captured
    except RuntimeError:
        raised = True
    check("KeepKV refuses to fall back to plain eviction when no prefill queries were captured", raised)


def test_registration():
    print("\n[2d] registered method strings resolve to the protocol's bases")
    sys.argv = ["x"]
    from main import get_cache_config
    base = {"compression_size": 0.05, "recent_size": 0.1, "sink_size": 4}
    for m, want in (("keepkv", (32, 7)), ("selkv;selector=snapkv", (32, 7)), ("selkv;selector=selkv", (32, 0)),
                    ("keepkv;selector=selkv", (32, 0)), ("keepkv;selector=selkv;merge=false", (32, 0)),
                    ("selkv;selector=snapkv;fallback=global", (32, 7))):
        cls, kw = get_cache_config(m, dict(base))
        c = cls(**kw)
        check(f"'{m}': observation window {c.observation_window}, pool kernel {c.pool_kernel}",
              (c.observation_window, c.pool_kernel) == want)


# ============================================================================ 3. SelKV
def test_selkv_merge_equals_loop():
    print("\n[3a] SelKV Eqs. 3-6 equal an explicit loop")
    torch.manual_seed(11)
    b, kv, g, L, d, B = 1, 2, 2, 72, 6, 8
    K, V, Q = torch.randn(b, kv, L, d), torch.randn(b, kv, L, d), torch.randn(b, kv * g, L, d)
    A = causal_attn(Q, K)
    cache = SelKVCache(selector="snapkv", bucket=B, selkv_window=16, **common(compression_size=0.4))
    cache.update(K.clone(), V.clone(), 0)
    scores, extra = cache.reduce_attention(A)
    ms, me = 4, L - 8
    torch.manual_seed(12)
    keep_mid = torch.randperm(me - ms)[:18].sort().values
    keep = torch.zeros(L, dtype=torch.bool)
    keep[:ms] = True
    keep[me:] = True
    keep[ms + keep_mid] = True
    cache._merge_layer(0, keep_mid, ms, me, extra)
    _, nv = cache._get_existing_cache(0)
    nk, _ = cache._get_existing_cache(0)
    bias = cache.logit_bias[0]

    Ag = A.reshape(b, kv, g, L, L).sum(2)[0]                    # (kv, L, L)
    win = Ag[:, -16:, :].sum(1)                                  # (kv, L)
    exp_v = V[0].clone()
    R = torch.ones(kv, L)
    for h in range(kv):
        num = torch.zeros(L, d)
        den = torch.zeros(L)
        for j in range(L):
            if keep[j]:
                continue
            cand = [i for i in range((j // B) * B, min((j // B) * B + B, L)) if keep[i]]
            if not cand:
                continue
            att = torch.tensor([float(Ag[h, j, i]) for i in cand])
            if float(att.max()) <= 0:
                continue
            i = cand[int(att.argmax())]
            gte = max(float(F.cosine_similarity(V[0, h, j], V[0, h, i], dim=0)), 0.0)
            num[i] += gte * win[h, j] * V[0, h, j]
            den[i] += gte * win[h, j]
        for i in range(L):
            if den[i] > 0:
                exp_v[h, i] = (win[h, i] * V[0, h, i] + num[i]) / (win[h, i] + den[i])
                if win[h, i] > 0:
                    R[h, i] = (win[h, i] + den[i]) / win[h, i]
    exp_bias = 0.5 * torch.log(R.mean(0))
    check("merged values equal Eq. 5 with Eq. 3 routing and Eq. 4 gate",
          torch.allclose(nv[0].float(), exp_v, atol=1e-5), f"max err {float((nv[0] - exp_v).abs().max()):.2e}")
    check("bias equals 0.5 * log(mean over KV heads of Eq. 6 R), same for every head",
          torch.allclose(bias[0, 0], exp_bias, atol=1e-5) and torch.equal(bias[0, 0], bias[0, 1]))
    check("keys are untouched", torch.equal(nk, K))
    check("an evicted token with no attended kept token in its bucket is dropped (routed < evicted)",
          cache.merge_stats["routed"] < cache.merge_stats["evicted"])


def test_selkv_head_sets_and_failfast():
    print("\n[3c] SelKV routes within the head's own S_h (Eq. 3) and refuses chunked prefill")
    kv, g, L, d, B = 2, 1, 16, 4, 8
    V = torch.randn(1, kv, L, d)
    K = torch.randn(1, kv, L, d)
    cache = SelKVCache(selector="selkv", bucket=B, selkv_window=4, **common(compression_size=0.5, recent_size=0))
    cache.update(K.clone(), V.clone(), 0)
    ms, me = 0, L
    keep_mid = torch.tensor([1, 2, 9, 10])                      # shared layout after union-and-trim
    hs = torch.zeros(kv, L, dtype=torch.bool)
    hs[0, [1, 9]] = True                                         # head 0 selected 1 and 9 itself
    hs[1, [2, 10]] = True                                        # head 1 selected 2 and 10 itself
    cache._head_sets[0] = hs
    bucket = torch.zeros(1, kv * g, L, B)
    win = torch.ones(1, kv * g, L)
    for h in range(kv):
        bucket[0, h, 3, 2] = 0.9                                 # token 3 attends most to position 2 ...
        bucket[0, h, 3, 1] = 0.1                                 # ... and a little to position 1
    V[0, :, 3] = V[0, :, 1] + V[0, :, 2]                         # positive gate to both
    cache._replace_existing_cache(0, K.clone(), V.clone())
    cache._merge_layer(0, keep_mid, ms, me, {"win_q": win, "bucket_q": bucket})
    _, nv = cache._get_existing_cache(0)
    check("head 0 cannot use position 2 (kept only by head 1): it routes token 3 to position 1",
          not torch.equal(nv[0, 0, 1], V[0, 0, 1]) and torch.equal(nv[0, 0, 2], V[0, 0, 2]))
    check("head 1 routes token 3 to position 2, the most-attended position in its own S_h",
          not torch.equal(nv[0, 1, 2], V[0, 1, 2]) and torch.equal(nv[0, 1, 1], V[0, 1, 1]))
    raised = False
    try:
        cache.reduce_attention(torch.softmax(torch.randn(1, kv, 5, L), -1))
    except RuntimeError:
        raised = True
    check("a prefill call with query length != key length raises", raised)


def test_selkv_selector():
    print("\n[3b] SelKV own selector")
    torch.manual_seed(21)
    b, kv, g, L, d = 1, 3, 2, 80, 6
    K, V, Q = torch.randn(b, kv, L, d), torch.randn(b, kv, L, d), torch.randn(b, kv * g, L, d)
    cache = SelKVCache(selector="selkv", selkv_window=16, **common(compression_size=0.3))
    cache.update(K, V, 0)
    _, extra = cache.reduce_attention(causal_attn(Q, K))
    ms, me = 4, L - 8
    budget = 12
    keep = cache._keep_middle(0, None, budget, ms, me, extra)
    a = extra["win_q"].reshape(b, kv, g, L).sum(2)[0, :, ms:me]
    c = avgpool_tokens(a * V[0].norm(dim=-1)[:, ms:me], 5)
    union = torch.zeros(me - ms, dtype=torch.bool)
    union[torch.topk(c, budget, dim=-1).indices.flatten()] = True
    mean = c.mean(0).masked_fill(~union, float("-inf"))
    exp = torch.topk(mean, budget).indices.sort().values
    check(f"keeps exactly the budget ({keep.numel()} = {budget})", keep.numel() == budget)
    check("union of per-head top-m_s trimmed by the mean score across heads", torch.equal(keep, exp))
    check("avg-pool keeps length and averages a constant to itself inside the window",
          torch.allclose(avgpool_tokens(torch.ones(20), 5)[2:-2], torch.ones(16)))


# ============================================================================ 4. decode
def test_decode_alignment():
    print("\n[4] logit bias stays aligned with the cache during decode maintenance")
    torch.manual_seed(31)
    b, kv, g, L, d = 1, 2, 2, 96, 8
    K, V, Q = torch.randn(b, kv, L, d), torch.randn(b, kv, L, d), torch.randn(b, kv * g, L, d)
    for c in range(3):
        K[0, :, 40 + c] = K[0, :, 39] + 0.05 * torch.randn(kv, d)   # guaranteed KeepKV candidates
    for name, cache in (("KeepKV", KeepKVCache(**common())),
                        ("SelKV(snapkv)", SelKVCache(selector="snapkv", **common())),
                        ("SelKV(selkv)", SelKVCache(selector="selkv", **common()))):
        if isinstance(cache, KeepKVCache):
            cache.window_queries[0] = Q[:, :, -(cache.ema_window + 1):, :]
        ok = True
        gen = torch.Generator().manual_seed(5)
        cache.update(K.clone(), V.clone(), 0)
        cache.current_attention_scores[0] = cache.reduce_attention(causal_attn(Q, K))
        for _ in range(5):
            cache.pre_attention_step(0)
            bias = cache.get_decode_bias(0)
            n = cache._get_existing_cache(0)[0].shape[-2]
            if bias is not None and tuple(bias.shape) != (1, kv * g, 1, n + 1):
                ok = False
            cache.update(torch.randn(b, kv, 1, d, generator=gen), torch.randn(b, kv, 1, d, generator=gen), 0)
            n2 = cache._get_existing_cache(0)[0].shape[-2]
            if 0 in cache.logit_bias and cache.logit_bias[0].shape[-1] != n2:
                ok = False
            kc, _ = cache._get_existing_cache(0)
            cache.current_attention_scores[0] = cache.reduce_attention(
                causal_attn(torch.randn(b, kv * g, 1, d, generator=gen), kc))
        check(f"{name}: bias shape matches cache+1 and length stays aligned over 5 steps "
              f"(bias active: {bool(cache._has_bias.get(0))})", ok)


def test_round2_additions():
    print("\n[5] round-2 audit additions")
    # (a) KeepKV: a merged key that overflows the cache dtype is skipped, not written.
    # The scale is lam = ln s_r / (pi_e ln s_e + pi_c ln s_c): with ln s_e ~ 0+ and ln s_c = -20,
    # pi_e ~ 1 so the denominator is ~1e-5 while ln s_r = ln s_e - ln 2 ~ -0.69, giving lam ~ -5.8e4.
    # The retained key's norm then decides whether lam * k fits in float16 (max 65504).
    d, L = 4, 12
    k = torch.zeros(1, 1, L, d)
    k[0, 0, :, 2] = 1.0
    k[0, 0, 3] = torch.tensor([4.0, -0.2, 0.0, 0.0])            # retained, logit -20
    k[0, 0, 8] = torch.tensor([4.0, 1e-7, 0.0, 0.0])            # evicted,  logit ~0, cosine ~1
    q = torch.tensor([0.0, 200.0, 0.0, 0.0]).view(1, 1, 1, d)   # logit = 100 * k[1]
    v = torch.randn(1, 1, L, d)
    kh = k.half().float()                                       # the cache stores half: use its values
    ln_se, ln_sc = 100 * kh[0, 0, 8, 1].item(), 100 * kh[0, 0, 3, 1].item()
    ln_W = float(torch.logaddexp(torch.tensor(ln_se), torch.tensor(ln_sc)))
    pi_e = math.exp(ln_se - ln_W)
    lam_exp = (ln_W - math.log(2.0)) / (pi_e * ln_se + (1 - pi_e) * ln_sc)
    kr_exp = lam_exp * (pi_e * kh[0, 0, 8] + (1 - pi_e) * kh[0, 0, 3])
    cache = KeepKVCache(merge_threshold=0.8, ema_alpha=1e-7, ema_window=0, compression_size=L, recent_size=0,
                        sink_size=0, observation_window=1, pool_kernel=0, per_head=False, mode="prefill")
    cache.update(k.half(), v.half(), 0)
    keep = torch.tensor([i for i in range(L) if i != 8])
    cache._merge_layer(0, keep, 0, L, q)
    nk, _ = cache._get_existing_cache(0)
    check(f"KeepKV: a key scale of {lam_exp:.2e} whose merged key ({kr_exp.abs().max():.3g}) overflows float16 "
          f"is skipped (skip_nonfinite {int(cache.merge_stats['skip_nonfinite'])}) and the cache stays finite",
          float(kr_exp.abs().max()) > 65504 and cache.merge_stats["skip_nonfinite"] == 1
          and bool(torch.isfinite(nk.float()).all()) and torch.equal(nk[0, 0, 3], k[0, 0, 3].half()))

    # (b) SelKV: zero and small-positive window attention on a routed-to target are floored and counted
    kv, L, d, B = 1, 16, 4, 8
    K, V = torch.randn(1, kv, L, d), torch.randn(1, kv, L, d)
    c = SelKVCache(selector="snapkv", bucket=B, **common(compression_size=0.5, recent_size=0))
    c.update(K.clone(), V.clone(), 0)
    win = torch.ones(1, kv, L)
    win[0, 0, 1] = 0.0
    win[0, 0, 9] = 1e-10
    bucket = torch.zeros(1, kv, L, B)
    bucket[0, 0, 3, 1] = 1.0                                     # token 3 -> position 1 (a_i = 0)
    bucket[0, 0, 11, 1] = 1.0                                    # token 11 -> position 9 (a_i = 1e-10)
    V[0, 0, 3] = V[0, 0, 1]
    V[0, 0, 11] = V[0, 0, 9]
    c._replace_existing_cache(0, K.clone(), V.clone())
    c._merge_layer(0, torch.tensor([1, 9]), 0, L, {"win_q": win, "bucket_q": bucket})
    bias = c.logit_bias[0][0, 0]
    check(f"SelKV: zero and 1e-10 target attention both counted (floored "
          f"{int(c.merge_stats['floored_attention_targets'])}) and the bias stays finite",
          c.merge_stats["floored_attention_targets"] == 2 and bool(torch.isfinite(bias).all()))

    # (c) integrated: the head sets _keep_middle builds are exactly the ones _merge_layer routes with
    torch.manual_seed(41)
    b, kv, g, L, d = 1, 2, 2, 96, 8
    K, V, Q = torch.randn(b, kv, L, d), torch.randn(b, kv, L, d), torch.randn(b, kv * g, L, d)
    c = SelKVCache(selector="selkv", selkv_window=16, **common())
    seen = {}
    orig = c._merge_layer
    def spy(layer_idx, keep_mid, ms, me, extra):
        seen["sets"] = c._head_sets.get(layer_idx).clone()
        seen["ms"], seen["me"] = ms, me
        return orig(layer_idx, keep_mid, ms, me, extra)
    c._merge_layer = spy
    c.update(K, V, 0)
    c.current_attention_scores[0] = c.reduce_attention(causal_attn(Q, K))
    c.finalize_prefill()
    ms, me = seen["ms"], seen["me"]
    a = (causal_attn(Q, K)[..., -16:, :].sum(2)).reshape(b, kv, g, L).sum(2)[0, :, ms:me]
    sc = avgpool_tokens(a * V[0].norm(dim=-1)[:, ms:me], 5)
    budget = c.get_middle_budget(0, L)
    exp = torch.zeros(kv, me - ms, dtype=torch.bool)
    exp.scatter_(1, torch.topk(sc, budget, dim=-1).indices, True)
    check("integrated: _merge_layer receives the per-head top-m_s sets built by _keep_middle, then they are freed",
          torch.equal(seen["sets"], exp) and 0 not in c._head_sets and c.merge_stats["routed"] > 0)

    # (d) no eviction at prefill: missing statistics must not raise
    for name, cache in (("KeepKV", KeepKVCache(**common(compression_size=1.0))),
                        ("SelKV", SelKVCache(selector="selkv", **common(compression_size=1.0)))):
        cache.update(torch.randn(1, 2, 40, 8), torch.randn(1, 2, 40, 8), 0)
        ok = True
        try:
            cache.finalize_prefill()
        except RuntimeError:
            ok = False
        check(f"{name}: a sample with no eviction at prefill does not raise without statistics", ok)

    # (e) chunked prefill with the snapkv selector and merge off stays plain SnapKV
    a = torch.softmax(torch.randn(1, 4, 5, 20), dim=-1)
    ref = SnapKVCache(**common()).reduce_attention(a)
    got = SelKVCache(selector="snapkv", merge=False, **common()).reduce_attention(a)
    check("SelKV(snapkv, merge=False): a chunked prefill call returns SnapKV's scores instead of raising",
          torch.is_tensor(got) and torch.equal(got, ref))

    # (f) the per-sample merge counters reach MERGE_STATS_PATH, labelled with the method string,
    # so a remote run's statistics are recoverable from the volume and not only from stdout
    import json
    import tempfile
    torch.manual_seed(3)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "merge_stats.jsonl")
        os.environ["MERGE_STATS_PATH"], os.environ["OTKV_METHOD"] = path, "selkv;selector=selkv"
        try:
            c = SelKVCache(selector="selkv", **common())
            K, V, Q = torch.randn(1, 2, 96, 8), torch.randn(1, 2, 96, 8), torch.randn(1, 4, 96, 8)
            c.update(K, V, 0)
            c.current_attention_scores[0] = c.reduce_attention(causal_attn(Q, K))
            c.pre_attention_step(0)
            recs = [json.loads(line) for line in open(path)]
        finally:
            os.environ.pop("MERGE_STATS_PATH"); os.environ.pop("OTKV_METHOD")
    ok = (len(recs) == 1 and recs[0]["cache"] == "SelKVCache" and recs[0]["method"] == "selkv;selector=selkv"
          and recs[0]["merge"] is True and recs[0]["stats"]["routed"] > 0
          and recs[0]["stats"] == {k: float(v) for k, v in c.merge_stats.items()})
    check("merge statistics are appended to MERGE_STATS_PATH as one labelled JSON line per sample", ok,
          str(recs)[:200])

    # (g) keys="fixed" isolates KeepKV's two channels: identical values and votes, keys untouched
    torch.manual_seed(11)
    kv, L, d = 2, 160, 16
    proto = torch.randn(1, kv, 3, d)
    K = proto[:, :, torch.randint(0, 3, (L,))] + 0.15 * torch.randn(1, kv, L, d)
    V, Qw = torch.randn(1, kv, L, d), torch.randn(1, 4, 33, d)      # one query draw for both modes
    out = {}
    for mode in ("moved", "fixed"):
        c = KeepKVCache(keys=mode, compression_size=0.25, recent_size=0.1, sink_size=4,
                        per_head=False, mode="prefill")
        c.update(K.clone(), V.clone(), 0)
        c._merge_layer(0, torch.arange(0, 136, 7), 4, 140, Qw)
        out[mode] = (*c._get_existing_cache(0), c.logit_bias[0], c.merge_stats["merged"])
    km, vm, bm, n = out["moved"]
    kf, vf, bf, _ = out["fixed"]
    check(f"KeepKV keys=fixed: same values and votes as keys=moved ({n:.0f} merges), keys left untouched",
          out["moved"][3] > 0 and torch.equal(vm, vf) and torch.equal(bm, bf)
          and torch.equal(kf, K) and not torch.equal(km, kf))


# ============================================================================ 5. 2026-09-19 knobs
def drive_prehook(cache, K, V, Q, steps, seed, queries=None):
    """Like drive(prehook=True) but also feeds KeepKV its window queries at prefill."""
    gen = torch.Generator().manual_seed(seed)
    b, kv, L, d = K.shape
    qh = Q.shape[1]
    cache.update(K, V, 0)
    cache.current_attention_scores[0] = cache.reduce_attention(causal_attn(Q, K))
    if queries is not None and hasattr(cache, "window_queries"):
        cache.window_queries[0] = queries
    for _ in range(steps):
        kn = torch.randn(b, kv, 1, d, generator=gen)
        vn = torch.randn(b, kv, 1, d, generator=gen)
        qn = torch.randn(b, qh, 1, d, generator=gen)
        cache.pre_attention_step(0)
        cache.update(kn, vn, 0)
        kc, _ = cache._get_existing_cache(0)
        cache.current_attention_scores[0] = cache.reduce_attention(causal_attn(qn, kc))
    return cache


def test_keepkv_selkv_selector():
    print("\n[5a] KeepKV on the SelKV selector")
    torch.manual_seed(51)
    b, kv, g, L, d = 1, 2, 2, 96, 8
    K, V, Q = torch.randn(b, kv, L, d), torch.randn(b, kv, L, d), torch.randn(b, kv * g, L, d)
    ref = drive_prehook(SelKVCache(selector="selkv", merge=False, **common()), K.clone(), V.clone(), Q, 6, 2)
    got = drive_prehook(KeepKVCache(selector="selkv", merge=False, **common()), K.clone(), V.clone(), Q, 6, 2)
    rk, rv = ref._get_existing_cache(0)
    ck, cv = got._get_existing_cache(0)
    check("keepkv;selector=selkv;merge=false is bit-identical to selkv;selector=selkv;merge=false "
          "(keys, values, scores) after prefill + 6 maintenance steps",
          ck.shape == rk.shape and torch.equal(ck, rk) and torch.equal(cv, rv)
          and torch.equal(got.attention_scores[0], ref.attention_scores[0]))
    check("no bias when merge is off", got.get_decode_bias(0) is None)
    check("selkv_window 32 / pool 0 for the SelKV selector (as SelKVCache); common()'s 16 / 7 for the SnapKV one",
          (got.observation_window, got.pool_kernel) == (32, 0)
          and (KeepKVCache(**common()).observation_window, KeepKVCache(**common()).pool_kernel) == (16, 7))

    # with merging on, the kept set is still SelKV's and KeepKV's Alg. 1 runs on it
    torch.manual_seed(52)
    proto = torch.randn(1, kv, 3, d)
    K2 = proto[:, :, torch.randint(0, 3, (L,))] + 0.15 * torch.randn(1, kv, L, d)   # near-duplicate keys
    Qw = torch.randn(1, kv * g, 17, d)
    seen = {}
    c = KeepKVCache(selector="selkv", **common())
    orig = c._merge_layer
    def spy(layer_idx, keep_mid, ms, me, extra):
        seen["keep"] = keep_mid.clone(); seen["extra"] = extra
        return orig(layer_idx, keep_mid, ms, me, extra)
    c._merge_layer = spy
    c.update(K2.clone(), V.clone(), 0)
    c.current_attention_scores[0] = c.reduce_attention(causal_attn(Q, K2))
    c.window_queries[0] = Qw
    c.pre_attention_step(0)
    s2 = SelKVCache(selector="selkv", merge=False, **common())
    s2.update(K2.clone(), V.clone(), 0)
    s2.current_attention_scores[0] = s2.reduce_attention(causal_attn(Q, K2))
    s2.pre_attention_step(0)
    ck, _ = c._get_existing_cache(0)
    sk, _ = s2._get_existing_cache(0)
    check("merge on: the same positions are kept as by SelKV's selector (cache lengths agree)",
          ck.shape == sk.shape and isinstance(seen["extra"], dict) and seen["extra"]["q"] is Qw
          and seen["extra"]["sel"] is not None)
    check(f"merge on: KeepKV merged {int(c.merge_stats['merged'])} evicted tokens and set a bias",
          c.merge_stats["merged"] > 0 and c.get_decode_bias(0) is not None)
    check("head sets are freed after selection", 0 not in c._head_sets)
    raised = False
    try:
        KeepKVCache(selector="pyramid", **common())
    except ValueError:
        raised = True
    check("an unknown selector is rejected", raised)


def selkv_loop(A, V, keep, head_sets, B, win, fallback):
    """Explicit per-token reference for Eqs. 3-6 with the empty-bucket fallback."""
    kv, L, d = V.shape[1], V.shape[2], V.shape[3]
    exp_v = V[0].clone()
    R = torch.ones(kv, L)
    n_fb = 0
    for h in range(kv):
        s_h = keep.clone()
        if head_sets is not None:
            s_h = s_h & head_sets[h]
        num = torch.zeros(L, d)
        den = torch.zeros(L)
        for j in range(L):
            if keep[j]:
                continue
            cand = [i for i in range((j // B) * B, min((j // B) * B + B, L)) if s_h[i]]
            att = [float(A[h, j, i]) for i in cand]
            if cand and max(att) > 0:
                i = cand[int(torch.tensor(att).argmax())]
            elif fallback == "global":
                cand = [i for i in range(L) if s_h[i]]
                att = [float(A[h, j, i]) for i in cand]
                if not cand or max(att) <= 0:
                    continue
                i = cand[int(torch.tensor(att).argmax())]
                n_fb += 1
            else:
                continue
            gte = max(float(F.cosine_similarity(V[0, h, j], V[0, h, i], dim=0)), 0.0)
            num[i] += gte * win[h, j] * V[0, h, j]
            den[i] += gte * win[h, j]
        for i in range(L):
            if den[i] > 0:
                exp_v[h, i] = (win[h, i] * V[0, h, i] + num[i]) / (win[h, i] + den[i])
                if win[h, i] > 0:
                    R[h, i] = (win[h, i] + den[i]) / win[h, i]
    return exp_v, 0.5 * torch.log(R.mean(0)), n_fb


def test_selkv_fallback():
    print("\n[5b] SelKV empty-bucket fallback")
    torch.manual_seed(61)
    b, kv, g, L, d, B = 1, 2, 2, 96, 6, 8
    K, V, Q = torch.randn(b, kv, L, d), torch.randn(b, kv, L, d), torch.randn(b, kv * g, L, d)
    A = causal_attn(Q, K)
    Ag = A.reshape(b, kv, g, L, L).sum(2)[0]
    win = Ag[:, -16:, :].sum(1)
    ms, me = 4, L - 8
    torch.manual_seed(62)
    keep_mid = torch.randperm(me - ms)[:10].sort().values          # sparse: many empty buckets
    keep = torch.zeros(L, dtype=torch.bool)
    keep[:ms] = True; keep[me:] = True; keep[ms + keep_mid] = True
    out = {}
    for fb in ("drop", "global"):
        cache = SelKVCache(selector="snapkv", bucket=B, selkv_window=16, fallback=fb, fallback_topk=L,
                           **common(compression_size=0.4))
        cache.update(K.clone(), V.clone(), 0)
        scores, extra = cache.reduce_attention(A)
        cache._merge_layer(0, keep_mid, ms, me, extra)
        _, nv = cache._get_existing_cache(0)
        out[fb] = (nv[0].float(), cache.logit_bias[0][0, 0], dict(cache.merge_stats), extra)
    for fb in ("drop", "global"):
        exp_v, exp_bias, n_fb = selkv_loop(Ag, V, keep, None, B, win, fb)
        nv, bias, st, extra = out[fb]
        check(f"fallback={fb}: merged values and bias equal the explicit loop",
              torch.allclose(nv, exp_v, atol=1e-5) and torch.allclose(bias, exp_bias, atol=1e-5),
              f"max err {float((nv - exp_v).abs().max()):.2e}")
        if fb == "global":
            check(f"fallback=global: {int(st['fallback_routed'])} tokens routed globally = loop's {n_fb}, "
                  f"routed = routed_bucket + fallback_routed, no misses with K = L",
                  int(st["fallback_routed"]) == n_fb and n_fb > 0 and st["fallback_miss"] == 0
                  and st["routed"] == st["routed_bucket"] + st["fallback_routed"]
                  and st["evicted"] == st["routed"] + st["dropped_empty_bucket"])
            check("global candidates are per KV head, sorted by descending head-summed attention",
                  tuple(extra["glob_idx"].shape) == (b, kv, L, L) and extra["glob_idx"].dtype == torch.int32
                  and bool((extra["glob_val"][..., :-1] >= extra["glob_val"][..., 1:]).all()))
        else:
            check("fallback=drop: no fallback counters, statistics unchanged from the port",
                  "fallback_routed" not in st and "glob_idx" not in extra and st["routed"] == st["routed_bucket"])
    check("drop and global differ (the fallback routes what drop evicts)",
          not torch.equal(out["drop"][0], out["global"][0]))
    # merge off: no candidates are captured, whatever the fallback says (bit-identical to SnapKV)
    c = SelKVCache(selector="snapkv", merge=False, fallback="global", **common())
    c.update(K.clone(), V.clone(), 0)
    r = c.reduce_attention(A)
    check("merge=False never captures global candidates", isinstance(r, tuple) and "glob_idx" not in r[1])
    # a truncated candidate list: with K = 1 only the top-attended position is available
    c = SelKVCache(selector="snapkv", bucket=B, selkv_window=16, fallback="global", fallback_topk=1,
                   **common(compression_size=0.4))
    c.update(K.clone(), V.clone(), 0)
    _, extra = c.reduce_attention(A)
    c._merge_layer(0, keep_mid, ms, me, extra)
    st = c.merge_stats
    check(f"fallback_topk=1: misses are counted ({int(st['fallback_miss'])} missed, "
          f"{int(st['fallback_routed'])} routed)",
          st["fallback_miss"] + st["fallback_routed"] == out["global"][2]["fallback_routed"]
          + out["global"][2]["fallback_miss"] and st["fallback_miss"] > 0)
    # own selector: the fallback respects the head-specific S_h
    torch.manual_seed(63)
    c = SelKVCache(selector="selkv", bucket=B, selkv_window=16, fallback="global", fallback_topk=L,
                   **common(compression_size=0.3))
    seen = {}
    orig = c._merge_layer
    def spy(layer_idx, keep_mid, ms, me, extra):
        seen["sets"] = c._head_sets.get(layer_idx).clone(); seen["keep"] = keep_mid.clone()
        seen["ms"], seen["me"] = ms, me
        return orig(layer_idx, keep_mid, ms, me, extra)
    c._merge_layer = spy
    c.update(K.clone(), V.clone(), 0)
    c.current_attention_scores[0] = c.reduce_attention(A)
    c.finalize_prefill()
    ms2, me2 = seen["ms"], seen["me"]
    keep2 = torch.zeros(L, dtype=torch.bool)
    keep2[:ms2] = True; keep2[me2:] = True; keep2[ms2 + seen["keep"]] = True
    hs = torch.ones(kv, L, dtype=torch.bool)
    hs[:, ms2:me2] = seen["sets"]
    exp_v, exp_bias, n_fb = selkv_loop(Ag, V, keep2, hs, B, win, "global")
    _, nv = c._get_existing_cache(0)
    kept_idx = torch.nonzero(keep2).flatten()
    check("own selector + global fallback: values equal the loop restricted to each head's S_h",
          torch.allclose(nv[0].float(), exp_v[:, kept_idx], atol=1e-5) and c.merge_stats["fallback_routed"] == n_fb)


def bitcheck(a_dir, a_suffix, b_dir, b_suffix):
    """Per-sample score files of two runs must agree exactly, task by task (the merge-off gate)."""
    import glob
    import json
    print(f"\n[bitcheck] {a_dir}/persample_*{a_suffix}  vs  {b_dir}/persample_*{b_suffix}")
    files = sorted(glob.glob(os.path.join(a_dir, "persample_*" + a_suffix)))
    if not files:
        check(f"per-sample files found in {a_dir}", False)
        return
    for f in files:
        J = json.load(open(f))
        t = J["task"]
        rf = os.path.join(b_dir, f"persample_{t}{b_suffix}")
        if not os.path.exists(rf):
            check(f"{t}: reference {os.path.basename(rf)} exists", False)
            continue
        R = json.load(open(rf))
        x, r = J["scores"], R["scores"]
        n = min(len(x), len(r))
        diff = [k for k in range(n) if abs(x[k] - r[k]) > 0]
        check(f"{t}: {n} samples identical" + (f" ({len(diff)} differ, max |d| "
              f"{max(abs(x[k] - r[k]) for k in diff) * 100:.2f})" if diff else ""),
              len(x) == len(r) and not diff)


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "--bitcheck":
        bitcheck(*sys.argv[2:6])
        print(f"\n{PASS} passed, {FAIL} failed")
        sys.exit(1 if FAIL else 0)
    test_fairness()
    test_keepkv_ema()
    test_keepkv_zip()
    test_keepkv_threshold_and_guard()
    test_registration()
    test_selkv_merge_equals_loop()
    test_selkv_head_sets_and_failfast()
    test_selkv_selector()
    test_decode_alignment()
    test_round2_additions()
    test_keepkv_selkv_selector()
    test_selkv_fallback()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
