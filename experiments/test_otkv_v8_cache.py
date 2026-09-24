"""OTKVv8Cache vs the offline harness protocol on random tensors.

Builds a one-layer cache state (random post-'RoPE' keys/queries, causal softmax
rows), lets the cache finalise its prefill (SnapKV selection + v8 merge + bias),
and checks that the retained values and the per-slot bias equal what the
module functions produce under the harness's protocol (fit 32 / select 16 of
the last 64 positions, joint (c, theta) fit, matching, gamma on select rows),
then runs three decode steps through the same code path the evaluator uses
(pre-hook -> bias -> update) and checks the bias stays aligned with the cache.

    python experiments/test_otkv_v8_cache.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.otkv_v8 as V8                                   # noqa: E402
from core.otkv_v8_cache import OTKVv8Cache, candidate_pairs_fast   # noqa: E402
from experiments.v8_offline import group_rows, retained_set, time_split, candidate_pairs  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        FAILS += 1


def make_state(seed, L=400, d=16, kv=2, g=2):
    gen = torch.Generator().manual_seed(seed)
    K = torch.randn(kv, L, d, generator=gen)
    for t in range(9, L - 80, 11):                          # some near-duplicate keys
        K[:, t] = K[:, t - 1] + 1e-7 * torch.randn(kv, d, generator=gen)
    V = torch.randn(kv, L, d, generator=gen)
    Q = torch.randn(kv * g, L, d, generator=gen)
    for h in range(kv * g):
        Q[h] += 0.4 * K[h // g]
    return K, V, Q


def dense_rows(Q, K, g):
    """Causal softmax rows (qh, L, L)."""
    qh, L, d = Q.shape
    out = []
    for h in range(qh):
        lg = (Q[h] @ K[h // g].T) * d ** -0.5
        mask = torch.arange(L)[None, :] > torch.arange(L)[:, None]
        out.append(torch.softmax(lg.masked_fill(mask, float("-inf")), -1))
    return torch.stack(out, 0)


def test_candidates():
    keep = torch.zeros(300, dtype=torch.bool); keep[::7] = True; keep[:4] = True; keep[-64:] = True
    prot = torch.zeros(300, dtype=torch.bool); prot[:4] = True; prot[-64:] = True
    ref = candidate_pairs(keep, prot, [], 8, 4)
    ii, jj = candidate_pairs_fast(keep.numpy(), prot.numpy(), 8, 4)
    a = sorted(zip(ref[:, 0].tolist(), ref[:, 1].tolist()))
    b = sorted(zip(ii.tolist(), jj.tolist()))
    check("vectorised candidates == harness candidates", a == b, f"{len(a)} vs {len(b)}")


def test_prefill_merge(seed):
    L, d, kv, g = 400, 16, 2, 2
    K, V, Q = make_state(seed, L, d, kv, g)
    budget = 0.30                                              # 120 slots: sink 4 + recent 64 + middle 52
    cache = OTKVv8Cache(compression_size=budget, recent_size=64, sink_size=4, observation_window=32,
                        pool_kernel=7, per_head=False, mode="prefill", joint_c=9)
    cache.update(K[None], V[None], 0)                          # prefill append
    rows = dense_rows(Q, K, g)                                 # (qh, L, L)
    cache.current_attention_scores[0] = cache.reduce_attention(rows[None])
    cache.finalize_prefill()
    k_c, v_c = cache._get_existing_cache(0)
    bias_c = cache.logit_bias.get(0)

    # ---- harness protocol
    keep, protect, _ = retained_set(Q, K, budget, 64, sink=4, obs_window=32, pool_kernel=7, sep_ids=())
    keep_idx = torch.nonzero(keep).flatten()
    check(f"seed {seed}: cache keeps the SnapKV set ({int(keep.sum())} slots)",
          k_c.shape[-2] == int(keep.sum()) and torch.allclose(k_c[0, 0], K[0][keep_idx]))
    class A: pass
    args = A(); args.window = 64; args.max_dist = 8; args.max_cand = 4; args.n_blocks = 4; args.c_max = 4.0
    args.joint_c = 9; args.min_gain = 0.05; args.gammas = (0.0, 0.25, 0.5, 1.0)
    for h in range(kv):
        q_cal, pos_cal, head_cal, _ = group_rows(Q[h * g:(h + 1) * g, L - 64:], L - 64)
        _, a_cal = V8.attention_rows(q_cal, K[h], causal_positions=pos_cal)
        o_cal = a_cal @ V[h]
        o0_cal, p_cal = V8.baseline_outputs(q_cal, K[h], V[h], keep, None, causal_positions=pos_cal)
        split = time_split(pos_cal, head_cal, L, 64)
        fit, _ = split["fit"]; sel, sel_ph = split["select"]
        edges = V8.build_sparse_costs(K[h], V[h], q_cal[fit], torch.arange(L), keep, protect, o_cal[fit],
                                      o0_cal[fit], p_cal[fit], a_cal[fit], max_dist=8, max_cand=4, n_blocks=4,
                                      c_max=4.0, sep_ids=None, joint_c=9)
        plan = V8.solve_partial_transport(edges, 0, None, method="matching")
        gamma, imp, per = V8.validate_plan(q_cal[sel], K[h], V[h], keep, plan, o_cal[sel], gammas=(0.0, 0.25, 0.5, 1.0),
                                          min_gain=0.05, per_head_rows=sel_ph, causal_positions=pos_cal[sel])
        v_ref, b_ref = V8.apply_plan(V[h], keep, plan, gamma)
        v_ok = torch.allclose(v_c[0, h].float(), v_ref[keep_idx], atol=1e-4, rtol=1e-4)
        b_ok = bias_c is not None and torch.allclose(bias_c[0, h].float(), b_ref[keep_idx], atol=1e-5)
        check(f"seed {seed} head {h}: values equal the harness plan (gamma {gamma}, {len(plan)} pairs)", v_ok,
              f"max diff {(v_c[0, h].float() - v_ref[keep_idx]).abs().max():.2e}")
        check(f"seed {seed} head {h}: bias equals the harness plan", b_ok)
    return cache, K, V, Q


def test_decode(cache, K, V, Q):
    kv, L, d = K.shape
    g = Q.shape[0] // kv
    n0 = cache._get_existing_cache(0)[0].shape[-2]
    for step in range(3):
        cache.pre_attention_step(0)
        bias = cache.get_decode_bias(0)
        n = cache._get_existing_cache(0)[0].shape[-2]
        check(f"decode step {step}: bias covers cache + incoming token",
              bias is None or bias.shape == (1, kv * g, 1, n + 1), f"{None if bias is None else tuple(bias.shape)} vs {n + 1}")
        kn, vn = torch.randn(1, kv, 1, d), torch.randn(1, kv, 1, d)
        # attention of the new query over the cache + itself (pre-hook already pruned)
        cache.current_attention_scores[0] = cache.reduce_attention(torch.softmax(torch.randn(1, kv * g, 1, n + 1), -1))
        cache.update(kn, vn, 0)
        n2 = cache._get_existing_cache(0)[0].shape[-2]
        check(f"decode step {step}: bias length == cache length after update",
              cache.logit_bias[0].shape[-1] == n2, f"{cache.logit_bias[0].shape[-1]} vs {n2}")
    check("decode keeps the cache at budget (+1 per step before the next prune)",
          cache._get_existing_cache(0)[0].shape[-2] <= n0 + 1)


if __name__ == "__main__":
    test_candidates()
    for seed in (0, 1):
        cache, K, V, Q = test_prefill_merge(seed)
    test_decode(cache, K, V, Q)
    print(f"\n{'OK' if FAILS == 0 else str(FAILS) + ' failures'}")
    sys.exit(1 if FAILS else 0)
