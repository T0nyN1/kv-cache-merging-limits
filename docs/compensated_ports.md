# KeepKV and SelKV — matched-protocol ports

Neither paper releases code. Both ports follow the paper text (equations and Algorithm 1 quoted in the module
docstrings of `baselines/keepkv.py` and `baselines/selkv.py`). This file lists every place the paper is silent and
the choice made, so a reader can judge whether a choice could have moved the result.

Protocol shared with every row of Table 2: Llama-3.1-8B-Instruct, 5 % KV budget, sink 4, recent window 10 % of the
budget, middle truncation at 8 000 tokens, LongBench dev (first 50 samples of 8 English tasks, 400 paired), one token
layout shared across heads, compression at the end of prefill and the protocol's decode-time maintenance.

## KeepKV (arXiv 2504.09936 v2, AAAI 2026)

| item | paper | port |
|---|---|---|
| votes, decode attention | p init 1, p_r = p_e + p_c; o = Σ p s v / Σ p s (Eqs. 3, 6) | as written; log p added to the logits |
| candidates | cosine of keys, argmax over retained, U > T = 0.8 (Alg. 1) | as written, per KV head |
| merge | ZIP-Merging (Alg. 1 / Eq. 7) | as written, in log space |
| scores | s = exp(q·k/√d), EMA over a window (Eq. 8) | as written, causal |
| EMA α, w | **not stated** | α = 0.9, w = 32 (33 queries) |
| GQA | **not discussed** | s averaged over a KV head's query heads; votes per KV head |
| eviction base | PyramidInfer allocation | pooled SnapKV of the protocol (PyramidInfer not implemented here) |
| retained set K_c | "retained cache" | the whole retained cache, sink and recent window included (`targets=middle` is a sensitivity) |
| several e → one c | Alg. 1 merges and replaces c | merged in position order; the entry's score becomes (w_e + w_c)/p_r |
| no candidate above T | **not stated** | evicted |
| key scale λ | Alg. 1 places no restriction on it | applied as written, negative values included; two numerical guards skip a merge and both are counted: a near-zero denominator \|π_e ln ŝ_e + π_c ln ŝ_c\| < 1e-6 (`skip_undefined` — this covers 0/0 but also small finite denominators, whose λ would be enormous) and a merged key that is not finite in the cache dtype (`skip_nonfinite`). `lam_policy=positive` and `cap:X` are sensitivities, not the published operator |
| merge-then-evict prose variant (§3.3) | mentioned, not in Alg. 1 | not implemented (Alg. 1 order) |
| key rewrite | Alg. 1 replaces k_c by the scaled mixture | as written; `keys=fixed` is a labelled sensitivity that keeps everything else and leaves the key in place |
| periodic re-compression | title only, no interval | none |
| eviction base, second selector (added 2026-09-19) | PyramidInfer | `selector=selkv` runs the identical merge on SelKV's own selection (Eqs. 1-2 of the SelKV port, through `SelKVSelectorMixin`, decode-time maintenance included), so the keys-fixed contrast can be read on a base other than pooled SnapKV. K_c stays the whole shared retained cache (KeepKV's rule), not SelKV's head-specific S_h. Gate: `keepkv;selector=selkv;merge=false` must equal `selkv;selector=selkv;merge=false` per sample, bit for bit (`experiments/test_compensated_ports.py`, unit test 5a and `--bitcheck`) |

## SelKV (arXiv 2607.16213 v1)

| item | paper | port |
|---|---|---|
| score | c = Σ_{q∈W} A · ‖v‖, W = 32, avg-pool K = 5 (Eq. 1) | as written (`selector=selkv`) |
| selection | per-KV-head top-m_s, union, trim by mean score (Eq. 2) | as written; m_s = m − m_r in the paper, i.e. the middle budget once the protocol's protected regions are used |
| recent window | m_r = 16, no sink | the protocol's sink 4 and recent window, for every method |
| routing | argmax over S_h ∩ B(j) of A_{h,j,i} (Eq. 3) | buckets = floor(j/32); A from the full prefill attention; S_h = positions kept after union-and-trim that were in head h's own top-m_s, plus the protected sink and recent window (with `selector=snapkv`, the shared retained set) |
| empty bucket | **not stated** | the evicted token is dropped (`fallback=drop`, the port). `fallback=global` (added 2026-09-19, reviewer request) applies Eq. 3 over the whole S_h instead, π_h(j) = argmax_{i∈S_h} A_{h,j,i}, with the same gate, merge and compensation; implemented from the `fallback_topk` = 128 most-attended positions of each token's head-summed prefill attention row (the full 8k×8k attention cannot be held for 32 layers), exact whenever one candidate lies in S_h — the protected sink is in every S_h and is the top-attended position of nearly every query — and counted as `fallback_miss` otherwise. `merge_stats.jsonl` carries `routed_bucket`, `fallback_routed`, `fallback_miss` |
| gate | g = max(cos(v_j, v_π(j)), 0) (Eq. 4) | as written |
| merge | attention-weighted, gated, values only (Eq. 5) | as written |
| compensation | R = (a_i + Σ g a_j)/a_i, averaged over KV heads, α log R, α = 0.5 (Eq. 6) | as written, not clipped; where a routed-to kept token has zero window attention (bf16 underflow) a_i is floored at 1e-8 and the incidence counted |
| RoPE repositioning (Eq. 7) | off by default | off |
| GQA attention | **not stated** | summed over the query heads of a KV head (cancels in every ratio) |
| second base | — | `selector=snapkv`: the same merge on the protocol's pooled-SnapKV set |
| cadence | "compresses the KV cache once after prefill" | merge once after prefill; the protocol's decode-time budget maintenance (as for every method) ranks by head-summed attention × head-mean value norm, average-pooled — an adaptation, identical in the own-selector merge-off pairing |
| chunked prefill | — | raises (never silently disables the merge) |

## Inconsistencies in the papers themselves

- KeepKV's perturbation-free identity is exact only when s_e and s_c are the unnormalised scores of ONE query. With
  the EMA of Eq. 8 there is in general no single query whose log-scores are the two EMA values, so the identity need
  not hold at any real decode query. The unit test validates the algebra with one query and α → 0; it does not
  validate the EMA construction's claim, and the paper table must not describe KeepKV as zero-perturbation.
- KeepKV's Eq. 8 sums a truncated window of w + 1 terms but divides by 1 − α^t, the correction for an EMA
  accumulated from the start: at t = L the weights sum to about 1 − α^{w+1}, not 1. The port implements Eq. 8 as
  written; the constant factor shifts every ln ŝ by the same amount, which changes the key scale λ.
- SelKV describes one shared kept set (union and trim) but routes within the head-specific S_h; the port takes
  Eq. 3 literally (see the routing row).

## What each row means

- **KeepKV operator on pooled SnapKV** — not KeepKV against its published PyramidInfer base. Paired against the
  existing pooled-SnapKV run, which the merge-off port reproduces bit for bit.
- **SelKV operator on pooled SnapKV** — same base and pairing.
- **SelKV, own selector** — paired against `selkv;selector=selkv;merge=false`, NOT against SnapKV.

## Per-sample statistics

Both ports append one JSON line per sample to `save_dir/merge_stats.jsonl` (`MERGE_STATS_PATH`, set by `main.py`
from `save_dir`), labelled with the method string, and print the same counters to stdout. KeepKV reports `evicted`,
`candidates` (cosine above T), `merged`, `skip_undefined`, `skip_policy`, `skip_nonfinite`, `lam_negative`,
`lam_abs_gt2/gt10`, `lam_absmax_seen` and per-layer λ quantile sums; SelKV reports `evicted`, `routed`,
`dropped_empty_bucket`, `gate_zero`, `gate_sum`, `floored_attention_targets`, `logR_max` and (own selector)
`union_size`. A run whose `merged` or `routed` is zero is a plain eviction run and must not be reported as a merge.

## What the runs showed (2026-09-16, Llama-3.1-8B, LongBench dev, 400 paired samples)

At a 5 % budget both operators are far less active than their published settings imply, and the counters say why:

| | KeepKV | SelKV (own selector) | SelKV (pooled SnapKV) |
|---|---|---|---|
| evicted tokens | 5.89e8 | 5.89e8 | 5.89e8 |
| acted on | 4.89e7 merged (8.3 %) | 2.24e7 routed (3.8 %) | 3.18e7 routed (5.4 %) |
| dropped | cosine below T | 96.2 % (no kept token in the 32-bucket) | 94.6 % |
| guards | 2 skips in 4.89e7 (near-zero denominator); 0 non-finite | 0 floored targets | 1 floored target |

KeepKV's merges concentrate: over 100 samples (triviaqa and qasper, `runs/ports_diag`) the most-merged slot of a
layer absorbs a median of 104 evicted tokens (max 184), so its logit bias reaches log 104 = 4.65, and a median of 248
slots per layer -- about 7 % -- carry five or more votes. Only 0.20 % of merges target a sink and 1.27 % the recent
window, so the protected regions are not what the operator disturbs. The key scale is well behaved at this protocol:
median lambda 1.006 per layer, p99 1.264, 0.12 % negative, 0.03 % above 10 in absolute value, and the two numerical
guards fired twice in 4.89e7 merges. That run also reproduces `runs/ports_a` sample for sample on both tasks, with the
counters added, so the counters changed nothing.

### The deltas

Every row is a 400-sample paired comparison against its own eviction base (`experiments/ports_report.py`):

| row | run | base | Δ | 95 % CI |
|---|---|---|---|---|
| SelKV operator, own selector, 5 % | `ports_b` | same selector, merge off | +0.10 | [-0.40, +0.58] |
| SelKV operator on pooled SnapKV, 5 % | `ports_c` | `dev8k` pooled SnapKV | +0.02 | [-0.26, +0.31] |
| SelKV operator, own selector, 25 % | `ports_d25` | same selector, merge off | +0.31 | [+0.01, +0.66] |
| KeepKV operator on pooled SnapKV, 5 % | `ports_a` | `dev8k` pooled SnapKV | -9.75 | [-11.98, -7.66] |
| ... same merges, retained keys left in place | `ports_h` | `dev8k` pooled SnapKV | **+0.37** | [-0.31, +1.15] |
| ... vote bias off | `ports_f` | `dev8k` pooled SnapKV | -9.39 | [-11.50, -7.42] |
| ... targets restricted to the middle | `ports_e` | `dev8k` pooled SnapKV | -9.88 | [-12.11, -7.74] |
| KeepKV operator, 25 % | `ports_g25` | `keepkv;merge=false` at 25 % | -4.24 | [-5.66, -2.89] |

The `keys=fixed` row is the isolation: identical candidates, identical merge weights, identical values and votes
(4.895e7 merges in both, and `experiments/test_compensated_ports.py` checks that the values and the bias are
bit-identical between the two modes), with only the key rewrite removed. It moves the operator by +10.12 [+8.00, +12.38], positive on all eight tasks.

At 25 % SelKV routes 21.5 % of evicted tokens (against 3.8 % at 5 %) and KeepKV merges 32.6 % (8.3 % at 5 %), but per
*retained slot* KeepKV's merges fall from 1.20 to 0.74, which is the quantity its loss tracks.

## Evaluator note

At decode (query length 1, batch 1, no padding) the evaluator has always replaced the attention mask with `None` for
every method; the removed mask is all zeros there, so it changes nothing. For the compensated caches the per-slot bias
is passed in its place. This path is shared with the consolidation and v8 rows already in the paper.

## Verification

`experiments/test_compensated_ports.py` (42 checks): with merge off the KeepKV port and the SelKV port with
`selector=snapkv` are bit-identical to `SnapKVCache` through prefill and decode (the SelKV own selector is not, by
design, and is paired with its own merge-off run); the registered method strings resolve to the protocol's bases;
KeepKV applies a negative key scale as written and it stays perturbation-free for the scoring query, while 0/0 is
skipped and both sensitivity policies work; SelKV routes within the head's own S_h; both ports refuse to degrade
silently (missing queries, chunked prefill); Eq. 8 matches a direct float64 evaluation; ZIP-Merging is perturbation-free for the scoring
query to ~1e-8, including four sequential merges into one entry; SelKV Eqs. 3-6 match an explicit per-token loop; the
bias stays aligned with the cache during decode. The round-2 additions check that a key scale whose merged key
overflows float16 is skipped and the cache stays finite; that a routed-to target with zero *and* with 1e-10 window
attention is floored and counted; that the per-head sets `_keep_middle` builds are exactly the ones `_merge_layer`
routes with and are then freed; that a sample with no eviction at prefill does not raise; that a chunked
`selector=snapkv, merge=False` call still returns SnapKV's scores; and that the merge counters reach
`MERGE_STATS_PATH`. The evaluator's window-query capture was checked against the
independent capture in `experiments/v8_offline.py`: identical to the bit on Qwen3-1.7B (layers 0, 13, 27, float32)
and on Llama-3.1-8B-Instruct (layers 0, 15, 31, bfloat16) — the target model and dtype.

Both ports went through two rounds of adversarial code review (12 findings, then a verification of those fixes
with five further corrections and the required-run list); every finding was dispositioned before the runs.
