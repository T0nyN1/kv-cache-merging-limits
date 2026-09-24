# KV merging, 2024–2026: what the field found, and what the theorem says about it

Survey conducted 2026-09-02 to answer one question: is the merging failure in
this project an artefact of the OT implementation, or is merging itself closed?
Companion to `docs/ot_kv_theory.md` (the theorem) and `core/consolidate.py`
(the constructive answer).

## 1. The landscape, by mechanism class

| class | works | merge rule | covered by the theorem? |
|---|---|---|---|
| **A. value-coefficient** | CaM (ICML'24), D2O, KVMerger ([2407.08454](https://arxiv.org/abs/2407.08454)), LOOK-M, WeightedKV (ICASSP'25) | evicted values folded into retained slots with fixed weights (attention/similarity/Gaussian kernel) | **yes — fully.** First-order error in the logit gap; ceiling e^{−σ²}, σ≈1.2 |
| **B. compensated** | KeepKV ([2504.09936](https://arxiv.org/abs/2504.09936)), SelKV ([2607.16213](https://arxiv.org/abs/2607.16213)), KVSlimmer ([2603.00907](https://arxiv.org/html/2603.00907v1)), AsymKV, SemantiCache ([2603.14303](https://arxiv.org/pdf/2603.14303)) | merged slot additionally carries a **mass channel** — vote counts scaling attention (KeepKV), a log-ratio logit bias (SelKV), or summed values with merged keys (KVSlimmer) | **no — outside the function class.** The theorem assumes a merge can only change values; a per-slot logit bias makes the slot represent e^{q·k+b}v |
| **C. trained** | DMC (Nawrot et al.'24), MLA-style architectures | model is fine-tuned/designed to emit mergeable KV | no — the model co-adapts; strong published results |
| **D. cross-layer** | MiniCache (NeurIPS'24), KVSharer | same token merged across adjacent layers | no — a different redundancy axis entirely |
| **E. reconstruction** | GRKV ([2605.31105](https://arxiv.org/pdf/2605.31105)) | fit synthesized representations to reproduce full-cache attention globally | no — this is the "synthesized anchors" direction the theory note left open |

## 2. What the class-B papers actually measure

The critical reading is their own ablations, not their headline tables:

- **SelKV** (the most careful): pure eviction 39.69 → blind merging **39.01**
  (merging *hurts* by −0.68) → + similarity gate 39.57 (still below eviction)
  → + attention compensation **39.81**. Net effect of the entire merging
  apparatus over pure eviction: **+0.12 LongBench** — inside the ±0.4
  conservation band we measured on 1,200 paired samples. Its margin over
  SnapKV: +0.02.
- **KeepKV**: "zero inference-perturbation" is exact **only at the
  merge-time queries** (votes p_i scale attention as o = Σ p_i s_i v_i / Σ p_i s_i;
  the merged key cannot track Σe^{q·k_j} as q varies — precisely the σ term).
  No ablation isolating merging from selection under identical rules.
- **KVSlimmer**: merges *adjacent* pairs (where σ is smallest — consistent
  with our Δpos measurements), sums values, merges keys with a Hessian-derived
  closed form. Explains K-homogeneity/V-heterogeneity spectrally, proves no
  bounds, does not discuss RoPE. Headline wins are vs H2O/CaM (+5.5/+6.8) —
  the weak baselines; no SnapKV-class comparison at matched budget, no
  merge-vs-evict ablation.
- **GRKV** argues *against* local merging (carrier-token bottleneck, shrinking
  feasible space) and regresses global representations instead.

**Verdict on the original question.** It was never the OT machinery: the
Sinkhorn plans solved the value-coefficient problem accurately, and every
2025-26 paper that gates and compensates carefully lands at +0.1–0.2 over its
own eviction baseline — exactly what a ceiling of e^{−σ²} with σ≈1.2 predicts.
The field converged, without stating it, on the two moves our theorem
identifies as the only ones available: *gate away dissimilar pairs* and
*compensate the mass*.

## 3. The crack the compensation channel opens — and its size

The theorem's exactness argument assumes the merged slot is a value edit under
unchanged attention. A per-slot logit bias b extends the representable family
to c·e^{q·k̄}: merging pair (1,2) with mass-weighted midpoint key and
b = E_w[log(a₁+a₂) − l̄] fits the pair's exp-mass exactly on observed queries
(softmax normaliser cancels), leaving only the **curvature residual**

    ε(q) = log(e^{l₁}+e^{l₂}) − l̄(q) − b  ≈  logcosh(Δl/2) − const,

**second order** in the logit gap where value-coefficient merging is first
order. For a gap std σ the residual std is ≈ 0.18σ² (vs σ): at the measured
adjacent-pair σ≈1.1 that is ≈0.22 — *under* the 0.285 break-even that
value-only merging misses by 4×. The pair error is also directly *measurable*
per pair from the stored window rows (`core/consolidate.py::pair_statistics`),
so the gate can threshold the deciding quantity itself instead of key cosine
(which tracks it at r=0.273 — the criterion every class-A/B paper gates on).

Implemented as v7's consolidation bypass (σ̂-gated, non-overlapping adjacent
pairs, freed slots admit more tokens at the same physical budget; merged slots
carry the bias into decode through a mask-injecting pre-hook):

- identical-pair merge is exactly lossless end-to-end (unit-tested);
- Qwen3-1.7B local: τ=0.3 consolidates ~28 of 77 middle slots per layer
  (**+36 % effective tokens** at the same physical cache), KL −2.5 %,
  ppl −2.0 % vs v7, NIAH 1.00 unchanged.

**Llama-3.1-8B validation (LongBench dev, paired vs v7, same physical budget):**

| budget | Δ LongBench | 95 % CI | NIAH | wikitext ppl |
|---:|---:|---|---:|---:|
| 5 % | **−0.72** | [−1.58, +0.09] | 1.00 (=) | 4.419 vs 4.431 (−0.3 %) |
| 2 % | **−1.83*** | [−3.00, −0.80] | 1.00 (=) | — |

Gate choice is irrelevant (σ̂ −0.72 vs cosine −0.73): once merges are
compensated and restricted to adjacent pairs, *which* similar pairs you pick
no longer matters — there is simply no end-task profit left to route toward.
Losses concentrate on multi-hop retrieval (2wikimqa −2.5, hotpotqa −1.7,
triviaqa −1.2 at 5 %; triviaqa −8.6 at 2 %): the merged slots are precisely
the high-attention pooled-block tokens QA decoding needs verbatim, and the
bias — fitted in-sample on the prefill window — drifts on the actual decode
queries, with no averaging across merged slots (the same missing term that
sank v5).

## 3b. The closing measurement

This was the strongest merging variant the theory permits: second-order
compensated, gated on the *measured* deciding quantity, merging only adjacent
pairs (where σ is smallest), exactly lossless on duplicates, +36 % effective
tokens. It improves the continuation-fidelity metrics (KL/ppl, where the
window queries resemble the decode queries, so the in-sample bias transfers)
and still loses LongBench — increasingly with compression. Combined with the
class-A theorem and the class-B papers' own +0.1-scale ablations, the
merging question is now closed at *both* orders for frozen RoPE models:
first-order (value-coefficient) merging cannot pay anywhere, and second-order
(compensated) merging pays only where its calibration queries match the
deployment queries — which long-context QA, the workload KV compression
exists for, does not satisfy.

## 3b-2. The matched-protocol merging table (Llama-3.1-8B, 5 % budget, paired)

Every merging class, measured against *its own* eviction base under one
protocol — the table the survey's argument rests on:

| operator | class | Δ LongBench vs own eviction | 95 % CI | Δ ppl |
|---|---|---:|---|---:|
| CaM (ICML'24, faithful port at published cadence) | A | **−0.14** | [−0.25, −0.04] | +0.003 % (differs on 30/400 samples) |
| σ̂-gated compensated consolidation @5 % (ours, strongest permitted) | B | **−0.72** | [−1.58, +0.09] | −0.3 % |
| — same @2 % | B | **−1.83** | [−3.00, −0.80] | — |
| SelKV (their own published ablation) | B | +0.12 | — | — |

Footnote on faithfulness: CaM's operator is streaming (one eviction per
decode step). Applied naively to a prefill-end mass eviction it *diverges*
(ppl 77.7 vs base 4.42 — thousands of superimposed v/32 smears, the v2
failure in another costume); the port applies it only at its published
cadence, where it is a measured no-op.

## 3c. The sigma mechanism generalises across model families

`experiments/sigma_families.py`, run 2026-09-02 (budget 0.15, oracle routing,
wikitext, probe = final 128 queries; A100):

| model | oracle median σ | frac ≤ 0.285 | frac ≤ 0.9 | verbatim-8x σ | same-token σ, adjacent → far |
|---|---:|---:|---:|---:|---|
| Qwen3-1.7B | 1.25 | 0.6 % | 17 % | 1.06 | 1.39 → 2.07 |
| Qwen2.5-7B-Instruct | 1.38 | 0.1 % | 15 % | 1.37 | 1.48 → 2.25 |
| Llama-3.1-8B-Instruct | 0.98 | 0.5 % | 44.5 % | 0.90 | 1.06 → 1.55 |
| Mistral-7B-Instruct-v0.3 | 0.94 | 5.1 % | 46 % | 0.88 | 1.07 → 1.55 |

Every family shows the same three signatures: the value-only break-even
fraction never exceeds 5 %, verbatim repetition never manufactures mergeable
keys, and σ grows monotonically with positional distance. The class-A closure
is now a claim about RoPE transformers, not about one model.

The family split is itself informative: Llama and Mistral sit a full 0.3-0.4
below the Qwen family in σ, with ~45 % of evicted tokens under the
compensated-pair reference (0.9) versus ~15 % for Qwen. That is why the
consolidation gate found a third of the budget mergeable on Llama — **and the
end task still lost.** With key geometry this permissive, the binding
constraint on second-order merging is not the geometry at all; it is the
calibration-query distribution shift of §3b. This sharpens the paper's causal
story: class A fails for geometric reasons, class B fails for distributional
ones, and the two failure modes are separable and separately measured.

## 4. Relation to the two candidate write-ups

- **Method paper ("v7 + consolidation")**: the novel elements over class B are
  (i) gating on the measured σ̂ rather than cosine, (ii) the break-even
  calibration derived from the same theory that closed class A, (iii) the
  controlled `consolidate_gate=cos` ablation showing the criterion matters.
- **Analysis paper ("why merging (mostly) cannot work")**: the taxonomy above
  + the class-A impossibility theorem + the class-B second-order analysis +
  re-reading of published ablations + the conservation finding. The
  constructive consolidation experiment is the paper's positive control:
  the theory predicts exactly which merges pay, and they do.

Sources: [KVMerger](https://arxiv.org/abs/2407.08454) ·
[KeepKV](https://arxiv.org/abs/2504.09936) ·
[SelKV](https://arxiv.org/html/2607.16213v1) ·
[KVSlimmer](https://arxiv.org/html/2603.00907v1) ·
[SemantiCache](https://arxiv.org/pdf/2603.14303) ·
[GRKV](https://arxiv.org/pdf/2605.31105) ·
[LightVLM](https://arxiv.org/abs/2509.00419) ·
[KV-cache management survey](https://arxiv.org/html/2412.19442v3) ·
[Awesome-KV-Cache-Compression](https://github.com/October2001/Awesome-KV-Cache-Compression)
