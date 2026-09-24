# Why optimal transport does not (yet) help — a derivation

Every OT-KV version so far merges evicted values into anchors with a
query-independent coefficient, and chooses which anchor by cosine distance
between post-RoPE keys. Measured end to end, the merge is worth 1-3 % and has
never reached significance. This note derives why, and what the derivation says
to change.

Notation: for a query `q`, the dense attention output is

```
o(q) = sum_t a_t(q) v_t ,   a_t = e^{q·k_t/√d} / Z ,   Z = sum_t e^{q·k_t/√d}
```

Compression keeps an anchor set `A`, evicts `E`, and replaces the anchor values
with `v'_j = v_j + δ_j` for some query-independent `δ`.

---

## 1. What eviction actually costs

**Lemma 1.** Let `ρ(q) = Σ_{i∈E} a_i(q)` be the evicted attention mass, and let
`v̄_A(q)`, `v̄_E(q)` be the attention-weighted value centroids of the retained
and evicted sets. Then

```
o_evict(q) − o(q) = ρ(q) · [ v̄_A(q) − v̄_E(q) ]
```

*Proof.* Renormalisation gives `â_j = a_j/(1−ρ)`, so `o_evict = v̄_A`. Also
`o = (1−ρ) v̄_A + ρ v̄_E`. Subtract. ∎

Two consequences worth stating plainly. Eviction is harmless when the evicted
values look like the retained ones *on average under the query* — dropping
redundant content costs nothing. And the error scales with `ρ`, so it grows as
the budget tightens, which is what the budget sweep shows.

---

## 2. What a merge can and cannot express

Merging adds `Σ_j â_j(q) δ_j`, a **linear functional of the retained attention
distribution**. Cancelling the error for every `q` therefore requires

```
Σ_j e^{q·k_j/√d} δ_j  ∝  Σ_i e^{q·k_i/√d} v_i        for all q
```

**Theorem (exactness).** Since `{q ↦ e^{q·k/√d}}` are linearly independent
functions for distinct `k`, this holds only if every evicted key coincides with
an anchor key.

So merging is exact exactly when it is unnecessary, and approximate everywhere
else. The interesting question is how good the approximation can be.

---

## 3. The quantity that decides it

Route evicted token `i` to anchor `j` and write `Δ = k_i − k_j`. The token's
true weight relative to its anchor is

```
R(q) = a_i/a_j = e^{q·Δ/√d}
```

A merge supplies a **constant** `c` in place of `R(q)`. Model `q·Δ/√d` over the
query distribution as `N(μ, σ²)`, so `R` is log-normal with

```
σ² = Δᵀ Σ_q Δ / d ,     Σ_q = second moment of the queries that will read the cache
```

The best constant is `c* = E[R] = e^{μ+σ²/2}`, with residual `Var(R)`. Not
merging is `c = 0`, with error `E[R²]`. Their ratio:

```
   error(best merge)     Var(R)      e^{σ²} − 1
   ───────────────── =  ───────  =  ────────────  =  1 − e^{−σ²}
   error(eviction)       E[R²]        e^{σ²}
```

**Proposition.** The largest fraction of squared error a query-independent merge
can remove is `e^{−σ²}`, and `σ` is a Mahalanobis distance between the two keys
under the query second moment.

This is the whole story:

- `σ → 0` (duplicate keys): merging removes everything, consistent with the
  exactness theorem.
- `σ = 1`: merging removes 37 % at best.
- `σ = 1.5`: 11 %.
- and the bound is for the *ideal* constant; any error in estimating `c` eats
  into it further.

---

## 4. Measurement — `experiments/merge_theory.py`

Qwen3-1.7B, post-RoPE keys, queries captured from a held-out continuation,
144 (layer, head) samples, budget 0.15:

| routing rule | mean σ | attainable reduction `e^{−σ²}` |
|---|---:|---:|
| cosine distance (v1–v4) | 1.500 | **21.1 %** |
| query-whitened distance | 1.211 | **29.0 %** |

and, most damningly,

```
correlation( cosine distance , true σ )  =  0.273
```

**The cost matrix every OT-KV version has minimised is nearly uncorrelated with
the quantity that determines whether a merge helps.** With cosine routing the
ceiling is 21 % of the squared error of the term the merge targets; after
coefficient-estimation noise the measured 1-3 % end-to-end is exactly what this
predicts. The optimal transport machinery has been solving the wrong problem
accurately.

This also explains the two earlier negative results without any extra
assumption. Softer plans were worse (`ε = 0.5` scored below not merging at all)
because spreading a value over several anchors averages several *wrong*
coefficients. Merging keys was worse because it lowers cosine distance while
leaving σ — the thing that matters — untouched.

---

## 5. What the derivation says to do instead

`σ² = Δᵀ Σ_q Δ / d` is estimable without any model-specific code, because the
attention weights already contain it. For the last `W` queries,

```
log A[w, i] − log A[w, j] = q_w·(k_i − k_j)/√d
```

— the per-query normaliser cancels in the difference. Therefore, writing
`z_t ∈ R^W` for the centred log-attention column of token `t` divided by `√W`:

```
σ²_ij = ‖ z_i − z_j ‖²          (the transport cost)
c*_ij = exp( μ_ij + σ²_ij / 2 ) (the merge coefficient)
```

Both the cost and the coefficient come from the same measured statistic, with no
Gaussian assumption needed for the cost and only a log-normal one for the
coefficient. The cost is an ordinary Euclidean distance, so the existing
top-`r` candidate search and sparse Sinkhorn apply unchanged — only the space
they operate in changes, from raw key space to **logit-response space**.

The prediction to test: replacing the cosine cost with this one should raise the
attainable reduction from 21 % to ~29 % and, unlike every knob tried so far,
should make the merge term itself measurable.

Implementation: `core/ot_kv_v5.py`, evaluated in `docs/results.md`.

---

## 6. The correction that closes the direction

§3 treats each evicted token independently, and on that basis the whitened
metric should have helped. It did not: `core/ot_kv_v5.py`, which implements
exactly the cost and coefficient derived above, is *monotonically worse* than not
merging (KL 0.466 no-merge, 0.472 / 0.512 / 0.576 at merge strength 0.2 / 0.5 /
1.0). The per-token analysis is missing a term.

**The missing term is that the coefficient errors do not cancel.** The error for
token `i` is `c_i − R_i(q)`, and every token is read by the *same* `q`. When `q`
lands in a direction where the `Δ_i` project positively, every coefficient is
wrong in the same direction at once. So the injected error scales with the
number of merged tokens `n`, exactly like the bias it is meant to remove —
there is no `1/√n` averaging to bank on.

Both terms therefore scale with `n`, and merging is net-positive only if the
relative coefficient error is smaller than the relative bias:

```
CV = sqrt(e^{σ²} − 1)   <   |v̄_A − v̄_E| / ‖v‖
```

**Measured** (Qwen3-1.7B, 144 layer-head samples, budget 0.15):

```
|v̄_A − v̄_E| / ‖v‖ = 0.291        =>   break-even at σ ≤ 0.285
```

The value centroids of the retained and evicted sets are close — averaging
thousands of value vectors concentrates them — so the bias available to be
removed is small, while `σ` is not.

### How far off is it, across every budget

Oracle routing (each evicted token sent to its globally best anchor, so no cost
function can do better):

| budget | median σ | tokens with σ ≤ 0.285 | ceiling `e^{−σ²}` |
|---:|---:|---:|---:|
| 0.05 | 1.401 | 0.05 % | 0.206 |
| 0.15 | 1.236 | 0.14 % | 0.283 |
| 0.30 | 1.121 | 0.19 % | 0.341 |
| 0.50 | 1.049 | 0.28 % | 0.392 |
| 0.80 | 0.998 | 0.33 % | 0.413 |
| 0.95 | 0.983 | 0.40 % | 0.418 |

σ needs to fall by **4.2x** to break even, and it plateaus near 1.0 even at a 95 %
budget where almost nothing is being evicted. Fewer than half a percent of
evicted tokens have a usable stand-in *anywhere in the cache*.

**Conclusion.** Value merging in KV compression is not an engineering problem
with the cost function — the oracle and the whitened metric give the same 0.07 %
at budget 0.15. Post-RoPE key space contains no near-duplicates at the
resolution the softmax resolves, so there is nothing for any transport plan to
exploit. This closes the merging direction; it is not a matter of trying a
better ε, a better `top_r`, or a better metric, all of which were tried.

### What is not closed

Every measured gain in this project came from **which tokens are kept**, not from
what happens to the evicted ones. That side is still an optimal-transport
problem, and a better-posed one: choosing anchors that minimise the Wasserstein
distance between the empirical key measure and the retained one is a discrete
quantisation problem, the anchors are its support, and attention mass is the
weight. `select_mode="kmeans:P"` is a crude solver for it and already beats
heavy-hitter selection at every budget (`docs/experiments.md` §5). The honest
version of "OT-KV" is transport used to *place* the anchors, not to move values
into them.


---

## 7. How far the closure actually extends

§6 measured one model, one corpus, one context length. Three of the four ways
that conclusion could have been an artefact were tested; all three held.

**Context length.** More tokens means more candidate anchors, so if near-
duplicates ever appear this is where they would. Oracle routing, budget fixed at
15 %:

| context | anchors | median σ | σ ≤ 0.285 |
|---:|---:|---:|---:|
| 1 024 | 153 | 1.265 | 0.07 % |
| 2 048 | 307 | 1.169 | 0.10 % |
| 4 096 | 614 | 1.154 | 0.12 % |
| 8 192 | 1 228 | 1.183 | 0.12 % |
| 16 384 | 2 457 | 1.184 | 0.12 % |

σ **saturates**. Sixteen times more candidates buys nothing.

**Text type.** Repetitive text is the most plausible place duplicate keys could
live. Context 4096, budget 15 %:

| text | median σ | σ ≤ 0.285 |
|---|---:|---:|
| wikitext prose | 1.163 | 0.07 % |
| Python source (this repository) | 1.220 | 0.02 % |
| one passage repeated verbatim 8x | 1.208 | 0.02 % |

**Literally repeating a passage does not produce mergeable keys.**

**Why.** σ between two positions holding the *same token id*, by positional
distance:

| \|Δpos\| | pairs | median σ | σ ≤ 0.285 |
|---:|---:|---:|---:|
| 0–8 | 224 | 1.135 | 1.79 % |
| 8–64 | 736 | 1.514 | 0 % |
| 64–512 | 6 272 | 1.704 | 0 % |
| 512–4 096 | 21 568 | 1.812 | 0 % |

Two mechanisms, both fatal. Keys are **rotated by position**, so σ grows
monotonically with distance — RoPE actively manufactures the difference that
merging needs to be absent. And keys are **contextual**: even adjacent repeats
of the same token sit at σ = 1.135, four times the break-even, because the
hidden state depends on the whole prefix. Identical content simply does not
produce identical keys.

This is a mechanism rather than a measurement, so it should hold for any
rotary-position transformer, and it explains why value-merging work in general
reports small gains rather than the large ones the redundancy statistics of
key space appear to promise.

### What remains genuinely open

1. **Query-dependent coefficients.** The theorem covers merges whose coefficient
   is fixed at compression time. A coefficient computed at decode from the real
   query escapes it — but computing one costs what keeping the token costs, so
   no practical method does this.
2. **Synthesised keys.** The exactness argument assumes anchors keep the keys the
   model produced. Allowing an anchor to carry an optimised key does not make
   the problem exactly solvable (a log-sum-exp of distinct exponentials is not an
   exponential), but it could improve the approximation. The naive version —
   moving the key to the cluster centroid — was measured and hurt; an optimised
   version has not been tried.
3. **Other models.** One model measured. The mechanism above predicts the result
   generalises to any RoPE transformer; that prediction is untested.
4. **Selection.** Nothing here closes the use of transport to *place* anchors.
   That remains open, and it is where every measured gain in this project came
   from.
