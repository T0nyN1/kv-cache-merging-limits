"""OT-KV v6 — transport as anchor *placement*, not value movement.

`docs/ot_kv_theory.md` closes the merging direction: a query-independent merge
is net-positive only where `sqrt(e^{sigma^2}-1) < |v_A - v_E| / ||v|| = 0.29`,
i.e. `sigma <= 0.285`, and oracle routing achieves `sigma >= 0.98` at every
budget from 5 % to 95 %, with under 0.4 % of evicted tokens clearing the bar.
There is nothing for a transport plan to move.

What the same derivation leaves open is where to *put* the anchors. Lemma 1 says

    o_evict(q) - o(q) = rho(q) * [ v_A(q) - v_E(q) ]

so the error is a product of two factors: the evicted attention mass `rho`, and
the gap between the retained and evicted value centroids. Heavy-hitter selection
(H2O, SnapKV) minimises the first and ignores the second; pure coverage
minimises the second and ignores the first. Neither is the objective.

v6 minimises a surrogate for the product: the **attention-mass-weighted
quantisation error of the token measure**,

    min_{A, |A| = m}   sum_t  s_t * d( z_t , A )^2

which is a discrete Wasserstein-2 quantisation problem — the anchors are the
support of an m-atom measure approximating the empirical one, and attention mass
is the weight. Weighting by `s_t` is what couples the two factors: mass pulls
the support towards the tokens queries actually read (small `rho`), while the
quantisation term keeps the support spread over the response distribution
(small centroid gap).

The space matters as much as the objective. `d` is measured in **logit-response
space**, not raw key space: `z_t` is token `t`'s centred log-attention column
over the last `W` prefill queries, for which

    || z_i - z_j ||^2  =  Var_w( log A[w,i] - log A[w,j] )  =  sigma_ij^2

the exact quantity the theory identifies. Raw key cosine, which v1-v4 clustered
in, correlates with it at only 0.273.

No values are modified. The cache after compression is a subset of the original
tokens, so nothing can blow up and the method composes with any decode-time
policy.
"""

from typing import Optional

import torch

from core.ot_kv_v3 import EPS
from core.ot_kv_v5 import OTKVv5Cache, logit_space_embedding


def mass_weighted_quantise(z: torch.Tensor, mass: torch.Tensor, budget: int,
                           iters: int = 4, heavy_frac: float = 0.0,
                           seed_from_mass: bool = True) -> torch.Tensor:
    """Mass-weighted k-medoids in `z` space; returns (B, H, budget) token indices.

    z    : (B, H, L, D) embedding the distance is measured in
    mass : (B, H, L)    attention mass, the weight of each atom

    Lloyd iterations give the centroids of the weighted measure; each centroid is
    then snapped to the nearest *real* token, because the cache can only hold
    keys the model actually produced.
    """
    b, h, L, D = z.shape
    budget = min(budget, L)
    n_heavy = min(budget, int(budget * heavy_frac))
    n_cl = max(1, budget - n_heavy)

    if seed_from_mass:
        # k-means++ style seeding biased by mass: start from the heaviest atoms,
        # which is both a good initialisation and keeps the seeds informative
        seed = torch.topk(mass, n_cl, dim=-1).indices
    else:
        seed = torch.linspace(0, L - 1, n_cl, device=z.device).long().view(1, 1, -1).expand(b, h, -1)
    centroids = torch.gather(z, 2, seed.unsqueeze(-1).expand(-1, -1, -1, D))

    w = mass.unsqueeze(-1)
    for _ in range(max(1, iters)):
        assign = torch.cdist(z, centroids).argmin(-1)                      # (B,H,L)
        oh = torch.zeros(b, h, L, n_cl, device=z.device, dtype=z.dtype)
        oh.scatter_(-1, assign.unsqueeze(-1), 1.0)
        oh = oh * w
        num = torch.einsum("bhlc,bhld->bhcd", oh, z)
        centroids = num / oh.sum(-2).unsqueeze(-1).clamp_min(EPS)

    medoids = torch.cdist(centroids, z).argmin(-1)                         # (B,H,n_cl)

    picked = torch.zeros(b, h, L, dtype=torch.bool, device=z.device)
    picked.scatter_(-1, medoids, True)
    if n_heavy:
        picked.scatter_(-1, torch.topk(mass, n_heavy, dim=-1).indices, True)

    # keep everything picked, then top up with the heaviest leftovers
    ranked = mass + picked.to(mass.dtype) * (mass.amax(dim=-1, keepdim=True) + 1.0)
    return torch.topk(ranked, budget, dim=-1).indices.sort(dim=-1).values


class OTKVv6Cache(OTKVv5Cache):
    """Wasserstein quantisation of the KV measure in logit-response space.

    Merging is off by design. `quant_space` and `heavy_frac` exist so the two
    claims -- that the space matters and that mass weighting is what balances
    `rho` against coverage -- can each be ablated.
    """

    def __init__(self, **kwargs):
        defaults = dict(merge=False, compress_interval=4, observation_window=32,
                        select_scope="global", per_head=True)
        defaults.update(kwargs)
        super().__init__(**defaults)
        self.quant_space = str(defaults.get("quant_space", "logit"))
        self.quant_iters = int(defaults.get("quant_iters", 4))
        # A small heavy-hitter floor. Pure mass-weighted quantisation collapsed
        # on one document at budget 0.075 (KL 1.90, ppl 109) when Lloyd landed in
        # a degenerate configuration; reserving a quarter of the budget for the
        # top-mass tokens bounds the damage at no measurable cost elsewhere.
        self.heavy_frac = float(defaults.get("heavy_frac", 0.25))
        self.mass_power = float(defaults.get("mass_power", 1.0))
        self.value_weight = float(defaults.get("value_weight", 1.0))

    def _compress_layer(self, layer_idx: int, total_tokens: int, reserve_tokens: int = 0,
                        allow_merge: bool = True, allow_select: bool = True):
        win = self._logattn.pop(layer_idx, None)
        if not allow_select or (win is None and self.quant_space == "logit"):
            # decode-time maintenance, or no window captured: inherited path
            return super(OTKVv5Cache, self)._compress_layer(
                layer_idx, total_tokens, reserve_tokens, allow_merge, allow_select)

        middle_k, middle_v = self.get_middle_cache(layer_idx)
        if middle_k is None or middle_k.shape[-2] == 0:
            return
        budget = max(0, int(self.get_middle_budget(layer_idx, total_tokens)) - reserve_tokens)
        if budget >= middle_k.shape[-2]:
            return

        seq_len = self._get_existing_cache(layer_idx)[0].shape[-2]
        ms, me = self.sink_size, max(self.sink_size, seq_len - self.recent_size)
        scores = self.attention_scores[layer_idx]
        mid_scores = scores[..., ms:me]
        if mid_scores.dim() == 1:
            mid_scores = mid_scores.view(1, 1, -1).expand(middle_k.shape[0], middle_k.shape[1], -1)

        if self.quant_space in ("logit", "joint"):
            if win is None or win.shape[-1] != seq_len:
                return super(OTKVv5Cache, self)._compress_layer(
                    layer_idx, total_tokens, reserve_tokens, allow_merge, allow_select)
            z, _ = logit_space_embedding(win[..., ms:me])          # (B,H,Lmid,Nq)
            if self.quant_space == "joint":
                # The error is rho(q) * [v_A(q) - v_E(q)]: the first factor lives
                # in response space, the second in *value* space. Two tokens with
                # the same response are interchangeable for rho but not for the
                # centroid gap, so the quantiser has to see both. Each block is
                # normalised to unit mean norm before mixing, otherwise the value
                # block (||v|| ~ 11) would swamp the response block (~1).
                v = middle_v.float()
                zs = z / z.norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True).clamp_min(EPS)
                vs = v / v.norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True).clamp_min(EPS)
                z = torch.cat([zs, self.value_weight * vs], dim=-1)
        else:
            z = torch.nn.functional.normalize(middle_k.float(), dim=-1)

        mass = mid_scores.float().clamp_min(EPS)
        if self.mass_power != 1.0:
            mass = mass.pow(self.mass_power)
        if self.select_scope == "global":
            mass = mass.sum(dim=1, keepdim=True).expand_as(mass)

        keep = mass_weighted_quantise(z, mass, budget, iters=self.quant_iters,
                                      heavy_frac=self.heavy_frac)

        g4 = keep.unsqueeze(-1).expand(-1, -1, -1, middle_k.shape[-1])
        new_k = torch.gather(middle_k, 2, g4)
        new_v = torch.gather(middle_v, 2, g4)
        new_scores = torch.gather(mid_scores, 2, keep)

        self.replace_middle_cache(layer_idx, new_k, new_v)
        if scores.dim() == 1:
            new_scores = new_scores.sum(dim=(0, 1))
        self.attention_scores[layer_idx] = torch.cat(
            [scores[..., :ms], new_scores, scores[..., me:]], dim=-1)
