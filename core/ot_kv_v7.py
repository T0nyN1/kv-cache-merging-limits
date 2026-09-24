"""OT-KV v7 — budget transport: spend every slot where the evicted mass is.

Where v7 comes from
-------------------
`docs/ot_kv_theory.md` closes value merging: a query-independent merge pays only
below sigma <= 0.285 and post-RoPE keys sit at sigma ~= 1.2 everywhere, so v7
does not merge. What the same Lemma 1 leaves is the *selection* problem:

    o_evict(q) - o(q) = rho(q) * [ v_bar_A(q) - v_bar_E(q) ]

The squared error a single evicted token contributes is, to first order,

    || a_i(q) * ( v_i - v_bar_A(q) ) ||^2  ~  a_i(q)^2 * || v_i - v_bar ||^2

so the right per-token score is not attention mass alone but attention mass
times how far the token's value sits from the retained centroid, and the right
*budget* is not "the same number of slots per layer" but whatever allocation
equalises the marginal evicted mass across layers. Three mechanisms follow, all
of them selection-side, all cheap, none of them a merge:

1. **Value-magnitude weighting** (`vnorm_tau`). Score each token by
   `attention * ||v||^tau`. Dropping a high-attention token whose value is tiny
   costs little; dropping a moderate-attention token with a large, unusual value
   costs a lot. This is the `||v_i - v_bar||` factor of Lemma 1 with the cheap
   uncentred norm standing in for the centred one.

2. **Local max-pooling of the selection score** (`pool_kernel`). The window
   score estimates E[a_i] for *future* queries from ~32 observed ones. Future
   queries hit the same semantic unit but not the same token position, so the
   per-position estimate has positional jitter; taking the max over a small
   neighbourhood keeps the whole unit a spike belongs to instead of one token
   of it. (SnapKV ships this and this framework's re-implementation had dropped
   it; v7 restores it for our method, and `snapkv;pool_kernel=7` restores it
   for the baseline so the comparison stays honest.)

3. **Cross-layer water-filling** (`layer_alloc="waterfill"`). Uniform per-layer
   budgets equalise token *counts*; the objective wants to equalise marginal
   evicted *mass*. Normalise each layer's (attention x vnorm) middle scores by
   the layer's total mass, pool the per-slot marginal gains across layers, and
   keep the globally largest `n_layers * uniform_budget` of them, subject to a
   per-layer floor and cap. This is a transportation problem -- moving a fixed
   stock of slots to layers -- whose LP is solved exactly by the greedy because
   the objective is separable and concave. Early layers (diffuse attention)
   draw more budget, late layers (attention sinks) less: PyramidKV's hand-tuned
   ramp is the shape this allocation discovers on its own, per prompt.

Everything else is inherited from the v3/v4 engine: dual score streams (window
mass selects, all-query mass is bookkept), the per-layer qa/continuation regime
detector, pooled cross-head evidence for window ranking, and the reserve that
caps the peak cache *at* budget rather than above it. Merging stays available
behind `merge=true` purely for ablation.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from core.consolidate import consolidate_pairs
from core.ot_kv_v3 import EPS, OTKVv3Cache, otkv_v3_compress

LOG_FLOOR = 1e-30


def maxpool_tokens(scores: torch.Tensor, kernel: int) -> torch.Tensor:
    """Max-pool along the token (last) axis with 'same' output length."""
    if kernel is None or int(kernel) <= 1:
        return scores
    kernel = int(kernel)
    length = scores.shape[-1]
    if length == 0:
        return scores
    flat = scores.reshape(-1, 1, length).float()
    pooled = F.max_pool1d(flat, kernel_size=min(kernel, length), stride=1,
                          padding=min(kernel, length) // 2)
    pooled = pooled[..., :length]                      # even kernels overshoot by 1
    return pooled.reshape(scores.shape).to(scores.dtype)


class OTKVv7Cache(OTKVv3Cache):
    """Selection-only compression with value-weighted, pooled scores and
    water-filled per-layer budgets."""

    # auto_regime is off by default: on Llama-3.1 the per-layer entropy-ratio
    # statistic does not separate question prompts from continuation prompts
    # (experiments/detector_probe.py: cont median 0.769 vs needle 0.745, fully
    # overlapping quartiles), and the qa path alone matches every threshold
    # variant on LongBench (25.64 vs 25.72) while scoring NIAH 1.00. The
    # detector remains available for models where the statistic separates
    # (Qwen3: 0.66-0.69 vs 0.77, threshold 0.73).
    V7_DEFAULTS = dict(
        merge=False,
        observation_window=32,
        split_scores=True,
        auto_regime=False,
        select_scope="global",
        select_mode="topk",
        per_head=True,
        compress_interval=4,
        top_r=4,
        epsilon=0.02,
    )

    def __init__(self, **kwargs):
        cfg = dict(self.V7_DEFAULTS)
        cfg.update(kwargs)
        super().__init__(**cfg)
        self.pool_kernel = int(cfg.get("pool_kernel", 7))
        self.vnorm_tau = float(cfg.get("vnorm_tau", 1.0))
        # "waterfill" | "uniform" | "pyramid" (pyramid delegates to the parent's
        # layer_budget knob so the ablation stays one flag)
        self.layer_alloc = str(cfg.get("layer_alloc", "waterfill"))
        if self.layer_alloc == "pyramid":
            self.layer_budget = "pyramid"
        self.alloc_floor = float(cfg.get("alloc_floor", 0.25))
        self.alloc_cap = float(cfg.get("alloc_cap", 2.5))
        self.cont_select_mode = str(cfg.get("cont_select_mode", "kmeans:50"))
        # "layer": each layer follows its own regime vote (v4 behaviour).
        # "global": majority vote across layers -- one mislabelled layer on a
        # retrieval prompt evicts exactly the needle, and the local NIAH
        # ablation shows per-layer mislabels are what loses it (4/5 vs 5/5).
        self.regime_scope = str(cfg.get("regime_scope", "layer"))
        # Which regime each mechanism applies in. "both" is the Lemma 1 default;
        # "continuation" restricts it to the all-query path if the dev ablations
        # say it taxes the window-scored one.
        self.vnorm_regime = str(cfg.get("vnorm_regime", "both"))
        self.alloc_regime = str(cfg.get("alloc_regime", "both"))

        # --- compensated pair consolidation (core/consolidate.py) ----------
        # Off by default: it changes decode-time attention (per-slot logit
        # bias), which needs the bias-aware pre-hook in the wrapper/bench.
        self.consolidate = bool(cfg.get("consolidate", False))
        self.consolidate_frac = float(cfg.get("consolidate_frac", 0.25))
        self.consolidate_tau = float(cfg.get("consolidate_tau", 0.3))
        self.consolidate_gap = int(cfg.get("consolidate_gap", 1))
        self.consolidate_gate = str(cfg.get("consolidate_gate", "sigma"))
        self.logit_bias: Dict[int, torch.Tensor] = {}
        self._logattn: Dict[int, torch.Tensor] = {}
        self._has_bias: Dict[int, bool] = {}
        self._gqa_groups = 1
        self._merged_slots = {"total": 0}
        self._alloc: Optional[Dict[int, int]] = None
        self._layer_regime: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # score plumbing
    # ------------------------------------------------------------------
    def _middle_bounds(self, layer_idx: int):
        seq_len = self._get_existing_cache(layer_idx)[0].shape[-2]
        return self.sink_size, max(self.sink_size, seq_len - self.recent_size)

    def _streams(self, layer_idx: int):
        """(all-query stream, window stream or None) over the full sequence."""
        s = self.attention_scores[layer_idx]
        if self._dual and s.dim() >= 3 and s.shape[0] == 2:
            return s[0], s[1]
        return s, None

    def _value_weight(self, values: torch.Tensor, like: torch.Tensor) -> Optional[torch.Tensor]:
        """||v||^tau shaped to broadcast against `like` (per-head or global)."""
        if self.vnorm_tau <= 0.0 or values is None:
            return None
        w = values.float().norm(dim=-1)                 # (b, h, L)
        if self.vnorm_tau != 1.0:
            w = w.pow(self.vnorm_tau)
        if like.dim() == 1:                             # global bookkeeping
            w = w.mean(dim=(0, 1))
        return w

    def _regime_for(self, layer_idx: int) -> str:
        return self._layer_regime.get(layer_idx, "qa")

    def _decide_regime(self, layer_idx: int) -> str:
        full, win = self._streams(layer_idx)
        if win is None:
            regime = "qa" if self.observation_window else "continuation"
        elif not self.auto_regime:
            regime = "qa"
        else:
            ms, me = self._middle_bounds(layer_idx)
            regime = self._regime(full[..., ms:me], win[..., ms:me])
        self._layer_regime[layer_idx] = regime
        self._regime_votes[regime] = self._regime_votes.get(regime, 0) + 1
        return regime

    def _selection_weights(self, layer_idx: int):
        """(selection scores, allocation gains) for the middle region.

        Selection scores are the regime's stream with pooling and the value
        weight applied; allocation gains are the same without pooling (pooling
        is a robustness prior on *which* token in a unit survives, not extra
        error mass, so it must not inflate a layer's claim on the budget).
        """
        ms, me = self._middle_bounds(layer_idx)
        full, win = self._streams(layer_idx)
        regime = self._regime_for(layer_idx)
        stream = win if (win is not None and regime == "qa") else full
        mid = stream[..., ms:me]

        sel = maxpool_tokens(mid, self.pool_kernel) if (win is not None and regime == "qa") else mid
        raw = mid

        if self.vnorm_regime in ("both", regime):
            _, middle_v = self.get_middle_cache(layer_idx)
            w = self._value_weight(middle_v, mid)
            if w is not None:
                sel = sel * w
                raw = raw * w
        return sel, raw, stream

    def _layer_total_mass(self, layer_idx: int, stream: torch.Tensor) -> torch.Tensor:
        """Total (attention x vnorm) mass of the whole layer, frozen regions
        included -- the normaliser that makes marginal gains comparable across
        layers. Follows the same vnorm gate as the middle weights so the
        marginal-gain fractions stay consistent."""
        w = None
        if self.vnorm_regime in ("both", self._regime_for(layer_idx)):
            _, v = self._get_existing_cache(layer_idx)
            w = self._value_weight(v, stream)
        weighted = stream if w is None else stream * w
        return weighted.sum()

    # ------------------------------------------------------------------
    # cross-layer budget allocation
    # ------------------------------------------------------------------
    def get_middle_budget(self, layer_idx: int, total_tokens: int) -> int:
        if self._alloc is not None and layer_idx in self._alloc and self.mode == "prefill":
            return self._alloc[layer_idx]
        return super().get_middle_budget(layer_idx, total_tokens)

    def _plan_allocation(self, layers):
        self._alloc = None
        uniform = int(self.middle_budget)
        if uniform <= 0 or self.mode != "prefill" or len(layers) < 2:
            return

        # Layers whose regime the allocation is not licensed for stay at the
        # uniform budget; the greedy runs over the rest with their slot share.
        pinned = [l for l in layers
                  if self.alloc_regime not in ("both", self._regime_for(l))]
        layers = [l for l in layers if l not in pinned]
        if len(layers) < 2:
            return

        total_slots = uniform * len(layers)
        floors, caps, lengths, gain_lists = {}, {}, {}, {}
        any_compress = False
        for l in layers:
            sel, raw, stream = self._selection_weights(l)
            g = raw.sum(dim=tuple(range(raw.dim() - 1))) if raw.dim() > 1 else raw
            lengths[l] = int(g.shape[-1])
            total = self._layer_total_mass(l, stream)
            g = (g / total.clamp_min(EPS)).sort(descending=True).values
            floor = min(lengths[l], max(0, int(uniform * self.alloc_floor)))
            cap = min(lengths[l], max(floor, int(math.ceil(uniform * self.alloc_cap))))
            floors[l], caps[l] = floor, cap
            gain_lists[l] = g
            if lengths[l] > uniform:
                any_compress = True
        if not any_compress:
            return

        remaining = total_slots - sum(floors.values())
        if remaining < 0:                                # floor > 1.0 misuse
            return

        pool_vals, pool_ids = [], []
        for l in layers:
            seg = gain_lists[l][floors[l]:caps[l]]
            pool_vals.append(seg)
            pool_ids.append(torch.full((seg.shape[-1],), l, dtype=torch.long, device=seg.device))
        vals = torch.cat(pool_vals) if pool_vals else torch.empty(0)
        ids = torch.cat(pool_ids) if pool_ids else torch.empty(0, dtype=torch.long)

        take = min(remaining, vals.shape[-1])
        alloc = dict(floors)
        if take > 0:
            top_ids = ids[torch.topk(vals, take).indices]
            counts = torch.bincount(top_ids.cpu(), minlength=max(layers) + 1)
            for l in layers:
                alloc[l] += int(counts[l])

        leftover = remaining - take                       # caps bound too hard
        while leftover > 0:
            moved = False
            for l in layers:
                room = lengths[l] - alloc[l]
                if room > 0 and leftover > 0:
                    add = min(room, leftover)
                    alloc[l] += add
                    leftover -= add
                    moved = True
            if not moved:
                break

        self._alloc = alloc

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def on_prefill_end(self):
        total = self.get_seq_length()
        self._update_budget(total, is_prefill_end=True)
        layers = sorted(self.attention_scores.keys())
        for l in layers:
            self._consume_attention_scores(l)
        for l in layers:
            self._decide_regime(l)
        if self.regime_scope == "global" and self._layer_regime:
            votes = list(self._layer_regime.values())
            majority = max(set(votes), key=votes.count)
            self._layer_regime = {l: majority for l in self._layer_regime}
        if self.layer_alloc == "waterfill":
            self._plan_allocation(layers)
        for l in layers:
            self._compress_layer(l, total, reserve_tokens=self._reserve(l, total))

    # ------------------------------------------------------------------
    # compensated pair consolidation (see core/consolidate.py)
    # ------------------------------------------------------------------
    def reduce_attention(self, attn_weights: torch.Tensor) -> torch.Tensor:
        a = attn_weights.detach()
        out = super().reduce_attention(a)
        W = int(self.observation_window or 0)
        if (not self.consolidate or W <= 0 or a.shape[-2] <= W
                or not torch.is_tensor(out) or out.dim() != 4):
            return out
        # piggyback the raw window rows on the dual stack; the consume side
        # knows the layer index and splits them apart (v5's trick)
        rows = a[..., -W:, :].float()                    # (b, qh, W, L)
        return torch.cat([out.permute(1, 2, 0, 3).float(), rows], dim=-2)

    def _consume_attention_scores(self, layer_idx: int):
        x = self.current_attention_scores.get(layer_idx)
        W = int(self.observation_window or 0)
        if (self.consolidate and x is not None and torch.is_tensor(x)
                and x.dim() == 4 and W > 0 and x.shape[-2] == 2 + W):
            dual = x[..., :2, :].permute(2, 0, 1, 3).contiguous()
            rows = x[..., 2:, :]                         # (b, qh, W, L)
            cur = self.attention_scores.get(layer_idx)
            kv = cur.shape[-2] if (cur is not None and cur.dim() >= 3) else rows.shape[1]
            b, qh, _, L = rows.shape
            if kv > 0 and qh % kv == 0:
                self._gqa_groups = qh // kv
                rows = rows.reshape(b, kv, self._gqa_groups * W, L)
            self._logattn[layer_idx] = rows
            self.current_attention_scores[layer_idx] = dual
        return super()._consume_attention_scores(layer_idx)

    def pre_attention_step(self, layer_idx: int):
        """Called by the bias-aware attention pre-hook before the mask for a
        decode step is built. Runs any pending maintenance compression *now*,
        so the logit-bias mask and the cache the model attends over agree."""
        if not self.consolidate:
            return
        self._prehook_seen = True
        self.finalize_prefill()
        self._consume_attention_scores(layer_idx)
        total = self.get_seq_length()
        self._update_budget(total + 1, is_prefill_end=False)
        step = self.decode_steps.get(layer_idx, 0)
        if step > 0 and step % self.compress_interval == 0:
            self._evict_simple(layer_idx, total + 1)

    def get_decode_bias(self, layer_idx: int) -> Optional[torch.Tensor]:
        """(b, n_query_heads, 1, L+1) additive logit bias for the next decode
        step, or None when no slot in this layer carries merged mass."""
        if not self.consolidate or not self._has_bias.get(layer_idx):
            return None
        bias = self.logit_bias.get(layer_idx)
        if bias is None:
            return None
        x = F.pad(bias, (0, 1))                          # incoming token: bias 0
        if self._gqa_groups > 1:
            x = x.repeat_interleave(self._gqa_groups, dim=1)
        return x.unsqueeze(2)

    def on_decode_step(self, key_states, value_states, layer_idx, cache_kwargs):
        if not self.consolidate:
            return super().on_decode_step(key_states, value_states, layer_idx, cache_kwargs)
        q_len = key_states.shape[-2]
        self._consume_attention_scores(layer_idx)
        total_after = self.get_seq_length() + q_len
        self._update_budget(total_after, is_prefill_end=False)
        step = self.decode_steps.get(layer_idx, 0)
        if not getattr(self, "_prehook_seen", False) and step > 0 \
                and step % self.compress_interval == 0:
            # legacy harness without the bias-aware pre-hook: keep the cache at
            # budget anyway (the bias then simply never reaches the logits)
            self._evict_simple(layer_idx, total_after)
        self.decode_steps[layer_idx] = step + q_len
        self._grow_scores(layer_idx, key_states)
        if layer_idx in self.logit_bias:
            self.logit_bias[layer_idx] = F.pad(self.logit_bias[layer_idx], (0, q_len))
        return key_states, value_states

    def _evict_simple(self, layer_idx: int, total_tokens: int):
        """Decode-time top-k eviction that keeps every channel aligned:
        K/V rows, both score streams, and the per-slot logit bias."""
        middle_k, middle_v = self.get_middle_cache(layer_idx)
        if middle_k is None or middle_k.shape[-2] == 0:
            return
        budget = max(0, int(self.get_middle_budget(layer_idx, total_tokens))
                     - self._reserve(layer_idx, total_tokens))
        Lm = middle_k.shape[-2]
        if budget >= Lm:
            return
        ms, me = self._middle_bounds(layer_idx)
        sel, _, _ = self._selection_weights(layer_idx)
        if sel.dim() == 3:
            pooled = sel.sum(dim=1, keepdim=True).expand_as(sel)
        else:
            pooled = sel.view(1, 1, -1).expand(middle_k.shape[0], middle_k.shape[1], -1)
        keep = torch.topk(pooled, max(1, budget), dim=-1).indices.sort(dim=-1).values

        g4 = keep.unsqueeze(-1).expand(-1, -1, -1, middle_k.shape[-1])
        self.replace_middle_cache(layer_idx,
                                  torch.gather(middle_k, 2, g4),
                                  torch.gather(middle_v, 2, g4))

        scores = self.attention_scores[layer_idx]
        mid = scores[..., ms:me]
        if mid.dim() == 4:                               # (2, b, h, Lm)
            idx = keep.unsqueeze(0).expand(mid.shape[0], -1, -1, -1)
            new_mid = torch.gather(mid, -1, idx)
        elif mid.dim() == 3:
            new_mid = torch.gather(mid, -1, keep)
        else:
            new_mid = mid[keep[0, 0]]
        self.attention_scores[layer_idx] = torch.cat(
            [scores[..., :ms], new_mid, scores[..., me:]], dim=-1)

        bias = self.logit_bias.get(layer_idx)
        if bias is not None:
            bm = torch.gather(bias[..., ms:me], -1, keep)
            self.logit_bias[layer_idx] = torch.cat(
                [bias[..., :ms], bm, bias[..., me:]], dim=-1)

    def _compress_layer_consolidate(self, layer_idx: int, total_tokens: int,
                                    reserve_tokens: int) -> bool:
        rows = self._logattn.get(layer_idx)
        if rows is None or self._regime_for(layer_idx) == "continuation":
            return False
        middle_k, middle_v = self.get_middle_cache(layer_idx)
        if middle_k is None or middle_k.shape[-2] == 0:
            return False
        budget = max(1, int(self.get_middle_budget(layer_idx, total_tokens)) - reserve_tokens)
        Lm = middle_k.shape[-2]
        if budget >= Lm:
            return False
        ms, me = self._middle_bounds(layer_idx)
        if rows.shape[-1] < me:                          # rows predate this shape
            return False
        scores = self.attention_scores[layer_idx]
        if not (scores.dim() == 4 and scores.shape[0] == 2):
            return False                                 # needs the dual layout

        sel, _, _ = self._selection_weights(layer_idx)
        pooled = sel.sum(dim=1, keepdim=True).expand_as(sel) if sel.dim() == 3 else sel
        b_plus = min(Lm, budget + max(1, math.ceil(budget * self.consolidate_frac)))
        anchor = torch.topk(pooled, b_plus, dim=-1).indices.sort(dim=-1).values  # (b,h,B+)

        g4 = anchor.unsqueeze(-1).expand(-1, -1, -1, middle_k.shape[-1])
        k_a = torch.gather(middle_k, 2, g4)
        v_a = torch.gather(middle_v, 2, g4)
        mid_scores = torch.gather(scores[..., ms:me], -1,
                                  anchor.unsqueeze(0).expand(2, -1, -1, -1))

        log_full = rows[..., ms:me].clamp_min(LOG_FLOOR).log().transpose(-2, -1)  # (b,kv,Lm,GW)
        gW = log_full.shape[-1]
        log_rows = torch.gather(log_full, 2, anchor.unsqueeze(-1).expand(-1, -1, -1, gW))

        new_k, new_v, new_scores, _, bias, n_merged = consolidate_pairs(
            k_a, v_a, anchor, log_rows, mid_scores, None,
            target_slots=budget, tau=self.consolidate_tau,
            gap=self.consolidate_gap, gate=self.consolidate_gate)

        self.replace_middle_cache(layer_idx, new_k, new_v)
        self.attention_scores[layer_idx] = torch.cat(
            [scores[..., :ms], new_scores, scores[..., me:]], dim=-1)

        phys = self._get_existing_cache(layer_idx)[0].shape[-2]
        full_bias = new_k.new_zeros(new_k.shape[0], new_k.shape[1], phys, dtype=torch.float32)
        full_bias[..., ms:ms + new_k.shape[2]] = bias
        self.logit_bias[layer_idx] = full_bias
        self._has_bias[layer_idx] = bool(n_merged > 0)
        self._merged_slots["total"] += n_merged
        self._logattn.pop(layer_idx, None)
        return True

    def _compress_layer(self, layer_idx: int, total_tokens: int, reserve_tokens: int = 0,
                        allow_merge: bool = True, allow_select: bool = True):
        if self.consolidate and layer_idx in self._logattn:
            if self._compress_layer_consolidate(layer_idx, total_tokens, reserve_tokens):
                return
            self._logattn.pop(layer_idx, None)
        middle_k, middle_v = self.get_middle_cache(layer_idx)
        if middle_k is None or middle_k.shape[-2] == 0:
            return

        budget = max(0, int(self.get_middle_budget(layer_idx, total_tokens)) - reserve_tokens)
        if budget >= middle_k.shape[-2]:
            return

        ms, me = self._middle_bounds(layer_idx)
        scores = self.attention_scores[layer_idx]

        if layer_idx not in self._layer_regime:            # decode-time first touch
            self._decide_regime(layer_idx)
        regime = self._regime_for(layer_idx)

        aux = None
        if self._dual and scores.dim() >= 3 and scores.shape[0] == 2:
            aux = scores[..., ms:me]
            base_scores = aux[0]
        else:
            base_scores = scores[..., ms:me]

        sel, _, _ = self._selection_weights(layer_idx)

        if regime == "continuation":
            # all-query evidence is per-head-rich (docs/experiments.md section 1);
            # coverage re-selection is prefill-only unless select_decode is set
            select_scope = "head"
            select_mode = self.cont_select_mode if allow_select else "topk"
        else:
            select_mode, select_scope = "topk", self.select_scope

        fz = scores[0] if aux is not None else scores
        frozen = (fz[..., :ms].sum(-1, keepdim=True)
                  + fz[..., me:].sum(-1, keepdim=True))

        out = otkv_v3_compress(
            middle_k, middle_v, budget, base_scores,
            merge=self.merge and allow_merge, top_r=self.top_r, epsilon=self.epsilon,
            sinkhorn_iters=self.sinkhorn_iters, capacity_beta=self.capacity_beta,
            sim_threshold=self.sim_threshold, merge_strength=self.merge_strength,
            score_mode=self.score_mode, chunk=self.chunk, key_merge=self.key_merge,
            aux_scores=aux, select_scores=sel,
            score_writeback=self.score_writeback,
            select_mode=select_mode,
            gate_sigma=self.gate_sigma, cost_alpha=self.cost_alpha, lambda_pos=self.lambda_pos,
            select_scope=select_scope,
            frozen_mass=frozen if self.frozen_correction else None,
        )

        if aux is None:
            new_k, new_v, new_scores = out
        else:
            new_k, new_v, _, new_scores = out

        self.replace_middle_cache(layer_idx, new_k, new_v)

        if new_scores.dim() != scores.dim():
            new_scores = new_scores.sum(dim=tuple(range(new_scores.dim() - 1)))
        self.attention_scores[layer_idx] = torch.cat(
            [scores[..., :ms], new_scores, scores[..., me:]], dim=-1)
