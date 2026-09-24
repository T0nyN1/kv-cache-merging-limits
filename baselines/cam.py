"""CaM (Zhang et al., ICML 2024) — faithful re-implementation.

Reference: github.com/zyxxmu/cam, `cam_hf/utils_hh/modify_llama_cam.py`,
`local_cam_mask`. The published operator, verbatim:

    merge_prob = attn(evicted) / mean(attn over the recent window)
    mask       = Bernoulli(clamp(merge_prob, 0, 1))
    v[next 32 kept positions] += v_evicted * mask / 32

On every eviction the evicted token's *value* is smeared (divided by a fixed
merge_budget of 32) over the retained tokens that follow it in sequence order,
gated by its attention prominence. Keys untouched, no mass compensation —
the class-A (value-coefficient) representative for the matched-protocol
comparison in docs/merging_survey.md. Selection is H2O's accumulated
attention, their base method. Run with per_head=false (the published code
keeps one token layout shared across heads with per-head values).

CaM is a *streaming* operator: the published code evicts exactly one token
per decode step, so each kept position accumulates at most a handful of v/32
contributions. Applying the same operator to a prefill-end mass eviction
super-imposes thousands of smears and the values diverge (measured: wikitext
ppl 77.7 against 4.42 for its own eviction base — the v2 failure mode in
another costume). The faithful port therefore applies the operator only where
the original defines it: decode-time prunes, which here evict one token per
step, matching the published cadence; the prefill-end mass eviction is plain
eviction.
"""

import torch

from baselines.h2o import H2OCache

MERGE_BUDGET = 32


class CaMCache(H2OCache):

    def _select_prune_middle(self, layer_idx: int, total_tokens: int):
        k, _ = self._get_existing_cache(layer_idx)
        if k is None:
            return super()._select_prune_middle(layer_idx, total_tokens)
        if not self._prefill_finalized:
            # prefill-end mass eviction: the streaming operator is undefined
            # here (see module docstring) — plain eviction, as published
            return super()._select_prune_middle(layer_idx, total_tokens)

        seq_len = k.shape[-2]
        ms = self.sink_size
        me = max(self.sink_size, seq_len - self.recent_size)
        if ms >= me:
            return super()._select_prune_middle(layer_idx, total_tokens)

        scores = self.attention_scores[layer_idx]
        middle_scores = scores[..., ms:me]
        budget = self.get_middle_budget(layer_idx, total_tokens)
        if budget >= middle_scores.shape[-1] or middle_scores.dim() != 1:
            # per-head bookkeeping not supported by the published operator
            return super()._select_prune_middle(layer_idx, total_tokens)

        keep = self._middle_keep_indices(self._rank_scores(middle_scores), budget)
        Lm = middle_scores.shape[-1]
        evict_mask = torch.ones(Lm, dtype=torch.bool, device=middle_scores.device)
        if keep.numel():
            evict_mask[keep] = False
        evict = torch.nonzero(evict_mask).flatten()

        if evict.numel() and keep.numel():
            middle_k, middle_v = self.get_middle_cache(layer_idx)
            recent_scores = scores[..., me:]
            norm = recent_scores.mean().clamp_min(1e-9) if recent_scores.numel() \
                else middle_scores.mean().clamp_min(1e-9)
            prob = (middle_scores[evict] / norm).clamp(0.0, 1.0)
            prob = torch.where(torch.isfinite(prob), prob, torch.zeros_like(prob))
            mask = torch.bernoulli(prob)

            ip = torch.searchsorted(keep, evict)          # first kept slot after e
            v = middle_v.float()
            contrib = v[..., evict, :] * (mask / MERGE_BUDGET).view(1, 1, -1, 1)
            new_v = v.clone()
            m = keep.numel()
            for off in range(MERGE_BUDGET):
                valid = (ip + off) < m
                if not bool(valid.any()):
                    break
                sel = torch.nonzero(valid).flatten()
                tgt = keep[(ip + off)[sel]]
                new_v.index_add_(2, tgt, contrib[..., sel, :])
            self.replace_middle_cache(layer_idx, middle_k, new_v.to(middle_v.dtype))

        # scores are untouched, so the parent recomputes the identical keep set
        # and performs the actual prune + score bookkeeping
        super()._select_prune_middle(layer_idx, total_tokens)
