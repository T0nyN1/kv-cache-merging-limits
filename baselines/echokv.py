from typing import Optional

import torch
import torch.nn.functional as F

from evaluation.models.base_cache import BaseCompressCache


class EchoKVCache(BaseCompressCache):
    requires_attention = False

    def __init__(self, max_representative_scan: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.max_representative_scan = max_representative_scan
        self.active_layers = set()

    def on_prefill(self, key_states: torch.Tensor, value_states: torch.Tensor,
                   layer_idx: int, cache_kwargs: dict):
        self.active_layers.add(layer_idx)
        self.current_attention_scores.pop(layer_idx, None)
        return key_states, value_states

    def on_prefill_end(self):
        total_tokens = self.get_seq_length()
        self._update_budget(total_tokens, is_prefill_end=True)
        for layer_idx in self.active_layers:
            self._prune_layer(layer_idx, total_tokens)

    def on_decode_step(self, key_states: torch.Tensor, value_states: torch.Tensor,
                       layer_idx: int, cache_kwargs: dict):
        self.current_attention_scores.pop(layer_idx, None)

        total_after = self.get_seq_length() + key_states.shape[-2]
        self._update_budget(total_after, is_prefill_end=False)
        self._prune_layer(layer_idx, total_after)

        return key_states, value_states

    def _prune_layer(self, layer_idx: int, total_tokens: int):
        middle_k, _ = self.get_middle_cache(layer_idx)
        if middle_k is None or middle_k.numel() == 0:
            return

        layer_middle_budget = self.get_middle_budget(layer_idx, total_tokens)
        seq_len = middle_k.shape[-2]

        if layer_middle_budget >= seq_len:
            return

        if layer_middle_budget == 0:
            keep_indices = torch.empty(0, dtype=torch.long, device=middle_k.device)
            if self.per_head:
                keep_indices = keep_indices.view(middle_k.shape[0], middle_k.shape[1], 0)
            self.prune_middle_cache(layer_idx, keep_indices)
            return

        candidates = torch.arange(seq_len, device=middle_k.device, dtype=torch.long)
        keep_indices = self._select_representatives(middle_k, candidates, layer_middle_budget)

        self.prune_middle_cache(layer_idx, keep_indices)

    # ------------------------------------------------------------------
    # Representative selection. In global mode a single set of positions is
    # kept across all heads; in per_head mode each (batch, kv-head) selects its
    # own representatives, matching the per-head contract of the other caches.
    # ------------------------------------------------------------------
    def _select_representatives(self, middle_k: torch.Tensor, candidates: torch.Tensor, budget: int) -> torch.Tensor:
        if candidates.numel() <= budget:
            return self._broadcast_candidates(middle_k, candidates)

        features = self._key_features(middle_k)

        if self.max_representative_scan is not None and candidates.numel() > self.max_representative_scan:
            candidates = self._evenly_spaced_candidates(candidates, self.max_representative_scan)
            budget = min(budget, candidates.numel())

        if self.per_head:
            return self._select_representatives_per_head(features, candidates, budget)
        return self._select_representatives_global(features, candidates, budget)

    def _select_representatives_global(self, features: torch.Tensor, candidates: torch.Tensor,
                                       budget: int) -> torch.Tensor:
        candidate_features = features.index_select(0, candidates)  # (Ncand, d)
        picked = []
        chunk_offsets = torch.tensor_split(
            torch.arange(candidates.numel(), device=candidates.device, dtype=torch.long), budget)

        for offsets in chunk_offsets:
            if offsets.numel() == 0:
                continue
            chunk_features = candidate_features.index_select(0, offsets)
            centroid = F.normalize(chunk_features.mean(dim=0, keepdim=True), dim=-1)
            similarities = torch.matmul(chunk_features, centroid.squeeze(0))
            best_offset = offsets[torch.argmax(similarities)]
            picked.append(best_offset)

        if not picked:
            return self._evenly_spaced_candidates(candidates, budget)

        picked_offsets = torch.stack(picked)
        return candidates.index_select(0, picked_offsets).sort().values

    def _select_representatives_per_head(self, features: torch.Tensor, candidates: torch.Tensor,
                                         budget: int) -> torch.Tensor:
        # features: (batch, kv_heads, seq, d); candidates index into seq (shared)
        cand_feat = features.index_select(2, candidates)  # (b, H, Ncand, d)
        picked = []
        chunk_offsets = torch.tensor_split(
            torch.arange(candidates.numel(), device=candidates.device, dtype=torch.long), budget)

        for offsets in chunk_offsets:
            if offsets.numel() == 0:
                continue
            chunk = cand_feat.index_select(2, offsets)  # (b, H, clen, d)
            centroid = F.normalize(chunk.mean(dim=2), dim=-1)  # (b, H, d)
            similarities = torch.einsum('bhcd,bhd->bhc', chunk, centroid)  # (b, H, clen)
            best_local = torch.argmax(similarities, dim=-1)  # (b, H) into offsets
            picked.append(offsets[best_local])  # (b, H) into candidates

        if not picked:
            return self._broadcast_candidates_from_positions(
                features, self._evenly_spaced_candidates(candidates, budget))

        picked_offsets = torch.stack(picked, dim=-1)  # (b, H, budget)
        keep = candidates[picked_offsets]  # (b, H, budget) middle positions
        return keep.sort(dim=-1).values

    def _broadcast_candidates(self, middle_k: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        if not self.per_head:
            return candidates
        b, h = middle_k.shape[0], middle_k.shape[1]
        return candidates.view(1, 1, -1).expand(b, h, -1).contiguous()

    def _broadcast_candidates_from_positions(self, features: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        b, h = features.shape[0], features.shape[1]
        return candidates.view(1, 1, -1).expand(b, h, -1).contiguous()

    def _key_features(self, key_states: torch.Tensor):
        keys = key_states.detach().float()  # (batch, kv_heads, seq, d)
        if self.per_head:
            return F.normalize(keys, dim=-1)
        seq_len = keys.shape[-2]
        keys = keys.reshape(-1, seq_len, keys.shape[-1]).mean(dim=0)  # (seq, d)
        return F.normalize(keys, dim=-1)

    @staticmethod
    def _evenly_spaced_candidates(candidates: torch.Tensor, budget: int):
        n = candidates.numel()
        if budget <= 0 or n == 0:
            return candidates[:0]
        if budget == 1:
            offsets = torch.tensor([n // 2], device=candidates.device, dtype=torch.long)
        else:
            offsets = torch.linspace(0, n - 1, steps=budget,
                                     device=candidates.device).round().to(dtype=torch.long)
        return candidates.index_select(0, offsets).sort().values
