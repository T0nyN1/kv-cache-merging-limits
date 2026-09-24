from typing import Union

import torch
from transformers.cache_utils import DynamicCache


class BaseCompressCache(DynamicCache):
    requires_attention: bool = True

    def __init__(self,
                 compression_size: Union[int, float],
                 mode: str = "prefill",
                 sink_size: Union[int, float] = 4,
                 recent_size: Union[int, float] = 256,
                 **kwargs):
        super().__init__()

        self.compression_size = compression_size
        self.mode = mode.lower()
        self.per_head = kwargs.get("per_head", False)

        self._init_sink_size = sink_size
        self._init_recent_size = recent_size
        self.kwargs = kwargs

        self.current_attention_scores = {}
        self.attention_scores = {}

        self.budget = 0
        self.sink_size = 0
        self.recent_size = 0
        self.middle_budget = 0
        self.prefill_length = 0
        self._prefill_finalized = False

        self._decode_step = 0

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs=None) -> tuple[
        torch.Tensor, torch.Tensor]:
        q_len = key_states.shape[-2]
        is_prefill = q_len > 1

        if is_prefill:
            self._consume_attention_scores(layer_idx)
            key_states, value_states = self.on_prefill(key_states, value_states, layer_idx, cache_kwargs)
            if layer_idx == 0:
                self._decode_step = 0
                self._print_monitor("Initialization", q_len, self._get_phys_length(0))
        else:
            if layer_idx == 0:
                flag = self.finalize_prefill()
                if flag:
                    self._print_monitor("Prefill", None, self._get_phys_length(0))
            key_states, value_states = self.on_decode_step(key_states, value_states, layer_idx, cache_kwargs)
            if layer_idx == 0:
                self._print_monitor("Decode", self._decode_step, self._get_phys_length(0))
                self._decode_step += 1

        result = super().update(key_states, value_states, layer_idx, cache_kwargs)
        return result

    def _print_monitor(self, stage: str, step_or_len: int, cache_len: int):
        budget = getattr(self, "budget", "N/A")
        sink = getattr(self, "sink_size", "N/A")
        recent = getattr(self, "recent_size", "N/A")
        middle = getattr(self, "middle_budget", "N/A")

        if stage == "Initialization":
            print(
                f"[KV Monitor] Stage: {stage} | "
                f"Input Tokens: {step_or_len:<4} | "
            )
        elif stage == "Prefill":
            print(
                f"[KV Monitor] Stage: {stage} | "
                f"Cache: {cache_len:<5} | "
                f"Budget: {budget:<4} (Sink:{sink} Middle:{middle} Recent:{recent})"
            )
        elif stage == "Decode":
            log_str = (f"[KV Monitor] Stage: {stage}  | Step: {step_or_len:<4} | "
                       f"Cache: {cache_len:<4} | "
                       f"Budget: {budget:<4} (Sink:{sink} Middle:{middle} Recent:{recent})")
            print(f"\r{log_str}\033[K", end="", flush=True)

    def finalize_prefill(self):
        if self._prefill_finalized:
            return False

        if self.prefill_length > 0:
            self._prefill_finalized = True
            return False

        if self.get_seq_length() == 0:
            return False

        self.on_prefill_end()
        self._prefill_finalized = True
        return True

    def on_prefill(self, key_states, value_states, layer_idx, cache_kwargs):
        raise NotImplementedError

    def on_decode_step(self, key_states, value_states, layer_idx, cache_kwargs):
        raise NotImplementedError

    def on_prefill_end(self):
        raise NotImplementedError

    def reduce_attention(self, attn_weights: torch.Tensor) -> torch.Tensor:
        assert attn_weights is not None
        return attn_weights.sum(dim=-2).detach()

    @staticmethod
    def _match_heads(scores: torch.Tensor, target_heads: int) -> torch.Tensor:
        """Reconcile a (batch, score_heads, k_len) tensor to `target_heads`.

        Attention weights are produced per *query* head, while the KV cache (and
        therefore our per-head scores) is indexed by *key/value* heads. For GQA
        models score_heads is a multiple of target_heads, so we sum each group of
        query heads back onto the KV head they share.
        """
        score_heads = scores.shape[1]
        if score_heads == target_heads:
            return scores
        if score_heads == 1:
            return scores.expand(-1, target_heads, -1)
        if score_heads % target_heads == 0:
            group = score_heads // target_heads
            return scores.reshape(scores.shape[0], target_heads, group, scores.shape[-1]).sum(dim=2)
        if target_heads % score_heads == 0:
            repeat = target_heads // score_heads
            return scores.repeat_interleave(repeat, dim=1)
        return scores.mean(dim=1, keepdim=True).expand(-1, target_heads, -1)

    def _attention_to_token_scores(self, attn_weights: torch.Tensor, target_shape=None) -> torch.Tensor:
        """Turn raw attention weights into a per-token importance tensor.

        `reduce_attention` has already collapsed the query-length axis, so the
        input is typically (batch, num_query_heads, k_len). We normalise it to
        (batch, heads, k_len) then either collapse to (k_len,) in global mode or,
        in per-head mode, fold query heads onto KV heads via `_match_heads`.
        """
        scores = attn_weights.detach().float()

        if scores.dim() == 1:
            scores = scores.view(1, 1, -1)
        elif scores.dim() == 2:
            scores = scores.unsqueeze(0)
        else:
            while scores.dim() > 3:
                scores = scores.sum(dim=-2)

        if not self.per_head:
            # collapse batch and every head into a single score per token
            return scores.sum(dim=(0, 1))

        if target_shape is not None and len(target_shape) >= 2:
            scores = self._match_heads(scores, target_shape[-2])
        return scores

    def _consume_attention_scores(self, layer_idx: int):
        attn_weights = self.current_attention_scores.pop(layer_idx, None)
        if attn_weights is None or layer_idx not in getattr(self, 'attention_scores', {}):
            return

        current_scores = self.attention_scores[layer_idx]
        if current_scores.numel() == 0:
            return

        score_update = self._attention_to_token_scores(attn_weights, current_scores.shape)
        score_update = score_update.to(device=current_scores.device, dtype=current_scores.dtype)

        usable = min(current_scores.shape[-1], score_update.shape[-1])
        if usable == 0:
            return

        current_scores[..., :usable] += score_update[..., :usable]

    # ------------------------------------------------------------------
    # Per-head aware score bookkeeping shared by every attention-based cache.
    # Subclasses build/extend/gather scores through these helpers so that the
    # per_head vs global distinction lives in exactly one place.
    # ------------------------------------------------------------------
    def _grow_scores(self, layer_idx: int, key_states: torch.Tensor):
        """Allocate (or right-extend by q_len) the score tensor for a layer.

        Global mode stores a 1-D (k_len,) tensor; per-head mode stores a
        (batch, num_kv_heads, k_len) tensor keyed by KV heads.
        """
        batch, num_kv_heads, q_len, _ = key_states.shape
        device = key_states.device
        if self.per_head:
            new_scores = torch.zeros((batch, num_kv_heads, q_len), device=device, dtype=torch.float32)
        else:
            new_scores = torch.zeros(q_len, device=device, dtype=torch.float32)

        if layer_idx not in self.attention_scores:
            self.attention_scores[layer_idx] = new_scores
        else:
            self.attention_scores[layer_idx] = torch.cat(
                [self.attention_scores[layer_idx], new_scores], dim=-1)

    def _middle_keep_indices(self, middle_scores: torch.Tensor, budget: int) -> torch.Tensor:
        """Top-k indices into the middle region, shaped for the active mode."""
        if budget <= 0:
            keep = torch.empty(0, dtype=torch.long, device=middle_scores.device)
            if self.per_head:
                keep = keep.view(middle_scores.shape[0], middle_scores.shape[1], 0)
            return keep
        _, keep = torch.topk(middle_scores, k=budget, dim=-1)
        return keep.sort(dim=-1).values

    def _gather_middle_scores(self, middle_scores: torch.Tensor, keep_indices: torch.Tensor) -> torch.Tensor:
        if self.per_head:
            return torch.gather(middle_scores, -1, keep_indices)
        return middle_scores[keep_indices]

    def _update_budget(self, total_tokens: int, is_prefill_end: bool = False):
        if is_prefill_end:
            self.prefill_length = total_tokens

            if isinstance(self.compression_size, int):
                self.budget = self.compression_size
            elif isinstance(self.compression_size, float):
                self.budget = int(total_tokens * self.compression_size)

            if isinstance(self._init_sink_size, int):
                self.sink_size = self._init_sink_size
            elif isinstance(self._init_sink_size, float):
                self.sink_size = int(self.budget * self._init_sink_size)

            if isinstance(self._init_recent_size, int):
                self.recent_size = self._init_recent_size
            elif isinstance(self._init_recent_size, float):
                self.recent_size = int(self.budget * self._init_recent_size)

            self.middle_budget = max(0, self.budget - self.sink_size - self.recent_size)


        else:
            if self.mode == "entire" and isinstance(self.compression_size, float):
                self.budget = int(total_tokens * self.compression_size)
                self.middle_budget = max(0, self.budget - self.sink_size - self.recent_size)

    def get_middle_budget(self, layer_idx: int, total_tokens: int) -> int:
        if self.mode == "entire" and isinstance(self.compression_size, float):
            budget = int(total_tokens * self.compression_size)
            return max(0, budget - self.sink_size - self.recent_size)

        return self.middle_budget

    def get_middle_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = self._get_existing_cache(layer_idx)
        if k is None or v is None:
            return None, None

        seq_len = k.shape[-2]
        middle_start = self.sink_size
        middle_end = max(self.sink_size, seq_len - self.recent_size)

        if middle_start >= middle_end:
            return None, None

        return k[..., middle_start:middle_end, :], v[..., middle_start:middle_end, :]

    def prune_middle_cache(self, layer_idx: int, keep_middle_indices: torch.Tensor):
        k, v = self._get_existing_cache(layer_idx)
        if k is None or v is None:
            return

        seq_len = k.shape[-2]
        middle_start = self.sink_size
        middle_end = max(self.sink_size, seq_len - self.recent_size)

        sink_k, sink_v = k[..., :middle_start, :], v[..., :middle_start, :]
        recent_k, recent_v = k[..., middle_end:, :], v[..., middle_end:, :]

        if middle_start < middle_end:
            middle_k, middle_v = k[..., middle_start:middle_end, :], v[..., middle_start:middle_end, :]
            keep_middle_indices = keep_middle_indices.to(device=k.device, dtype=torch.long)
            if keep_middle_indices.dim() >= 3:
                gather_idx_k = keep_middle_indices.unsqueeze(-1).expand(-1, -1, -1, middle_k.shape[-1])
                gather_idx_v = keep_middle_indices.unsqueeze(-1).expand(-1, -1, -1, middle_v.shape[-1])
                pruned_middle_k = torch.gather(middle_k, 2, gather_idx_k)
                pruned_middle_v = torch.gather(middle_v, 2, gather_idx_v)
            else:
                pruned_middle_k = middle_k.index_select(-2, keep_middle_indices)
                pruned_middle_v = middle_v.index_select(-2, keep_middle_indices)
        else:
            pruned_middle_k = k.new_empty((*k.shape[:-2], 0, k.shape[-1]))
            pruned_middle_v = v.new_empty((*v.shape[:-2], 0, v.shape[-1]))

        new_k = torch.cat([sink_k, pruned_middle_k, recent_k], dim=-2)
        new_v = torch.cat([sink_v, pruned_middle_v, recent_v], dim=-2)

        self._replace_existing_cache(layer_idx, new_k, new_v)

    def replace_middle_cache(self, layer_idx: int, new_middle_k: torch.Tensor, new_middle_v: torch.Tensor):
        k, v = self._get_existing_cache(layer_idx)
        if k is None or v is None:
            return

        seq_len = k.shape[-2]
        middle_start = self.sink_size
        middle_end = max(self.sink_size, seq_len - self.recent_size)

        sink_k, sink_v = k[..., :middle_start, :], v[..., :middle_start, :]
        recent_k, recent_v = k[..., middle_end:, :], v[..., middle_end:, :]

        new_k = torch.cat([sink_k, new_middle_k, recent_k], dim=-2)
        new_v = torch.cat([sink_v, new_middle_v, recent_v], dim=-2)

        self._replace_existing_cache(layer_idx, new_k, new_v)

    def _get_existing_cache(self, layer_idx: int):
        if hasattr(self, "layers"):
            if layer_idx >= len(self.layers):
                return None, None
            layer = self.layers[layer_idx]
            return layer.keys, layer.values

        if hasattr(self, "key_cache"):
            if layer_idx >= len(self.key_cache):
                return None, None
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        return None, None

    def _replace_existing_cache(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor):
        if key_states is None or value_states is None:
            return

        if hasattr(self, "layers"):
            if layer_idx >= len(self.layers):
                return
            layer = self.layers[layer_idx]
            layer.keys = key_states
            layer.values = value_states
            self._set_layer_length(layer, key_states.shape[-2])
            return

        if hasattr(self, "key_cache"):
            if layer_idx >= len(self.key_cache):
                return
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states

    @staticmethod
    def _set_layer_length(layer, seq_len: int):
        if hasattr(layer, "cumulative_length"):
            if isinstance(layer.cumulative_length, torch.Tensor):
                layer.cumulative_length.fill_(seq_len)
            else:
                layer.cumulative_length = seq_len
        if hasattr(layer, "cumulative_length_int"):
            layer.cumulative_length_int = seq_len

    def _get_phys_length(self, layer_idx=0):
        if hasattr(self, "layers"):
            if layer_idx < len(self.layers):
                k_tensor = self.layers[layer_idx].keys
                if k_tensor is not None and k_tensor.numel() > 0:
                    return k_tensor.shape[-2]
        elif hasattr(self, "key_cache"):
            k_cache = self.key_cache
            if layer_idx < len(k_cache) and k_cache[layer_idx] is not None:
                if k_cache[layer_idx].numel() > 0:
                    return k_cache[layer_idx].shape[-2]
        return self.get_seq_length(layer_idx)


class SelectionCompressCache(BaseCompressCache):
    """Base class for attention-driven *selection* caches (e.g. H2O, SnapKV).

    Subclasses only decide how raw attention becomes per-token importance (via
    `reduce_attention`) and, optionally, how much budget each layer gets (via
    `get_middle_budget`). All score bookkeeping and the per_head vs global
    top-k pruning are handled here, so every such subclass supports multi-head
    compression out of the box.
    """

    def on_prefill(self, key_states, value_states, layer_idx, cache_kwargs):
        self._grow_scores(layer_idx, key_states)
        return key_states, value_states

    def on_prefill_end(self):
        total_tokens = self.get_seq_length()
        self._update_budget(total_tokens, is_prefill_end=True)
        for layer_idx in list(self.attention_scores.keys()):
            self._consume_attention_scores(layer_idx)
            self._select_prune_middle(layer_idx, total_tokens)

    def on_decode_step(self, key_states, value_states, layer_idx, cache_kwargs):
        self._consume_attention_scores(layer_idx)

        total_after = self.get_seq_length() + key_states.shape[-2]
        self._update_budget(total_after, is_prefill_end=False)
        self._select_prune_middle(layer_idx, total_after)

        self._grow_scores(layer_idx, key_states)
        return key_states, value_states

    def _rank_scores(self, middle_scores: torch.Tensor) -> torch.Tensor:
        """Transform scores for *ranking only* -- what is written back after the
        prune is always the untransformed accumulation, so the transform cannot
        compound across repeated compressions."""
        return middle_scores

    def _select_prune_middle(self, layer_idx: int, total_tokens: int):
        k, _ = self._get_existing_cache(layer_idx)
        if k is None:
            return

        seq_len = k.shape[-2]
        middle_start = self.sink_size
        middle_end = max(self.sink_size, seq_len - self.recent_size)
        if middle_start >= middle_end:
            return

        scores = self.attention_scores[layer_idx]
        middle_scores = scores[..., middle_start:middle_end]

        budget = self.get_middle_budget(layer_idx, total_tokens)
        if budget >= middle_scores.shape[-1]:
            return

        keep_indices = self._middle_keep_indices(self._rank_scores(middle_scores), budget)
        self.prune_middle_cache(layer_idx, keep_indices)

        sink_scores = scores[..., :middle_start]
        recent_scores = scores[..., middle_end:]
        pruned_middle_scores = self._gather_middle_scores(middle_scores, keep_indices)
        self.attention_scores[layer_idx] = torch.cat(
            [sink_scores, pruned_middle_scores, recent_scores], dim=-1)
