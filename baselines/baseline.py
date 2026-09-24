import torch

from evaluation.models.base_cache import BaseCompressCache


class BaselineCache(BaseCompressCache):
    requires_attention = False

    def __init__(self, **kwargs):
        kwargs["compression_size"] = 1.0
        kwargs["mode"] = "entire"
        kwargs["sink_size"] = 0
        kwargs["recent_size"] = 0

        super().__init__(**kwargs)

    def on_prefill(self, key_states: torch.Tensor, value_states: torch.Tensor,
                   layer_idx: int, cache_kwargs: dict):
        self.current_attention_scores.pop(layer_idx, None)
        return key_states, value_states

    def on_prefill_end(self):
        total_tokens = self.get_seq_length()
        self._update_budget(total_tokens, is_prefill_end=True)

    def on_decode_step(self, key_states: torch.Tensor, value_states: torch.Tensor,
                       layer_idx: int, cache_kwargs: dict):
        self.current_attention_scores.pop(layer_idx, None)

        total_after = self.get_seq_length() + key_states.shape[-2]
        self._update_budget(total_after, is_prefill_end=False)

        return key_states, value_states
