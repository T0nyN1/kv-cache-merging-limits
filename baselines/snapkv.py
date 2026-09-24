from typing import Optional

import torch

from evaluation.models.base_cache import SelectionCompressCache


class SnapKVCache(SelectionCompressCache):
    """SnapKV: score tokens by attention within a recent observation window.

    Only differs from H2O by restricting the attention used for scoring to the
    last `observation_window` query positions; scoring, per_head/global top-k
    pruning and bookkeeping are inherited from SelectionCompressCache.

    `pool_kernel` restores the token-axis max-pooling the published SnapKV
    applies to its window scores before top-k (this re-implementation had
    dropped it). Default 0 keeps the historical behaviour so existing cached
    results stay comparable; `snapkv;pool_kernel=7` is the faithful method.
    """
    requires_attention = True

    def __init__(self, observation_window: Optional[int] = None,
                 pool_kernel: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.observation_window = observation_window
        self.pool_kernel = int(pool_kernel or 0)

    def reduce_attention(self, attn_weights: torch.Tensor) -> torch.Tensor:
        scores = attn_weights.detach()
        if self.observation_window is not None and self.observation_window > 0 \
                and scores.dim() >= 2:
            scores = scores[..., -self.observation_window:, :]
        return scores.sum(dim=-2)

    def _rank_scores(self, middle_scores: torch.Tensor) -> torch.Tensor:
        if self.pool_kernel <= 1:
            return middle_scores
        from core.ot_kv_v7 import maxpool_tokens
        return maxpool_tokens(middle_scores, self.pool_kernel)
