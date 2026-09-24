"""Shared plumbing for compensated (class-B) merge ports: a per-slot logit bias
that rides along the cache and is injected at every decode query.

The bias path is the one core/otkv_v8_cache.py and core/consolidate.py use: the
evaluator's attention pre-hook (evaluation/models/wrapper.py) sees the duck-typed
`consolidate = True`, calls `pre_attention_step` so any maintenance pruning
happens *before* the mask is built, and passes `get_decode_bias` as an additive
attention mask. Everything else -- selection, budget accounting, decode-time
maintenance -- is SnapKVCache's, so a subclass with `merge=False` is the
identical eviction cache (checked by experiments/test_compensated_ports.py).

Subclasses provide:
  _take_pending(layer_idx)        -> prefill statistics for this layer, or None
  _keep_middle(layer_idx, mid, budget, ms, me, extra) -> sorted 1-D keep indices
                                     into the middle region (default: SnapKV's rule)
  _merge_layer(layer_idx, keep_mid, ms, me, extra)    -> rewrite cache K/V in place
                                     and set self.logit_bias[layer_idx] (1, kv, L)
"""
import json
import os
from collections import defaultdict
from typing import Optional

import torch
import torch.nn.functional as F

from baselines.snapkv import SnapKVCache


def _as_bool(x):
    return x if isinstance(x, bool) else str(x).lower() in ("1", "true", "yes")


class BiasCompensatedCache(SnapKVCache):

    consolidate = True          # read by the evaluator's bias-aware pre-hook

    def __init__(self, merge=True, **kwargs):
        super().__init__(**kwargs)
        if self.per_head:
            raise NotImplementedError(f"{type(self).__name__} shares one token layout across heads: "
                                      f"run with per_head=false")
        self.merge = _as_bool(merge)
        self.logit_bias = {}
        self._has_bias = {}
        self._gqa_groups = 1
        self._prehook_seen = False
        self._pending = {}
        self._merged_layers = set()
        self.merge_stats = defaultdict(float)

    # ------------------------------------------------------------ subclass hooks
    def _take_pending(self, layer_idx: int):
        return self._pending.pop(layer_idx, None)

    def _keep_middle(self, layer_idx, mid, budget, ms, me, extra):
        return self._middle_keep_indices(self._rank_scores(mid), budget)

    def _merge_layer(self, layer_idx, keep_mid, ms, me, extra):
        raise NotImplementedError

    def _require_pending(self, layer_idx, extra):
        """Called only when this layer evicts at prefill and merging is on; subclasses raise here
        if the statistics the operator needs are missing, instead of silently evicting."""

    def _after_selection(self, layer_idx):
        """Per-layer cleanup after a selection step."""

    # ------------------------------------------------------------ selection + merge
    def _select_prune_middle(self, layer_idx: int, total_tokens: int):
        k, _ = self._get_existing_cache(layer_idx)
        if k is None:
            return
        seq_len = k.shape[-2]
        ms = self.sink_size
        me = max(self.sink_size, seq_len - self.recent_size)
        if ms >= me:
            return
        scores = self.attention_scores[layer_idx]
        mid = scores[..., ms:me]
        budget = self.get_middle_budget(layer_idx, total_tokens)
        extra = self._take_pending(layer_idx)
        if budget >= mid.shape[-1]:
            return
        keep = self._keep_middle(layer_idx, mid, budget, ms, me, extra)
        if self.merge and layer_idx not in self._merged_layers:
            self._require_pending(layer_idx, extra)
            if extra is not None:
                self._merge_layer(layer_idx, keep, ms, me, extra)
                self._merged_layers.add(layer_idx)
        self._after_selection(layer_idx)
        self.prune_middle_cache(layer_idx, keep)
        self.attention_scores[layer_idx] = torch.cat(
            [scores[..., :ms], self._gather_middle_scores(mid, keep), scores[..., me:]], dim=-1)
        bias = self.logit_bias.get(layer_idx)
        if bias is not None:
            bm = bias[..., ms:me].index_select(-1, keep.to(bias.device))
            self.logit_bias[layer_idx] = torch.cat([bias[..., :ms], bm, bias[..., me:]], dim=-1)

    # ------------------------------------------------------------ decode-time bias
    def _emit_merge_stats(self):
        """One line per sample: to stdout, and to MERGE_STATS_PATH when the runner sets it
        (misc/modal/evaluate.py does, from save_dir), so a remote run's counters are recoverable."""
        stats = {k: float(v) for k, v in sorted(self.merge_stats.items())}
        print(f"[{type(self).__name__}] " + " ".join(f"{k} {v:.4g}" for k, v in stats.items()), flush=True)
        path = os.environ.get("MERGE_STATS_PATH")
        if not path:
            return
        rec = {"cache": type(self).__name__, "method": os.environ.get("OTKV_METHOD", ""),
               "merge": self.merge, "stats": stats}
        try:
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    def pre_attention_step(self, layer_idx: int):
        self._prehook_seen = True
        if self.finalize_prefill() and layer_idx == 0 and self.merge_stats:
            self._emit_merge_stats()
        self._consume_attention_scores(layer_idx)
        total = self.get_seq_length()
        self._update_budget(total + 1, is_prefill_end=False)
        self._select_prune_middle(layer_idx, total + 1)

    def get_decode_bias(self, layer_idx: int) -> Optional[torch.Tensor]:
        if not self._has_bias.get(layer_idx):
            return None
        bias = self.logit_bias.get(layer_idx)
        if bias is None:
            return None
        x = F.pad(bias, (0, 1))                               # incoming token: bias 0
        if self._gqa_groups > 1:
            x = x.repeat_interleave(self._gqa_groups, dim=1)
        return x.unsqueeze(2)

    def on_decode_step(self, key_states, value_states, layer_idx, cache_kwargs):
        q_len = key_states.shape[-2]
        self._consume_attention_scores(layer_idx)
        total_after = self.get_seq_length() + q_len
        self._update_budget(total_after, is_prefill_end=False)
        if not self._prehook_seen:
            self._select_prune_middle(layer_idx, total_after)
        self._grow_scores(layer_idx, key_states)
        if layer_idx in self.logit_bias:
            self.logit_bias[layer_idx] = F.pad(self.logit_bias[layer_idx], (0, q_len))
        return key_states, value_states

    def _set_bias(self, layer_idx, bias):
        """bias: (1, kv, L) float32, 0 where the slot carries no compensation."""
        self.logit_bias[layer_idx] = bias
        self._has_bias[layer_idx] = bool((bias != 0).any())
