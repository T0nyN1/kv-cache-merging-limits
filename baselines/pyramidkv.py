import math

from baselines.snapkv import SnapKVCache


class PyramidKVCache(SnapKVCache):

    def __init__(self, pyramid_low_scale: float = 1.5, pyramid_high_scale: float = 0.5,
                 num_layers: int = None, **kwargs):
        super().__init__(**kwargs)
        self.pyramid_low_scale = float(pyramid_low_scale)
        self.pyramid_high_scale = float(pyramid_high_scale)
        # True model depth. When known up front (injected by the wrapper), the
        # per-layer pyramid budget is correct regardless of pruning order; the
        # max() in get_middle_budget only serves as a fallback for the legacy
        # path where depth is discovered incrementally.
        self.num_layers = num_layers

    def get_middle_budget(self, layer_idx: int, total_tokens: int) -> int:
        """Depth-dependent budget.

        Scales the *base* middle budget the parent computed, rather than
        recomputing a layer budget from `total_tokens`. Recomputing is wrong once
        compression has happened: during decode `total_tokens` is the length of
        the already-compressed cache, so each pass shrinks the budget again and
        the cache collapses (measured: [90,70,50,30] -> [11,11,11,11] over 30
        decode steps in mode="prefill", where the budget must stay fixed).
        Scaling the parent's value inherits the correct prefill/entire semantics.
        """
        self.num_layers = max(self.num_layers or 0, layer_idx + 1)
        num_layers = max(1, self.num_layers)

        if num_layers == 1:
            scale = 1.0
        else:
            depth = layer_idx / float(num_layers - 1)
            scale = self.pyramid_low_scale + depth * (self.pyramid_high_scale - self.pyramid_low_scale)

        mean_scale = 0.5 * (self.pyramid_low_scale + self.pyramid_high_scale)
        if mean_scale <= 0:
            mean_scale = 1.0

        base_middle = super().get_middle_budget(layer_idx, total_tokens)
        return max(0, math.floor(base_middle * scale / mean_scale))
