"""OT-KV v4 — the measured configuration of the v3 engine.

v3 introduced the mass-conserving transport (see `core/ot_kv_v3.py` and
`docs/ot_kv_v3.md`). v4 changes no maths; it changes four defaults that the
measurements in `docs/experiments.md` say were wrong, and adds one mechanism.

1. **Near-hard transport.** Sweeping the entropic regularisation showed the
   softness is a cost, not a benefit: KL 0.1748 at `epsilon=0.02` against 0.1808
   at 0.5, which is worse than not merging at all. The value of OT here is the
   *capacity constraint*, not the spread of the plan. v4 defaults to
   `epsilon=0.02`.

2. **Four candidates, not sixteen.** `top_r=4` matches `top_r=16` on quality
   (0.1766 vs 0.1798) and beats `top_r=64` (0.1839), while cutting the
   compression pass from 8.0 s to 3.4 s. Spreading a value over many anchors
   dilutes it.

3. **Compress often enough to match the budget.** v3 compressed every 32 decode
   steps, so its cache grew to `budget + 32` between passes -- about 8 % more KV
   than the baselines it is compared against. Capping the peak at budget instead
   cost ~0.6 LongBench points, which was the entire margin being claimed. v4
   compresses every 4 steps, which holds 99.3 % of budget on average and is the
   knee of the curve: interval 1 costs 174 ms/token for identical quality,
   interval 4 costs 87 ms, and interval 16 saves only 4 ms more while giving up
   cache parity.

4. **Pooled evidence for selection, per-head transport.** `select_scope="global"`
   sums the per-head scores before ranking and gives every head the same token
   set, while the transport and the merged values stay per-head. This is the
   fix for the finding that per-head selection *hurts* short-window scoring:
   with GQA the 32 rows of a SnapKV-style window are split eight ways, which is
   too little evidence to rank on (SnapKV itself scores 23.46 global against
   22.87 per-head on LongBench). All-query scoring has evidence to spare and
   prefers `select_scope="head"`.

The two operating points, both exported below as presets, differ only in how
importance is estimated -- which is a property of the task, not a tuned knob:

* `preset="qa"` — a question at the end of the prompt decides what matters, so
  score on the recent window and pool it across heads.
* `preset="continuation"` — future queries resemble past ones, so accumulate
  over every query per head and spend part of the budget covering the key cloud.
"""

from core.ot_kv_v3 import OTKVv3Cache

PRESETS = {
    # Decide the regime per layer from the attention itself (see
    # OTKVv3Cache._regime): both score streams are kept, and each compression
    # ranks on whichever one the prompt's shape says is informative.
    "auto": dict(
        observation_window=32,
        split_scores=True,
        auto_regime=True,
        select_scope="global",
        select_mode="topk",
    ),
    # A question at the tail of the prompt defines relevance. Nothing earlier
    # marks the answer as important, so all-query accumulation cannot find it.
    "qa": dict(
        observation_window=32,
        split_scores=True,
        select_scope="global",
        select_mode="topk",
    ),
    # Open-ended continuation: future queries look like past ones, so all-query
    # mass is the right predictor and coverage of the key cloud pays.
    "continuation": dict(
        observation_window=None,
        split_scores=False,
        select_scope="head",
        select_mode="kmeans:50",
        select_decode=True,
        # Re-running the k-means quantisation costs ~6 ms per layer per token,
        # so it cannot run every step the way the cheap top-k path can. Every
        # 8 steps keeps the mean cache at 98 % of budget for ~0.8 ms/layer/token.
        compress_interval=8,
    ),
}

V4_DEFAULTS = dict(
    epsilon=0.02,
    top_r=4,
    compress_interval=4,
    merge=True,
    merge_strength=0.5,
    gate_sigma=0.3,
    cost_alpha=0.8,
    capacity_beta=1.0,
    per_head=True,
)


class OTKVv4Cache(OTKVv3Cache):
    """v3's transport with the defaults the sweeps selected.

    `preset` picks one of the two operating points; any explicit kwarg still
    wins, so ablations stay expressible.
    """

    def __init__(self, **kwargs):
        preset = kwargs.pop("preset", "qa")
        if preset not in PRESETS:
            raise ValueError(f"preset must be one of {sorted(PRESETS)}, got {preset!r}")

        merged = dict(V4_DEFAULTS)
        merged.update(PRESETS[preset])
        merged.update(kwargs)          # caller's explicit choices win
        super().__init__(**merged)
        self.preset = preset
