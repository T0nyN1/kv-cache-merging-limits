from evaluation.models.base_cache import SelectionCompressCache


class H2OCache(SelectionCompressCache):
    """Heavy-Hitter Oracle: keep tokens with the largest accumulated attention.

    All behaviour (per-token score accumulation, per_head/global top-k pruning)
    is inherited from SelectionCompressCache. H2O uses the default
    `reduce_attention`, which sums attention over the full query window.
    """
    pass
