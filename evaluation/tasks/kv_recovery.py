from typing import Dict, Any

import scipy.stats
import torch
from transformers.cache_utils import DynamicCache

from .base_evaluator import BaseEvaluator
from .registry import register_task


@register_task("kv_recovery")
class KVRecoveryEvaluator(BaseEvaluator):
    def evaluate(self) -> Dict[str, Any]:
        print("\n[*] Running KV State Recovery (Wasserstein Distance) Experiment...")
        tokenizer = self.model_wrapper.tokenizer
        model = self.model_wrapper._model

        sample_text = "Machine learning focuses on the development of computer programs that can access data and use it learn for themselves. The process of learning begins with observations or data."
        inputs = tokenizer(sample_text, return_tensors="pt").to(model.device)

        custom_cache = self.model_wrapper._setup_cache_and_hooks()
        try:
            with torch.no_grad():
                outputs_compressed = model(
                    inputs.input_ids,
                    use_cache=True,
                    past_key_values=custom_cache,
                )
                cache_compressed = outputs_compressed.past_key_values
                if hasattr(cache_compressed, "on_prefill_end"):
                    cache_compressed.on_prefill_end()
        finally:
            self._cleanup_cache_and_hooks(custom_cache)

        dense_cache = DynamicCache()
        with torch.no_grad():
            outputs_dense = model(
                inputs.input_ids,
                use_cache=True,
                past_key_values=dense_cache
            )
            cache_dense = outputs_dense.past_key_values

        layer_idx = -1

        def get_v_matrix(cache, idx):

            if hasattr(cache, "value_cache"):
                return cache.value_cache[idx]

            if hasattr(cache, "layers"):
                return cache.layers[idx].values

            if isinstance(cache, (tuple, list)):
                return cache[idx][1]
            raise TypeError(f"Unsupported cache type: {type(cache)}")

        v_dense = get_v_matrix(cache_dense, layer_idx).cpu().float().numpy().flatten()
        v_compressed = get_v_matrix(cache_compressed, layer_idx).cpu().float().numpy().flatten()

        w_dist = scipy.stats.wasserstein_distance(v_dense, v_compressed)

        print(f"-> Last layer V Matrix 1D Wasserstein Distance: {w_dist:.4f}")

        return {"kv_recovery": {"wasserstein_distance_1d": w_dist.item()}}
