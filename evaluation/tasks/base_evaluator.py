from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseEvaluator(ABC):

    def __init__(self, model_wrapper, **kwargs):
        self.model_wrapper = model_wrapper
        self.args = kwargs

    def _cleanup_cache_and_hooks(self, cache=None):
        for hook in getattr(self.model_wrapper, "_hooks", []):
            hook.remove()
        self.model_wrapper._hooks.clear()

        if cache is not None and hasattr(cache, "current_attention_scores"):
            cache.current_attention_scores.clear()

    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:

        pass
