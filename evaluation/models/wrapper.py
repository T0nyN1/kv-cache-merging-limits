import torch
import torch.nn.functional as F
from lm_eval.models.huggingface import HFLM
from tqdm import tqdm
from transformers.cache_utils import DynamicCache


def _capture_window_queries(module, hidden_states, kwargs, n_window):
    """Post-RoPE queries of the last `n_window` positions, (b, q_heads, W, d),
    computed along the projection -> (q_norm) -> RoPE path the attention module
    itself uses. Only called for caches that declare `wants_window_queries`
    (KeepKV's zero-perturbation merge needs unnormalised scores exp(q.k/sqrt d),
    which the post-softmax attention weights do not determine)."""
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    pos = kwargs.get("position_embeddings")
    if pos is None:
        return None
    head_dim = getattr(module, "head_dim", None) or module.q_proj.out_features // module.config.num_attention_heads
    hs = hidden_states[:, -n_window:, :]
    q = module.q_proj(hs).view(*hs.shape[:-1], -1, head_dim)
    if hasattr(module, "q_norm"):
        q = module.q_norm(q)
    q = q.transpose(1, 2)
    cos, sin = pos
    cos, sin = cos[:, -n_window:, :], sin[:, -n_window:, :]
    q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos, sin)
    return q.detach()


def _get_attention_pre_hook(cache_obj=None, layer_idx=None):
    def pre_hook(module, args, kwargs):
        hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states")

        n_q = int(getattr(cache_obj, "wants_window_queries", 0) or 0) if cache_obj is not None else 0
        if n_q > 0 and hidden_states is not None and hidden_states.shape[1] > 1:
            with torch.no_grad():
                q = _capture_window_queries(module, hidden_states, kwargs, min(n_q, hidden_states.shape[1]))
            if q is not None:
                cache_obj.window_queries[layer_idx] = q

        if hidden_states is not None and hidden_states.shape[1] == 1:
            if "attention_mask" in kwargs:
                kwargs["attention_mask"] = None
            # Consolidating caches (core/consolidate.py) carry a per-slot logit
            # bias for merged entries. Any pending maintenance compression must
            # run *before* the bias mask is built, so mask and cache agree.
            if cache_obj is not None and getattr(cache_obj, "consolidate", False):
                cache_obj.pre_attention_step(layer_idx)
                bias = cache_obj.get_decode_bias(layer_idx)
                if bias is not None and hidden_states is not None:
                    kwargs["attention_mask"] = bias.to(hidden_states.dtype)

        return args, kwargs

    return pre_hook


def _get_attention_hook(cache_obj, layer_idx):
    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple) and len(outputs) > 1:
            attn_weights = outputs[1]
            if attn_weights is None:
                raise RuntimeError(
                    "Attention weights are required for this KV cache method but the "
                    "attention module returned None. This method needs eager attention "
                    "that emits attention scores; ensure attn_implementation='eager' and "
                    "that the transformers version returns attn_weights from self_attn."
                )
            with torch.no_grad():
                accumulated_score = cache_obj.reduce_attention(attn_weights)
                cache_obj.current_attention_scores[layer_idx] = accumulated_score
            try:
                attn_weights.untyped_storage().resize_(0)
            except RuntimeError:
                pass

            new_outputs = list(outputs)
            new_outputs[1] = None
            return tuple(new_outputs)

        return outputs

    return hook


class EvaluatorHFLM(HFLM):
    def __init__(self, pretrained: str, cache_class=None, cache_kwargs=None,
                 prefill_fraction=0.1, max_length=4096, **kwargs):

        kwargs["attn_implementation"] = "eager"
        super().__init__(pretrained=pretrained, max_length=max_length, **kwargs)

        self.prefill_fraction = prefill_fraction
        self.cache_class = cache_class
        self.cache_kwargs = cache_kwargs or {}
        self._hooks = []

    def _setup_cache_and_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        if self.cache_class is None:
            return DynamicCache()

        cache_instance = self.cache_class(**self.cache_kwargs)
        needs_attn = getattr(cache_instance, "requires_attention", True)

        layers = self._model.model.layers

        # Provide the true model depth up front to depth-dependent caches (e.g.
        # PyramidKV), so per-layer budgets don't depend on pruning traversal order.
        if getattr(cache_instance, "num_layers", None) is None and hasattr(cache_instance, "num_layers"):
            cache_instance.num_layers = len(layers)
        for layer_idx, layer in enumerate(layers):
            if needs_attn:
                hook_handle = layer.self_attn.register_forward_hook(
                    _get_attention_hook(cache_instance, layer_idx)
                )
                self._hooks.append(hook_handle)

            pre_hook_handle = layer.self_attn.register_forward_pre_hook(
                _get_attention_pre_hook(cache_instance, layer_idx), with_kwargs=True
            )
            self._hooks.append(pre_hook_handle)

        return cache_instance

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        results = []
        iterator = requests if disable_tqdm else tqdm(requests, desc="Autoregressive PPL")

        self._model.eval()
        device = self._model.device

        with torch.no_grad():
            for req in iterator:
                text = req.args[0]
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)

                if getattr(self.tokenizer, "bos_token_id", None) is not None:
                    if len(token_ids) == 0 or token_ids[0] != self.tokenizer.bos_token_id:
                        token_ids = [self.tokenizer.bos_token_id] + token_ids

                if len(token_ids) < 3:
                    results.append(0.0)
                    continue

                split_idx = max(1, int(len(token_ids) * self.prefill_fraction))
                split_idx = min(split_idx, len(token_ids) - 1)

                prefix_ids = torch.tensor([token_ids[:split_idx]], device=device)
                target_ids = torch.tensor([token_ids[split_idx:]], device=device)

                print(f"\n" + "=" * 55)
                print(f"[Doc Monitor] Processing New Document")
                print(f"[Doc Monitor] Total Tokens   : {len(token_ids)}")
                print(f"[Doc Monitor] Prefill Tokens : {prefix_ids.shape[1]}")
                print(f"[Doc Monitor] Decode Steps   : {target_ids.shape[1]}")
                print("=" * 55)
                print()

                past_key_values = self._setup_cache_and_hooks()

                outputs = self._model(
                    input_ids=prefix_ids,
                    use_cache=True,
                    past_key_values=past_key_values,
                    return_dict=True
                )

                last_logit = outputs.logits[:, -1:, :]
                total_logprob = 0.0
                decode_seq_len = target_ids.shape[1]

                for i in range(decode_seq_len):
                    target_token = target_ids[:, i:i + 1]

                    log_probs = F.log_softmax(last_logit, dim=-1)
                    token_logprob = log_probs.gather(-1, target_token.unsqueeze(-1)).squeeze()
                    total_logprob += token_logprob.item()

                    if i == decode_seq_len - 1:
                        break

                    outputs = self._model(
                        input_ids=target_token,
                        past_key_values=past_key_values,
                        use_cache=True,
                        position_ids=torch.tensor([[split_idx + i]], device=device),
                        return_dict=True
                    )

                    last_logit = outputs.logits

                print()
                results.append(total_logprob)

        for h in self._hooks:
            h.remove()

        return results
