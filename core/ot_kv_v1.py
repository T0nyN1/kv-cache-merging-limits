"""Frozen replica of the pre-v2 OT-KV implementation (kept for A/B only).

Identical to the original `core/ot_kv.py` before the v2 performance rework, so
that regressions introduced by v2 can be measured rather than argued about.
Do not develop against this file.
"""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from evaluation.models.base_cache import BaseCompressCache

COST_SCALE_EPS = 1e-6


def _validate_transport_mode(transport_mode: str):
    if transport_mode not in {"soft", "hard"}:
        raise ValueError("transport_mode must be 'soft' or 'hard'.")


def _sinkhorn_matrix_space_v1(cost_matrix: torch.Tensor, epsilon: float = 0.05,
                          max_iter: int = 20,
                          target_marginal: Optional[torch.Tensor] = None) -> torch.Tensor:
    K = torch.exp(-cost_matrix.float() / epsilon)

    n_evict = cost_matrix.shape[-2]
    m_anchor = cost_matrix.shape[-1]

    mu = torch.full((n_evict,), 1.0 / n_evict, device=cost_matrix.device, dtype=K.dtype)

    if target_marginal is None:
        nu = torch.full((m_anchor,), 1.0 / m_anchor, device=cost_matrix.device, dtype=K.dtype)
    else:
        nu = target_marginal.to(device=cost_matrix.device, dtype=K.dtype)
        nu = nu.clamp_min(COST_SCALE_EPS)
        nu = nu / nu.sum(dim=-1, keepdim=True)

    u = torch.ones_like(mu)
    v = torch.ones_like(nu)

    for _ in range(int(max_iter)):
        u = mu / torch.matmul(K, v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-9)
        v = nu / torch.matmul(K.transpose(-1, -2), u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-9)

    transport = u.unsqueeze(-1) * K * v.unsqueeze(-2)
    return transport

def _sinkhorn_log_space_v1(cost_matrix: torch.Tensor, epsilon: float = 0.01,
                       max_iter: int = 50,
                       target_marginal: Optional[torch.Tensor] = None) -> torch.Tensor:
    n_evict = cost_matrix.shape[-2]
    m_anchor = cost_matrix.shape[-1]

    C_eps = cost_matrix.float() / epsilon
    f = torch.zeros_like(C_eps[:, :, :, 0])
    g = torch.zeros_like(C_eps[:, :, 0, :])

    mu = torch.full((n_evict,), 1.0 / n_evict, device=cost_matrix.device, dtype=cost_matrix.dtype).log()

    if target_marginal is None:
        nu = torch.full((m_anchor,), 1.0 / m_anchor, device=cost_matrix.device, dtype=cost_matrix.dtype).log()
    else:
        target_marginal = target_marginal.to(device=cost_matrix.device, dtype=cost_matrix.dtype)
        target_marginal = target_marginal.clamp_min(COST_SCALE_EPS)
        target_marginal = target_marginal / target_marginal.sum(dim=-1, keepdim=True).clamp_min(COST_SCALE_EPS)
        nu = target_marginal.log()

    for _ in range(int(max_iter)):
        f = mu - torch.logsumexp(g.unsqueeze(-2) - C_eps, dim=-1)
        g = nu - torch.logsumexp(f.unsqueeze(-1) - C_eps, dim=-2)

    log_t = f.unsqueeze(-1) + g.unsqueeze(-2) - C_eps
    return torch.exp(log_t)


def _anchor_target_capacity(relative_anchor_weight: torch.Tensor, target_beta: float) -> torch.Tensor:
    target_beta = float(target_beta)
    if target_beta == 0.0:
        return torch.ones_like(relative_anchor_weight)
    return relative_anchor_weight.clamp_min(COST_SCALE_EPS).pow(-target_beta)


def _normalize_target_capacity(target_capacity: torch.Tensor) -> torch.Tensor:
    return target_capacity / target_capacity.sum(dim=-1, keepdim=True).clamp_min(COST_SCALE_EPS)


def _build_anchor_target_marginal(relative_anchor_weight: torch.Tensor, target_beta: float) -> Optional[torch.Tensor]:
    if float(target_beta) == 0.0:
        return None
    target_capacity = _anchor_target_capacity(relative_anchor_weight, target_beta)
    return _normalize_target_capacity(target_capacity)


def _fallback_importance_scores(key_states: torch.Tensor) -> torch.Tensor:
    seq_len = key_states.shape[-2]
    if seq_len == 0:
        return key_states.new_empty(key_states.shape[:-1], dtype=torch.float32)

    key_scores = key_states.float()
    weights = torch.norm(key_scores, dim=-1)
    position_weights = 1.0 / (seq_len - torch.arange(seq_len, device=key_states.device, dtype=weights.dtype) + 1.0)
    return weights * position_weights.view(1, 1, -1)


def _prepare_importance_scores(key_states: torch.Tensor, importance_scores: Optional[torch.Tensor]) -> torch.Tensor:
    if importance_scores is None:
        return _fallback_importance_scores(key_states).clamp_min(COST_SCALE_EPS)

    scores = importance_scores.detach().to(device=key_states.device, dtype=torch.float32)
    scores = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores)).clamp_min(0.0)

    empty_heads = scores.sum(dim=-1, keepdim=True) <= COST_SCALE_EPS
    if empty_heads.any():
        fallback_scores = _fallback_importance_scores(key_states)
        scores = torch.where(empty_heads, fallback_scores, scores)

    return scores.clamp_min(COST_SCALE_EPS)


def _build_ot_cost_matrix(dists: torch.Tensor, w_anchor: torch.Tensor, gamma: float) -> Tuple[
    torch.Tensor, torch.Tensor]:
    anchor_mean = w_anchor.mean(dim=-1, keepdim=True).clamp_min(COST_SCALE_EPS)
    relative_anchor_weight = w_anchor / anchor_mean
    anchor_penalty = relative_anchor_weight.clamp_min(COST_SCALE_EPS).pow(gamma)

    raw_cost = dists / anchor_penalty.unsqueeze(-2)
    # cost_scale = raw_cost.flatten(-2).median(dim=-1).values
    cost_scale = raw_cost.flatten(-2).mean(dim=-1)
    cost_scale = cost_scale.clamp_min(COST_SCALE_EPS).view(*cost_scale.shape, 1, 1)

    return raw_cost / cost_scale, relative_anchor_weight


def otkv_compress_v1(key_states: torch.Tensor, value_states: torch.Tensor, budget: int,
                  gamma: float = 1.0, epsilon: float = 0.01,
                  transport_mode: str = "soft", target_beta: float = 0.0,
                  sinkhorn_iters: int = 50,
                  importance_scores: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_transport_mode(transport_mode)
    batch, num_heads, seq_len, head_dim = key_states.shape

    corrected_weights = _prepare_importance_scores(key_states, importance_scores)

    budget = min(max(int(budget), 1), seq_len)
    if seq_len <= budget:
        return key_states, value_states, corrected_weights

    _, anchor_indices = torch.topk(corrected_weights, budget, dim=-1)
    anchor_indices = anchor_indices.sort(dim=-1).values

    gather_index = anchor_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    k_anchor = torch.gather(key_states, 2, gather_index)
    v_anchor = torch.gather(value_states, 2, gather_index)
    w_anchor = torch.gather(corrected_weights, 2, anchor_indices).float()

    evict_mask = torch.ones(batch, num_heads, seq_len, device=key_states.device, dtype=torch.bool)
    evict_mask.scatter_(-1, anchor_indices, False)

    k_evict = key_states[evict_mask].view(batch, num_heads, -1, head_dim)
    v_evict = value_states[evict_mask].view(batch, num_heads, -1, head_dim)

    if k_evict.shape[-2] == 0:
        return k_anchor, v_anchor, w_anchor

    k_evict_norm = F.normalize(k_evict, dim=-1)
    k_anchor_norm = F.normalize(k_anchor, dim=-1)

    dists = 1.0 - torch.matmul(
        k_evict_norm,
        k_anchor_norm.transpose(-1, -2),
    ).float()

    cost_matrix, relative_anchor_weight = _build_ot_cost_matrix(dists, w_anchor, gamma)
    target_marginal = _build_anchor_target_marginal(relative_anchor_weight, target_beta)

    if transport_mode == "soft":
        transport = _sinkhorn_matrix_space_v1(
            cost_matrix,
            epsilon=epsilon,
            max_iter=sinkhorn_iters,
            target_marginal=target_marginal,
        )
        transport = transport.to(value_states.dtype)
        v_merged = v_anchor + torch.matmul(transport.transpose(-1, -2), v_evict)
    else:
        best_anchor = torch.argmin(cost_matrix, dim=-1)
        v_merged = v_anchor.float().clone()
        v_merged.scatter_add_(
            2,
            best_anchor.unsqueeze(-1).expand(-1, -1, -1, head_dim),
            v_evict.float(),
        )

    return k_anchor, v_merged.to(value_states.dtype), w_anchor


class OTKVv1Cache(BaseCompressCache):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gamma = float(kwargs.get('gamma', 1.0))
        self.epsilon = float(kwargs.get('epsilon', 0.01))
        self.transport_mode = kwargs.get('transport_mode', 'soft')
        self.compress_interval = int(kwargs.get('compress_interval', 32))
        self.target_beta = float(kwargs.get('target_beta', 0.0))
        self.sinkhorn_iters = int(kwargs.get('sinkhorn_iters', 50))

        self.attn_scores = {}
        self.decode_steps = {}

    def on_prefill(self, key_states: torch.Tensor, value_states: torch.Tensor,
                   layer_idx: int, cache_kwargs: dict):
        device = key_states.device
        dtype = torch.float32
        batch, num_heads, q_len, _ = key_states.shape

        if layer_idx not in self.attn_scores:
            self.attn_scores[layer_idx] = torch.zeros((batch, num_heads, q_len), device=device, dtype=dtype)
        else:
            new_scores = torch.zeros((batch, num_heads, q_len), device=device, dtype=dtype)
            self.attn_scores[layer_idx] = torch.cat([self.attn_scores[layer_idx], new_scores], dim=-1)

        return key_states, value_states

    def on_prefill_end(self):
        total_tokens = self.get_seq_length()
        self._update_budget(total_tokens, is_prefill_end=True)

        for layer_idx in list(self.attn_scores.keys()):
            self._consume_attention_scores(layer_idx)
            self._compress_layer(layer_idx, total_tokens)

    def on_decode_step(self, key_states: torch.Tensor, value_states: torch.Tensor,
                       layer_idx: int, cache_kwargs: dict):
        device = key_states.device
        dtype = torch.float32
        batch, num_heads, q_len, _ = key_states.shape

        self._consume_attention_scores(layer_idx)

        total_tokens = self.get_seq_length()
        total_after = total_tokens + q_len
        self._update_budget(total_after, is_prefill_end=False)

        decode_step = self.decode_steps.get(layer_idx, 0)

        if decode_step % self.compress_interval == 0:
            self._compress_layer(
                layer_idx,
                total_after,
                reserve_tokens=max(q_len, self.compress_interval)
            )

        self.decode_steps[layer_idx] = decode_step + q_len

        self.attn_scores[layer_idx] = F.pad(self.attn_scores[layer_idx], (0, q_len))

        return key_states, value_states

    def _compress_layer(self, layer_idx: int, total_tokens: int, reserve_tokens: int = 0):
        middle_k, middle_v = self.get_middle_cache(layer_idx)
        if middle_k is None or middle_k.shape[-2] == 0:
            return

        layer_middle_budget = max(
            0,
            int(self.get_middle_budget(layer_idx, total_tokens)) - reserve_tokens
        )

        if layer_middle_budget >= middle_k.shape[-2]:
            return

        seq_len = self._get_existing_cache(layer_idx)[0].shape[-2]
        middle_start = self.sink_size
        middle_end = max(self.sink_size, seq_len - self.recent_size)

        middle_scores = self.attn_scores[layer_idx][..., middle_start:middle_end]

        new_middle_k, new_middle_v, new_middle_scores = otkv_compress_v1(
            key_states=middle_k,
            value_states=middle_v,
            budget=layer_middle_budget,
            gamma=self.gamma,
            epsilon=self.epsilon,
            transport_mode=self.transport_mode,
            target_beta=self.target_beta,
            sinkhorn_iters=self.sinkhorn_iters,
            importance_scores=middle_scores
        )

        self.replace_middle_cache(layer_idx, new_middle_k, new_middle_v)

        sink_scores = self.attn_scores[layer_idx][..., :middle_start]
        recent_scores = self.attn_scores[layer_idx][..., middle_end:]
        self.attn_scores[layer_idx] = torch.cat([sink_scores, new_middle_scores, recent_scores], dim=-1)

    def _consume_attention_scores(self, layer_idx: int):
        attn_weights = self.current_attention_scores.pop(layer_idx, None)
        if attn_weights is None or layer_idx not in self.attn_scores:
            return

        current_scores = self.attn_scores[layer_idx]
        if current_scores.numel() == 0:
            return

        score_update = self._attention_to_token_scores(attn_weights, current_scores.shape)
        score_update = score_update.to(device=current_scores.device, dtype=current_scores.dtype)

        usable = min(current_scores.shape[-1], score_update.shape[-1])
        if usable == 0:
            return

        current_scores[..., :usable] += score_update[..., :usable]

    @staticmethod
    def _attention_to_token_scores(attn_weights: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        scores = attn_weights.detach().float()
        target_batch, target_heads, _ = target_shape

        if scores.dim() == 1:
            scores = scores.view(1, 1, -1)
        elif scores.dim() == 2:
            scores = scores.unsqueeze(0)
        elif scores.dim() > 3:
            while scores.dim() > 3:
                scores = scores.sum(dim=-2)

        score_heads = scores.shape[1]
        if score_heads != target_heads:
            if score_heads == 1:
                scores = scores.expand(-1, target_heads, -1)
            elif score_heads % target_heads == 0:
                group = score_heads // target_heads
                scores = scores.reshape(scores.shape[0], target_heads, group, scores.shape[-1]).sum(dim=2)
            elif target_heads % score_heads == 0:
                repeat = target_heads // score_heads
                scores = scores.repeat_interleave(repeat, dim=1)
            else:
                scores = scores.mean(dim=1, keepdim=True).expand(-1, target_heads, -1)

        return scores
