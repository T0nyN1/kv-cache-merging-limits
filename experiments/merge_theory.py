"""Test the theory's prediction about which distance decides mergeability.

The derivation (docs/ot_kv_theory.md) says a query-independent merge can only
cancel the eviction error exactly when an evicted key duplicates an anchor key,
and that the achievable reduction in squared error for evicted token `i` routed
to anchor `j` is

    e^{-sigma_ij^2},      sigma_ij^2 = (k_i - k_j)^T  Sigma_q  (k_i - k_j)

where `Sigma_q` is the second moment of the *queries* that will read the cache.
Everything hinges on sigma, and sigma is a Mahalanobis distance under the query
covariance -- not the cosine distance in raw key space that v1-v4 all use.

This script measures, on a real model:

1. how well cosine distance predicts sigma (if it predicts it badly, the
   transport cost is optimising the wrong thing);
2. whether routing to the whitened-nearest anchor gives a smaller sigma than
   routing to the cosine-nearest one, and by how much;
3. what that implies for the attainable error reduction, e^{-sigma^2}.

Queries are captured post-RoPE from a held-out continuation, so they are the
queries the compressed cache will actually face.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from experiments.data import load_wikitext_docs


def capture_kq(model, ids, n_probe, device):
    """Return per-layer post-RoPE keys (KV heads) and probe queries (query heads).

    The queries come from the last `n_probe` positions, which stand in for the
    queries a compressed cache will be read with.
    """
    keys, queries = {}, {}

    def pre_hook(layer_idx):
        def hook(module, args, kwargs):
            hs = args[0] if args else kwargs.get("hidden_states")
            pos = kwargs.get("position_embeddings")
            if hs is None or pos is None:
                return args, kwargs
            with torch.no_grad():
                from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
                shape = (*hs.shape[:-1], -1, module.head_dim)
                q = module.q_norm(module.q_proj(hs).view(shape)).transpose(1, 2)
                k = module.k_norm(module.k_proj(hs).view(shape)).transpose(1, 2)
                cos, sin = pos
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                keys[layer_idx] = k.detach().float()               # (b, kv_h, L, d)
                queries[layer_idx] = q.detach().float()[:, :, -n_probe:, :]
            return args, kwargs

        return hook

    handles = [layer.self_attn.register_forward_pre_hook(pre_hook(i), with_kwargs=True)
               for i, layer in enumerate(model.model.layers)]
    try:
        with torch.no_grad():
            model(input_ids=ids, use_cache=True, past_key_values=DynamicCache())
    finally:
        for h in handles:
            h.remove()
    return keys, queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--docs", type=int, default=3)
    ap.add_argument("--prefill", type=int, default=3072)
    ap.add_argument("--probe", type=int, default=128)
    ap.add_argument("--budget", type=float, default=0.15)
    ap.add_argument("--layers", type=int, default=8, help="evaluate every Nth layer")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="sdpa").to(args.device).eval()
    n_kv = model.config.num_key_value_heads
    n_q = model.config.num_attention_heads
    group = n_q // n_kv

    docs = load_wikitext_docs(min_tokens=args.prefill + args.probe, max_docs=args.docs, tokenizer=tok)
    stats = {k: [] for k in ("cos_sigma", "maha_sigma", "corr", "red_cos", "red_maha")}

    for text in docs:
        ids = torch.tensor([tok.encode(text, add_special_tokens=False)[:args.prefill]], device=args.device)
        keys, queries = capture_kq(model, ids, args.probe, args.device)

        for li in range(0, len(keys), args.layers):
            K = keys[li][0]                                    # (kv_h, L, d)
            Q = queries[li][0]                                 # (q_h, P, d)
            for h in range(K.shape[0]):
                k = K[h]                                       # (L, d)
                q = Q[h * group:(h + 1) * group].reshape(-1, k.shape[-1])   # (group*P, d)

                # sigma^2 = Delta^T Sigma_q Delta ; use the *centred* query
                # second moment, since a shared mean shifts every token equally
                qc = q - q.mean(0, keepdim=True)
                Sig = (qc.T @ qc) / qc.shape[0]

                budget = max(2, int(k.shape[0] * args.budget))
                # anchors: evenly spaced stand-ins for "some selection rule"; the
                # question here is the metric, not the rule
                a_idx = torch.linspace(0, k.shape[0] - 1, budget, device=k.device).long()
                e_idx = torch.tensor([i for i in range(k.shape[0]) if i not in set(a_idx.tolist())],
                                     device=k.device)
                if len(e_idx) < 32:
                    continue
                e_idx = e_idx[torch.randperm(len(e_idx), device=k.device)[:256]]

                ka, ke = k[a_idx], k[e_idx]                    # (m,d) (n,d)
                D = ke.unsqueeze(1) - ka.unsqueeze(0)          # (n,m,d)

                # true sigma of the log attention ratio, per (evicted, anchor).
                # The attention logit is q.k / sqrt(d), so the ratio's log is
                # q.(k_i - k_j) / sqrt(d) -- the scaling matters, without it
                # sigma comes out sqrt(128)x too large.
                scale = k.shape[-1] ** -0.5
                logits = (D @ q.T) * scale                     # (n,m,Nq)
                sigma = logits.std(dim=-1)                     # (n,m)

                cos = 1 - torch.nn.functional.normalize(ke, dim=-1) @ \
                      torch.nn.functional.normalize(ka, dim=-1).T
                maha = (torch.einsum("nmd,de,nme->nm", D, Sig, D).clamp_min(0).sqrt() * scale)

                # 1. does cosine track sigma?
                f = torch.stack([cos.flatten(), sigma.flatten()])
                stats["corr"].append(float(torch.corrcoef(f)[0, 1]))

                # 2. sigma achieved by each routing rule
                s_cos = sigma.gather(1, cos.argmin(1, keepdim=True)).squeeze(1)
                s_mah = sigma.gather(1, maha.argmin(1, keepdim=True)).squeeze(1)
                stats["cos_sigma"].append(float(s_cos.mean()))
                stats["maha_sigma"].append(float(s_mah.mean()))

                # 3. attainable squared-error reduction e^{-sigma^2}
                stats["red_cos"].append(float((-s_cos.pow(2)).exp().mean()))
                stats["red_maha"].append(float((-s_mah.pow(2)).exp().mean()))
        del keys, queries

    n = len(stats["corr"])
    m = {k: sum(v) / len(v) for k, v in stats.items()}
    print(f"\n{n} (layer, head) samples over {len(docs)} documents, budget {args.budget}\n")
    print(f"correlation( cosine distance , true sigma )   : {m['corr']:.3f}")
    print(f"                                                (1.0 would mean cosine is the right metric)\n")
    print(f"mean sigma, routing by cosine                 : {m['cos_sigma']:.3f} nats")
    print(f"mean sigma, routing by query-whitened distance : {m['maha_sigma']:.3f} nats")
    print(f"  -> sigma reduced by {100 * (1 - m['maha_sigma'] / m['cos_sigma']):.1f}%\n")
    print(f"attainable squared-error reduction e^-sigma^2 :")
    print(f"  cosine routing         : {m['red_cos']:.4f}  (merging can remove {100 * m['red_cos']:.1f}% of the error)")
    print(f"  query-whitened routing : {m['red_maha']:.4f}  (merging can remove {100 * m['red_maha']:.1f}% of the error)")


if __name__ == "__main__":
    main()
