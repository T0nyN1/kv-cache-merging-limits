"""Verify the sigma mechanism across model families.

The closure argument (docs/ot_kv_theory.md §6-7) was measured on Qwen3-1.7B.
Its mechanism — RoPE rotation plus contextualisation manufacture the key
differences merging needs to be absent — predicts the same numbers for any
rotary transformer. This script measures, for a given model:

  1. oracle-routing sigma at budget 0.15: each evicted token routed to its
     globally best anchor; median sigma, fraction under the value-only
     break-even (0.285) and under the compensated-pair reference (0.9);
  2. same-token-id pair sigma by positional distance (RoPE vs context split);
  3. oracle sigma on a passage repeated verbatim 8x (the best case for
     redundancy, and the strongest refutation when it fails).

Architecture support: llama / mistral / qwen2 (plain RoPE attention) and
qwen3 (extra q/k RMSNorm). Everything is captured post-RoPE with a probe of
held-out final-position queries, exactly as the original measurement did.

    python experiments/sigma_families.py --model <id> --device cuda --out out.json
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from experiments.data import load_wikitext_docs

BREAK_EVEN = 0.285          # value-only merging (docs/ot_kv_theory.md §6)
PAIR_REF = 0.90             # compensated adjacent-pair reference scale


def capture_kq(model, ids, n_probe):
    """Post-RoPE keys (KV heads) and probe queries (query heads), any
    llama-family attention module (optional q_norm/k_norm)."""
    keys, queries = {}, {}

    def pre_hook(layer_idx):
        def hook(module, args, kwargs):
            hs = args[0] if args else kwargs.get("hidden_states")
            pos = kwargs.get("position_embeddings")
            if hs is None or pos is None:
                return args, kwargs
            with torch.no_grad():
                head_dim = getattr(module, "head_dim", None) or \
                    module.q_proj.out_features // module.config.num_attention_heads
                shape = (*hs.shape[:-1], -1, head_dim)
                q = module.q_proj(hs).view(shape)
                k = module.k_proj(hs).view(shape)
                if hasattr(module, "q_norm"):
                    q = module.q_norm(q)
                if hasattr(module, "k_norm"):
                    k = module.k_norm(k)
                q, k = q.transpose(1, 2), k.transpose(1, 2)
                cos, sin = pos
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                keys[layer_idx] = k.detach().float()
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


def head_iter(model, keys, queries, layer_stride):
    group = model.config.num_attention_heads // model.config.num_key_value_heads
    for li in range(0, len(keys), layer_stride):
        K, Q = keys[li][0], queries[li][0]
        for h in range(K.shape[0]):
            k = K[h]
            q = Q[h * group:(h + 1) * group].reshape(-1, k.shape[-1])
            yield k, q


def oracle_sigma(model, keys, queries, budget, layer_stride, n_evict_sample=192, thresholds=()):
    """Median oracle sigma + break-even fractions, pooled over heads/layers.
    `thresholds` adds frac_le_<t> entries for extra break-even values (e.g. a
    model's own measured centroid-gap threshold)."""
    med, f_be, f_pair = [], [], []
    f_extra = {float(t): [] for t in thresholds}
    for k, q in head_iter(model, keys, queries, layer_stride):
        L, d = k.shape
        sc = d ** -0.5
        m = max(2, int(L * budget))
        a = torch.linspace(0, L - 1, m, device=k.device).long()
        mask = torch.ones(L, dtype=torch.bool, device=k.device)
        mask[a] = False
        e = torch.nonzero(mask).flatten()
        e = e[torch.randperm(len(e), device=k.device)[:n_evict_sample]]
        # sigma to every anchor: std over probe queries of q·(k_e - k_a)/sqrt(d)
        proj_e = (k[e] @ q.T) * sc                     # (n, P)
        proj_a = (k[a] @ q.T) * sc                     # (m, P)
        diff = proj_e.unsqueeze(1) - proj_a.unsqueeze(0)
        sigma_best = diff.std(dim=-1).min(dim=1).values
        med.append(float(sigma_best.median()))
        f_be.append(float((sigma_best <= BREAK_EVEN).float().mean()))
        f_pair.append(float((sigma_best <= PAIR_REF).float().mean()))
        for t in f_extra:
            f_extra[t].append(float((sigma_best <= t).float().mean()))
    n = len(med)
    out = {"median_sigma": sum(med) / n, "frac_le_0.285": sum(f_be) / n,
           "frac_le_0.9": sum(f_pair) / n, "heads": n}
    for t, v in f_extra.items():
        out[f"frac_le_{t:g}"] = sum(v) / n
    return out


def same_token_sigma(model, keys, queries, token_ids, layer_stride,
                     bins=((0, 8), (8, 64), (64, 512), (512, 100000))):
    """Sigma between positions holding the same token id, binned by distance."""
    ids = token_ids.tolist()
    pos_by_tok = {}
    for p, t in enumerate(ids):
        pos_by_tok.setdefault(t, []).append(p)
    pairs = []
    for plist in pos_by_tok.values():
        if len(plist) < 2:
            continue
        for i in range(len(plist) - 1):
            for j in range(i + 1, min(i + 5, len(plist))):
                pairs.append((plist[i], plist[j]))
    if not pairs:
        return {}
    pairs = pairs[:4000]
    pi = torch.tensor([p[0] for p in pairs])
    pj = torch.tensor([p[1] for p in pairs])
    dist = (pj - pi).abs()

    out = {}
    sig_sum = None
    n_heads = 0
    for k, q in head_iter(model, keys, queries, layer_stride):
        sc = k.shape[-1] ** -0.5
        d = (k[pi.to(k.device)] - k[pj.to(k.device)])
        sigma = ((d @ q.T) * sc).std(dim=-1).cpu()
        sig_sum = sigma if sig_sum is None else sig_sum + sigma
        n_heads += 1
    sigma = sig_sum / n_heads
    for lo, hi in bins:
        m = (dist >= lo) & (dist < hi)
        if int(m.sum()) == 0:
            continue
        out[f"{lo}-{hi}"] = {"pairs": int(m.sum()),
                             "median_sigma": float(sigma[m].median()),
                             "frac_le_0.285": float((sigma[m] <= BREAK_EVEN).float().mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--docs", type=int, default=3)
    ap.add_argument("--prefill", type=int, default=2560)
    ap.add_argument("--probe", type=int, default=128)
    ap.add_argument("--budget", type=float, default=0.15)
    ap.add_argument("--layer-stride", type=int, default=4)
    ap.add_argument("--thresholds", type=float, nargs="*", default=[],
                    help="extra break-even sigma values to report fractions for (e.g. a model's own "
                         "centroid-gap threshold from experiments/centroid_gap.py)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype),
        attn_implementation="eager").to(args.device).eval()

    docs = load_wikitext_docs(min_tokens=args.prefill + args.probe,
                              max_docs=args.docs, tokenizer=tok)
    agg = {"model": args.model, "oracle": [], "same_token": [], "repeat8x": []}

    for text in docs:
        enc = tok.encode(text, add_special_tokens=False)[:args.prefill]
        ids = torch.tensor([enc], device=args.device)
        keys, queries = capture_kq(model, ids, args.probe)
        agg["oracle"].append(oracle_sigma(model, keys, queries, args.budget, args.layer_stride,
                                          thresholds=args.thresholds))
        agg["same_token"].append(same_token_sigma(model, keys, queries,
                                                  torch.tensor(enc), args.layer_stride))
        del keys, queries

    # verbatim 8x repetition — redundancy's best case
    base = tok.encode(docs[0], add_special_tokens=False)[:args.prefill // 8]
    rep = (base * 8)[:args.prefill]
    ids = torch.tensor([rep], device=args.device)
    keys, queries = capture_kq(model, ids, args.probe)
    agg["repeat8x"].append(oracle_sigma(model, keys, queries, args.budget, args.layer_stride,
                                        thresholds=args.thresholds))
    del keys, queries

    def avg(dicts, key):
        vals = [d[key] for d in dicts if key in d]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "model": args.model,
        "oracle_median_sigma": avg(agg["oracle"], "median_sigma"),
        "oracle_frac_le_0.285": avg(agg["oracle"], "frac_le_0.285"),
        "oracle_frac_le_0.9": avg(agg["oracle"], "frac_le_0.9"),
        "repeat8x_median_sigma": avg(agg["repeat8x"], "median_sigma"),
        "repeat8x_frac_le_0.285": avg(agg["repeat8x"], "frac_le_0.285"),
        "same_token_by_distance": {},
    }
    for t in args.thresholds:
        summary[f"oracle_frac_le_{float(t):g}"] = avg(agg["oracle"], f"frac_le_{float(t):g}")
        summary[f"repeat8x_frac_le_{float(t):g}"] = avg(agg["repeat8x"], f"frac_le_{float(t):g}")
    for binkey in ("0-8", "8-64", "64-512", "512-100000"):
        meds = [d[binkey]["median_sigma"] for d in agg["same_token"] if binkey in d]
        if meds:
            summary["same_token_by_distance"][binkey] = sum(meds) / len(meds)

    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "raw": agg}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
