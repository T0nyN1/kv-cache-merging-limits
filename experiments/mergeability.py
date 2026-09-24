"""How interchangeable are an evicted token and its nearest anchor, really?

`redundancy.py` shows evicted keys sit at cosine ~0.85 from their nearest
anchor, which sounds close. But attention weights are exponentials of dot
products: with ||k|| ~ 11 and head_dim 128, a cosine gap of 0.1 is worth roughly
||q|| ||k|| sqrt(2 - 2 cos) / sqrt(d) ~ 5 nats of logit, i.e. a factor of ~100 in
attention weight. Cosine similarity is the wrong yardstick.

The quantity that actually decides whether token i can be carried by anchor j is
the log-ratio of the attention weights they receive from a real query,

    gap = | log a_i(q) - log a_j(q) |,

measured over held-out queries. gap << 1 means the two are interchangeable and
merging is near-lossless. gap >> 1 means the anchor receives a wildly different
weight, so whatever value mass we route into it is applied with the wrong
coefficient -- noise, not recovery.

This script measures that distribution directly from the model's own attention
weights, for anchors picked by cosine and by L2, and reports how much of the
evicted attention mass sits in the mergeable regime.
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from experiments.data import load_wikitext_docs


def capture(model, ids, num_kv_heads, window):
    """Dense prefill keeping (a) summed scores and (b) the raw attention rows of
    the last `window` queries, which act as held-out-ish probe queries."""
    summed, rows = {}, {}

    def make_hook(li):
        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple) and len(outputs) > 1 and outputs[1] is not None:
                with torch.no_grad():
                    a = outputs[1].detach().float()
                    w = a.sum(dim=-2)
                    b, qh, kl = w.shape
                    grp = qh // num_kv_heads
                    summed[li] = w.reshape(b, num_kv_heads, grp, kl).sum(dim=2)
                    # keep per-query rows, folded to KV heads by averaging the
                    # query heads that share a KV head
                    rows[li] = a[..., -window:, :].reshape(b, num_kv_heads, grp, window, kl).mean(dim=2)
                    del a
                try:
                    outputs[1].untyped_storage().resize_(0)
                except RuntimeError:
                    pass
                o = list(outputs)
                o[1] = None
                return tuple(o)
            return outputs

        return hook

    handles = [l.self_attn.register_forward_hook(make_hook(i)) for i, l in enumerate(model.model.layers)]
    try:
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True, past_key_values=DynamicCache(), return_dict=True)
    finally:
        for h in handles:
            h.remove()
    return out.past_key_values, summed, rows


def select_anchors(rule, k, s, budget):
    """Pick `budget` anchor positions per head. k: (1,H,L,D) float, s: (1,H,L)."""
    b, h, L, d = k.shape
    if rule == "topk":
        idx = torch.topk(s, budget, dim=-1).indices
    elif rule == "random":
        idx = torch.stack([torch.stack([torch.randperm(L, device=k.device)[:budget] for _ in range(h)])
                           for _ in range(b)])
    elif rule == "mass_random":
        idx = torch.multinomial(s.reshape(b * h, L).clamp_min(1e-9), budget, replacement=False).view(b, h, budget)
    elif rule.startswith("kmeans"):
        # mass-weighted k-means on keys, then take the real token nearest each
        # centroid (a medoid): a Wasserstein quantisation of the key cloud.
        half = int(rule.split(":")[1]) if ":" in rule else 0
        n_heavy = int(budget * half / 100)
        heavy = torch.topk(s, n_heavy, dim=-1).indices if n_heavy else None
        n_cl = budget - n_heavy
        c = k[:, :, torch.randperm(L, device=k.device)[:n_cl], :].clone()
        w = s.unsqueeze(-1)
        for _ in range(5):
            assign = torch.cdist(k, c).argmin(-1)                       # (b,h,L)
            oh = F.one_hot(assign, n_cl).to(k.dtype) * w                # (b,h,L,n_cl)
            num = torch.einsum("bhlc,bhld->bhcd", oh, k)
            den = oh.sum(-2).unsqueeze(-1).clamp_min(1e-6)
            c = num / den
        idx = torch.cdist(c, k).argmin(-1)                              # medoid per centroid
        if heavy is not None:
            idx = torch.cat([heavy, idx], dim=-1)
        # de-duplicate by keeping the first occurrence, topping up with heavy hitters
        idx = _dedup_topup(idx, s, budget)
    else:
        raise ValueError(rule)
    return idx.sort(dim=-1).values


def _dedup_topup(idx, s, budget):
    b, h, _ = idx.shape
    L = s.shape[-1]
    out = torch.zeros((b, h, budget), dtype=torch.long, device=idx.device)
    order = torch.argsort(s, dim=-1, descending=True)
    for bi in range(b):
        for hi in range(h):
            seen, keep = set(), []
            for t in idx[bi, hi].tolist():
                if t not in seen:
                    seen.add(t)
                    keep.append(t)
                if len(keep) == budget:
                    break
            for t in order[bi, hi].tolist():
                if len(keep) >= budget:
                    break
                if t not in seen:
                    seen.add(t)
                    keep.append(t)
            out[bi, hi] = torch.tensor(keep[:budget], device=idx.device)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--prefill", type=int, default=3072)
    ap.add_argument("--budget", type=float, default=0.15)
    ap.add_argument("--recent", type=float, default=0.1)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--docs", type=int, default=2)
    ap.add_argument("--rules", nargs="+", default=["topk", "random", "mass_random", "kmeans:0", "kmeans:50"])
    ap.add_argument("--layers", type=int, default=0, help="only probe every Nth layer (0 = all)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation="eager").to(args.device).eval()
    n_kv = model.config.num_key_value_heads
    docs = load_wikitext_docs(min_tokens=args.prefill + 8, max_docs=args.docs, tokenizer=tok)

    buckets = {}
    for text in docs:
        ids = tok.encode(text, add_special_tokens=False)[:args.prefill]
        cache, summed, rows = capture(model, torch.tensor([ids], device=args.device), n_kv, args.window)

        total = cache.layers[0].keys.shape[-2]
        budget = int(total * args.budget)
        recent = int(budget * args.recent)
        m_budget = max(1, budget - args.sink - recent)
        m_end = total - recent

        step = max(1, args.layers)
        for li in range(0, len(cache.layers), step):
            k = cache.layers[li].keys[..., args.sink:m_end, :]
            a = rows[li][..., args.sink:m_end]                  # (1, n_kv, W, Lm)
            grp = model.config.num_attention_heads // n_kv
            # remove the probe window's own contribution so the score used for
            # selection and for the merge coefficient is genuinely held out
            s = summed[li][..., args.sink:m_end] - a.sum(dim=-2) * grp
            s = s.clamp_min(1e-9)
            if k.shape[-2] <= m_budget:
                continue

            for rule in args.rules:
                aidx = select_anchors(rule, k.float(), s, m_budget)
                mask = torch.ones(s.shape, device=k.device, dtype=torch.bool)
                mask.scatter_(-1, aidx, False)
                n = int(mask.sum(-1)[0, 0])

                g = aidx.unsqueeze(-1).expand(-1, -1, -1, k.shape[-1])
                ka = torch.gather(k, 2, g).float()
                ke = k[mask].view(1, n_kv, n, -1).float()
                se = s[mask].view(1, n_kv, n)

                a_anchor = torch.gather(a, -1, aidx.unsqueeze(-2).expand(-1, -1, args.window, -1))
                a_evict = a.transpose(-1, -2)[mask].view(1, n_kv, n, args.window).transpose(-1, -2)

                bi = torch.cdist(ke, ka).argmin(-1)
                name = rule
                mass_kept = (torch.gather(s, -1, aidx).sum() / s.sum()).item()

                # log-attention gap between each evicted token and its chosen anchor
                sel = torch.gather(a_anchor, -1, bi.unsqueeze(-2).expand(-1, -1, args.window, -1))
                lg_e = torch.log(a_evict.clamp_min(1e-12))
                lg_a = torch.log(sel.clamp_min(1e-12))
                gap = (lg_e - lg_a).abs().mean(dim=-2).flatten()          # mean over probe queries
                w = se.flatten()
                w = w / w.sum().clamp_min(1e-9)
                b = buckets.setdefault(name, {"gap": [], "m0.5": [], "m1": [], "m2": [], "frac1": [], "kept": []})
                b["kept"].append(mass_kept)
                b["gap"].append(gap.median().item())
                b["m0.5"].append((w * (gap < 0.5).float()).sum().item())
                b["m1"].append((w * (gap < 1.0).float()).sum().item())
                b["m2"].append((w * (gap < 2.0).float()).sum().item())
                b["frac1"].append((gap < 1.0).float().mean().item())

                # Does the *historical* attention ratio predict the future one?
                # That ratio is exactly the coefficient the merge applies, so the
                # residual below is the error merging actually commits.
                s_a = torch.gather(s, -1, aidx)
                ratio_hist = torch.log(se.clamp_min(1e-9)) - torch.log(
                    torch.gather(s_a, -1, bi).clamp_min(1e-9))
                resid = ((lg_e - lg_a).mean(dim=-2) - ratio_hist).abs().flatten()
                b.setdefault("resid", []).append(resid.median().item())
                b.setdefault("resid_m1", []).append((w * (resid < 1.0).float()).sum().item())
        del cache, summed, rows

    print(f"\nmodel={args.model} prefill={args.prefill} budget={args.budget} window={args.window}")
    print("gap = |log a_evicted - log a_anchor| over held-out queries; < 1 nat means interchangeable\n")
    print(f"{'anchor rule':<14} {'median gap':>11} {'frac gap<1':>11} {'mass gap<0.5':>13} "
          f"{'mass gap<1':>11} {'mass gap<2':>11} {'attn kept':>10} {'resid':>7} {'mass res<1':>11}")
    print("-" * 108)
    for name, b in buckets.items():
        f = lambda kk: sum(b[kk]) / len(b[kk])
        print(f"{name:<14} {f('gap'):>11.2f} {f('frac1'):>11.3f} {f('m0.5'):>13.3f} "
              f"{f('m1'):>11.3f} {f('m2'):>11.3f} {f('kept'):>10.3f} {f('resid'):>7.2f} {f('resid_m1'):>11.3f}")
    print()


if __name__ == "__main__":
    main()
