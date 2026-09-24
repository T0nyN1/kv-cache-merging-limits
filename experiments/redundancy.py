"""Is the KV cache actually redundant in the sense merging needs?

The premise behind token merging is that some tokens are interchangeable, so one
slot can carry both. For attention the precise condition is that their *keys*
agree: a query scores token t by q.k_t, so tokens i and j receive the same
weight for every query iff k_i ~= k_j. If evicted keys sit far from every
retained key, no merging rule -- optimal transport or otherwise -- can recover
their contribution, and eviction is already near the best you can do.

This script measures that directly on a real cache: for each evicted token, how
close is its nearest anchor, and how much attention mass sits on tokens that do
have a close anchor. It also reports the same statistic for value vectors and
for adjacent-position pairs, as reference points.
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.data import load_wikitext_docs
from experiments.probe import capture_dense, split_regions


def stats(sim, weights=None):
    q = torch.tensor([0.5, 0.9, 0.99], device=sim.device)
    out = {
        "mean": sim.mean().item(),
        "p50": torch.quantile(sim.float(), q[0]).item(),
        "p90": torch.quantile(sim.float(), q[1]).item(),
        "p99": torch.quantile(sim.float(), q[2]).item(),
        "frac>0.7": (sim > 0.7).float().mean().item(),
        "frac>0.9": (sim > 0.9).float().mean().item(),
    }
    if weights is not None:
        w = weights / weights.sum().clamp_min(1e-9)
        out["mass>0.7"] = (w * (sim > 0.7).float()).sum().item()
        out["mass>0.9"] = (w * (sim > 0.9).float()).sum().item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--prefill", type=int, default=3584)
    ap.add_argument("--budget", type=float, default=0.15)
    ap.add_argument("--recent", type=float, default=0.1)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--docs", type=int, default=2)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation="eager").to(args.device).eval()
    n_kv = model.config.num_key_value_heads
    docs = load_wikitext_docs(min_tokens=args.prefill + 8, max_docs=args.docs, tokenizer=tok)

    agg = {}
    for text in docs:
        ids = tok.encode(text, add_special_tokens=False)[:args.prefill]
        cache, scores = capture_dense(model, torch.tensor([ids], device=args.device), args.device, n_kv)
        scores = scores["full"]

        total = cache.layers[0].keys.shape[-2]
        budget = int(total * args.budget)
        recent = int(budget * args.recent)
        middle_budget = max(1, budget - args.sink - recent)

        for li in range(len(cache.layers)):
            k, v = cache.layers[li].keys, cache.layers[li].values
            _, (mk, mv, ms), _ = split_regions(k, v, scores[li], args.sink, recent)
            if mk.shape[-2] <= middle_budget:
                continue

            _, aidx = torch.topk(ms, middle_budget, dim=-1)
            mask = torch.ones(ms.shape, device=k.device, dtype=torch.bool)
            mask.scatter_(-1, aidx, False)

            g = aidx.unsqueeze(-1).expand(-1, -1, -1, k.shape[-1])
            ka, va = torch.gather(mk, 2, g), torch.gather(mv, 2, g)
            n = mask.sum(-1)[0, 0].item()
            ke = mk[mask].view(1, n_kv, n, -1)
            ve = mv[mask].view(1, n_kv, n, -1)
            se = ms[mask].view(1, n_kv, n)

            sim_k = torch.matmul(F.normalize(ke.float(), dim=-1),
                                 F.normalize(ka.float(), dim=-1).transpose(-1, -2)).max(-1).values
            sim_v = torch.matmul(F.normalize(ve.float(), dim=-1),
                                 F.normalize(va.float(), dim=-1).transpose(-1, -2)).max(-1).values
            # reference point: how similar is a key to its immediate neighbour?
            nb = F.cosine_similarity(mk[..., :-1, :].float(), mk[..., 1:, :].float(), dim=-1)

            for name, s, w in (("key->anchor", sim_k, se), ("value->anchor", sim_v, se),
                               ("key->neighbour", nb, None)):
                d = stats(s.flatten(), None if w is None else w.flatten())
                bucket = agg.setdefault((li, name), {kk: [] for kk in d})
                for kk, vv in d.items():
                    bucket[kk].append(vv)
        del cache, scores

    print(f"\nmodel={args.model} prefill={args.prefill} budget={args.budget} "
          f"(middle keeps {args.budget:.0%} of the cache)")
    print("=" * 104)
    print(f"{'layer':>5} {'quantity':<15} {'mean':>7} {'p50':>7} {'p90':>7} {'p99':>7} "
          f"{'>0.7':>7} {'>0.9':>7} {'mass>0.7':>9} {'mass>0.9':>9}")
    print("-" * 104)
    n_layers = max(li for li, _ in agg) + 1
    show = sorted({0, 1, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1})
    for li in show:
        for name in ("key->anchor", "value->anchor", "key->neighbour"):
            d = agg.get((li, name))
            if d is None:
                continue
            m = {kk: sum(vv) / len(vv) for kk, vv in d.items()}
            print(f"{li:>5} {name:<15} {m['mean']:>7.3f} {m['p50']:>7.3f} {m['p90']:>7.3f} {m['p99']:>7.3f} "
                  f"{m['frac>0.7']:>7.3f} {m['frac>0.9']:>7.3f} "
                  f"{m.get('mass>0.7', float('nan')):>9.3f} {m.get('mass>0.9', float('nan')):>9.3f}")
        print()

    print("all layers averaged:")
    for name in ("key->anchor", "value->anchor", "key->neighbour"):
        vals = [v for (li, nm), v in agg.items() if nm == name]
        if not vals:
            continue
        m = {kk: sum(sum(d[kk]) / len(d[kk]) for d in vals) / len(vals) for kk in vals[0]}
        print(f"      {name:<15} {m['mean']:>7.3f} {m['p50']:>7.3f} {m['p90']:>7.3f} {m['p99']:>7.3f} "
              f"{m['frac>0.7']:>7.3f} {m['frac>0.9']:>7.3f} "
              f"{m.get('mass>0.7', float('nan')):>9.3f} {m.get('mass>0.9', float('nan')):>9.3f}")
    print("=" * 104)


if __name__ == "__main__":
    main()
