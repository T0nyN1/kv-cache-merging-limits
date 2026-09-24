"""The break-even constant on any model: |v_A - v_E| / mean||v||.

experiments/cancel.py measured 0.2914 on Qwen3-1.7B (budget 0.15, top-k anchors
by all-query attention mass, 144 (layer, head) samples), and the paper turns it
into the value-only break-even sigma <= 0.285 for every model. This script
measures the same quantity for any HF causal LM so the constant can be checked
on the families where sigma is lower (Llama-3.1, Mistral), as requested in
review round 1.

Both selectors the paper uses are reported: all-query mass (H2O-style, the
original measurement) and last-32-query window mass (SnapKV-style, what v7
selects on). Also reports the implied break-even sigma = sqrt(ln(1 + gap^2)).

    python experiments/centroid_gap.py --model models/Llama-3.1-8B-Instruct --device mps --dtype bfloat16 \
        --out runs/local_repro/centroid_gap_llama.json
"""
import argparse
import json
import math
import os
import statistics as st
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.data import load_wikitext_docs
from experiments.probe import capture_dense


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--docs", type=int, default=3)
    ap.add_argument("--prefill", type=int, default=3000)
    ap.add_argument("--budget", type=float, default=0.15)
    ap.add_argument("--layer-stride", type=int, default=5)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="eager").to(args.device).eval()
    n_kv = model.config.num_key_value_heads
    docs = load_wikitext_docs(min_tokens=args.prefill, max_docs=args.docs, tokenizer=tok)

    rows = {"full": [], "win": []}
    vnorms = []
    for text in docs:
        ids = torch.tensor([tok.encode(text, add_special_tokens=False)[:args.prefill]], device=args.device)
        cache, sc = capture_dense(model, ids, args.device, n_kv, window=args.window)
        n_layers = len(cache.layers)
        for li in range(0, n_layers, args.layer_stride):
            V = cache.layers[li].values[0].float()                # (H, L, d)
            H, L, d = V.shape
            m = max(2, int(L * args.budget))
            for name in ("full", "win"):
                s = sc[name][li][0].float()                        # (H, L)
                idx = s.topk(m, dim=-1).indices
                for h in range(H):
                    a = idx[h]
                    mask = torch.ones(L, dtype=torch.bool, device=V.device)
                    mask[a] = False
                    wa, we = s[h][a], s[h][mask]
                    vA = (V[h][a] * wa.unsqueeze(-1)).sum(0) / wa.sum().clamp_min(1e-9)
                    vE = (V[h][mask] * we.unsqueeze(-1)).sum(0) / we.sum().clamp_min(1e-9)
                    vnorm = V[h].norm(dim=-1).mean()
                    rows[name].append(float((vA - vE).norm() / vnorm.clamp_min(1e-9)))
                    if name == "full":
                        vnorms.append(float(vnorm))
        del cache, sc

    out = {"model": args.model, "budget": args.budget, "prefill": args.prefill, "docs": len(docs),
           "samples": len(rows["full"])}
    print(f"\n{args.model}: {len(rows['full'])} (layer, head) samples, budget {args.budget}, prefill {args.prefill}")
    for name in ("full", "win"):
        g = rows[name]
        mean, med = st.mean(g), st.median(g)
        out[name] = {"gap_mean": mean, "gap_median": med,
                     "sigma_be_from_mean": math.sqrt(math.log(1 + mean ** 2)),
                     "sigma_be_from_median": math.sqrt(math.log(1 + med ** 2))}
        print(f"  selector={name:<4} |v_A - v_E|/mean||v||: mean {mean:.4f}  median {med:.4f}"
              f"  -> break-even sigma {out[name]['sigma_be_from_mean']:.3f} (mean) / {out[name]['sigma_be_from_median']:.3f} (median)")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
