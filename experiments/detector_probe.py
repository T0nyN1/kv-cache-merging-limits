"""Measure candidate regime-detector statistics on real prompts.

The v3/v4/v7 detector uses the ratio of normalised entropies H(win)/H(full),
threshold 0.73 (Qwen-calibrated). On Llama-3.1-8B the needle-prompt and
wikitext-continuation distributions overlap: 0.73 mislabels needle layers as
continuation (NIAH 0.76), 0.9 labels wikitext as qa (ppl 4.04 -> 4.42). This
probe collects, per layer, on three prompt classes:

  cont     : plain wikitext continuation (regime should be "continuation")
  needle   : wikitext + needle + trailing question (should be "qa")
  summary  : wikitext + trailing summarisation instruction (empirically wants
             "continuation": that regime wins gov_report/multi_news by +2.5)

statistics:
  h_ratio  : H(win)/H(full), the current detector
  pbar     : attention-weighted mean relative position of the window stream,
             sink excluded ("do the last queries look back, or stay local?")
  sharpfar : (1 - H(win)) * (1 - pbar) — sharp AND far ⇒ retrieval question

Run on any HF model:  python experiments/detector_probe.py --model <id> --device mps
"""
import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer

NEEDLE = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
QUESTION = "\n\nQuestion: What is the best thing to do in San Francisco?\nAnswer:"
SUMMARY = "\n\nSummarize the passage above in three sentences.\nSummary:"


def norm_entropy(p: torch.Tensor) -> torch.Tensor:
    q = p.clamp_min(1e-9)
    q = q / q.sum(-1, keepdim=True)
    h = -(q * q.log()).sum(-1)
    return h / math.log(max(2, p.shape[-1]))


def collect(model, tok, text, device, window=32, sink=4):
    ids = tok(text, return_tensors="pt").input_ids[:, :3000].to(device)
    stats = {}
    handles = []

    def make_hook(li):
        def hook(module, inputs, outputs):
            a = outputs[1]          # (b, qh, q, k)
            if a is None:
                return outputs
            a = a.detach().float()
            full = a.sum(dim=-2)                     # (b, qh, k)
            win = a[..., -window:, :].sum(dim=-2)
            # sink excluded from both statistics
            fullm = full[..., sink:]
            winm = win[..., sink:]
            L = winm.shape[-1]
            pos = torch.arange(L, device=a.device, dtype=torch.float32) / max(1, L - 1)
            pbar = (winm * pos).sum(-1) / winm.sum(-1).clamp_min(1e-9)   # (b, qh)
            h_win = norm_entropy(winm)
            h_full = norm_entropy(fullm)
            stats[li] = dict(
                h_ratio=(h_win.mean() / h_full.mean().clamp_min(1e-9)).item(),
                pbar=pbar.mean().item(),
                h_win=h_win.mean().item(),
            )
            new_out = list(outputs)
            new_out[1] = None
            return tuple(new_out)
        return hook

    for li, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_hook(make_hook(li)))
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Llama-3.1-8B-Instruct")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--depths", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype),
        attn_implementation="eager").to(args.device).eval()

    from experiments.data import load_wikitext_docs
    doc = load_wikitext_docs(min_tokens=2500, max_docs=1)[0]
    base_ids = tok(doc, add_special_tokens=False).input_ids[:2600]
    base = tok.decode(base_ids)

    prompts = {"cont": base}
    for d in args.depths:
        cut = int(len(base) * d)
        prompts[f"needle@{d}"] = base[:cut] + "\n" + NEEDLE + "\n" + base[cut:] + QUESTION
    prompts["summary"] = base + SUMMARY

    print(f"model={args.model}")
    print(f"{'prompt':<12} {'h_ratio p25/med/p75':>24} {'pbar p25/med/p75':>24} {'h_win med':>10}")
    for name, text in prompts.items():
        stats = collect(model, tok, text, args.device)
        for key in ("h_ratio", "pbar", "h_win"):
            vals = sorted(s[key] for s in stats.values())
            n = len(vals)
            q = lambda f: vals[int(f * (n - 1))]
            if key == "h_ratio":
                line = f"{name:<12} {q(.25):>7.3f}/{q(.5):>6.3f}/{q(.75):>6.3f}"
            elif key == "pbar":
                line += f"  {q(.25):>7.3f}/{q(.5):>6.3f}/{q(.75):>6.3f}"
            else:
                line += f"  {q(.5):>8.3f}"
        print(line)


if __name__ == "__main__":
    main()
