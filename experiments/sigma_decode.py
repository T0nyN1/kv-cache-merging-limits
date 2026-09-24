"""Is sigma the same under DECODE-TIME queries as under the prefill probe?

Every sigma number in the paper is measured with a probe of the final 128
prefill positions. The paper's own explanation of why compensated merging
fails is that decode-time queries drift away from those calibration queries.
Two things could then be true and neither has been measured:

  1. the spread of q.Delta/sqrt(d) over the 64 decode queries of one answer
     could be SMALLER than over 128 diverse prefill positions (lower sigma:
     merging with decode-fitted coefficients would be more viable than the
     prefill probe suggests), or larger;
  2. the MEAN of q.Delta/sqrt(d) could shift between prefill and decode
     (mu-drift), which a prefill-fitted constant coefficient cannot absorb
     even when sigma is small.

This script measures both on a real model, for three prompt kinds (a
question-at-the-end retrieval prompt, plain continuation, and a summarisation
instruction), with the same oracle-routing construction as sigma_families.py:
evenly spaced anchors at a 15 % budget, 192 sampled evicted tokens per head,
post-RoPE keys and queries, every `layer_stride`-th layer, all KV heads.

Reported per prompt kind (medians over heads and sampled tokens):
  sigma_prefill          oracle sigma under the prefill probe (paper's quantity)
  sigma_decode_oracle    oracle sigma under the decode queries (best anchor re-chosen)
  sigma_decode_at_pre    decode sigma at the anchor the prefill probe would choose
  mu_shift_over_sigma    |mean_decode - mean_prefill| / sigma_prefill at that anchor
  frac_le_0.285_*        fraction of sampled evicted tokens under value-only break-even

    python experiments/sigma_decode.py --model Qwen/Qwen3-1.7B --device mps --out runs/local_repro/sigma_decode_qwen3.json
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from experiments.data import load_wikitext_docs, build_niah_samples
from experiments.sigma_families import capture_kq

BREAK_EVEN = 0.285


def decode_queries(model, ids, n_new, layer_stride, no_repeat_ngram=0, top_k=0, seed=0):
    """Decode n_new tokens (greedy by default; `no_repeat_ngram` blocks repeated
    n-grams and `top_k` > 0 samples, to avoid the degenerate greedy loops that
    review round 3 found in two of six prompts); return {layer: (T, q_heads, d)}
    post-RoPE decode queries for every layer_stride-th layer, plus the ids."""
    qs = {}
    gen_t = torch.Generator(device="cpu").manual_seed(seed)

    def pre_hook(li):
        def hook(module, args, kwargs):
            hs = args[0] if args else kwargs.get("hidden_states")
            pos = kwargs.get("position_embeddings")
            if hs is None or pos is None or hs.shape[1] != 1:
                return args, kwargs
            with torch.no_grad():
                head_dim = getattr(module, "head_dim", None) or \
                    module.q_proj.out_features // module.config.num_attention_heads
                q = module.q_proj(hs).view(*hs.shape[:-1], -1, head_dim)
                if hasattr(module, "q_norm"):
                    q = module.q_norm(q)
                q = q.transpose(1, 2)
                cos, sin = pos
                q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos, sin)
                qs.setdefault(li, []).append(q.detach().float()[0, :, 0, :])
            return args, kwargs
        return hook

    handles = [layer.self_attn.register_forward_pre_hook(pre_hook(i), with_kwargs=True)
               for i, layer in enumerate(model.model.layers) if i % layer_stride == 0]
    gen = []

    def pick(logits):
        lg = logits.float().clone()
        if no_repeat_ngram > 0 and len(gen) >= no_repeat_ngram - 1:
            n = no_repeat_ngram
            hist = ids[0].tolist() + gen
            prefix = tuple(hist[-(n - 1):]) if n > 1 else tuple()
            for i in range(len(hist) - n + 1):
                if tuple(hist[i:i + n - 1]) == prefix:
                    lg[hist[i + n - 1]] = float("-inf")
        if top_k > 0:
            v, ix = torch.topk(lg, top_k)
            p = torch.softmax(v, -1).cpu()
            return ix[torch.multinomial(p, 1, generator=gen_t).item()].view(1, 1)
        return lg.argmax().view(1, 1)

    try:
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True, past_key_values=DynamicCache())
            nxt = pick(out.logits[0, -1])
            for _ in range(n_new):
                out = model(input_ids=nxt, use_cache=True, past_key_values=out.past_key_values)
                gen.append(int(nxt))
                nxt = pick(out.logits[0, -1])
    finally:
        for h in handles:
            h.remove()
    return {li: torch.stack(v, 0) for li, v in qs.items()}, gen


def analyse(model, keys, q_pre, q_dec, budget, layer_stride, n_sample=192):
    group = model.config.num_attention_heads // model.config.num_key_value_heads
    rows = []
    for li in range(0, len(keys), layer_stride):
        if li not in q_dec:
            continue
        K = keys[li][0]                                        # (kv_h, L, d)
        Qp = q_pre[li][0]                                      # (q_h, P, d)
        Qd = q_dec[li]                                         # (T, q_h, d)
        for h in range(K.shape[0]):
            k = K[h]
            L, d = k.shape
            sc = d ** -0.5
            qp = Qp[h * group:(h + 1) * group].reshape(-1, d)                   # (group*P, d)
            qd = Qd[:, h * group:(h + 1) * group, :].reshape(-1, d)             # (T*group, d)
            m = max(2, int(L * budget))
            a = torch.linspace(0, L - 1, m, device=k.device).long()
            mask = torch.ones(L, dtype=torch.bool, device=k.device)
            mask[a] = False
            e = torch.nonzero(mask).flatten()
            e = e[torch.randperm(len(e), device=k.device)[:n_sample]]
            D = k[e].unsqueeze(1) - k[a].unsqueeze(0)                          # (n, m, d)
            gp = torch.einsum("nmd,pd->nmp", D, qp) * sc                       # (n, m, P')
            gd = torch.einsum("nmd,td->nmt", D, qd) * sc                       # (n, m, T')
            s_pre = gp.std(-1)                                                 # (n, m)
            s_dec = gd.std(-1)
            j_pre = s_pre.argmin(1)                                            # anchor chosen by prefill probe
            sp = s_pre.gather(1, j_pre[:, None]).squeeze(1)
            sd_at = s_dec.gather(1, j_pre[:, None]).squeeze(1)
            sd_or = s_dec.min(1).values
            mu_p = gp.mean(-1).gather(1, j_pre[:, None]).squeeze(1)
            mu_d = gd.mean(-1).gather(1, j_pre[:, None]).squeeze(1)
            shift = (mu_d - mu_p).abs() / sp.clamp_min(1e-6)
            rows.append(torch.stack([sp, sd_at, sd_or, shift], 1).cpu())
    R = torch.cat(rows, 0)
    return {
        "n_tokens": int(R.shape[0]),
        "sigma_prefill_median": float(R[:, 0].median()),
        "sigma_decode_at_prefill_anchor_median": float(R[:, 1].median()),
        "sigma_decode_oracle_median": float(R[:, 2].median()),
        "mu_shift_over_sigma_median": float(R[:, 3].median()),
        "mu_shift_over_sigma_p90": float(R[:, 3].quantile(0.9)),
        "frac_le_0.285_prefill": float((R[:, 0] <= BREAK_EVEN).float().mean()),
        "frac_le_0.285_decode_oracle": float((R[:, 2] <= BREAK_EVEN).float().mean()),
        "frac_decode_sigma_smaller_than_prefill": float((R[:, 1] < R[:, 0]).float().mean()),
        "ratio_decode_over_prefill_median": float((R[:, 1] / R[:, 0].clamp_min(1e-6)).median()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--context", type=int, default=2500)
    ap.add_argument("--probe", type=int, default=128)
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--budget", type=float, default=0.15)
    ap.add_argument("--layer-stride", type=int, default=4)
    ap.add_argument("--prompts", type=int, default=2, help="prompts per kind")
    ap.add_argument("--no-repeat-ngram", dest="no_repeat_ngram", type=int, default=0,
                    help="block repeated n-grams during decoding (0 = plain greedy)")
    ap.add_argument("--top-k", dest="top_k", type=int, default=0, help="sample from the top-k (0 = greedy)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="eager").to(args.device).eval()

    docs = load_wikitext_docs(min_tokens=args.context + 200, max_docs=args.prompts + 4, tokenizer=tok)
    kinds = {}
    niah = build_niah_samples(docs[args.prompts:], tok, context_tokens=args.context,
                              depths=(0.35, 0.7)[:args.prompts], seed=0)
    kinds["qa_needle"] = [s["prompt"] for s in niah]
    kinds["continuation"] = [tok.decode(tok.encode(d, add_special_tokens=False)[:args.context])
                             for d in docs[:args.prompts]]
    kinds["summary_instruction"] = [t + "\n\nSummarize the passage above in three sentences.\nSummary:"
                                    for t in kinds["continuation"]]

    results = {"model": args.model, "context": args.context, "probe": args.probe, "new_tokens": args.new,
               "budget": args.budget, "kinds": {}}
    for kind, prompts in kinds.items():
        agg = []
        for text in prompts:
            ids = torch.tensor([tok.encode(text, add_special_tokens=False)[-(args.context + 200):]],
                               device=args.device)
            keys, q_pre = capture_kq(model, ids, args.probe)
            q_dec, gen = decode_queries(model, ids, args.new, args.layer_stride,
                                        no_repeat_ngram=args.no_repeat_ngram, top_k=args.top_k)
            r = analyse(model, keys, q_pre, q_dec, args.budget, args.layer_stride)
            r["distinct_generated_tokens"] = len(set(gen))
            r["generated"] = tok.decode(gen)[:120]
            agg.append(r)
            del keys, q_pre, q_dec
        results["kinds"][kind] = agg
        print(f"\n[{kind}]")
        for r in agg:
            print(f"  sigma prefill {r['sigma_prefill_median']:.3f} | decode@prefill-anchor {r['sigma_decode_at_prefill_anchor_median']:.3f}"
                  f" | decode oracle {r['sigma_decode_oracle_median']:.3f} | |dmu|/sigma med {r['mu_shift_over_sigma_median']:.2f}"
                  f" p90 {r['mu_shift_over_sigma_p90']:.2f} | frac<=0.285 pre {100*r['frac_le_0.285_prefill']:.2f}%"
                  f" dec-oracle {100*r['frac_le_0.285_decode_oracle']:.2f}% | decode sigma smaller in {100*r['frac_decode_sigma_smaller_than_prefill']:.0f}% of tokens")
            print(f"    gen: {r['generated']!r}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
