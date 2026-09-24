"""Fast probe: how much does prefill compression change the model's beliefs?

The end-to-end benchmark decodes token by token and costs ~30 s per
(method, document). That is far too slow to search a six-dimensional
hyper-parameter space, and most of that time measures the decode loop rather
than the compression rule.

This probe isolates the compression rule:

1. dense prefill over `L` tokens, capturing per-layer / per-KV-head accumulated
   attention with the same hook the real evaluator uses;
2. compress the resulting (K, V) *offline* with each candidate rule;
3. push a held-out probe block of `P` tokens through the model **in a single
   forward** against each compressed cache, and compare next-token
   distributions against the dense run.

One forward instead of P sequential steps, so a full sweep costs seconds.
What it does not cover is decode-time recompression; use `bench.py` for that.
"""
import argparse
import copy
import itertools
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from experiments.data import load_wikitext_docs


# ---------------------------------------------------------------------------
def capture_dense(model, ids, device, num_kv_heads, window=32):
    """Dense prefill; returns (cache, {"full": scores, "win": scores}).

    "full" sums attention over every query (H2O-style); "win" sums only the last
    `window` queries (SnapKV-style), which is a much better predictor of what
    future queries will attend to.
    """
    scores = {"full": {}, "win": {}}

    def make_hook(layer_idx):
        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple) and len(outputs) > 1 and outputs[1] is not None:
                with torch.no_grad():
                    a = outputs[1].detach().float()
                    for name, sl in (("full", a), ("win", a[..., -window:, :])):
                        w = sl.sum(dim=-2)                          # (b, n_q_heads, k_len)
                        b, qh, kl = w.shape
                        if qh != num_kv_heads and qh % num_kv_heads == 0:
                            w = w.reshape(b, num_kv_heads, qh // num_kv_heads, kl).sum(dim=2)
                        scores[name][layer_idx] = w
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

    handles = [layer.self_attn.register_forward_hook(make_hook(i))
               for i, layer in enumerate(model.model.layers)]
    try:
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True, past_key_values=DynamicCache(), return_dict=True)
    finally:
        for h in handles:
            h.remove()
    return out.past_key_values, scores


def cache_from_kv(kv_list):
    """Wrap a list of (k, v) into a fresh DynamicCache."""
    c = DynamicCache()
    for i, (k, v) in enumerate(kv_list):
        c.update(k, v, i, {})
    return c


def split_regions(k, v, s, sink_size, recent_size):
    n = k.shape[-2]
    end = max(sink_size, n - recent_size)
    return (k[..., :sink_size, :], v[..., :sink_size, :], s[..., :sink_size]), \
           (k[..., sink_size:end, :], v[..., sink_size:end, :], s[..., sink_size:end]), \
           (k[..., end:, :], v[..., end:, :], s[..., end:])


# ---------------------------------------------------------------------------
def compress_all_layers(dense_cache, all_scores, method, budget_frac, recent_frac, sink_size, kwargs):
    """Apply one compression rule to every layer, returning a new cache."""
    from core.ot_kv import otkv_compress
    from core.ot_kv_v1 import otkv_compress_v1
    from core.ot_kv_v3 import otkv_v3_compress

    kwargs = dict(kwargs)
    scores = all_scores[kwargs.pop("score_src", "full")]

    n_layers = len(dense_cache.layers)
    total = dense_cache.layers[0].keys.shape[-2]
    budget = int(total * budget_frac)
    recent_size = int(budget * recent_frac)
    middle_budget = max(1, budget - sink_size - recent_size)

    out = []
    for li in range(n_layers):
        k = dense_cache.layers[li].keys
        v = dense_cache.layers[li].values
        s = scores[li]
        (sk, sv, ss), (mk, mv, ms), (rk, rv, rs) = split_regions(k, v, s, sink_size, recent_size)

        if mk.shape[-2] <= middle_budget:
            out.append((k, v))
            continue

        if method == "evict":
            _, idx = torch.topk(ms, middle_budget, dim=-1)
            idx = idx.sort(dim=-1).values
            g = idx.unsqueeze(-1).expand(-1, -1, -1, k.shape[-1])
            nk, nv = torch.gather(mk, 2, g), torch.gather(mv, 2, g)
        elif method == "otkv_v2":
            nk, nv, _ = otkv_compress(mk, mv, middle_budget, importance_scores=ms, **kwargs)
        elif method == "otkv_v1":
            nk, nv, _ = otkv_compress_v1(mk, mv, middle_budget, importance_scores=ms, **kwargs)
        elif method == "v3":
            frozen = ss.sum(-1, keepdim=True) + rs.sum(-1, keepdim=True)
            nk, nv, _ = otkv_v3_compress(mk, mv, middle_budget, ms, frozen_mass=frozen, **kwargs)
        else:
            raise ValueError(method)

        out.append((torch.cat([sk, nk, rk], dim=-2), torch.cat([sv, nv, rv], dim=-2)))
    return cache_from_kv(out)


# ---------------------------------------------------------------------------
def _null_mask_hook():
    """Compressed caches are shorter than the true history, so the model-built
    causal mask no longer lines up. The probe block is fully visible to itself
    plus the whole (already causal) cache, so the mask can simply be dropped for
    the cache columns and rebuilt causally for the probe columns."""

    def pre_hook(module, args, kwargs):
        hs = args[0] if args else kwargs.get("hidden_states")
        if hs is None:
            return args, kwargs
        q_len = hs.shape[1]
        mask = kwargs.get("attention_mask")
        if mask is None:
            return args, kwargs
        kv_len = mask.shape[-1]
        new = torch.zeros((1, 1, q_len, kv_len), device=hs.device, dtype=mask.dtype)
        if q_len > 1:
            past = kv_len - q_len
            causal = torch.full((q_len, q_len), torch.finfo(mask.dtype).min, device=hs.device, dtype=mask.dtype)
            causal = torch.triu(causal, diagonal=1)
            new[..., past:] = causal
        kwargs["attention_mask"] = new
        return args, kwargs

    return pre_hook


@torch.no_grad()
def probe_logprobs(model, cache, probe_ids, start_pos, device):
    """Next-token log-probs for every probe position, in one forward."""
    handles = [layer.self_attn.register_forward_pre_hook(_null_mask_hook(), with_kwargs=True)
               for layer in model.model.layers]
    try:
        pos = torch.arange(start_pos, start_pos + probe_ids.shape[1], device=device).unsqueeze(0)
        out = model(input_ids=probe_ids, past_key_values=cache, use_cache=True,
                    position_ids=pos, return_dict=True)
    finally:
        for h in handles:
            h.remove()
    return F.log_softmax(out.logits.float(), dim=-1)[0]


def kl_against(ref_lp, lp):
    return torch.sum(ref_lp.exp() * (ref_lp - lp), dim=-1).mean().item()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--docs", type=int, default=4)
    ap.add_argument("--prefill", type=int, default=3584)
    ap.add_argument("--probe", type=int, default=128)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.15])
    ap.add_argument("--recent", type=float, default=0.1)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--configs", type=str, default=None,
                    help="JSON list of explicit v3 kwarg dicts (takes precedence over --grid)")
    ap.add_argument("--grid", type=str, default=None,
                    help="JSON dict of v3 kwarg -> list of values; the cartesian product is swept")
    ap.add_argument("--baselines", nargs="+", default=["evict", "evict:win", "otkv_v1", "otkv_v2"],
                    help="rule[:score_src]; score_src is full (H2O) or win (SnapKV)")
    ap.add_argument("--score_window", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="eager"
    ).to(args.device).eval()
    n_kv = model.config.num_key_value_heads

    need = args.prefill + args.probe + 8
    docs = load_wikitext_docs(min_tokens=need, max_docs=args.docs, tokenizer=tok)
    print(f"[probe] {len(docs)} docs | prefill={args.prefill} probe={args.probe} "
          f"budgets={args.budgets} recent={args.recent} sink={args.sink}")

    if args.configs:
        configs = json.loads(args.configs)
    elif args.grid:
        grid = json.loads(args.grid)
        keys = list(grid)
        configs = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]
    else:
        configs = [{}]

    acc = {}   # (budget, label) -> [kl per doc]
    for di, text in enumerate(docs):
        ids = tok.encode(text, add_special_tokens=False)[:need]
        if getattr(tok, "bos_token_id", None) is not None:
            ids = [tok.bos_token_id] + ids
        prefix = torch.tensor([ids[:args.prefill]], device=args.device)
        probe = torch.tensor([ids[args.prefill:args.prefill + args.probe]], device=args.device)

        t0 = time.time()
        dense_cache, scores = capture_dense(model, prefix, args.device, n_kv, window=args.score_window)
        ref_lp = probe_logprobs(model, copy.deepcopy(dense_cache), probe, args.prefill, args.device)
        print(f"  doc {di}: dense prefill+probe {time.time() - t0:.1f}s")

        for budget in args.budgets:
            runs = []
            for b in args.baselines:
                rule, _, src = b.partition(":")
                runs.append((rule, b, {"score_src": src or "full"}))
            for cfg in configs:
                label = "v3" if not cfg else "v3[" + ",".join(f"{k}={v}" for k, v in cfg.items()) + "]"
                runs.append(("v3", label, cfg))

            for method, label, kw in runs:
                t0 = time.time()
                c = compress_all_layers(dense_cache, scores, method, budget, args.recent, args.sink, kw)
                comp_s = time.time() - t0
                lp = probe_logprobs(model, c, probe, args.prefill, args.device)
                kl = kl_against(ref_lp, lp)
                acc.setdefault((budget, label), {"kl": [], "compress_s": [], "len": 0})
                acc[(budget, label)]["kl"].append(kl)
                acc[(budget, label)]["compress_s"].append(comp_s)
                acc[(budget, label)]["len"] = c.layers[0].keys.shape[-2]
                del c
        del dense_cache, scores, ref_lp

    print("\n" + "=" * 84)
    print(f"{'budget':>7} {'config':<44} {'KL(dense||c)':>13} {'cache':>7} {'compress s':>11}")
    print("-" * 84)
    rows = []
    for (budget, label), d in sorted(acc.items(), key=lambda x: (x[0][0], sum(x[1]["kl"]) / len(x[1]["kl"]))):
        kl = sum(d["kl"]) / len(d["kl"])
        cs = sum(d["compress_s"]) / len(d["compress_s"])
        rows.append({"budget": budget, "config": label, "kl": kl, "cache": d["len"], "compress_s": cs})
        print(f"{budget:>7} {label:<44} {kl:>13.5f} {d['len']:>7} {cs:>11.3f}")
    print("=" * 84)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"args": vars(args), "rows": rows}, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
