"""Local A/B benchmark for KV-cache compression methods.

Runs entirely offline on one GPU/MPS device with a small model so that
algorithmic changes can be measured in minutes instead of hours on Modal.

Metrics
-------
ppl   : teacher-forced perplexity over the decoded half of a long document.
kl    : mean KL(dense || compressed) of the next-token distribution at every
        decoded position. Far more sensitive than ppl -- it measures how much
        the compressed cache changed the model's beliefs, not just whether the
        argmax stayed lucky.
niah  : needle-in-a-haystack accuracy (greedy decode, exact substring match).
speed : prefill / decode wall-clock and the physical cache length actually kept.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from experiments.data import load_wikitext_docs, build_niah_samples


# --------------------------------------------------------------------------
# hooks (mirrors evaluation/models/wrapper.py so the bench exercises the same
# code path the real evaluator uses)
# --------------------------------------------------------------------------
def _attention_hook(cache_obj, layer_idx):
    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple) and len(outputs) > 1:
            attn_weights = outputs[1]
            if attn_weights is None:
                raise RuntimeError("eager attention required: attn_weights is None")
            with torch.no_grad():
                cache_obj.current_attention_scores[layer_idx] = cache_obj.reduce_attention(attn_weights)
            try:
                attn_weights.untyped_storage().resize_(0)
            except RuntimeError:
                pass
            new_outputs = list(outputs)
            new_outputs[1] = None
            return tuple(new_outputs)
        return outputs

    return hook


def _attention_pre_hook(cache_obj=None, layer_idx=None):
    def pre_hook(module, args, kwargs):
        hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states")
        if hidden_states is not None and hidden_states.shape[1] == 1:
            if "attention_mask" in kwargs:
                kwargs["attention_mask"] = None
            if cache_obj is not None and getattr(cache_obj, "consolidate", False):
                cache_obj.pre_attention_step(layer_idx)
                bias = cache_obj.get_decode_bias(layer_idx)
                if bias is not None:
                    kwargs["attention_mask"] = bias.to(hidden_states.dtype)
        return args, kwargs

    return pre_hook


class CacheHarness:
    """Builds a fresh cache + attention hooks per document."""

    def __init__(self, model, cache_class, cache_kwargs):
        self.model = model
        self.cache_class = cache_class
        self.cache_kwargs = cache_kwargs or {}
        self._hooks = []

    def __enter__(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        if self.cache_class is None:
            self.cache = DynamicCache()
            return self.cache

        self.cache = self.cache_class(**self.cache_kwargs)
        needs_attn = getattr(self.cache, "requires_attention", True)
        layers = self.model.model.layers
        if hasattr(self.cache, "num_layers") and getattr(self.cache, "num_layers", None) is None:
            self.cache.num_layers = len(layers)
        for layer_idx, layer in enumerate(layers):
            if needs_attn:
                self._hooks.append(layer.self_attn.register_forward_hook(_attention_hook(self.cache, layer_idx)))
            self._hooks.append(layer.self_attn.register_forward_pre_hook(
                _attention_pre_hook(self.cache, layer_idx), with_kwargs=True))
        return self.cache

    def __exit__(self, *exc):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        if hasattr(self.cache, "current_attention_scores"):
            self.cache.current_attention_scores.clear()
        self.cache = None
        return False


# --------------------------------------------------------------------------
# method registry
# --------------------------------------------------------------------------
def get_method(name, budget, recent, sink, per_head, extra=None):
    extra = dict(extra or {})
    common = dict(compression_size=budget, recent_size=recent, sink_size=sink,
                  per_head=per_head, mode="prefill")

    if name == "dense":
        return None, {}
    if name == "h2o":
        from baselines.h2o import H2OCache
        return H2OCache, common
    if name.startswith("snapkv"):
        from baselines.snapkv import SnapKVCache
        return SnapKVCache, {**common, "observation_window": extra.pop("observation_window", 32), **extra}
    if name == "pyramidkv":
        from baselines.pyramidkv import PyramidKVCache
        return PyramidKVCache, common
    if name == "streamingllm":
        from baselines.streamingllm import StreamingLLMCache
        return StreamingLLMCache, common
    if name == "echokv":
        from baselines.echokv import EchoKVCache
        return EchoKVCache, common
    if name.startswith("otkv_v1"):
        from core.ot_kv_v1 import OTKVv1Cache
        defaults = dict(gamma=1.0, epsilon=0.01, transport_mode="soft",
                        compress_interval=32, target_beta=0.0, sinkhorn_iters=50)
        defaults.update(extra)
        return OTKVv1Cache, {**common, **defaults}
    if name.startswith("otkv7") or name.startswith("v7"):
        from core.ot_kv_v7 import OTKVv7Cache
        return OTKVv7Cache, {**common, **extra}
    if name.startswith("otkv6") or name.startswith("v6"):
        from core.ot_kv_v6 import OTKVv6Cache
        return OTKVv6Cache, {**common, **extra}
    if name.startswith("otkv5") or name.startswith("v5"):
        from core.ot_kv_v5 import OTKVv5Cache
        return OTKVv5Cache, {**common, **extra}
    if name.startswith("otkv4") or name.startswith("v4"):
        from core.ot_kv_v4 import OTKVv4Cache
        return OTKVv4Cache, {**common, **extra}
    if name.startswith("otkv3") or name.startswith("v3"):
        from core.ot_kv_v3 import OTKVv3Cache
        return OTKVv3Cache, {**common, **extra}
    if name.startswith("otkv"):
        from core.ot_kv import OTKVCache
        defaults = dict(gamma=1.0, epsilon=0.01, transport_mode="soft",
                        compress_interval=32, target_beta=0.0, sinkhorn_iters=50)
        defaults.update(extra)
        return OTKVCache, {**common, **defaults}
    raise ValueError(f"unknown method {name}")


# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------
@dataclass
class DocResult:
    nll: float = 0.0
    n: int = 0
    kl: float = 0.0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    cache_len: int = 0
    cache_peak: int = 0
    cache_mean: float = 0.0
    dense_len: int = 0
    ref_logprobs: list = field(default_factory=list)


def _sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def run_ppl_doc(model, tok, text, harness, prefill_frac, max_tokens, device, ref=None, keep_ref=False):
    ids = tok.encode(text, add_special_tokens=False)[:max_tokens]
    if getattr(tok, "bos_token_id", None) is not None and (not ids or ids[0] != tok.bos_token_id):
        ids = [tok.bos_token_id] + ids

    split = max(1, int(len(ids) * prefill_frac))
    prefix = torch.tensor([ids[:split]], device=device)
    target = torch.tensor([ids[split:]], device=device)

    res = DocResult()
    with harness as cache:
        _sync(device)
        t0 = time.time()
        out = model(input_ids=prefix, use_cache=True, past_key_values=cache, return_dict=True)
        _sync(device)
        res.prefill_s = time.time() - t0

        logit = out.logits[:, -1:, :].float()
        del out

        n_steps = target.shape[1]
        _sync(device)
        t0 = time.time()
        for i in range(n_steps):
            lp = F.log_softmax(logit[:, -1, :], dim=-1)
            tgt = target[0, i].item()
            res.nll -= lp[0, tgt].item()
            res.n += 1

            if keep_ref:
                res.ref_logprobs.append(lp[0].to(torch.float16).cpu())
            elif ref is not None:
                p = ref[i].to(device=lp.device, dtype=torch.float32)
                res.kl += torch.sum(p.exp() * (p - lp[0])).item()

            if i == n_steps - 1:
                break
            pos = split + i
            out = model(input_ids=target[:, i:i + 1], use_cache=True, past_key_values=cache,
                        position_ids=torch.tensor([[pos]], device=device), return_dict=True)
            plen = _phys_len(cache)
            res.cache_peak = max(res.cache_peak, plen)
            res.cache_mean += plen
            logit = out.logits.float()
            del out
        _sync(device)
        res.decode_s = time.time() - t0
        res.cache_len = _phys_len(cache)
        res.cache_peak = max(res.cache_peak, res.cache_len)
        res.cache_mean = res.cache_mean / max(1, n_steps - 1) if n_steps > 1 else res.cache_len
        res.dense_len = split + n_steps - 1
    return res


def _phys_len(cache):
    """Mean KV length across layers.

    Reporting layer 0 alone misrepresents any method with a depth-dependent
    budget: PyramidKV gives its first layer 1.5x the flat budget, so a layer-0
    reading suggests it holds 44 % more cache than it does. The per-layer mean is
    what the memory footprint actually scales with.
    """
    if hasattr(cache, "layers") and len(cache.layers):
        lens = [l.keys.shape[-2] for l in cache.layers if getattr(l, "keys", None) is not None]
        if lens:
            return sum(lens) / len(lens)
    return 0


def _digit_prefix_score(answer, text):
    """Partial credit: fraction of the answer's leading digits reproduced.

    At tight budgets models routinely emit the first two or three digits of the
    needle and then drift, so exact match collapses every method to zero and
    hides real differences.
    """
    import re
    m = re.search(r"\d+", text)
    got = m.group(0) if m else ""
    n = 0
    for a, b in zip(answer, got):
        if a != b:
            break
        n += 1
    return n / len(answer)


@torch.no_grad()
def run_niah_sample(model, tok, sample, harness, device, max_new=12):
    ids = tok.encode(sample["prompt"], add_special_tokens=False)
    if getattr(tok, "bos_token_id", None) is not None and (not ids or ids[0] != tok.bos_token_id):
        ids = [tok.bos_token_id] + ids
    inp = torch.tensor([ids], device=device)

    with harness as cache:
        out = model(input_ids=inp, use_cache=True, past_key_values=cache, return_dict=True)
        cur = out.logits[:, -1:, :].argmax(-1)
        gen = [cur.item()]
        for i in range(max_new - 1):
            pos = inp.shape[1] + i
            out = model(input_ids=cur, use_cache=True, past_key_values=cache,
                        position_ids=torch.tensor([[pos]], device=device), return_dict=True)
            cur = out.logits[:, -1:, :].argmax(-1)
            gen.append(cur.item())
        text = tok.decode(gen)
    return sample["answer"] in text, _digit_prefix_score(sample["answer"], text), text


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--methods", nargs="+", default=["dense", "h2o", "otkv"])
    ap.add_argument("--tasks", nargs="+", default=["ppl"], choices=["ppl", "niah"])
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.5])
    ap.add_argument("--recent", type=float, default=0.1)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--per_head", type=lambda x: str(x).lower() == "true", default=True)
    ap.add_argument("--docs", type=int, default=3)
    ap.add_argument("--max_tokens", type=int, default=3072)
    ap.add_argument("--prefill_frac", type=float, default=0.5)
    ap.add_argument("--niah_tokens", type=int, default=3000)
    ap.add_argument("--niah_depths", type=float, nargs="+", default=[0.15, 0.35, 0.55, 0.75, 0.92])
    ap.add_argument("--extra", type=str, default="{}", help="JSON of extra per-method kwargs, keyed by method name")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--quiet", action="store_true", default=True)
    args = ap.parse_args()

    extras = json.loads(args.extra)

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="eager"
    ).to(args.device).eval()

    if args.quiet:
        import builtins
        _real_print = builtins.print

        def quiet_print(*a, **kw):
            if a and isinstance(a[0], str) and a[0].startswith(("[KV Monitor]", "\r[KV Monitor]")):
                return
            _real_print(*a, **kw)

        builtins.print = quiet_print

    rows = []          # flat list of result dicts, one per (budget, method)
    methods = [m for m in args.methods if m != "dense"]

    # ---------------- ppl / kl ----------------
    if "ppl" in args.tasks:
        docs = load_wikitext_docs(min_tokens=args.max_tokens, max_docs=args.docs, tokenizer=tok)
        print(f"[ppl] {len(docs)} docs, max_tokens={args.max_tokens}, prefill_frac={args.prefill_frac}, "
              f"per_head={args.per_head}")
        refs, dense_rs = [], []
        for text in docs:
            h = CacheHarness(model, None, {})
            r = run_ppl_doc(model, tok, text, h, args.prefill_frac, args.max_tokens, args.device, keep_ref=True)
            refs.append(r.ref_logprobs)
            r.ref_logprobs = []
            dense_rs.append(r)
        rows.append(_agg("dense", None, dense_rs))

        for budget in args.budgets:
            for method in methods:
                cls, kw = get_method(method, budget, args.recent, args.sink, args.per_head, extras.get(method))
                per_doc = []
                for di, text in enumerate(docs):
                    h = CacheHarness(model, cls, kw)
                    per_doc.append(run_ppl_doc(model, tok, text, h, args.prefill_frac,
                                               args.max_tokens, args.device, ref=refs[di]))
                row = _agg(method, budget, per_doc)
                rows.append(row)
                print(f"  budget={budget:<5} {method:<14} ppl={row['ppl']:.4f}  KL={row['kl']:.5f}  "
                      f"cache={row['cache']:.0f}  prefill={row['prefill_s']:.2f}s")

    # ---------------- niah ----------------
    niah = {}
    if "niah" in args.tasks:
        hay = load_wikitext_docs(min_tokens=1000, max_docs=60, tokenizer=None)
        samples = build_niah_samples(hay, tok, context_tokens=args.niah_tokens, depths=tuple(args.niah_depths))
        print(f"[niah] {len(samples)} samples, ~{args.niah_tokens} ctx tokens")
        for budget in args.budgets:
            for method in ["dense"] + methods:
                key = (method, None if method == "dense" else budget)
                if key in niah:
                    continue
                cls, kw = get_method(method, budget, args.recent, args.sink, args.per_head, extras.get(method))
                hits, partial, detail = 0, 0.0, []
                for smp in samples:
                    h = CacheHarness(model, cls, kw)
                    ok, frac, text = run_niah_sample(model, tok, smp, h, args.device)
                    hits += int(ok)
                    partial += frac
                    detail.append((smp["depth"], ok, text.strip().replace("\n", " ")[:30]))
                niah[key] = {"acc": hits / len(samples), "partial": partial / len(samples), "detail": detail}
                print(f"  budget={budget if method != 'dense' else '-':<5} {method:<14} "
                      f"niah={hits / len(samples):.2f} partial={partial / len(samples):.3f}")

    # ---------------- report ----------------
    print("\n" + "=" * 100)
    hdr = f"{'budget':>7} {'method':<16} {'ppl':>9} {'KL(dense||c)':>13} {'cache mean/peak':>14} {'prefill s':>10} {'dec ms/tok':>11}"
    if niah:
        hdr += f" {'niah':>6} {'partial':>8}"
    print(hdr)
    print("-" * 100)
    for row in rows:
        key = (row["method"], row["budget"])
        nd = niah.get(key, {})
        n = nd.get("acc")
        line = (f"{(row['budget'] if row['budget'] is not None else '-'):>7} {row['method']:<16} "
                f"{row['ppl']:>9.4f} {row['kl']:>13.5f} "
                f"{row.get('cache_mean', 0):>6.0f}/{row.get('cache_peak', 0):<6.0f} "
                f"{row['prefill_s']:>10.2f} {row['decode_ms']:>11.1f}")
        if niah:
            line += (f" {n if n is not None else float('nan'):>6.2f}"
                     f" {nd.get('partial', float('nan')):>8.3f}")
        print(line)
    print("=" * 100)

    if niah:
        print("\nniah detail (depth:hit:generated)")
        for (method, budget), d in niah.items():
            print(f"  {method}@{budget}: " + " | ".join(f"{dp:.2f}:{'Y' if ok else 'n'}:{t[:18]}" for dp, ok, t in d["detail"]))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        payload = {"args": vars(args), "rows": rows,
                   "niah": {f"{m}@{b}": v for (m, b), v in niah.items()}}
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.out}")


def _agg(method, budget, docs_r):
    nll = sum(d.nll for d in docs_r) / max(1, sum(d.n for d in docs_r))
    kl = sum(d.kl for d in docs_r) / max(1, sum(d.n for d in docs_r))
    per_doc_kl = [d.kl / max(1, d.n) for d in docs_r]
    per_doc_nll = [d.nll / max(1, d.n) for d in docs_r]
    clen = sum(d.cache_len for d in docs_r) / len(docs_r)
    dlen = sum(d.dense_len for d in docs_r) / len(docs_r)
    return {
        "method": method, "budget": budget,
        "ppl": float(torch.tensor(nll).exp()), "nll": nll, "kl": kl,
        "cache": clen, "cache_frac": clen / max(1, dlen),
        "cache_peak": sum(d.cache_peak for d in docs_r) / len(docs_r),
        "cache_mean": sum(d.cache_mean for d in docs_r) / len(docs_r),
        "prefill_s": sum(d.prefill_s for d in docs_r) / len(docs_r),
        "decode_ms": sum(d.decode_s for d in docs_r) / max(1, sum(d.n for d in docs_r)) * 1000,
        # per-document values: the unit of analysis for a paired comparison.
        # Positions inside one document are heavily autocorrelated, so treating
        # them as independent samples would badly overstate significance.
        "per_doc_kl": per_doc_kl,
        "per_doc_nll": per_doc_nll,
    }


if __name__ == "__main__":
    main()
