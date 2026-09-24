"""Class-A oracle fitted on the DECODE queries themselves (the anti-causal bound).

experiments/classa_oracle.py fits the unrestricted query-independent value edit
on prefill queries and tests it on later queries; under a deployment-faithful
protocol that removes at most ~25 % of held-out eviction error, and more prefill
calibration makes it worse because prefill queries drift from decode queries.
Review round 1 asked for the asymptote: if the edit is fitted on the actual
decode queries (which no deployable method can see), how much can *any*
query-independent value edit remove?

Protocol per prompt: greedy-decode T tokens, capture the post-RoPE decode
queries, use the prefill cache as the cache (prefix), anchors = top mass of the
last 128 prefill queries (causal), then three fits of delta in R^{m x d}:
  decode-fit   : fit on decode steps [0, T/2), test on [T/2, T)   <- anti-causal upper bound
  prefill-128  : fit on the last 128 prefill queries, test on all T decode steps  (deployment)
  prefill-512  : same with 512 prefill queries
Reported: median over heads of the held-out fraction of squared eviction error removed.

    python experiments/classa_oracle_decode.py --device mps --out runs/local_repro/classa_oracle_decode_qwen3.json
"""
import argparse
import json
import os
import statistics as st
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.classa_oracle import capture_kqv
from experiments.data import load_wikitext_docs, build_niah_samples
from experiments.sigma_decode import decode_queries


def solve(A_fit, E_fit, A_test, E_test, lams, m, device):
    G = A_fit.T @ A_fit
    base_in = float((E_fit ** 2).sum())
    base_out = float((E_test ** 2).sum())
    out = {}
    for lam in lams:
        delta = torch.linalg.solve(G + lam * torch.eye(m, device=device), A_fit.T @ E_fit)
        res_in = float(((A_fit @ delta - E_fit) ** 2).sum())
        res_out = float(((A_test @ delta - E_test) ** 2).sum())
        out[lam] = (1.0 - res_in / max(base_in, 1e-12), 1.0 - res_out / max(base_out, 1e-12))
    return out


_GROUP = {"g": 1}


def group_size_hint(q_dec, T):
    """rows per decode step (= GQA group size), recorded by main()."""
    return max(1, _GROUP["g"])


def head_fits(k, v, q_pre, pos_pre, q_dec, budget, lams, device, anchor_window=128, pre_fit=(128, 512)):
    """k, v: (L, d) prefill cache; q_pre: (Wp, d) prefill probe rows with absolute
    positions pos_pre; q_dec: (T*group, d) decode rows ordered step-major."""
    L, d = k.shape
    sc = d ** -0.5
    T = q_dec.shape[0]

    # anchors from the last `anchor_window` prefill positions, causal attention
    lg_pre = (q_pre @ k.T) * sc
    future = torch.arange(L, device=device)[None, :] > pos_pre[:, None]
    lg_pre = lg_pre.masked_fill(future, float("-inf"))
    at_pre = torch.softmax(lg_pre, dim=-1)
    sel = pos_pre >= (L - anchor_window)
    m = max(2, int(L * budget))
    a_idx = torch.topk(at_pre[sel].sum(0), m).indices.sort().values
    a_mask = torch.zeros(L, dtype=torch.bool, device=device)
    a_mask[a_idx] = True

    def errs(lg, at):
        o_full = at @ v
        ak = torch.softmax(lg[:, a_mask], dim=-1)
        return ak, o_full - ak @ v[a_mask]

    A_pre, E_pre = errs(lg_pre, at_pre)
    lg_dec = (q_dec @ k.T) * sc                       # decode queries see the whole prefix
    A_dec, E_dec = errs(lg_dec, torch.softmax(lg_dec, dim=-1))

    res = {}
    half = T // 2
    res["decode_fit"] = solve(A_dec[:half], E_dec[:half], A_dec[half:], E_dec[half:], lams, m, device)
    # interleaved split (even steps fit, odd steps test): removes the within-decode
    # drift between the two halves, so the number isolates the fit-size question
    steps = torch.arange(T, device=device) // group_size_hint(q_dec, T)
    even = (steps % 2 == 0)
    res["decode_fit_interleaved"] = solve(A_dec[even], E_dec[even], A_dec[~even], E_dec[~even], lams, m, device)
    for W in pre_fit:
        rows = pos_pre >= (L - W)
        res[f"prefill_{W}"] = solve(A_pre[rows], E_pre[rows], A_dec, E_dec, lams, m, device)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--context", type=int, default=2500)
    ap.add_argument("--probe", type=int, default=512, help="prefill probe positions captured (>= max prefill fit)")
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--budget", type=float, default=0.15)
    ap.add_argument("--layer-stride", type=int, default=4)
    ap.add_argument("--prompts", type=int, default=2)
    ap.add_argument("--lams", type=float, nargs="+", default=[1e-3, 1e-1, 1.0])
    ap.add_argument("--no-repeat-ngram", dest="no_repeat_ngram", type=int, default=0)
    ap.add_argument("--top-k", dest="top_k", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="eager").to(args.device).eval()
    group = model.config.num_attention_heads // model.config.num_key_value_heads
    _GROUP["g"] = group

    docs = load_wikitext_docs(min_tokens=args.context + 200, max_docs=args.prompts + 4, tokenizer=tok)
    kinds = {}
    niah = build_niah_samples(docs[args.prompts:], tok, context_tokens=args.context,
                              depths=(0.35, 0.7)[:args.prompts], seed=0)
    kinds["qa_needle"] = [s["prompt"] for s in niah]
    kinds["continuation"] = [tok.decode(tok.encode(d, add_special_tokens=False)[:args.context])
                             for d in docs[:args.prompts]]

    results = {"model": args.model, "context": args.context, "new_tokens": args.new, "budget": args.budget,
               "kinds": {}}
    for kind, prompts in kinds.items():
        agg = {}
        for text in prompts:
            ids = torch.tensor([tok.encode(text, add_special_tokens=False)[-(args.context + 200):]],
                               device=args.device)
            K, Q, V = capture_kqv(model, ids, args.probe)
            Qd, gen = decode_queries(model, ids, args.new, args.layer_stride,
                                     no_repeat_ngram=args.no_repeat_ngram, top_k=args.top_k)
            print(f"  [{kind}] distinct generated tokens: {len(set(gen))}/{len(gen)}")
            L = ids.shape[1]
            for li in range(0, len(K), args.layer_stride):
                if li not in Qd:
                    continue
                for h in range(K[li].shape[1]):
                    k, v = K[li][0, h], V[li][0, h]
                    d = k.shape[-1]
                    q_pre = Q[li][0, h * group:(h + 1) * group].reshape(-1, d)
                    pos_pre = torch.arange(L - args.probe, L, device=args.device).repeat(group)
                    q_dec = Qd[li][:, h * group:(h + 1) * group, :].reshape(-1, d)
                    res = head_fits(k, v, q_pre, pos_pre, q_dec, args.budget, args.lams, args.device)
                    for name, r in res.items():
                        for lam, (ri, ro) in r.items():
                            agg.setdefault(name, {}).setdefault(lam, {"in": [], "out": []})
                            agg[name][lam]["in"].append(ri)
                            agg[name][lam]["out"].append(ro)
            del K, Q, V, Qd
        summary = {}
        print(f"\n[{kind}]  (median over heads; fraction of squared eviction error removed)")
        for name, per in agg.items():
            summary[name] = {}
            line = f"  {name:<12}"
            for lam, io in per.items():
                fi = [x for x in io["in"] if x == x and abs(x) < 1e6]
                fo = [x for x in io["out"] if x == x and abs(x) < 1e6]
                mi, mo = st.median(fi), st.median(fo)
                summary[name][str(lam)] = {"in_median": mi, "out_median": mo, "n": len(fo)}
                line += f"  lam={lam:g}: in {mi:.3f} out {mo:.3f}"
            print(line)
        results["kinds"][kind] = summary
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
