"""The jointly optimal class-A merge, measured.

Reviewer objection: Proposition 1 bounds a single evicted token routed to a
single anchor with a scalar coefficient, but a general class-A operator edits
every retained value with an arbitrary vector, and joint correlations across
values could in principle cancel error the pair analysis cannot see.

This script answers empirically, with the *unrestricted* class-A oracle. For
each head, the merge that minimises the true squared output error over a set
of real fit queries is a linear least-squares problem: with renormalised
retained attention A-hat in R^{W x m} and eviction errors E in R^{W x d},

    delta* = argmin_delta || A-hat delta - E ||_F^2 + lam ||delta||_F^2

is the best possible query-independent value edit -- any published class-A
method (CaM, D2O, KVMerger, any transport plan) realises a special case of it.
We fit on the first half of the probe queries and report the fraction of
squared eviction error removed on the held-out second half (temporal split,
mimicking prefill-calibration -> decode drift), plus in-sample removal and a
diagnostic of the Gaussian logit-gap model.

    python experiments/classa_oracle.py --model Qwen/Qwen3-1.7B --device mps
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

from experiments.data import load_wikitext_docs


def capture_kqv(model, ids, n_probe):
    """Post-RoPE keys, probe queries, and values for every layer."""
    keys, queries, values = {}, {}, {}

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
                v = module.v_proj(hs).view(shape)
                if hasattr(module, "q_norm"):
                    q = module.q_norm(q)
                if hasattr(module, "k_norm"):
                    k = module.k_norm(k)
                q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                cos, sin = pos
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                keys[layer_idx] = k.detach().float()
                queries[layer_idx] = q.detach().float()[:, :, -n_probe:, :]
                values[layer_idx] = v.detach().float()
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
    return keys, queries, values


def head_solve(k, v, q, budget, lams, device, positions=None, split="head", causal=False,
               prefix_cache=False, test_len=0, anchor_window=0):
    """Residual fraction of the jointly-optimal class-A merge for one head.

    k, v : (L, d) post-RoPE keys / values
    q    : (W, d) probe queries. Rows are laid out head-major by the caller:
           for a GQA group of g query heads and P probe positions, row r = g_i*P + p.
    positions : (W,) absolute position of each query row (needed for the
           temporal split and the causal mask); None = rows assumed ordered in time.
    split : "head"     -- the original behaviour: first half of the rows vs second
                          half. With a GQA group of 2 this is query-head 0 vs
                          query-head 1 at IDENTICAL positions, i.e. a cross-head
                          split, not a temporal one (audit finding 2026-09-03).
            "temporal" -- fit on the earlier half of the probe positions (all
                          query heads), test on the later half: the split the
                          docstring above describes.
    causal : mask keys later than the query position (the model's real attention);
           False reproduces the original numbers, where probe queries attend to
           every cached key including later ones.
    prefix_cache : (temporal split only) restrict the cache to the keys that
           exist at the end of the fit window, i.e. positions < cut. The test
           queries then read exactly the cache the fit queries calibrated --
           the deployment situation (prefill cache compressed once, decode
           queries read it) -- instead of also attending to their own later
           keys, which no anchor set can represent and which would deflate the
           held-out number for a reason unrelated to merging.
    Returns dict of {lam: (in-sample removal, out-of-sample removal)} plus
    logit-gap Gaussianity diagnostics.
    """
    L, d = k.shape
    sc = d ** -0.5
    W = q.shape[0]
    if positions is None:
        positions = torch.arange(L - W, L, device=device)
    if split == "temporal":
        # test window = the last `test_len` positions (a decode-length horizon)
        # when given, else the later half of the probe; fit = everything before
        cut = L - test_len if test_len > 0 else L - (L - int(positions.min())) // 2
        fit_rows = positions < cut
        if prefix_cache:
            k, v = k[:cut], v[:cut]
            L = cut
    else:
        fit_rows = torch.zeros(W, dtype=torch.bool, device=device)
        fit_rows[: W // 2] = True
    q_fit, q_test = q[fit_rows], q[~fit_rows]

    logits = (q @ k.T) * sc                                   # (W, L)
    if causal:
        future = torch.arange(L, device=device)[None, :] > positions[:, None]
        logits = logits.masked_fill(future, float("-inf"))
    attn = torch.softmax(logits, dim=-1)
    o_full = attn @ v                                         # (W, d)

    # anchors: top-budget by mass accumulated over the *fit* queries -- or, with
    # anchor_window > 0, over only the last `anchor_window` fit positions
    # (SnapKV-style), so that the anchor set does not change with the size of
    # the calibration set. Without this, a large causal calibration window
    # elects its own recent positions as anchors (2026-09-03 finding).
    m = max(2, int(L * budget))
    if anchor_window > 0 and split == "temporal":
        cut_pos = int(positions[~fit_rows].min()) if (~fit_rows).any() else int(positions.max()) + 1
        sel_rows = fit_rows & (positions >= cut_pos - anchor_window)
        mass = attn[sel_rows].sum(0)
    else:
        mass = attn[fit_rows].sum(0)
    a_idx = torch.topk(mass, m).indices.sort().values
    a_mask = torch.zeros(L, dtype=torch.bool, device=device)
    a_mask[a_idx] = True

    attn_kept = torch.softmax(logits[:, a_mask], dim=-1)      # renormalised (W, m)
    o_evict = attn_kept @ v[a_mask]
    err = o_full - o_evict                                    # (W, d)

    A_fit, A_test = attn_kept[fit_rows], attn_kept[~fit_rows]
    E_fit, E_test = err[fit_rows], err[~fit_rows]
    base_in = float((E_fit ** 2).sum())
    base_out = float((E_test ** 2).sum())

    out = {}
    G = A_fit.T @ A_fit
    for lam in lams:
        delta = torch.linalg.solve(G + lam * torch.eye(m, device=device),
                                   A_fit.T @ E_fit)           # (m, d)
        res_in = float(((A_fit @ delta - E_fit) ** 2).sum())
        res_out = float(((A_test @ delta - E_test) ** 2).sum())
        out[lam] = (1.0 - res_in / max(base_in, 1e-12),
                    1.0 - res_out / max(base_out, 1e-12))

    # Gaussianity of the logit gap q.(k_i - k_j)/sqrt(d) for sampled evicted->
    # nearest-anchor pairs: median |skew| and median excess kurtosis
    e_idx = torch.nonzero(~a_mask).flatten()
    e_idx = e_idx[torch.randperm(len(e_idx), device=device)[:64]]
    # under a causal mask only rows whose gaps are all finite are usable for the
    # moment diagnostic (a masked key gives -inf); the test rows always are in
    # prefix-cache mode, and the original mode has no masking at all
    lg = logits if not causal else logits[torch.isfinite(logits[:, e_idx]).all(1)
                                          & torch.isfinite(logits[:, a_idx]).all(1)]
    Wd = lg.shape[0]
    diffs = lg[:, e_idx].unsqueeze(-1) - lg[:, a_idx].unsqueeze(1)          # (Wd, n, m)
    best = diffs.std(dim=0).argmin(dim=-1)
    gaps = torch.gather(diffs, 2, best.view(1, -1, 1).expand(Wd, -1, 1)).squeeze(-1)  # (Wd, n)
    z = (gaps - gaps.mean(0)) / gaps.std(0).clamp_min(1e-9)
    skew = (z ** 3).mean(0).abs().median()
    kurt = ((z ** 4).mean(0) - 3.0).median()
    return out, float(skew), float(kurt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--docs", type=int, default=2)
    ap.add_argument("--prefill", type=int, default=2048)
    ap.add_argument("--probe", type=int, default=256)
    ap.add_argument("--budget", type=float, default=0.15)
    ap.add_argument("--layer-stride", type=int, default=4)
    ap.add_argument("--lams", type=float, nargs="+", default=[1e-6, 1e-3, 1e-1, 1.0])
    ap.add_argument("--split", choices=["head", "temporal"], default="head",
                    help="head = original first-half/second-half row split (cross-query-head under GQA); "
                         "temporal = fit on the earlier half of the probe positions, test on the later half")
    ap.add_argument("--causal", action="store_true",
                    help="mask keys later than each probe query (the model's real attention)")
    ap.add_argument("--prefix-cache", dest="prefix_cache", action="store_true",
                    help="(with --split temporal) test queries read only the cache that existed at the "
                         "end of the fit window, as at deployment")
    ap.add_argument("--test-len", dest="test_len", type=int, default=0,
                    help="(with --split temporal) fixed test horizon: the last N probe positions are the "
                         "test set, everything earlier in the probe is the fit set (0 = half/half)")
    ap.add_argument("--anchor-window", dest="anchor_window", type=int, default=0,
                    help="(with --split temporal) choose anchors by the mass of only the last N fit "
                         "positions, independent of the calibration size (0 = all fit queries)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32,
        attn_implementation="eager").to(args.device).eval()
    group = model.config.num_attention_heads // model.config.num_key_value_heads

    docs = load_wikitext_docs(min_tokens=args.prefill + args.probe,
                              max_docs=args.docs, tokenizer=tok)
    per_lam = {lam: {"in": [], "out": []} for lam in args.lams}
    skews, kurts = [], []

    for text in docs:
        ids = torch.tensor([tok.encode(text, add_special_tokens=False)[:args.prefill]],
                           device=args.device)
        K, Q, V = capture_kqv(model, ids, args.probe)
        for li in range(0, len(K), args.layer_stride):
            for h in range(K[li].shape[1]):
                k = K[li][0, h]
                v = V[li][0, h]
                q = Q[li][0, h * group:(h + 1) * group].reshape(-1, k.shape[-1])
                L = k.shape[0]
                positions = torch.arange(L - args.probe, L, device=args.device).repeat(group)
                res, s, x = head_solve(k, v, q, args.budget, args.lams, args.device,
                                       positions=positions, split=args.split, causal=args.causal,
                                       prefix_cache=args.prefix_cache, test_len=args.test_len,
                                       anchor_window=args.anchor_window)
                for lam, (r_in, r_out) in res.items():
                    per_lam[lam]["in"].append(r_in)
                    per_lam[lam]["out"].append(r_out)
                skews.append(s)
                kurts.append(x)
        del K, Q, V

    import math
    import statistics as st

    def finite(xs):
        # statistics.median on a list containing nan/inf is undefined (sorting
        # with nan is not a total order); aggregate over finite heads only
        return [x for x in xs if math.isfinite(x)]

    summary = {"model": args.model, "budget": args.budget,
               "heads": len(skews),
               "gaussian_median_abs_skew": st.median(finite(skews)),
               "gaussian_median_excess_kurtosis": st.median(finite(kurts)),
               "removal": {}}
    print(f"\njointly-optimal class-A merge, budget {args.budget}, "
          f"{len(skews)} heads   (fraction of squared eviction error removed)")
    print(f"{'lambda':>8} {'in-sample':>12} {'held-out':>12} {'finite heads':>13}")
    for lam in args.lams:
        fi, fo = finite(per_lam[lam]["in"]), finite(per_lam[lam]["out"])
        mi = st.median(fi) if fi else float("nan")
        mo = st.median(fo) if fo else float("nan")
        summary["removal"][lam] = {"in_median": mi, "out_median": mo,
                                   "out_mean": st.mean(fo) if fo else float("nan"),
                                   "n_finite_out": len(fo), "n_heads": len(per_lam[lam]["out"])}
        print(f"{lam:>8g} {mi:>12.3f} {mo:>12.3f} {len(fo):>6}/{len(per_lam[lam]['out']):<6}")
    print(f"\nlogit-gap Gaussianity: median |skew| {st.median(skews):.3f}, "
          f"median excess kurtosis {st.median(kurts):.3f}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
