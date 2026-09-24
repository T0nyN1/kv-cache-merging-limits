"""Review check: the sigma statistic on an ABSOLUTE-position model (GPT-2 medium),
same protocol as experiments/sigma_families.py (evenly spaced anchors at the
budget, 192 sampled evicted tokens/head, probe = final 128 queries; the model has
no RoPE, so keys are captured as the model uses them), plus verbatim 8x repeat and
same-token sigma by distance. Context is limited to 1024 by the model.
"""
import argparse, json, os, sys
import torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.data import load_wikitext_docs

BE, PAIR = 0.285, 0.9


def capture(model, ids, n_probe):
    keys, queries = {}, {}

    def pre_hook(li):
        def hook(module, args, kwargs):
            hs = args[0] if args else kwargs.get("hidden_states")
            if hs is None:
                return args, kwargs
            with torch.no_grad():
                q, k, v = module.c_attn(hs).split(module.split_size, dim=2)
                shape = (*hs.shape[:-1], -1, module.head_dim)
                q = q.view(shape).transpose(1, 2); k = k.view(shape).transpose(1, 2)
                keys[li] = k.detach().float()[0]
                queries[li] = q.detach().float()[0, :, -n_probe:, :]
            return args, kwargs
        return hook

    hs = [l.attn.register_forward_pre_hook(pre_hook(i), with_kwargs=True) for i, l in enumerate(model.transformer.h)]
    try:
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
    finally:
        for h in hs:
            h.remove()
    return keys, queries


def oracle(K, Q, budget, stride, dev):
    heads = []
    for li in range(0, len(K), stride):
        for h in range(K[li].shape[0]):
            k = K[li][h]; q = Q[li][h]
            L, d = k.shape; sc = d ** -0.5
            m = max(2, int(L * budget))
            anc = torch.linspace(0, L - 1, m, device=dev).long()
            mask = torch.ones(L, dtype=torch.bool, device=dev); mask[anc] = False
            e = torch.nonzero(mask).flatten()
            e = e[torch.randperm(len(e), device=dev)[:192]]
            pe = (k[e] @ q.T) * sc; pa = (k[anc] @ q.T) * sc
            sb = (pe.unsqueeze(1) - pa.unsqueeze(0)).std(-1).min(1).values
            heads.append({"layer": li, "head": h, "median": float(sb.median()),
                          "frac_be": float((sb <= BE).float().mean()), "frac_pair": float((sb <= PAIR).float().mean())})
    return heads


def same_token(K, Q, enc, stride, dev, bins):
    pos_by = {}
    for p, t in enumerate(enc):
        pos_by.setdefault(t, []).append(p)
    pairs = []
    for pl in pos_by.values():
        for i in range(len(pl) - 1):
            for j in range(i + 1, min(i + 5, len(pl))):
                pairs.append((pl[i], pl[j]))
    pairs = pairs[:4000]
    pi = torch.tensor([p[0] for p in pairs], device=dev); pj = torch.tensor([p[1] for p in pairs], device=dev)
    dist = (pj - pi).cpu()
    ssum, smin, nh = None, None, 0
    for li in range(0, len(K), stride):
        for h in range(K[li].shape[0]):
            k = K[li][h]; q = Q[li][h]; sc = k.shape[-1] ** -0.5
            s = (((k[pi] - k[pj]) @ q.T) * sc).std(-1).cpu()
            ssum = s if ssum is None else ssum + s
            smin = s if smin is None else torch.minimum(smin, s)
            nh += 1
    s = ssum / nh
    out = {}
    for b in bins:
        msk = (dist >= b[0]) & (dist <= b[1])
        if int(msk.sum()) == 0:
            continue
        out[f"{b[0]}-{b[1]}"] = {"pairs": int(msk.sum()), "sigma_median": float(s[msk].median()),
                                 "frac_be": float((s[msk] <= BE).float().mean()),
                                 "headmin_median": float(smin[msk].median()), "headmin_frac_be": float((smin[msk] <= BE).float().mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-medium")
    ap.add_argument("--docs", type=int, default=3)
    ap.add_argument("--prefill", type=int, default=896)
    ap.add_argument("--probe", type=int, default=128)
    ap.add_argument("--budget", type=float, default=0.15)
    ap.add_argument("--layer-stride", type=int, default=1)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = a.device
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="eager").to(dev).eval()
    docs = load_wikitext_docs(min_tokens=a.prefill + a.probe, max_docs=a.docs, tokenizer=tok)
    bins = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 64), (65, 512), (513, 10**6)]
    res = {"model": a.model, "prefill": a.prefill, "probe": a.probe, "budget": a.budget, "oracle_docs": [], "same_token_docs": []}
    allheads = []
    for text in docs:
        enc = tok.encode(text, add_special_tokens=False)[: a.prefill]
        ids = torch.tensor([enc], device=dev)
        K, Q = capture(model, ids, a.probe)
        heads = oracle(K, Q, a.budget, a.layer_stride, dev)
        allheads += heads
        res["oracle_docs"].append({"mean_head_median": sum(h["median"] for h in heads) / len(heads),
                                   "mean_frac_be": sum(h["frac_be"] for h in heads) / len(heads),
                                   "mean_frac_pair": sum(h["frac_pair"] for h in heads) / len(heads), "heads": len(heads)})
        res["same_token_docs"].append(same_token(K, Q, enc, a.layer_stride, dev, bins))
        del K, Q
    base = tok.encode(docs[0], add_special_tokens=False)[: a.prefill // 8]
    rep = (base * 8)[: a.prefill]
    K, Q = capture(model, torch.tensor([rep], device=dev), a.probe)
    heads = oracle(K, Q, a.budget, a.layer_stride, dev)
    res["repeat8x"] = {"mean_head_median": sum(h["median"] for h in heads) / len(heads),
                       "mean_frac_be": sum(h["frac_be"] for h in heads) / len(heads),
                       "mean_frac_pair": sum(h["frac_pair"] for h in heads) / len(heads)}
    res["repeat8x_same_token"] = same_token(K, Q, rep, a.layer_stride, dev, bins)
    med = sorted(h["median"] for h in allheads)
    res["head_median_min_p10_p50_p90_max"] = [med[0], med[len(med) // 10], med[len(med) // 2], med[9 * len(med) // 10], med[-1]]
    res["heads_frac_be_gt_0.20"] = sum(h["frac_be"] > 0.2 for h in allheads)
    res["heads_median_le_0.5"] = sum(h["median"] <= 0.5 for h in allheads)
    res["n_head_rows"] = len(allheads)
    res["head_medians"] = [h["median"] for h in allheads]
    res["head_frac_be"] = [h["frac_be"] for h in allheads]
    # same-token summary averaged over docs
    st = {}
    for key in res["same_token_docs"][0]:
        rows = [d[key] for d in res["same_token_docs"] if key in d]
        st[key] = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    res["same_token_summary"] = st
    res["oracle_summary"] = {k: sum(d[k] for d in res["oracle_docs"]) / len(res["oracle_docs"]) for k in ("mean_head_median", "mean_frac_be", "mean_frac_pair")}
    print(json.dumps({k: v for k, v in res.items() if k not in ("same_token_docs",)}, indent=1))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
