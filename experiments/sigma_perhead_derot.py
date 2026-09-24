"""Review check: per-head sigma distribution, mass-weighted sigma, and the
RoPE-vs-contextualisation split of same-token sigma at fine distances.

Same protocol as experiments/sigma_families.py (evenly spaced anchors at the
budget, 192 sampled evicted tokens/head, probe = final 128 queries, post-RoPE),
plus:
  * per-(layer, head) medians / fractions instead of the head-averaged summary;
  * attention-mass-weighted fractions (mass of each evicted token under the probe);
  * same-token pairs at distance 1, 2, 3-4, 5-8, 9-64, 65-512, 513+, with
      sigma_post  : as the paper (both keys post-RoPE)
      sigma_derot : key j re-rotated to position i (removes the RoPE part
                    exactly for identical content; what remains is context)
"""
import argparse, json, os, sys, math
import torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from experiments.data import load_wikitext_docs

BE, PAIR = 0.285, 0.9


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def capture(model, ids, n_probe):
    keys, keys_pre, queries, cs = {}, {}, {}, {}

    def pre_hook(li):
        def hook(module, args, kwargs):
            hs = args[0] if args else kwargs.get("hidden_states")
            pos = kwargs.get("position_embeddings")
            if hs is None or pos is None:
                return args, kwargs
            with torch.no_grad():
                hd = getattr(module, "head_dim", None) or module.q_proj.out_features // module.config.num_attention_heads
                shape = (*hs.shape[:-1], -1, hd)
                q = module.q_proj(hs).view(shape)
                k = module.k_proj(hs).view(shape)
                if hasattr(module, "q_norm"):
                    q = module.q_norm(q)
                if hasattr(module, "k_norm"):
                    k = module.k_norm(k)
                q, k = q.transpose(1, 2), k.transpose(1, 2)
                cos, sin = pos
                qr, kr = apply_rotary_pos_emb(q, k, cos, sin)
                keys[li] = kr.detach().float()[0]                # (kv, L, d)
                keys_pre[li] = k.detach().float()[0]
                queries[li] = qr.detach().float()[0, :, -n_probe:, :]  # (qh, P, d)
                cs[li] = (cos.detach().float()[0], sin.detach().float()[0])  # (L, d)
            return args, kwargs
        return hook

    hs = [l.self_attn.register_forward_pre_hook(pre_hook(i), with_kwargs=True) for i, l in enumerate(model.model.layers)]
    try:
        with torch.no_grad():
            model(input_ids=ids, use_cache=True, past_key_values=DynamicCache())
    finally:
        for h in hs:
            h.remove()
    return keys, keys_pre, queries, cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--docs", type=int, default=3)
    ap.add_argument("--prefill", type=int, default=2560)
    ap.add_argument("--probe", type=int, default=128)
    ap.add_argument("--budget", type=float, default=0.15)
    ap.add_argument("--layer-stride", type=int, default=1)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = a.device
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), attn_implementation="eager").to(dev).eval()
    group = model.config.num_attention_heads // model.config.num_key_value_heads
    docs = load_wikitext_docs(min_tokens=a.prefill + a.probe, max_docs=a.docs, tokenizer=tok)

    per_head = {}   # (li,h) -> list over docs of dicts
    bins = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 64), (65, 512), (513, 10**6)]
    same_post = {b: [] for b in bins}    # head-averaged sigma per pair (paper's aggregation)
    same_derot = {b: [] for b in bins}
    same_post_headmin = {b: [] for b in bins}   # per pair: min over heads
    same_derot_headmin = {b: [] for b in bins}

    for text in docs:
        enc = tok.encode(text, add_special_tokens=False)[: a.prefill]
        ids = torch.tensor([enc], device=dev)
        K, Kp, Q, CS = capture(model, ids, a.probe)
        # same-token pairs
        pos_by = {}
        for p, t in enumerate(enc):
            pos_by.setdefault(t, []).append(p)
        pairs = []
        for pl in pos_by.values():
            for i in range(len(pl) - 1):
                for j in range(i + 1, min(i + 5, len(pl))):
                    pairs.append((pl[i], pl[j]))
        pairs = pairs[:4000]
        pi = torch.tensor([p[0] for p in pairs], device=dev)
        pj = torch.tensor([p[1] for p in pairs], device=dev)
        dist = (pj - pi).cpu()
        sp_sum, sd_sum, sp_min, sd_min, nh = None, None, None, None, 0
        for li in range(0, len(K), a.layer_stride):
            cos, sin = CS[li]
            for h in range(K[li].shape[0]):
                k = K[li][h]; kp = Kp[li][h]
                q = Q[li][h * group:(h + 1) * group].reshape(-1, k.shape[-1])
                L, d = k.shape; sc = d ** -0.5
                # ---- oracle sigma, per head, with probe attention mass
                m = max(2, int(L * a.budget))
                anc = torch.linspace(0, L - 1, m, device=dev).long()
                mask = torch.ones(L, dtype=torch.bool, device=dev); mask[anc] = False
                e = torch.nonzero(mask).flatten()
                e = e[torch.randperm(len(e), device=dev)[:192]]
                pe = (k[e] @ q.T) * sc; pa = (k[anc] @ q.T) * sc
                diff = pe.unsqueeze(1) - pa.unsqueeze(0)
                sb = diff.std(-1).min(1).values                    # (n,)
                logits = (q @ k.T) * sc                            # (P', L) full (probe queries see all keys: they are the last positions, causal ok for all but the probe block itself)
                att = torch.softmax(logits, -1)
                mass_e = att[:, e].mean(0)                         # mean probe mass of each sampled evicted token
                mass_e = mass_e / mass_e.sum()
                rec = per_head.setdefault(f"{li}:{h}", [])
                rec.append({
                    "median": float(sb.median()),
                    "frac_be": float((sb <= BE).float().mean()),
                    "frac_pair": float((sb <= PAIR).float().mean()),
                    "wfrac_be": float(((sb <= BE).float() * mass_e).sum()),
                    "wfrac_pair": float(((sb <= PAIR).float() * mass_e).sum()),
                    "wmean_sigma": float((sb * mass_e).sum()),
                    "evicted_mass_total": float(att[:, mask].sum(-1).mean()),
                })
                # ---- same-token sigma: post-RoPE and de-rotated
                dpost = k[pi] - k[pj]
                kj_re = kp[pj] * cos[pi] + rotate_half(kp[pj]) * sin[pi]   # key j content, rotated to position i
                dder = k[pi] - kj_re
                s_post = ((dpost @ q.T) * sc).std(-1).cpu()
                s_der = ((dder @ q.T) * sc).std(-1).cpu()
                sp_sum = s_post if sp_sum is None else sp_sum + s_post
                sd_sum = s_der if sd_sum is None else sd_sum + s_der
                sp_min = s_post if sp_min is None else torch.minimum(sp_min, s_post)
                sd_min = s_der if sd_min is None else torch.minimum(sd_min, s_der)
                nh += 1
        sp, sd = sp_sum / nh, sd_sum / nh
        for b in bins:
            msk = (dist >= b[0]) & (dist <= b[1])
            same_post[b] += sp[msk].tolist(); same_derot[b] += sd[msk].tolist()
            same_post_headmin[b] += sp_min[msk].tolist(); same_derot_headmin[b] += sd_min[msk].tolist()
        del K, Kp, Q, CS

    # aggregate
    out = {"model": a.model, "prefill": a.prefill, "probe": a.probe, "budget": a.budget, "layer_stride": a.layer_stride,
           "n_heads": len(per_head)}
    heads = []
    for key, recs in per_head.items():
        li, h = map(int, key.split(":"))
        avg = {k: sum(r[k] for r in recs) / len(recs) for k in recs[0]}
        avg.update(layer=li, head=h)
        heads.append(avg)
    out["heads"] = heads
    med = sorted(x["median"] for x in heads)
    out["head_median_sigma_min_p10_p50_p90_max"] = [med[0], med[len(med) // 10], med[len(med) // 2], med[9 * len(med) // 10], med[-1]]
    out["heads_frac_be_gt_0.05"] = sum(x["frac_be"] > 0.05 for x in heads)
    out["heads_frac_be_gt_0.20"] = sum(x["frac_be"] > 0.20 for x in heads)
    out["heads_median_le_0.5"] = sum(x["median"] <= 0.5 for x in heads)
    out["mean_frac_be"] = sum(x["frac_be"] for x in heads) / len(heads)
    out["mean_wfrac_be"] = sum(x["wfrac_be"] for x in heads) / len(heads)
    out["mean_frac_pair"] = sum(x["frac_pair"] for x in heads) / len(heads)
    out["mean_wfrac_pair"] = sum(x["wfrac_pair"] for x in heads) / len(heads)
    out["mean_head_median"] = sum(x["median"] for x in heads) / len(heads)
    out["mean_wmean_sigma"] = sum(x["wmean_sigma"] for x in heads) / len(heads)
    out["top5_lowest_heads"] = sorted(heads, key=lambda x: x["median"])[:5]
    out["top5_highest_frac_be"] = sorted(heads, key=lambda x: -x["frac_be"])[:5]
    st = {}
    for b in bins:
        v = torch.tensor(same_post[b]); w = torch.tensor(same_derot[b])
        vm = torch.tensor(same_post_headmin[b]); wm = torch.tensor(same_derot_headmin[b])
        if len(v) == 0:
            continue
        st[f"{b[0]}-{b[1]}"] = {"pairs": len(v),
                                "sigma_post_median": float(v.median()), "sigma_derot_median": float(w.median()),
                                "post_frac_be": float((v <= BE).float().mean()), "derot_frac_be": float((w <= BE).float().mean()),
                                "post_headmin_median": float(vm.median()), "derot_headmin_median": float(wm.median()),
                                "derot_headmin_frac_be": float((wm <= BE).float().mean())}
    out["same_token"] = st
    print(json.dumps({k: v for k, v in out.items() if k != "heads"}, indent=1))
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
