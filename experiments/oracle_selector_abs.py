"""Exp A: is the ~25% oracle ceiling selector-conditional, and what happens in ABSOLUTE error?

Same protocol as runs/local_repro/oracle_dec_W*.json (prefill 2560, causal, prefix cache,
test = last 128 positions, 2 wikitext docs, every 4th layer, all KV heads, budget 0.15),
but for each head we compute, for two anchor selectors,
  snap : anchors = top-mass of the last 128 fit positions (paper's choice, --anchor-window 128)
  h2o  : anchors = top-mass accumulated over ALL W fit positions (--anchor-window 0)
the held-out eviction error E_S and the ridge-merged residual R_S(lam), both normalised by
||o_full||^2 over the test rows, plus anchor-position statistics.
"""
import json, os, sys, statistics as st
import torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.classa_oracle import capture_kqv
from experiments.data import load_wikitext_docs

dev = "mps"
PREFILL, TEST, BUDGET, STRIDE = 2560, 128, 0.15, 4
WS = [128, 1024, 2048]
LAMS = [1e-3, 1e-1, 1.0]
PROBE = max(WS) + TEST

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float32,
                                             attn_implementation="eager").to(dev).eval()
group = model.config.num_attention_heads // model.config.num_key_value_heads
docs = load_wikitext_docs(min_tokens=PREFILL + PROBE, max_docs=2, tokenizer=tok)

rows = []  # per head dicts
for text in docs:
    ids = torch.tensor([tok.encode(text, add_special_tokens=False)[:PREFILL]], device=dev)
    K, Q, V = capture_kqv(model, ids, PROBE)
    for li in range(0, len(K), STRIDE):
        for h in range(K[li].shape[1]):
            k, v = K[li][0, h], V[li][0, h]
            L, d = k.shape
            sc = d ** -0.5
            q = Q[li][0, h * group:(h + 1) * group].reshape(-1, d)
            pos = torch.arange(L - PROBE, L, device=dev).repeat(group)
            cut = L - TEST
            kc, vc = k[:cut], v[:cut]
            Lc = cut
            m = max(2, int(Lc * BUDGET))
            logits = (q @ kc.T) * sc
            future = torch.arange(Lc, device=dev)[None, :] > pos[:, None]
            logits = logits.masked_fill(future, float("-inf"))
            attn = torch.softmax(logits, -1)
            o_full = attn @ vc
            test_rows = pos >= cut
            N = float((o_full[test_rows] ** 2).sum())
            rec = {"layer": li, "head": h, "N": N}
            for W in WS:
                fit_rows = (pos < cut) & (pos >= cut - W)
                for sel in ["snap", "h2o"]:
                    if sel == "snap":
                        sel_rows = fit_rows & (pos >= cut - 128)
                    else:
                        sel_rows = fit_rows
                    if sel == "snap" and W != 128 and W != WS[1]:
                        pass
                    mass = attn[sel_rows].sum(0)
                    a_idx = torch.topk(mass, m).indices.sort().values
                    a_mask = torch.zeros(Lc, dtype=torch.bool, device=dev); a_mask[a_idx] = True
                    ak = torch.softmax(logits[:, a_mask], -1)
                    err = o_full - ak @ vc[a_mask]
                    A_fit, E_fit = ak[fit_rows], err[fit_rows]
                    A_te, E_te = ak[test_rows], err[test_rows]
                    base_out = float((E_te ** 2).sum())
                    G = A_fit.T @ A_fit
                    key = f"W{W}_{sel}"
                    rec[key + "_evict"] = base_out / N
                    rec[key + "_anchor_med_pos"] = float(a_idx.float().median())
                    rec[key + "_anchor_frac_last128"] = float((a_idx >= cut - 128).float().mean())
                    rec[key + "_anchor_frac_fitwin"] = float((a_idx >= cut - W).float().mean())
                    for lam in LAMS:
                        delta = torch.linalg.solve(G + lam * torch.eye(m, device=dev), A_fit.T @ E_fit)
                        res_out = float(((A_te @ delta - E_te) ** 2).sum())
                        rec[key + f"_merged_l{lam:g}"] = res_out / N
                        rec[key + f"_removed_l{lam:g}"] = 1 - res_out / max(base_out, 1e-12)
            rows.append(rec)
    del K, Q, V

def med(key):
    xs = [r[key] for r in rows if r.get(key) == r.get(key)]
    return st.median(xs)

print(f"heads={len(rows)}")
keys = sorted(set(k for r in rows for k in r if k not in ("layer", "head", "N")))
for k in keys:
    print(f"{k:34s} median {med(k):9.4f}")
# head-wise comparisons in absolute error
print("\n--- head-wise absolute comparisons (fraction of heads) ---")
for W in WS:
    for lam in LAMS:
        a = [r[f"W{W}_h2o_merged_l{lam:g}"] < r["W128_snap_evict"] for r in rows]
        b = [r[f"W{W}_h2o_merged_l{lam:g}"] < r[f"W128_snap_merged_l{lam:g}"] for r in rows]
        c = [r[f"W{W}_h2o_evict"] < r["W128_snap_evict"] for r in rows]
        ratio = st.median([r[f"W{W}_h2o_merged_l{lam:g}"] / max(r["W128_snap_evict"], 1e-12) for r in rows])
        print(f"W={W:5d} lam={lam:g}: h2o-merged < snap-evict in {sum(a)/len(a):.2f} of heads; "
              f"h2o-merged < snap128-merged in {sum(b)/len(b):.2f}; h2o-evict < snap-evict in {sum(c)/len(c):.2f}; "
              f"median h2o-merged/snap-evict = {ratio:.3f}")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expA_selector.json")
json.dump(rows, open(out, "w"), indent=1)
print("wrote", out)
