"""Exp C: does the decode mean shift (|mu_dec - mu_pre| / sigma_pre = 0.35-0.46 median) exceed a
within-prefill null?  Same construction as experiments/sigma_decode.py (evenly spaced anchors,
192 sampled evicted tokens per head, anchor chosen by min prefill std over the LAST 128 positions),
but we also compute the same statistic against
  nullA : prefill positions [L-192, L-128)  -> 64 positions x 2 heads = 128 rows, same size as decode
  nullB : the two halves of the calibration window itself, [L-128,L-64) vs [L-64,L)
and, to expose the selection effect, the shift at a RANDOM anchor instead of the min-std anchor.
"""
import json, os, sys, statistics as st
import torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.data import load_wikitext_docs, build_niah_samples
from experiments.sigma_families import capture_kq
from experiments.sigma_decode import decode_queries

dev = "mps"
CONTEXT, PROBE, NEW, BUDGET, STRIDE, PROMPTS, NS = 2500, 192, 64, 0.15, 4, 2, 192
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float32,
                                             attn_implementation="eager").to(dev).eval()
group = model.config.num_attention_heads // model.config.num_key_value_heads
docs = load_wikitext_docs(min_tokens=CONTEXT + 200, max_docs=PROMPTS + 4, tokenizer=tok)
niah = build_niah_samples(docs[PROMPTS:], tok, context_tokens=CONTEXT, depths=(0.35, 0.7)[:PROMPTS], seed=0)
kinds = {"qa_needle": [s["prompt"] for s in niah],
         "continuation": [tok.decode(tok.encode(d, add_special_tokens=False)[:CONTEXT]) for d in docs[:PROMPTS]]}
kinds["summary_instruction"] = [t + "\n\nSummarize the passage above in three sentences.\nSummary:" for t in kinds["continuation"]]

def stats(gp_cal, gp_other, j):
    """gp_*: (n, m, rows). anchor j: (n,). returns |mu_other - mu_cal| / sigma_cal at j"""
    mu_c = gp_cal.mean(-1).gather(1, j[:, None]).squeeze(1)
    sd_c = gp_cal.std(-1).gather(1, j[:, None]).squeeze(1)
    mu_o = gp_other.mean(-1).gather(1, j[:, None]).squeeze(1)
    return (mu_o - mu_c).abs() / sd_c.clamp_min(1e-6)

results = {}
for kind, prompts in kinds.items():
    results[kind] = []
    for text in prompts:
        ids = torch.tensor([tok.encode(text, add_special_tokens=False)[-(CONTEXT + 200):]], device=dev)
        keys, q_pre = capture_kq(model, ids, PROBE)
        q_dec, gen = decode_queries(model, ids, NEW, STRIDE)
        R = []
        for li in range(0, len(keys), STRIDE):
            if li not in q_dec:
                continue
            Kl = keys[li][0]; Qp = q_pre[li][0]; Qd = q_dec[li]
            for h in range(Kl.shape[0]):
                k = Kl[h]; L, d = k.shape; sc = d ** -0.5
                qp_all = Qp[h * group:(h + 1) * group]                    # (group, PROBE, d)
                cal = qp_all[:, -128:, :].reshape(-1, d)                   # paper's calibration probe: last 128
                nullA = qp_all[:, -192:-128, :].reshape(-1, d)             # 64 earlier prefill positions
                calB1 = qp_all[:, -128:-64, :].reshape(-1, d)
                calB2 = qp_all[:, -64:, :].reshape(-1, d)
                qd = Qd[:, h * group:(h + 1) * group, :].reshape(-1, d)    # (64*group, d)
                m = max(2, int(L * BUDGET))
                a = torch.linspace(0, L - 1, m, device=dev).long()
                mask = torch.ones(L, dtype=torch.bool, device=dev); mask[a] = False
                e = torch.nonzero(mask).flatten()
                e = e[torch.randperm(len(e), device=dev)[:NS]]
                D = k[e].unsqueeze(1) - k[a].unsqueeze(0)                  # (n, m, d)
                g = lambda qq: torch.einsum("nmd,pd->nmp", D, qq) * sc
                gc, gA, gB1, gB2, gd = g(cal), g(nullA), g(calB1), g(calB2), g(qd)
                j = gc.std(-1).argmin(1)                                   # anchor chosen on the calibration probe
                jr = torch.randint(0, m, (gc.shape[0],), device=dev)       # random anchor
                s_dec = stats(gc, gd, j)
                s_nullA = stats(gc, gA, j)
                s_nullB = stats(gB1, gB2, j)                               # halves of the calibration window
                s_dec_rand = stats(gc, gd, jr)
                s_nullA_rand = stats(gc, gA, jr)
                # decode sigma at chosen anchor / calibration sigma, and same for nullA
                sd_c = gc.std(-1).gather(1, j[:, None]).squeeze(1)
                r_dec = gd.std(-1).gather(1, j[:, None]).squeeze(1) / sd_c
                r_A = gA.std(-1).gather(1, j[:, None]).squeeze(1) / sd_c
                R.append(torch.stack([s_dec, s_nullA, s_nullB, s_dec_rand, s_nullA_rand, r_dec, r_A], 1).cpu())
        R = torch.cat(R, 0)
        names = ["shift_decode", "shift_nullA_prefill64", "shift_nullB_calhalves", "shift_decode_randanchor",
                 "shift_nullA_randanchor", "sigma_ratio_decode", "sigma_ratio_nullA"]
        rec = {"generated": tok.decode(gen)[:100], "distinct_gen_tokens": len(set(gen)), "n": int(R.shape[0])}
        for i, nm in enumerate(names):
            rec[nm + "_median"] = float(R[:, i].median()); rec[nm + "_p90"] = float(R[:, i].quantile(0.9))
        results[kind].append(rec)
        print(f"[{kind}] gen={rec['generated']!r} distinct={rec['distinct_gen_tokens']}")
        for nm in names:
            print(f"    {nm:28s} median {rec[nm+'_median']:.3f}  p90 {rec[nm+'_p90']:.3f}")
        del keys, q_pre, q_dec
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "runs", "local_repro", "sigma_drift_null_qwen3.json")
json.dump(results, open(out, "w"), indent=1)
print("wrote", out)
