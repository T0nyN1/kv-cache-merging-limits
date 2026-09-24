"""Eviction bias is a difference of centroids (it cancels); merge noise is an
incoherent sum (it does not). Measure both at the scale they actually occur."""
import sys, torch; sys.path.insert(0,'.')
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.probe import capture_dense
from experiments.data import load_wikitext_docs
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float32,
        attn_implementation="eager").to("mps").eval()
docs=load_wikitext_docs(min_tokens=3000, max_docs=3, tokenizer=tok)
BUD=0.15
rows=[]
for text in docs:
    ids=torch.tensor([tok.encode(text, add_special_tokens=False)[:3000]], device="mps")
    cache, sc = capture_dense(model, ids, "mps", model.config.num_key_value_heads, window=32)
    for li in range(0, len(cache.layers), 5):
        V=cache.layers[li].values[0]                 # (H, L, d)
        s=sc["full"][li][0]                          # (H, L)
        H,L,d=V.shape
        m=max(2,int(L*BUD))
        idx=s.topk(m,dim=-1).indices
        for h in range(H):
            a=idx[h]; mask=torch.ones(L,dtype=torch.bool,device=V.device); mask[a]=False
            wa=s[h][a]; we=s[h][mask]
            vA=(V[h][a]*wa.unsqueeze(-1)).sum(0)/wa.sum()
            vE=(V[h][mask]*we.unsqueeze(-1)).sum(0)/we.sum()
            vnorm=V[h].norm(dim=-1).mean()
            rows.append((float((vA-vE).norm()/vnorm), float(mask.sum()), float(vnorm)))
    del cache, sc
import statistics
gap=[r[0] for r in rows]; n=[r[1] for r in rows]
print(f"\n{len(rows)} (layer, head) samples, budget {BUD}\n")
print(f"|v_A - v_E| / mean||v||   (the bias merging removes)  : {statistics.mean(gap):.4f}")
print(f"evicted tokens per head n                              : {statistics.mean(n):.0f}")
import math
for sigma in (1.0, 1.21, 1.5):
    cv=math.sqrt(math.exp(sigma**2)-1)
    thresh=cv/math.sqrt(statistics.mean(n))
    print(f"  sigma={sigma:<5} CV={cv:5.2f}  merge helps only if gap > CV/sqrt(n) = {thresh:.4f}"
          f"   -> {'HELPS' if statistics.mean(gap)>thresh else 'HURTS'}")
