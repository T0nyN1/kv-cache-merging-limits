"""Does sigma fall as the context grows? More tokens means more candidate
anchors, so if near-duplicates ever appear this is where they would."""
import sys, math, torch; sys.path.insert(0,'.')
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.merge_theory import capture_kq
from experiments.data import load_wikitext_docs
import statistics as st
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float32,
        attn_implementation="sdpa").to("mps").eval()
grp=model.config.num_attention_heads//model.config.num_key_value_heads
need=math.sqrt(math.log(1+0.2914**2))
docs=load_wikitext_docs(min_tokens=1200, max_docs=12, tokenizer=tok)
print(f"break-even sigma = {need:.3f}   (budget fixed at 15%)\n")
print(f"{'context':>8} {'anchors':>8} {'median sigma':>13} {'frac<=thr':>11} {'ceiling':>9}")
print("-"*54)
for L in (1024, 2048, 4096, 8192, 16384):
    text=" ".join(docs)                      # concatenate to reach long contexts
    ids=torch.tensor([tok.encode(text, add_special_tokens=False)[:L]], device="mps")
    if ids.shape[1] < L: print(f"{L:>8}  (corpus too short: {ids.shape[1]})"); continue
    K,Q=capture_kq(model, ids, 96, "mps")
    sig, frac, ceil, m_ = [], [], [], 0
    for li in range(0, len(K), 9):
        for h in range(K[li].shape[1]):
            k=K[li][0,h]; q=Q[li][0,h*grp:(h+1)*grp].reshape(-1,k.shape[-1]); sc=k.shape[-1]**-0.5
            m=max(2,int(k.shape[0]*0.15)); m_=m
            a=torch.linspace(0,k.shape[0]-1,m,device=k.device).long()
            e=torch.tensor([i for i in range(k.shape[0]) if i not in set(a.tolist())],device=k.device)
            e=e[torch.randperm(len(e),device=k.device)[:128]]
            D=k[e].unsqueeze(1)-k[a].unsqueeze(0)
            s_best=(((D@q.T)*sc).std(-1)).min(1).values          # oracle routing
            sig.append(float(s_best.median())); frac.append(float((s_best<=need).float().mean()))
            ceil.append(float((-s_best.pow(2)).exp().mean()))
            del D
    print(f"{L:>8} {m_:>8} {st.mean(sig):>13.3f} {100*st.mean(frac):>10.2f}% {st.mean(ceil):>9.3f}")
    del K,Q
