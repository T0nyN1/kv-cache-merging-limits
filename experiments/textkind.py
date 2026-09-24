"""Do repetitive contexts (code, duplicated passages) produce the near-duplicate
keys that prose does not? This is the most plausible place merging could work."""
import sys, math, glob, torch; sys.path.insert(0,'.')
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.merge_theory import capture_kq
from experiments.data import load_wikitext_docs
import statistics as st
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float32,
        attn_implementation="sdpa").to("mps").eval()
grp=model.config.num_attention_heads//model.config.num_key_value_heads
need=math.sqrt(math.log(1+0.2914**2))

prose = " ".join(load_wikitext_docs(min_tokens=1200, max_docs=8, tokenizer=None))
code  = "\n".join(open(f).read() for f in sorted(glob.glob("core/*.py")+glob.glob("experiments/*.py")
                                                +glob.glob("baselines/*.py")+glob.glob("evaluation/**/*.py", recursive=True)))
para  = (load_wikitext_docs(min_tokens=1200, max_docs=1, tokenizer=None)[0][:6000] + "\n") * 8   # literal repeats

print(f"break-even sigma = {need:.3f}, context 4096, budget 15%\n")
print(f"{'text':<26} {'median sigma':>13} {'frac<=thr':>11} {'ceiling':>9}")
print("-"*62)
for name, text in (("wikitext prose", prose), ("python source (this repo)", code),
                   ("literally repeated passage", para)):
    ids=torch.tensor([tok.encode(text, add_special_tokens=False)[:4096]], device="mps")
    if ids.shape[1] < 4096: print(f"{name:<26}  too short ({ids.shape[1]})"); continue
    K,Q=capture_kq(model, ids, 96, "mps")
    sig,frac,ceil=[],[],[]
    for li in range(0,len(K),9):
        for h in range(K[li].shape[1]):
            k=K[li][0,h]; q=Q[li][0,h*grp:(h+1)*grp].reshape(-1,k.shape[-1]); sc=k.shape[-1]**-0.5
            m=max(2,int(k.shape[0]*0.15))
            a=torch.linspace(0,k.shape[0]-1,m,device=k.device).long()
            e=torch.tensor([i for i in range(k.shape[0]) if i not in set(a.tolist())],device=k.device)
            e=e[torch.randperm(len(e),device=k.device)[:128]]
            D=k[e].unsqueeze(1)-k[a].unsqueeze(0)
            sb=(((D@q.T)*sc).std(-1)).min(1).values
            sig.append(float(sb.median())); frac.append(float((sb<=need).float().mean()))
            ceil.append(float((-sb.pow(2)).exp().mean())); del D
    print(f"{name:<26} {st.mean(sig):>13.3f} {100*st.mean(frac):>10.2f}% {st.mean(ceil):>9.3f}")
    del K,Q
