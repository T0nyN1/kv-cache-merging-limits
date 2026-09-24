"""Is RoPE what destroys the key duplication merging needs?

Take pairs of positions holding the *same token id* and measure sigma as a
function of their positional distance. If RoPE is the cause, sigma should start
near zero for adjacent repeats and grow with |delta position|."""
import sys, math, torch, collections; sys.path.insert(0,'.')
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.merge_theory import capture_kq
from experiments.data import load_wikitext_docs
import statistics as st
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float32,
        attn_implementation="sdpa").to("mps").eval()
grp=model.config.num_attention_heads//model.config.num_key_value_heads
need=math.sqrt(math.log(1+0.2914**2))
text=" ".join(load_wikitext_docs(min_tokens=1200, max_docs=6, tokenizer=None))
ids_list=tok.encode(text, add_special_tokens=False)[:4096]
ids=torch.tensor([ids_list], device="mps")
K,Q=capture_kq(model, ids, 96, "mps")

# group positions by token id, keep ids that repeat
pos=collections.defaultdict(list)
for p,t in enumerate(ids_list): pos[t].append(p)
pairs=[]
for t,ps in pos.items():
    if len(ps)<2: continue
    for i in range(len(ps)):
        for j in range(i+1,len(ps)):
            pairs.append((ps[i],ps[j],ps[j]-ps[i]))
import random; random.Random(0).shuffle(pairs); pairs=pairs[:4000]
buckets={(0,8):[], (8,64):[], (64,512):[], (512,4096):[]}
for li in range(0,len(K),9):
    for h in range(K[li].shape[1]):
        k=K[li][0,h]; q=Q[li][0,h*grp:(h+1)*grp].reshape(-1,k.shape[-1]); sc=k.shape[-1]**-0.5
        for (a,b,d) in pairs[:900]:
            for lo,hi in buckets:
                if lo<=d<hi:
                    s=float((((k[a]-k[b])@q.T)*sc).std()); buckets[(lo,hi)].append(s); break
print(f"\nsame-token pairs, sigma vs positional distance (break-even {need:.3f})\n")
print(f"{'|delta pos|':>14} {'pairs':>8} {'median sigma':>13} {'frac<=thr':>11}")
print("-"*52)
for (lo,hi),v in buckets.items():
    if not v: continue
    print(f"{f'{lo}-{hi}':>14} {len(v):>8} {st.median(v):>13.3f} {100*sum(1 for x in v if x<=need)/len(v):>10.2f}%")
