"""Decisive test: the coefficient error is driven by the shared query, so it
scales with n exactly like the bias it removes. Merging then helps only when the
relative coefficient error CV = sqrt(e^{sigma^2}-1) is below the relative
centroid gap. Measure what fraction of evicted tokens clear that bar."""
import sys, math, torch; sys.path.insert(0,'.')
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.merge_theory import capture_kq
from experiments.data import load_wikitext_docs
import torch.nn.functional as F

GAP = 0.2914                                   # measured |v_A - v_E| / ||v||
sigma_needed = math.sqrt(math.log(1 + GAP**2))
print(f"relative centroid gap (bias merging can remove) : {GAP:.4f}")
print(f"=> merging helps only where CV < {GAP:.4f}, i.e. sigma < {sigma_needed:.4f}\n")

tok=AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float32,
        attn_implementation="eager").to("mps").eval()
docs=load_wikitext_docs(min_tokens=2700, max_docs=3, tokenizer=tok)
grp = model.config.num_attention_heads // model.config.num_key_value_heads
frac_cos, frac_mah, sig_cos, sig_mah = [], [], [], []
for text in docs:
    ids=torch.tensor([tok.encode(text, add_special_tokens=False)[:2560]], device="mps")
    K,Q = capture_kq(model, ids, 128, "mps")
    for li in range(0, len(K), 5):
        for h in range(K[li].shape[1]):
            k=K[li][0,h]; q=Q[li][0,h*grp:(h+1)*grp].reshape(-1,k.shape[-1])
            qc=q-q.mean(0,keepdim=True); Sig=(qc.T@qc)/qc.shape[0]; sc=k.shape[-1]**-0.5
            m=max(2,int(k.shape[0]*0.15))
            a=torch.linspace(0,k.shape[0]-1,m,device=k.device).long()
            e=torch.tensor([i for i in range(k.shape[0]) if i not in set(a.tolist())],device=k.device)
            e=e[torch.randperm(len(e),device=k.device)[:256]]
            D=k[e].unsqueeze(1)-k[a].unsqueeze(0)
            sigma=((D@q.T)*sc).std(-1)
            cos=1-F.normalize(k[e],dim=-1)@F.normalize(k[a],dim=-1).T
            mah=(torch.einsum("nmd,de,nme->nm",D,Sig,D).clamp_min(0).sqrt()*sc)
            s_c=sigma.gather(1,cos.argmin(1,keepdim=True)).squeeze(1)
            s_m=sigma.gather(1,mah.argmin(1,keepdim=True)).squeeze(1)
            s_best=sigma.min(1).values
            sig_cos.append(float(s_c.mean())); sig_mah.append(float(s_m.mean()))
            frac_cos.append(float((s_m<=sigma_needed).float().mean()))
            frac_mah.append(float((s_best<=sigma_needed).float().mean()))
    del K,Q
import statistics as st
print(f"sigma to the best anchor found by the whitened metric : {st.mean(sig_mah):.3f}")
print(f"sigma to the *oracle* best anchor (lower bound)       : n/a per-token, see below\n")
print(f"fraction of evicted tokens with sigma <= {sigma_needed:.3f}:")
print(f"  routed by the query-whitened metric : {100*st.mean(frac_cos):.2f} %")
print(f"  routed by an oracle (min over all anchors) : {100*st.mean(frac_mah):.2f} %")
print(f"\nsigma reduction still required to break even: {st.mean(sig_mah)/sigma_needed:.1f}x")
