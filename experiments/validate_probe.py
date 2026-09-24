import sys, copy, torch, torch.nn.functional as F
sys.path.insert(0,'.')
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from experiments.probe import capture_dense, probe_logprobs
from experiments.data import load_wikitext_docs

dev="mps"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float16, attn_implementation="eager").to(dev).eval()
docs = load_wikitext_docs(min_tokens=800, max_docs=1, tokenizer=tok)
ids = tok.encode(docs[0], add_special_tokens=False)[:600]
L, P = 500, 32
prefix = torch.tensor([ids[:L]], device=dev); probe = torch.tensor([ids[L:L+P]], device=dev)

# reference 1: one plain forward over the whole thing, no cache tricks
with torch.no_grad():
    full = model(torch.tensor([ids[:L+P]], device=dev), use_cache=False).logits.float()
ref  = F.log_softmax(full[0, L:L+P], dim=-1)      # aligned with the probe forward
refs = F.log_softmax(full[0, L-1:L-1+P], dim=-1)  # aligned with sequential decode

# reference 2: probe path through the hooked single forward on a dense cache
dense_cache, scores = capture_dense(model, prefix, dev, model.config.num_key_value_heads)
lp_probe = probe_logprobs(model, copy.deepcopy(dense_cache), probe, L, dev)

# reference 3: sequential decode, one token at a time
c = copy.deepcopy(dense_cache); seq=[]
with torch.no_grad():
    # first prediction comes from the prefill's last logit
    o = model(input_ids=prefix, use_cache=True, past_key_values=DynamicCache(), return_dict=True)
    seq.append(F.log_softmax(o.logits[0,-1].float(), -1))
    c3 = o.past_key_values
    for i in range(P-1):
        o = model(input_ids=probe[:, i:i+1], past_key_values=c3, use_cache=True,
                  position_ids=torch.tensor([[L+i]], device=dev), return_dict=True)
        seq.append(F.log_softmax(o.logits[0,-1].float(), -1))
seq = torch.stack(seq)

def kl(a,b): return torch.sum(a.exp()*(a-b), -1).mean().item()
print(f"KL(full-forward || probe-path)      = {kl(ref, lp_probe):.3e}   <- must be ~0")
print(f"KL(full-forward || sequential)      = {kl(refs, seq):.3e}   <- must be ~0")
print(f"max |logprob diff| probe vs full    = {(ref-lp_probe).abs().max():.3e}")
