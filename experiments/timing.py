import sys, time, torch
sys.path.insert(0,'.')
from core.ot_kv_v3 import otkv_v3_compress
from core.ot_kv import otkv_compress
dev="mps"; torch.manual_seed(0)
B,H,D = 1,8,128
print(f"{'setting':<42} {'ms / layer':>11} {'x evict':>8}")
print("-"*64)
for L, budget in [(3300, 660), (7000, 1400)]:
    k=torch.randn(B,H,L,D,device=dev,dtype=torch.float16); v=torch.randn_like(k)
    s=(torch.rand(B,H,L,device=dev)**3).float()
    def timeit(fn, n=3):
        fn(); torch.mps.synchronize(); t=time.time()
        for _ in range(n): fn()
        torch.mps.synchronize(); return (time.time()-t)/n*1000
    base = timeit(lambda: torch.gather(k,2,torch.topk(s,budget,-1).indices.unsqueeze(-1).expand(-1,-1,-1,D)))
    rows = [
        (f"L={L}->{budget} evict (top-k)", base),
        (f"L={L}->{budget} v3 topk+merge", timeit(lambda: otkv_v3_compress(k,v,budget,s))),
        (f"L={L}->{budget} v3 kmeans:50+merge", timeit(lambda: otkv_v3_compress(k,v,budget,s,select_mode='kmeans:50'))),
        (f"L={L}->{budget} v3 kmeans:50 no-merge", timeit(lambda: otkv_v3_compress(k,v,budget,s,select_mode='kmeans:50',merge=False))),
        (f"L={L}->{budget} v2 otkv (dense sinkhorn,50it)", timeit(lambda: otkv_compress(k,v,budget,importance_scores=s), n=1)),
    ]
    for name, ms in rows:
        print(f"{name:<42} {ms:>11.1f} {ms/base:>8.0f}x")
    print()
