import torch, sys
sys.path.insert(0, '.')
from core.ot_kv import sinkhorn_matrix_space, sinkhorn_log_space, _build_ot_cost_matrix

torch.manual_seed(0)
B, H, N, M, D = 1, 4, 512, 512, 128

# realistic-ish keys: correlated directions
k_evict = torch.randn(B, H, N, D)
k_anchor = torch.randn(B, H, M, D)
ke = torch.nn.functional.normalize(k_evict, dim=-1)
ka = torch.nn.functional.normalize(k_anchor, dim=-1)
dists = 1.0 - torch.matmul(ke, ka.transpose(-1, -2)).float()
w_anchor = torch.rand(B, H, M).float() + 0.05

C, rel = _build_ot_cost_matrix(dists, w_anchor, gamma=1.0)
print(f"cost stats: min={C.min():.4f} mean={C.mean():.4f} max={C.max():.4f}")

for eps in [0.01, 0.05, 0.2, 1.0]:
    K = torch.exp(-C.float() / eps)
    zero_frac = (K == 0).float().mean().item()
    row_zero = (K.sum(-1) == 0).float().mean().item()
    print(f"eps={eps:<5} exp(-C/eps): zero entries={zero_frac:.3%}  fully-zero rows={row_zero:.3%}  max={K.max():.3e}")

print()
for eps in [0.01, 0.05, 0.2]:
    T = sinkhorn_matrix_space(C, epsilon=eps, max_iter=50)
    Tl = sinkhorn_log_space(C, epsilon=eps, max_iter=50)
    for name, t in (("matrix", T), ("log   ", Tl)):
        row = t.sum(-1)      # should be 1/N
        col = t.sum(-2)      # should be 1/M
        print(f"eps={eps:<5} {name}: total mass={t.sum(dim=(-1,-2)).mean():.4f} (want 1.0) | "
              f"row err={(row - 1.0/N).abs().max():.3e} (want<1e-6) | col err={(col - 1.0/M).abs().max():.3e} | "
              f"has_nan={torch.isnan(t).any().item()}")
    print()

# what the *n_evict rescale does to the merged value magnitude
v_evict = torch.randn(B, H, N, D)
v_anchor = torch.randn(B, H, M, D)
for eps in [0.01, 0.2]:
    for name, fn in (("matrix", sinkhorn_matrix_space), ("log", sinkhorn_log_space)):
        T = fn(C, epsilon=eps, max_iter=50)
        delta_v1 = torch.matmul(T.transpose(-1, -2), v_evict)
        delta_v2 = delta_v1 * N
        print(f"eps={eps} {name:<7}: ||v_anchor||={v_anchor.norm(dim=-1).mean():.3f}  "
              f"||delta_v1||={delta_v1.norm(dim=-1).mean():.4f}  ||delta_v2 (xN)||={delta_v2.norm(dim=-1).mean():.4f}")
