"""Verify: weights inside logsumexp reduce effective temperature by factor m."""
import torch

mu = 0.1

for m, label in [(3, "m=3 (real-world)"), (10, "m=10 (DTLZ2)")]:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    
    torch.manual_seed(42)
    inner = torch.rand(m) * 0.8 + 0.1  # values in [0.1, 0.9]
    ref = torch.zeros(m)
    weights = torch.ones(m) / m  # uniform, normalized

    # --- Forward values ---
    S_lin = mu * torch.logsumexp((inner - ref) / mu, dim=-1)
    S_ours = mu * torch.logsumexp(weights * (inner - ref) / mu, dim=-1)
    S_equiv = mu * torch.logsumexp((inner - ref) / (m * mu), dim=-1)
    S_logadd = mu * torch.logsumexp((inner - ref) / mu + torch.log(weights), dim=-1)

    print(f"\n  Forward values:")
    print(f"    Lin et al (no weights):      S = {S_lin.item():.6f}")
    print(f"    Ours (w*x/mu):               S = {S_ours.item():.6f}")
    print(f"    Equivalent (x/(m*mu)):        S = {S_equiv.item():.6f}")
    print(f"    Fix (x/mu + log(w)):          S = {S_logadd.item():.6f}")
    print(f"    Ours == x/(m*mu)?             {torch.allclose(S_ours, S_equiv)}")

    # --- Gradients ---
    def get_grad(fn):
        x = inner.clone().requires_grad_(True)
        s = fn(x, ref, weights, mu)
        s.backward()
        return x.grad.clone()

    def fn_broken(x, ref, w, mu):
        return mu * torch.logsumexp(w * (x - ref) / mu, dim=-1)

    def fn_lin(x, ref, w, mu):
        return mu * torch.logsumexp((x - ref) / mu, dim=-1)

    def fn_logadd(x, ref, w, mu):
        return mu * torch.logsumexp((x - ref) / mu + torch.log(w), dim=-1)

    g_broken = get_grad(fn_broken)
    g_lin = get_grad(fn_lin)
    g_logadd = get_grad(fn_logadd)

    print(f"\n  Gradients:")
    print(f"    {'obj':>5s}  {'inner':>6s}  {'broken':>8s}  {'lin':>8s}  {'logadd':>8s}")
    for i in range(m):
        print(f"    {i:>5d}  {inner[i].item():>6.3f}  {g_broken[i].item():>8.5f}  {g_lin[i].item():>8.5f}  {g_logadd[i].item():>8.5f}")

    print(f"\n  Gradient entropy (higher = more uniform signal):")
    def entropy(g):
        p = torch.abs(g) / torch.abs(g).sum()
        return -(p * torch.log(p + 1e-10)).sum().item()
    print(f"    Broken:  {entropy(g_broken):.4f}")
    print(f"    Lin:     {entropy(g_lin):.4f}")  
    print(f"    Logadd:  {entropy(g_logadd):.4f}")
    
    print(f"\n  Max gradient magnitude:")
    print(f"    Broken:  {g_broken.abs().max().item():.6f}")
    print(f"    Lin:     {g_lin.abs().max().item():.6f}")
    print(f"    Ratio (broken/lin): {(g_broken.abs().max()/g_lin.abs().max()).item():.4f}")
