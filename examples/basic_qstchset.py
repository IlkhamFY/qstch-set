"""
Minimal example: Many-objective BO with qSTCH-Set.

Optimizes a simple 5-objective problem in 2 minutes on CPU.
Run: python examples/basic_qstchset.py
"""
import torch
from stch_botorch import run_mobo

def simple_5obj(X):
    """
    5 conflicting objectives for 6D input.
    Each objective i wants x[i] to be 1, while x[j] (j!=i) contributes penalty.
    """
    # X shape: (n, 6)
    # Objectives: f_i(x) = - (x[i] - 1)^2 - 0.1 * sum_{j!=i} x[j]^2
    # We want to maximize these (minimize distance to 1)
    
    n, d = X.shape
    m = 5
    objs = []
    
    for i in range(m):
        # Term 1: (x[i] - 1)^2
        t1 = (X[:, i] - 1) ** 2
        # Term 2: sum of squares of other vars
        # Create mask for j!=i
        mask = torch.ones(d, dtype=torch.bool, device=X.device)
        mask[i] = False
        t2 = (X[:, mask] ** 2).sum(dim=1)
        
        # Maximize: negate the loss
        obj = -(t1 + 0.1 * t2)
        objs.append(obj)
        
    return torch.stack(objs, dim=-1) # (n, 5)

if __name__ == "__main__":
    bounds = torch.stack([torch.zeros(6), torch.ones(6)])
    print("Running 5-objective optimization...")
    
    results = run_mobo(
        f=simple_5obj, 
        bounds=bounds, 
        n_obj=5, 
        budget=60, 
        seed=42, 
        verbose=True
    )
    
    print(f"\nFinal HV: {results.hypervolume_history[-1]:.4f}")
    print(f"Pareto front size: {results.pareto_Y.shape[0]}")
