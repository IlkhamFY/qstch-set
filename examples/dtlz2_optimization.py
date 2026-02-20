"""
Benchmark example: Optimize DTLZ2 (5 objectives) using qSTCH-Set.

Run: python examples/dtlz2_optimization.py
"""
import matplotlib.pyplot as plt
import torch
from botorch.test_functions.multi_objective import DTLZ2
from stch_botorch import run_mobo

def main():
    print("Running DTLZ2 (m=5) optimization...")
    
    # 1. Setup Problem
    n_obj = 5
    dim = 14  # standard dim for DTLZ2(m=5) is usually m + k - 1
    # BoTorch defaults? DTLZ2(dim=..., num_objectives=...)
    # Standard: k=10. dim = m + k - 1 = 5 + 10 - 1 = 14.
    
    problem = DTLZ2(dim=dim, num_objectives=n_obj, negate=True) # negate=True for maximization
    
    # Bounds: [0, 1]^d
    bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
    
    # 2. Run Optimization
    # Budget 100, Batch 5
    results = run_mobo(
        f=problem, 
        bounds=bounds, 
        n_obj=n_obj, 
        budget=100, 
        batch_size=5, 
        seed=123, 
        verbose=True
    )
    
    # 3. Analyze Results
    print("\nOptimization Complete!")
    print(f"Final Hypervolume: {results.hypervolume_history[-1]:.4f}")
    print(f"Pareto Front Size: {len(results.pareto_Y)}")
    
    # Reference value for DTLZ2(m=5)?
    # Max HV is roughly volume of unit sphere sector?
    # Actually depends heavily on reference point used for HV calculation.
    # We used adaptive ref point in run_mobo, so absolute value might vary.
    
    # 4. Plot Convergence
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(results.hypervolume_history)), results.hypervolume_history, marker='o')
    plt.title(f"DTLZ2 (m={n_obj}) Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Hypervolume (adaptive ref point)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("examples/dtlz2_convergence.png")
    print("Convergence plot saved to examples/dtlz2_convergence.png")
    
    # 5. Print Comparison Table
    print("\n--- Final Stats ---")
    print(f"{'Metric':<20} | {'Value':<10}")
    print("-" * 35)
    print(f"{'Hypervolume':<20} | {results.hypervolume_history[-1]:.4f}")
    print(f"{'Pareto Points':<20} | {len(results.pareto_Y)}")
    print(f"{'Total Evals':<20} | {results.n_evals}")

if __name__ == "__main__":
    main()
