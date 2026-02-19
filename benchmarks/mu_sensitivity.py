"""
Mu Sensitivity Analysis for STCH-Set.

Runs qSTCH-Set on DTLZ2 m=5 with μ ∈ {0.01, 0.05, 0.1, 0.5, 1.0}.
"""

import argparse
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from multiprocessing import Pool

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import runner from existing benchmark
# Ensure benchmarks dir is in path or import relatively
sys.path.insert(0, str(Path(__file__).parent))
from dtlz_benchmark_v2 import run_stch_set_bo, get_problem, tkwargs

def worker_mu_run(kwargs):
    mu = kwargs["mu"]
    seed = kwargs["seed"]
    m = kwargs["m"]
    d = kwargs["d"]
    n_iters = kwargs["n_iters"]
    device = kwargs.get("device", "cpu")
    
    # Re-instantiate problem
    problem, d, ref_point = get_problem("DTLZ2", m, device=device)
    
    # Run
    return run_stch_set_bo(
        problem, d, m, ref_point, 
        n_init=2*(d+1), 
        n_iters=n_iters, 
        q=kwargs["K"], 
        seed=seed, 
        mu=mu,
        warm_start=True,
        device=device
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    mu_values = [0.01, 0.05, 0.1, 0.5, 1.0]
    
    # Dummy call for dims
    _, d, _ = get_problem("DTLZ2", args.m, device="cpu")
    
    results = {}
    
    print(f"Running Mu Sensitivity: m={args.m}, K={args.K}, seeds={args.seeds}")
    
    for mu in mu_values:
        print(f"\nEvaluating mu={mu}")
        tasks = []
        for s in range(args.seeds):
            tasks.append({
                "mu": mu,
                "seed": s,
                "m": args.m,
                "d": d,
                "n_iters": args.iters,
                "K": args.K,
                "device": args.device
            })
            
        with Pool(processes=min(args.seeds, 4)) as pool:
            # Returns list of (hv_history, times, errors)
            outputs = pool.map(worker_mu_run, tasks)
            
        # Aggregate
        hv_histories = np.array([out[0] for out in outputs])
        results[mu] = {
            "mean": hv_histories.mean(axis=0).tolist(),
            "std": hv_histories.std(axis=0).tolist(),
            "final_mean": float(hv_histories[:, -1].mean()),
            "final_std": float(hv_histories[:, -1].std())
        }
        print(f"  Final HV: {results[mu]['final_mean']:.4f} ± {results[mu]['final_std']:.4f}")

    # Save
    out_dir = Path("results/mu_sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "mu_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Plot
    plt.figure(figsize=(10, 6))
    for mu, res in results.items():
        mean = np.array(res["mean"])
        std = np.array(res["std"])
        iters = np.arange(len(mean))
        plt.plot(iters, mean, label=f"mu={mu}")
        plt.fill_between(iters, mean-std, mean+std, alpha=0.1)
        
    plt.xlabel("Iteration")
    plt.ylabel("Hypervolume")
    plt.title(f"Mu Sensitivity (DTLZ2 m={args.m}, K={args.K})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "mu_sensitivity_plot.png")
    print(f"\nSaved results and plot to {out_dir}")
    
    # Find best
    best_mu = max(results.items(), key=lambda x: x[1]['final_mean'])
    print(f"Best mu: {best_mu[0]} (HV={best_mu[1]['final_mean']:.4f})")

if __name__ == "__main__":
    main()
