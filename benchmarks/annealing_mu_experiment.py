"""
Quick experiment: Does STCH-Set with ANNEALED mu perform better?

Schedule: mu_t = 0.1 / log(t + 2)
At t=0: mu = 0.1/log(2) ≈ 0.144 (smoother, more exploration)
At t=19: mu = 0.1/log(21) ≈ 0.033 (tighter, exploiting)

Hypothesis: Annealing recovers exact Chebyshev at convergence while 
keeping smooth gradients early. Could be a practical tip for the paper.

Compares:
  - STCH-Set with fixed mu=0.1
  - STCH-Set with annealed mu_t = 0.1/log(t+2) 
  - STCH-Set with annealed mu_t = 0.05/log(t+2)

On DTLZ2 m=5, K=5, 3 seeds, 20 iterations (quick run).
"""

import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.test_functions.multi_objective import DTLZ2
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from stch_botorch.acquisition.stch_set_bo import qSTCHSet

tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}


def get_initial_data(problem, d, n_init, seed):
    torch.manual_seed(seed)
    train_X = draw_sobol_samples(
        bounds=torch.stack([torch.zeros(d, **tkwargs), torch.ones(d, **tkwargs)]),
        n=n_init, q=1,
    ).squeeze(1)
    train_Y = problem(train_X)
    return train_X, train_Y


def fit_model(train_X, train_Y):
    model = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=train_Y.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def compute_hv(train_Y, ref_point):
    pareto_mask = is_non_dominated(train_Y)
    pareto_Y = train_Y[pareto_mask]
    if pareto_Y.shape[0] == 0:
        return 0.0
    valid = (pareto_Y > ref_point).all(dim=-1)
    pareto_Y = pareto_Y[valid]
    if pareto_Y.shape[0] == 0:
        return 0.0
    try:
        hv = Hypervolume(ref_point=ref_point)
        return hv.compute(pareto_Y)
    except Exception:
        return 0.0


def run_stch_set_with_schedule(problem, d, m, ref_point, n_init, n_iters, K, seed, mu_schedule):
    """Run STCH-Set with a mu schedule function: mu_schedule(iteration) -> mu."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    bounds = torch.stack([torch.zeros(d, **tkwargs), torch.ones(d, **tkwargs)])
    train_X, train_Y = get_initial_data(problem, d, n_init, seed)
    hv_history = [compute_hv(train_Y, ref_point)]
    mu_history = []
    
    for i in range(n_iters):
        t0 = time.time()
        mu_t = mu_schedule(i)
        mu_history.append(mu_t)
        
        try:
            model = fit_model(train_X, train_Y)
            acqf = qSTCHSet(
                model=model,
                ref_point=ref_point,
                mu=mu_t,
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64])),
                maximize=True,
            )
            candidates, _ = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=K,
                num_restarts=4,
                raw_samples=64,
                options={"batch_limit": 5, "maxiter": 50},
            )
            new_Y = problem(candidates)
            train_X = torch.cat([train_X, candidates])
            train_Y = torch.cat([train_Y, new_Y])
        except Exception as e:
            print(f"  iter {i} failed: {e}")
            candidates = torch.rand(K, d, **tkwargs)
            new_Y = problem(candidates)
            train_X = torch.cat([train_X, candidates])
            train_Y = torch.cat([train_Y, new_Y])
        
        hv = compute_hv(train_Y, ref_point)
        hv_history.append(hv)
        dt = time.time() - t0
        print(f"  iter {i+1}/{n_iters}: mu={mu_t:.4f}, HV={hv:.4f}, time={dt:.1f}s", flush=True)
    
    return hv_history, mu_history


def main():
    m = 5
    d = m + 1  # DTLZ2: d = m + k - 1, k=2
    K = m
    n_init = 2 * (d + 1)
    n_iters = 20
    seeds = [0, 1, 2]
    
    problem = DTLZ2(num_objectives=m, dim=d, negate=True)
    ref_point = torch.full((m,), -1.5, **tkwargs)
    
    schedules = {
        "fixed_0.1": lambda t: 0.1,
        "anneal_0.1": lambda t: 0.1 / math.log(t + 2),
        "anneal_0.05": lambda t: 0.05 / math.log(t + 2),
        "fixed_0.01": lambda t: 0.01,
    }
    
    results = {}
    
    for name, schedule in schedules.items():
        print(f"\n{'='*60}")
        print(f"Schedule: {name}")
        print(f"{'='*60}")
        
        all_hv = []
        all_mu = []
        
        for seed in seeds:
            print(f"\n  Seed {seed}")
            hv_hist, mu_hist = run_stch_set_with_schedule(
                problem, d, m, ref_point, n_init, n_iters, K, seed, schedule
            )
            all_hv.append(hv_hist)
            all_mu.append(mu_hist)
        
        final_hvs = [h[-1] for h in all_hv]
        results[name] = {
            "hv_histories": all_hv,
            "mu_histories": all_mu,
            "final_hvs": final_hvs,
            "mean_hv": float(np.mean(final_hvs)),
            "std_hv": float(np.std(final_hvs)),
        }
        
        print(f"\n  {name}: {np.mean(final_hvs):.4f} ± {np.std(final_hvs):.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Annealing mu experiment (DTLZ2, m=5, K=5, 3 seeds)")
    print("="*60)
    for name, r in results.items():
        print(f"  {name:20s}: {r['mean_hv']:.4f} ± {r['std_hv']:.4f}")
    
    # Save
    out_path = Path(__file__).parent / "results" / "annealing_mu_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
