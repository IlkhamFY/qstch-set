#!/usr/bin/env python3
"""mu ablation: STCH-Set on DTLZ2 m=5 with mu in {0.01, 0.05, 0.1, 0.5, 1.0}."""
import sys, os, time, json, traceback
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
torch.set_default_dtype(torch.double)

from botorch.test_functions.multi_objective import DTLZ2
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples
from stch_botorch.acquisition.stch_set_bo import qSTCHSet
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR = ROOT / "benchmarks" / "results"
tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}

def compute_hv(Y, ref_point):
    pareto_mask = is_non_dominated(Y)
    pareto_Y = Y[pareto_mask]
    if pareto_Y.shape[0] == 0: return 0.0
    valid = (pareto_Y > ref_point).all(dim=-1)
    pareto_Y = pareto_Y[valid]
    if pareto_Y.shape[0] == 0: return 0.0
    try: return Hypervolume(ref_point=ref_point).compute(pareto_Y)
    except: return 0.0

def run(m=5, K=5, n_init=20, n_iters=20, seeds=[0,1,2], mus=[0.01, 0.05, 0.1, 0.5, 1.0]):
    d = m + 1
    problem = DTLZ2(num_objectives=m, dim=d, negate=True)
    ref_point = torch.full((m,), -1.5, **tkwargs)
    bounds = torch.stack([torch.zeros(d, **tkwargs), torch.ones(d, **tkwargs)])
    
    results = {}
    for mu in mus:
        print(f"\n--- mu={mu} ---", flush=True)
        seed_results = []
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            X = draw_sobol_samples(bounds=bounds, n=n_init, q=1).squeeze(1)
            Y = problem(X)
            hv_hist = [compute_hv(Y, ref_point)]
            
            for i in range(n_iters):
                t0 = time.time()
                try:
                    model = SingleTaskGP(X, Y, outcome_transform=Standardize(m=m))
                    mll = ExactMarginalLogLikelihood(model.likelihood, model)
                    fit_gpytorch_mll(mll)
                    acqf = qSTCHSet(model=model, ref_point=ref_point, mu=mu, maximize=True,
                        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64])))
                    cands, _ = optimize_acqf(acqf, bounds=bounds, q=K, num_restarts=4, raw_samples=64,
                        options={"batch_limit": 5, "maxiter": 50})
                    X = torch.cat([X, cands]); Y = torch.cat([Y, problem(cands)])
                except Exception as e:
                    print(f"  mu={mu} seed={seed} iter {i} error: {e}", flush=True)
                    cands = torch.rand(K, d, **tkwargs)
                    X = torch.cat([X, cands]); Y = torch.cat([Y, problem(cands)])
                hv = compute_hv(Y, ref_point)
                hv_hist.append(hv)
                print(f"  mu={mu} seed={seed} iter {i+1}/{n_iters}: HV={hv:.6f} ({time.time()-t0:.1f}s)", flush=True)
            
            seed_results.append({"seed": seed, "final_hv": float(hv_hist[-1]), "hv_history": [float(h) for h in hv_hist]})
        
        hvs = [r["final_hv"] for r in seed_results]
        results[f"mu={mu}"] = {"mean_hv": float(np.mean(hvs)), "std_hv": float(np.std(hvs)), "runs": seed_results}
        print(f"  mu={mu}: HV = {np.mean(hvs):.6f} +/- {np.std(hvs):.6f}", flush=True)
    
    with open(RESULTS_DIR / "ablation_mu_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'ablation_mu_results.json'}")

if __name__ == "__main__":
    run()
