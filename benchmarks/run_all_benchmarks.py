"""
Full DTLZ2 benchmark suite: m=5,8,10 with multiple seeds.
Runs directly, no sub-agents, no conda run.
Saves results incrementally to JSON files.
"""
import sys, os, time, json, traceback
from pathlib import Path
from datetime import datetime

# Setup paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
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
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization

from stch_botorch.acquisition.stch_set_bo import qSTCHSet
from stch_botorch.scalarization import smooth_chebyshev

import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR = ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}


def compute_hv(Y, ref_point):
    """Compute hypervolume. Y in BoTorch maximization convention."""
    pareto_mask = is_non_dominated(Y)
    pareto_Y = Y[pareto_mask]
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


def fit_gp(X, Y, m):
    model = SingleTaskGP(X, Y, outcome_transform=Standardize(m=m))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def run_stch_set(problem, d, m, ref_point, bounds, n_init, n_iters, q, seed, mu=0.1):
    """STCH-Set-BO: our method. q=K candidates jointly optimized."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = draw_sobol_samples(bounds=bounds, n=n_init, q=1).squeeze(1)
    Y = problem(X)
    hv_hist = [compute_hv(Y, ref_point)]
    
    mc = 64 if m <= 5 else 32
    nr = 4 if m <= 5 else 3
    rs = 64 if m <= 5 else 32
    mi = 50 if m <= 5 else 30
    
    for i in range(n_iters):
        t0 = time.time()
        try:
            model = fit_gp(X, Y, m)
            acqf = qSTCHSet(
                model=model, ref_point=ref_point, mu=mu, maximize=True,
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([mc])),
            )
            cands, _ = optimize_acqf(
                acq_function=acqf, bounds=bounds, q=q,
                num_restarts=nr, raw_samples=rs,
                options={"batch_limit": 5, "maxiter": mi},
            )
            new_Y = problem(cands)
            X = torch.cat([X, cands])
            Y = torch.cat([Y, new_Y])
        except Exception as e:
            print(f"    STCH-Set iter {i} error: {e}", flush=True)
            cands = torch.rand(q, d, **tkwargs)
            new_Y = problem(cands)
            X = torch.cat([X, cands])
            Y = torch.cat([Y, new_Y])
        
        hv = compute_hv(Y, ref_point)
        hv_hist.append(hv)
        print(f"    STCH-Set(q={q}) iter {i+1}/{n_iters}: HV={hv:.6f} ({time.time()-t0:.1f}s)", flush=True)
    
    return hv_hist


def run_qnparego(problem, d, m, ref_point, bounds, n_init, n_iters, q, seed):
    """qNParEGO baseline: random Chebyshev scalarization each iter."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = draw_sobol_samples(bounds=bounds, n=n_init, q=1).squeeze(1)
    Y = problem(X)
    hv_hist = [compute_hv(Y, ref_point)]
    
    mc = 64 if m <= 5 else 32
    nr = 4 if m <= 5 else 3
    rs = 64 if m <= 5 else 32
    mi = 50 if m <= 5 else 30
    
    for i in range(n_iters):
        t0 = time.time()
        try:
            model = fit_gp(X, Y, m)
            weights = torch.distributions.Dirichlet(torch.ones(m)).sample()
            scalarization = get_chebyshev_scalarization(weights=weights, Y=Y)
            mc_obj = GenericMCObjective(scalarization)
            
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc]))
            acqf = qLogNoisyExpectedImprovement(
                model=model, X_baseline=X, sampler=sampler,
                objective=mc_obj, prune_baseline=True,
            )
            cands, _ = optimize_acqf(
                acq_function=acqf, bounds=bounds, q=q,
                num_restarts=nr, raw_samples=rs,
                options={"batch_limit": 5, "maxiter": mi},
            )
            new_Y = problem(cands)
            X = torch.cat([X, cands])
            Y = torch.cat([Y, new_Y])
        except Exception as e:
            print(f"    qNParEGO iter {i} error: {e}", flush=True)
            cands = torch.rand(q, d, **tkwargs)
            new_Y = problem(cands)
            X = torch.cat([X, cands])
            Y = torch.cat([Y, new_Y])
        
        hv = compute_hv(Y, ref_point)
        hv_hist.append(hv)
        print(f"    qNParEGO(q={q}) iter {i+1}/{n_iters}: HV={hv:.6f} ({time.time()-t0:.1f}s)", flush=True)
    
    return hv_hist


def run_stch_nparego(problem, d, m, ref_point, bounds, n_init, n_iters, q, seed, mu=0.1):
    """STCH-NParEGO: single-point STCH scalarization (not set-based)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = draw_sobol_samples(bounds=bounds, n=n_init, q=1).squeeze(1)
    Y = problem(X)
    hv_hist = [compute_hv(Y, ref_point)]
    
    mc = 64 if m <= 5 else 32
    nr = 4 if m <= 5 else 3
    rs = 64 if m <= 5 else 32
    mi = 50 if m <= 5 else 30
    
    for i in range(n_iters):
        t0 = time.time()
        try:
            model = fit_gp(X, Y, m)
            weights = torch.distributions.Dirichlet(torch.ones(m)).sample()
            # Use negated ref for maximization convention
            neg_ref = -ref_point
            
            def make_obj(w, rp, mu_val):
                def fn(samples, X=None):
                    # samples in max convention, negate for smooth_chebyshev (min convention)
                    return smooth_chebyshev(-samples, w, rp, mu_val)
                return fn
            
            # ref_point for STCH is in minimization convention
            min_ref = Y.min(dim=0).values  # best observed (most negative = best in max convention)
            min_ref_neg = -min_ref  # convert to min convention
            
            mc_obj = GenericMCObjective(make_obj(weights, min_ref_neg, mu))
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc]))
            acqf = qLogNoisyExpectedImprovement(
                model=model, X_baseline=X, sampler=sampler,
                objective=mc_obj, prune_baseline=True,
            )
            cands, _ = optimize_acqf(
                acq_function=acqf, bounds=bounds, q=q,
                num_restarts=nr, raw_samples=rs,
                options={"batch_limit": 5, "maxiter": mi},
            )
            new_Y = problem(cands)
            X = torch.cat([X, cands])
            Y = torch.cat([Y, new_Y])
        except Exception as e:
            print(f"    STCH-NParEGO iter {i} error: {e}", flush=True)
            cands = torch.rand(q, d, **tkwargs)
            new_Y = problem(cands)
            X = torch.cat([X, cands])
            Y = torch.cat([Y, new_Y])
        
        hv = compute_hv(Y, ref_point)
        hv_hist.append(hv)
        print(f"    STCH-NParEGO(q={q}) iter {i+1}/{n_iters}: HV={hv:.6f} ({time.time()-t0:.1f}s)", flush=True)
    
    return hv_hist


def run_random(problem, d, m, ref_point, bounds, n_init, n_iters, q, seed):
    """Random baseline."""
    torch.manual_seed(seed)
    X = draw_sobol_samples(bounds=bounds, n=n_init, q=1).squeeze(1)
    Y = problem(X)
    hv_hist = [compute_hv(Y, ref_point)]
    
    for i in range(n_iters):
        cands = torch.rand(q, d, **tkwargs)
        new_Y = problem(cands)
        X = torch.cat([X, cands])
        Y = torch.cat([Y, new_Y])
        hv = compute_hv(Y, ref_point)
        hv_hist.append(hv)
    
    return hv_hist


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIGS = [
    {"m": 5,  "n_init": 20, "n_iters": 30, "seeds": [0,1,2,3,4], "K": 5},
    {"m": 8,  "n_init": 30, "n_iters": 25, "seeds": [0,1,2],     "K": 8},
    {"m": 10, "n_init": 40, "n_iters": 20, "seeds": [0,1,2],     "K": 10},
]

def main():
    start_time = time.time()
    all_results = {}
    
    for cfg in CONFIGS:
        m = cfg["m"]
        d = m + 1  # DTLZ2: d = m + k - 1, k=2
        K = cfg["K"]
        n_init = cfg["n_init"]
        n_iters = cfg["n_iters"]
        seeds = cfg["seeds"]
        
        problem = DTLZ2(num_objectives=m, dim=d, negate=True)
        ref_point = torch.full((m,), -1.5, **tkwargs)
        bounds = torch.stack([torch.zeros(d, **tkwargs), torch.ones(d, **tkwargs)])
        
        print(f"\n{'='*70}", flush=True)
        print(f"  DTLZ2 m={m}, d={d}, K={K}, n_init={n_init}, n_iters={n_iters}, seeds={seeds}", flush=True)
        print(f"{'='*70}", flush=True)
        
        results = {
            "config": cfg,
            "methods": {}
        }
        
        for method_name, run_fn, q in [
            ("STCH-Set", lambda *a, **kw: run_stch_set(*a, **kw), K),
            ("STCH-NParEGO", lambda *a, **kw: run_stch_nparego(*a, **kw), 1),
            ("qNParEGO", lambda *a, **kw: run_qnparego(*a, **kw), 1),
            ("Random", lambda *a, **kw: run_random(*a, **kw), 1),
        ]:
            print(f"\n  --- {method_name} (q={q}) ---", flush=True)
            method_results = []
            
            for seed in seeds:
                print(f"  Seed {seed}:", flush=True)
                t0 = time.time()
                try:
                    hv_hist = run_fn(problem, d, m, ref_point, bounds, n_init, n_iters, q, seed)
                    method_results.append({
                        "seed": seed,
                        "hv_history": [float(h) for h in hv_hist],
                        "final_hv": float(hv_hist[-1]),
                        "time": time.time() - t0,
                        "error": None,
                    })
                    print(f"    Final HV={hv_hist[-1]:.6f} in {time.time()-t0:.0f}s", flush=True)
                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"    FAILED: {e}\n{tb}", flush=True)
                    method_results.append({
                        "seed": seed,
                        "hv_history": [],
                        "final_hv": 0.0,
                        "time": time.time() - t0,
                        "error": str(e),
                    })
            
            final_hvs = [r["final_hv"] for r in method_results if r["error"] is None]
            results["methods"][method_name] = {
                "q": q,
                "runs": method_results,
                "mean_hv": float(np.mean(final_hvs)) if final_hvs else 0.0,
                "std_hv": float(np.std(final_hvs)) if final_hvs else 0.0,
            }
            
            if final_hvs:
                print(f"  {method_name}: HV = {np.mean(final_hvs):.6f} +/- {np.std(final_hvs):.6f}", flush=True)
        
        # Save incrementally
        outfile = RESULTS_DIR / f"dtlz2_m{m}_results.json"
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved: {outfile}", flush=True)
        
        all_results[f"m={m}"] = {
            method: f"{data['mean_hv']:.6f} +/- {data['std_hv']:.6f}"
            for method, data in results["methods"].items()
        }
    
    # Final summary
    elapsed = time.time() - start_time
    print(f"\n{'='*70}", flush=True)
    print(f"  ALL DONE in {elapsed/3600:.1f}h", flush=True)
    print(f"{'='*70}", flush=True)
    for key, methods in all_results.items():
        print(f"\n  {key}:", flush=True)
        for method, hv_str in methods.items():
            print(f"    {method:20s} HV = {hv_str}", flush=True)
    
    # Save combined summary
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump({"results": all_results, "elapsed_hours": elapsed/3600, "timestamp": datetime.now().isoformat()}, f, indent=2)


if __name__ == "__main__":
    main()
