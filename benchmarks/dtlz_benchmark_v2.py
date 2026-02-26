"""
DTLZ Benchmark Suite v2 for STCH-Set-BO.

Fixes from v1:
  - qNParEGO: wrap get_chebyshev_scalarization in GenericMCObjective
  - STCH-Set: test with q=K (the SET size), not q=1
  - K ablation: K={3,5,10}
  - mu ablation: mu={0.01, 0.1, 0.5, 1.0}
  - Increased num_restarts/raw_samples for high-m
  - 5 seeds minimum for statistical significance
  - Detailed JSON output with per-seed, per-iteration data
  - Parallel execution for seeds
  - Warm-starting for STCH-Set

Usage:
    python dtlz_benchmark_v2.py --problem DTLZ2 --m 5 --seeds 5 --iters 30
    python dtlz_benchmark_v2.py --problem DTLZ2 --m 5 --seeds 5 --ablation k
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count, current_process

import numpy as np
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.objective import GenericMCObjective
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.test_functions.multi_objective import DTLZ2
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

warnings.filterwarnings("ignore")

# Adjust path to include src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from stch_botorch.acquisition.stch_set_bo import qSTCHSet, qSTCHSetPure
from stch_botorch.scalarization import smooth_chebyshev

# Global default, but will be set per process
tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}


# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------

def get_problem(name: str, m: int, device=None):
    """Get test problem, input dim, and reference point."""
    if device is None:
        device = tkwargs["device"]
    kwargs = {"dtype": torch.double, "device": device}
    
    if name == "DTLZ2":
        d = m + 1  # d = m + k - 1, k=2
        problem = DTLZ2(num_objectives=m, dim=d, negate=True)
        problem.to(**kwargs)
        ref_point = torch.full((m,), -1.5, **kwargs)
    elif name == "ZDT2":
        assert m == 2, "ZDT2 is bi-objective"
        from botorch.test_functions.multi_objective import ZDT2 as ZDT2Func
        d = 30
        problem = ZDT2Func(num_objectives=2, dim=d, negate=True)
        problem.to(**kwargs)
        ref_point = torch.tensor([-11.0, -11.0], **kwargs)
    elif name == "ZDT3":
        assert m == 2, "ZDT3 is bi-objective"
        from botorch.test_functions.multi_objective import ZDT3 as ZDT3Func
        d = 30
        problem = ZDT3Func(num_objectives=2, dim=d, negate=True)
        problem.to(**kwargs)
        ref_point = torch.tensor([-11.0, -11.0], **kwargs)
    else:
        raise ValueError(f"Unknown problem: {name}")
    return problem, d, ref_point


def get_opt_params(m: int):
    """Get optimization hyperparameters scaled to objective count."""
    if m <= 3:
        return dict(num_restarts=10, raw_samples=256, mc_samples=128, maxiter=100)
    elif m <= 5:
        return dict(num_restarts=4, raw_samples=64, mc_samples=64, maxiter=50)
    elif m <= 8:
        return dict(num_restarts=6, raw_samples=128, mc_samples=64, maxiter=80)
    else:
        return dict(num_restarts=4, raw_samples=64, mc_samples=32, maxiter=30)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def get_initial_data(problem, d, n_init, seed, device):
    """Generate initial Sobol samples with a specific seed."""
    kwargs = {"dtype": torch.double, "device": device}
    # Temporarily set CPU for sampling if needed, or just direct
    # draw_sobol_samples handles device
    bounds = torch.stack([torch.zeros(d, **kwargs), torch.ones(d, **kwargs)])
    
    # Seeding
    # Note: torch.manual_seed sets the seed for the current device.
    torch.manual_seed(seed)
    
    train_X = draw_sobol_samples(
        bounds=bounds,
        n=n_init,
        q=1,
        seed=seed 
    ).squeeze(1)
    train_Y = problem(train_X)
    return train_X, train_Y


def fit_model(train_X, train_Y):
    """Fit a multi-output GP."""
    model = SingleTaskGP(
        train_X, train_Y,
        outcome_transform=Standardize(m=train_Y.shape[-1]),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def compute_hv(train_Y, ref_point):
    """Compute hypervolume of current Pareto front."""
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


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------

def run_stch_set_bo(problem, d, m, ref_point, n_init, n_iters, q, seed, mu=0.1, warm_start=True, device=None):
    """Run STCH-Set-BO with q candidates optimized jointly."""
    if device is None: device = tkwargs["device"]
    kwargs = {"dtype": torch.double, "device": device}
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = get_opt_params(m)

    bounds = torch.stack([torch.zeros(d, **kwargs), torch.ones(d, **kwargs)])
    train_X, train_Y = get_initial_data(problem, d, n_init, seed, device)
    hv_history = [compute_hv(train_Y, ref_point)]
    times = []
    errors = []
    
    # Warm-start state
    prev_candidates = None

    for i in range(n_iters):
        t0 = time.time()
        try:
            model = fit_model(train_X, train_Y)

            # Normalize using posterior mean predictions — mirrors
            # get_chebyshev_scalarization(weights, Y=pred) pattern.
            with torch.no_grad():
                pred = model.posterior(train_X).mean
            Y_bounds = torch.stack([pred.min(dim=0).values, pred.max(dim=0).values])

            acqf = qSTCHSet(
                model=model,
                ref_point=ref_point,
                mu=mu,
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([opt["mc_samples"]])),
                Y_bounds=Y_bounds,
                maximize=True,
            )

            # Prepare initial conditions for warm-start
            batch_initial_conditions = None
            if warm_start and prev_candidates is not None:
                # optimize_acqf expects batch_initial_conditions of shape (num_restarts, q, d)
                # We use prev_candidates (q, d) as one restart, and generate others randomly
                num_restarts = opt["num_restarts"]
                if num_restarts > 1:
                    # Generate random restarts
                    raw_random = torch.rand(num_restarts - 1, q, d, **kwargs)
                    # Scale to bounds
                    raw_random = bounds[0] + (bounds[1] - bounds[0]) * raw_random
                    # Prepend previous solution
                    batch_initial_conditions = torch.cat([
                        prev_candidates.unsqueeze(0),
                        raw_random
                    ], dim=0)
                else:
                    batch_initial_conditions = prev_candidates.unsqueeze(0)

            candidates, acq_value = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=q,  # This is K, the set size
                num_restarts=opt["num_restarts"],
                raw_samples=opt["raw_samples"],
                options={"batch_limit": 5, "maxiter": opt["maxiter"]},
                batch_initial_conditions=batch_initial_conditions
            )

            # Store for next iteration
            prev_candidates = candidates.detach().clone()

            new_Y = problem(candidates)
            train_X = torch.cat([train_X, candidates])
            train_Y = torch.cat([train_Y, new_Y])
        except Exception as e:
            errors.append({"iter": i, "error": str(e)})
            # print(f"  STCH-Set iter {i} failed: {e}")
            candidates = torch.rand(q, d, **kwargs)
            new_Y = problem(candidates)
            train_X = torch.cat([train_X, candidates])
            train_Y = torch.cat([train_Y, new_Y])
            prev_candidates = None # Reset warm start on failure

        t1 = time.time()
        hv = compute_hv(train_Y, ref_point)
        hv_history.append(hv)
        times.append(t1 - t0)

        # print(f"  STCH-Set(q={q},mu={mu}) iter {i+1}/{n_iters}: HV={hv:.4f}, time={t1-t0:.1f}s", flush=True)

    return hv_history, times, errors


def run_stch_set_pure(problem, d, m, ref_point, n_init, n_iters, q, seed, mu=0.1, device=None):
    """Run qSTCHSetPure — Lin et al.'s exact formula, no weights, no ref point.

    Used as a diagnostic baseline to verify that qSTCHSet (with uniform weights
    and ref_point) behaves equivalently. If they produce similar results, it
    confirms our additions don't meaningfully change the optimization landscape.
    """
    if device is None: device = tkwargs["device"]
    kwargs = {"dtype": torch.double, "device": device}

    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = get_opt_params(m)

    bounds = torch.stack([torch.zeros(d, **kwargs), torch.ones(d, **kwargs)])
    train_X, train_Y = get_initial_data(problem, d, n_init, seed, device)
    hv_history = [compute_hv(train_Y, ref_point)]
    times = []
    errors = []
    prev_candidates = None

    for i in range(n_iters):
        t0 = time.time()
        try:
            model = fit_model(train_X, train_Y)

            acqf = qSTCHSetPure(
                model=model,
                mu=mu,
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([opt["mc_samples"]])),
                maximize=True,
            )

            batch_initial_conditions = None
            if prev_candidates is not None:
                num_restarts = opt["num_restarts"]
                if num_restarts > 1:
                    raw_random = torch.rand(num_restarts - 1, q, d, **kwargs)
                    raw_random = bounds[0] + (bounds[1] - bounds[0]) * raw_random
                    batch_initial_conditions = torch.cat([prev_candidates.unsqueeze(0), raw_random], dim=0)
                else:
                    batch_initial_conditions = prev_candidates.unsqueeze(0)

            candidates, _ = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=q,
                num_restarts=opt["num_restarts"],
                raw_samples=opt["raw_samples"],
                options={"batch_limit": 5, "maxiter": opt["maxiter"]},
                batch_initial_conditions=batch_initial_conditions,
            )

            prev_candidates = candidates.detach().clone()
            new_Y = problem(candidates)
            train_X = torch.cat([train_X, candidates])
            train_Y = torch.cat([train_Y, new_Y])
        except Exception as e:
            errors.append({"iter": i, "error": str(e)})
            candidates = torch.rand(q, d, **kwargs)
            new_Y = problem(candidates)
            train_X = torch.cat([train_X, candidates])
            train_Y = torch.cat([train_Y, new_Y])
            prev_candidates = None

        t1 = time.time()
        hv = compute_hv(train_Y, ref_point)
        hv_history.append(hv)
        times.append(t1 - t0)

    return hv_history, times, errors


def run_qnparego(problem, d, m, ref_point, n_init, n_iters, q, seed, device=None):
    if device is None: device = tkwargs["device"]
    kwargs = {"dtype": torch.double, "device": device}
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = get_opt_params(m)

    bounds = torch.stack([torch.zeros(d, **kwargs), torch.ones(d, **kwargs)])
    train_X, train_Y = get_initial_data(problem, d, n_init, seed, device)
    hv_history = [compute_hv(train_Y, ref_point)]
    times = []
    errors = []

    for i in range(n_iters):
        t0 = time.time()
        try:
            model = fit_model(train_X, train_Y)

            weights = torch.rand(m, **kwargs)
            weights = weights / weights.sum()

            chebyshev_fn = get_chebyshev_scalarization(weights=weights, Y=train_Y)
            objective = GenericMCObjective(chebyshev_fn)

            acqf = qLogNoisyExpectedImprovement(
                model=model,
                X_baseline=train_X,
                objective=objective,
                prune_baseline=True,
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([opt["mc_samples"]])),
            )

            candidates, acq_value = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=q,
                num_restarts=opt["num_restarts"],
                raw_samples=opt["raw_samples"],
                options={"batch_limit": 5, "maxiter": opt["maxiter"]},
            )

            new_Y = problem(candidates)
            train_X = torch.cat([train_X, candidates])
            train_Y = torch.cat([train_Y, new_Y])
        except Exception as e:
            errors.append({"iter": i, "error": str(e)})
            candidates = torch.rand(q, d, **kwargs)
            new_Y = problem(candidates)
            train_X = torch.cat([train_X, candidates])
            train_Y = torch.cat([train_Y, new_Y])

        t1 = time.time()
        hv = compute_hv(train_Y, ref_point)
        hv_history.append(hv)
        times.append(t1 - t0)

    return hv_history, times, errors


def run_stch_nparego(problem, d, m, ref_point, n_init, n_iters, q, seed, mu=0.1, device=None):
    if device is None: device = tkwargs["device"]
    kwargs = {"dtype": torch.double, "device": device}
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = get_opt_params(m)

    bounds = torch.stack([torch.zeros(d, **kwargs), torch.ones(d, **kwargs)])
    train_X, train_Y = get_initial_data(problem, d, n_init, seed, device)
    hv_history = [compute_hv(train_Y, ref_point)]
    times = []
    errors = []

    for i in range(n_iters):
        t0 = time.time()
        try:
            model = fit_model(train_X, train_Y)

            weights = torch.rand(m, **kwargs)
            weights = weights / weights.sum()

            def make_stch_objective(w, rp, mu_val):
                def obj_fn(samples, X=None):
                    return smooth_chebyshev(Y=samples, weights=w, ref_point=rp, mu=mu_val)
                return obj_fn

            objective = GenericMCObjective(make_stch_objective(weights, ref_point, mu))

            acqf = qLogNoisyExpectedImprovement(
                model=model,
                X_baseline=train_X,
                objective=objective,
                prune_baseline=True,
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([opt["mc_samples"]])),
            )

            candidates, acq_value = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=q,
                num_restarts=opt["num_restarts"],
                raw_samples=opt["raw_samples"],
                options={"batch_limit": 5, "maxiter": opt["maxiter"]},
            )

            new_Y = problem(candidates)
            train_X = torch.cat([train_X, candidates])
            train_Y = torch.cat([train_Y, new_Y])
        except Exception as e:
            errors.append({"iter": i, "error": str(e)})
            candidates = torch.rand(q, d, **kwargs)
            new_Y = problem(candidates)
            train_X = torch.cat([train_X, candidates])
            train_Y = torch.cat([train_Y, new_Y])

        t1 = time.time()
        hv = compute_hv(train_Y, ref_point)
        hv_history.append(hv)
        times.append(t1 - t0)

    return hv_history, times, errors


def run_qehvi(problem, d, m, ref_point, n_init, n_iters, q, seed, device=None):
    if device is None: device = tkwargs["device"]
    kwargs = {"dtype": torch.double, "device": device}
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = get_opt_params(m)

    bounds = torch.stack([torch.zeros(d, **kwargs), torch.ones(d, **kwargs)])
    train_X, train_Y = get_initial_data(problem, d, n_init, seed, device)
    hv_history = [compute_hv(train_Y, ref_point)]
    times = []
    errors = []

    for i in range(n_iters):
        t0 = time.time()
        try:
            model = fit_model(train_X, train_Y)

            acqf = qLogNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=ref_point.tolist(),
                X_baseline=train_X,
                prune_baseline=True,
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([opt["mc_samples"]])),
            )

            candidates, acq_value = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=q,
                num_restarts=opt["num_restarts"],
                raw_samples=opt["raw_samples"],
                options={"batch_limit": 5, "maxiter": opt["maxiter"]},
            )

            new_Y = problem(candidates)
            train_X = torch.cat([train_X, candidates])
            train_Y = torch.cat([train_Y, new_Y])
        except Exception as e:
            errors.append({"iter": i, "error": str(e)})
            candidates = torch.rand(q, d, **kwargs)
            new_Y = problem(candidates)
            train_X = torch.cat([train_X, candidates])
            train_Y = torch.cat([train_Y, new_Y])

        t1 = time.time()
        hv = compute_hv(train_Y, ref_point)
        hv_history.append(hv)
        times.append(t1 - t0)

    return hv_history, times, errors


def run_random(problem, d, m, ref_point, n_init, n_iters, q, seed, device=None):
    if device is None: device = tkwargs["device"]
    kwargs = {"dtype": torch.double, "device": device}
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_X, train_Y = get_initial_data(problem, d, n_init, seed, device)
    hv_history = [compute_hv(train_Y, ref_point)]
    times = []

    for i in range(n_iters):
        t0 = time.time()
        candidates = torch.rand(q, d, **kwargs)
        new_Y = problem(candidates)
        train_X = torch.cat([train_X, candidates])
        train_Y = torch.cat([train_Y, new_Y])
        t1 = time.time()
        hv_history.append(compute_hv(train_Y, ref_point))
        times.append(t1 - t0)

    return hv_history, times, []

# ---------------------------------------------------------------------------
# Worker Function
# ---------------------------------------------------------------------------

def worker_run_method(kwargs):
    """Worker function to run a single seed of a method."""
    method_name = kwargs["method_name"]
    seed = kwargs["seed"]
    problem_name = kwargs["problem_name"]
    m = kwargs["m"]
    d = kwargs["d"]
    n_init = kwargs["n_init"]
    n_iters = kwargs["n_iters"]
    device_str = kwargs["device"]
    
    # Configure device
    if device_str == "cuda" and torch.cuda.is_available():
        # Simple round-robin or just use default
        # If we have multiple GPUs, we could map process to GPU
        n_gpus = torch.cuda.device_count()
        # process name usually 'SpawnPoolWorker-1', etc.
        # But this is brittle. Just use cuda:0 if not specified.
        # Or just use the default device which will be picked up
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        
    # Re-instantiate problem and ref_point on correct device
    problem, _, ref_point = get_problem(problem_name, m, device=device)
    
    # Run
    if method_name == "stch_set_pure":
        return run_stch_set_pure(
            problem, d, m, ref_point, n_init, n_iters, kwargs["K"], seed, kwargs["mu"], device=device
        )
    elif method_name.startswith("stch_set"):
        return run_stch_set_bo(
            problem, d, m, ref_point, n_init, n_iters, kwargs["K"], seed, kwargs["mu"], device=device
        )
    elif method_name == "stch_nparego":
        return run_stch_nparego(
            problem, d, m, ref_point, n_init, n_iters, kwargs["q"], seed, kwargs["mu"], device=device
        )
    elif method_name == "qnparego":
        return run_qnparego(
            problem, d, m, ref_point, n_init, n_iters, kwargs["q"], seed, device=device
        )
    elif method_name == "qehvi":
        return run_qehvi(
            problem, d, m, ref_point, n_init, n_iters, kwargs["q"], seed, device=device
        )
    elif method_name == "random":
        return run_random(
            problem, d, m, ref_point, n_init, n_iters, kwargs["q"], seed, device=device
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DTLZ Benchmark v2 for STCH-Set-BO")
    parser.add_argument("--problem", default="DTLZ2")
    parser.add_argument("--m", type=int, default=5, help="Number of objectives")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30, help="BO iterations per method")
    parser.add_argument("--q", type=int, default=1, help="Batch size for non-set methods")
    parser.add_argument("--mu", type=float, default=0.1, help="Default STCH mu")
    parser.add_argument("--K", type=int, default=5, help="Default set size for STCH-Set")
    parser.add_argument("--skip-qehvi", action="store_true", help="Skip qEHVI (slow for m>5)")
    parser.add_argument(
        "--ablation", choices=["none", "k", "mu", "both"], default="none",
        help="Run ablation study: k={3,5,10}, mu={0.01,0.1,0.5,1.0}"
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=["stch_set", "stch_set_pure", "stch_nparego", "qnparego", "qehvi", "random"],
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default=None, help="Directory for output (overrides --output)")
    parser.add_argument("--seed-offset", type=int, default=0, help="Offset for seed range (for SLURM array jobs)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for torch tensors")
    parser.add_argument("--n-workers", type=int, default=None, help="Number of parallel workers (defaults to min(seeds, 4))")
    args = parser.parse_args()

    # Device info
    print(f"Requested device: {args.device}")
    if args.device == "cuda" and not torch.cuda.is_available():
         print("WARNING: CUDA requested but not available. Falling back to CPU.")
         args.device = "cpu"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"DTLZ Benchmark v2: {args.problem} m={args.m}, seeds={args.seeds}, "
          f"iters={args.iters}, K={args.K}, mu={args.mu}")
    print(f"Ablation: {args.ablation}, seed_offset={args.seed_offset}")

    # Dummy call to get d
    _, d, _ = get_problem(args.problem, args.m, device=torch.device("cpu"))
    n_init = 2 * (d + 1)
    print(f"Input dim: {d}, Initial points: {n_init}\n")

    # Skip qEHVI for large m
    if (args.m > 6 or args.skip_qehvi) and "qehvi" in args.methods:
        print(f"Skipping qEHVI for m={args.m}")
        args.methods = [m for m in args.methods if m != "qehvi"]

    results = {
        "version": 2,
        "timestamp": timestamp,
        "problem": args.problem,
        "m": args.m,
        "d": d,
        "n_init": n_init,
        "n_iters": args.iters,
        "q_default": args.q,
        "K_default": args.K,
        "mu_default": args.mu,
        "seeds": args.seeds,
        "ablation": args.ablation,
        "methods": {},
    }

    # -----------------------------------------------------------------------
    # Build configurations
    # -----------------------------------------------------------------------
    configs_to_run = [] # List of dicts

    if args.ablation in ("k", "both"):
        for K in [3, 5, 10]:
            configs_to_run.append({
                "method_name": f"stch_set_K{K}",
                "real_method": "stch_set",
                "K": K,
                "mu": args.mu
            })
    
    if args.ablation in ("mu", "both"):
        for mu_val in [0.01, 0.1, 0.5, 1.0]:
            configs_to_run.append({
                "method_name": f"stch_set_mu{mu_val}",
                "real_method": "stch_set",
                "K": args.K,
                "mu": mu_val
            })

    if args.ablation == "none":
        for method_name in args.methods:
            configs_to_run.append({
                "method_name": method_name,
                "real_method": method_name if method_name != "stch_set" else "stch_set", # handle naming
                "K": args.K,
                "mu": args.mu,
                "q": args.q
            })
            
    # Clean up "real_method" which was internal helper
    # The worker expects "method_name" to be passed through for results, 
    # but we need to pass enough info to know what to run.
    # Actually, let's just make the worker kwargs complete.

    # -----------------------------------------------------------------------
    # Run experiments
    # -----------------------------------------------------------------------
    
    n_workers = args.n_workers if args.n_workers is not None else min(args.seeds, 4)
    print(f"Parallel execution with {n_workers} workers.")

    for cfg in configs_to_run:
        display_name = cfg["method_name"]
        print(f"\n{'='*60}")
        print(f"Running {display_name}")
        print(f"{'='*60}")

        # Prepare tasks
        tasks = []
        for seed_idx in range(args.seeds):
            seed = seed_idx + args.seed_offset
            
            task_kwargs = {
                "method_name": cfg.get("real_method", display_name), # e.g. "stch_set"
                "seed": seed,
                "problem_name": args.problem,
                "m": args.m,
                "d": d,
                "n_init": n_init,
                "n_iters": args.iters,
                "device": args.device,
                "K": cfg.get("K", args.K),
                "mu": cfg.get("mu", args.mu),
                "q": cfg.get("q", args.q)
            }
            # Special handling for ablation names mapping back to real methods
            if "stch_set" in display_name and "K" in display_name:
                 task_kwargs["method_name"] = "stch_set"
            if "stch_set" in display_name and "mu" in display_name:
                 task_kwargs["method_name"] = "stch_set"
                 
            tasks.append(task_kwargs)

        all_hv = []
        all_times = []
        all_errors = []

        # Run parallel (or sequential if n_workers==1 to avoid CUDA fork issues)
        if n_workers <= 1:
            results_list = [worker_run_method(t) for t in tasks]
        else:
            with Pool(processes=n_workers) as pool:
                # We map the worker over tasks
                # worker_run_method returns (hv_history, times, errors)
                results_list = pool.map(worker_run_method, tasks)
            
        for res in results_list:
            hv_history, times, errors = res
            all_hv.append(hv_history)
            all_times.append(times)
            all_errors.append(errors)
            # print(f"  Final HV: {hv_history[-1]:.4f}")

        if all_hv:
            hv_array = np.array(all_hv)
            time_array = np.array(all_times)

            results["methods"][display_name] = {
                "hv_mean": hv_array.mean(axis=0).tolist(),
                "hv_std": hv_array.std(axis=0).tolist(),
                "hv_all": hv_array.tolist(),
                "time_mean": time_array.mean(axis=0).tolist(),
                "final_hv_mean": float(hv_array[:, -1].mean()),
                "final_hv_std": float(hv_array[:, -1].std()),
                "mean_iter_time": float(time_array.mean()),
                "n_errors": sum(len(e) for e in all_errors),
                "errors": all_errors,
            }

            print(f"\n  {display_name}: HV={hv_array[:,-1].mean():.4f}"
                  f"±{hv_array[:,-1].std():.4f}, "
                  f"avg time={time_array.mean():.1f}s/iter, "
                  f"errors={sum(len(e) for e in all_errors)}")

    # Save
    if args.output_dir:
        output_path = os.path.join(
            args.output_dir,
            f"{args.problem}_m{args.m}_{args.ablation}_seed{args.seed_offset}_{timestamp}.json"
        )
    else:
        output_path = args.output or (
            f"benchmarks/results/{args.problem}_m{args.m}_{args.ablation}_{timestamp}.json"
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY: {args.problem} m={args.m}")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'Final HV':>12} {'±Std':>10} {'Errors':>8} {'Time/iter':>12}")
    print("-" * 70)
    for name, data in results["methods"].items():
        print(f"{name:<25} {data['final_hv_mean']:>12.4f} "
              f"±{data['final_hv_std']:.4f}  {data['n_errors']:>6} "
              f"{data['mean_iter_time']:>9.1f}s")


if __name__ == "__main__":
    main()
