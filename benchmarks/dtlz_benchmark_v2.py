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

Usage:
    python dtlz_benchmark_v2.py --problem DTLZ2 --m 5 --seeds 5 --iters 30
    python dtlz_benchmark_v2.py --problem DTLZ2 --m 5 --seeds 5 --ablation k
    python dtlz_benchmark_v2.py --problem DTLZ2 --m 5 --seeds 5 --ablation mu
    python dtlz_benchmark_v2.py --problem DTLZ2 --m 8 --seeds 3 --skip-qehvi
    python dtlz_benchmark_v2.py --problem ZDT2 --m 2 --seeds 3
"""

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from stch_botorch.acquisition.stch_set_bo import qSTCHSet
from stch_botorch.scalarization import smooth_chebyshev

tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}


# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------

def get_problem(name: str, m: int):
    """Get test problem, input dim, and reference point."""
    if name == "DTLZ2":
        d = m + 1  # d = m + k - 1, k=2
        problem = DTLZ2(num_objectives=m, dim=d, negate=True)
        ref_point = torch.full((m,), -1.5, **tkwargs)
    elif name == "ZDT2":
        assert m == 2, "ZDT2 is bi-objective"
        from botorch.test_functions.multi_objective import ZDT2 as ZDT2Func
        d = 30
        problem = ZDT2Func(num_objectives=2, dim=d, negate=True)
        ref_point = torch.tensor([-11.0, -11.0], **tkwargs)
    elif name == "ZDT3":
        assert m == 2, "ZDT3 is bi-objective"
        from botorch.test_functions.multi_objective import ZDT3 as ZDT3Func
        d = 30
        problem = ZDT3Func(num_objectives=2, dim=d, negate=True)
        ref_point = torch.tensor([-11.0, -11.0], **tkwargs)
    else:
        raise ValueError(f"Unknown problem: {name}")
    return problem, d, ref_point


def get_opt_params(m: int):
    """Get optimization hyperparameters scaled to objective count."""
    if m <= 3:
        return dict(num_restarts=10, raw_samples=256, mc_samples=128, maxiter=100)
    elif m <= 5:
        return dict(num_restarts=20, raw_samples=512, mc_samples=256, maxiter=150)
    else:
        return dict(num_restarts=32, raw_samples=1024, mc_samples=256, maxiter=200)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def get_initial_data(problem, d, n_init, seed):
    """Generate initial Sobol samples with a specific seed."""
    torch.manual_seed(seed)
    train_X = draw_sobol_samples(
        bounds=torch.stack([torch.zeros(d, **tkwargs), torch.ones(d, **tkwargs)]),
        n=n_init,
        q=1,
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

def run_stch_set_bo(problem, d, m, ref_point, n_init, n_iters, q, seed, mu=0.1):
    """Run STCH-Set-BO with q candidates optimized jointly.
    
    CRITICAL: q here IS the set size K. The whole point of STCH-Set is
    jointly optimizing K candidates. With q=1 it degenerates to single-point.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = get_opt_params(m)

    bounds = torch.stack([torch.zeros(d, **tkwargs), torch.ones(d, **tkwargs)])
    train_X, train_Y = get_initial_data(problem, d, n_init, seed)
    hv_history = [compute_hv(train_Y, ref_point)]
    times = []
    errors = []

    for i in range(n_iters):
        t0 = time.time()
        try:
            model = fit_model(train_X, train_Y)

            acqf = qSTCHSet(
                model=model,
                ref_point=ref_point,
                mu=mu,
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([opt["mc_samples"]])),
                maximize=True,
            )

            candidates, acq_value = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=q,  # This is K, the set size
                num_restarts=opt["num_restarts"],
                raw_samples=opt["raw_samples"],
                options={"batch_limit": 5, "maxiter": opt["maxiter"]},
            )

            new_Y = problem(candidates)
            train_X = torch.cat([train_X, candidates])
            train_Y = torch.cat([train_Y, new_Y])
        except Exception as e:
            errors.append({"iter": i, "error": str(e)})
            print(f"  STCH-Set iter {i} failed: {e}")
            candidates = torch.rand(q, d, **tkwargs)
            new_Y = problem(candidates)
            train_X = torch.cat([train_X, candidates])
            train_Y = torch.cat([train_Y, new_Y])

        t1 = time.time()
        hv = compute_hv(train_Y, ref_point)
        hv_history.append(hv)
        times.append(t1 - t0)

        if (i + 1) % 5 == 0:
            print(f"  STCH-Set(q={q},mu={mu}) iter {i+1}/{n_iters}: HV={hv:.4f}, time={t1-t0:.1f}s")

    return hv_history, times, errors


def run_qnparego(problem, d, m, ref_point, n_init, n_iters, q, seed):
    """Run qNParEGO with FIXED API: GenericMCObjective wrapper."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = get_opt_params(m)

    bounds = torch.stack([torch.zeros(d, **tkwargs), torch.ones(d, **tkwargs)])
    train_X, train_Y = get_initial_data(problem, d, n_init, seed)
    hv_history = [compute_hv(train_Y, ref_point)]
    times = []
    errors = []

    for i in range(n_iters):
        t0 = time.time()
        try:
            model = fit_model(train_X, train_Y)

            # Random Chebyshev weights
            weights = torch.rand(m, **tkwargs)
            weights = weights / weights.sum()

            # FIX: get_chebyshev_scalarization returns a callable(Y, X=None)
            # but qLogNoisyExpectedImprovement.objective must be an
            # MCAcquisitionObjective. Wrap it in GenericMCObjective.
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
            print(f"  qNParEGO iter {i} failed: {e}")
            candidates = torch.rand(q, d, **tkwargs)
            new_Y = problem(candidates)
            train_X = torch.cat([train_X, candidates])
            train_Y = torch.cat([train_Y, new_Y])

        t1 = time.time()
        hv = compute_hv(train_Y, ref_point)
        hv_history.append(hv)
        times.append(t1 - t0)

        if (i + 1) % 5 == 0:
            print(f"  qNParEGO iter {i+1}/{n_iters}: HV={hv:.4f}, time={t1-t0:.1f}s")

    return hv_history, times, errors


def run_stch_nparego(problem, d, m, ref_point, n_init, n_iters, q, seed, mu=0.1):
    """Run STCH-NParEGO (single-point STCH scalarization + qNEI)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = get_opt_params(m)

    bounds = torch.stack([torch.zeros(d, **tkwargs), torch.ones(d, **tkwargs)])
    train_X, train_Y = get_initial_data(problem, d, n_init, seed)
    hv_history = [compute_hv(train_Y, ref_point)]
    times = []
    errors = []

    for i in range(n_iters):
        t0 = time.time()
        try:
            model = fit_model(train_X, train_Y)

            weights = torch.rand(m, **tkwargs)
            weights = weights / weights.sum()

            def make_stch_objective(w, rp, mu_val):
                """Create a closure to avoid late-binding issues."""
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
            print(f"  STCH-NParEGO iter {i} failed: {e}")
            candidates = torch.rand(q, d, **tkwargs)
            new_Y = problem(candidates)
            train_X = torch.cat([train_X, candidates])
            train_Y = torch.cat([train_Y, new_Y])

        t1 = time.time()
        hv = compute_hv(train_Y, ref_point)
        hv_history.append(hv)
        times.append(t1 - t0)

        if (i + 1) % 5 == 0:
            print(f"  STCH-NParEGO iter {i+1}/{n_iters}: HV={hv:.4f}, time={t1-t0:.1f}s")

    return hv_history, times, errors


def run_qehvi(problem, d, m, ref_point, n_init, n_iters, q, seed):
    """Run qEHVI. Exponential in m, skip for m>6."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = get_opt_params(m)

    bounds = torch.stack([torch.zeros(d, **tkwargs), torch.ones(d, **tkwargs)])
    train_X, train_Y = get_initial_data(problem, d, n_init, seed)
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
            print(f"  qEHVI iter {i} failed: {e}")
            candidates = torch.rand(q, d, **tkwargs)
            new_Y = problem(candidates)
            train_X = torch.cat([train_X, candidates])
            train_Y = torch.cat([train_Y, new_Y])

        t1 = time.time()
        hv = compute_hv(train_Y, ref_point)
        hv_history.append(hv)
        times.append(t1 - t0)

        if (i + 1) % 5 == 0:
            print(f"  qEHVI iter {i+1}/{n_iters}: HV={hv:.4f}, time={t1-t0:.1f}s")

    return hv_history, times, errors


def run_random(problem, d, m, ref_point, n_init, n_iters, q, seed):
    """Random baseline."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_X, train_Y = get_initial_data(problem, d, n_init, seed)
    hv_history = [compute_hv(train_Y, ref_point)]
    times = []

    for i in range(n_iters):
        t0 = time.time()
        candidates = torch.rand(q, d, **tkwargs)
        new_Y = problem(candidates)
        train_X = torch.cat([train_X, candidates])
        train_Y = torch.cat([train_Y, new_Y])
        t1 = time.time()
        hv_history.append(compute_hv(train_Y, ref_point))
        times.append(t1 - t0)

    return hv_history, times, []


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
        default=["stch_set", "stch_nparego", "qnparego", "qehvi", "random"],
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"DTLZ Benchmark v2: {args.problem} m={args.m}, seeds={args.seeds}, "
          f"iters={args.iters}, K={args.K}, mu={args.mu}")
    print(f"Ablation: {args.ablation}")

    problem, d, ref_point = get_problem(args.problem, args.m)
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
    # Build method configs
    # -----------------------------------------------------------------------
    method_configs = []

    if args.ablation in ("k", "both"):
        # K ablation for STCH-Set
        for K in [3, 5, 10]:
            method_configs.append({
                "name": f"stch_set_K{K}",
                "fn": lambda s, K=K: run_stch_set_bo(
                    problem, d, args.m, ref_point, n_init, args.iters, K, s, args.mu
                ),
            })
    
    if args.ablation in ("mu", "both"):
        # mu ablation for STCH-Set
        for mu_val in [0.01, 0.1, 0.5, 1.0]:
            method_configs.append({
                "name": f"stch_set_mu{mu_val}",
                "fn": lambda s, mu=mu_val: run_stch_set_bo(
                    problem, d, args.m, ref_point, n_init, args.iters, args.K, s, mu
                ),
            })

    if args.ablation == "none":
        # Standard comparison
        fn_map = {
            "stch_set": lambda s: run_stch_set_bo(
                problem, d, args.m, ref_point, n_init, args.iters, args.K, s, args.mu
            ),
            "stch_nparego": lambda s: run_stch_nparego(
                problem, d, args.m, ref_point, n_init, args.iters, args.q, s, args.mu
            ),
            "qnparego": lambda s: run_qnparego(
                problem, d, args.m, ref_point, n_init, args.iters, args.q, s
            ),
            "qehvi": lambda s: run_qehvi(
                problem, d, args.m, ref_point, n_init, args.iters, args.q, s
            ),
            "random": lambda s: run_random(
                problem, d, args.m, ref_point, n_init, args.iters, args.q, s
            ),
        }
        for method_name in args.methods:
            if method_name in fn_map:
                method_configs.append({"name": method_name, "fn": fn_map[method_name]})

    # -----------------------------------------------------------------------
    # Run experiments
    # -----------------------------------------------------------------------
    for cfg in method_configs:
        name = cfg["name"]
        print(f"\n{'='*60}")
        print(f"Running {name}")
        print(f"{'='*60}")

        all_hv = []
        all_times = []
        all_errors = []

        for seed in range(args.seeds):
            print(f"\n  Seed {seed+1}/{args.seeds}")
            try:
                hv_history, times, errors = cfg["fn"](seed)
                all_hv.append(hv_history)
                all_times.append(times)
                all_errors.append(errors)
                print(f"  Final HV: {hv_history[-1]:.4f}, errors: {len(errors)}")
            except Exception as e:
                print(f"  FAILED: {e}")
                traceback.print_exc()

        if all_hv:
            hv_array = np.array(all_hv)
            time_array = np.array(all_times)

            results["methods"][name] = {
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

            print(f"\n  {name}: HV={hv_array[:,-1].mean():.4f}"
                  f"±{hv_array[:,-1].std():.4f}, "
                  f"avg time={time_array.mean():.1f}s/iter, "
                  f"errors={sum(len(e) for e in all_errors)}")

    # Save
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
