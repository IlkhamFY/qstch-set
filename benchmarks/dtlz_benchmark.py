"""
DTLZ Benchmark Suite for STCH-Set-BO.

Compares STCH-Set-BO (our method) vs qNParEGO, qEHVI, and random
on the standard DTLZ test problems with scalable objectives.

Usage:
    python dtlz_benchmark.py --problem DTLZ2 --m 5 --seeds 5 --iters 30 --q 1
    python dtlz_benchmark.py --problem DTLZ2 --m 10 --seeds 3 --iters 20 --q 1
"""

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.test_functions.multi_objective import DTLZ2
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood

warnings.filterwarnings("ignore")

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from stch_botorch.acquisition.stch_set_bo import qSTCHSet
from stch_botorch.scalarization import smooth_chebyshev

tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}


def get_dtlz2(m: int):
    """Get DTLZ2 problem with m objectives."""
    d = m + 1  # Standard: d = m + k - 1, k=2 for DTLZ2
    problem = DTLZ2(num_objectives=m, dim=d, negate=True)  # negate for maximization
    # Reference point: slightly worse than nadir
    ref_point = torch.full((m,), -1.5, **tkwargs)
    return problem, d, ref_point


def get_initial_data(problem, d, n_init):
    """Generate initial Sobol samples."""
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
        train_X,
        train_Y,
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
    # Filter points that dominate ref_point
    valid = (pareto_Y > ref_point).all(dim=-1)
    pareto_Y = pareto_Y[valid]
    if pareto_Y.shape[0] == 0:
        return 0.0
    try:
        hv = Hypervolume(ref_point=ref_point)
        return hv.compute(pareto_Y)
    except Exception:
        return 0.0


def run_stch_set_bo(problem, d, m, ref_point, n_init, n_iters, q, seed, mu=0.1):
    """Run STCH-Set-BO (our method)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    bounds = torch.stack([torch.zeros(d, **tkwargs), torch.ones(d, **tkwargs)])
    train_X, train_Y = get_initial_data(problem, d, n_init)
    hv_history = [compute_hv(train_Y, ref_point)]
    times = []

    for i in range(n_iters):
        t0 = time.time()
        try:
            model = fit_model(train_X, train_Y)

            # STCH-Set ref_point: for the scalarization, we use ref_point
            # in BoTorch maximization convention
            acqf = qSTCHSet(
                model=model,
                ref_point=ref_point,
                mu=mu,
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
                maximize=True,
            )

            candidates, acq_value = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=q,
                num_restarts=10,
                raw_samples=256,
                options={"batch_limit": 5, "maxiter": 100},
            )

            new_Y = problem(candidates)
            train_X = torch.cat([train_X, candidates])
            train_Y = torch.cat([train_Y, new_Y])
        except Exception as e:
            print(f"  STCH-Set iter {i} failed: {e}")
            # Fallback: random point
            candidates = torch.rand(q, d, **tkwargs)
            new_Y = problem(candidates)
            train_X = torch.cat([train_X, candidates])
            train_Y = torch.cat([train_Y, new_Y])

        t1 = time.time()
        hv = compute_hv(train_Y, ref_point)
        hv_history.append(hv)
        times.append(t1 - t0)

        if (i + 1) % 5 == 0:
            print(f"  STCH-Set iter {i+1}/{n_iters}: HV={hv:.4f}, time={t1-t0:.1f}s")

    return hv_history, times


def run_qnparego(problem, d, m, ref_point, n_init, n_iters, q, seed):
    """Run qNParEGO (Chebyshev scalarization + qNEI)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    bounds = torch.stack([torch.zeros(d, **tkwargs), torch.ones(d, **tkwargs)])
    train_X, train_Y = get_initial_data(problem, d, n_init)
    hv_history = [compute_hv(train_Y, ref_point)]
    times = []

    for i in range(n_iters):
        t0 = time.time()
        try:
            model = fit_model(train_X, train_Y)

            # Random Chebyshev weights (ParEGO style)
            weights = torch.rand(m, **tkwargs)
            weights = weights / weights.sum()

            # Scalarize training targets for qNEI
            from botorch.utils.multi_objective.scalarization import (
                get_chebyshev_scalarization,
            )

            chebyshev_scalarization = get_chebyshev_scalarization(
                weights=weights, Y=train_Y
            )

            # Use qLogNEI with scalarized objective
            acqf = qLogNoisyExpectedImprovement(
                model=model,
                X_baseline=train_X,
                objective=chebyshev_scalarization,
                prune_baseline=True,
            )

            candidates, acq_value = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=q,
                num_restarts=10,
                raw_samples=256,
                options={"batch_limit": 5, "maxiter": 100},
            )

            new_Y = problem(candidates)
            train_X = torch.cat([train_X, candidates])
            train_Y = torch.cat([train_Y, new_Y])
        except Exception as e:
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

    return hv_history, times


def run_stch_nparego(problem, d, m, ref_point, n_init, n_iters, q, seed, mu=0.1):
    """Run STCH-NParEGO (single-point STCH + qNEI, Pires-style)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    bounds = torch.stack([torch.zeros(d, **tkwargs), torch.ones(d, **tkwargs)])
    train_X, train_Y = get_initial_data(problem, d, n_init)
    hv_history = [compute_hv(train_Y, ref_point)]
    times = []

    for i in range(n_iters):
        t0 = time.time()
        try:
            model = fit_model(train_X, train_Y)

            # Random weight for this iteration
            weights = torch.rand(m, **tkwargs)
            weights = weights / weights.sum()

            # STCH scalarization as objective
            def stch_objective(samples, X=None):
                """Scalarize samples using smooth Chebyshev."""
                return smooth_chebyshev(
                    Y=samples, weights=weights, ref_point=ref_point, mu=mu
                )

            acqf = qLogNoisyExpectedImprovement(
                model=model,
                X_baseline=train_X,
                objective=stch_objective,
                prune_baseline=True,
            )

            candidates, acq_value = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=q,
                num_restarts=10,
                raw_samples=256,
                options={"batch_limit": 5, "maxiter": 100},
            )

            new_Y = problem(candidates)
            train_X = torch.cat([train_X, candidates])
            train_Y = torch.cat([train_Y, new_Y])
        except Exception as e:
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

    return hv_history, times


def run_qehvi(problem, d, m, ref_point, n_init, n_iters, q, seed):
    """Run qEHVI (qLogNoisyExpectedHypervolumeImprovement)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    bounds = torch.stack([torch.zeros(d, **tkwargs), torch.ones(d, **tkwargs)])
    train_X, train_Y = get_initial_data(problem, d, n_init)
    hv_history = [compute_hv(train_Y, ref_point)]
    times = []

    for i in range(n_iters):
        t0 = time.time()
        try:
            model = fit_model(train_X, train_Y)

            acqf = qLogNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=ref_point.tolist(),
                X_baseline=train_X,
                prune_baseline=True,
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
            )

            candidates, acq_value = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=q,
                num_restarts=10,
                raw_samples=256,
                options={"batch_limit": 5, "maxiter": 100},
            )

            new_Y = problem(candidates)
            train_X = torch.cat([train_X, candidates])
            train_Y = torch.cat([train_Y, new_Y])
        except Exception as e:
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

    return hv_history, times


def run_random(problem, d, m, ref_point, n_init, n_iters, q, seed):
    """Run random baseline."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    bounds = torch.stack([torch.zeros(d, **tkwargs), torch.ones(d, **tkwargs)])
    train_X, train_Y = get_initial_data(problem, d, n_init)
    hv_history = [compute_hv(train_Y, ref_point)]
    times = []

    for i in range(n_iters):
        t0 = time.time()
        candidates = torch.rand(q, d, **tkwargs)
        new_Y = problem(candidates)
        train_X = torch.cat([train_X, candidates])
        train_Y = torch.cat([train_Y, new_Y])
        t1 = time.time()

        hv = compute_hv(train_Y, ref_point)
        hv_history.append(hv)
        times.append(t1 - t0)

    return hv_history, times


def main():
    parser = argparse.ArgumentParser(description="DTLZ Benchmark for STCH-Set-BO")
    parser.add_argument("--problem", default="DTLZ2", help="Problem name")
    parser.add_argument("--m", type=int, default=3, help="Number of objectives")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--iters", type=int, default=30, help="BO iterations")
    parser.add_argument("--q", type=int, default=1, help="Batch size")
    parser.add_argument("--mu", type=float, default=0.1, help="STCH smoothing param")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["stch_set", "stch_nparego", "qnparego", "qehvi", "random"],
        help="Methods to run",
    )
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    print(f"DTLZ Benchmark: {args.problem}, m={args.m}, seeds={args.seeds}, "
          f"iters={args.iters}, q={args.q}, mu={args.mu}")
    print(f"Methods: {args.methods}")
    print()

    # Get problem
    if args.problem == "DTLZ2":
        problem, d, ref_point = get_dtlz2(args.m)
    else:
        raise ValueError(f"Unknown problem: {args.problem}")

    n_init = 2 * (d + 1)
    print(f"Input dim: {d}, Initial points: {n_init}")

    # Skip qEHVI for large m (exponential cost)
    if args.m > 6 and "qehvi" in args.methods:
        print(f"WARNING: Skipping qEHVI for m={args.m} (exponential HV cost)")
        args.methods = [m for m in args.methods if m != "qehvi"]

    results = {
        "problem": args.problem,
        "m": args.m,
        "d": d,
        "n_init": n_init,
        "n_iters": args.iters,
        "q": args.q,
        "mu": args.mu,
        "seeds": args.seeds,
        "methods": {},
    }

    method_fns = {
        "stch_set": lambda s: run_stch_set_bo(
            problem, d, args.m, ref_point, n_init, args.iters, args.q, s, args.mu
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

    for method in args.methods:
        if method not in method_fns:
            print(f"Unknown method: {method}, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Running {method}")
        print(f"{'='*60}")

        all_hv = []
        all_times = []

        for seed in range(args.seeds):
            print(f"\n  Seed {seed+1}/{args.seeds}")
            try:
                hv_history, times = method_fns[method](seed)
                all_hv.append(hv_history)
                all_times.append(times)
                print(f"  Final HV: {hv_history[-1]:.4f}")
            except Exception as e:
                print(f"  FAILED: {e}")
                traceback.print_exc()

        if all_hv:
            # Compute stats
            hv_array = np.array(all_hv)
            mean_hv = hv_array.mean(axis=0).tolist()
            std_hv = hv_array.std(axis=0).tolist()

            time_array = np.array(all_times)
            mean_time = time_array.mean(axis=0).tolist()

            results["methods"][method] = {
                "hv_mean": mean_hv,
                "hv_std": std_hv,
                "hv_all": hv_array.tolist(),
                "time_mean": mean_time,
                "final_hv_mean": float(hv_array[:, -1].mean()),
                "final_hv_std": float(hv_array[:, -1].std()),
                "mean_iter_time": float(time_array.mean()),
            }

            print(f"\n  {method} summary: HV={hv_array[:,-1].mean():.4f}"
                  f"±{hv_array[:,-1].std():.4f}, "
                  f"avg iter time={time_array.mean():.1f}s")

    # Save results
    output_path = args.output or f"benchmarks/results/dtlz_{args.problem}_m{args.m}_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.problem} m={args.m}")
    print(f"{'='*60}")
    print(f"{'Method':<20} {'Final HV':>12} {'±Std':>10} {'Avg Time/iter':>15}")
    print("-" * 60)
    for method, data in results["methods"].items():
        std_str = f"±{data['final_hv_std']:.4f}"
        print(f"{method:<20} {data['final_hv_mean']:>12.4f} {std_str:>10} "
              f"{data['mean_iter_time']:>12.1f}s")


if __name__ == "__main__":
    main()
