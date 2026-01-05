"""Benchmark runner for comparing acquisition functions."""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from botorch.acquisition.multi_objective import qLogNParEGO, qNoisyExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.hypervolume import Hypervolume
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from stch_botorch.benchmarks.synthetic import SyntheticProblem
from stch_botorch.integration.stch_qpmhi import optimize_stch_qpmhi


def run_benchmark(
    problem: SyntheticProblem,
    methods: Optional[List[str]] = None,
    num_iterations: int = 20,
    batch_size: int = 4,
    num_initial_points: int = 8,
    num_seeds: int = 5,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Run benchmark comparing different acquisition functions.

    Args:
        problem: Synthetic test problem.
        methods: List of methods to compare. Default: ['stch_qpmhi', 'qnehvi', 'qparago', 'random'].
        num_iterations: Number of BO iterations.
        batch_size: Batch size (q).
        num_initial_points: Number of initial random points.
        num_seeds: Number of random seeds.
        output_dir: Output directory for results. If None, uses experiments/results/{timestamp}/.

    Returns:
        Dictionary with results for each method.
    """
    if methods is None:
        methods = ["stch_qpmhi", "qnehvi", "qparago", "random"]

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("experiments") / "results" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for method in methods:
        print(f"\nRunning {method}...")
        method_results = []

        for seed in range(num_seeds):
            print(f"  Seed {seed + 1}/{num_seeds}")
            torch.manual_seed(seed)

            # Initialize data
            train_X = _random_initial_points(problem.bounds, num_initial_points)
            train_Y = problem(train_X)

            # Run BO loop
            hypervolumes = []
            times = []

            for iteration in range(num_iterations):
                start_time = time.time()

                # Fit model
                model = SingleTaskGP(train_X, train_Y)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)

                # Compute Pareto front and reference point
                pareto_Y = _compute_pareto(train_Y)
                ref_point = problem.get_ref_point()

                # Select batch
                if method == "stch_qpmhi":
                    batch = optimize_stch_qpmhi(
                        model,
                        problem.bounds,
                        pareto_Y,
                        ref_point,
                        q=batch_size,
                        num_candidates=min(200, 50 * batch_size),
                        stch_kwargs={"num_weights": 50, "num_restarts": 3},
                    )
                elif method == "qnehvi":
                    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=pareto_Y)
                    acq = qNoisyExpectedHypervolumeImprovement(
                        model=model,
                        ref_point=ref_point.tolist(),
                        partitioning=partitioning,
                        X_baseline=train_X,
                    )
                    from botorch.optim import optimize_acqf

                    batch, _ = optimize_acqf(
                        acq_function=acq,
                        bounds=problem.bounds,
                        q=batch_size,
                        num_restarts=10,
                        raw_samples=512,
                    )
                elif method == "qparago":
                    acq = qLogNParEGO(model=model, ref_point=ref_point.tolist())
                    from botorch.optim import optimize_acqf

                    batch, _ = optimize_acqf(
                        acq_function=acq,
                        bounds=problem.bounds,
                        q=batch_size,
                        num_restarts=10,
                        raw_samples=512,
                    )
                elif method == "random":
                    batch = _random_initial_points(problem.bounds, batch_size)
                else:
                    raise ValueError(f"Unknown method: {method}")

                # Evaluate batch
                batch_Y = problem(batch)

                # Update data
                train_X = torch.cat([train_X, batch], dim=0)
                train_Y = torch.cat([train_Y, batch_Y], dim=0)

                # Compute hypervolume
                pareto_Y = _compute_pareto(train_Y)
                if pareto_Y.shape[0] > 0:
                    hv_calc = Hypervolume(ref_point=ref_point)
                    hv = hv_calc.compute(pareto_Y)
                    hypervolumes.append(hv.item())
                else:
                    hypervolumes.append(0.0)

                elapsed = time.time() - start_time
                times.append(elapsed)

            method_results.append(
                {
                    "seed": seed,
                    "hypervolumes": hypervolumes,
                    "times": times,
                    "final_pareto": pareto_Y.cpu().numpy(),
                }
            )

        results[method] = method_results

        # Save results
        import numpy as np

        np.savez(output_dir / f"{method}_results.npz", results=method_results)

    return results


def _random_initial_points(bounds: Tensor, n: int) -> Tensor:
    """Generate random initial points within bounds."""
    d = bounds.shape[1]
    points = torch.rand(n, d, dtype=bounds.dtype)
    points = points * (bounds[1] - bounds[0]) + bounds[0]
    return points


def _compute_pareto(Y: Tensor) -> Tensor:
    """Compute Pareto front (maximization)."""
    if Y.shape[0] == 0:
        return Y

    n, m = Y.shape
    is_pareto = torch.ones(n, dtype=torch.bool, device=Y.device)

    for i in range(n):
        if not is_pareto[i]:
            continue
        y_i = Y[i : i + 1]

        for j in range(i + 1, n):
            if not is_pareto[j]:
                continue
            y_j = Y[j : j + 1]

            if (y_j >= y_i).all() and (y_j > y_i).any():
                is_pareto[i] = False
                break
            if (y_i >= y_j).all() and (y_i > y_j).any():
                is_pareto[j] = False

    return Y[is_pareto]
