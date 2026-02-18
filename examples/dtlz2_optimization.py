"""Full MOBO loop: qSTCHSet on DTLZ2 (3 objectives).

Runs a complete multi-objective Bayesian optimization loop using qSTCHSet
on the DTLZ2 test problem and reports hypervolume at each iteration.
"""

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.test_functions.multi_objective import DTLZ2
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.mlls import ExactMarginalLogLikelihood

from stch_botorch.acquisition import qSTCHSet


def main() -> None:
    # --- Configuration ---
    d = 4          # input dimensions (d >= m for DTLZ2)
    m = 3          # objectives
    n_init = 2 * d # initial points
    n_iters = 15   # BO iterations
    q = 2          # batch size per iteration

    torch.manual_seed(42)

    # --- Problem ---
    problem = DTLZ2(dim=d, num_objectives=m, negate=True)  # negate → maximization
    bounds = torch.stack([torch.zeros(d), torch.ones(d)])
    ref_point = torch.full((m,), -1.5)  # dominated reference for hypervolume

    hv_computer = Hypervolume(ref_point=ref_point)

    # --- Initial data ---
    train_X = torch.rand(n_init, d)
    train_Y = problem(train_X)

    print(f"DTLZ2 (d={d}, m={m}), q={q}, {n_iters} iterations")
    print("-" * 50)

    for i in range(n_iters):
        # Fit GP
        model = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=m))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # Build acquisition
        acqf = qSTCHSet(
            model=model,
            ref_point=ref_point,
            mu=0.1,
        )

        # Optimize
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=q,
            num_restarts=10,
            raw_samples=256,
        )

        # Evaluate and append
        new_Y = problem(candidates)
        train_X = torch.cat([train_X, candidates], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)

        # Compute hypervolume of current Pareto front
        pareto_mask = is_non_dominated(train_Y)
        pareto_Y = train_Y[pareto_mask]
        hv = hv_computer.compute(pareto_Y)

        print(f"Iter {i+1:3d} | n={train_X.shape[0]:4d} | Pareto={pareto_mask.sum().item():3d} | HV={hv:.4f}")

    print("-" * 50)
    print(f"Final hypervolume: {hv:.4f}")


if __name__ == "__main__":
    main()
