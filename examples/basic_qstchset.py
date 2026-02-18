"""Minimal example: qSTCHSet acquisition function with BoTorch.

Demonstrates how to use qSTCHSet for multi-objective Bayesian optimization
on a simple 2-objective problem.
"""

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from stch_botorch.acquisition import qSTCHSet


def main() -> None:
    # --- Problem setup ---
    d = 3  # input dimensions
    m = 2  # objectives
    n_init = 10  # initial points

    bounds = torch.stack([torch.zeros(d), torch.ones(d)])

    # Simple bi-objective: minimize f1 = sum(x), f2 = sum((x-1)^2)
    # BoTorch maximizes, so we negate.
    def objective(X: torch.Tensor) -> torch.Tensor:
        f1 = -X.sum(dim=-1)
        f2 = -((X - 1) ** 2).sum(dim=-1)
        return torch.stack([f1, f2], dim=-1)

    # --- Initial data ---
    train_X = torch.rand(n_init, d)
    train_Y = objective(train_X)

    # --- Fit GP ---
    model = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=m))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # --- Build qSTCHSet acquisition ---
    ref_point = train_Y.min(dim=0).values - 0.1
    acqf = qSTCHSet(model=model, ref_point=ref_point, mu=0.1)

    # --- Optimize: jointly select q=4 candidates ---
    candidates, value = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=4,
        num_restarts=10,
        raw_samples=256,
    )

    print(f"Selected candidates:\n{candidates}")
    print(f"Acquisition value: {value.item():.4f}")
    print(f"Objective values:\n{objective(candidates)}")


if __name__ == "__main__":
    main()
