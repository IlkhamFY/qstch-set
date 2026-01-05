"""Synthetic test problems for multi-objective optimization."""

from typing import Optional

import torch
from botorch.test_functions.multi_objective import BraninCurrin, DTLZ2
from torch import Tensor


class SyntheticProblem:
    """Wrapper for synthetic test problems.

    Args:
        name: Problem name ('branin_currin', 'dtlz2', etc.).
        dim: Input dimension (for DTLZ2).
        num_objectives: Number of objectives.
    """

    def __init__(self, name: str, dim: Optional[int] = None, num_objectives: Optional[int] = None):
        self.name = name.lower()

        if self.name == "branin_currin":
            self.func = BraninCurrin(negate=True)  # Negate to maximize
            self.dim = 2
            self.num_objectives = 2
            self.bounds = torch.tensor([[-5.0, 0.0], [10.0, 15.0]])
        elif self.name == "dtlz2":
            if dim is None:
                dim = 6
            if num_objectives is None:
                num_objectives = 2
            self.func = DTLZ2(dim=dim, num_objectives=num_objectives, negate=True)
            self.dim = dim
            self.num_objectives = num_objectives
            self.bounds = torch.tensor([[0.0] * dim, [1.0] * dim])
        else:
            raise ValueError(f"Unknown problem: {name}")

    def __call__(self, X: Tensor) -> Tensor:
        """Evaluate problem at points X.

        Args:
            X: Input points of shape (..., d).

        Returns:
            Objective values of shape (..., m).
        """
        return self.func(X)

    def get_ref_point(self) -> Tensor:
        """Get reference point for hypervolume computation.

        Returns:
            Reference point of shape (m,).
        """
        # Use a point worse than typical observed values
        # For maximization, use negative values
        return torch.zeros(self.num_objectives) - 1.0
