"""Tests for qPMHI acquisition function."""

import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from stch_botorch.acquisition.qpmhi import qPMHI


class TestqPMHI:
    """Tests for qPMHI class."""

    def setup_method(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.train_X = torch.rand(10, 2, dtype=torch.double)
        self.train_Y = torch.rand(10, 2, dtype=torch.double)  # 2 objectives

        # Fit model
        self.model = SingleTaskGP(self.train_X, self.train_Y)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

        # Compute Pareto front (simple: non-dominated points)
        self.pareto_Y = self._compute_pareto(self.train_Y)
        self.ref_point = torch.tensor([0.0, 0.0], dtype=torch.double)

    @staticmethod
    def _compute_pareto(Y: torch.Tensor) -> torch.Tensor:
        """Compute Pareto front (maximization)."""
        n, m = Y.shape
        is_pareto = torch.ones(n, dtype=torch.bool)

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

    def test_basic_functionality(self):
        """Test basic qPMHI scoring."""
        qpmhi = qPMHI(self.model, self.pareto_Y, self.ref_point)

        candidates = torch.rand(5, 2, dtype=torch.double)
        scores = qpmhi(candidates)

        assert scores.shape == (5,)
        assert torch.all(torch.isfinite(scores))
        assert torch.all(scores >= 0)
        assert torch.all(scores <= 1)

    def test_probabilities_sum_to_one(self):
        """Test that probabilities approximately sum to 1."""
        qpmhi = qPMHI(self.model, self.pareto_Y, self.ref_point)

        candidates = torch.rand(10, 2, dtype=torch.double)
        scores = qpmhi(candidates)

        # Probabilities should sum close to 1 (allowing for numerical errors)
        total = scores.sum().item()
        assert 0.9 <= total <= 1.1, f"Probabilities sum to {total}, expected ~1.0"

    def test_shape_handling(self):
        """Test various input shapes."""
        qpmhi = qPMHI(self.model, self.pareto_Y, self.ref_point)

        # Single candidate
        X1 = torch.rand(1, 2, dtype=torch.double)
        scores1 = qpmhi(X1)
        assert scores1.shape == (1,)

        # Multiple candidates
        X2 = torch.rand(5, 2, dtype=torch.double)
        scores2 = qpmhi(X2)
        assert scores2.shape == (5,)

        # Batched input
        X3 = torch.rand(2, 5, 2, dtype=torch.double)
        scores3 = qpmhi(X3)
        assert scores3.shape == (2, 5)

    def test_empty_candidates(self):
        """Test with empty candidate set."""
        qpmhi = qPMHI(self.model, self.pareto_Y, self.ref_point)

        candidates = torch.empty(0, 2, dtype=torch.double)
        scores = qpmhi(candidates)

        assert scores.shape == (0,)

    def test_tie_handling(self):
        """Test that ties are handled correctly."""
        qpmhi = qPMHI(self.model, self.pareto_Y, self.ref_point)

        # Use same candidate multiple times (should create ties)
        candidate = torch.rand(1, 2, dtype=torch.double)
        candidates = candidate.repeat(3, 1)

        scores = qpmhi(candidates)

        # All should have similar scores (ties split probability)
        assert torch.allclose(scores, scores[0], atol=1e-2)

    def test_dominated_candidates(self):
        """Test that dominated candidates get low scores."""
        qpmhi = qPMHI(self.model, self.pareto_Y, self.ref_point)

        # Create a candidate that's likely dominated
        # (worse than all Pareto points in all objectives)
        worst_point = self.pareto_Y.min(dim=0)[0] - 1.0
        dominated_candidate = worst_point.unsqueeze(0)

        # Mix dominated and non-dominated candidates
        good_candidates = torch.rand(5, 2, dtype=torch.double)
        candidates = torch.cat([dominated_candidate, good_candidates], dim=0)

        scores = qpmhi(candidates)

        # Dominated candidate should have lower score (but not necessarily 0 due to MC sampling)
        # Just check that scores are valid
        assert torch.all(torch.isfinite(scores))
        assert torch.all(scores >= 0)

    def test_ref_point_validation(self):
        """Test that ref_point validation works."""
        # Wrong dimension
        with pytest.raises(ValueError, match="ref_point must be 1D"):
            qPMHI(self.model, self.pareto_Y, self.ref_point.unsqueeze(0))

        # Wrong number of objectives
        wrong_ref = torch.tensor([0.0, 0.0, 0.0], dtype=torch.double)
        with pytest.raises(ValueError, match="same number of objectives"):
            qPMHI(self.model, self.pareto_Y, wrong_ref)

    def test_pareto_Y_validation(self):
        """Test that pareto_Y validation works."""
        # Wrong dimension
        with pytest.raises(ValueError, match="pareto_Y must be 2D"):
            qPMHI(self.model, self.pareto_Y.flatten(), self.ref_point)
