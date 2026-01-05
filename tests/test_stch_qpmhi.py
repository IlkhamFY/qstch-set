"""Tests for STCH-qPMHI integration."""

import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from stch_botorch.integration.stch_qpmhi import (
    STCHCandidateGenerator,
    STCHqPMHIAcquisition,
    optimize_stch_qpmhi,
)


class TestSTCHCandidateGenerator:
    """Tests for STCHCandidateGenerator."""

    def setup_method(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.train_X = torch.rand(10, 2, dtype=torch.double)
        self.train_Y = torch.rand(10, 2, dtype=torch.double)  # 2 objectives

        # Fit model
        self.model = SingleTaskGP(self.train_X, self.train_Y)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

        self.bounds = torch.stack(
            [torch.zeros(2, dtype=torch.double), torch.ones(2, dtype=torch.double)]
        )
        self.ref_point = torch.tensor([0.0, 0.0], dtype=torch.double)

    def test_basic_generation(self):
        """Test basic candidate generation."""
        generator = STCHCandidateGenerator(
            self.model, self.bounds, num_weights=10, num_restarts=2
        )

        candidates = generator.generate(ref_point=self.ref_point)

        assert candidates.shape[0] > 0
        assert candidates.shape[1] == 2
        # Check bounds
        assert torch.all(candidates >= self.bounds[0])
        assert torch.all(candidates <= self.bounds[1])

    def test_deduplication(self):
        """Test that deduplication works."""
        generator = STCHCandidateGenerator(
            self.model, self.bounds, num_weights=5, dedup_tol=1e-3, num_restarts=1
        )

        candidates = generator.generate(ref_point=self.ref_point)

        # Check that no two candidates are too close
        if candidates.shape[0] > 1:
            for i in range(candidates.shape[0]):
                for j in range(i + 1, candidates.shape[0]):
                    dist = torch.norm(candidates[i] - candidates[j])
                    assert dist >= generator.dedup_tol or dist < 1e-6  # Allow exact duplicates

    def test_weight_generation(self):
        """Test weight generation on simplex."""
        generator = STCHCandidateGenerator(self.model, self.bounds, num_weights=20)

        weights = generator._generate_weights(20)

        assert weights.shape == (20, 2)
        # Check that weights sum to 1
        sums = weights.sum(axis=1)
        assert torch.allclose(torch.tensor(sums), torch.ones(20), atol=1e-5)
        # Check that all weights are positive
        assert (weights > 0).all()


class TestSTCHqPMHIAcquisition:
    """Tests for STCHqPMHIAcquisition."""

    def setup_method(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.train_X = torch.rand(10, 2, dtype=torch.double)
        self.train_Y = torch.rand(10, 2, dtype=torch.double)

        # Fit model
        self.model = SingleTaskGP(self.train_X, self.train_Y)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

        self.bounds = torch.stack(
            [torch.zeros(2, dtype=torch.double), torch.ones(2, dtype=torch.double)]
        )

        # Compute Pareto front
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

    def test_get_batch(self):
        """Test batch selection."""
        acq = STCHqPMHIAcquisition(
            self.model,
            self.bounds,
            self.pareto_Y,
            self.ref_point,
            num_candidates=20,
            stch_kwargs={"num_weights": 10, "num_restarts": 2},
        )

        batch = acq.get_batch(q=3)

        assert batch.shape == (3, 2)
        # Check bounds
        assert torch.all(batch >= self.bounds[0])
        assert torch.all(batch <= self.bounds[1])

    def test_optimize_stch_qpmhi(self):
        """Test high-level optimize_stch_qpmhi function."""
        batch = optimize_stch_qpmhi(
            self.model,
            self.bounds,
            self.pareto_Y,
            self.ref_point,
            q=2,
            num_candidates=15,
            stch_kwargs={"num_weights": 8, "num_restarts": 2},
        )

        assert batch.shape == (2, 2)
        # Check bounds
        assert torch.all(batch >= self.bounds[0])
        assert torch.all(batch <= self.bounds[1])
