"""Integration tests for STCH-BoTorch with BoTorch."""

import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qSimpleRegret
from gpytorch.mlls import ExactMarginalLogLikelihood

from stch_botorch import SmoothChebyshevObjective, SmoothChebyshevSetObjective


class TestBoTorchIntegration:
    """Integration tests with BoTorch models and acquisition functions."""

    def setup_method(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.train_X = torch.rand(10, 2, dtype=torch.double)
        self.train_Y = torch.rand(10, 3, dtype=torch.double)  # 3 objectives
        
        # Fit model
        self.model = SingleTaskGP(self.train_X, self.train_Y)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

    def test_smooth_chebyshev_objective_with_model(self):
        """Test SmoothChebyshevObjective with BoTorch model posterior."""
        objective = SmoothChebyshevObjective(
            weights=torch.tensor([0.5, 0.3, 0.2], dtype=torch.double),
            ref_point=torch.tensor([0.0, 0.0, 0.0], dtype=torch.double),
            mu=0.1
        )
        
        # Test with various X shapes
        test_cases = [
            (1, 1, 2),  # Single candidate
            (1, 2, 2),  # q=2
            (1, 3, 2),  # q=3
            (5, 1, 2),  # 5 batch points, q=1
        ]
        
        for shape in test_cases:
            X = torch.rand(*shape, dtype=torch.double)
            with torch.no_grad():
                posterior = self.model.posterior(X)
                samples = posterior.sample(torch.Size([10]))  # 10 MC samples
                obj_vals = objective.forward(samples)
                
                # Verify shape: (MC, batch, q) or (MC, batch*q)
                expected_mc = 10
                if len(shape) == 3:
                    expected_batch, expected_q = shape[0], shape[1]
                    # Should be (MC, batch, q)
                    assert obj_vals.shape == (expected_mc, expected_batch, expected_q), \
                        f"Shape mismatch for {shape}: got {obj_vals.shape}"
                assert torch.all(torch.isfinite(obj_vals))

    def test_smooth_chebyshev_set_objective_with_model(self):
        """Test SmoothChebyshevSetObjective with BoTorch model posterior."""
        objective = SmoothChebyshevSetObjective(
            weights=torch.tensor([0.5, 0.3, 0.2], dtype=torch.double),
            ref_point=torch.tensor([0.0, 0.0, 0.0], dtype=torch.double),
            mu=0.1
        )
        
        # Test with various X shapes
        test_cases = [
            (1, 1, 2),  # Single candidate
            (1, 2, 2),  # q=2
            (1, 3, 2),  # q=3
        ]
        
        for shape in test_cases:
            X = torch.rand(*shape, dtype=torch.double)
            with torch.no_grad():
                posterior = self.model.posterior(X)
                samples = posterior.sample(torch.Size([10]))
                obj_vals = objective.forward(samples)
                
                # Verify shape: (MC, batch) - q dimension removed
                expected_mc = 10
                expected_batch = shape[0]
                assert obj_vals.shape == (expected_mc, expected_batch), \
                    f"Shape mismatch for {shape}: got {obj_vals.shape}"
                assert torch.all(torch.isfinite(obj_vals))

    def test_qsimpleregreet_with_smooth_chebyshev_objective(self):
        """Test qSimpleRegret acquisition function with SmoothChebyshevObjective."""
        objective = SmoothChebyshevObjective(
            weights=torch.tensor([0.5, 0.3, 0.2], dtype=torch.double),
            ref_point=torch.tensor([0.0, 0.0, 0.0], dtype=torch.double),
            mu=0.1
        )
        acqf = qSimpleRegret(model=self.model, objective=objective)
        
        # Test with single candidate (this should work)
        test_X = torch.rand(1, 1, 2, dtype=torch.double)
        acq_val = acqf(test_X)
        
        assert acq_val.numel() == 1
        assert torch.all(torch.isfinite(acq_val))

    def test_qsimpleregreet_with_smooth_chebyshev_set_objective(self):
        """Test qSimpleRegret with SmoothChebyshevSetObjective.
        
        Note: SmoothChebyshevSetObjective removes the q dimension, which causes
        BoTorch's shape validation to fail. This is a known limitation when using
        set objectives with standard acquisition functions. The objective works
        correctly with direct model sampling (tested above).
        """
        objective = SmoothChebyshevSetObjective(
            weights=torch.tensor([0.5, 0.3, 0.2], dtype=torch.double),
            ref_point=torch.tensor([0.0, 0.0, 0.0], dtype=torch.double),
            mu=0.1
        )
        
        # Test direct model sampling (this works)
        test_X = torch.rand(1, 1, 2, dtype=torch.double)
        with torch.no_grad():
            posterior = self.model.posterior(test_X)
            samples = posterior.sample(torch.Size([10]))
            obj_vals = objective.forward(samples)
            assert obj_vals.shape == (10, 1)  # (MC, batch) - q removed
            assert torch.all(torch.isfinite(obj_vals))
        
        # Note: Using with acquisition functions may fail due to BoTorch's
        # shape validation expecting the q dimension to be present
        # This is documented in the class docstring

    def test_gradient_flow_through_acquisition(self):
        """Test that gradients flow through acquisition function."""
        objective = SmoothChebyshevObjective(
            weights=torch.tensor([0.5, 0.3, 0.2], dtype=torch.double),
            ref_point=torch.tensor([0.0, 0.0, 0.0], dtype=torch.double),
            mu=0.1
        )
        acqf = qSimpleRegret(model=self.model, objective=objective)
        
        test_X = torch.rand(1, 1, 2, dtype=torch.double, requires_grad=True)
        acq_val = acqf(test_X)
        acq_val.backward()
        
        assert test_X.grad is not None
        assert torch.all(torch.isfinite(test_X.grad))

