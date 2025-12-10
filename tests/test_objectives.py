"""Tests for BoTorch objective wrapper classes."""

import pytest
import torch

from stch_botorch.objectives import SmoothChebyshevObjective, SmoothChebyshevSetObjective


class TestSmoothChebyshevObjective:
    """Tests for SmoothChebyshevObjective class."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        objective = SmoothChebyshevObjective(
            weights=torch.tensor([0.5, 0.5]),
            ref_point=torch.tensor([0.0, 0.0]),
            mu=0.1,
        )

        # samples shape: (sample_shape x batch_shape x q x m)
        # Example: (2, 3, 4, 2) -> (2, 3, 4)
        samples = torch.randn(2, 3, 4, 2)  # 2 MC samples, 3 batch, 4 candidates, 2 objectives
        result = objective.forward(samples)

        assert result.shape == (2, 3, 4)  # q dimension preserved

    def test_shape_mapping(self):
        """Test that shape mapping is correct: ... x q x m -> ... x q."""
        objective = SmoothChebyshevObjective(
            weights=torch.tensor([0.5, 0.5]),
            ref_point=torch.tensor([0.0, 0.0]),
            mu=0.1,
        )

        # Various shapes
        test_cases = [
            (1, 2, 2),  # (sample, batch, q, m) -> (sample, batch, q)
            (5, 2, 2),  # (sample, batch, q, m) -> (sample, batch, q)
            (1, 1, 3, 2),  # (sample, batch, q, m) -> (sample, batch, q)
        ]

        for shape in test_cases:
            if len(shape) == 3:
                # Add m dimension
                samples = torch.randn(*shape, 2)
                expected_shape = shape
            else:
                samples = torch.randn(*shape)
                expected_shape = shape[:-1]

            result = objective.forward(samples)
            assert result.shape == expected_shape, f"Failed for shape {shape}"

    def test_mc_sample_handling(self):
        """Test that MC samples are handled correctly."""
        objective = SmoothChebyshevObjective(
            weights=torch.tensor([0.5, 0.5]),
            ref_point=torch.tensor([0.0, 0.0]),
            mu=0.1,
        )

        # Multiple MC samples
        samples = torch.randn(10, 1, 2, 2)  # 10 MC samples
        result = objective.forward(samples)

        assert result.shape == (10, 1, 2)
        assert torch.all(torch.isfinite(result))

    def test_gradient_flow(self):
        """Test that gradients flow through the objective."""
        objective = SmoothChebyshevObjective(
            weights=torch.tensor([0.5, 0.5]),
            ref_point=torch.tensor([0.0, 0.0]),
            mu=0.1,
        )

        samples = torch.randn(2, 3, 4, 2, requires_grad=True)
        result = objective.forward(samples)
        loss = result.sum()
        loss.backward()

        assert samples.grad is not None
        assert torch.all(torch.isfinite(samples.grad))

    def test_ideal_point_default(self):
        """Test that ideal point is used when ref_point is None."""
        objective = SmoothChebyshevObjective(
            weights=torch.tensor([0.5, 0.5]),
            ref_point=None,
            mu=0.1,
        )

        samples = torch.tensor([[[[1.0, 3.0], [2.0, 1.0], [3.0, 2.0]]]])
        result = objective.forward(samples)

        assert result.shape == (1, 1, 3)
        assert torch.all(torch.isfinite(result))

    def test_custom_mu(self):
        """Test that custom mu values work."""
        objective = SmoothChebyshevObjective(
            weights=torch.tensor([0.5, 0.5]),
            ref_point=torch.tensor([0.0, 0.0]),
            mu=0.01,  # Smaller mu
        )

        samples = torch.randn(2, 3, 4, 2)
        result = objective.forward(samples)

        assert result.shape == (2, 3, 4)
        assert torch.all(torch.isfinite(result))


class TestSmoothChebyshevSetObjective:
    """Tests for SmoothChebyshevSetObjective class."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        objective = SmoothChebyshevSetObjective(
            weights=torch.tensor([0.5, 0.5]),
            ref_point=torch.tensor([0.0, 0.0]),
            mu=0.1,
        )

        # samples shape: (sample_shape x batch_shape x q x m)
        # Example: (2, 3, 4, 2) -> (2, 3) - q dimension removed
        samples = torch.randn(2, 3, 4, 2)
        result = objective.forward(samples)

        assert result.shape == (2, 3)  # q dimension removed

    def test_shape_mapping(self):
        """Test that shape mapping is correct: ... x q x m -> (sample_shape x batch_shape)."""
        objective = SmoothChebyshevSetObjective(
            weights=torch.tensor([0.5, 0.5]),
            ref_point=torch.tensor([0.0, 0.0]),
            mu=0.1,
        )

        # Various shapes
        test_cases = [
            ((1, 2, 2, 2), (1, 2)),  # (sample, batch, q, m) -> (sample, batch)
            ((5, 3, 4, 2), (5, 3)),  # (sample, batch, q, m) -> (sample, batch)
            ((1, 1, 3, 2), (1, 1)),  # (sample, batch, q, m) -> (sample, batch)
        ]

        for input_shape, expected_shape in test_cases:
            samples = torch.randn(*input_shape)
            result = objective.forward(samples)
            assert result.shape == expected_shape, f"Failed for shape {input_shape}"

    def test_q_dimension_removed(self):
        """Test that q dimension is correctly removed."""
        objective = SmoothChebyshevSetObjective(
            weights=torch.tensor([0.5, 0.5]),
            ref_point=torch.tensor([0.0, 0.0]),
            mu=0.1,
        )

        # Different q values should all result in same output shape
        for q in [2, 4, 8]:
            samples = torch.randn(2, 3, q, 2)
            result = objective.forward(samples)
            assert result.shape == (2, 3), f"Failed for q={q}"

    def test_mc_sample_handling(self):
        """Test that MC samples are handled correctly."""
        objective = SmoothChebyshevSetObjective(
            weights=torch.tensor([0.5, 0.5]),
            ref_point=torch.tensor([0.0, 0.0]),
            mu=0.1,
        )

        # Multiple MC samples
        samples = torch.randn(10, 1, 2, 2)  # 10 MC samples
        result = objective.forward(samples)

        assert result.shape == (10, 1)  # q dimension removed
        assert torch.all(torch.isfinite(result))

    def test_gradient_flow(self):
        """Test that gradients flow through the objective."""
        objective = SmoothChebyshevSetObjective(
            weights=torch.tensor([0.5, 0.5]),
            ref_point=torch.tensor([0.0, 0.0]),
            mu=0.1,
        )

        samples = torch.randn(2, 3, 4, 2, requires_grad=True)
        result = objective.forward(samples)
        loss = result.sum()
        loss.backward()

        assert samples.grad is not None
        assert torch.all(torch.isfinite(samples.grad))

    def test_ideal_point_default(self):
        """Test that ideal point is used when ref_point is None."""
        objective = SmoothChebyshevSetObjective(
            weights=torch.tensor([0.5, 0.5]),
            ref_point=None,
            mu=0.1,
        )

        samples = torch.tensor([[[[1.0, 3.0], [2.0, 1.0], [3.0, 2.0]]]])
        result = objective.forward(samples)

        assert result.shape == (1, 1)  # q dimension removed
        assert torch.all(torch.isfinite(result))

    def test_custom_mu(self):
        """Test that custom mu values work."""
        objective = SmoothChebyshevSetObjective(
            weights=torch.tensor([0.5, 0.5]),
            ref_point=torch.tensor([0.0, 0.0]),
            mu=0.01,  # Smaller mu
        )

        samples = torch.randn(2, 3, 4, 2)
        result = objective.forward(samples)

        assert result.shape == (2, 3)  # q dimension removed
        assert torch.all(torch.isfinite(result))

    def test_batch_aggregation(self):
        """Test that the objective correctly aggregates over the batch dimension."""
        objective = SmoothChebyshevSetObjective(
            weights=torch.tensor([0.5, 0.5]),
            ref_point=torch.tensor([0.0, 0.0]),
            mu=0.1,
        )

        # Create samples where different candidates excel at different objectives
        samples = torch.tensor([[[[1.0, 10.0], [10.0, 1.0]]]])  # shape: (1, 1, 2, 2)
        result = objective.forward(samples)

        assert result.shape == (1, 1)
        assert torch.all(torch.isfinite(result))


class TestBoTorchIntegration:
    """Tests for integration with BoTorch components."""

    def test_objective_is_mcmultioutput(self):
        """Test that objectives inherit from MCMultiOutputObjective."""
        from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective

        obj1 = SmoothChebyshevObjective(
            weights=torch.tensor([0.5, 0.5]),
            ref_point=torch.tensor([0.0, 0.0]),
        )
        obj2 = SmoothChebyshevSetObjective(
            weights=torch.tensor([0.5, 0.5]),
            ref_point=torch.tensor([0.0, 0.0]),
        )

        assert isinstance(obj1, MCMultiOutputObjective)
        assert isinstance(obj2, MCMultiOutputObjective)

    def test_different_output_shapes(self):
        """Test that the two objectives produce different output shapes."""
        samples = torch.randn(5, 3, 4, 2)  # (sample, batch, q, m)

        obj1 = SmoothChebyshevObjective(
            weights=torch.tensor([0.5, 0.5]),
            ref_point=torch.tensor([0.0, 0.0]),
        )
        obj2 = SmoothChebyshevSetObjective(
            weights=torch.tensor([0.5, 0.5]),
            ref_point=torch.tensor([0.0, 0.0]),
        )

        result1 = obj1.forward(samples)
        result2 = obj2.forward(samples)

        # obj1 preserves q dimension, obj2 removes it
        assert result1.shape == (5, 3, 4)
        assert result2.shape == (5, 3)

