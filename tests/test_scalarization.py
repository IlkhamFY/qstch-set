"""Tests for smooth Tchebycheff scalarization functions."""

import pytest
import torch

from stch_botorch.scalarization import smooth_chebyshev, smooth_chebyshev_set


class TestSmoothChebyshev:
    """Tests for smooth_chebyshev function."""

    def test_basic_functionality(self):
        """Test basic scalarization with simple inputs."""
        Y = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        weights = torch.tensor([0.5, 0.5])
        ref_point = torch.tensor([0.0, 0.0])
        mu = 0.1

        result = smooth_chebyshev(Y, weights, ref_point, mu)
        assert result.shape == (2,)
        assert torch.all(torch.isfinite(result))

    def test_weight_normalization(self):
        """Test that weights are automatically normalized."""
        Y = torch.tensor([[1.0, 2.0]])
        weights_unormalized = torch.tensor([2.0, 2.0])  # Sums to 4
        weights_normalized = torch.tensor([1.0, 1.0])  # Sums to 2
        ref_point = torch.tensor([0.0, 0.0])
        mu = 0.1

        result1 = smooth_chebyshev(Y, weights_unormalized, ref_point, mu)
        result2 = smooth_chebyshev(Y, weights_normalized, ref_point, mu)

        # Results should be the same after normalization
        assert torch.allclose(result1, result2)

    def test_ideal_point_default(self):
        """Test that ideal point is computed correctly when ref_point is None."""
        Y = torch.tensor([[1.0, 3.0], [2.0, 1.0], [3.0, 2.0]])
        weights = torch.tensor([0.5, 0.5])
        mu = 0.1

        # Ideal point should be [1.0, 1.0] (min of each column)
        result = smooth_chebyshev(Y, weights, ref_point=None, mu=mu)
        assert result.shape == (3,)
        assert torch.all(torch.isfinite(result))

        # Should match explicit ideal point
        explicit_result = smooth_chebyshev(
            Y, weights, ref_point=torch.tensor([1.0, 1.0]), mu=mu
        )
        assert torch.allclose(result, explicit_result)

    def test_shape_handling(self):
        """Test various input shapes."""
        weights = torch.tensor([0.5, 0.5])
        ref_point = torch.tensor([0.0, 0.0])
        mu = 0.1

        # 1D: (m,)
        Y1 = torch.tensor([1.0, 2.0])
        result1 = smooth_chebyshev(Y1, weights, ref_point, mu)
        assert result1.shape == ()

        # 2D: (n, m)
        Y2 = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        result2 = smooth_chebyshev(Y2, weights, ref_point, mu)
        assert result2.shape == (2,)

        # 3D: (b, n, m)
        Y3 = torch.tensor([[[1.0, 2.0], [2.0, 1.0]], [[3.0, 1.0], [1.0, 3.0]]])
        result3 = smooth_chebyshev(Y3, weights, ref_point, mu)
        assert result3.shape == (2, 2)

    def test_gradient_computation(self):
        """Test that gradients are computed for all objectives."""
        Y = torch.tensor([[1.0, 2.0], [2.0, 1.0]], requires_grad=True)
        weights = torch.tensor([0.5, 0.5])
        ref_point = torch.tensor([0.0, 0.0])
        mu = 0.1

        result = smooth_chebyshev(Y, weights, ref_point, mu)
        loss = result.sum()
        loss.backward()

        # Check that gradients exist and are non-zero
        assert Y.grad is not None
        assert torch.all(torch.isfinite(Y.grad))
        # Unlike hard Tchebycheff, smooth version should have gradients for all objectives
        assert not torch.allclose(Y.grad, torch.zeros_like(Y.grad))

    def test_convergence_to_tchebyshev(self):
        """Test that STCH converges to TCH as mu -> 0."""
        Y = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        weights = torch.tensor([0.5, 0.5])
        ref_point = torch.tensor([0.0, 0.0])

        # Compute true Tchebycheff: utility = -max(w_i * (Y_i - z*_i))
        weighted_distances = weights * (Y - ref_point)
        tch_values = -weighted_distances.max(dim=-1)[0]  # Negative for maximization

        # Compute STCH with decreasing mu
        mus = [1.0, 0.1, 0.01, 0.001]
        errors = []
        for mu in mus:
            stch_values = smooth_chebyshev(Y, weights, ref_point, mu)
            error = torch.abs(stch_values - tch_values).max().item()
            errors.append(error)

        # Error should decrease as mu decreases
        assert errors[-1] < errors[0]
        # For very small mu, should be very close
        assert errors[-1] < 0.1

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Large values
        Y_large = torch.tensor([[100.0, 200.0], [200.0, 100.0]])
        weights = torch.tensor([0.5, 0.5])
        ref_point = torch.tensor([0.0, 0.0])
        mu = 0.1

        result = smooth_chebyshev(Y_large, weights, ref_point, mu)
        assert torch.all(torch.isfinite(result))

        # Small mu
        mu_small = 1e-6
        result_small = smooth_chebyshev(Y_large, weights, ref_point, mu_small)
        assert torch.all(torch.isfinite(result_small))

    def test_positive_weights_requirement(self):
        """Test that negative weights raise an error."""
        Y = torch.tensor([[1.0, 2.0]])
        weights_negative = torch.tensor([-0.5, 0.5])
        ref_point = torch.tensor([0.0, 0.0])
        mu = 0.1

        with pytest.raises(ValueError, match="All weights must be positive"):
            smooth_chebyshev(Y, weights_negative, ref_point, mu)

    def test_positive_mu_requirement(self):
        """Test that non-positive mu raises an error."""
        Y = torch.tensor([[1.0, 2.0]])
        weights = torch.tensor([0.5, 0.5])
        ref_point = torch.tensor([0.0, 0.0])

        with pytest.raises(ValueError, match="mu must be positive"):
            smooth_chebyshev(Y, weights, ref_point, mu=0.0)

        with pytest.raises(ValueError, match="mu must be positive"):
            smooth_chebyshev(Y, weights, ref_point, mu=-0.1)

    def test_shape_mismatch_errors(self):
        """Test that shape mismatches raise appropriate errors."""
        Y = torch.tensor([[1.0, 2.0]])
        weights = torch.tensor([0.5, 0.5])
        ref_point = torch.tensor([0.0, 0.0, 0.0])  # Wrong size
        mu = 0.1

        with pytest.raises(ValueError, match="ref_point must have"):
            smooth_chebyshev(Y, weights, ref_point, mu)

        weights_wrong = torch.tensor([0.5, 0.5, 0.5])  # Wrong size
        ref_point = torch.tensor([0.0, 0.0])

        with pytest.raises(ValueError, match="weights must have"):
            smooth_chebyshev(Y, weights_wrong, ref_point, mu)


class TestSmoothChebyshevSet:
    """Tests for smooth_chebyshev_set function."""

    def test_basic_functionality(self):
        """Test basic set scalarization."""
        Y = torch.tensor([[[1.0, 2.0], [2.0, 1.0]]])  # shape: (1, 2, 2)
        weights = torch.tensor([0.5, 0.5])
        ref_point = torch.tensor([0.0, 0.0])
        mu = 0.1

        result = smooth_chebyshev_set(Y, weights, ref_point, mu)
        assert result.shape == (1,)  # q dimension removed
        assert torch.all(torch.isfinite(result))

    def test_q_dimension_removed(self):
        """Test that the q dimension is correctly removed."""
        Y = torch.tensor([[[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]]])  # shape: (1, 3, 2)
        weights = torch.tensor([0.5, 0.5])
        ref_point = torch.tensor([0.0, 0.0])
        mu = 0.1

        result = smooth_chebyshev_set(Y, weights, ref_point, mu)
        assert result.shape == (1,)  # q=3 dimension removed

    def test_shape_handling(self):
        """Test various input shapes."""
        weights = torch.tensor([0.5, 0.5])
        ref_point = torch.tensor([0.0, 0.0])
        mu = 0.1

        # 2D: (q, m) -> scalar
        Y1 = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        result1 = smooth_chebyshev_set(Y1, weights, ref_point, mu)
        assert result1.shape == ()

        # 3D: (b, q, m) -> (b,)
        Y2 = torch.tensor([[[1.0, 2.0], [2.0, 1.0]], [[3.0, 1.0], [1.0, 3.0]]])
        result2 = smooth_chebyshev_set(Y2, weights, ref_point, mu)
        assert result2.shape == (2,)

    def test_gradient_computation(self):
        """Test that gradients are computed correctly."""
        Y = torch.tensor([[[1.0, 2.0], [2.0, 1.0]]], requires_grad=True)
        weights = torch.tensor([0.5, 0.5])
        ref_point = torch.tensor([0.0, 0.0])
        mu = 0.1

        result = smooth_chebyshev_set(Y, weights, ref_point, mu)
        loss = result.sum()
        loss.backward()

        assert Y.grad is not None
        assert torch.all(torch.isfinite(Y.grad))
        assert not torch.allclose(Y.grad, torch.zeros_like(Y.grad))

    def test_ideal_point_default(self):
        """Test that ideal point is computed correctly when ref_point is None."""
        Y = torch.tensor([[[1.0, 3.0], [2.0, 1.0], [3.0, 2.0]]])
        weights = torch.tensor([0.5, 0.5])
        mu = 0.1

        result = smooth_chebyshev_set(Y, weights, ref_point=None, mu=mu)
        assert result.shape == (1,)
        assert torch.all(torch.isfinite(result))

    def test_weight_normalization(self):
        """Test that weights are automatically normalized."""
        Y = torch.tensor([[[1.0, 2.0], [2.0, 1.0]]])
        weights_unormalized = torch.tensor([2.0, 2.0])
        weights_normalized = torch.tensor([1.0, 1.0])
        ref_point = torch.tensor([0.0, 0.0])
        mu = 0.1

        result1 = smooth_chebyshev_set(Y, weights_unormalized, ref_point, mu)
        result2 = smooth_chebyshev_set(Y, weights_normalized, ref_point, mu)

        assert torch.allclose(result1, result2)

    def test_nested_smoothing_logic(self):
        """Test that nested smoothing (min over batch, max over objectives) works."""
        # Create a case where one candidate is better for obj1, another for obj2
        Y = torch.tensor([[[1.0, 10.0], [10.0, 1.0]]])  # shape: (1, 2, 2)
        weights = torch.tensor([0.5, 0.5])
        ref_point = torch.tensor([0.0, 0.0])
        mu = 0.1

        result = smooth_chebyshev_set(Y, weights, ref_point, mu)
        assert result.shape == (1,)
        assert torch.all(torch.isfinite(result))

    def test_minimum_dimensions(self):
        """Test that function requires at least 2 dimensions."""
        weights = torch.tensor([0.5, 0.5])
        ref_point = torch.tensor([0.0, 0.0])
        mu = 0.1

        # 1D input should raise error
        Y_1d = torch.tensor([1.0, 2.0])
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            smooth_chebyshev_set(Y_1d, weights, ref_point, mu)

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        Y = torch.tensor([[[100.0, 200.0], [200.0, 100.0]]])
        weights = torch.tensor([0.5, 0.5])
        ref_point = torch.tensor([0.0, 0.0])
        mu = 0.1

        result = smooth_chebyshev_set(Y, weights, ref_point, mu)
        assert torch.all(torch.isfinite(result))

        # Small mu
        mu_small = 1e-6
        result_small = smooth_chebyshev_set(Y, weights, ref_point, mu_small)
        assert torch.all(torch.isfinite(result_small))

