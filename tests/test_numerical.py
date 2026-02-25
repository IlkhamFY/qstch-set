"""
Numerical stability and mathematical correctness tests for stch-botorch.

Tests verify:
1. Sign convention correctness against Lin et al. (2405.19650)
2. Convergence: as mu→0, STCH → TCH (max operator)
3. Gradient flow through scalarization
4. Batch dimension handling
5. Edge cases (extreme mu, large/small objectives)
"""

import pytest
import torch
import torch.autograd as autograd

from stch_botorch.scalarization import smooth_chebyshev, smooth_chebyshev_set


class TestSignConvention:
    """Verify the sign convention matches the paper.
    
    Paper (Eq 5): g^(STCH) = μ log(Σ exp(λ_i(f_i - z*_i)/μ))  [to MINIMIZE]
    BoTorch utility = -g^(STCH)  [to MAXIMIZE]
    
    So utility should DECREASE as objectives get worse (higher for minimization).
    """

    def test_dominated_point_has_lower_utility(self):
        """A dominated point (all objectives worse) must have lower utility."""
        weights = torch.tensor([0.5, 0.5])
        ref_point = torch.tensor([0.0, 0.0])
        
        Y_good = torch.tensor([[1.0, 1.0]])  # Better (lower objectives)
        Y_bad = torch.tensor([[2.0, 2.0]])    # Worse (higher objectives)
        
        u_good = smooth_chebyshev(Y_good, weights, ref_point, mu=0.1)
        u_bad = smooth_chebyshev(Y_bad, weights, ref_point, mu=0.1)
        
        assert u_good > u_bad, (
            f"BUG: Dominated point has HIGHER utility ({u_bad.item():.4f}) "
            f"than dominating point ({u_good.item():.4f}). "
            f"Sign convention is inverted!"
        )

    def test_ideal_point_has_highest_utility(self):
        """The ideal point (ref_point itself) should have the highest utility."""
        weights = torch.tensor([0.5, 0.5])
        ref_point = torch.tensor([0.0, 0.0])
        
        Y_ideal = torch.tensor([[0.0, 0.0]])
        Y_other = torch.tensor([[1.0, 0.5]])
        
        u_ideal = smooth_chebyshev(Y_ideal, weights, ref_point, mu=0.1)
        u_other = smooth_chebyshev(Y_other, weights, ref_point, mu=0.1)
        
        assert u_ideal > u_other, (
            f"BUG: Non-ideal point has higher utility ({u_other.item():.4f}) "
            f"than ideal point ({u_ideal.item():.4f})."
        )

    def test_paper_equation_5_exact(self):
        """Directly verify against paper Eq 5 with log-additive weights.
        
        g^(STCH) = μ log(Σ w_i * exp((f_i - z*_i)/μ))
                 = μ logsumexp((f_i - z*_i)/μ + log(w_i))
        utility = -g^(STCH)
        
        Note: weights are log-additive to preserve the temperature μ.
        The old formula w*(f-z*)/μ reduced effective temperature to μ/m.
        See docs/ACQUISITION_ANALYSIS.md for derivation.
        """
        weights = torch.tensor([0.5, 0.5])
        ref_point = torch.tensor([0.0, 0.0])
        Y = torch.tensor([[3.0, 1.0]])
        mu = 0.1
        
        # Manual computation: logsumexp((f_i - z*_i)/μ + log(w_i))
        deviations = (Y - ref_point) / mu + torch.log(weights)
        g_paper = mu * torch.logsumexp(deviations, dim=-1)
        expected_utility = -g_paper
        
        actual_utility = smooth_chebyshev(Y, weights, ref_point, mu=mu)
        
        assert torch.allclose(actual_utility, expected_utility, atol=1e-6), (
            f"Code utility ({actual_utility.item():.6f}) != "
            f"paper utility ({expected_utility.item():.6f})"
        )


class TestConvergenceMuToZero:
    """As μ→0, STCH should converge to TCH (hard max)."""
    
    def test_single_solution_converges_to_tch(self):
        """smooth_chebyshev → -max_i(Y_i - z*_i) as μ→0.
        
        With log-additive weights, the limit as μ→0 is the unweighted hard max
        of (Y_i - z*_i), because μ*log(w_i) vanishes. This matches Lin et al.'s
        formulation which has no weights.
        """
        weights = torch.tensor([0.3, 0.7])
        ref_point = torch.tensor([0.0, 0.0])
        Y = torch.tensor([[2.0, 3.0]])
        
        # True TCH value (unweighted): max(2-0, 3-0) = 3.0
        # Utility = -3.0
        tch_utility = -(Y - ref_point).max(dim=-1).values
        
        mus = [1.0, 0.1, 0.01, 0.001]
        utilities = [smooth_chebyshev(Y, weights, ref_point, mu=m).item() for m in mus]
        
        # Should converge monotonically toward tch_utility
        for i in range(len(utilities) - 1):
            assert abs(utilities[i+1] - tch_utility.item()) <= abs(utilities[i] - tch_utility.item()) + 1e-8, (
                f"Not converging: mu={mus[i+1]}, util={utilities[i+1]:.6f}, "
                f"tch={tch_utility.item():.6f}"
            )
        
        # Final value should be very close
        assert abs(utilities[-1] - tch_utility.item()) < 0.01, (
            f"mu=0.001 utility ({utilities[-1]:.6f}) far from TCH ({tch_utility.item():.6f})"
        )

    def test_set_scalarization_converges(self):
        """smooth_chebyshev_set → -max_i(min_k(Y_ik - z*_i)) as μ→0.
        
        With log-additive weights, the limit is the unweighted hard max of
        the per-objective minimums (matching Lin et al.'s formulation).
        """
        weights = torch.tensor([0.5, 0.5])
        ref_point = torch.tensor([0.0, 0.0])
        # Two solutions, two objectives
        Y = torch.tensor([[[1.0, 4.0], [3.0, 2.0]]])  # (1, 2, 2)
        
        # TCH-Set (unweighted): max_i(min_k(Y_ik - z*_i))
        # obj 0: min(1,3)=1
        # obj 1: min(4,2)=2
        # max(1, 2) = 2.0, utility = -2.0
        tch_set_utility = -2.0
        
        mus = [1.0, 0.1, 0.01, 0.001]
        utilities = [smooth_chebyshev_set(Y, weights, ref_point, mu=m).item() for m in mus]
        
        assert abs(utilities[-1] - tch_set_utility) < 0.05, (
            f"mu=0.001 set utility ({utilities[-1]:.6f}) far from TCH-Set ({tch_set_utility})"
        )


class TestNumericalStability:
    """Test with extreme values."""
    
    @pytest.mark.parametrize("mu", [1e-6, 1e-3, 0.1, 1.0, 10.0])
    def test_extreme_mu_no_nan(self, mu):
        """No NaN/Inf for various mu values."""
        weights = torch.tensor([0.5, 0.5])
        ref_point = torch.tensor([0.0, 0.0])
        Y = torch.tensor([[1.0, 2.0]])
        
        result = smooth_chebyshev(Y, weights, ref_point, mu=mu)
        assert torch.isfinite(result).all(), f"Non-finite result for mu={mu}: {result}"

    def test_very_large_objectives(self):
        """Test with large objective values."""
        weights = torch.tensor([0.5, 0.5])
        ref_point = torch.tensor([0.0, 0.0])
        Y = torch.tensor([[1e6, 1e6]])
        
        result = smooth_chebyshev(Y, weights, ref_point, mu=0.1)
        assert torch.isfinite(result).all(), f"Non-finite result for large Y: {result}"

    def test_very_small_objectives(self):
        """Test with very small objective values near ideal."""
        weights = torch.tensor([0.5, 0.5])
        ref_point = torch.tensor([0.0, 0.0])
        Y = torch.tensor([[1e-10, 1e-10]])
        
        result = smooth_chebyshev(Y, weights, ref_point, mu=0.1)
        assert torch.isfinite(result).all(), f"Non-finite result for small Y: {result}"

    def test_equal_weights_symmetry(self):
        """With equal weights, swapping objectives should give same utility."""
        weights = torch.tensor([0.5, 0.5])
        ref_point = torch.tensor([0.0, 0.0])
        
        Y1 = torch.tensor([[1.0, 3.0]])
        Y2 = torch.tensor([[3.0, 1.0]])
        
        u1 = smooth_chebyshev(Y1, weights, ref_point, mu=0.1)
        u2 = smooth_chebyshev(Y2, weights, ref_point, mu=0.1)
        
        assert torch.allclose(u1, u2, atol=1e-6), (
            f"Symmetry violated: {u1.item():.6f} != {u2.item():.6f}"
        )

    def test_single_objective(self):
        """With m=1, should reduce to simple negation (utility = -(Y - z*))."""
        weights = torch.tensor([1.0])
        ref_point = torch.tensor([0.0])
        Y = torch.tensor([[5.0]])
        
        result = smooth_chebyshev(Y, weights, ref_point, mu=0.1)
        # With one objective: μ log(exp(λ(f-z*)/μ)) = λ(f-z*) = 5.0
        # utility = -5.0
        expected = -5.0
        assert torch.allclose(result, torch.tensor([expected]), atol=1e-5), (
            f"Single objective: got {result.item():.6f}, expected {expected}"
        )


class TestGradientFlow:
    """Verify gradients flow correctly through scalarization."""
    
    def test_gradient_exists(self):
        """Gradients should be non-zero and finite."""
        Y = torch.tensor([[2.0, 3.0]], requires_grad=True)
        weights = torch.tensor([0.5, 0.5])
        ref_point = torch.tensor([0.0, 0.0])
        
        utility = smooth_chebyshev(Y, weights, ref_point, mu=0.1)
        grad = autograd.grad(utility.sum(), Y)[0]
        
        assert torch.isfinite(grad).all(), f"Non-finite gradient: {grad}"
        assert (grad != 0).any(), f"Zero gradient: {grad}"

    def test_gradient_direction_minimization(self):
        """Gradient should point toward DECREASING objectives (for minimization).
        
        Since utility should increase when objectives decrease, dU/dY should be negative.
        """
        Y = torch.tensor([[2.0, 3.0]], requires_grad=True)
        weights = torch.tensor([0.5, 0.5])
        ref_point = torch.tensor([0.0, 0.0])
        
        utility = smooth_chebyshev(Y, weights, ref_point, mu=0.1)
        grad = autograd.grad(utility.sum(), Y)[0]
        
        # For correct minimization: dU/dY_i should be negative
        # (increasing Y_i should decrease utility)
        assert (grad < 0).all(), (
            f"BUG: Gradient is non-negative ({grad}). "
            f"Utility increases with Y, but should decrease for minimization."
        )

    def test_set_scalarization_gradient(self):
        """Gradients flow through set scalarization."""
        Y = torch.tensor([[[1.0, 4.0], [3.0, 2.0]]], requires_grad=True)
        weights = torch.tensor([0.5, 0.5])
        ref_point = torch.tensor([0.0, 0.0])
        
        utility = smooth_chebyshev_set(Y, weights, ref_point, mu=0.1)
        grad = autograd.grad(utility.sum(), Y)[0]
        
        assert torch.isfinite(grad).all(), f"Non-finite gradient: {grad}"


class TestBatchDimensions:
    """Test with various batch dimensions (MC samples × batch × q × m)."""
    
    def test_mc_batch_q_m(self):
        """Standard BoTorch shape: (num_mc, batch, q, m) → (num_mc, batch, q)."""
        num_mc, batch, q, m = 128, 4, 5, 3
        Y = torch.randn(num_mc, batch, q, m)
        weights = torch.tensor([0.3, 0.3, 0.4])
        ref_point = torch.zeros(m)
        
        result = smooth_chebyshev(Y, weights, ref_point, mu=0.1)
        assert result.shape == (num_mc, batch, q), f"Shape mismatch: {result.shape}"

    def test_set_scalarization_batch(self):
        """Set scalarization: (num_mc, batch, q, m) → (num_mc, batch)."""
        num_mc, batch, q, m = 64, 3, 4, 2
        Y = torch.randn(num_mc, batch, q, m)
        weights = torch.tensor([0.5, 0.5])
        ref_point = torch.zeros(m)
        
        result = smooth_chebyshev_set(Y, weights, ref_point, mu=0.1)
        assert result.shape == (num_mc, batch), f"Shape mismatch: {result.shape}"


if __name__ == "__main__":
    print("=" * 70)
    print("STCH-BoTorch Mathematical Correctness Tests")
    print("=" * 70)
    
    # Run critical sign convention tests manually
    tests = TestSignConvention()
    
    print("\n--- Sign Convention Tests ---")
    try:
        tests.test_dominated_point_has_lower_utility()
        print("✓ Dominated point has lower utility")
    except AssertionError as e:
        print(f"✗ FAIL: {e}")
    
    try:
        tests.test_ideal_point_has_highest_utility()
        print("✓ Ideal point has highest utility")
    except AssertionError as e:
        print(f"✗ FAIL: {e}")
    
    try:
        tests.test_paper_equation_5_exact()
        print("✓ Matches paper Eq 5")
    except AssertionError as e:
        print(f"✗ FAIL: {e}")
    
    print("\n--- Convergence Tests ---")
    conv = TestConvergenceMuToZero()
    try:
        conv.test_single_solution_converges_to_tch()
        print("✓ Converges to TCH as μ→0")
    except AssertionError as e:
        print(f"✗ FAIL: {e}")
    
    print("\n--- Gradient Tests ---")
    grad_tests = TestGradientFlow()
    try:
        grad_tests.test_gradient_direction_minimization()
        print("✓ Gradient direction correct for minimization")
    except AssertionError as e:
        print(f"✗ FAIL: {e}")
    
    print("\n--- Single Objective Test ---")
    stab = TestNumericalStability()
    try:
        stab.test_single_objective()
        print("✓ Single objective reduces correctly")
    except AssertionError as e:
        print(f"✗ FAIL: {e}")
