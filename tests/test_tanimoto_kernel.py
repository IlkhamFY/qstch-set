"""Tests for Tanimoto kernel."""
import torch
import pytest
from stch_botorch.kernels.tanimoto import TanimotoKernel


class TestTanimotoKernel:
    """Test suite for TanimotoKernel."""

    def test_identical_vectors(self):
        """k(x, x) = 1 for any non-zero x."""
        kernel = TanimotoKernel()
        x = torch.tensor([[1, 0, 1, 1, 0], [0, 1, 1, 0, 1]], dtype=torch.double)
        K = kernel(x, x).to_dense()
        # Diagonal should be 1.0
        assert torch.allclose(K.diag(), torch.ones(2, dtype=torch.double), atol=1e-6)

    def test_disjoint_vectors(self):
        """k(x, x') = 0 when x and x' have no overlap."""
        kernel = TanimotoKernel()
        x1 = torch.tensor([[1, 1, 0, 0]], dtype=torch.double)
        x2 = torch.tensor([[0, 0, 1, 1]], dtype=torch.double)
        K = kernel(x1, x2).to_dense()
        assert torch.allclose(K, torch.zeros(1, 1, dtype=torch.double), atol=1e-6)

    def test_known_value(self):
        """Test against hand-computed Tanimoto similarity."""
        kernel = TanimotoKernel()
        # x1 = {1,1,0,1,0}, x2 = {1,0,1,1,0}
        # dot = 1*1 + 1*0 + 0*1 + 1*1 + 0*0 = 2
        # |x1|^2 = 3, |x2|^2 = 3
        # Tanimoto = 2 / (3 + 3 - 2) = 2/4 = 0.5
        x1 = torch.tensor([[1, 1, 0, 1, 0]], dtype=torch.double)
        x2 = torch.tensor([[1, 0, 1, 1, 0]], dtype=torch.double)
        K = kernel(x1, x2).to_dense()
        assert torch.allclose(K, torch.tensor([[0.5]], dtype=torch.double), atol=1e-6)

    def test_symmetry(self):
        """K(x1, x2) should be symmetric."""
        kernel = TanimotoKernel()
        x = torch.randint(0, 2, (5, 20), dtype=torch.double)
        K = kernel(x, x).to_dense()
        assert torch.allclose(K, K.T, atol=1e-6)

    def test_positive_semidefinite(self):
        """Kernel matrix should be PSD."""
        kernel = TanimotoKernel()
        x = torch.randint(0, 2, (10, 50), dtype=torch.double)
        K = kernel(x, x).to_dense()
        eigenvalues = torch.linalg.eigvalsh(K)
        # All eigenvalues >= -epsilon (numerical tolerance)
        assert (eigenvalues > -1e-5).all(), f"Negative eigenvalue: {eigenvalues.min()}"

    def test_diag_mode(self):
        """diag=True should return only diagonal."""
        kernel = TanimotoKernel()
        x = torch.randint(0, 2, (8, 30), dtype=torch.double)
        K_full = kernel(x, x).to_dense()
        K_diag = kernel(x, x, diag=True)
        assert K_diag.shape == (8,)
        assert torch.allclose(K_diag, K_full.diag(), atol=1e-6)

    def test_batch_mode(self):
        """Should handle batch dimensions."""
        kernel = TanimotoKernel()
        x1 = torch.randint(0, 2, (3, 5, 20), dtype=torch.double)
        x2 = torch.randint(0, 2, (3, 7, 20), dtype=torch.double)
        K = kernel(x1, x2).to_dense()
        assert K.shape == (3, 5, 7)

    def test_values_in_unit_interval(self):
        """All kernel values should be in [0, 1]."""
        kernel = TanimotoKernel()
        x = torch.randint(0, 2, (20, 100), dtype=torch.double)
        K = kernel(x, x).to_dense()
        assert (K >= -1e-6).all()
        assert (K <= 1.0 + 1e-6).all()

    def test_count_vectors(self):
        """Should work with count (non-binary) vectors too."""
        kernel = TanimotoKernel()
        x1 = torch.tensor([[2, 0, 3, 1]], dtype=torch.double)
        x2 = torch.tensor([[1, 1, 2, 0]], dtype=torch.double)
        # dot = 2*1 + 0*1 + 3*2 + 1*0 = 8
        # |x1|^2 = 4+0+9+1 = 14, |x2|^2 = 1+1+4+0 = 6
        # Tanimoto = 8 / (14 + 6 - 8) = 8/12 = 2/3
        K = kernel(x1, x2).to_dense()
        assert torch.allclose(K, torch.tensor([[2.0 / 3.0]], dtype=torch.double), atol=1e-6)

    def test_zero_vector_handling(self):
        """k(0, x) should be 0 (no structural overlap)."""
        kernel = TanimotoKernel()
        x1 = torch.zeros(1, 10, dtype=torch.double)
        x2 = torch.ones(1, 10, dtype=torch.double)
        K = kernel(x1, x2).to_dense()
        assert torch.allclose(K, torch.zeros(1, 1, dtype=torch.double), atol=1e-6)

    def test_with_scale_kernel(self):
        """Should compose with ScaleKernel for GP use."""
        import gpytorch
        kernel = gpytorch.kernels.ScaleKernel(TanimotoKernel())
        x = torch.randint(0, 2, (5, 20), dtype=torch.double)
        K = kernel(x, x).to_dense()
        assert K.shape == (5, 5)
        # ScaleKernel multiplies by outputscale, so values can exceed 1
        assert (K >= -1e-6).all()

    def test_gp_integration(self):
        """Tanimoto kernel should work inside a SingleTaskGP."""
        import gpytorch
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize

        torch.manual_seed(42)
        n, d = 20, 50
        X = torch.randint(0, 2, (n, d), dtype=torch.double)
        Y = torch.randn(n, 1, dtype=torch.double)

        covar = gpytorch.kernels.ScaleKernel(TanimotoKernel())
        model = SingleTaskGP(X, Y, covar_module=covar, outcome_transform=Standardize(m=1))

        # Should be able to compute posterior
        model.eval()
        with torch.no_grad():
            posterior = model.posterior(X[:3])
            mean = posterior.mean
            var = posterior.variance
        assert mean.shape == (3, 1)
        assert (var > 0).all()
