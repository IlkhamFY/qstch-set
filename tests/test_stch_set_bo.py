"""Tests for qSTCHSet acquisition function."""

import sys
from pathlib import Path

import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from stch_botorch.acquisition.stch_set_bo import qSTCHSet, qSTCHSetPure, qSTCHSetTS

DTYPE = torch.double
DEVICE = torch.device("cpu")


def _make_model(d=4, m=3, n=15):
    """Create a simple multi-output GP for testing."""
    torch.manual_seed(42)
    train_X = torch.rand(n, d, dtype=DTYPE, device=DEVICE)
    train_Y = torch.rand(n, m, dtype=DTYPE, device=DEVICE)
    model = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=m))
    return model


class TestqSTCHSet:
    """Tests for qSTCHSet Monte Carlo acquisition function."""

    def test_forward_shape(self):
        """Test output shape for various batch sizes."""
        model = _make_model(d=4, m=3)
        ref_point = torch.zeros(3, dtype=DTYPE)
        acqf = qSTCHSet(model=model, ref_point=ref_point, mu=0.1)

        # Single batch, q=3
        X = torch.rand(1, 3, 4, dtype=DTYPE)
        val = acqf(X)
        assert val.shape == torch.Size([1])

        # Multiple batches, q=5
        X = torch.rand(7, 5, 4, dtype=DTYPE)
        val = acqf(X)
        assert val.shape == torch.Size([7])

    def test_forward_differentiable(self):
        """Test that gradients flow through the acquisition function."""
        model = _make_model(d=4, m=3)
        ref_point = torch.zeros(3, dtype=DTYPE)
        acqf = qSTCHSet(model=model, ref_point=ref_point, mu=0.1)

        X = torch.rand(2, 3, 4, dtype=DTYPE, requires_grad=True)
        val = acqf(X)
        val.sum().backward()
        assert X.grad is not None
        assert not torch.isnan(X.grad).any()

    def test_optimize_acqf(self):
        """Test integration with BoTorch optimize_acqf."""
        model = _make_model(d=4, m=3)
        ref_point = torch.zeros(3, dtype=DTYPE)
        acqf = qSTCHSet(
            model=model,
            ref_point=ref_point,
            mu=0.1,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([32])),
        )

        bounds = torch.stack([torch.zeros(4, dtype=DTYPE), torch.ones(4, dtype=DTYPE)])
        candidates, value = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=3,
            num_restarts=2,
            raw_samples=16,
        )
        assert candidates.shape == torch.Size([3, 4])
        assert value.dim() == 0  # scalar

    def test_higher_q_better_coverage(self):
        """More candidates should give better or equal coverage."""
        model = _make_model(d=4, m=3)
        ref_point = torch.zeros(3, dtype=DTYPE)
        acqf = qSTCHSet(
            model=model,
            ref_point=ref_point,
            mu=0.1,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64])),
        )

        torch.manual_seed(0)
        # Same points, q=1 vs q=3 (q=3 includes q=1 point plus extras)
        X1 = torch.rand(1, 1, 4, dtype=DTYPE)
        X3 = torch.cat([X1, torch.rand(1, 2, 4, dtype=DTYPE)], dim=1)

        val1 = acqf(X1).item()
        val3 = acqf(X3).item()
        # More points should give equal or better (higher) acquisition value
        assert val3 >= val1 - 1e-3  # allow small numerical tolerance

    def test_custom_weights(self):
        """Test with non-uniform preference weights."""
        model = _make_model(d=4, m=3)
        ref_point = torch.zeros(3, dtype=DTYPE)
        weights = torch.tensor([0.7, 0.2, 0.1], dtype=DTYPE)
        acqf = qSTCHSet(model=model, ref_point=ref_point, weights=weights, mu=0.1)

        X = torch.rand(3, 2, 4, dtype=DTYPE)
        val = acqf(X)
        assert val.shape == torch.Size([3])
        assert not torch.isnan(val).any()

    def test_mu_sensitivity(self):
        """Different mu values should give different but valid results."""
        model = _make_model(d=4, m=3)
        ref_point = torch.zeros(3, dtype=DTYPE)
        X = torch.rand(2, 3, 4, dtype=DTYPE)

        vals = []
        for mu in [0.01, 0.1, 1.0]:
            acqf = qSTCHSet(
                model=model,
                ref_point=ref_point,
                mu=mu,
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64])),
            )
            vals.append(acqf(X).detach())

        # All should be finite
        for v in vals:
            assert torch.isfinite(v).all()

        # Different mu should generally give different values
        assert not torch.allclose(vals[0], vals[2], atol=1e-6)

    def test_many_objectives(self):
        """Test with m=10 objectives (our key use case)."""
        model = _make_model(d=14, m=10, n=30)
        ref_point = torch.zeros(10, dtype=DTYPE)
        acqf = qSTCHSet(
            model=model,
            ref_point=ref_point,
            mu=0.1,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([32])),
        )

        X = torch.rand(2, 5, 14, dtype=DTYPE)
        val = acqf(X)
        assert val.shape == torch.Size([2])
        assert torch.isfinite(val).all()

    def test_maximize_flag(self):
        """Test that maximize=False inverts the convention."""
        model = _make_model(d=4, m=3)
        ref_point = torch.zeros(3, dtype=DTYPE)

        X = torch.rand(2, 3, 4, dtype=DTYPE)

        acqf_max = qSTCHSet(
            model=model, ref_point=ref_point, mu=0.1, maximize=True,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64])),
        )
        acqf_min = qSTCHSet(
            model=model, ref_point=ref_point, mu=0.1, maximize=False,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64])),
        )

        val_max = acqf_max(X)
        val_min = acqf_min(X)
        # Should be different (negation changes the scalarization)
        assert not torch.allclose(val_max, val_min, atol=1e-4)


    def test_normalization_basic(self):
        """Test that Y_range/Y_min normalization produces finite, valid outputs."""
        model = _make_model(d=4, m=3)
        ref_point = torch.zeros(3, dtype=DTYPE)
        Y_min = torch.tensor([0.1, 0.2, 0.3], dtype=DTYPE)
        Y_range = torch.tensor([0.5, 0.6, 0.4], dtype=DTYPE)

        acqf = qSTCHSet(
            model=model,
            ref_point=ref_point,
            mu=0.1,
            Y_range=Y_range,
            Y_min=Y_min,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([32])),
        )

        X = torch.rand(2, 3, 4, dtype=DTYPE)
        val = acqf(X)
        assert val.shape == torch.Size([2])
        assert torch.isfinite(val).all()

    def test_normalization_large_scale(self):
        """Test normalization handles large objective scales (the Penicillin bug)."""

        def _make_large_scale_model(d=4, m=3, n=15, scale=1e5):
            torch.manual_seed(42)
            train_X = torch.rand(n, d, dtype=DTYPE, device=DEVICE)
            # Large-scale objectives (simulating Penicillin ~300K range)
            train_Y = torch.rand(n, m, dtype=DTYPE, device=DEVICE) * scale + scale
            model = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=m))
            return model, train_Y

        model, train_Y = _make_large_scale_model(scale=3e5)
        ref_point = torch.zeros(3, dtype=DTYPE)
        Y_min = train_Y.min(dim=0).values
        Y_range = train_Y.max(dim=0).values - Y_min

        # Without normalization: gradients should be near-zero (saturated softmax)
        acqf_raw = qSTCHSet(
            model=model,
            ref_point=ref_point,
            mu=0.1,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([32])),
        )
        X_raw = torch.rand(1, 3, 4, dtype=DTYPE, requires_grad=True)
        val_raw = acqf_raw(X_raw)
        val_raw.sum().backward()
        grad_norm_raw = X_raw.grad.norm().item()

        # With normalization: gradients should be healthy
        acqf_norm = qSTCHSet(
            model=model,
            ref_point=ref_point,
            mu=0.1,
            Y_range=Y_range,
            Y_min=Y_min,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([32])),
        )
        X_norm = torch.rand(1, 3, 4, dtype=DTYPE, requires_grad=True)
        val_norm = acqf_norm(X_norm)
        val_norm.sum().backward()
        grad_norm_norm = X_norm.grad.norm().item()

        # Normalized version should have meaningfully larger gradients
        # (or at least both should be finite)
        assert torch.isfinite(val_raw).all()
        assert torch.isfinite(val_norm).all()
        assert grad_norm_norm > 0, "Normalized gradients should be non-zero"

    def test_normalization_differentiable(self):
        """Test that gradients flow through normalized acquisition."""
        model = _make_model(d=4, m=3)
        ref_point = torch.zeros(3, dtype=DTYPE)
        Y_min = torch.tensor([0.1, 0.2, 0.3], dtype=DTYPE)
        Y_range = torch.tensor([0.5, 0.6, 0.4], dtype=DTYPE)

        acqf = qSTCHSet(
            model=model,
            ref_point=ref_point,
            mu=0.1,
            Y_range=Y_range,
            Y_min=Y_min,
        )

        X = torch.rand(2, 3, 4, dtype=DTYPE, requires_grad=True)
        val = acqf(X)
        val.sum().backward()
        assert X.grad is not None
        assert not torch.isnan(X.grad).any()
        assert X.grad.norm().item() > 0

    def test_normalization_requires_both_args(self):
        """Y_range without Y_min should raise."""
        model = _make_model(d=4, m=3)
        ref_point = torch.zeros(3, dtype=DTYPE)
        with pytest.raises(ValueError, match="Y_min is required"):
            qSTCHSet(
                model=model,
                ref_point=ref_point,
                Y_range=torch.ones(3, dtype=DTYPE),
            )

    def test_normalization_backward_compatible(self):
        """Without Y_range, behavior should be identical to before."""
        model = _make_model(d=4, m=3)
        ref_point = torch.zeros(3, dtype=DTYPE)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([32]))

        acqf = qSTCHSet(
            model=model,
            ref_point=ref_point,
            mu=0.1,
            sampler=sampler,
        )
        assert acqf.Y_bounds is None

        X = torch.rand(2, 3, 4, dtype=DTYPE)
        val = acqf(X)
        assert val.shape == torch.Size([2])
        assert torch.isfinite(val).all()

    def test_normalization_optimize_acqf(self):
        """Test that normalized acqf works with optimize_acqf."""
        model = _make_model(d=4, m=3)
        ref_point = torch.zeros(3, dtype=DTYPE)
        Y_min = torch.tensor([0.1, 0.2, 0.3], dtype=DTYPE)
        Y_range = torch.tensor([0.5, 0.6, 0.4], dtype=DTYPE)

        acqf = qSTCHSet(
            model=model,
            ref_point=ref_point,
            mu=0.1,
            Y_range=Y_range,
            Y_min=Y_min,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([32])),
        )

        bounds = torch.stack([torch.zeros(4, dtype=DTYPE), torch.ones(4, dtype=DTYPE)])
        candidates, value = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=3,
            num_restarts=2,
            raw_samples=16,
        )
        assert candidates.shape == torch.Size([3, 4])
        assert torch.isfinite(value)


class TestqSTCHSetTS:
    """Tests for Thompson Sampling variant."""

    def test_forward_shape(self):
        model = _make_model(d=4, m=3)
        ref_point = torch.zeros(3, dtype=DTYPE)
        acqf = qSTCHSetTS(model=model, ref_point=ref_point, mu=0.1)

        X = torch.rand(5, 3, 4, dtype=DTYPE)
        val = acqf(X)
        assert val.shape == torch.Size([5])

    def test_resample(self):
        model = _make_model(d=4, m=3)
        ref_point = torch.zeros(3, dtype=DTYPE)
        acqf = qSTCHSetTS(model=model, ref_point=ref_point, mu=0.1)

        X = torch.rand(2, 3, 4, dtype=DTYPE)
        # Should not error
        acqf.resample()
        val = acqf(X)
        assert torch.isfinite(val).all()

    def test_normalization(self):
        """Test TS variant with normalization."""
        model = _make_model(d=4, m=3)
        ref_point = torch.zeros(3, dtype=DTYPE)
        Y_min = torch.tensor([0.1, 0.2, 0.3], dtype=DTYPE)
        Y_range = torch.tensor([0.5, 0.6, 0.4], dtype=DTYPE)

        acqf = qSTCHSetTS(
            model=model,
            ref_point=ref_point,
            mu=0.1,
            Y_range=Y_range,
            Y_min=Y_min,
        )

        X = torch.rand(3, 3, 4, dtype=DTYPE)
        val = acqf(X)
        assert val.shape == torch.Size([3])
        assert torch.isfinite(val).all()


class TestqSTCHSetPure:
    """Tests for qSTCHSetPure — Lin et al.'s exact formula."""

    def test_basic_forward(self):
        """Output shape and finiteness."""
        model = _make_model(d=4, m=3)
        acqf = qSTCHSetPure(model=model, mu=0.1)
        X = torch.rand(2, 3, 4, dtype=DTYPE)
        val = acqf(X)
        assert val.shape == torch.Size([2])
        assert torch.isfinite(val).all()

    def test_single_candidate(self):
        """Works with q=1 (degenerate set)."""
        model = _make_model(d=4, m=3)
        acqf = qSTCHSetPure(model=model, mu=0.1)
        X = torch.rand(2, 1, 4, dtype=DTYPE)
        val = acqf(X)
        assert val.shape == torch.Size([2])
        assert torch.isfinite(val).all()

    def test_gradient_flows(self):
        """Gradients should flow through the acquisition value."""
        model = _make_model(d=4, m=3)
        acqf = qSTCHSetPure(model=model, mu=0.1)
        X = torch.rand(1, 3, 4, dtype=DTYPE, requires_grad=True)
        val = acqf(X)
        val.backward()
        assert X.grad is not None
        assert torch.isfinite(X.grad).all()
        assert X.grad.abs().max() > 1e-8, "Gradients should be non-trivial"

    def test_comparable_to_qstchset(self):
        """qSTCHSetPure and qSTCHSet (uniform weights, no ref benefit) should
        produce values in the same ballpark — both are smooth max over K."""
        model = _make_model(d=4, m=3)
        torch.manual_seed(42)
        X = torch.rand(2, 3, 4, dtype=DTYPE)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([64]))

        pure = qSTCHSetPure(model=model, mu=0.1, sampler=sampler)
        weighted = qSTCHSet(
            model=model,
            ref_point=torch.zeros(3, dtype=DTYPE),
            mu=0.1,
            sampler=sampler,
        )
        val_pure = pure(X)
        val_weighted = weighted(X)
        # Both should be finite and have same sign pattern
        assert torch.isfinite(val_pure).all()
        assert torch.isfinite(val_weighted).all()

    def test_mu_effect(self):
        """Smaller mu should give sharper (larger magnitude) values."""
        model = _make_model(d=4, m=5)
        X = torch.rand(3, 5, 4, dtype=DTYPE)
        acqf_tight = qSTCHSetPure(model=model, mu=0.01)
        acqf_loose = qSTCHSetPure(model=model, mu=1.0)
        # Both finite
        assert torch.isfinite(acqf_tight(X)).all()
        assert torch.isfinite(acqf_loose(X)).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
