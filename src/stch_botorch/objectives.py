"""
BoTorch objective wrappers for Smooth Tchebycheff scalarization.

This module provides objective classes that can be used with BoTorch
acquisition functions for multi-objective Bayesian optimization.
"""

from typing import Optional

import torch
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective

from stch_botorch.scalarization import smooth_chebyshev, smooth_chebyshev_set


class SmoothChebyshevObjective(MCMultiOutputObjective):
    """Smooth Tchebycheff objective for multi-output optimization.

    BoTorch maximizes this utility. Lower Y is preferred (minimization semantics).
    Formula: -mu * log( sum( exp( weights * (Y - ref_point) / mu ) ) )

    This objective maps samples of shape (sample_shape x batch_shape x q x m)
    to shape (sample_shape x batch_shape x q), where:
    - sample_shape: Monte Carlo sample dimension
    - batch_shape: Batch dimension
    - q: Number of candidates
    - m: Number of objectives

    Args:
        weights: Weight vector of shape (m,). Will be automatically normalized.
        ref_point: Reference point of shape (m,). If None, uses ideal point.
        mu: Smoothing parameter. Default is 0.1.

    Example:
        >>> from botorch.acquisition import qLogNParEGO
        >>> objective = SmoothChebyshevObjective(
        ...     weights=torch.tensor([0.5, 0.5]),
        ...     ref_point=torch.tensor([0.0, 0.0]),
        ...     mu=0.1
        ... )
        >>> acq = qLogNParEGO(model=model, ref_point=ref_point, objective=objective)
    """

    def __init__(
        self,
        weights: torch.Tensor,
        ref_point: Optional[torch.Tensor] = None,
        mu: float = 0.1,
    ) -> None:
        super().__init__()
        self.weights = weights
        self.ref_point = ref_point
        self.mu = mu

    def forward(self, samples: torch.Tensor, X: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Transform multi-output samples to scalarized utility.

        Args:
            samples: Monte Carlo samples of shape (sample_shape x batch_shape x q x m).
            X: Input points (optional, not used but required by base class).

        Returns:
            Scalarized utility of shape (sample_shape x batch_shape x q).
        """
        # samples shape: (sample_shape x batch_shape x q x m)
        # smooth_chebyshev expects (..., m) and returns (...)
        # We want to apply it along the q dimension, keeping MC and batch dimensions
        # Reshape to (..., q, m) where ... = (sample_shape x batch_shape)
        original_shape = samples.shape
        
        # Flatten leading dimensions for processing
        # (sample_shape x batch_shape x q x m) -> (N x q x m)
        N = samples.numel() // (samples.shape[-2] * samples.shape[-1])
        q = samples.shape[-2]
        m = samples.shape[-1]
        samples_flat = samples.view(N, q, m)
        
        # Apply smooth_chebyshev to each (q x m) slice
        # This returns (N x q)
        utilities = smooth_chebyshev(
            Y=samples_flat,
            weights=self.weights,
            ref_point=self.ref_point,
            mu=self.mu,
        )
        
        # Reshape back: (N x q) -> (sample_shape x batch_shape x q)
        return utilities.view(*original_shape[:-1])


class SmoothChebyshevSetObjective(MCMultiOutputObjective):
    """Smooth Tchebycheff Set objective for batch optimization.

    Advanced use only. Returns (sample_shape x batch_shape), not standard (... x q) shape.
    Use with `qSimpleRegret` or `FixedFeatureAcquisition`, not standard `qEI`.
    BoTorch maximizes this utility. Lower Y is preferred (minimization semantics).
    Per Lin et al. ICLR 2025 Eq. 12.

    This objective aggregates over the q dimension internally, mapping samples of shape
    (sample_shape x batch_shape x q x m) to shape (sample_shape x batch_shape).

    Args:
        weights: Weight vector of shape (m,). Will be automatically normalized.
        ref_point: Reference point of shape (m,). If None, uses ideal point.
        mu: Smoothing parameter. Default is 0.1.

    Example:
        >>> from botorch.acquisition import qSimpleRegret
        >>> objective = SmoothChebyshevSetObjective(
        ...     weights=torch.tensor([0.5, 0.5]),
        ...     ref_point=torch.tensor([0.0, 0.0]),
        ...     mu=0.1
        ... )
        >>> acq = qSimpleRegret(model=model, objective=objective)
    """

    def __init__(
        self,
        weights: torch.Tensor,
        ref_point: Optional[torch.Tensor] = None,
        mu: float = 0.1,
    ) -> None:
        super().__init__()
        self.weights = weights
        self.ref_point = ref_point
        self.mu = mu

    def forward(self, samples: torch.Tensor, X: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Transform multi-output samples to scalarized utility (aggregated over q).

        Args:
            samples: Monte Carlo samples of shape (sample_shape x batch_shape x q x m).
            X: Input points (optional, not used but required by base class).

        Returns:
            Scalarized utility of shape (sample_shape x batch_shape).
            The q dimension is removed by aggregation.
        """
        # samples shape: (sample_shape x batch_shape x q x m)
        # smooth_chebyshev_set expects (..., q, m) and returns (...)
        # We want to apply it to each MC/batch combination
        original_shape = samples.shape
        
        # Flatten leading dimensions for processing
        # (sample_shape x batch_shape x q x m) -> (N x q x m)
        N = samples.numel() // (samples.shape[-2] * samples.shape[-1])
        q = samples.shape[-2]
        m = samples.shape[-1]
        samples_flat = samples.view(N, q, m)
        
        # Apply smooth_chebyshev_set to each (q x m) slice
        # This returns (N,) - the q dimension is removed
        utilities = smooth_chebyshev_set(
            Y=samples_flat,
            weights=self.weights,
            ref_point=self.ref_point,
            mu=self.mu,
        )
        
        # Reshape back: (N,) -> (sample_shape x batch_shape)
        return utilities.view(*original_shape[:-2])

