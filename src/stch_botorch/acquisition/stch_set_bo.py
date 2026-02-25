"""
STCH-Set Bayesian Optimization Acquisition Function.

The core novel contribution: adapting Lin et al.'s STCH-Set scalarization
(ICLR 2025) to sample-efficient Bayesian optimization of expensive black-box
multi-objective problems.

Key idea: Jointly optimize K candidates to collectively cover m objectives
by applying STCH-Set to GP posterior samples. The smooth log-sum-exp
formulation enables gradient-based acquisition optimization via L-BFGS-B,
and scales as O(Km) — independent of hypervolume computation.

This fills the empty cell in the 2x2 matrix:
                    Single Solution    Set of K Solutions
  Gradient-based:   STCH (Lin ICML24)  STCH-Set (Lin ICLR25)
  BO (expensive):   Pires&Coelho 2025  **THIS (STCH-Set-BO)**
"""

from typing import Optional, Union

import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
)
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    normalize,
    t_batch_mode_transform,
)
from torch import Tensor

from stch_botorch.scalarization import smooth_chebyshev_set


class qSTCHSet(MCAcquisitionFunction):
    """Monte Carlo STCH-Set acquisition function for many-objective BO.

    Jointly evaluates a batch of q candidates by their collective coverage
    of m objectives under GP posterior uncertainty. Uses STCH-Set scalarization
    (smooth min over candidates, smooth max over objectives) applied to
    MC posterior samples.

    Objective normalization:
        Following BoTorch's get_chebyshev_scalarization pattern, this
        acquisition function normalizes posterior samples to [0,1] using
        observed training data bounds (Y_bounds). This ensures the STCH-Set
        scalarization operates on comparable scales regardless of raw
        objective magnitudes.

        Pass ``Y_bounds`` (a 2×m tensor of [min, max] per objective) to enable
        normalization. When Y_bounds is None, no normalization is applied —
        appropriate when objectives are already O(1) scale (e.g., DTLZ
        benchmarks with Standardize).

    Args:
        model: A fitted multi-output BoTorch model.
        ref_point: Reference point of shape (m,). In BoTorch maximization
            convention (higher is better). Used as z* in STCH-Set after
            normalization.
        weights: Preference weights of shape (m,). Default: uniform (1/m).
        mu: Smoothing parameter for STCH-Set. Default 0.1.
        Y_bounds: A (2, m) tensor with [Y_min, Y_max] per objective in
            BoTorch maximization space. Used to normalize posterior samples
            to [0,1]. Computed from training data as:
                Y_bounds = torch.stack([Y.min(dim=0).values, Y.max(dim=0).values])
            When provided, ref_point is also normalized. This matches the
            normalization in BoTorch's get_chebyshev_scalarization.
        sampler: MC sampler. Default: SobolQMCNormalSampler(256).
        X_pending: Pending points of shape (p, d) already selected.
        maximize: If True (default), assumes BoTorch maximization convention.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> Y_bounds = torch.stack([train_Y.min(0).values, train_Y.max(0).values])
        >>> acqf = qSTCHSet(
        ...     model=model,
        ...     ref_point=ref_point,
        ...     mu=0.1,
        ...     Y_bounds=Y_bounds,
        ... )
    """

    def __init__(
        self,
        model: Model,
        ref_point: Tensor,
        weights: Optional[Tensor] = None,
        mu: float = 0.1,
        Y_bounds: Optional[Tensor] = None,
        # Legacy API (deprecated, use Y_bounds instead):
        Y_range: Optional[Tensor] = None,
        Y_min: Optional[Tensor] = None,
        sampler: Optional[SobolQMCNormalSampler] = None,
        X_pending: Optional[Tensor] = None,
        maximize: bool = True,
    ) -> None:
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))

        objective = IdentityMCMultiOutputObjective()
        super().__init__(model=model, sampler=sampler, objective=objective)

        m = ref_point.shape[-1]
        self.register_buffer("ref_point", ref_point)

        if weights is None:
            weights = torch.ones(m, dtype=ref_point.dtype, device=ref_point.device) / m
        self.register_buffer("weights", weights)

        self.mu = mu
        self.maximize = maximize

        # Handle Y_bounds (new API) or Y_range/Y_min (legacy)
        if Y_bounds is not None:
            self.register_buffer("Y_bounds", Y_bounds)
        elif Y_range is not None:
            # Legacy: construct Y_bounds from Y_min and Y_range
            if Y_min is None:
                raise ValueError("Y_min is required when Y_range is provided")
            Y_range = torch.clamp(Y_range, min=1e-8)
            Y_bounds = torch.stack([Y_min, Y_min + Y_range])
            self.register_buffer("Y_bounds", Y_bounds)
        else:
            self.Y_bounds = None

        if X_pending is not None:
            self.set_X_pending(X_pending)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the STCH-Set acquisition value for candidate sets.

        Args:
            X: Candidate points of shape (batch_shape x q x d).

        Returns:
            Acquisition values of shape (batch_shape,). Higher is better.
        """
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)
        samples = self.objective(samples, X)
        # samples shape: (num_samples x batch_shape x q x m)

        if self.maximize:
            Y = -samples  # Convert to minimization space for STCH-Set
            ref = -self.ref_point
        else:
            Y = samples
            ref = self.ref_point

        # Normalize to [0,1] using observed data bounds.
        # This matches BoTorch's get_chebyshev_scalarization(weights, Y=pred):
        #   Y_normalized = normalize(-Y, bounds=Y_bounds)
        # In our case, Y is already negated (minimization space), so we negate
        # Y_bounds accordingly.
        if self.Y_bounds is not None:
            # Y_bounds is in maximization space: [Y_min_max, Y_max_max]
            # In minimization space: min = -Y_max_max, max = -Y_min_max
            bounds_min = -self.Y_bounds[1]  # -Y_max in max space = min in min space
            bounds_max = -self.Y_bounds[0]  # -Y_min in max space = max in min space
            # Clamp range to avoid division by zero
            obj_range = torch.clamp(bounds_max - bounds_min, min=1e-8)
            # Normalize Y and ref to [0, 1]
            Y = (Y - bounds_min) / obj_range
            ref = (ref - bounds_min) / obj_range

        # Apply STCH-Set scalarization
        acq_values = smooth_chebyshev_set(
            Y=Y,
            weights=self.weights,
            ref_point=ref,
            mu=self.mu,
        )

        # Average over MC samples (dim 0)
        return acq_values.mean(dim=0)


class qSTCHSetTS(MCAcquisitionFunction):
    """Thompson Sampling variant of STCH-Set for many-objective BO.

    Uses a single MC sample (Thompson sample) instead of averaging over many.
    Cheaper per evaluation but noisier. Use with multiple random restarts.

    Args:
        model: A fitted multi-output BoTorch model.
        ref_point: Reference point of shape (m,).
        weights: Preference weights of shape (m,). Default: uniform.
        mu: Smoothing parameter. Default 0.1.
        Y_bounds: A (2, m) tensor for normalization. See qSTCHSet.
        maximize: BoTorch maximization convention. Default True.
    """

    def __init__(
        self,
        model: Model,
        ref_point: Tensor,
        weights: Optional[Tensor] = None,
        mu: float = 0.1,
        Y_bounds: Optional[Tensor] = None,
        Y_range: Optional[Tensor] = None,
        Y_min: Optional[Tensor] = None,
        maximize: bool = True,
    ) -> None:
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1]))
        objective = IdentityMCMultiOutputObjective()
        super().__init__(model=model, sampler=sampler, objective=objective)

        m = ref_point.shape[-1]
        self.register_buffer("ref_point", ref_point)

        if weights is None:
            weights = torch.ones(m, dtype=ref_point.dtype, device=ref_point.device) / m
        self.register_buffer("weights", weights)

        self.mu = mu
        self.maximize = maximize

        if Y_bounds is not None:
            self.register_buffer("Y_bounds", Y_bounds)
        elif Y_range is not None:
            if Y_min is None:
                raise ValueError("Y_min is required when Y_range is provided")
            Y_range = torch.clamp(Y_range, min=1e-8)
            Y_bounds = torch.stack([Y_min, Y_min + Y_range])
            self.register_buffer("Y_bounds", Y_bounds)
        else:
            self.Y_bounds = None

    def resample(self) -> None:
        """Draw a fresh Thompson sample on next forward() call."""
        self.sampler.base_samples = None

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate STCH-Set on a Thompson sample."""
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)
        samples = self.objective(samples, X)

        if self.maximize:
            Y = -samples
            ref = -self.ref_point
        else:
            Y = samples
            ref = self.ref_point

        if self.Y_bounds is not None:
            bounds_min = -self.Y_bounds[1]
            bounds_max = -self.Y_bounds[0]
            obj_range = torch.clamp(bounds_max - bounds_min, min=1e-8)
            Y = (Y - bounds_min) / obj_range
            ref = (ref - bounds_min) / obj_range

        acq_values = smooth_chebyshev_set(Y=Y, weights=self.weights, ref_point=ref, mu=self.mu)
        return acq_values.squeeze(0)
