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

    This is a direct replacement for qNParEGO / qEHVI that:
    - Scales to m >> 5 objectives (O(qm) vs exponential for HV)
    - Produces K coordinated candidates (vs independent random scalarizations)
    - Is fully differentiable (smooth log-sum-exp enables L-BFGS-B)

    The acquisition value for a set X_q = {x_1, ..., x_q} is:
        alpha(X_q) = E_f~GP[ -STCH-Set(f(X_q)) ]
    where STCH-Set = mu * log sum_i exp(lambda_i * (-mu * log sum_k exp(-f_i(x_k)/mu) - z*_i) / mu)

    Higher alpha means the set better covers all objectives (minimax sense).

    Objective normalization:
        STCH-Set's logsumexp aggregation is sensitive to the scale of
        objective values. When objectives have very different magnitudes
        (e.g., VehicleSafety ~150 vs Penicillin ~300K), a fixed mu=0.1
        causes the softmax to saturate, killing gradient flow.

        To handle arbitrary objective scales, qSTCHSet normalizes posterior
        samples to [0, 1] per objective using the observed training data
        range (Y_range). Pass ``Y_range`` to enable this. When Y_range is
        None (default), no normalization is applied -- appropriate when
        objectives are already O(1) scale (e.g., DTLZ benchmarks).

    Args:
        model: A fitted multi-output BoTorch model.
        ref_point: Reference point of shape (m,). Used as the ideal/utopia
            point z* in STCH-Set. Should be at or below the best observed
            value for each objective (minimization convention: lower is better).
            For BoTorch maximization convention, negate accordingly.
        weights: Preference weights of shape (m,). Default: uniform (1/m).
            Positive weights required. Will be normalized to sum to 1.
        mu: Smoothing parameter for STCH-Set. Controls the approximation
            tightness to true Tchebycheff. Default 0.1.
            - Smaller mu: tighter approximation, larger gradients
            - Larger mu: smoother but less faithful
            - Approximation gap <= mu * log(m) + mu * log(q)
        Y_range: Per-objective range tensor of shape (m,) for normalization.
            Computed as max(Y_obs, dim=0) - min(Y_obs, dim=0) over observed
            training data. When provided, posterior samples and ref_point are
            normalized to [0, 1] scale before scalarization, ensuring mu=0.1
            works across arbitrary objective scales. Default: None (no
            normalization, suitable for O(1)-scale objectives like DTLZ).
        Y_min: Per-objective minimum tensor of shape (m,). Required when
            Y_range is provided. The minimum observed value per objective,
            used as the normalization offset: Y_norm = (Y - Y_min) / Y_range.
        sampler: MC sampler. Default: SobolQMCNormalSampler(256).
        X_pending: Pending points of shape (p, d) already selected.
        maximize: If True (default), assumes BoTorch maximization convention
            (higher objective values are better). Internally negates for
            STCH-Set which assumes minimization.

    Example:
        >>> from botorch.models import SingleTaskGP
        >>> from botorch.optim import optimize_acqf
        >>> model = SingleTaskGP(train_X, train_Y)  # train_Y: (n, m)
        >>> # Compute normalization from training data
        >>> Y_min = train_Y.min(dim=0).values
        >>> Y_range = train_Y.max(dim=0).values - Y_min
        >>> acqf = qSTCHSet(
        ...     model=model,
        ...     ref_point=torch.zeros(m),
        ...     mu=0.1,
        ...     Y_range=Y_range,
        ...     Y_min=Y_min,
        ... )
        >>> candidates, value = optimize_acqf(
        ...     acq_function=acqf,
        ...     bounds=bounds,
        ...     q=5,
        ...     num_restarts=20,
        ...     raw_samples=512,
        ... )
    """

    def __init__(
        self,
        model: Model,
        ref_point: Tensor,
        weights: Optional[Tensor] = None,
        mu: float = 0.1,
        Y_range: Optional[Tensor] = None,
        Y_min: Optional[Tensor] = None,
        sampler: Optional[SobolQMCNormalSampler] = None,
        X_pending: Optional[Tensor] = None,
        maximize: bool = True,
    ) -> None:
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))

        # BoTorch requires an objective for multi-output models.
        # We use IdentityMCMultiOutputObjective to pass through all outputs,
        # then apply STCH-Set scalarization ourselves in forward().
        objective = IdentityMCMultiOutputObjective()
        super().__init__(model=model, sampler=sampler, objective=objective)

        m = ref_point.shape[-1]
        self.register_buffer("ref_point", ref_point)

        if weights is None:
            weights = torch.ones(m, dtype=ref_point.dtype, device=ref_point.device) / m
        self.register_buffer("weights", weights)

        self.mu = mu
        self.maximize = maximize

        # Objective normalization buffers
        if Y_range is not None:
            if Y_min is None:
                raise ValueError("Y_min is required when Y_range is provided")
            # Clamp range to avoid division by zero for constant objectives
            Y_range = torch.clamp(Y_range, min=1e-8)
            self.register_buffer("Y_range", Y_range)
            self.register_buffer("Y_min", Y_min)
        else:
            self.Y_range = None
            self.Y_min = None

        if X_pending is not None:
            self.set_X_pending(X_pending)

    def _normalize(self, Y: Tensor, ref: Tensor) -> tuple:
        """Normalize objectives to [0, 1] using observed data range.

        Args:
            Y: Objective values of shape (..., q, m) in minimization space.
            ref: Reference point of shape (m,) in minimization space.

        Returns:
            (Y_norm, ref_norm) with objectives scaled to [0, 1].
        """
        if self.Y_range is None:
            return Y, ref
        # In minimization space: lower is better.
        # Y_min and Y_range were computed from the BoTorch (maximization) space,
        # so we need to account for the negation.
        if self.maximize:
            # Y is already negated (minimization). Y_min/Y_range are in max space.
            # In max space: Y_max_obs is the best. After negation: -Y_max_obs is
            # the min in min space. range stays the same (just sign-flipped).
            # Normalize: Y_norm = (Y - min_minspace) / range
            # min_minspace = -max_maxspace = -(Y_min + Y_range)
            min_minspace = -(self.Y_min + self.Y_range)
            Y_norm = (Y - min_minspace) / self.Y_range
            ref_norm = (ref - min_minspace) / self.Y_range
        else:
            Y_norm = (Y - self.Y_min) / self.Y_range
            ref_norm = (ref - self.Y_min) / self.Y_range
        return Y_norm, ref_norm

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the STCH-Set acquisition value for candidate sets.

        Args:
            X: Candidate points of shape (batch_shape x q x d).

        Returns:
            Acquisition values of shape (batch_shape,). Higher is better.
        """
        # Get posterior samples and apply identity objective
        # samples shape: (num_samples x batch_shape x q x m)
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)
        # Apply the objective (IdentityMCMultiOutputObjective passes through)
        samples = self.objective(samples, X)

        # If maximizing (BoTorch convention), negate samples so that
        # STCH-Set (which assumes minimization) works correctly.
        if self.maximize:
            Y = -samples
            ref = -self.ref_point
        else:
            Y = samples
            ref = self.ref_point

        # Normalize objectives to [0, 1] if Y_range is provided.
        # This ensures mu=0.1 works regardless of objective scale.
        Y, ref = self._normalize(Y, ref)

        # Apply STCH-Set scalarization
        # Y shape: (num_samples x batch_shape x q x m)
        # smooth_chebyshev_set expects (..., q, m) -> (...)
        # Returns utility (higher = better set coverage)
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
        Y_range: Per-objective range for normalization. See qSTCHSet.
        Y_min: Per-objective minimum for normalization. See qSTCHSet.
        maximize: BoTorch maximization convention. Default True.
    """

    def __init__(
        self,
        model: Model,
        ref_point: Tensor,
        weights: Optional[Tensor] = None,
        mu: float = 0.1,
        Y_range: Optional[Tensor] = None,
        Y_min: Optional[Tensor] = None,
        maximize: bool = True,
    ) -> None:
        # Single sample = Thompson sampling
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

        # Objective normalization buffers
        if Y_range is not None:
            if Y_min is None:
                raise ValueError("Y_min is required when Y_range is provided")
            Y_range = torch.clamp(Y_range, min=1e-8)
            self.register_buffer("Y_range", Y_range)
            self.register_buffer("Y_min", Y_min)
        else:
            self.Y_range = None
            self.Y_min = None

    def _normalize(self, Y: Tensor, ref: Tensor) -> tuple:
        """Normalize objectives to [0, 1] using observed data range."""
        if self.Y_range is None:
            return Y, ref
        if self.maximize:
            min_minspace = -(self.Y_min + self.Y_range)
            Y_norm = (Y - min_minspace) / self.Y_range
            ref_norm = (ref - min_minspace) / self.Y_range
        else:
            Y_norm = (Y - self.Y_min) / self.Y_range
            ref_norm = (ref - self.Y_min) / self.Y_range
        return Y_norm, ref_norm

    def resample(self) -> None:
        """Draw a fresh Thompson sample on next forward() call."""
        self.sampler.base_samples = None

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate STCH-Set on a Thompson sample.

        Args:
            X: Candidate points of shape (batch_shape x q x d).

        Returns:
            Acquisition values of shape (batch_shape,).
        """
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)  # (1, batch_shape, q, m)
        samples = self.objective(samples, X)

        if self.maximize:
            Y = -samples
            ref = -self.ref_point
        else:
            Y = samples
            ref = self.ref_point

        Y, ref = self._normalize(Y, ref)

        # (1, batch_shape, q, m) -> apply STCH-Set -> (1, batch_shape) -> squeeze
        acq_values = smooth_chebyshev_set(Y=Y, weights=self.weights, ref_point=ref, mu=self.mu)
        return acq_values.squeeze(0)  # (batch_shape,)
