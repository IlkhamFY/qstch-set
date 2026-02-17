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
        α(X_q) = E_f~GP[ -STCH-Set(f(X_q)) ]
    where STCH-Set = μ·log Σ_i exp(λ_i · (-μ·log Σ_k exp(-f_i(x_k)/μ) - z*_i) / μ)

    Higher α means the set better covers all objectives (minimax sense).

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
            - Smaller μ → tighter approximation, larger gradients
            - Larger μ → smoother but less faithful
            - Approximation gap ≤ μ·log(m) + μ·log(q)
        sampler: MC sampler. Default: SobolQMCNormalSampler(256).
        X_pending: Pending points of shape (p, d) already selected.
        maximize: If True (default), assumes BoTorch maximization convention
            (higher objective values are better). Internally negates for
            STCH-Set which assumes minimization.

    Example:
        >>> from botorch.models import SingleTaskGP
        >>> from botorch.optim import optimize_acqf
        >>> model = SingleTaskGP(train_X, train_Y)  # train_Y: (n, m)
        >>> acqf = qSTCHSet(
        ...     model=model,
        ...     ref_point=torch.zeros(m),  # or best observed per objective
        ...     mu=0.1,
        ... )
        >>> # Jointly optimize q=5 candidates
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

        # Apply STCH-Set scalarization
        # Y shape: (num_samples x batch_shape x q x m)
        # smooth_chebyshev_set expects (..., q, m) → (...)
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
        maximize: BoTorch maximization convention. Default True.
    """

    def __init__(
        self,
        model: Model,
        ref_point: Tensor,
        weights: Optional[Tensor] = None,
        mu: float = 0.1,
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

        # (1, batch_shape, q, m) → apply STCH-Set → (1, batch_shape) → squeeze
        acq_values = smooth_chebyshev_set(Y=Y, weights=self.weights, ref_point=ref, mu=self.mu)
        return acq_values.squeeze(0)  # (batch_shape,)
