"""
Smooth Tchebycheff scalarization functions.

This module implements differentiable alternatives to standard Tchebycheff
scalarization for multi-objective optimization.
"""

from typing import Optional

import torch


def _normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    """Normalize weights to sum to 1.

    Args:
        weights: Weight tensor of shape (m,) or (..., m).

    Returns:
        Normalized weights with the same shape.
    """
    return weights / weights.sum(dim=-1, keepdim=True)


def _get_ideal_point(Y: torch.Tensor) -> torch.Tensor:
    """Compute ideal point (minimum along last dimension).

    Args:
        Y: Objective values of shape (..., m).

    Returns:
        Ideal point of shape (m,).
    """
    if Y.dim() == 1:
        return Y
    # For (n, m) or (..., n, m), take min along all dimensions except the last
    # This gives us the minimum value for each objective across all samples
    # Reshape to (N, m) where N is product of all leading dims, then min along dim=0
    m = Y.shape[-1]
    Y_flat = Y.view(-1, m)  # Flatten all leading dimensions
    return Y_flat.min(dim=0)[0]


def smooth_chebyshev(
    Y: torch.Tensor,
    weights: torch.Tensor,
    ref_point: Optional[torch.Tensor] = None,
    mu: float = 0.1,
) -> torch.Tensor:
    """Smooth Tchebycheff scalarization.

    BoTorch maximizes this utility. Lower Y is preferred (minimization semantics).
    Formula: -mu * log( sum( exp( weights * (Y - ref_point) / mu ) ) )

    This function approximates the max operator in standard Tchebycheff using
    LogSumExp for differentiability.

    Args:
        Y: Objective values of shape (..., m), where m is the number of objectives.
        weights: Weight vector of shape (m,) or broadcastable to (..., m).
            Will be automatically normalized to sum to 1.
        ref_point: Reference point of shape (m,). If None, uses ideal point
            (minimum of each objective).
        mu: Smoothing parameter. Smaller values make the approximation tighter
            to the true max. Default is 0.1.

    Returns:
        Scalarized utility of shape (...). Higher values are better (for maximization).

    Example:
        >>> Y = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        >>> weights = torch.tensor([0.5, 0.5])
        >>> ref_point = torch.tensor([0.0, 0.0])
        >>> utility = smooth_chebyshev(Y, weights, ref_point, mu=0.1)
    """
    # Input validation
    if Y.dim() < 1:
        raise ValueError(f"Y must have at least 1 dimension, got {Y.dim()}")
    if mu <= 0:
        raise ValueError(f"mu must be positive, got {mu}")

    # Ensure weights are positive
    weights = torch.as_tensor(weights, dtype=Y.dtype, device=Y.device)
    if (weights <= 0).any():
        raise ValueError("All weights must be positive")

    # Normalize weights
    weights = _normalize_weights(weights)

    # Get reference point (ideal point if not provided)
    if ref_point is None:
        ref_point = _get_ideal_point(Y)
    else:
        ref_point = torch.as_tensor(ref_point, dtype=Y.dtype, device=Y.device)

    # Ensure shapes are compatible
    m = Y.shape[-1]
    if ref_point.shape[-1] != m:
        raise ValueError(
            f"ref_point must have {m} elements (number of objectives), "
            f"got {ref_point.shape[-1]}"
        )
    if weights.shape[-1] != m:
        raise ValueError(
            f"weights must have {m} elements (number of objectives), "
            f"got {weights.shape[-1]}"
        )

    # Compute weighted deviations: D_i = w_i * (y_i - z*_i)
    # Per Lin et al. ICLR 2025 Eq. 5: g_STCH = mu * log(sum(exp(w_i(f_i - z*_i) / mu)))
    # Shape: (..., m)
    weighted_distances = weights * (Y - ref_point)

    # BoTorch maximizes utility, so negate the scalarization:
    # utility = -g_STCH = -mu * logsumexp(w_i(Y_i - z*_i) / mu)
    # Lower Y → lower g_STCH → higher utility (correct for minimization objectives)
    utility = -mu * torch.logsumexp(weighted_distances / mu, dim=-1)

    return utility


def smooth_chebyshev_set(
    Y: torch.Tensor,
    weights: torch.Tensor,
    ref_point: Optional[torch.Tensor] = None,
    mu: float = 0.1,
) -> torch.Tensor:
    """Smooth Tchebycheff Set scalarization ("Few for Many").

    BoTorch maximizes this utility. Lower Y is preferred (minimization semantics).
    Per Lin et al. ICLR 2025 Eq. 12.

    This function optimizes a batch of candidates to collectively cover all
    objectives. It uses nested smoothing: smooth min over the batch, then smooth
    max over objectives.

    Args:
        Y: Objective values of shape (..., q, m), where q is the batch size
            and m is the number of objectives.
        weights: Weight vector of shape (m,) or broadcastable to (..., m).
            Will be automatically normalized to sum to 1.
        ref_point: Reference point of shape (m,). If None, uses ideal point
            (minimum of each objective).
        mu: Smoothing parameter. Smaller values make the approximation tighter.
            Default is 0.1.

    Returns:
        Scalarized utility of shape (...). The q dimension is removed.
        Higher values are better (for maximization).

    Example:
        >>> Y = torch.tensor([[[1.0, 2.0], [2.0, 1.0]]])  # shape: (1, 2, 2)
        >>> weights = torch.tensor([0.5, 0.5])
        >>> ref_point = torch.tensor([0.0, 0.0])
        >>> utility = smooth_chebyshev_set(Y, weights, ref_point, mu=0.1)
    """
    # Input validation
    if Y.dim() < 2:
        raise ValueError(
            f"Y must have at least 2 dimensions (..., q, m), got {Y.dim()}"
        )
    if mu <= 0:
        raise ValueError(f"mu must be positive, got {mu}")

    # Ensure weights are positive
    weights = torch.as_tensor(weights, dtype=Y.dtype, device=Y.device)
    if (weights <= 0).any():
        raise ValueError("All weights must be positive")

    # Normalize weights
    weights = _normalize_weights(weights)

    # Get reference point (ideal point if not provided)
    if ref_point is None:
        ref_point = _get_ideal_point(Y)
    else:
        ref_point = torch.as_tensor(ref_point, dtype=Y.dtype, device=Y.device)

    # Ensure shapes are compatible
    m = Y.shape[-1]
    q = Y.shape[-2]
    if ref_point.shape[-1] != m:
        raise ValueError(
            f"ref_point must have {m} elements (number of objectives), "
            f"got {ref_point.shape[-1]}"
        )
    if weights.shape[-1] != m:
        raise ValueError(
            f"weights must have {m} elements (number of objectives), "
            f"got {weights.shape[-1]}"
        )

    # Inner aggregation: Smooth min over batch (dim=-2, the q dimension)
    # inner = -mu * logsumexp(-f_values / mu)
    # Shape: (..., m)
    inner = -mu * torch.logsumexp(-Y / mu, dim=-2)

    # Outer aggregation: Smooth max over objectives (dim=-1, the m dimension)
    # S = mu * logsumexp(weights * (inner - z*) / mu)
    # Shape: (...)
    S = mu * torch.logsumexp(weights * (inner - ref_point) / mu, dim=-1)

    # Return negative for maximization (BoTorch maximizes utility)
    # Utility = -S
    utility = -S

    return utility
