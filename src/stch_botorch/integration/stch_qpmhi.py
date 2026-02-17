"""
STCH-qPMHI Integration: Two-stage framework for batch selection.

Stage 1: Generate diverse candidate pool using STCH scalarization.
Stage 2: Select optimal batch using qPMHI ranking.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from botorch.models.model import Model
from scipy.optimize import minimize
from scipy.stats import qmc
from torch import Tensor

from stch_botorch.acquisition.qpmhi import qPMHI
from stch_botorch.scalarization import smooth_chebyshev


class STCHCandidateGenerator:
    """Generate diverse candidate pool using STCH scalarization.

    Generates candidates by optimizing STCH scalarization with multiple
    weight vectors, enabling efficient exploration of the Pareto front.

    Args:
        model: Fitted BoTorch model.
        bounds: Optimization bounds of shape (2, d).
        num_weights: Number of weight vectors to use. Default is 100.
        mu: Smoothing parameter for STCH. Default is 0.1.
        ref_point: Reference point for STCH. If None, will be computed
            from current observations.
        dedup_tol: Tolerance for candidate deduplication. Default is 1e-4.
        num_restarts: Number of random restarts per weight vector. Default is 5.
    """

    def __init__(
        self,
        model: Model,
        bounds: Tensor,
        num_weights: int = 100,
        mu: float = 0.1,
        ref_point: Optional[Tensor] = None,
        dedup_tol: float = 1e-4,
        num_restarts: int = 5,
    ) -> None:
        self.model = model
        self.bounds = bounds
        self.num_weights = num_weights
        self.mu = mu
        self.ref_point = ref_point
        self.dedup_tol = dedup_tol
        self.num_restarts = num_restarts

        # Get number of objectives from model
        if hasattr(model, "num_outputs"):
            self.num_objectives = model.num_outputs
        else:
            # Try to infer from a dummy evaluation
            dummy_X = bounds[0:1, :].unsqueeze(0)
            with torch.no_grad():
                posterior = model.posterior(dummy_X)
                self.num_objectives = posterior.mean.shape[-1]

    def generate(
        self,
        ref_point: Optional[Tensor] = None,
        warm_start: Optional[Tensor] = None,
    ) -> Tensor:
        """Generate candidate pool.

        Args:
            ref_point: Reference point for STCH. If None, uses self.ref_point
                or computes from model.
            warm_start: Optional warm-start candidates to include.

        Returns:
            Candidate pool of shape (n_candidates, d).
        """
        # Use provided ref_point or self.ref_point
        stch_ref_point = ref_point if ref_point is not None else self.ref_point

        # Generate weight vectors on simplex using Sobol sampling
        weights = self._generate_weights(self.num_weights)

        # Generate candidates for each weight
        candidates = []
        if warm_start is not None:
            candidates.append(warm_start)

        for weight in weights:
            weight_tensor = torch.tensor(weight, dtype=self.bounds.dtype, device=self.bounds.device)
            candidate = self._optimize_stch(weight_tensor, stch_ref_point)
            if candidate is not None:
                candidates.append(candidate.unsqueeze(0))

        if len(candidates) == 0:
            # Fallback: return random candidates
            n_fallback = min(self.num_weights, 100)
            return self._random_candidates(n_fallback)

        candidates_tensor = torch.cat(candidates, dim=0)

        # Deduplicate
        candidates_tensor = self._deduplicate(candidates_tensor)

        # Clip to bounds
        candidates_tensor = torch.clamp(
            candidates_tensor, min=self.bounds[0], max=self.bounds[1]
        )

        return candidates_tensor

    def _generate_weights(self, n: int) -> np.ndarray:
        """Generate weight vectors on the (m-1)-simplex using Sobol sampling.

        Args:
            n: Number of weight vectors.

        Returns:
            Weight vectors of shape (n, m) that sum to 1.
        """
        m = self.num_objectives
        if m == 1:
            return np.ones((n, 1))

        # Generate Sobol samples in (m-1) dimensions
        sampler = qmc.Sobol(d=m - 1, scramble=True, seed=42)
        samples = sampler.random(n)  # (n, m-1)

        # Transform to simplex using stick-breaking
        weights = np.zeros((n, m))
        weights[:, 0] = samples[:, 0]
        for i in range(1, m - 1):
            weights[:, i] = samples[:, i] * (1 - weights[:, :i].sum(axis=1))
        weights[:, -1] = 1 - weights[:, :-1].sum(axis=1)

        # Ensure all weights are positive (add small epsilon if needed)
        weights = np.maximum(weights, 1e-6)
        weights = weights / weights.sum(axis=1, keepdims=True)

        return weights

    def _optimize_stch(
        self, weights: Tensor, ref_point: Optional[Tensor]
    ) -> Optional[Tensor]:
        """Optimize STCH scalarization for a single weight vector.

        Args:
            weights: Weight vector of shape (m,).
            ref_point: Reference point of shape (m,). If None, computes from model.

        Returns:
            Optimal candidate of shape (d,) or None if optimization failed.
        """
        d = self.bounds.shape[1]
        bounds_np = self.bounds.cpu().numpy()

        # Compute ref_point if not provided (use model's training data ideal point)
        if ref_point is None:
            ref_point = self._compute_ideal_point()

        best_x = None
        best_value = float("inf")

        # Multiple random restarts
        for _ in range(self.num_restarts):
            # Random initial point
            x0 = np.random.uniform(bounds_np[0], bounds_np[1])

            # Objective function: negative STCH (since minimize)
            def objective(x):
                x_tensor = torch.tensor(x, dtype=self.bounds.dtype, device=self.bounds.device).unsqueeze(0)
                x_tensor = x_tensor.unsqueeze(0)  # (1, 1, d)

                with torch.no_grad():
                    posterior = self.model.posterior(x_tensor)
                    mean = posterior.mean.squeeze(0).squeeze(0)  # (m,)

                # STCH utility (to maximize), so negate for minimization
                utility = smooth_chebyshev(
                    mean.unsqueeze(0), weights, ref_point, self.mu
                )
                return -utility.item()

            try:
                result = minimize(
                    objective,
                    x0,
                    method="L-BFGS-B",
                    bounds=list(zip(bounds_np[0], bounds_np[1])),
                    options={"maxiter": 100},
                )

                if result.success and result.fun < best_value:
                    best_value = result.fun
                    best_x = result.x
            except Exception:
                continue

        if best_x is None:
            return None

        return torch.tensor(best_x, dtype=self.bounds.dtype, device=self.bounds.device)

    def _deduplicate(self, candidates: Tensor) -> Tensor:
        """Remove duplicate candidates.

        Args:
            candidates: Candidates of shape (n, d).

        Returns:
            Deduplicated candidates.
        """
        if candidates.shape[0] == 0:
            return candidates

        # Compute pairwise distances
        n = candidates.shape[0]
        keep = torch.ones(n, dtype=torch.bool, device=candidates.device)

        for i in range(n):
            if not keep[i]:
                continue
            for j in range(i + 1, n):
                if not keep[j]:
                    continue
                dist = torch.norm(candidates[i] - candidates[j])
                if dist < self.dedup_tol:
                    keep[j] = False

        return candidates[keep]

    def _compute_ideal_point(self) -> Tensor:
        """Compute ideal point from model's training data.

        Returns:
            Ideal point of shape (m,).
        """
        # Try to get training data from model
        if hasattr(self.model, "train_inputs") and self.model.train_inputs[0] is not None:
            train_X = self.model.train_inputs[0]
            if hasattr(self.model, "train_targets") and self.model.train_targets is not None:
                train_Y = self.model.train_targets
            else:
                # Evaluate model at training inputs
                with torch.no_grad():
                    posterior = self.model.posterior(train_X)
                    train_Y = posterior.mean  # (n, m)
            
            # Ideal point is minimum of each objective (for minimization semantics in STCH)
            ideal = train_Y.min(dim=0)[0]
            return ideal.to(dtype=self.bounds.dtype, device=self.bounds.device)
        else:
            # Fallback: use a default (zeros)
            return torch.zeros(
                self.num_objectives, dtype=self.bounds.dtype, device=self.bounds.device
            )

    def _random_candidates(self, n: int) -> Tensor:
        """Generate random candidates as fallback.

        Args:
            n: Number of candidates.

        Returns:
            Random candidates of shape (n, d).
        """
        d = self.bounds.shape[1]
        candidates = torch.rand(n, d, dtype=self.bounds.dtype, device=self.bounds.device)
        candidates = candidates * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        return candidates


class STCHqPMHIAcquisition:
    """Two-stage acquisition: STCH candidate generation + qPMHI batch selection.

    Args:
        model: Fitted BoTorch model.
        bounds: Optimization bounds of shape (2, d).
        pareto_Y: Current Pareto front of shape (n_pareto, m).
        ref_point: Reference point for hypervolume computation.
        num_candidates: Number of candidates to generate in stage 1.
            Default is 1000.
        stch_kwargs: Additional kwargs for STCHCandidateGenerator.
        qpmhi_kwargs: Additional kwargs for qPMHI.
    """

    def __init__(
        self,
        model: Model,
        bounds: Tensor,
        pareto_Y: Tensor,
        ref_point: Tensor,
        num_candidates: int = 1000,
        stch_kwargs: Optional[Dict] = None,
        qpmhi_kwargs: Optional[Dict] = None,
    ) -> None:
        self.model = model
        self.bounds = bounds
        self.pareto_Y = pareto_Y
        self.ref_point = ref_point
        self.num_candidates = num_candidates

        stch_kwargs = stch_kwargs or {}
        stch_kwargs.pop("num_weights", None)
        self.candidate_generator = STCHCandidateGenerator(
            model=model, bounds=bounds, num_weights=num_candidates, **stch_kwargs
        )

        qpmhi_kwargs = qpmhi_kwargs or {}
        self.qpmhi = qPMHI(
            model=model, pareto_Y=pareto_Y, ref_point=ref_point, **qpmhi_kwargs
        )

    def get_batch(self, q: int, warm_start: Optional[Tensor] = None) -> Tensor:
        """Generate and select batch of candidates.

        Args:
            q: Batch size.
            warm_start: Optional warm-start candidates.

        Returns:
            Selected batch of shape (q, d).
        """
        # Stage 1: Generate candidate pool
        candidates = self.candidate_generator.generate(warm_start=warm_start)

        # Limit pool size if needed
        if candidates.shape[0] > self.num_candidates:
            # Randomly sample to desired size
            indices = torch.randperm(candidates.shape[0])[: self.num_candidates]
            candidates = candidates[indices]

        # Stage 2: Score with qPMHI and select top-q
        with torch.no_grad():
            scores = self.qpmhi(candidates)

        # Select top-q
        _, top_indices = torch.topk(scores, k=min(q, candidates.shape[0]))
        batch = candidates[top_indices]

        return batch


def optimize_stch_qpmhi(
    model: Model,
    bounds: Tensor,
    pareto_Y: Tensor,
    ref_point: Tensor,
    q: int,
    num_candidates: int = 1000,
    stch_kwargs: Optional[Dict] = None,
    qpmhi_kwargs: Optional[Dict] = None,
) -> Tensor:
    """High-level interface for STCH-qPMHI batch selection.

    Args:
        model: Fitted BoTorch model.
        bounds: Optimization bounds of shape (2, d).
        pareto_Y: Current Pareto front of shape (n_pareto, m).
        ref_point: Reference point for hypervolume computation.
        q: Batch size.
        num_candidates: Number of candidates to generate. Default is 1000.
        stch_kwargs: Additional kwargs for STCHCandidateGenerator.
        qpmhi_kwargs: Additional kwargs for qPMHI.

    Returns:
        Selected batch of shape (q, d).
    """
    acq = STCHqPMHIAcquisition(
        model=model,
        bounds=bounds,
        pareto_Y=pareto_Y,
        ref_point=ref_point,
        num_candidates=num_candidates,
        stch_kwargs=stch_kwargs,
        qpmhi_kwargs=qpmhi_kwargs,
    )
    return acq.get_batch(q=q)
