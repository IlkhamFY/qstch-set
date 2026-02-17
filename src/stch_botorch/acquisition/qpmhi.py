"""
qPMHI: Probability of Maximum Hypervolume Improvement.

Multi-point Probability of Maximum Hypervolume Improvement (qPMHI) acquisition
function for pool-based batch selection in multi-objective Bayesian optimization.
"""

from typing import Optional

import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume
from torch import Tensor


class qPMHI(MCAcquisitionFunction):
    """Probability of Maximum Hypervolume Improvement (qPMHI).

    qPMHI scores each candidate in a pool by the probability that it achieves
    the maximum hypervolume improvement. This enables exact batch selection
    by ranking candidates and selecting the top-q.

    Args:
        model: A fitted BoTorch model.
        pareto_Y: Current Pareto front of shape (n_pareto, m) where m is
            the number of objectives. Objectives should follow BoTorch's
            maximization convention (higher is better).
        ref_point: Reference point for hypervolume computation of shape (m,).
            Should be dominated by all feasible outcomes.
        sampler: Monte Carlo sampler. Defaults to SobolQMCNormalSampler
            with 512 samples.
        objective: Optional objective transform (MCAcquisitionObjective).
            If None, uses raw model outputs.
        max_candidates_per_chunk: Optional chunking for memory efficiency.
            If None, processes all candidates at once.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> pareto_Y = compute_pareto_front(train_Y)
        >>> ref_point = torch.tensor([0.0, 0.0])
        >>> qpmhi = qPMHI(model, pareto_Y, ref_point)
        >>> candidates = torch.rand(100, d)  # 100 candidates
        >>> scores = qpmhi(candidates)  # Shape: (100,)
        >>> top_q = candidates[torch.topk(scores, k=5).indices]
    """

    def __init__(
        self,
        model: Model,
        pareto_Y: Tensor,
        ref_point: Tensor,
        sampler: Optional[SobolQMCNormalSampler] = None,
        objective: Optional[torch.nn.Module] = None,
        max_candidates_per_chunk: Optional[int] = None,
    ) -> None:
        # Validate inputs
        if pareto_Y.dim() != 2:
            raise ValueError(f"pareto_Y must be 2D, got shape {pareto_Y.shape}")
        if ref_point.dim() != 1:
            raise ValueError(f"ref_point must be 1D, got shape {ref_point.shape}")
        if pareto_Y.shape[-1] != ref_point.shape[-1]:
            raise ValueError(
                f"pareto_Y and ref_point must have same number of objectives, "
                f"got {pareto_Y.shape[-1]} and {ref_point.shape[-1]}"
            )

        # Set default sampler
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))

        # For multi-output models, we need an objective to avoid BoTorch error
        if objective is None and model.num_outputs > 1:
            objective = IdentityMCMultiOutputObjective()

        # Initialize parent class
        super().__init__(model=model, sampler=sampler, objective=objective)

        self.pareto_Y = pareto_Y
        self.ref_point = ref_point
        self.max_candidates_per_chunk = max_candidates_per_chunk

        # Initialize hypervolume calculator for baseline
        self._hv_calculator = Hypervolume(ref_point=ref_point)
        # Filter pareto_Y to ensure it's actually Pareto-optimal
        self._pareto_Y_filtered = self._filter_pareto(pareto_Y)
        self._baseline_hv = self._hv_calculator.compute(self._pareto_Y_filtered)

    def forward(self, X: Tensor) -> Tensor:
        """Compute qPMHI probability scores for candidates.

        Args:
            X: Candidate points of shape (n, d) or (batch, n, d).
                If batched, processes each batch independently.

        Returns:
            Probability scores of shape (n,) or (batch, n).
            Each score is P(candidate achieves max HVI).
        """
        # Handle batched input
        if X.dim() == 3:
            batch_shape = X.shape[:-2]
            X_flat = X.view(-1, X.shape[-2], X.shape[-1])
            scores_flat = torch.stack([self._forward_single(x) for x in X_flat])
            return scores_flat.view(*batch_shape, X.shape[-2])

        return self._forward_single(X)

    def _forward_single(self, X: Tensor) -> Tensor:
        """Compute scores for a single batch of candidates.

        Args:
            X: Candidates of shape (n, d).

        Returns:
            Scores of shape (n,).
        """
        n = X.shape[0]
        if n == 0:
            return torch.tensor([], dtype=X.dtype, device=X.device)

        # Get posterior samples
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)  # Shape: (sample_shape, n, m)

        # Apply objective transform if provided
        if self.objective is not None:
            samples = self.objective(samples, X=X)

        # Ensure samples are 3D: (num_samples, n, m)
        if samples.dim() == 2:
            samples = samples.unsqueeze(0)
        elif samples.dim() == 4:
            # Handle case where samples have q dimension: (sample_shape, batch, q, m)
            # Flatten batch and q dimensions
            samples = samples.view(samples.shape[0], -1, samples.shape[-1])
        num_samples, n_candidates, m = samples.shape

        # Compute HVI for each sample and candidate
        hvi_per_sample = torch.zeros(num_samples, n_candidates, dtype=X.dtype, device=X.device)

        # Process in chunks if specified
        if self.max_candidates_per_chunk is not None and n_candidates > self.max_candidates_per_chunk:
            for i in range(0, n_candidates, self.max_candidates_per_chunk):
                end_idx = min(i + self.max_candidates_per_chunk, n_candidates)
                chunk_samples = samples[:, i:end_idx, :]
                chunk_hvi = self._compute_hvi_batch(chunk_samples)
                hvi_per_sample[:, i:end_idx] = chunk_hvi
        else:
            hvi_per_sample = self._compute_hvi_batch(samples)

        # For each MC sample, find which candidate(s) achieve max HVI
        # Assign probability mass (split evenly if ties)
        max_hvi_per_sample = hvi_per_sample.max(dim=1, keepdim=True)[0]  # (num_samples, 1)
        # Use tolerance for numerical stability
        is_max = (hvi_per_sample >= max_hvi_per_sample - 1e-6).float()  # (num_samples, n)
        # Normalize so each sample contributes 1.0 total probability
        num_max_per_sample = is_max.sum(dim=1, keepdim=True).clamp(min=1.0)  # (num_samples, 1)
        prob_per_sample = is_max / num_max_per_sample  # (num_samples, n)

        # Average over MC samples
        scores = prob_per_sample.mean(dim=0)  # (n,)

        return scores

    def _compute_hvi_batch(self, samples: Tensor) -> Tensor:
        """Compute HVI for all candidates across all MC samples.

        Args:
            samples: Posterior samples of shape (num_samples, n, m).

        Returns:
            HVI values of shape (num_samples, n).
        """
        num_samples, n, m = samples.shape
        device = samples.device
        dtype = samples.dtype

        hvi = torch.zeros(num_samples, n, dtype=dtype, device=device)

        for s in range(num_samples):
            sample_y = samples[s]  # (n, m)

            for i in range(n):
                candidate_y = sample_y[i : i + 1]  # (1, m)

                # Check if candidate is dominated by current Pareto front
                if self._pareto_Y_filtered.shape[0] > 0:
                    # For maximization: candidate is dominated if any pareto point >= candidate (all) and > candidate (any)
                    dominated = (
                        (self._pareto_Y_filtered >= candidate_y).all(dim=1)
                        & (self._pareto_Y_filtered > candidate_y).any(dim=1)
                    )
                    if dominated.any():
                        hvi[s, i] = 0.0
                        continue

                # Compute hypervolume of (pareto_Y ∪ {candidate})
                extended_pareto = torch.cat([self._pareto_Y_filtered, candidate_y], dim=0)
                extended_pareto_filtered = self._filter_pareto(extended_pareto)

                # Compute hypervolume
                if extended_pareto_filtered.shape[0] == 0:
                    hv_extended = 0.0
                else:
                    hv_extended = self._hv_calculator.compute(extended_pareto_filtered)

                # HVI is the difference
                hvi[s, i] = hv_extended - self._baseline_hv

        return hvi

    @staticmethod
    def _filter_pareto(Y: Tensor) -> Tensor:
        """Filter to keep only Pareto-optimal points (maximization).

        Args:
            Y: Objective values of shape (n, m).

        Returns:
            Pareto-optimal subset of shape (n_pareto, m).
        """
        if Y.shape[0] == 0:
            return Y

        n, m = Y.shape
        is_pareto = torch.ones(n, dtype=torch.bool, device=Y.device)

        for i in range(n):
            if not is_pareto[i]:
                continue
            y_i = Y[i : i + 1]  # (1, m)

            # Check if any other point dominates y_i
            # For maximization: point j dominates i if Y[j] >= Y[i] (all) and Y[j] > Y[i] (any)
            for j in range(i + 1, n):
                if not is_pareto[j]:
                    continue
                y_j = Y[j : j + 1]  # (1, m)

                # Check if y_j dominates y_i
                if (y_j >= y_i).all() and (y_j > y_i).any():
                    is_pareto[i] = False
                    break

                # Check if y_i dominates y_j
                if (y_i >= y_j).all() and (y_i > y_j).any():
                    is_pareto[j] = False

        return Y[is_pareto]
