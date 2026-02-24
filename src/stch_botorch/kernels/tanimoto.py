"""
Tanimoto (Jaccard) similarity kernel for molecular fingerprints.

The Tanimoto kernel is the standard GP kernel for binary molecular
fingerprints in cheminformatics:

    k(x, x') = (x . x') / (|x|^2 + |x'|^2 - x . x')

where x, x' are binary (or count) fingerprint vectors. This is a proper
positive-definite kernel (it's the intersection kernel on sets, which is
PD by Haussler's convolution theorem).

Why not RBF? RBF on binary vectors treats Hamming distance as Euclidean,
which is geometrically wrong for sparse binary vectors (most bits are 0).
Tanimoto measures structural overlap directly: two molecules sharing 80%
of their substructures get k=0.8, regardless of the 1024-dim embedding.

References:
    - Ralaivola et al. (2005): "Graph kernels for chemical informatics"
    - Griffiths & Hernandez-Lobato (2020): "Constrained Bayesian optimization
      for automatic chemical design using variational autoencoders"
"""

import torch
from gpytorch.kernels import Kernel
from torch import Tensor


class TanimotoKernel(Kernel):
    """Tanimoto (Jaccard) kernel for binary/count fingerprint vectors.

    k(x, x') = (x . x') / (|x|^2 + |x'|^2 - x . x')

    For binary vectors, this equals the Jaccard index:
        |A ∩ B| / |A ∪ B|

    Works with:
        - Binary Morgan/ECFP fingerprints (standard)
        - Count Morgan fingerprints
        - Any non-negative feature vectors

    The kernel is stationary in fingerprint space and outputs values
    in [0, 1], with k(x, x) = 1 for all non-zero x.

    Compatible with GPyTorch's kernel API: use as a drop-in replacement
    for RBFKernel in SingleTaskGP or other BoTorch models.

    Example::

        from stch_botorch.kernels import TanimotoKernel
        from botorch.models import SingleTaskGP

        model = SingleTaskGP(
            train_X, train_Y,
            covar_module=gpytorch.kernels.ScaleKernel(TanimotoKernel()),
        )
    """

    has_lengthscale = False

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> Tensor:
        """Compute Tanimoto kernel matrix.

        Args:
            x1: (... x n x d) tensor
            x2: (... x m x d) tensor
            diag: if True, return only diagonal elements (n == m required)

        Returns:
            (... x n x m) kernel matrix, or (... x n) if diag=True
        """
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        if diag:
            # Diagonal: k(x_i, x_i) for paired inputs
            # For binary vectors, k(x, x) = |x|^2 / |x|^2 = 1 (if x != 0)
            dot = (x1 * x2).sum(dim=-1)
            x1_sq = (x1 ** 2).sum(dim=-1)
            x2_sq = (x2 ** 2).sum(dim=-1)
            denom = x1_sq + x2_sq - dot
            # Avoid division by zero (both vectors are zero)
            return dot / denom.clamp(min=1e-8)
        else:
            # Full kernel matrix
            x1_sq = (x1 ** 2).sum(dim=-1, keepdim=True)  # (... x n x 1)
            x2_sq = (x2 ** 2).sum(dim=-1, keepdim=True)  # (... x m x 1)
            dot = torch.matmul(x1, x2.transpose(-2, -1))  # (... x n x m)
            # |x1|^2 + |x2|^2 - x1.x2, broadcast: (n x 1) + (1 x m) - (n x m)
            denom = x1_sq + x2_sq.transpose(-2, -1) - dot
            return dot / denom.clamp(min=1e-8)
