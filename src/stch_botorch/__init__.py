"""
STCH-BoTorch: Smooth Tchebycheff scalarization for multi-objective Bayesian optimization.

This package provides differentiable alternatives to standard Tchebycheff scalarization
for use with BoTorch acquisition functions.
"""

__version__ = "0.1.0"

from stch_botorch.objectives import (
    SmoothChebyshevObjective,
    SmoothChebyshevSetObjective,
)
from stch_botorch.scalarization import smooth_chebyshev, smooth_chebyshev_set

from stch_botorch.acquisition import qPMHI
from stch_botorch.integration import (
    STCHCandidateGenerator,
    STCHqPMHIAcquisition,
    optimize_stch_qpmhi,
)

__all__ = [
    "__version__",
    "SmoothChebyshevObjective",
    "SmoothChebyshevSetObjective",
    "smooth_chebyshev",
    "smooth_chebyshev_set",
    "qPMHI",
    "STCHCandidateGenerator",
    "STCHqPMHIAcquisition",
    "optimize_stch_qpmhi",
]

