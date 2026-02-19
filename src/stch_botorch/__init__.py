"""
STCH-BoTorch: Smooth Tchebycheff Set scalarization for many-objective Bayesian optimization.

Provides qSTCH-Set, a Monte Carlo acquisition function that jointly optimizes batches of
candidates for collective Pareto coverage using Smooth Tchebycheff Set scalarization.
Scales to 10+ objectives in O(Km) time with no hypervolume computation.

See: https://github.com/IlkhamFY/stch-botorch
"""

__version__ = "0.1.0"

from stch_botorch.objectives import (
    SmoothChebyshevObjective,
    SmoothChebyshevSetObjective,
)
from stch_botorch.scalarization import smooth_chebyshev, smooth_chebyshev_set

from stch_botorch.acquisition import qPMHI, qSTCHSet, qSTCHSetTS
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
    "qSTCHSet",
    "qSTCHSetTS",
    "STCHCandidateGenerator",
    "STCHqPMHIAcquisition",
    "optimize_stch_qpmhi",
]

