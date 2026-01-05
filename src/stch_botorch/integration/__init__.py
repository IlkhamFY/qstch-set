"""Integration modules combining STCH with other acquisition methods."""

from stch_botorch.integration.stch_qpmhi import (
    STCHCandidateGenerator,
    STCHqPMHIAcquisition,
    optimize_stch_qpmhi,
)

__all__ = ["STCHCandidateGenerator", "STCHqPMHIAcquisition", "optimize_stch_qpmhi"]
