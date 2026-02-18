# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-02-17

### Added

- `qSTCHSet` — Monte Carlo STCH-Set acquisition function for many-objective BO
- `qSTCHSetTS` — Thompson Sampling variant of qSTCHSet
- `smooth_chebyshev` — single-point Smooth Tchebycheff scalarization
- `smooth_chebyshev_set` — set-based Smooth Tchebycheff scalarization (Lin et al. ICLR 2025)
- `SmoothChebyshevObjective` — BoTorch objective wrapper for use with `qLogNParEGO` etc.
- `SmoothChebyshevSetObjective` — BoTorch objective wrapper aggregating over batch dimension
- `qPMHI` — Probability of Maximum Hypervolume Improvement acquisition function
- `optimize_stch_qpmhi` — two-stage STCH + qPMHI batch selection
- DTLZ and ZDT benchmark scripts
- Comprehensive test suite
