"""Benchmarking utilities for multi-objective optimization."""

from stch_botorch.benchmarks.plotting import (
    plot_hypervolume_progression,
    plot_pareto_fronts,
    plot_runtime_comparison,
)
from stch_botorch.benchmarks.runner import run_benchmark
from stch_botorch.benchmarks.synthetic import SyntheticProblem

__all__ = [
    "run_benchmark",
    "SyntheticProblem",
    "plot_hypervolume_progression",
    "plot_runtime_comparison",
    "plot_pareto_fronts",
]
