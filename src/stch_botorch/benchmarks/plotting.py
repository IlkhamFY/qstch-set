"""Plotting utilities for benchmark results."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_hypervolume_progression(
    results: Dict,
    output_path: Optional[Path] = None,
    figsize: tuple = (8, 6),
):
    """Plot hypervolume progression over iterations.

    Args:
        results: Results dictionary from run_benchmark.
        output_path: Path to save figure. If None, displays.
        figsize: Figure size.
    """
    plt.figure(figsize=figsize)

    for method, method_results in results.items():
        # Collect hypervolumes across seeds
        all_hvs = []
        for seed_result in method_results:
            all_hvs.append(seed_result["hypervolumes"])

        # Compute mean and std
        all_hvs = np.array(all_hvs)
        mean_hv = all_hvs.mean(axis=0)
        std_hv = all_hvs.std(axis=0)

        iterations = np.arange(1, len(mean_hv) + 1)
        plt.plot(iterations, mean_hv, label=method, linewidth=2)
        plt.fill_between(iterations, mean_hv - std_hv, mean_hv + std_hv, alpha=0.2)

    plt.xlabel("Iteration")
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume Progression")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_runtime_comparison(
    results: Dict,
    output_path: Optional[Path] = None,
    figsize: tuple = (8, 6),
):
    """Plot runtime comparison across methods.

    Args:
        results: Results dictionary from run_benchmark.
        output_path: Path to save figure. If None, displays.
        figsize: Figure size.
    """
    plt.figure(figsize=figsize)

    methods = []
    mean_times = []
    std_times = []

    for method, method_results in results.items():
        # Collect total times across seeds
        total_times = []
        for seed_result in method_results:
            total_times.append(np.sum(seed_result["times"]))

        methods.append(method)
        mean_times.append(np.mean(total_times))
        std_times.append(np.std(total_times))

    x_pos = np.arange(len(methods))
    plt.bar(x_pos, mean_times, yerr=std_times, capsize=5, alpha=0.7)
    plt.xticks(x_pos, methods, rotation=45, ha="right")
    plt.ylabel("Total Runtime (seconds)")
    plt.title("Runtime Comparison")
    plt.grid(True, alpha=0.3, axis="y")

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_pareto_fronts(
    results: Dict,
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 8),
):
    """Plot final Pareto fronts for 2D objectives.

    Args:
        results: Results dictionary from run_benchmark.
        output_path: Path to save figure. If None, displays.
        figsize: Figure size.
    """
    n_methods = len(results)
    n_cols = 2
    n_rows = (n_methods + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_methods == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (method, method_results) in enumerate(results.items()):
        ax = axes[idx]

        # Plot Pareto fronts from all seeds
        for seed_result in method_results:
            pareto = seed_result["final_pareto"]
            if pareto.shape[0] > 0 and pareto.shape[1] == 2:
                # Sort by first objective for plotting
                sort_idx = np.argsort(pareto[:, 0])
                pareto_sorted = pareto[sort_idx]
                ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1], "o-", alpha=0.5, markersize=3)

        ax.set_xlabel("Objective 1")
        ax.set_ylabel("Objective 2")
        ax.set_title(method)
        ax.grid(True, alpha=0.3)

    # Hide extra subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()
