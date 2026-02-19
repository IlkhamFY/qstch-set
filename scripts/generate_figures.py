#!/usr/bin/env python3
"""
Generate NeurIPS-style convergence figures for qSTCH-Set paper.

Reads benchmark result JSONs and produces:
  - paper/figures/convergence_m5.pdf  (+ .png)
  - paper/figures/convergence_m8.pdf  (+ .png)
  - paper/figures/convergence_m10.pdf (+ .png) — bar chart (no per-iter data)

Style: Clean axes, no box, colorblind-friendly (Tol bright palette).
"""

import json
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
BENCH_RESULTS = ROOT / "benchmarks" / "results"
NIBI_RESULTS = ROOT / "results"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── NeurIPS style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "text.usetex": False,  # safe fallback
})

# Tol bright palette — colorblind-friendly
COLORS = {
    "STCH-Set":      "#4477AA",  # blue
    "qNParEGO":      "#EE6677",  # red/coral
    "STCH-NParEGO":  "#228833",  # green
    "Random":        "#BBBBBB",  # gray
}
MARKERS = {
    "STCH-Set":      "o",
    "qNParEGO":      "s",
    "STCH-NParEGO":  "^",
    "Random":        "D",
}
# Plot order (our method first)
METHOD_ORDER = ["STCH-Set", "qNParEGO", "STCH-NParEGO", "Random"]

# Display names for the legend
DISPLAY_NAMES = {
    "STCH-Set":      "qSTCH-Set (ours)",
    "qNParEGO":      "qNParEGO",
    "STCH-NParEGO":  "STCH-NParEGO",
    "Random":        "Random",
}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_hv_curves(data):
    """
    From a benchmark JSON, extract per-method HV curves.
    Returns dict: method_name -> { 'mean': array, 'std': array, 'iters': array }
    """
    methods = data["methods"]
    result = {}
    for name, mdata in methods.items():
        if "runs" in mdata:
            # Raw runs with hv_history
            histories = [np.array(r["hv_history"]) for r in mdata["runs"]
                        if r.get("error") is None and r.get("hv_history")]
            if not histories:
                continue
            # Truncate to shortest run
            min_len = min(len(h) for h in histories)
            arr = np.array([h[:min_len] for h in histories])
            result[name] = {
                "mean": arr.mean(axis=0),
                "std": arr.std(axis=0),
                "iters": np.arange(min_len),
                "n_seeds": len(histories),
            }
        elif "hv_mean" in mdata:
            # Pre-aggregated (like m=3 results)
            result[name] = {
                "mean": np.array(mdata["hv_mean"]),
                "std": np.array(mdata["hv_std"]),
                "iters": np.arange(len(mdata["hv_mean"])),
                "n_seeds": mdata.get("seeds", "?"),
            }
    return result


def plot_convergence(curves, m, K, n_init, fig_path_stem, title_extra=""):
    """Plot convergence curves with error bands."""
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.8))

    for name in METHOD_ORDER:
        if name not in curves:
            continue
        c = curves[name]
        mean, std, iters = c["mean"], c["std"], c["iters"]
        color = COLORS.get(name, "#999999")
        marker = MARKERS.get(name, ".")
        label = DISPLAY_NAMES.get(name, name)

        # Add seed count to label
        n_seeds = c.get("n_seeds", "?")
        label += f" ({n_seeds} seeds)"

        # Plot mean line
        ax.plot(iters, mean, color=color, marker=marker,
                markevery=max(1, len(iters) // 8), label=label, zorder=3)
        # Error band (mean ± 1 std)
        ax.fill_between(iters, mean - std, mean + std,
                        color=color, alpha=0.15, zorder=1)

    ax.set_xlabel("BO Iteration")
    ax.set_ylabel("Hypervolume")
    title = f"DTLZ2, $m={m}$"
    if K:
        title += f", $K={K}$"
    if title_extra:
        title += f" — {title_extra}"
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25, linewidth=0.5)

    # Save
    for ext in ["pdf", "png"]:
        fig.savefig(f"{fig_path_stem}.{ext}")
        print(f"  Saved {fig_path_stem}.{ext}")
    plt.close(fig)


def plot_m10_bar(nibi_data, fig_path_stem):
    """
    m=10: only final HV available, so create a bar chart.
    """
    methods_data = nibi_data["methods"]
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5))

    names = []
    means = []
    stds = []
    colors = []

    method_map = {
        "stch_set": "STCH-Set",
        "qnparego": "qNParEGO",
    }
    for key in ["stch_set", "qnparego"]:
        if key not in methods_data:
            continue
        md = methods_data[key]
        display = method_map.get(key, key)
        names.append(DISPLAY_NAMES.get(display, display))
        means.append(md.get("final_hv_mean", 0))
        stds.append(md.get("final_hv_std", 0))
        colors.append(COLORS.get(display, "#999999"))

    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  edgecolor="white", linewidth=0.8, width=0.5, zorder=3)

    # Add value labels on bars
    for bar, m_val, s_val in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s_val + 0.3,
                f"{m_val:.1f}±{s_val:.1f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Final Hypervolume")
    ax.set_title("DTLZ2, $m=10$, $K=10$ (3 seeds)")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)

    # Set y-axis to start from reasonable value
    ymin = min(means) - max(stds) - 2
    ymax = max(means) + max(stds) + 2
    ax.set_ylim(ymin, ymax)

    for ext in ["pdf", "png"]:
        fig.savefig(f"{fig_path_stem}.{ext}")
        print(f"  Saved {fig_path_stem}.{ext}")
    plt.close(fig)


def plot_k_ablation(ablation_data, bench_m5_curves, fig_path_stem):
    """
    K-ablation at m=5: bar chart of final HV for K=3,5,10
    plus qNParEGO baseline from full benchmark.
    """
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))

    methods = ablation_data["methods"]
    k_labels = []
    k_means = []
    k_stds = []
    k_colors = []

    # Parse K from method names and sort
    k_items = []
    for name, md in methods.items():
        # Extract K value from name like "stch_set_K3"
        k_val = int(name.split("K")[1])
        k_items.append((k_val, md))
    k_items.sort(key=lambda x: x[0])

    blue_shades = ["#AACCEE", "#4477AA", "#223366"]
    for i, (k_val, md) in enumerate(k_items):
        k_labels.append(f"$K={k_val}$")
        k_means.append(md["final_hv_mean"])
        k_stds.append(md["final_hv_std"])
        k_colors.append(blue_shades[i])

    # Add qNParEGO baseline
    if "qNParEGO" in bench_m5_curves:
        qn = bench_m5_curves["qNParEGO"]
        k_labels.append("qNParEGO")
        k_means.append(qn["mean"][-1])
        k_stds.append(qn["std"][-1])
        k_colors.append(COLORS["qNParEGO"])

    x = np.arange(len(k_labels))
    bars = ax.bar(x, k_means, yerr=k_stds, capsize=5, color=k_colors,
                  edgecolor="white", linewidth=0.8, width=0.55, zorder=3)

    for bar, m_val, s_val in zip(bars, k_means, k_stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s_val + 0.03,
                f"{m_val:.2f}",
                ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(k_labels)
    ax.set_ylabel("Final Hypervolume")
    ax.set_title("$K$-Ablation on DTLZ2, $m=5$ (3 seeds)")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)

    ymin = min(k_means) - max(k_stds) - 0.5
    ymax = max(k_means) + max(k_stds) + 0.5
    ax.set_ylim(ymin, ymax)

    for ext in ["pdf", "png"]:
        fig.savefig(f"{fig_path_stem}.{ext}")
        print(f"  Saved {fig_path_stem}.{ext}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("qSTCH-Set Paper — Figure Generation")
    print("=" * 60)

    # ── 1. m=5 convergence ─────────────────────────────────────────────
    m5_path = BENCH_RESULTS / "dtlz2_m5_results.json"
    if m5_path.exists():
        print(f"\n[1] Loading m=5 data: {m5_path}")
        m5_data = load_json(m5_path)
        m5_curves = extract_hv_curves(m5_data)
        print(f"    Methods: {list(m5_curves.keys())}")
        for name, c in m5_curves.items():
            print(f"    {name}: {c['n_seeds']} seeds, {len(c['mean'])} iters, "
                  f"final HV = {c['mean'][-1]:.3f} ± {c['std'][-1]:.3f}")
        plot_convergence(
            m5_curves, m=5, K=5,
            n_init=m5_data["config"]["n_init"],
            fig_path_stem=str(FIG_DIR / "convergence_m5"),
        )
    else:
        print(f"\n[1] MISSING: {m5_path}")

    # ── 2. m=8 convergence ─────────────────────────────────────────────
    m8_path = BENCH_RESULTS / "dtlz2_m8_results.json"
    if m8_path.exists():
        print(f"\n[2] Loading m=8 data: {m8_path}")
        m8_data = load_json(m8_path)
        m8_curves = extract_hv_curves(m8_data)
        print(f"    Methods: {list(m8_curves.keys())}")
        for name, c in m8_curves.items():
            print(f"    {name}: {c['n_seeds']} seeds, {len(c['mean'])} iters, "
                  f"final HV = {c['mean'][-1]:.3f} ± {c['std'][-1]:.3f}")
        plot_convergence(
            m8_curves, m=8, K=8,
            n_init=m8_data["config"]["n_init"],
            fig_path_stem=str(FIG_DIR / "convergence_m8"),
        )
    else:
        print(f"\n[2] MISSING: {m8_path}")

    # ── 3. m=10 bar chart (no per-iteration data) ──────────────────────
    m10_path = NIBI_RESULTS / "nibi_m10_K10_results.json"
    if m10_path.exists():
        print(f"\n[3] Loading m=10 data: {m10_path}")
        m10_data = load_json(m10_path)
        for name, md in m10_data["methods"].items():
            print(f"    {name}: final HV = {md.get('final_hv_mean', '?')} "
                  f"± {md.get('final_hv_std', '?')}")
        plot_m10_bar(
            m10_data,
            fig_path_stem=str(FIG_DIR / "convergence_m10"),
        )
    else:
        print(f"\n[3] MISSING: {m10_path}")

    # ── 4. K-ablation at m=5 ──────────────────────────────────────────
    abl_path = NIBI_RESULTS / "nibi_ablation_K_m5_results.json"
    if abl_path.exists() and m5_path.exists():
        print(f"\n[4] Loading K-ablation data: {abl_path}")
        abl_data = load_json(abl_path)
        plot_k_ablation(
            abl_data,
            m5_curves if m5_path.exists() else {},
            fig_path_stem=str(FIG_DIR / "k_ablation_m5"),
        )
    else:
        print(f"\n[4] MISSING: {abl_path}")

    print("\n" + "=" * 60)
    print("Done! Figures in:", FIG_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
