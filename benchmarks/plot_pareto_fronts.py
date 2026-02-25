"""
Generate Pareto front plots from existing real-world benchmark summary.json files.
Run: python benchmarks/plot_pareto_fronts.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from botorch.test_functions.multi_objective import VehicleSafety, Penicillin, CarSideImpact
from botorch.utils.multi_objective.pareto import is_non_dominated
import torch

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 10, "legend.fontsize": 8,
    "lines.linewidth": 1.5, "figure.dpi": 150,
})

RESULTS_BASE = Path(__file__).parent.parent / "results"

PROBLEMS = {
    "VehicleSafety":  {"cls": VehicleSafety,  "obj_names": ["Mass", "Ain", "Intrusion"]},
    "Penicillin":     {"cls": Penicillin,      "obj_names": ["Titer", "CO2", "Time"]},
    "CarSideImpact":  {"cls": CarSideImpact,   "obj_names": ["Mass", "Ain", "MBP", "FD"]},
}

COLORS = {
    "random":    "#888888",
    "qnparego":  "#e41a1c",
    "qehvi":     "#377eb8",
    "qnehvi":    "#4daf4a",
    "qstch_set": "#984ea3",
}
LABELS = {
    "random":    "Sobol",
    "qnparego":  "qNParEGO",
    "qehvi":     "qEHVI",
    "qnehvi":    "qNEHVI",
    "qstch_set": "qSTCH-Set (ours)",
}
METHODS = ["random", "qnparego", "qehvi", "qnehvi", "qstch_set"]

def get_true_pareto(problem_cls):
    """Sample a dense reference Pareto front from the true problem."""
    try:
        prob = problem_cls(negate=False)
        # Sample 10000 random points, take Pareto
        X = torch.rand(10000, prob.dim, dtype=torch.double)
        X = prob.bounds[0] + X * (prob.bounds[1] - prob.bounds[0])
        Y = prob(X)
        mask = is_non_dominated(-Y)  # minimization
        return Y[mask].numpy()
    except Exception:
        return None

def plot_problem(problem_name, summary_path, out_dir):
    with open(summary_path) as f:
        summary = json.load(f)

    info = PROBLEMS[problem_name]
    obj_names = info["obj_names"]
    num_obj = summary["num_objectives"]
    n_seeds = summary["n_seeds"]

    # All 2D pairs
    pairs = [(i, j) for i in range(num_obj) for j in range(i+1, num_obj)]
    ncols = min(3, len(pairs))
    nrows = (len(pairs) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    # Try to get true Pareto front
    true_pf = get_true_pareto(info["cls"])

    for ax_idx, (i, j) in enumerate(pairs):
        ax = axes_flat[ax_idx]

        # Plot true Pareto front if available
        if true_pf is not None:
            ax.scatter(true_pf[:, i], true_pf[:, j],
                      c="gold", s=8, alpha=0.4, label="True PF", zorder=0, marker="*")

        # Plot each method's Pareto points from summary
        # Summary has hv_mean but not raw points — use pareto_fronts if available
        pareto_key = "_pareto_fronts"
        if pareto_key in summary:
            for method in METHODS:
                if method not in summary[pareto_key]:
                    continue
                pts_all = []
                for seed_pts in summary[pareto_key][method]:
                    pts_all.extend(seed_pts)
                if not pts_all:
                    continue
                arr = np.array(pts_all)
                # negate back to minimization space
                ax.scatter(-arr[:, i], -arr[:, j],
                          c=COLORS[method], label=LABELS[method],
                          alpha=0.5, s=15, zorder=2)
        else:
            # No raw pareto data — annotate with final HV
            ax.text(0.5, 0.5, f"No raw Pareto data\n(re-run with updated script)",
                   transform=ax.transAxes, ha="center", va="center",
                   fontsize=9, color="gray")

        ax.set_xlabel(f"$f_{{{i+1}}}$ ({obj_names[i]})")
        ax.set_ylabel(f"$f_{{{j+1}}}$ ({obj_names[j]})")
        ax.grid(True, alpha=0.25)
        if ax_idx == 0:
            ax.legend(fontsize=7, markerscale=2, loc="upper right")

    # Hide unused axes
    for ax in axes_flat[len(pairs):]:
        ax.set_visible(False)

    fig.suptitle(f"{problem_name} — Final Pareto Fronts ({n_seeds} seeds, all methods)", fontsize=12)
    plt.tight_layout()
    out_path = out_dir / f"pareto_{problem_name.lower()}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
    return out_path

def main():
    out_dir = RESULTS_BASE / "pareto_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for problem_name in PROBLEMS:
        summary_path = RESULTS_BASE / f"real_world_{problem_name.lower()}" / "summary.json"
        if not summary_path.exists():
            print(f"MISSING: {summary_path} — skipping")
            continue
        print(f"\nPlotting {problem_name}...")
        path = plot_problem(problem_name, summary_path, out_dir)
        saved.append(path)

    print(f"\nDone. {len(saved)} plots saved to {out_dir}")

if __name__ == "__main__":
    main()
