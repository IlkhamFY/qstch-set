"""
Generate convergence plots for m=5, m=8, m=10 from local benchmark JSON files.
Produces:
  - convergence_all.pdf/png  — 1x3 multi-panel figure (main paper)
  - scaling_summary.pdf/png  — final HV vs m scaling plot
  - convergence_m5/m8/m10.pdf/png — individual panels (appendix)
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent.parent / "benchmarks" / "results"
OUT_DIR = Path(__file__).parent

# NeurIPS-friendly style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "figure.dpi": 150,
})

METHODS = {
    "STCH-Set":     {"color": "#2166ac", "marker": "o", "label": "qSTCH-Set (ours)", "zorder": 4},
    "qNParEGO":     {"color": "#d73027", "marker": "s", "label": "qNParEGO",          "zorder": 3},
    "STCH-NParEGO": {"color": "#4dac26", "marker": "^", "label": "STCH-NParEGO",      "zorder": 2},
    "Random":       {"color": "#999999", "marker": "D", "label": "Random",            "zorder": 1},
}

def load_results(json_path):
    with open(json_path) as f:
        d = json.load(f)
    config = d["config"]
    methods = {}
    for name, mdata in d["methods"].items():
        runs = [r for r in mdata["runs"] if r.get("error") is None]
        if not runs:
            continue
        histories = [r["hv_history"] for r in runs]
        min_len = min(len(h) for h in histories)
        arr = np.array([h[:min_len] for h in histories])
        methods[name] = {
            "mean":    arr.mean(axis=0),
            "std":     arr.std(axis=0),
            "n_seeds": len(runs),
            "final_hv": float(arr[:, -1].mean()),
            "final_std": float(arr[:, -1].std()),
        }
    return config, methods

def plot_panel(ax, config, results, show_legend=False, show_ylabel=True):
    m = config["m"]
    K = config["K"]
    n_seeds = results.get("STCH-Set", next(iter(results.values())))["n_seeds"]

    for name, style in METHODS.items():
        if name not in results:
            continue
        r = results[name]
        mean, std = r["mean"], r["std"]
        x = np.arange(len(mean))
        markevery = max(1, len(x) // 8)
        ax.plot(x, mean, color=style["color"], marker=style["marker"],
                label=style["label"], markevery=markevery,
                zorder=style["zorder"])
        ax.fill_between(x, mean - std, mean + std,
                        alpha=0.12, color=style["color"], zorder=style["zorder"])

    ax.set_title(f"$m={{{m}}}$, $K={{{K}}}$ ({n_seeds} seeds)", pad=4)
    ax.set_xlabel("BO Iteration")
    if show_ylabel:
        ax.set_ylabel("Hypervolume")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlim(0, max(len(r["mean"]) for r in results.values()) - 1)
    if show_legend:
        ax.legend(loc="lower right", framealpha=0.9)

def main():
    datasets = [
        ("dtlz2_m5_results.json",  5),
        ("dtlz2_m8_results.json",  8),
        ("dtlz2_m10_results.json", 10),
    ]

    all_configs = []
    all_results = []
    for fname, m in datasets:
        path = RESULTS_DIR / fname
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        config, results = load_results(path)
        all_configs.append(config)
        all_results.append(results)
        print(f"m={m}: " + ", ".join(
            f"{k}={v['final_hv']:.3f}±{v['final_std']:.3f}({v['n_seeds']}s)"
            for k, v in results.items()
        ))

    # --- Multi-panel figure (main paper) ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=False)
    for i, (ax, config, results) in enumerate(zip(axes, all_configs, all_results)):
        plot_panel(ax, config, results,
                   show_legend=(i == 2),
                   show_ylabel=(i == 0))

    # Shared legend below
    handles, labels = axes[2].get_legend_handles_labels()
    axes[2].get_legend().remove()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.04), framealpha=0.9, fontsize=9)

    fig.suptitle("DTLZ2 Convergence: Hypervolume vs. BO Iteration", fontsize=12, y=1.01)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    for ext in ["pdf", "png"]:
        out = OUT_DIR / f"convergence_all.{ext}"
        fig.savefig(out, dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close()

    # --- Individual panels (appendix) ---
    for config, results in zip(all_configs, all_results):
        m = config["m"]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        plot_panel(ax, config, results, show_legend=True, show_ylabel=True)
        plt.tight_layout()
        for ext in ["pdf", "png"]:
            out = OUT_DIR / f"convergence_m{m}.{ext}"
            fig.savefig(out, dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
            print(f"Saved {out}")
        plt.close()

    # --- Scaling summary: final HV vs m ---
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ms = [c["m"] for c in all_configs]
    for name, style in METHODS.items():
        hvs, stds = [], []
        for results in all_results:
            if name in results:
                hvs.append(results[name]["final_hv"])
                stds.append(results[name]["final_std"])
            else:
                hvs.append(None)
                stds.append(None)
        valid = [(m, hv, std) for m, hv, std in zip(ms, hvs, stds) if hv is not None]
        if not valid:
            continue
        xs, ys, es = zip(*valid)
        ax.errorbar(xs, ys, yerr=es, color=style["color"], marker=style["marker"],
                    label=style["label"], capsize=3, linewidth=2.0, markersize=6)

    ax.set_xlabel("Number of Objectives ($m$)")
    ax.set_ylabel("Final Hypervolume")
    ax.set_title("Scaling with Number of Objectives (DTLZ2)")
    ax.set_xticks(ms)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        out = OUT_DIR / f"scaling_summary.{ext}"
        fig.savefig(out, dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close()

    print("\nAll figures generated.")

if __name__ == "__main__":
    main()
