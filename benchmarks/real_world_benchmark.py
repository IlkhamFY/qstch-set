"""
Real-world multi-objective benchmark: CarSideImpact, VehicleSafety, Penicillin
Compares: Sobol, qNParEGO, qEHVI, qNEHVI, qSTCH-Set
Based on: https://archive.botorch.org/tutorials/multi_objective_bo
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.test_functions.multi_objective import (
    CarSideImpact,
    Penicillin,
    VehicleSafety,
)
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples, sample_simplex
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from stch_botorch.acquisition.stch_set_bo import qSTCHSet

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

PROBLEM_MAP = {
    "VehicleSafety": VehicleSafety,
    "Penicillin": Penicillin,
    "CarSideImpact": CarSideImpact,
}

N_BATCH = 20
MC_SAMPLES = 128  # overridden by --mc-samples arg


def get_problem(name: str, device, dtype):
    cls = PROBLEM_MAP[name]
    return cls(negate=True).to(device=device, dtype=dtype)


def generate_initial_data(problem, n, device, dtype):
    train_x = draw_sobol_samples(
        bounds=problem.bounds, n=n, q=1
    ).squeeze(1).to(device=device, dtype=dtype)
    train_obj_true = problem(train_x)
    train_obj = train_obj_true + torch.randn_like(train_obj_true) * 1e-6
    return train_x, train_obj, train_obj_true


def initialize_model(train_x, train_obj, bounds):
    train_x_norm = normalize(train_x, bounds)
    models = [
        SingleTaskGP(
            train_x_norm,
            train_obj[..., i: i + 1],
            outcome_transform=Standardize(m=1),
        )
        for i in range(train_obj.shape[-1])
    ]
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model


def unit_bounds(d, device, dtype):
    return torch.stack([torch.zeros(d, device=device, dtype=dtype),
                        torch.ones(d, device=device, dtype=dtype)])


def optimize_qnparego(problem, model, train_x, train_obj, sampler, bounds, batch_size, device, dtype):
    from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
    train_x_norm = normalize(train_x, bounds)
    with torch.no_grad():
        pred = model.posterior(train_x_norm).mean
    candidates_list = []
    for _ in range(batch_size):
        # Fix: ensure weights on same device/dtype as Y
        weights = sample_simplex(train_obj.shape[-1]).squeeze().to(device=device, dtype=dtype)
        objective = get_chebyshev_scalarization(weights=weights, Y=pred)
        acq = qNoisyExpectedImprovement(
            model=model,
            X_baseline=train_x_norm,
            sampler=sampler,
            objective=objective,
            prune_baseline=True,
        )
        cand, _ = optimize_acqf(
            acq_function=acq,
            bounds=unit_bounds(bounds.shape[1], device, dtype),
            q=1,
            num_restarts=10,
            raw_samples=512,
            options={"batch_limit": 5, "maxiter": 200},
        )
        candidates_list.append(cand)
    candidates = torch.cat(candidates_list, dim=-2)
    new_x = unnormalize(candidates.detach(), bounds)
    new_obj_true = problem(new_x)
    new_obj = new_obj_true + torch.randn_like(new_obj_true) * 1e-6
    return new_x, new_obj, new_obj_true


def optimize_qehvi(problem, model, train_x, train_obj_true, sampler, bounds, batch_size, ref_point, device, dtype):
    train_x_norm = normalize(train_x, bounds)
    partitioning = DominatedPartitioning(ref_point=ref_point, Y=train_obj_true)
    acq = qLogExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        sampler=sampler,
    )
    candidates, _ = optimize_acqf(
        acq_function=acq,
        bounds=unit_bounds(bounds.shape[1], device, dtype),
        q=batch_size,
        num_restarts=10,
        raw_samples=512,
        options={"batch_limit": 5, "maxiter": 200},
    )
    new_x = unnormalize(candidates.detach(), bounds)
    new_obj_true = problem(new_x)
    new_obj = new_obj_true + torch.randn_like(new_obj_true) * 1e-6
    return new_x, new_obj, new_obj_true


def optimize_qnehvi(problem, model, train_x, train_obj, sampler, bounds, batch_size, ref_point, device, dtype):
    train_x_norm = normalize(train_x, bounds)
    acq = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        X_baseline=train_x_norm,
        sampler=sampler,
        prune_baseline=True,
    )
    candidates, _ = optimize_acqf(
        acq_function=acq,
        bounds=unit_bounds(bounds.shape[1], device, dtype),
        q=batch_size,
        num_restarts=10,
        raw_samples=512,
        options={"batch_limit": 5, "maxiter": 200},
    )
    new_x = unnormalize(candidates.detach(), bounds)
    new_obj_true = problem(new_x)
    new_obj = new_obj_true + torch.randn_like(new_obj_true) * 1e-6
    return new_x, new_obj, new_obj_true


def optimize_qstch_set(problem, model, train_x, train_obj, sampler, bounds, batch_size, ref_point, device, dtype):
    acq = qSTCHSet(
        model=model,
        ref_point=ref_point,
        mu=0.1,
        sampler=sampler,
    )
    candidates, _ = optimize_acqf(
        acq_function=acq,
        bounds=unit_bounds(bounds.shape[1], device, dtype),
        q=batch_size,
        num_restarts=10,
        raw_samples=512,
        options={"batch_limit": 5, "maxiter": 200},
    )
    new_x = unnormalize(candidates.detach(), bounds)
    new_obj_true = problem(new_x)
    new_obj = new_obj_true + torch.randn_like(new_obj_true) * 1e-6
    return new_x, new_obj, new_obj_true


def run_benchmark(problem_name: str, n_seeds: int, output_dir: Path, device, dtype, mc_samples: int = 128):
    problem = get_problem(problem_name, device, dtype)
    num_obj = problem.num_objectives
    batch_size = num_obj  # K = m rule
    ref_point = problem.ref_point.to(device=device, dtype=dtype)
    bounds = problem.bounds.to(device=device, dtype=dtype)

    print(f"\n{'='*60}")
    print(f"Problem: {problem_name} | m={num_obj} | d={problem.dim} | K={batch_size} | seeds={n_seeds} | MC={mc_samples}")
    print(f"{'='*60}")

    methods = ["random", "qnparego", "qehvi", "qnehvi", "qstch_set"]
    all_results = {m: [] for m in methods}
    pareto_fronts = {m: [] for m in methods}

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        print(f"\n--- Seed {seed+1}/{n_seeds} ---")

        n_init = 2 * (problem.dim + 1)
        train_x, train_obj, train_obj_true = generate_initial_data(problem, n_init, device, dtype)

        data = {
            method: {
                "train_x": train_x.clone(),
                "train_obj": train_obj.clone(),
                "train_obj_true": train_obj_true.clone(),
            }
            for method in methods
        }

        bd = DominatedPartitioning(ref_point=ref_point, Y=train_obj_true)
        init_hv = bd.compute_hypervolume().item()
        hvs = {m: [init_hv] for m in methods}

        for iteration in range(1, N_BATCH + 1):
            t0 = time.monotonic()

            for method in ["qnparego", "qehvi", "qnehvi", "qstch_set"]:
                d = data[method]
                mll, model = initialize_model(d["train_x"], d["train_obj"], bounds)
                fit_gpytorch_mll(mll)
                sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))

                if method == "qnparego":
                    new_x, new_obj, new_obj_true = optimize_qnparego(
                        problem, model, d["train_x"], d["train_obj"],
                        sampler, bounds, batch_size, device, dtype
                    )
                elif method == "qehvi":
                    new_x, new_obj, new_obj_true = optimize_qehvi(
                        problem, model, d["train_x"], d["train_obj_true"],
                        sampler, bounds, batch_size, ref_point, device, dtype
                    )
                elif method == "qnehvi":
                    new_x, new_obj, new_obj_true = optimize_qnehvi(
                        problem, model, d["train_x"], d["train_obj"],
                        sampler, bounds, batch_size, ref_point, device, dtype
                    )
                elif method == "qstch_set":
                    new_x, new_obj, new_obj_true = optimize_qstch_set(
                        problem, model, d["train_x"], d["train_obj"],
                        sampler, bounds, batch_size, ref_point, device, dtype
                    )

                d["train_x"] = torch.cat([d["train_x"], new_x])
                d["train_obj"] = torch.cat([d["train_obj"], new_obj])
                d["train_obj_true"] = torch.cat([d["train_obj_true"], new_obj_true])

            # random baseline
            new_x_rand, new_obj_rand, new_obj_true_rand = generate_initial_data(
                problem, batch_size, device, dtype
            )
            data["random"]["train_x"] = torch.cat([data["random"]["train_x"], new_x_rand])
            data["random"]["train_obj"] = torch.cat([data["random"]["train_obj"], new_obj_rand])
            data["random"]["train_obj_true"] = torch.cat([data["random"]["train_obj_true"], new_obj_true_rand])

            for method in methods:
                bd = DominatedPartitioning(
                    ref_point=ref_point, Y=data[method]["train_obj_true"]
                )
                hvs[method].append(bd.compute_hypervolume().item())

            t1 = time.monotonic()
            hv_str = " | ".join(
                f"{m}={hvs[m][-1]:.3f}" for m in ["random", "qnparego", "qehvi", "qnehvi", "qstch_set"]
            )
            print(f"  Iter {iteration:>2}/{N_BATCH}: {hv_str} ({t1-t0:.1f}s)")

        for method in methods:
            all_results[method].append(hvs[method])

        # Save final Pareto fronts for plotting
        for method in methods:
            Y = data[method]["train_obj_true"]
            mask = is_non_dominated(Y)
            pareto_fronts[method].append(Y[mask].cpu().tolist())

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np

    summary = {
        "problem": problem_name,
        "n_seeds": n_seeds,
        "n_batch": N_BATCH,
        "batch_size": batch_size,
        "num_objectives": num_obj,
        "dim": problem.dim,
    }
    for method in methods:
        runs = all_results[method]
        arr = np.array(runs)
        summary[method] = {
            "hv_mean": arr.mean(axis=0).tolist(),
            "hv_std": arr.std(axis=0).tolist(),
            "final_hv_mean": float(arr[:, -1].mean()),
            "final_hv_std": float(arr[:, -1].std()),
            "hv_all": arr.tolist(),
        }

    print(f"\n{'='*60}")
    print(f"Final HV ({problem_name}):")
    labels = {"random": "Sobol", "qnparego": "qNParEGO", "qehvi": "qEHVI",
              "qnehvi": "qNEHVI", "qstch_set": "qSTCH-Set"}
    for method in methods:
        r = summary[method]
        print(f"  {labels[method]:15s}: {r['final_hv_mean']:.4f} ± {r['final_hv_std']:.4f}")
    print(f"{'='*60}")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {output_dir / 'summary.json'}")

    # Plot convergence
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {"random": "#888", "qnparego": "#e41a1c", "qehvi": "#377eb8",
                  "qnehvi": "#4daf4a", "qstch_set": "#984ea3"}
        fig, ax = plt.subplots(figsize=(7, 4.5))
        iters = list(range(N_BATCH + 1))
        for method in methods:
            mean = np.array(summary[method]["hv_mean"])
            std = np.array(summary[method]["hv_std"])
            ax.plot(iters, mean, label=labels[method], color=colors[method], linewidth=2)
            ax.fill_between(iters, mean - std, mean + std, alpha=0.15, color=colors[method])
        ax.set_xlabel("BO Iteration")
        ax.set_ylabel("Hypervolume")
        ax.set_title(f"{problem_name} (m={num_obj}, K={batch_size}, {n_seeds} seeds, MC={mc_samples})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "hv_plot.png", dpi=150, bbox_inches="tight")
        print(f"Convergence plot saved.")
    except Exception as e:
        print(f"Convergence plot failed: {e}")

    # Plot Pareto fronts (2D projections for m=3; obj 0 vs 1, 0 vs 2, 1 vs 2 for m=4)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pareto_data = pareto_fronts
        if pareto_data and num_obj <= 4:
            pairs = [(0, 1)] if num_obj == 2 else [(0, 1), (0, 2), (1, 2)]
            fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4.5))
            if len(pairs) == 1:
                axes = [axes]
            for ax, (i, j) in zip(axes, pairs):
                for method in methods:
                    pts_all = pareto_data.get(method, [])
                    # Aggregate all seeds
                    all_pts = []
                    for seed_pts in pts_all:
                        all_pts.extend(seed_pts)
                    if not all_pts:
                        continue
                    arr = np.array(all_pts)
                    # negate back to original minimization space for plotting
                    ax.scatter(-arr[:, i], -arr[:, j],
                               label=labels[method], color=colors[method],
                               alpha=0.5, s=15)
                ax.set_xlabel(f"$f_{{{i+1}}}$")
                ax.set_ylabel(f"$f_{{{j+1}}}$")
                ax.set_title(f"{problem_name}: $f_{{{i+1}}}$ vs $f_{{{j+1}}}$")
                ax.legend(fontsize=7, markerscale=2)
                ax.grid(True, alpha=0.3)
            plt.suptitle(f"{problem_name} — Final Pareto Fronts (all seeds)", fontsize=12)
            plt.tight_layout()
            plt.savefig(output_dir / "pareto_fronts.png", dpi=150, bbox_inches="tight")
            print(f"Pareto front plot saved.")
    except Exception as e:
        print(f"Pareto front plot failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True, choices=list(PROBLEM_MAP.keys()))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--mc-samples", type=int, default=128,
                        help="Number of MC samples for acquisition function (default 128, use 512 for higher fidelity)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dtype = torch.double
    device = torch.device(args.device)

    if args.output_dir is None:
        out = Path(__file__).parent.parent / "results" / f"real_world_{args.problem.lower()}_mc{args.mc_samples}"
    else:
        out = Path(args.output_dir)

    run_benchmark(args.problem, args.seeds, out, device, dtype, mc_samples=args.mc_samples)
