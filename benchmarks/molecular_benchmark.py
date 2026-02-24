"""
Pool-based multi-objective molecular optimization benchmark.

Extends Kristiadi et al. (ICML 2024) single-objective setting to
multi-objective: discrete BO over a fixed molecular pool, using Morgan
fingerprints as features for a multi-output GP surrogate. Evaluates
hypervolume improvement over BO iterations.

Datasets:
  - multi_redox: BTZ derivatives, 1407 molecules, 4 objectives
      * Ered (reduction potential)     -> minimize (negate for max convention)
      * HOMO (HOMO energy level)       -> maximize (already max convention)
      * Gsol (solvation free energy)   -> minimize (negate for max convention)
      * Absorption Wavelength (nm)     -> maximize (already max convention)

Methods compared:
  - Random: uniform random selection from pool
  - qNParEGO: scalarized Thompson sampling with Chebyshev scalarization
  - qSTCH-Set: our joint set-based STCH scalarization (K=m)

Key difference from continuous BO: acquisition is maximized over a discrete
candidate pool (not via L-BFGS-B). Each iteration picks the top-K molecules
from the remaining pool by acquisition value.

Note: Kristiadi et al. only optimize Ered (single-objective). We extend to
all 4 DFT properties simultaneously, creating a many-objective molecular
optimization benchmark that tests STCH-Set in its intended regime (m >= 4).
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from stch_botorch.acquisition.stch_set_bo import qSTCHSet
from stch_botorch.kernels.tanimoto import TanimotoKernel

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_multi_redox(data_path: Path, device, dtype):
    """Load multi-redox BTZ dataset with 4 objectives.

    Objectives (all converted to maximization for BoTorch):
      - Ered: reduction potential (minimize -> negate)
      - HOMO: HOMO energy level (maximize -> keep as-is)
      - Gsol: solvation free energy (minimize -> negate)
      - Absorption Wavelength: nm (maximize -> keep as-is)

    Features: Morgan fingerprints (radius=2, 1024 bits).

    Returns:
        X_pool: (N, 1024) fingerprint tensor
        Y_pool: (N, 4) objective tensor (all maximization convention)
        smiles: list of SMILES strings
        obj_names: list of objective names
    """
    import csv

    rows = []
    with open(data_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    smiles_list = [r["SMILES"] for r in rows]
    ered = np.array([float(r["Ered"]) for r in rows])
    homo = np.array([float(r["HOMO"]) for r in rows])
    gsol = np.array([float(r["Gsol"]) for r in rows])
    absorp = np.array([float(r["Absorption Wavelength"]) for r in rows])

    # Compute Morgan fingerprints
    fps = _compute_morgan_fps(smiles_list, radius=2, n_bits=1024)

    X_pool = torch.tensor(fps, device=device, dtype=dtype)
    # Convert to maximization convention (BoTorch standard)
    # Ered: minimize -> negate
    # HOMO: maximize -> keep (higher HOMO = better electron donor)
    # Gsol: minimize -> negate (more negative = better solvation)
    # Absorption: maximize -> keep (red-shifted = desirable)
    Y_pool = torch.tensor(
        np.stack([-ered, homo, -gsol, absorp], axis=1),
        device=device, dtype=dtype,
    )
    obj_names = ["neg_Ered", "HOMO", "neg_Gsol", "Absorption"]
    return X_pool, Y_pool, smiles_list, obj_names


def _compute_morgan_fps(smiles_list, radius=2, n_bits=1024):
    """Compute Morgan fingerprints. Falls back to random if rdkit unavailable."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator

        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                fps.append(np.zeros(n_bits))
            else:
                fp = gen.GetFingerprintAsNumPy(mol)
                fps.append(fp)
        return np.array(fps, dtype=np.float64)
    except ImportError:
        print("WARNING: rdkit not available, using random features (results not meaningful)")
        rng = np.random.RandomState(42)
        return rng.rand(len(smiles_list), n_bits)


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_model(train_x, train_y, kernel="rbf"):
    """Fit independent single-task GP per objective (ModelListGP).

    Args:
        kernel: "rbf" (default BoTorch) or "tanimoto" (Tanimoto similarity)
    """
    import gpytorch

    models = []
    for i in range(train_y.shape[-1]):
        if kernel == "tanimoto":
            covar = gpytorch.kernels.ScaleKernel(TanimotoKernel())
            gp = SingleTaskGP(
                train_x,
                train_y[..., i:i+1],
                covar_module=covar,
                outcome_transform=Standardize(m=1),
            )
        else:
            gp = SingleTaskGP(
                train_x,
                train_y[..., i:i+1],
                outcome_transform=Standardize(m=1),
            )
        models.append(gp)
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


# ---------------------------------------------------------------------------
# Acquisition: discrete pool selection
# ---------------------------------------------------------------------------

def select_random(pool_indices, batch_size, rng):
    """Random selection from pool."""
    chosen = rng.choice(pool_indices, size=min(batch_size, len(pool_indices)), replace=False)
    return chosen.tolist()


def select_qnparego(model, X_pool_remaining, Y_train, batch_size, mc_samples, device, dtype):
    """qNParEGO-style discrete pool selection.

    For each of K batch slots, draw random Chebyshev weights, scalarize
    the multi-output posterior, and pick the pool molecule with highest
    scalarized posterior mean (Thompson-sampling-inspired greedy).

    This matches Kristiadi et al.'s use of scalarized Thompson sampling
    (Paria et al. 2020) adapted to discrete pools.
    """
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
    num_obj = Y_train.shape[-1]
    n_remaining = X_pool_remaining.shape[0]

    selected_indices = []

    for k in range(batch_size):
        weights = sample_simplex(num_obj).squeeze().to(device=device, dtype=dtype)

        # Scalarize using Chebyshev with training data as reference
        scalarization = get_chebyshev_scalarization(weights=weights, Y=Y_train)
        objective = GenericMCObjective(scalarization)

        # Build qLogNEI with scalarized objective
        acq = qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=X_pool_remaining[:1],  # need at least 1 baseline point
            sampler=sampler,
            objective=objective,
            prune_baseline=True,
        )

        # Evaluate on all remaining pool candidates (q=1 each)
        acq_vals = torch.full((n_remaining,), float("-inf"), device=device, dtype=dtype)
        batch_eval_size = 256
        with torch.no_grad():
            for start in range(0, n_remaining, batch_eval_size):
                end = min(start + batch_eval_size, n_remaining)
                X_batch = X_pool_remaining[start:end].unsqueeze(1)  # (B, 1, d)
                vals = acq(X_batch)
                acq_vals[start:end] = vals

        # Mask already selected
        for idx in selected_indices:
            acq_vals[idx] = float("-inf")

        best_idx = acq_vals.argmax().item()
        selected_indices.append(best_idx)

    return selected_indices


def select_qstch_set(model, X_pool_remaining, Y_train, batch_size, mc_samples,
                     ref_point, device, dtype):
    """qSTCH-Set: evaluate joint set quality for all K-subsets.

    For tractability with large pools, use greedy sequential construction:
    1. Score all candidates for the first slot
    2. Fix best, score remaining for second slot, etc.

    This is a greedy approximation to joint set optimization.
    """
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))

    # Compute normalization from training data
    Y_min = Y_train.min(dim=0).values
    Y_max = Y_train.max(dim=0).values
    Y_range = Y_max - Y_min
    # Clamp range to avoid division by zero
    Y_range = torch.clamp(Y_range, min=1e-6)

    selected_indices = []
    selected_X = []

    for k in range(batch_size):
        best_val = float("-inf")
        best_idx = -1

        # Evaluate each remaining candidate
        batch_eval_size = 128
        n_remaining = X_pool_remaining.shape[0]

        all_vals = []
        for start in range(0, n_remaining, batch_eval_size):
            end = min(start + batch_eval_size, n_remaining)
            vals_batch = []
            for i in range(start, end):
                if i in selected_indices:
                    vals_batch.append(float("-inf"))
                    continue

                # Build candidate set: previously selected + this candidate
                if selected_X:
                    candidate_set = torch.stack(selected_X + [X_pool_remaining[i]], dim=0)
                else:
                    candidate_set = X_pool_remaining[i:i+1]

                # Evaluate qSTCH-Set acquisition
                acq = qSTCHSet(
                    model=model,
                    ref_point=ref_point,
                    mu=0.1,
                    sampler=sampler,
                    Y_range=Y_range,
                    Y_min=Y_min,
                )
                with torch.no_grad():
                    val = acq(candidate_set.unsqueeze(0)).item()
                vals_batch.append(val)
            all_vals.extend(vals_batch)

        # Find best
        for i, v in enumerate(all_vals):
            if v > best_val and i not in selected_indices:
                best_val = v
                best_idx = i

        selected_indices.append(best_idx)
        selected_X.append(X_pool_remaining[best_idx])

    return selected_indices


def select_qstch_set_fast(model, X_pool_remaining, Y_train, batch_size, mc_samples,
                          ref_point, device, dtype):
    """Fast qSTCH-Set: greedy sequential set construction.

    For each slot k=1..K:
      1. Build qSTCHSet acquisition with current set as context
      2. Evaluate each remaining candidate as q=1 addition
      3. Pick the best and add to set

    For K=1 this is exact; for K>1 it's a greedy approximation to joint
    set optimization. The key advantage: STCH-Set's scalarization naturally
    scores set diversity, so greedy tends to select complementary candidates.
    """
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))

    Y_min = Y_train.min(dim=0).values
    Y_max = Y_train.max(dim=0).values
    Y_range = torch.clamp(Y_max - Y_min, min=1e-6)

    selected_indices = []
    selected_X = []
    n_remaining = X_pool_remaining.shape[0]
    selected_set = set()

    for k in range(batch_size):
        # Build acquisition function once per slot
        acq = qSTCHSet(
            model=model,
            ref_point=ref_point,
            mu=0.1,
            sampler=sampler,
            Y_range=Y_range,
            Y_min=Y_min,
        )

        acq_vals = torch.full((n_remaining,), float("-inf"), device=device, dtype=dtype)

        with torch.no_grad():
            for i in range(n_remaining):
                if i in selected_set:
                    continue
                if selected_X:
                    candidate_set = torch.stack(selected_X + [X_pool_remaining[i]], dim=0)
                else:
                    candidate_set = X_pool_remaining[i:i+1]
                acq_vals[i] = acq(candidate_set.unsqueeze(0))

        best_idx = acq_vals.argmax().item()
        selected_indices.append(best_idx)
        selected_X.append(X_pool_remaining[best_idx])
        selected_set.add(best_idx)

    return selected_indices


# ---------------------------------------------------------------------------
# Main BO loop
# ---------------------------------------------------------------------------

def run_molecular_benchmark(
    dataset: str,
    n_seeds: int,
    n_init: int,
    n_iters: int,
    batch_size: int,
    mc_samples: int,
    output_dir: Path,
    device,
    dtype,
    fast_stch: bool = True,
    kernel: str = "rbf",
):
    """Pool-based multi-objective BO following Kristiadi et al. Algorithm 1."""

    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    if dataset == "multi_redox":
        data_path = data_dir / "redox_mer.csv"
        X_pool, Y_pool, smiles, obj_names = load_multi_redox(data_path, device, dtype)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    N, d = X_pool.shape
    num_obj = Y_pool.shape[1]

    # Compute reference point (worst observed - small margin)
    ref_point = Y_pool.min(dim=0).values - 0.1 * (Y_pool.max(dim=0).values - Y_pool.min(dim=0).values)

    # Compute oracle hypervolume (full pool Pareto front)
    bd_oracle = DominatedPartitioning(ref_point=ref_point, Y=Y_pool)
    oracle_hv = bd_oracle.compute_hypervolume().item()

    print(f"\n{'='*70}")
    print(f"Molecular Benchmark: {dataset}")
    print(f"  Pool size: {N} | Features: {d} | Objectives: {num_obj}")
    print(f"  Objectives: {obj_names}")
    for i, name in enumerate(obj_names):
        ymin, ymax = Y_pool[:, i].min().item(), Y_pool[:, i].max().item()
        print(f"    {name}: [{ymin:.4f}, {ymax:.4f}]")
    print(f"  n_init: {n_init} | n_iters: {n_iters} | batch_size (K): {batch_size}")
    print(f"  MC samples: {mc_samples} | Seeds: {n_seeds} | Kernel: {kernel}")
    print(f"  Oracle HV (full pool): {oracle_hv:.6e}")
    print(f"  Ref point: {[f'{x:.4f}' for x in ref_point.cpu().tolist()]}")
    print(f"{'='*70}")

    methods = ["random", "qnparego", "qstch_set"]
    labels = {"random": "Random", "qnparego": "qNParEGO", "qstch_set": "qSTCH-Set"}
    all_results = {m: [] for m in methods}

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        print(f"\n--- Seed {seed+1}/{n_seeds} ---")

        # Random initial pool indices
        init_indices = rng.choice(N, size=n_init, replace=False).tolist()

        # Per-method state
        state = {}
        for method in methods:
            state[method] = {
                "observed_indices": list(init_indices),
                "remaining_indices": [i for i in range(N) if i not in init_indices],
            }

        # Initial HV
        Y_init = Y_pool[init_indices]
        bd = DominatedPartitioning(ref_point=ref_point, Y=Y_init)
        init_hv = bd.compute_hypervolume().item()
        hvs = {m: [init_hv] for m in methods}

        for iteration in range(1, n_iters + 1):
            t0 = time.monotonic()

            for method in methods:
                s = state[method]
                obs_idx = s["observed_indices"]
                rem_idx = s["remaining_indices"]

                if len(rem_idx) == 0:
                    hvs[method].append(hvs[method][-1])
                    continue

                X_train = X_pool[obs_idx]
                Y_train = Y_pool[obs_idx]
                X_remaining = X_pool[rem_idx]

                if method == "random":
                    chosen_local = select_random(
                        np.arange(len(rem_idx)), batch_size, rng
                    )
                    chosen_pool_idx = [rem_idx[i] for i in chosen_local]

                elif method == "qnparego":
                    model = fit_model(X_train, Y_train, kernel=kernel)
                    chosen_local = select_qnparego(
                        model, X_remaining, Y_train, batch_size,
                        mc_samples, device, dtype
                    )
                    chosen_pool_idx = [rem_idx[i] for i in chosen_local]

                elif method == "qstch_set":
                    model = fit_model(X_train, Y_train, kernel=kernel)
                    select_fn = select_qstch_set_fast if fast_stch else select_qstch_set
                    chosen_local = select_fn(
                        model, X_remaining, Y_train, batch_size,
                        mc_samples, ref_point, device, dtype
                    )
                    chosen_pool_idx = [rem_idx[i] for i in chosen_local]

                # Update observed / remaining
                s["observed_indices"].extend(chosen_pool_idx)
                s["remaining_indices"] = [i for i in rem_idx if i not in chosen_pool_idx]

                # Compute HV
                Y_obs = Y_pool[s["observed_indices"]]
                bd = DominatedPartitioning(ref_point=ref_point, Y=Y_obs)
                hvs[method].append(bd.compute_hypervolume().item())

            t1 = time.monotonic()
            hv_str = " | ".join(
                f"{labels[m]}={hvs[m][-1]:.4f}" for m in methods
            )
            print(f"  Iter {iteration:>2}/{n_iters}: {hv_str} ({t1-t0:.1f}s)")

        for method in methods:
            all_results[method].append(hvs[method])

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": dataset,
        "pool_size": N,
        "feature_dim": d,
        "num_objectives": num_obj,
        "objective_names": obj_names,
        "kernel": kernel,
        "n_init": n_init,
        "n_iters": n_iters,
        "batch_size": batch_size,
        "mc_samples": mc_samples,
        "n_seeds": n_seeds,
        "oracle_hv": oracle_hv,
        "ref_point": ref_point.cpu().tolist(),
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

    print(f"\n{'='*70}")
    print(f"Final HV ({dataset}, m={num_obj}) after {n_iters} iterations:")
    for method in methods:
        r = summary[method]
        pct = r["final_hv_mean"] / oracle_hv * 100
        print(f"  {labels[method]:15s}: {r['final_hv_mean']:.6e} +/- {r['final_hv_std']:.6e} ({pct:.1f}% of oracle)")
    print(f"  Oracle (full pool): {oracle_hv:.6e}")
    print(f"{'='*70}")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {output_dir / 'summary.json'}")

    # Plot convergence
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"random": "#888", "qnparego": "#e41a1c", "qstch_set": "#984ea3"}
        fig, ax = plt.subplots(figsize=(8, 5))
        iters = list(range(n_iters + 1))
        for method in methods:
            mean = np.array(summary[method]["hv_mean"])
            std = np.array(summary[method]["hv_std"])
            ax.plot(iters, mean, label=labels[method], color=colors[method], linewidth=2)
            ax.fill_between(iters, mean - std, mean + std, alpha=0.15, color=colors[method])

        ax.axhline(oracle_hv, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Oracle")
        ax.set_xlabel("BO Iteration (each selects K molecules)")
        ax.set_ylabel("Hypervolume")
        ax.set_title(f"Molecular Optimization: {dataset} (m={num_obj})\n"
                     f"(pool={N}, K={batch_size}, {n_seeds} seeds, {n_init} init)")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "convergence.png", dpi=150, bbox_inches="tight")
        print(f"Convergence plot saved to {output_dir / 'convergence.png'}")
    except Exception as e:
        print(f"Plot failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pool-based molecular multi-objective BO benchmark")
    parser.add_argument("--dataset", default="multi_redox",
                        choices=["multi_redox"],
                        help="Dataset name")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--n-init", type=int, default=20, help="Initial random observations")
    parser.add_argument("--n-iters", type=int, default=30, help="Number of BO iterations")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Candidates per iteration (K). Default=4 (num_objectives)")
    parser.add_argument("--mc-samples", type=int, default=64, help="MC samples for acquisition")
    parser.add_argument("--fast-stch", action="store_true", default=True,
                        help="Use fast greedy STCH-Set selection (default)")
    parser.add_argument("--no-fast-stch", action="store_false", dest="fast_stch")
    parser.add_argument("--kernel", type=str, default="rbf",
                        choices=["rbf", "tanimoto"],
                        help="GP kernel: rbf (default) or tanimoto")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dtype = torch.double
    device = torch.device(args.device)

    if args.output_dir is None:
        out = Path(__file__).parent.parent / "results" / f"molecular_{args.dataset}_{args.kernel}"
    else:
        out = Path(args.output_dir)

    run_molecular_benchmark(
        dataset=args.dataset,
        n_seeds=args.seeds,
        n_init=args.n_init,
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        mc_samples=args.mc_samples,
        output_dir=out,
        device=device,
        dtype=dtype,
        fast_stch=args.fast_stch,
        kernel=args.kernel,
    )
