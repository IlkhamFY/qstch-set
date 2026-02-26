"""
DMTA (Design-Make-Test-Analyze) Molecular Benchmark.

Evaluates batch selection strategies for multi-objective molecular optimization
under realistic DMTA cycle constraints: small coordinated batches (K=5),
many objectives (m=4-8), and the need for SET-level quality rather than
individual point quality.

Key insight: qSTCH-Set optimizes the batch AS A SET (minimax: best worst-covered
objective), while qNParEGO selects candidates independently with random
scalarizations. This benchmark measures whether coordinated set selection
actually produces better batches.

Metrics beyond HV:
  - Worst-Objective Coverage (WOC): min across objectives of normalized coverage
  - Min-Max Regret: max across objectives of (oracle - best_selected)
  - Per-Batch Diversity: avg pairwise Tanimoto distance within each batch
  - Objective Specialization: how many distinct "specialist roles" the batch fills

Dataset: BTZ multi-redox derivatives (1407 molecules)
  DFT properties: Ered, HOMO, Gsol, Absorption
  RDKit properties: QED, LogP, MolWt, Fsp3
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
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from stch_botorch.acquisition.stch_set_bo import qSTCHSet

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dmta_dataset(data_path: Path, num_objectives: int, device, dtype):
    """Load BTZ dataset with configurable number of objectives.

    m=4: Ered, HOMO, Gsol, Absorption (DFT only)
    m=6: + QED, neg_LogP
    m=8: + neg_MolWt, Fsp3

    All converted to maximization convention.
    """
    import csv

    rows = []
    with open(data_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    smiles_list = [r["SMILES"] for r in rows]

    # DFT properties (always included)
    ered = np.array([float(r["Ered"]) for r in rows])
    homo = np.array([float(r["HOMO"]) for r in rows])
    gsol = np.array([float(r["Gsol"]) for r in rows])
    absorp = np.array([float(r["Absorption Wavelength"]) for r in rows])

    # All objectives in maximization convention
    obj_arrays = [
        -ered,    # neg_Ered: minimize reduction potential
        homo,     # HOMO: maximize
        -gsol,    # neg_Gsol: minimize solvation energy
        absorp,   # Absorption: maximize wavelength
    ]
    obj_names = ["neg_Ered", "HOMO", "neg_Gsol", "Absorption"]

    if num_objectives >= 6:
        # Compute RDKit descriptors
        from rdkit import Chem
        from rdkit.Chem import Descriptors, QED as QEDModule

        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        qed_vals = np.array([QEDModule.qed(m) if m else 0.0 for m in mols])
        logp_vals = np.array([Descriptors.MolLogP(m) if m else 0.0 for m in mols])

        obj_arrays.extend([
            qed_vals,   # QED: maximize drug-likeness
            -logp_vals, # neg_LogP: minimize for solubility
        ])
        obj_names.extend(["QED", "neg_LogP"])

    if num_objectives >= 8:
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        if 'mols' not in dir():
            mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        mw_vals = np.array([Descriptors.MolWt(m) if m else 500.0 for m in mols])
        fsp3_vals = np.array([Descriptors.FractionCSP3(m) if m else 0.0 for m in mols])

        obj_arrays.extend([
            -mw_vals,   # neg_MolWt: prefer lighter molecules
            fsp3_vals,  # Fsp3: maximize (escape flatland)
        ])
        obj_names.extend(["neg_MolWt", "Fsp3"])

    # Truncate to requested number
    obj_arrays = obj_arrays[:num_objectives]
    obj_names = obj_names[:num_objectives]

    # Compute Morgan fingerprints
    fps = _compute_morgan_fps(smiles_list)

    X_pool = torch.tensor(fps, device=device, dtype=dtype)
    Y_pool = torch.tensor(
        np.stack(obj_arrays, axis=1),
        device=device, dtype=dtype,
    )

    return X_pool, Y_pool, smiles_list, obj_names


def _compute_morgan_fps(smiles_list, radius=2, n_bits=1024):
    """Compute Morgan fingerprints."""
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
                fps.append(gen.GetFingerprintAsNumPy(mol))
        return np.array(fps, dtype=np.float64)
    except ImportError:
        rng = np.random.RandomState(42)
        return rng.rand(len(smiles_list), n_bits)


# ---------------------------------------------------------------------------
# SET-QUALITY METRICS
# ---------------------------------------------------------------------------

def compute_worst_objective_coverage(Y_selected, Y_pool):
    """Worst-Objective Coverage (WOC).

    For each objective i:
      coverage_i = (best_selected_i - worst_pool_i) / (best_pool_i - worst_pool_i)

    WOC = min across objectives of coverage_i.

    Range: [0, 1]. Higher = better. Measures: does the selected set
    cover ALL objectives well? A method that neglects one objective
    will score low.
    """
    pool_min = Y_pool.min(dim=0).values
    pool_max = Y_pool.max(dim=0).values
    pool_range = (pool_max - pool_min).clamp(min=1e-8)

    best_selected = Y_selected.max(dim=0).values
    coverage_per_obj = (best_selected - pool_min) / pool_range
    coverage_per_obj = coverage_per_obj.clamp(0.0, 1.0)

    woc = coverage_per_obj.min().item()
    per_obj = coverage_per_obj.cpu().tolist()
    return woc, per_obj


def compute_minmax_regret(Y_selected, Y_pool):
    """Min-Max Regret.

    For each objective i:
      regret_i = best_pool_i - best_selected_i

    Report max(regret) across objectives. Lower = better.
    Measures: what's the WORST gap between oracle and our selection?
    """
    pool_max = Y_pool.max(dim=0).values
    best_selected = Y_selected.max(dim=0).values
    regret_per_obj = pool_max - best_selected
    regret_per_obj = regret_per_obj.clamp(min=0.0)

    # Normalize by pool range for comparability
    pool_range = (Y_pool.max(dim=0).values - Y_pool.min(dim=0).values).clamp(min=1e-8)
    norm_regret = regret_per_obj / pool_range

    max_regret = norm_regret.max().item()
    per_obj = norm_regret.cpu().tolist()
    return max_regret, per_obj


def compute_batch_diversity(X_batch):
    """Average pairwise Tanimoto distance within a batch.

    For binary fingerprints: Tanimoto(a,b) = |a&b| / |a|b|
    Distance = 1 - Tanimoto similarity.

    Higher = more diverse batch.
    """
    K = X_batch.shape[0]
    if K < 2:
        return 0.0

    # Compute pairwise Tanimoto similarity
    # For binary FPs: sim = dot(a,b) / (|a|^2 + |b|^2 - dot(a,b))
    dots = X_batch @ X_batch.T
    norms_sq = (X_batch ** 2).sum(dim=-1)
    denom = norms_sq.unsqueeze(0) + norms_sq.unsqueeze(1) - dots
    denom = denom.clamp(min=1e-8)
    sim_matrix = dots / denom

    # Average off-diagonal distance
    mask = ~torch.eye(K, dtype=torch.bool, device=X_batch.device)
    avg_distance = (1.0 - sim_matrix[mask]).mean().item()
    return avg_distance


def compute_specialization_score(Y_batch, Y_pool):
    """Objective Specialization Score.

    For each candidate in the batch, determine which objective(s) it is
    "best at" relative to the pool. Count how many distinct objectives
    have a specialist in the batch.

    Score = number of objectives that have at least one specialist / m.
    Range: [0, 1]. Higher = better role differentiation.
    """
    K, m = Y_batch.shape
    pool_max = Y_pool.max(dim=0).values
    pool_min = Y_pool.min(dim=0).values
    pool_range = (pool_max - pool_min).clamp(min=1e-8)

    # Normalize batch objectives to [0,1]
    Y_norm = (Y_batch - pool_min) / pool_range

    # For each candidate, find its best (highest normalized) objective
    best_obj_per_candidate = Y_norm.argmax(dim=-1)  # (K,)

    # Count unique objectives covered
    unique_objectives = len(set(best_obj_per_candidate.cpu().tolist()))
    score = unique_objectives / m
    specialists = best_obj_per_candidate.cpu().tolist()

    return score, specialists


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_model(train_x, train_y):
    """Fit independent SingleTaskGP per objective with Standardize."""
    models = []
    for i in range(train_y.shape[-1]):
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
# Selection methods
# ---------------------------------------------------------------------------

def select_random(n_remaining, batch_size, rng):
    """Random selection."""
    return rng.choice(n_remaining, size=min(batch_size, n_remaining), replace=False).tolist()


def select_qnparego(model, X_remaining, Y_train, batch_size, mc_samples, device, dtype):
    """qNParEGO: independent random Chebyshev scalarization per candidate."""
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
    num_obj = Y_train.shape[-1]
    n_remaining = X_remaining.shape[0]
    selected = []

    for k in range(batch_size):
        weights = sample_simplex(num_obj).squeeze().to(device=device, dtype=dtype)
        scalarization = get_chebyshev_scalarization(weights=weights, Y=Y_train)
        objective = GenericMCObjective(scalarization)

        acq = qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=X_remaining[:1],
            sampler=sampler,
            objective=objective,
            prune_baseline=True,
        )

        acq_vals = torch.full((n_remaining,), float("-inf"), device=device, dtype=dtype)
        with torch.no_grad():
            for start in range(0, n_remaining, 256):
                end = min(start + 256, n_remaining)
                X_batch = X_remaining[start:end].unsqueeze(1)
                acq_vals[start:end] = acq(X_batch)

        for idx in selected:
            acq_vals[idx] = float("-inf")

        best_idx = acq_vals.argmax().item()
        selected.append(best_idx)

    return selected


def select_qstch_set(model, X_remaining, Y_train, batch_size, mc_samples,
                     ref_point, device, dtype, weight_mode="original", mu=0.1):
    """qSTCH-Set: greedy sequential set construction with joint scoring.

    For each slot k=1..K:
      Build qSTCHSet with current partial set, evaluate each remaining
      candidate as the next addition, pick best.

    The STCH-Set scalarization scores the joint set quality (minimax over
    objectives of min-over-candidates), so even greedily, it tends to select
    candidates that COMPLEMENT the existing set.
    """
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))

    Y_min = Y_train.min(dim=0).values
    Y_max = Y_train.max(dim=0).values
    Y_bounds = torch.stack([Y_min, Y_max])

    selected = []
    selected_X = []
    selected_set = set()
    n_remaining = X_remaining.shape[0]
    EVAL_BATCH = 128  # candidates evaluated in one forward pass

    for k in range(batch_size):
        acq = qSTCHSet(
            model=model,
            ref_point=ref_point,
            mu=mu,
            sampler=sampler,
            Y_bounds=Y_bounds,
            weight_mode=weight_mode,
        )

        acq_vals = torch.full((n_remaining,), float("-inf"), device=device, dtype=dtype)

        with torch.no_grad():
            if selected_X:
                # Partial set exists: evaluate each candidate as an addition.
                # Build batched sets: (EVAL_BATCH, k+1, d)
                context = torch.stack(selected_X, dim=0)  # (k, d)
                for start in range(0, n_remaining, EVAL_BATCH):
                    end = min(start + EVAL_BATCH, n_remaining)
                    B = end - start
                    # Expand context for each candidate
                    ctx_exp = context.unsqueeze(0).expand(B, -1, -1)  # (B, k, d)
                    new_c = X_remaining[start:end].unsqueeze(1)        # (B, 1, d)
                    sets = torch.cat([ctx_exp, new_c], dim=1)          # (B, k+1, d)
                    acq_vals[start:end] = acq(sets)
            else:
                # First slot: evaluate singletons
                for start in range(0, n_remaining, EVAL_BATCH):
                    end = min(start + EVAL_BATCH, n_remaining)
                    singletons = X_remaining[start:end].unsqueeze(1)   # (B, 1, d)
                    acq_vals[start:end] = acq(singletons)

        # Mask already selected
        for sel_idx in selected:
            acq_vals[sel_idx] = float("-inf")

        best_idx = acq_vals.argmax().item()
        selected.append(best_idx)
        selected_X.append(X_remaining[best_idx])
        selected_set.add(best_idx)

    return selected


# ---------------------------------------------------------------------------
# Main DMTA loop
# ---------------------------------------------------------------------------

def run_dmta_benchmark(
    num_objectives: int,
    n_seeds: int,
    n_init: int,
    n_iters: int,
    batch_size: int,
    mc_samples: int,
    output_dir: Path,
    device,
    dtype,
    weight_mode: str = "original",
    mu: float = 0.1,
):
    """Pool-based DMTA benchmark with set-quality metrics."""

    data_path = Path(__file__).parent.parent / "data" / "redox_mer.csv"
    X_pool, Y_pool, smiles, obj_names = load_dmta_dataset(
        data_path, num_objectives, device, dtype
    )

    N, d = X_pool.shape
    m = Y_pool.shape[1]

    # Reference point: worst - 10% margin
    ref_point = Y_pool.min(dim=0).values - 0.1 * (
        Y_pool.max(dim=0).values - Y_pool.min(dim=0).values
    )

    # Oracle HV
    bd_oracle = DominatedPartitioning(ref_point=ref_point, Y=Y_pool)
    oracle_hv = bd_oracle.compute_hypervolume().item()

    print(f"\n{'='*70}")
    print(f"DMTA Molecular Benchmark (m={m}, K={batch_size})")
    print(f"  Pool: {N} | Features: {d} | Objectives: {m}")
    print(f"  Objectives: {obj_names}")
    for i, name in enumerate(obj_names):
        ymin, ymax = Y_pool[:, i].min().item(), Y_pool[:, i].max().item()
        print(f"    {name}: [{ymin:.4f}, {ymax:.4f}]")
    print(f"  n_init={n_init} | n_iters={n_iters} | K={batch_size} | MC={mc_samples}")
    print(f"  Seeds: {n_seeds} | weight_mode: {weight_mode} | mu: {mu}")
    print(f"  Oracle HV: {oracle_hv:.6e}")
    print(f"{'='*70}")

    methods = ["random", "qnparego", "qstch_set"]
    labels = {"random": "Random", "qnparego": "qNParEGO", "qstch_set": "qSTCH-Set"}

    all_results = {m_name: {
        "hv": [], "woc": [], "regret": [], "diversity": [], "specialization": [],
        "woc_per_obj": [], "regret_per_obj": [],
    } for m_name in methods}

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        print(f"\n--- Seed {seed+1}/{n_seeds} ---")

        init_indices = rng.choice(N, size=n_init, replace=False).tolist()

        state = {}
        for method in methods:
            state[method] = {
                "observed": list(init_indices),
                "remaining": [i for i in range(N) if i not in init_indices],
            }

        # Initial metrics
        Y_init = Y_pool[init_indices]
        bd = DominatedPartitioning(ref_point=ref_point, Y=Y_init)
        init_hv = bd.compute_hypervolume().item()
        init_woc, init_woc_obj = compute_worst_objective_coverage(Y_init, Y_pool)
        init_regret, init_regret_obj = compute_minmax_regret(Y_init, Y_pool)

        hvs = {m_name: [init_hv] for m_name in methods}
        wocs = {m_name: [init_woc] for m_name in methods}
        regrets = {m_name: [init_regret] for m_name in methods}
        diversities = {m_name: [] for m_name in methods}
        specializations = {m_name: [] for m_name in methods}
        woc_per_obj_hist = {m_name: [init_woc_obj] for m_name in methods}
        regret_per_obj_hist = {m_name: [init_regret_obj] for m_name in methods}

        for iteration in range(1, n_iters + 1):
            t0 = time.monotonic()

            for method in methods:
                s = state[method]
                obs_idx = s["observed"]
                rem_idx = s["remaining"]

                if len(rem_idx) == 0:
                    hvs[method].append(hvs[method][-1])
                    wocs[method].append(wocs[method][-1])
                    regrets[method].append(regrets[method][-1])
                    diversities[method].append(0.0)
                    specializations[method].append(0.0)
                    woc_per_obj_hist[method].append(woc_per_obj_hist[method][-1])
                    regret_per_obj_hist[method].append(regret_per_obj_hist[method][-1])
                    continue

                X_train = X_pool[obs_idx]
                Y_train = Y_pool[obs_idx]
                X_remaining = X_pool[rem_idx]

                # Select batch
                if method == "random":
                    chosen_local = select_random(len(rem_idx), batch_size, rng)
                elif method == "qnparego":
                    model = fit_model(X_train, Y_train)
                    chosen_local = select_qnparego(
                        model, X_remaining, Y_train, batch_size,
                        mc_samples, device, dtype
                    )
                elif method == "qstch_set":
                    model = fit_model(X_train, Y_train)
                    chosen_local = select_qstch_set(
                        model, X_remaining, Y_train, batch_size,
                        mc_samples, ref_point, device, dtype,
                        weight_mode=weight_mode, mu=mu,
                    )

                chosen_pool_idx = [rem_idx[i] for i in chosen_local]

                # --- Per-batch metrics ---
                X_batch = X_pool[chosen_pool_idx]
                Y_batch = Y_pool[chosen_pool_idx]

                batch_div = compute_batch_diversity(X_batch)
                spec_score, _ = compute_specialization_score(Y_batch, Y_pool)
                diversities[method].append(batch_div)
                specializations[method].append(spec_score)

                # Update state
                s["observed"].extend(chosen_pool_idx)
                s["remaining"] = [i for i in rem_idx if i not in chosen_pool_idx]

                # --- Cumulative metrics ---
                Y_obs = Y_pool[s["observed"]]
                bd = DominatedPartitioning(ref_point=ref_point, Y=Y_obs)
                hvs[method].append(bd.compute_hypervolume().item())

                woc, woc_obj = compute_worst_objective_coverage(Y_obs, Y_pool)
                wocs[method].append(woc)
                woc_per_obj_hist[method].append(woc_obj)

                regret, regret_obj = compute_minmax_regret(Y_obs, Y_pool)
                regrets[method].append(regret)
                regret_per_obj_hist[method].append(regret_obj)

            t1 = time.monotonic()
            hv_str = " | ".join(f"{labels[mn]}={hvs[mn][-1]:.2f}" for mn in methods)
            woc_str = " | ".join(f"{labels[mn]}={wocs[mn][-1]:.3f}" for mn in methods)
            print(f"  Iter {iteration:>2}: HV: {hv_str} | WOC: {woc_str} ({t1-t0:.1f}s)")

        # Store seed results
        for method in methods:
            all_results[method]["hv"].append(hvs[method])
            all_results[method]["woc"].append(wocs[method])
            all_results[method]["regret"].append(regrets[method])
            all_results[method]["diversity"].append(diversities[method])
            all_results[method]["specialization"].append(specializations[method])
            all_results[method]["woc_per_obj"].append(woc_per_obj_hist[method])
            all_results[method]["regret_per_obj"].append(regret_per_obj_hist[method])

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "benchmark": "dmta",
        "num_objectives": m,
        "objective_names": obj_names,
        "pool_size": N,
        "n_init": n_init,
        "n_iters": n_iters,
        "batch_size": batch_size,
        "mc_samples": mc_samples,
        "n_seeds": n_seeds,
        "oracle_hv": oracle_hv,
        "ref_point": ref_point.cpu().tolist(),
        "weight_mode": weight_mode,
        "mu": mu,
    }

    for method in methods:
        r = all_results[method]
        hv_arr = np.array(r["hv"])
        woc_arr = np.array(r["woc"])
        reg_arr = np.array(r["regret"])
        div_arr = np.array(r["diversity"])
        spec_arr = np.array(r["specialization"])

        summary[method] = {
            "hv_mean": hv_arr.mean(axis=0).tolist(),
            "hv_std": hv_arr.std(axis=0).tolist(),
            "final_hv": float(hv_arr[:, -1].mean()),
            "woc_mean": woc_arr.mean(axis=0).tolist(),
            "woc_std": woc_arr.std(axis=0).tolist(),
            "final_woc": float(woc_arr[:, -1].mean()),
            "regret_mean": reg_arr.mean(axis=0).tolist(),
            "regret_std": reg_arr.std(axis=0).tolist(),
            "final_regret": float(reg_arr[:, -1].mean()),
            "diversity_mean": div_arr.mean(axis=0).tolist() if div_arr.size else [],
            "final_diversity": float(div_arr.mean()) if div_arr.size else 0.0,
            "specialization_mean": spec_arr.mean(axis=0).tolist() if spec_arr.size else [],
            "final_specialization": float(spec_arr.mean()) if spec_arr.size else 0.0,
            "hv_all": hv_arr.tolist(),
            "woc_all": woc_arr.tolist(),
            "regret_all": reg_arr.tolist(),
        }

    # Print final summary
    print(f"\n{'='*70}")
    print(f"DMTA Benchmark Results (m={m}, K={batch_size}, {n_seeds} seeds)")
    print(f"{'='*70}")
    header = f"{'Method':>15} | {'HV':>12} | {'WOC':>8} | {'Regret':>8} | {'Diversity':>9} | {'Special.':>8}"
    print(header)
    print("-" * len(header))
    for method in methods:
        r = summary[method]
        print(f"{labels[method]:>15} | {r['final_hv']:>12.2f} | "
              f"{r['final_woc']:>8.4f} | {r['final_regret']:>8.4f} | "
              f"{r['final_diversity']:>9.4f} | {r['final_specialization']:>8.4f}")
    print(f"{'Oracle':>15} | {oracle_hv:>12.2f} |")
    print(f"{'='*70}")

    out_file = output_dir / f"dmta_m{m}_K{batch_size}_{weight_mode}.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {out_file}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DMTA Molecular Benchmark")
    parser.add_argument("--m", type=int, default=6, choices=[4, 6, 8],
                        help="Number of objectives (4=DFT only, 6=+QED/LogP, 8=+MolWt/Fsp3)")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--n-iters", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=5, help="K: candidates per DMTA batch")
    parser.add_argument("--mc-samples", type=int, default=64)
    parser.add_argument("--weight-mode", type=str, default="original",
                        choices=["original", "log_additive"])
    parser.add_argument("--mu", type=float, default=0.1)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dtype = torch.double
    device = torch.device(args.device)

    if args.output_dir is None:
        out = Path(__file__).parent.parent / "results" / f"dmta_m{args.m}_K{args.batch_size}"
    else:
        out = Path(args.output_dir)

    run_dmta_benchmark(
        num_objectives=args.m,
        n_seeds=args.seeds,
        n_init=args.n_init,
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        mc_samples=args.mc_samples,
        output_dir=out,
        device=device,
        dtype=dtype,
        weight_mode=args.weight_mode,
        mu=args.mu,
    )
