"""
ZDT Benchmark Suite for STCH-BoTorch
Publication-quality comparison: STCH-NParEGO vs Vanilla qNParEGO vs qEHVI vs NSGA-II

Authors: stch-botorch team
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure stch_botorch is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem as PymooProblem
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.problems import get_problem
from pymoo.operators.sampling.rnd import FloatRandomSampling

from stch_botorch.objectives import SmoothChebyshevObjective

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.double)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ─── Problem Setup ───────────────────────────────────────────────────────────

class ZDTProblem:
    """Wrapper around pymoo ZDT problems for BoTorch."""
    
    def __init__(self, name: str, n_var: int = 30):
        self.name = name
        self.n_var = n_var
        self.n_obj = 2
        self.pymoo_problem = get_problem(name, n_var=n_var)
        self.bounds = torch.tensor(
            [[0.0] * n_var, [1.0] * n_var], dtype=torch.double
        )
        # Reference point for HV (worst-case upper bound for minimization objectives)
        # ZDT objectives are in [0, ~1-11] range; use 11.0 to be safe
        self.ref_point_hv = torch.tensor([11.0, 11.0], dtype=torch.double)
        
        # Get true Pareto front from pymoo
        pf = self.pymoo_problem.pareto_front()
        self.true_pf = torch.tensor(pf, dtype=torch.double) if pf is not None else None
        
        # Compute true HV
        if self.true_pf is not None:
            partitioning = DominatedPartitioning(
                ref_point=self.ref_point_hv, Y=-self.true_pf  # negate: BoTorch uses maximization
            )
            # Wait — HV computation with negated objectives needs negated ref_point too
            # Let me think carefully...
            # BoTorch's DominatedPartitioning expects maximization convention
            # Our objectives are to be minimized. For HV calculation:
            # We compute HV in the original (minimization) space using custom logic
            # Actually, let's just compute HV directly in minimization space
            self.true_hv = self._compute_hv(self.true_pf)
        else:
            self.true_hv = None
    
    def _compute_hv(self, Y_min: torch.Tensor) -> float:
        """Compute hypervolume in minimization space.
        
        Args:
            Y_min: Objective values to minimize, shape (n, m)
        
        Returns:
            Hypervolume indicator value.
        """
        if Y_min.shape[0] == 0:
            return 0.0
        # Negate for BoTorch (which assumes maximization)
        Y_max = -Y_min
        ref_point_max = -self.ref_point_hv
        # Filter to non-dominated in maximization space
        pareto_mask = is_non_dominated(Y_max)
        Y_pareto = Y_max[pareto_mask]
        if Y_pareto.shape[0] == 0:
            return 0.0
        partitioning = DominatedPartitioning(ref_point=ref_point_max, Y=Y_pareto)
        return partitioning.compute_hypervolume().item()

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate objectives (minimization).
        
        Args:
            X: Input tensor of shape (n, n_var) in [0, 1]^n_var.
            
        Returns:
            Y: Objective values of shape (n, 2), to be minimized.
        """
        X_np = X.detach().cpu().numpy()
        Y_np = np.zeros((X_np.shape[0], self.n_obj))
        for i in range(X_np.shape[0]):
            out = {}
            self.pymoo_problem._evaluate(X_np[i:i+1], out)
            Y_np[i] = out["F"][0]
        return torch.tensor(Y_np, dtype=torch.double)


# ─── BO Methods ──────────────────────────────────────────────────────────────

def generate_initial_data(problem: ZDTProblem, n_init: int, seed: int):
    """Generate initial random data points."""
    torch.manual_seed(seed)
    X = draw_sobol_samples(bounds=problem.bounds, n=n_init, q=1).squeeze(1)
    Y = problem.evaluate(X)
    return X, Y


def fit_model(X: torch.Tensor, Y: torch.Tensor, bounds: torch.Tensor):
    """Fit a SingleTaskGP model. Y is in minimization convention (negate for BoTorch)."""
    # Normalize X to [0,1] (already in [0,1] for ZDT)
    # Negate Y because BoTorch models assume maximization
    model = SingleTaskGP(X, -Y, outcome_transform=Standardize(m=Y.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def run_stch_nparego(problem: ZDTProblem, X_init, Y_init, n_iter: int, seed: int):
    """STCH-NParEGO: Our method."""
    X, Y = X_init.clone(), Y_init.clone()
    hv_history = []
    
    for i in range(n_iter):
        try:
            model = fit_model(X, Y, problem.bounds)
        except Exception as e:
            print(f"  [STCH] iter {i}: GP fit failed: {e}")
            hv_history.append(hv_history[-1] if hv_history else 0.0)
            continue
        
        # Random Dirichlet weights each iteration (NParEGO style)
        torch.manual_seed(seed * 1000 + i)
        weights = torch.distributions.Dirichlet(torch.ones(problem.n_obj)).sample()
        
        # Use worst observed values as reference point for scalarization
        ref_point_scalar = Y.max(dim=0).values  # worst per objective (max since minimizing)
        
        objective = SmoothChebyshevObjective(
            weights=weights,
            ref_point=ref_point_scalar,
            mu=0.1,
        )
        
        # Use qLogNoisyExpectedImprovement with the scalarized objective
        # The model outputs -Y (negated), and objective maps posterior samples
        # We need a single-output acquisition function on the scalarized output
        # Actually, for NParEGO pattern: scalarize then use single-objective acq
        # Let's use qNoisyExpectedImprovement on scalarized posterior
        
        # Build scalarized model: apply objective to model posterior samples
        # BoTorch's qNParEGO pattern: use objective with qLogNoisyExpectedImprovement
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        
        from botorch.acquisition.logei import qLogNoisyExpectedImprovement
        
        # For NParEGO, we scalarize the multi-output model to single output
        # then optimize EI on that. But qLogNoisyEI expects single-output model.
        # Instead, let's manually scalarize: use objective inside acq function.
        # BoTorch's qNoisyExpectedHypervolumeImprovement won't work here.
        # 
        # Proper approach: use get_chebyshev_scalarization pattern but with STCH
        # The scalarization is applied to the model's posterior samples.
        
        # We'll use a custom approach: sample from posterior, scalarize, pick best
        from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
        
        # Create a wrapper that scalarizes multi-output to single output
        # Using GenericMCObjective
        from botorch.acquisition.objective import GenericMCObjective
        
        def make_stch_obj(w, rp, mu_val):
            """Create a closure for STCH scalarization."""
            def obj_fn(samples, X=None):
                # samples: (sample_shape x batch x q x m) — model outputs are -Y (negated min)
                # So samples represent "to be maximized" values
                # STCH expects minimization-convention Y, so negate back
                Y_min = -samples
                return smooth_chebyshev_from_min(Y_min, w, rp, mu_val)
            return obj_fn
        
        def smooth_chebyshev_from_min(Y_min, w, rp, mu_val):
            """Apply STCH to minimization-convention Y, return utility to maximize."""
            from stch_botorch.scalarization import smooth_chebyshev
            # smooth_chebyshev already returns utility for maximization
            # It expects Y in minimization convention
            return smooth_chebyshev(Y_min, w, rp, mu_val)
        
        mc_objective = GenericMCObjective(make_stch_obj(weights, ref_point_scalar, 0.1))
        
        acq = qNoisyExpectedImprovement(
            model=model,
            X_baseline=X,
            sampler=sampler,
            objective=mc_objective,
            prune_baseline=True,
        )
        
        # Optimize acquisition function
        try:
            candidates, _ = optimize_acqf(
                acq_function=acq,
                bounds=problem.bounds,
                q=1,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
            )
        except Exception as e:
            print(f"  [STCH] iter {i}: acqf optim failed: {e}")
            hv_history.append(hv_history[-1] if hv_history else 0.0)
            continue
        
        new_Y = problem.evaluate(candidates)
        X = torch.cat([X, candidates])
        Y = torch.cat([Y, new_Y])
        
        hv = problem._compute_hv(Y)
        hv_history.append(hv)
        
        if (i + 1) % 10 == 0:
            print(f"  [STCH] iter {i+1}/{n_iter}: HV={hv:.4f}")
    
    return X, Y, hv_history


def run_vanilla_nparego(problem: ZDTProblem, X_init, Y_init, n_iter: int, seed: int):
    """Vanilla qNParEGO with BoTorch's built-in Chebyshev scalarization."""
    from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
    from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
    from botorch.acquisition.objective import GenericMCObjective
    
    X, Y = X_init.clone(), Y_init.clone()
    hv_history = []
    
    for i in range(n_iter):
        try:
            model = fit_model(X, Y, problem.bounds)
        except Exception as e:
            print(f"  [NParEGO] iter {i}: GP fit failed: {e}")
            hv_history.append(hv_history[-1] if hv_history else 0.0)
            continue
        
        torch.manual_seed(seed * 1000 + i)
        weights = torch.distributions.Dirichlet(torch.ones(problem.n_obj)).sample()
        
        # BoTorch's get_chebyshev_scalarization expects maximization-convention Y
        # Our model outputs -Y (negated minimization), so the model IS in max convention
        # The scalarization needs the observed Y in max convention too
        chebyshev = get_chebyshev_scalarization(weights=weights, Y=-Y)
        mc_objective = GenericMCObjective(chebyshev)
        
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        acq = qNoisyExpectedImprovement(
            model=model,
            X_baseline=X,
            sampler=sampler,
            objective=mc_objective,
            prune_baseline=True,
        )
        
        try:
            candidates, _ = optimize_acqf(
                acq_function=acq,
                bounds=problem.bounds,
                q=1,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
            )
        except Exception as e:
            print(f"  [NParEGO] iter {i}: acqf optim failed: {e}")
            hv_history.append(hv_history[-1] if hv_history else 0.0)
            continue
        
        new_Y = problem.evaluate(candidates)
        X = torch.cat([X, candidates])
        Y = torch.cat([Y, new_Y])
        
        hv = problem._compute_hv(Y)
        hv_history.append(hv)
        
        if (i + 1) % 10 == 0:
            print(f"  [NParEGO] iter {i+1}/{n_iter}: HV={hv:.4f}")
    
    return X, Y, hv_history


def run_qehvi(problem: ZDTProblem, X_init, Y_init, n_iter: int, seed: int):
    """qEHVI: Gold standard multi-objective BO."""
    X, Y = X_init.clone(), Y_init.clone()
    hv_history = []
    
    # Reference point for EHVI (in maximization space = negated)
    ref_point_max = (-problem.ref_point_hv).tolist()
    
    for i in range(n_iter):
        try:
            model = fit_model(X, Y, problem.bounds)
        except Exception as e:
            print(f"  [qEHVI] iter {i}: GP fit failed: {e}")
            hv_history.append(hv_history[-1] if hv_history else 0.0)
            continue
        
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        
        try:
            acq = qNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=ref_point_max,
                X_baseline=X,
                sampler=sampler,
                prune_baseline=True,
            )
            
            candidates, _ = optimize_acqf(
                acq_function=acq,
                bounds=problem.bounds,
                q=1,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
            )
        except Exception as e:
            print(f"  [qEHVI] iter {i}: acqf failed: {e}")
            hv_history.append(hv_history[-1] if hv_history else 0.0)
            continue
        
        new_Y = problem.evaluate(candidates)
        X = torch.cat([X, candidates])
        Y = torch.cat([Y, new_Y])
        
        hv = problem._compute_hv(Y)
        hv_history.append(hv)
        
        if (i + 1) % 10 == 0:
            print(f"  [qEHVI] iter {i+1}/{n_iter}: HV={hv:.4f}")
    
    return X, Y, hv_history


def run_nsga2(problem: ZDTProblem, n_evals: int, seed: int):
    """NSGA-II baseline with pymoo, limited to n_evals function evaluations."""
    from pymoo.termination.default import DefaultSingleObjectiveTermination
    from pymoo.core.termination import Termination
    
    algo = NSGA2(pop_size=n_evals, sampling=FloatRandomSampling())
    
    res = pymoo_minimize(
        problem.pymoo_problem,
        algo,
        ("n_gen", 1),  # Just 1 generation with pop_size = n_evals
        seed=seed,
        verbose=False,
    )
    
    Y = torch.tensor(res.F, dtype=torch.double)
    hv = problem._compute_hv(Y)
    
    return Y, hv


# ─── Main Benchmark ──────────────────────────────────────────────────────────

NUM_RESTARTS = 5
RAW_SAMPLES = 128
MC_SAMPLES = 64


def run_benchmark(problem_name: str, n_var: int = 30, n_init: int = 10, 
                  n_iter: int = 40, seeds: list = [42, 123, 456]):
    """Run full benchmark for one ZDT problem."""
    problem = ZDTProblem(problem_name, n_var=n_var)
    n_total = n_init + n_iter
    
    print(f"\n{'='*70}")
    print(f"  Benchmark: {problem_name.upper()} ({n_var} vars, {n_init}+{n_iter}={n_total} evals)")
    if problem.true_hv is not None:
        print(f"  True Pareto HV: {problem.true_hv:.4f}")
    print(f"{'='*70}")
    
    methods = ["STCH-NParEGO", "Vanilla-NParEGO", "qEHVI", "NSGA-II"]
    all_results = {m: {"hv_final": [], "hv_history": [], "n_pareto": []} for m in methods}
    all_pareto_fronts = {m: [] for m in methods}
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        X_init, Y_init = generate_initial_data(problem, n_init, seed)
        init_hv = problem._compute_hv(Y_init)
        print(f"  Initial HV: {init_hv:.4f}")
        
        # 1. STCH-NParEGO
        print(f"\n  Running STCH-NParEGO...")
        t0 = time.time()
        X_stch, Y_stch, hv_stch = run_stch_nparego(problem, X_init, Y_init, n_iter, seed)
        t_stch = time.time() - t0
        pareto_mask = is_non_dominated(-Y_stch)
        all_results["STCH-NParEGO"]["hv_final"].append(hv_stch[-1] if hv_stch else init_hv)
        all_results["STCH-NParEGO"]["hv_history"].append([init_hv] + hv_stch)
        all_results["STCH-NParEGO"]["n_pareto"].append(pareto_mask.sum().item())
        all_pareto_fronts["STCH-NParEGO"].append(Y_stch[pareto_mask].tolist())
        print(f"  STCH-NParEGO done in {t_stch:.1f}s, final HV={hv_stch[-1] if hv_stch else init_hv:.4f}")
        
        # 2. Vanilla NParEGO
        print(f"\n  Running Vanilla NParEGO...")
        t0 = time.time()
        X_npar, Y_npar, hv_npar = run_vanilla_nparego(problem, X_init, Y_init, n_iter, seed)
        t_npar = time.time() - t0
        pareto_mask = is_non_dominated(-Y_npar)
        all_results["Vanilla-NParEGO"]["hv_final"].append(hv_npar[-1] if hv_npar else init_hv)
        all_results["Vanilla-NParEGO"]["hv_history"].append([init_hv] + hv_npar)
        all_results["Vanilla-NParEGO"]["n_pareto"].append(pareto_mask.sum().item())
        all_pareto_fronts["Vanilla-NParEGO"].append(Y_npar[pareto_mask].tolist())
        print(f"  Vanilla NParEGO done in {t_npar:.1f}s, final HV={hv_npar[-1] if hv_npar else init_hv:.4f}")
        
        # 3. qEHVI
        print(f"\n  Running qEHVI...")
        t0 = time.time()
        X_ehvi, Y_ehvi, hv_ehvi = run_qehvi(problem, X_init, Y_init, n_iter, seed)
        t_ehvi = time.time() - t0
        pareto_mask = is_non_dominated(-Y_ehvi)
        all_results["qEHVI"]["hv_final"].append(hv_ehvi[-1] if hv_ehvi else init_hv)
        all_results["qEHVI"]["hv_history"].append([init_hv] + hv_ehvi)
        all_results["qEHVI"]["n_pareto"].append(pareto_mask.sum().item())
        all_pareto_fronts["qEHVI"].append(Y_ehvi[pareto_mask].tolist())
        print(f"  qEHVI done in {t_ehvi:.1f}s, final HV={hv_ehvi[-1] if hv_ehvi else init_hv:.4f}")
        
        # 4. NSGA-II
        print(f"\n  Running NSGA-II...")
        Y_nsga, hv_nsga = run_nsga2(problem, n_total, seed)
        pareto_mask = is_non_dominated(-Y_nsga)
        all_results["NSGA-II"]["hv_final"].append(hv_nsga)
        all_results["NSGA-II"]["hv_history"].append([hv_nsga])  # single value
        all_results["NSGA-II"]["n_pareto"].append(pareto_mask.sum().item())
        all_pareto_fronts["NSGA-II"].append(Y_nsga[pareto_mask].tolist())
        print(f"  NSGA-II HV={hv_nsga:.4f}, {pareto_mask.sum().item()} Pareto points")
    
    # ─── Summary Table ───
    print(f"\n{'='*70}")
    print(f"  RESULTS: {problem_name.upper()}")
    print(f"{'='*70}")
    print(f"  {'Method':<20} {'HV (mean±std)':<22} {'# Pareto (mean±std)':<22}")
    print(f"  {'-'*60}")
    
    summary = {}
    for m in methods:
        hvs = np.array(all_results[m]["hv_final"])
        nps = np.array(all_results[m]["n_pareto"])
        print(f"  {m:<20} {hvs.mean():.4f} ± {hvs.std():.4f}     {nps.mean():.1f} ± {nps.std():.1f}")
        summary[m] = {
            "hv_mean": float(hvs.mean()),
            "hv_std": float(hvs.std()),
            "n_pareto_mean": float(nps.mean()),
            "n_pareto_std": float(nps.std()),
        }
    
    if problem.true_hv:
        print(f"\n  True Pareto HV: {problem.true_hv:.4f}")
    
    # ─── Save Results ───
    results_data = {
        "problem": problem_name,
        "n_var": n_var,
        "n_init": n_init,
        "n_iter": n_iter,
        "seeds": seeds,
        "true_hv": problem.true_hv,
        "summary": summary,
        "hv_histories": {m: all_results[m]["hv_history"] for m in methods},
    }
    
    json_path = RESULTS_DIR / f"{problem_name}_results.json"
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\n  Results saved to {json_path}")
    
    # ─── HV Convergence Plot ───
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    colors = {"STCH-NParEGO": "#e63946", "Vanilla-NParEGO": "#457b9d", 
              "qEHVI": "#2a9d8f", "NSGA-II": "#e9c46a"}
    
    for m in ["STCH-NParEGO", "Vanilla-NParEGO", "qEHVI"]:
        histories = all_results[m]["hv_history"]
        min_len = min(len(h) for h in histories)
        arr = np.array([h[:min_len] for h in histories])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        iters = np.arange(min_len)
        ax.plot(iters, mean, label=m, color=colors[m], linewidth=2)
        ax.fill_between(iters, mean - std, mean + std, alpha=0.2, color=colors[m])
    
    # NSGA-II: horizontal line
    nsga_hvs = np.array(all_results["NSGA-II"]["hv_final"])
    ax.axhline(nsga_hvs.mean(), color=colors["NSGA-II"], linestyle="--", 
               linewidth=2, label=f"NSGA-II ({n_total} evals)")
    
    if problem.true_hv:
        ax.axhline(problem.true_hv, color="black", linestyle=":", linewidth=1, 
                    label="True Pareto HV", alpha=0.5)
    
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Hypervolume", fontsize=12)
    ax.set_title(f"{problem_name.upper()} — HV Convergence", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = RESULTS_DIR / f"{problem_name}_hv_convergence.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  HV convergence plot saved to {plot_path}")
    
    # ─── Pareto Front Plot ───
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # True Pareto front
    if problem.true_pf is not None:
        ax.plot(problem.true_pf[:, 0].numpy(), problem.true_pf[:, 1].numpy(),
                "k-", linewidth=1.5, label="True Pareto Front", alpha=0.5, zorder=1)
    
    markers = {"STCH-NParEGO": "o", "Vanilla-NParEGO": "s", "qEHVI": "^", "NSGA-II": "D"}
    
    # Plot best seed (seed 0) Pareto fronts
    for m in methods:
        if all_pareto_fronts[m]:
            pf = np.array(all_pareto_fronts[m][0])
            if len(pf) > 0:
                ax.scatter(pf[:, 0], pf[:, 1], c=colors[m], marker=markers[m],
                          s=40, label=m, alpha=0.7, edgecolors="white", linewidths=0.5, zorder=2)
    
    ax.set_xlabel("$f_1$", fontsize=12)
    ax.set_ylabel("$f_2$", fontsize=12)
    ax.set_title(f"{problem_name.upper()} — Pareto Fronts (seed={seeds[0]})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = RESULTS_DIR / f"{problem_name}_pareto_front.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Pareto front plot saved to {plot_path}")
    
    return results_data


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("ZDT Benchmark Suite for STCH-BoTorch")
    print("=" * 70)
    
    problems = ["zdt1", "zdt2", "zdt3"]
    all_data = {}
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-var", type=int, default=6)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--problems", nargs="+", default=["zdt1", "zdt2", "zdt3"])
    args = parser.parse_args()
    
    problems = args.problems
    for pname in problems:
        data = run_benchmark(pname, n_var=args.n_var, n_init=args.n_init, 
                            n_iter=args.n_iter, seeds=args.seeds)
        all_data[pname] = data
    
    # Final combined summary
    print(f"\n\n{'='*70}")
    print("  COMBINED SUMMARY")
    print(f"{'='*70}")
    for pname in problems:
        print(f"\n  {pname.upper()}:")
        for m, s in all_data[pname]["summary"].items():
            print(f"    {m:<20} HV={s['hv_mean']:.4f}±{s['hv_std']:.4f}")
    
    print("\nDone! All results in:", RESULTS_DIR)
