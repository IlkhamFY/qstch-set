"""
High-level API for running Many-Objective Bayesian Optimization with STCH-Set.
"""

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

# Internal imports
from stch_botorch.acquisition.stch_set_bo import qSTCHSet


@dataclass
class MOBOResult:
    """Container for MOBO optimization results."""
    X: Tensor
    Y: Tensor
    pareto_X: Tensor
    pareto_Y: Tensor
    hypervolume_history: List[float]
    n_evals: int
    n_obj: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def run_mobo(
    f: Callable[[Union[Tensor, np.ndarray]], Union[Tensor, np.ndarray]],
    bounds: Tensor,
    n_obj: int,
    budget: int,
    batch_size: Optional[int] = None,
    n_init: Optional[int] = None,
    mu: float = 0.1,
    seed: int = 0,
    verbose: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> MOBOResult:
    """
    Run Many-Objective Bayesian Optimization using qSTCH-Set.

    Args:
        f: Black-box function. Accepts (n, d) inputs, returns (n, m) objectives.
           Can return Tensor or numpy array.
        bounds: Tensor of shape (2, d) specifying lower and upper bounds.
        n_obj: Number of objectives (m).
        budget: Total number of function evaluations allowed.
        batch_size: Number of candidates to evaluate in parallel per iteration (q).
                    Defaults to n_obj (m) if None.
        n_init: Number of initial Sobol points. Defaults to 2*(d+1) if None.
        mu: Smoothing parameter for STCH-Set scalarization.
        seed: Random seed for reproducibility.
        verbose: Whether to print progress updates.
        device: 'cuda' or 'cpu'. Auto-detected if None.

    Returns:
        MOBOResult object containing history and Pareto set.
    """
    # 1. Setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if isinstance(device, str):
            device = torch.device(device)

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Validate inputs
    bounds = bounds.to(device=device, dtype=torch.double)
    dim = bounds.shape[-1]
    
    if batch_size is None:
        batch_size = n_obj
    
    if n_init is None:
        n_init = 2 * (dim + 1)

    if verbose:
        print(f"Starting MOBO (qSTCH-Set) on {device}")
        print(f"  Objectives: {n_obj}, Dim: {dim}")
        print(f"  Budget: {budget}, Batch size: {batch_size}, Initial: {n_init}")

    # Helper to evaluate f safely
    def eval_f(x: Tensor) -> Tensor:
        # Handle numpy conversion if needed
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x_in = x
            if x.device.type != 'cpu':
                x_np = x.detach().cpu().numpy()
            else:
                x_np = x.detach().numpy()
        else:
            x_in = torch.from_numpy(x).to(device=device, dtype=torch.double)
            x_np = x
        
        # Call user function (it might expect numpy or tensor)
        # We try passing tensor first if it accepts it, otherwise numpy
        try:
            y = f(x_in)
        except (TypeError, ValueError, AttributeError):
            y = f(x_np)
        
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        
        y = y.to(dtype=bounds.dtype, device=device)
        if y.ndim == 1:
            y = y.unsqueeze(0)
        return y

    # 2. Initialization (Sobol)
    # Draw Sobol samples on CPU then move to device
    X_init = draw_sobol_samples(bounds=bounds.cpu(), n=n_init, q=1).squeeze(1).to(device=device, dtype=bounds.dtype)
    Y_init = eval_f(X_init)

    train_X = X_init
    train_Y = Y_init
    
    hv_history = []
    
    start_time = time.time()
    
    # Initial HV
    # Use adaptive reference point for calculation as requested
    if train_Y.shape[0] > 0:
        hv_ref_point = train_Y.min(dim=0).values - 0.1 * (train_Y.std(dim=0) + 1e-6)
        # For HV calculation, we assume minimization of -Y ? 
        # BoTorch HV assumes maximization. If we have minimization problem (lower better),
        # we should negate Y. 
        # Prompt: "ref_point for HV: train_Y.min ... - 0.1 * std".
        # This implies train_Y are values to be MAXIMIZED (standard BoTorch).
        # If user function returns minimization objectives, they should negate them before passing to run_mobo?
        # Or run_mobo assumes maximization?
        # Standard BoTorch assumes maximization. We stick to that.
        
        # Calculate HV on Pareto front
        try:
            pareto_mask = is_non_dominated(train_Y)
            pareto_Y_curr = train_Y[pareto_mask]
            hv_metric = Hypervolume(ref_point=hv_ref_point)
            current_hv = hv_metric.compute(pareto_Y_curr)
        except Exception as e:
            current_hv = 0.0
            if verbose:
                print(f"Warning: HV calculation failed: {e}")
        
        hv_history.append(current_hv)
        
        if verbose:
             print(f"Iter 0/{(budget - n_init) // batch_size} | n={len(train_X)} | Pareto={is_non_dominated(train_Y).sum().item()} | HV={current_hv:.4f}")

    # 3. Main BO Loop
    iteration = 1
    while len(train_X) < budget:
        
        # 3a. Fit Model
        # Standardize Y to mean 0 std 1 for better GP fitting
        model = SingleTaskGP(
            train_X, 
            train_Y, 
            outcome_transform=Standardize(m=n_obj)
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(mll)

        # 3b. Construct Acquisition Function
        # Ref point for acquisition: train_Y.min - 0.1 * std
        # Note: Standardize transform handles scaling, but qSTCHSet needs unscaled ref_point?
        # qSTCHSet uses the model which outputs Standardized values if outcome_transform is used?
        # Wait, if we use Standardize, the model posterior is in standardized space? 
        # No, BoTorch models with outcome_transform usually untransform the output in posterior if not specified otherwise,
        # BUT SingleTaskGP posterior returns transformed values if we don't handle it?
        # Actually SingleTaskGP with outcome_transform: the posterior is in the ORIGINAL scale.
        # The transform is applied internally. 
        # So we should pass ref_point in ORIGINAL scale.
        
        # Ensure we don't divide by zero if std is 0
        y_std = train_Y.std(dim=0)
        y_std[y_std == 0] = 1.0
        
        ref_point = train_Y.min(dim=0).values - 0.1 * y_std
        
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        
        acqf = qSTCHSet(
            model=model,
            ref_point=ref_point,
            mu=mu,
            sampler=sampler,
            maximize=True  # Assumes maximization
        )
        
        # 3c. Optimize Acquisition
        # Prepare warm start
        # We want `num_restarts` initial conditions.
        # We'll take top pareto points as warm starts.
        num_restarts = 20
        raw_samples = 512
        
        # Generate raw samples first to get initial conditions
        # But we want to explicitly inject warm starts.
        # We can pass `batch_initial_conditions` to optimize_acqf.
        # Usually optimize_acqf generates them from raw_samples. 
        # We can implement a custom initialization or just let optimize_acqf handle it 
        # via `initial_conditions` (deprecated?) or `batch_initial_conditions`.
        
        # For simplicity and robustness, we'll let optimize_acqf handle initialization from raw_samples,
        # but we can seed it by passing `batch_initial_conditions`.
        # Construct batch_initial_conditions manually:
        # Take best observed point (or points), replicate them, add noise? 
        # The prompt says: "Warm-start: pass previous best X_K as one of the initial conditions".
        
        # Let's find the Pareto set of current train_X
        pareto_mask = is_non_dominated(train_Y)
        pareto_X_curr = train_X[pareto_mask]
        
        batch_initial_conditions = None
        if len(pareto_X_curr) >= batch_size:
            # Pick a random subset of Pareto points to form a batch
            idx = torch.randperm(len(pareto_X_curr))[:batch_size]
            warm_X = pareto_X_curr[idx] # (q, d)
            # Make it (1, q, d)
            warm_X = warm_X.unsqueeze(0)
            
            # We need (num_restarts, q, d). 
            # We can use this warm_X for the first restart, and random for others.
            # But optimize_acqf doesn't easily support mixed initialization unless we generate all.
            # So we generate random ones and replace the first one.
            
            # Generate random restarts
            # We use a helper to generate random points from bounds
            # Or just let optimize_acqf do it.
            # For "pass previous best X_K as one of the initial conditions", 
            # we can use `gen_candidates_scipy` style manually or just trust `optimize_acqf` with `raw_samples`.
            # Standard `optimize_acqf` with `raw_samples` evaluates acqf on raw_samples and picks best ones as initial conditions.
            # If we include our warm start in `raw_samples`? No, raw_samples are random.
            
            # Let's try to pass `batch_initial_conditions` explicitly.
            # We'll generate (num_restarts, q, d) random points, and replace index 0 with warm_X.
            random_starts = draw_sobol_samples(bounds=bounds.cpu(), n=num_restarts, q=batch_size).to(device=device, dtype=bounds.dtype)
            random_starts[0] = warm_X
            batch_initial_conditions = random_starts
        
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            batch_initial_conditions=batch_initial_conditions,
            options={"batch_limit": 5, "maxiter": 200},
        )
        
        # 3d. Evaluate and Update
        new_Y = eval_f(candidates)
        
        train_X = torch.cat([train_X, candidates], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)
        
        # 3e. Logging
        # Update HV ref point
        hv_ref_point = train_Y.min(dim=0).values - 0.1 * (train_Y.std(dim=0) + 1e-6)
        
        pareto_mask = is_non_dominated(train_Y)
        try:
            hv_metric = Hypervolume(ref_point=hv_ref_point)
            current_hv = hv_metric.compute(train_Y[pareto_mask])
        except:
            current_hv = 0.0
        
        hv_history.append(current_hv)
        
        if verbose:
            n_evals = len(train_X)
            n_pareto = pareto_mask.sum().item()
            total_iters = (budget - n_init) // batch_size
            print(f"Iter {iteration}/{total_iters} | n={n_evals} | Pareto={n_pareto} | HV={current_hv:.4f}")
        
        iteration += 1

    # 4. Finalize
    is_pareto = is_non_dominated(train_Y)
    pareto_X = train_X[is_pareto]
    pareto_Y = train_Y[is_pareto]
    
    return MOBOResult(
        X=train_X,
        Y=train_Y,
        pareto_X=pareto_X,
        pareto_Y=pareto_Y,
        hypervolume_history=hv_history,
        n_evals=len(train_X),
        n_obj=n_obj,
        metadata={"device": str(device), "mu": mu, "seed": seed}
    )
