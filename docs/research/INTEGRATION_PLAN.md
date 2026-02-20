# STCH-BoTorch × RxnFlow Integration Plan

## Architecture Overview

```
RxnFlow (GFlowNet)          stch-botorch (MOBO)
┌─────────────────┐         ┌──────────────────────┐
│ Synthesis-aware  │         │ SmoothChebyshevObj    │
│ molecular gen    │────────▶│ (multi-obj reward)    │
│                  │         ├──────────────────────┤
│ Action space:    │         │ STCHCandidateGenerator│
│ - building blocks│◀────────│ (diverse candidates)  │
│ - rxn templates  │         ├──────────────────────┤
│                  │         │ qPMHI                 │
│ Output: diverse  │────────▶│ (batch selection)     │
│ synthesizable    │         └──────────────────────┘
│ molecules        │
└─────────────────┘
```

## Your Codebase Summary

### Core API (stch_botorch)
| Class/Function | Purpose |
|---|---|
| `smooth_chebyshev(Y, weights, ref_point, mu)` | Single-point STCH scalarization → scalar utility |
| `smooth_chebyshev_set(Y, weights, ref_point, mu)` | Batch STCH → aggregates over q candidates |
| `SmoothChebyshevObjective` | BoTorch MCObjective wrapper (per-candidate) |
| `SmoothChebyshevSetObjective` | BoTorch MCObjective wrapper (batch-aggregated) |
| `qPMHI` | Pool-based batch selection by hypervolume improvement probability |
| `STCHCandidateGenerator` | Diverse candidate pool via multi-weight STCH optimization |
| `STCHqPMHIAcquisition` / `optimize_stch_qpmhi` | Two-stage pipeline: STCH pool → qPMHI selection |

## Integration Points

### 1. STCH as GFlowNet Reward Function

**Where:** RxnFlow's `reward_fn` / proxy model output

RxnFlow generates molecules with probability ∝ reward. Currently uses single-objective rewards (e.g., docking score). We replace with STCH multi-objective scalarization:

```python
# Adapter: multi-objective reward for RxnFlow
import torch
from stch_botorch.scalarization import smooth_chebyshev

class STCHReward:
    """Multi-objective reward using STCH scalarization."""
    
    def __init__(self, objectives: list, weights: torch.Tensor, 
                 ref_point: torch.Tensor, mu: float = 0.1):
        self.objectives = objectives  # [qed_fn, sa_fn, docking_fn, ...]
        self.weights = weights
        self.ref_point = ref_point
        self.mu = mu
    
    def __call__(self, mol_batch) -> torch.Tensor:
        # Evaluate each objective → (batch, m)
        Y = torch.stack([obj(mol_batch) for obj in self.objectives], dim=-1)
        # STCH scalarization → (batch,)
        return smooth_chebyshev(Y, self.weights, self.ref_point, self.mu)
```

**Why STCH > weighted sum here:** Tchebycheff scalarization can recover non-convex Pareto points. With smoothing, gradients flow through the reward → GFlowNet learns properly.

**Sampling diversity:** Randomize weights each training batch (like NParEGO) to cover the full Pareto front:
```python
# Per-batch random weight sampling
weights = torch.distributions.Dirichlet(torch.ones(m)).sample()
reward = smooth_chebyshev(Y, weights, ref_point, mu)
```

### 2. STCH-Set for GFlowNet Batch Training

**Where:** GFlowNet trajectory balance loss / batch reward

STCH-Set optimizes a batch of q candidates to collectively cover all objectives. This is directly useful for GFlowNet batch training:

```python
from stch_botorch.scalarization import smooth_chebyshev_set

class STCHSetBatchReward:
    """Reward a batch of molecules for collective objective coverage."""
    
    def __call__(self, mol_batch) -> torch.Tensor:
        # mol_batch: q molecules from one GFlowNet rollout
        Y = self.evaluate_objectives(mol_batch)  # (q, m)
        # Single scalar reward for the entire batch
        return smooth_chebyshev_set(Y.unsqueeze(0), self.weights, 
                                    self.ref_point, self.mu).squeeze()
```

**Use case:** Train GFlowNet to produce *batches* that are diverse across objectives, not just individual high-reward molecules.

### 3. Two-Stage Pipeline: GFlowNet → STCH-qPMHI Selection

**Where:** After GFlowNet generates a candidate pool

This is the highest-leverage integration — use GFlowNet as a *generator* and STCH-qPMHI as a *selector*:

```
Step 1: RxnFlow generates N diverse synthesizable molecules (N=1000+)
Step 2: Surrogate model predicts objectives for all N
Step 3: STCHCandidateGenerator scores candidates (already done by GFlowNet)
Step 4: qPMHI selects optimal batch of q molecules for synthesis/testing
Step 5: Lab results → update surrogate → repeat (DMTA loop)
```

```python
from stch_botorch.integration import optimize_stch_qpmhi

# After GFlowNet generates pool
gfn_molecules = rxnflow.sample(n=1000)  # Synthesizable molecules
X_pool = featurize(gfn_molecules)       # Molecular fingerprints/descriptors

# Fit surrogate on existing data
model = fit_surrogate(train_X, train_Y)  # GP or neural proxy

# Select batch using qPMHI
result = optimize_stch_qpmhi(
    model=model,
    bounds=bounds,           # Not needed for pool-based
    ref_point=ref_point,
    batch_size=10,           # Molecules to synthesize
    pool_X=X_pool,           # GFlowNet-generated pool
)
selected_indices = result["selected_indices"]
to_synthesize = [gfn_molecules[i] for i in selected_indices]
```

### 4. Multi-Fidelity Reward Cascade

Combine cheap and expensive oracles:

```
Fidelity 0: RDKit descriptors (QED, SA, LogP)     → free, instant
Fidelity 1: ML proxy (ADMET, binding affinity)     → cheap, fast  
Fidelity 2: Docking (AutoDock/DiffDock)             → moderate
Fidelity 3: FEP/MD simulation                       → expensive, slow
```

STCH handles this naturally — use cheap fidelities during GFlowNet training, expensive ones for final qPMHI selection.

## Adapter Code Needed

| Module | Purpose | Effort |
|---|---|---|
| `rxnflow_adapter.py` | Wrap STCH reward for RxnFlow's reward interface | ~100 lines |
| `mol_featurizer.py` | SMILES → fingerprint tensor for GP surrogate | ~50 lines (use datamol) |
| `surrogate.py` | GP/neural proxy for objective prediction | ~150 lines (BoTorch SingleTaskGP) |
| `dmta_loop.py` | Orchestrate generate → predict → select → update | ~200 lines |

## Recommended Sequence

1. **Clone RxnFlow** and run their demo (verify it works on our env)
2. **Build `rxnflow_adapter.py`** — STCH reward with random weight sampling
3. **Train toy GFlowNet** — 2 objectives (QED + SA) on ZINC subset
4. **Add surrogate + qPMHI selection** — pool-based batch selection
5. **Benchmark** — compare vs. single-objective GFlowNet + random selection
6. **Paper-ready** — add multi-fidelity, scale to real targets

## Key Insight

Your stch-botorch already solves the hard part (differentiable multi-objective scalarization + batch selection). RxnFlow solves another hard part (synthesis-aware generation). The gap is just the adapter layer connecting them — maybe 500 lines of Python.

This is a genuine research contribution: **no one has published GFlowNet + STCH scalarization for DMTA**.
