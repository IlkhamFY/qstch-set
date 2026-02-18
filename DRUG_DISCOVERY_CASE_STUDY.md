# Drug Discovery Case Study: Multi-Objective Lead Optimization with qSTCHSet

> *A realistic application of STCH-Set-BO for simultaneous ADMET optimization of diverse lead candidates*

## 1. Motivation

In drug discovery, the transition from hit identification to lead optimization requires simultaneously satisfying 5–10 competing ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) criteria. A compound with excellent target affinity but poor hERG safety or metabolic stability is worthless. The standard workflow—sequential optimization of individual properties by medicinal chemists—leads to "molecular obesity" and whack-a-mole failures where fixing one liability introduces another (Waring et al., *Nat. Rev. Drug Discov.* 2015).

**Why current methods fall short:**

| Method | Limitation for this problem |
|--------|---------------------------|
| qEHVI (BoTorch) | Hypervolume scales exponentially with m; infeasible for m ≥ 6 |
| qNParEGO | Random scalarization weights → uncoordinated candidates; no set-level diversity |
| GFlowNets / REINVENT | Sample-inefficient; require 10K+ oracle calls (PMO benchmark, Gao et al. 2022) |
| Weighted-sum scalarization | Misses non-convex Pareto regions; requires arbitrary weight choices |

**Why qSTCHSet is the right fit:**
1. **Many objectives (m=7):** O(Km) scaling, no hypervolume computation
2. **Coordinated diverse leads (K=5):** The set-level min-max formulation *inherently* produces candidates that divide-and-conquer the objective space
3. **Expensive evaluations:** BO sample efficiency matters—each ADMET assay costs $500–5,000 and takes days; we budget ~200 evaluations total
4. **Differentiable:** Smooth log-sum-exp enables gradient-based optimization of candidate batches via L-BFGS-B

## 2. Objective Definitions (m = 7)

We define 7 objectives capturing the key axes of a realistic lead optimization campaign. All objectives are formulated for **minimization** (lower = better).

| # | Objective | Type | Oracle / Model | Target | Source |
|---|-----------|------|----------------|--------|--------|
| 1 | **QED** (Drug-likeness) | Maximize → min(-QED) | RDKit `Chem.QED.qed()` | > 0.6 | Bickerton et al., *Nat. Chem.* 2012 |
| 2 | **SA Score** (Synthetic Accessibility) | Minimize | RDKit `sascorer` | < 4.0 | Ertl & Schuffenhauer, *JCIM* 2009 |
| 3 | **LogP** (Lipophilicity) | Target 2.5 → min |LogP - 2.5| | Crippen LogP (RDKit) | 1.0–3.5 | Lipinski et al., *Adv. Drug Deliv. Rev.* 2001 |
| 4 | **hERG Inhibition** (Cardiac Toxicity) | Minimize prob(active) | TDC `herg_central` classifier (648 cpds, AUROC) | prob < 0.1 | TDC (Huang et al., *NeurIPS* 2021) |
| 5 | **CYP3A4 Inhibition** (Metabolic Liability) | Minimize prob(inhibitor) | TDC `cyp3a4_veith` classifier (12,328 cpds) | prob < 0.2 | Veith et al., *Nat. Biotechnol.* 2009; TDC |
| 6 | **Caco-2 Permeability** (Absorption) | Maximize → min(-perm) | TDC `caco2_wang` regressor (906 cpds, MAE) | > -5.15 log cm/s | Wang et al., *J. Med. Chem.* 2016; TDC |
| 7 | **Aqueous Solubility** (LogS) | Maximize → min(-LogS) | ESOL (Delaney) via RDKit | > -4.0 | Delaney, *JCIM* 2004 |

**Conflict structure** (why this needs Pareto optimization, not scalarization):
- QED ↔ SA: Complex drug-like scaffolds are harder to synthesize
- LogP ↔ Solubility: Lipophilic molecules have poor aqueous solubility
- Permeability ↔ Solubility: Permeable molecules tend to be lipophilic → insoluble
- hERG ↔ LogP: hERG liability correlates with lipophilicity (Aronov, *Drug Discov. Today* 2005)
- CYP3A4 ↔ QED: Drug-like, aromatic molecules often inhibit CYP enzymes

## 3. Design Space & Oracle Specification

### Chemical Space

We use a **pool-based** approach over a curated library of ~10,000 drug-like molecules from ZINC250K (Irwin et al., 2012), filtered to:
- MW ∈ [200, 500], RO5-compliant
- Represented as **Morgan fingerprint (r=2, 2048-bit)** → continuous via PCA to d=50 dimensions

This is more realistic than de novo generation for lead optimization, where chemists explore SAR around known scaffolds.

### Oracle Construction

Each "evaluation" of a candidate molecule computes all 7 objectives:

```python
def oracle(smiles: str) -> torch.Tensor:
    """Returns 7-objective vector (minimization convention)."""
    mol = Chem.MolFromSmiles(smiles)
    return torch.tensor([
        -qed(mol),                              # 1. QED (negate for min)
        sascorer.calculateScore(mol),           # 2. SA Score
        abs(Crippen.MolLogP(mol) - 2.5),        # 3. LogP deviation
        herg_model.predict_proba(fp(mol)),      # 4. hERG prob
        cyp3a4_model.predict_proba(fp(mol)),    # 5. CYP3A4 prob
        -caco2_model.predict(fp(mol)),          # 6. Caco-2 (negate)
        -esol_logS(mol),                        # 7. Solubility (negate)
    ])
```

**Oracle cost model:** We treat each evaluation as "expensive" (budget N=200). In practice, objectives 4–6 use ML surrogates of wet-lab assays; the BO loop adds a second layer of GP surrogate on top—a common setup in real pharma pipelines where even running a docking calculation takes minutes.

### ADMET Surrogate Models

For objectives 4–6, we train gradient-boosted classifiers/regressors on the TDC datasets:

| Endpoint | TDC Dataset | Training Size | Model | Expected AUROC/MAE |
|----------|------------|---------------|-------|--------------------|
| hERG | `herg_central` | 648 | LightGBM on ECFP4 | ~0.83 AUROC |
| CYP3A4 | `cyp3a4_veith` | 12,328 | LightGBM on ECFP4 | ~0.88 AUROC |
| Caco-2 | `caco2_wang` | 906 | RF on ECFP4 | ~0.32 MAE |

These surrogates serve as our "ground truth" oracles—the BO campaign treats them as black boxes.

## 4. Experimental Protocol

### 4.1 Baselines

| Method | Implementation | Notes |
|--------|---------------|-------|
| **qSTCHSet (ours)** | `stch_botorch.qSTCHSet` | K=5, μ=0.1, q=5 |
| **qNParEGO** | BoTorch `qNParEGO` | K=5 via random Chebyshev scalarizations |
| **qEHVI** | BoTorch `qExpectedHypervolumeImprovement` | K=5, if tractable at m=7 (likely fails) |
| **DGEMO** | DGEMO (Lukovic et al., 2020) | Diversity-guided evolutionary MO |
| **Random** | Uniform from pool | K=5 random selection each round |
| **Thompson Sampling** | Independent per-objective TS | No coordination between K candidates |

### 4.2 Protocol

```
For each method, repeat 10 seeds:
  1. Initialize: 20 random molecules from pool, evaluate all 7 objectives
  2. Fit: Independent GP per objective (Matérn 5/2 kernel, d=50 PCA space)
  3. Acquire: Select K=5 candidates (batch q=5) by optimizing acquisition
  4. Evaluate: Query oracle for 5 new molecules
  5. Repeat steps 2-4 for 36 rounds (20 + 36×5 = 200 total evaluations)
  6. Report: Final Pareto front quality, diversity of recommended set
```

### 4.3 Metrics

| Metric | What it measures |
|--------|-----------------|
| **Hypervolume** of final Pareto front | Overall quality (gold standard for MO) |
| **Set Coverage (C-metric)** | Does method A's Pareto front dominate B's? |
| **Diversity** of top-K set | Average pairwise Tanimoto distance between recommended K=5 molecules |
| **Objective Coverage** | For each of 7 objectives, is at least one candidate in the recommended set within the "acceptable" threshold? |
| **Sample Efficiency** | HV as function of oracle calls (AUC-HV curve) |

The **Objective Coverage** metric is novel and directly measures the set-level property that qSTCHSet optimizes: a good set of K=5 leads should have *every* objective well-covered by at least one member.

## 5. Expected Results Narrative

### 5.1 qEHVI Cannot Scale

At m=7, hypervolume computation is exponential. We expect qEHVI to either fail outright or require prohibitive compute per acquisition step (>1 hour vs. seconds for qSTCHSet). This immediately demonstrates the practical advantage.

### 5.2 qSTCHSet Produces Coordinated Diverse Leads

The min-max structure of STCH-Set naturally produces "specialist" molecules:

| Candidate | Specialty | Trade-off accepted |
|-----------|----------|-------------------|
| Lead 1 | Best QED + SA | Moderate hERG |
| Lead 2 | Lowest hERG + CYP | Lower permeability |
| Lead 3 | Best solubility + permeability | Higher SA score |
| Lead 4 | Balanced across all 7 | No single best |
| Lead 5 | Best LogP + low CYP | Moderate QED |

qNParEGO with random weights will produce less coordinated sets—some candidates will redundantly optimize the same trade-off while leaving objective gaps.

### 5.3 Quantitative Predictions

| Metric | qSTCHSet | qNParEGO | Random |
|--------|----------|----------|--------|
| Final HV (normalized) | **0.72 ± 0.04** | 0.61 ± 0.06 | 0.35 ± 0.08 |
| Objective Coverage (7/7) | **85%** of seeds | 55% of seeds | 10% |
| Diversity (Tanimoto) | **0.68 ± 0.05** | 0.52 ± 0.08 | 0.71 ± 0.03 |
| Time per acquisition | **2.3s** | 1.8s | 0.01s |

Key insight: Random achieves high diversity but terrible coverage. qNParEGO achieves decent HV but poor coordination. qSTCHSet uniquely achieves both high HV *and* high objective coverage through its set-level optimization.

### 5.4 Ablations

1. **K sensitivity (K=1,3,5,7,10):** Show diminishing returns after K=5; at K=1 reduces to single-point STCH
2. **μ sensitivity (0.01, 0.1, 1.0, 10.0):** Low μ → hard min-max (sharper specialists); high μ → soft averaging (more balanced)
3. **m scaling (3,5,7,10 objectives):** Add AMES toxicity, BBB penetration, half-life as extra objectives; show graceful O(Km) scaling

## 6. Code Sketch

```python
"""
Drug discovery lead optimization with qSTCHSet.
Simultaneously optimize K=5 diverse leads across 7 ADMET objectives.
"""
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from stch_botorch import qSTCHSet

# ── Oracle Setup ──────────────────────────────────────────────
from drug_oracles import DrugOracle  # wrapper around RDKit + TDC surrogates
oracle = DrugOracle(
    objectives=["qed", "sa", "logp_dev", "herg", "cyp3a4", "caco2", "solubility"],
    pool_smiles="data/zinc250k_filtered.smi",  # 10K drug-like pool
    representation="morgan_pca50",              # Morgan FP → PCA(50)
)

# ── Initialization ────────────────────────────────────────────
N_INIT = 20
N_ROUNDS = 36
K = 5  # candidates per round = leads we want
M = 7  # objectives

train_X = oracle.random_sample(N_INIT)          # (20, 50)
train_Y = oracle.evaluate(train_X)              # (20, 7) — minimization

# ── BO Loop ───────────────────────────────────────────────────
for round_idx in range(N_ROUNDS):
    # Fit independent GPs (one per objective)
    models = []
    for j in range(M):
        gp = SingleTaskGP(train_X, train_Y[:, j:j+1])
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        models.append(gp)
    
    # Combine into multi-output model
    from botorch.models import ModelListGP
    model = ModelListGP(*models)
    
    # Compute reference point from observed data
    ref_point = train_Y.min(dim=0).values - 0.1  # slightly below best observed
    
    # qSTCHSet acquisition: jointly optimize K=5 candidates
    acq = qSTCHSet(
        model=model,
        ref_point=ref_point,
        mu=0.1,                    # smoothing temperature
        num_samples=128,           # MC posterior samples
        K=K,                       # set size for STCH-Set
    )
    
    # Optimize — returns K candidates as a single batch
    candidates, acq_value = optimize_acqf(
        acq_function=acq,
        bounds=oracle.bounds,      # (2, 50) PCA bounds
        q=K,                       # batch size = K leads
        num_restarts=10,
        raw_samples=512,
    )
    
    # Evaluate new candidates
    new_Y = oracle.evaluate(candidates)            # (K, 7)
    train_X = torch.cat([train_X, candidates])
    train_Y = torch.cat([train_Y, new_Y])
    
    # Log progress
    pareto_Y = compute_pareto_front(train_Y)
    hv = compute_hypervolume(pareto_Y, ref_point)
    obj_coverage = objective_coverage(candidates, thresholds={
        "qed": -0.6, "sa": 4.0, "logp_dev": 1.0,
        "herg": 0.1, "cyp3a4": 0.2, "caco2": 5.15, "solubility": 4.0,
    })
    print(f"Round {round_idx+1}: HV={hv:.4f}, "
          f"Coverage={obj_coverage}/7, "
          f"Diversity={pairwise_tanimoto(candidates):.3f}")

# ── Final Recommendation ──────────────────────────────────────
# Select best K=5 set from all evaluated molecules
from stch_botorch.scalarization import smooth_chebyshev_set

best_set_idx = select_best_set(
    Y=train_Y, K=5, mu=0.1,
    scalarization=smooth_chebyshev_set,
)
recommended_smiles = oracle.idx_to_smiles(best_set_idx)

print("\n🧪 Recommended Lead Set:")
for i, smi in enumerate(recommended_smiles):
    props = oracle.property_profile(smi)
    print(f"  Lead {i+1}: {smi}")
    print(f"    QED={props['qed']:.2f}  SA={props['sa']:.1f}  "
          f"LogP={props['logp']:.1f}  hERG={props['herg']:.2f}  "
          f"CYP3A4={props['cyp3a4']:.2f}  Caco2={props['caco2']:.2f}  "
          f"LogS={props['solubility']:.1f}")
```

## 7. Why This Matters for the Paper

This case study addresses three likely reviewer objections:

1. **"Where's the real application?"** — Lead optimization with 7 ADMET objectives is an actual problem pharma teams face daily. The TDC endpoints are standard benchmarks.

2. **"Why not just use qEHVI?"** — It literally cannot run at m=7. The exponential hypervolume computation is the bottleneck that motivates alternatives like STCH-Set.

3. **"Why optimize a *set* of K solutions?"** — In drug discovery, you never advance a single molecule. You always advance a *portfolio* of 3–5 diverse leads to hedge against attrition in later stages (toxicology, formulation, clinical trials). The K=5 set formulation directly matches industry practice.

### Connection to Paper Narrative

The STCH-Set-BO paper claims three advantages: (a) graceful scaling to many objectives, (b) coordinated set-level optimization, and (c) full differentiability. This case study exercises all three simultaneously in a compelling real-world setting that synthetic benchmarks cannot capture.

## References

- Bickerton et al. "Quantifying the chemical beauty of drugs." *Nat. Chem.* 4, 90–98 (2012).
- Ertl & Schuffenhauer. "Estimation of synthetic accessibility score." *JCIM* 49, 1669–1676 (2009).
- Lipinski et al. "Experimental and computational approaches." *Adv. Drug Deliv. Rev.* 46, 3–26 (2001).
- Huang et al. "Therapeutics Data Commons." *NeurIPS Datasets & Benchmarks* (2021).
- Gao et al. "Sample efficiency matters: A benchmark for practical molecular optimization." *NeurIPS* (2022).
- Brown et al. "GuacaMol: Benchmarking models for de novo molecular design." *JCIM* 59, 1096–1108 (2019).
- Lin et al. "Smooth Tchebycheff Scalarization for Multi-Objective Optimization." *ICML* (2024).
- Lin et al. "STCH-Set: Set Scalarization for Many-Objective Optimization." *ICLR* (2025).
- Waring et al. "An analysis of the attrition of drug candidates." *Nat. Rev. Drug Discov.* 14, 475–486 (2015).
- Aronov. "Predictive in silico modeling for hERG channel blockers." *Drug Discov. Today* 10, 149–155 (2005).
- Delaney. "ESOL: Estimating aqueous solubility directly from molecular structure." *JCIM* 44, 1000–1005 (2004).
