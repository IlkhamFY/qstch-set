# Molecular BO Experiment Plan
**Created:** 2026-02-22 (while Ilkham sleeps)
**Goal:** Run qSTCH-Set on real molecules, show Rodrigo it works on drug-relevant properties

---

## The Dataset: redox_mer (BTZ molecules)

From `wiseodd/lapeft-bayesopt/examples/data/redox_mer.csv`:
- **1407 benzothiazine (BTZ) molecules** (SMILES + 4 DFT properties)
- **Properties:**
  - `Ered` — redox potential (eV) — relevant for battery/redox-flow applications
  - `HOMO` — HOMO energy (eV) — stability proxy
  - `Gsol` — solvation free energy (kcal/mol) — solubility proxy
  - `Absorption Wavelength` — nm — photophysics

**This is a discrete BO problem** — we have a fixed candidate pool of 1407 molecules, not a continuous space. qSTCH-Set must select K molecules per iteration from the pool.

---

## The Experiment

### Setup
- **m = 4 objectives**: simultaneously optimize Ered, HOMO, Gsol, Absorption Wavelength
- **K = 4** (K=m rule)
- **n_init = 10** random molecules
- **T = 30** BO iterations → 10 + 30×4 = 130 molecules evaluated total
- **5 seeds**
- **Featurization**: Morgan fingerprints (radius=2, 2048 bits) — no LLM needed, simpler and faster

### Why Morgan fingerprints (not LLM features)?
- Faster, no GPU needed for feature extraction
- More interpretable, standard in drug discovery
- Allows fair comparison: qSTCH-Set vs qNParEGO vs qEHVI all use same features
- LLM features can be added later as ablation

### Baselines
- `qSTCH-Set` (K=4)
- `qNParEGO` (q=1, single-point)
- `qEHVI` (exact HV, tractable at m=4)
- `qNEHVI` (noisy HV)
- `Random` (random selection from pool)

### Metric
- **Hypervolume** at each iteration over the 4-objective space
- **Reference point**: slightly below minimum observed values in dataset
- Track HV trajectory + final Pareto front

---

## Architecture: Discrete BO Loop

Key difference from DTLZ2: **no `optimize_acqf`** over continuous space.
Instead: evaluate acquisition function on ALL remaining candidates, pick top K.

```python
# Discrete BO loop pseudocode
for t in range(T):
    # Fit GP on observed data
    model = fit_gp(train_x, train_y)  # train_x: fingerprints, train_y: 4 properties
    
    # Evaluate acquisition on all unobserved candidates
    with torch.no_grad():
        acq_values = acqf(candidate_pool)  # shape: (n_remaining,)
    
    # Select top K candidates
    top_k_idx = acq_values.topk(K).indices
    new_x = candidate_pool[top_k_idx]
    
    # "Evaluate" (lookup from dataset — it's pre-computed!)
    new_y = dataset_lookup(new_x)
    
    # Update training set
    train_x = cat(train_x, new_x)
    train_y = cat(train_y, new_y)
    
    # Remove selected from pool
    candidate_pool = remove(candidate_pool, top_k_idx)
```

For qSTCH-Set in discrete setting: evaluate over all K-subsets is intractable. Instead use **fantasy model + greedy K-step** or evaluate joint batch acquisition via the same MC estimator but over candidate points.

---

## Implementation Plan

### Step 1: Data preparation script
`benchmarks/molecular_benchmark.py`

```python
# Download redox_mer.csv
# Compute Morgan fingerprints for all 1407 molecules
# Normalize objectives (all to maximization)
# Set reference point = [min_ered-0.1, min_homo-0.1, min_gsol-0.1, min_abs-0.1]
```

### Step 2: Discrete BO loop
For qSTCH-Set specifically:
- Option A: **Joint batch evaluation** — for each candidate x_i, evaluate qSTCH-Set(x_i, x_i+1, ..., x_i+K-1) over random batches → slow
- Option B: **Greedy sequential** — optimize each of K candidates one at a time conditioned on previous (like qNParEGO but with STCH scalarization)
- Option C: **Thompson Sampling** — use qSTCHSetTS variant, sample K weight vectors, pick argmax for each → fast, O(K×n_candidates)

**Recommended: Option C (Thompson Sampling)** — fastest for discrete setting, already implemented as `qSTCHSetTS`.

### Step 3: Surrogate model
MultiTask GP with Matern-5/2 on Morgan fingerprints:
```python
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
# One GP per objective, shared input
```

### Step 4: Results
- HV trajectory plot per method
- Final Pareto front scatter (2D projections of 4D front)
- Table: final HV mean ± std

---

## The Colab Connection

The Colab notebook (https://colab.research.google.com/drive/1WdwO7Fj0QSfWnqM8hdu8YT7YLNNM5c6p) is Rodrigo's reference implementation — it shows the LLM feature extraction + Laplace BO loop for single-objective optimization of `Ered`.

**What we need to add on top:**
1. Multi-objective targets (all 4 properties simultaneously)
2. qSTCH-Set acquisition instead of Thompson sampling over single objective
3. HV metric instead of best observed value

**We do NOT need the LLM features** — Morgan fingerprints are standard, faster, and allow clean comparison. The LLM features are orthogonal to the acquisition function comparison.

---

## Where to Run

**Option A: Nibi (recommended)**
- 1 GPU job, ~2-4h per seed, 5 seeds → ~1 day
- Same setup as existing benchmarks
- SLURM script needed

**Option B: Colab**
- Free T4 GPU, fine for 1-2 seeds
- Good for prototyping/showing Rodrigo
- Can share notebook directly

**Recommend: prototype on Colab first (1 seed, fast check), then submit full 5-seed run to Nibi.**

---

## What to Tell Rodrigo

> "Running qSTCH-Set on the redox_mer dataset (1407 BTZ molecules, 4 DFT properties simultaneously: Ered, HOMO, Gsol, Absorption). Using Morgan fingerprints as features, GP surrogate, discrete candidate selection. Comparing against qNParEGO, qEHVI, qNEHVI. This is a real multi-objective molecular design problem — we're treating all 4 properties as simultaneous objectives and optimizing the Pareto front. Results in ~1 day."

---

## Files to Create

1. `benchmarks/molecular_benchmark.py` — main script
2. `benchmarks/data/redox_mer.csv` — download from lapeft-bayesopt
3. `slurm/job_k_molecular.sh` — Nibi job script
4. `examples/molecular_bo_demo.ipynb` — Colab-compatible notebook for Rodrigo

---

## Key Technical Note: qSTCH-Set in Discrete Space

Our current implementation uses `optimize_acqf` which optimizes over a continuous box. For discrete candidate pools, we need `optimize_acqf_discrete` from BoTorch:

```python
from botorch.optim import optimize_acqf_discrete

candidates, values = optimize_acqf_discrete(
    acq_function=acqf,
    q=K,
    choices=candidate_pool,  # shape (n_candidates, d)
    max_batch_size=256,
)
```

This evaluates the joint acquisition over all q-subsets (expensive for large K) OR uses a greedy approach. For K=4 and n_candidates~1400, greedy is tractable.

---

## Priority Order for When Ilkham Wakes Up

1. ✅ **Plan is here** (this file)
2. **Create `molecular_benchmark.py`** — implement the discrete BO loop
3. **Test locally** with 1 seed, 5 iterations to verify it runs
4. **Submit to Nibi** with 5 seeds, 30 iterations
5. **Create Colab notebook** for Rodrigo to see and run
6. **Update paper** — add molecular results as Section 5.6

---

*Estimated time to implement: 2-3h coding + 4-8h Nibi run = results by tomorrow evening.*
