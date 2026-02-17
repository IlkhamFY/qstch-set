# Dr. Rodrigo Vargas — TODO for stch-botorch

## Goal: Bullet-proof the code with existing MO problems BEFORE molecules

### 1. Check Pareto front with other existing packages
- Compare stch-botorch Pareto fronts against pymoo (NSGA-II, NSGA-III)
- Verify we recover the same/better Pareto front on standard problems
- Use pymoo as ground truth: `from pymoo.problems import get_problem`

### 2. ZDT Suite — find shared minima
- ZDT1 (convex), ZDT2 (non-convex), ZDT3 (disconnected), ZDT4 (multimodal), ZDT6 (degenerate)
- Run STCH scalarization with BoTorch GP surrogate on each
- Compare against: vanilla BoTorch qEHVI, qNParEGO (weighted sum), pymoo NSGA-II
- Metrics: Hypervolume, IGD, Spacing, Pareto coverage

### 3. Try BoTorch WITHOUT stch-botorch (baseline)
- Run same ZDT suite with vanilla BoTorch (qEHVI, qNParEGO)
- This establishes the baseline that STCH must beat
- Fair comparison: same budget, same GP model, same initial points

### Reference Papers
- **Lin et al. (ICLR 2025)** arXiv:2405.19650 — "Few for Many: Tchebycheff Set Scalarization for Many-Objective Optimization" — THIS IS THE THEORY PAPER for STCH-Set
- **Nature Comp Sci 2025** s41524-025-01924-8 — "Active learning enables generation of molecules that advance the known Pareto front" — active learning + Pareto front advancement
- **J. Med. Chem. 2023** acs.jmedchem.3c01083 — bPK score: deep learning for small-molecule developability (~100 ADMET assays aggregated)
- **ScopeBO** (Sven Roediger / Doyle / Sigman) — ML tool to maximize info content of substrate scopes, balancing performance + diversity

### Target
- Highest-tier journal (Nature-level or top ML venue)
- Dense, high-quality paper
- Mathematical rigor first, molecules second
