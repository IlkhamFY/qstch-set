# STCH-Set Bayesian Optimization: Experiment Strategy

*Generated 2026-02-17 | For NeurIPS 2025/2026 submission*

---

## Executive Summary

This document defines the experimental validation plan for our STCH-Set MOBO paper. The strategy is designed to meet the bar for a **strong accept** at NeurIPS: novel method + comprehensive synthetic benchmarks + at least one compelling real-world application + ablations + wall-clock analysis + statistical rigor (≥10 seeds).

---

## 1. Synthetic Benchmarks (MUST-HAVE)

### 1.1 Standard DTLZ Suite (Scalable Objectives)

| Problem | d (input) | m (objectives) | Pareto Front Shape | Why |
|---------|-----------|-----------------|--------------------|----|
| DTLZ1 | 6 | 3 | Linear hyperplane | Tests convergence (many local fronts) |
| DTLZ2 | 6 | 3, 5, 8, 10 | Concave hypersphere | Standard; scale m to show many-objective strength |
| DTLZ3 | 6 | 3 | Concave (hard) | Many local fronts; tests robustness |
| DTLZ5 | 6 | 5 | Degenerate (curve) | Tests ability to find low-dimensional Pareto manifold |
| DTLZ7 | 6 | 3 | Disconnected | Tests handling of disconnected fronts |

**BoTorch classes:** `botorch.test_functions.multi_objective.DTLZ1`, `DTLZ2`, `DTLZ3`, `DTLZ5`, `DTLZ7`

### 1.2 ZDT Suite (2-objective sanity checks)

| Problem | d | m | Notes |
|---------|---|---|-------|
| ZDT1 | 30 | 2 | Convex front |
| ZDT2 | 30 | 2 | Non-convex front |
| ZDT3 | 30 | 2 | Disconnected front |

**Purpose:** Verify no regression in bi-objective case; fast to run.

### 1.3 Many-Objective Stress Tests (KEY DIFFERENTIATOR)

| Problem | d | m | Why Critical |
|---------|---|---|-------------|
| DTLZ2 | m+4 | 5 | Standard many-objective threshold |
| DTLZ2 | m+4 | 8 | Where qNEHVI becomes expensive |
| DTLZ2 | m+4 | 10 | Few MOBO papers go here |
| DTLZ2 | m+4 | 15 | **Our headline result** — show STCH-Set scales where competitors can't |
| DTLZ2 | m+4 | 20 | Stretch goal; would be unprecedented for BO |

**This is where STCH-Set must shine.** The key claim is that softmax-based hypervolume approximation via STCH scales gracefully to many objectives while exact HV computation is exponential in m.

---

## 2. Real-World Benchmarks

### 2.1 MUST-HAVE: Multi-Objective Drug Discovery (TDC/ADMET)

**Problem:** Optimize a molecule generation oracle with multiple ADMET objectives simultaneously.

**Setup using TDC (Therapeutics Data Commons):**
- Use TDC's Oracle interface for molecular property evaluation
- Objectives (pick 4-6 from): LogP, QED, SA Score, Caco2 permeability, CYP2D6 inhibition, hERG toxicity, Lipophilicity, Solubility
- Input: molecular fingerprint or latent space (d ≈ 128-256)
- m = 4, 6 (show scaling)

**Why compelling:** Drug discovery is the #1 cited application of MOBO. Reviewers expect it.

**Alternative/complement:** Use PMO (Practical Molecular Optimization) benchmark from Gao et al. which provides standardized oracle budgets.

### 2.2 MUST-HAVE: Vehicle Crashworthiness / Engineering Design

| Problem | d | m | Source |
|---------|---|---|--------|
| Vehicle Crashworthiness | 5 | 3 | RE benchmark suite (Tanabe & Ishibuchi 2020) |
| Welded Beam Design | 4 | 2+1 constraint | Classic engineering benchmark |
| Car Side Impact | 7 | 3 | RE benchmark suite |
| Water Problem | 3 | 5 | RE benchmark suite — many-objective! |

**RE suite** (Real-world Engineering problems) is becoming standard. Use `pymoo` implementations.

### 2.3 NICE-TO-HAVE: Materials Discovery

Inspired by the Nature Comp. Mat. 2025 paper (Saar et al.):
- **Problem:** Optimize density (ρ) and heat of formation (ΔH_f,s) for energetic molecules
- **Setup:** Use surrogate trained on their 10k CSD dataset
- **m = 2-3** (add stability as 3rd objective)
- Shows relevance to active learning / generative design pipeline

### 2.4 NICE-TO-HAVE: Neural Network Hyperparameter Optimization

- Optimize accuracy, latency, model size, fairness metrics simultaneously
- m = 3-5
- Practical and relatable to ML audience

---

## 3. Baselines (Exact BoTorch Names)

### Tier 1: MUST compare against

| Method | BoTorch Implementation | Notes |
|--------|----------------------|-------|
| **qNEHVI** | `qNoisyExpectedHypervolumeImprovement` | State-of-the-art for noiseless/noisy MOBO |
| **qParEGO** | `qExpectedImprovement` + Chebyshev scalarization | Decomposition baseline |
| **qEHVI** | `qExpectedHypervolumeImprovement` | Exact (noiseless) variant |
| **MORBO** | `morbo` (separate package) | SOTA for high-d MOBO |
| **Random** | Sobol/random sampling | Lower bound |
| **NSGA-II** | `pymoo` implementation | Evolutionary baseline |

### Tier 2: SHOULD compare against

| Method | Source | Notes |
|--------|--------|-------|
| **DGEMO** | Original code | Diversity-guided MOBO |
| **MESMO/PFES** | BoTorch `qMultiObjectiveMaxValueEntropy` | Information-theoretic |
| **USeMO** | Original code | Uncertainty-guided |
| **MOBO-OSD** | OpenReview code | NeurIPS 2025 — orthogonal search directions |
| **NBI-MOBO** | OpenReview code | NeurIPS 2025 — normal boundary intersection |

### Tier 3: Nice-to-have

| Method | Notes |
|--------|-------|
| **BOFormer** | ICML 2024 — RL-based MOBO |
| **PSL (Pareto Set Learning)** | Learning the full Pareto set |
| **TSEMO** | Thompson sampling for MOBO |

---

## 4. Metrics

### Primary (report in ALL experiments)

| Metric | What It Measures | Implementation |
|--------|-----------------|----------------|
| **Log Hypervolume Difference** | Gap to true/reference Pareto front | `botorch.utils.multi_objective.hypervolume.Hypervolume` |
| **Hypervolume** (absolute) | Volume dominated by Pareto set | Same |

### Secondary (report in main experiments)

| Metric | What It Measures | Notes |
|--------|-----------------|-------|
| **IGD (Inverted Generational Distance)** | Convergence + diversity | `pymoo.indicators.igd` |
| **IGD+** | Modified IGD (Pareto-compliant) | `pymoo.indicators.igd_plus` |
| **Spacing** | Uniformity of Pareto front | Custom implementation |
| **Epsilon Indicator** | Worst-case approximation ratio | `pymoo.indicators.epsilon` |

### Operational (report for scalability story)

| Metric | What It Measures |
|--------|-----------------|
| **Wall-clock time per iteration** | Computational cost |
| **Wall-clock time vs. m** | Scaling behavior (our key advantage) |
| **GPU memory vs. m** | Memory scaling |
| **Number of Pareto-optimal points found** | Solution quality |

---

## 5. Statistical Rigor

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Seeds per experiment** | 20 | NeurIPS standard; 10 is minimum, 20 is strong |
| **Confidence intervals** | Mean ± 1 SEM (shaded in plots) | Standard practice |
| **Statistical test** | Wilcoxon signed-rank test | Non-parametric; report p-values in appendix |
| **Budget per run** | 200 evaluations (synthetic), 100 (real-world) | Standard for MOBO |
| **Initial points** | 2*(d+1) via Sobol | BoTorch convention |
| **Batch size** | q=1 (sequential) and q=4 (batch) | Test both settings |

---

## 6. Ablation Studies (MUST-HAVE)

| Ablation | What It Tests |
|----------|--------------|
| **STCH temperature (τ)** | Sensitivity to softmax temperature |
| **Set size (k)** | How many points in the STCH set matter |
| **MC samples** | Convergence of STCH approximation |
| **STCH vs exact HV** | Approximation quality as m grows |
| **With/without gradient through STCH** | Value of differentiable approximation |

---

## 7. Key Figures for the Paper

1. **Figure 1:** Convergence plots (log HV difference vs. evaluations) on DTLZ2 for m=2,3,5,8,10,15
2. **Figure 2:** Wall-clock time vs. number of objectives (log scale) — STCH-Set vs qNEHVI vs qParEGO
3. **Figure 3:** Real-world drug discovery results (Pareto fronts, HV curves)
4. **Figure 4:** Ablation grid (temperature, set size, MC samples)
5. **Figure 5:** Pareto front visualizations (parallel coordinates for m>3)

---

## 8. Experiment Priority & Compute Estimates

### Phase 1: MUST-HAVE (submit without these = desk reject)
*Estimated: 3-5 days on single GPU (A100)*

| Experiment | Runs | Est. Time |
|-----------|------|-----------|
| DTLZ2 m=3,5,8,10 × 6 methods × 20 seeds | 480 | 24h |
| DTLZ1,3,5,7 m=3 × 6 methods × 20 seeds | 480 | 24h |
| ZDT1,2,3 m=2 × 6 methods × 20 seeds | 360 | 8h |
| Wall-clock scaling study (m=2→20) | 100 | 4h |
| Drug discovery (TDC, m=4,6) × 4 methods × 20 seeds | 160 | 12h |
| Ablations (temperature, set size, MC) × 20 seeds | 200 | 8h |
| **Phase 1 Total** | **~1800 runs** | **~80h (3.3 days)** |

### Phase 2: SHOULD-HAVE (strengthens to strong accept)
*Estimated: 2-3 days on single GPU*

| Experiment | Runs | Est. Time |
|-----------|------|-----------|
| DTLZ2 m=15,20 (stretch objectives) | 240 | 16h |
| RE engineering problems × 4 methods × 20 seeds | 240 | 12h |
| Additional baselines (DGEMO, MESMO, MOBO-OSD) | 600 | 24h |
| Batch setting (q=4) on DTLZ2 subset | 200 | 8h |
| **Phase 2 Total** | **~1280 runs** | **~60h (2.5 days)** |

### Phase 3: NICE-TO-HAVE (cherry on top)
*Estimated: 1-2 days*

| Experiment | Runs | Est. Time |
|-----------|------|-----------|
| Materials design (surrogate) | 160 | 8h |
| HPO benchmark (m=4) | 160 | 8h |
| Constrained MOBO variants | 200 | 12h |
| **Phase 3 Total** | **~520 runs** | **~28h** |

### Total Compute Budget
- **Phase 1+2:** ~140h ≈ 6 days on 1× A100 (or ~1.5 days on 4× A100)
- **Phase 1+2+3:** ~168h ≈ 7 days on 1× A100

---

## 9. Nature Comp. Mat. 2025 Paper Analysis

**Paper:** "Active learning enables generation of molecules that advance the known Pareto front" (Saar et al., Nature Computational Materials 2025)

**What they did:**
- Active learning loop: JANUS genetic algorithm → DFT validation → MPNN retraining
- 2-objective optimization (density ρ, heat of formation ΔH_f,s) for energetic molecules
- Key insight: generative models fail to extrapolate beyond training Pareto front because surrogate models don't generalize OOD
- Active learning fixed this: 19× RMSE reduction, precision improved 7% → 86%

**How STCH-Set complements:**
- They use simple scalarization for multi-objective handling; STCH-Set would provide principled Pareto-aware acquisition
- They optimize only m=2 objectives; STCH-Set enables scaling to many more (add stability, synthesizability, cost, etc.)
- Their active learning loop could use STCH-Set as the acquisition function for selecting which molecules to simulate next
- **Potential collaboration angle:** Apply STCH-Set to their pipeline with m=4-6 objectives

---

## 10. The "bPK Score" and Multi-Objective Drug Discovery

The "balanced potency-kinetics" concept refers to **multiparameter optimization (MPO)** scores used in pharma (Novartis, Pfizer, etc.) that compress 5-10+ ADMET properties into a single scalar:
- Typically includes: potency (IC50/Ki), selectivity, permeability (Caco2), metabolic stability (CLint), plasma protein binding, hERG liability, CYP inhibition, solubility
- **The problem:** Scalarization loses Pareto information — a compound scoring 0.7 on the composite might dominate one scoring 0.8 on the objectives that actually matter

**Why STCH-Set is relevant:**
- Instead of compressing 8+ ADMET objectives into one number, optimize the actual Pareto front
- STCH-Set can handle m=8-10 objectives where qNEHVI becomes computationally intractable
- **This is our killer application story:** "Pharma currently uses MPO scores because existing MOBO can't handle m>4 objectives efficiently. STCH-Set changes this."

---

## 11. What Makes a "Strong Accept" MOBO Paper at NeurIPS

Based on analysis of recent accepted papers (MORBO NeurIPS 2022, DGEMO NeurIPS 2021, qNEHVI NeurIPS 2021):

### Must demonstrate:
1. **Clear theoretical contribution** — convergence guarantees or approximation bounds for STCH
2. **Synthetic benchmarks at scale** — DTLZ with m≥5 is table stakes; m≥10 is differentiating
3. **≥1 compelling real-world problem** — drug discovery or engineering design
4. **Head-to-head vs. qNEHVI and qParEGO** — these are the BoTorch defaults everyone knows
5. **Wall-clock comparison** — show computational advantage explicitly
6. **Statistical rigor** — ≥10 seeds with error bars; 20 is better
7. **Ablations** — show each design choice matters

### Differentiators that push to strong accept:
- Scaling to m=10-20 objectives (unprecedented in BO literature)
- Real drug discovery with actual ADMET objectives (not just toy)
- Open-source BoTorch integration
- Gradient-based optimization through the STCH approximation (unique selling point)

---

## 12. Quick-Start Checklist

- [ ] Implement STCH-Set acquisition function in BoTorch
- [ ] Set up DTLZ benchmark harness with configurable m
- [ ] Run ZDT sanity checks (should match or beat qNEHVI)
- [ ] Run DTLZ2 scaling study: m = 3, 5, 8, 10, 15
- [ ] Implement wall-clock timing infrastructure
- [ ] Set up TDC drug discovery benchmark
- [ ] Run Phase 1 experiments (20 seeds each)
- [ ] Generate convergence plots + scaling plots
- [ ] Run ablations
- [ ] Run Phase 2 (RE problems, additional baselines)
- [ ] Statistical significance tests
- [ ] Write results section

---

*Strategy authored by research assistant. Review with Dr. Vargas before execution.*
