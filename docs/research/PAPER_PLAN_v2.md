# STCH-Set-BO Paper Plan v2 — Refined & Honest

**Date:** 2026-02-17  
**Authors:** Ilkham Zolotarev, Rodrigo Vargas-Hernández (McMaster University)  
**Target:** NeurIPS 2026 (deadline ~May 2026)

---

## Novelty Statement (Be Precise)

**What we did NOT invent:**
- STCH scalarization (Lin & Zhang, ICML 2024)
- STCH-Set scalarization (Lin et al., ICLR 2025)
- Combining STCH with Bayesian optimization (Pires & Coelho, SSRN 2025 — single-point composite BO)
- Exact hypervolume-based acquisition for BO (qEHVI, ε-PoHVI, etc.)

**What we contribute (narrow, defensible):**
1. First integration of **STCH-Set** (not single-point STCH) into sample-efficient Bayesian optimization with GP surrogates
2. A BO method that scales to **many objectives (m ≫ 5)** where hypervolume-based methods are intractable
3. Sample-efficient discovery of **K complementary solutions** that collectively cover m objectives — the "few-for-many" problem solved with orders of magnitude fewer evaluations than evolutionary methods
4. Open-source BoTorch-compatible implementation with practical guidelines

**The gap we fill:** Lin et al. (ICLR 2025) solve few-for-many with gradients. Pires & Coelho (2025) do single-point STCH in BO. Nobody does **set-based STCH in BO** — i.e., finding K complementary solutions sample-efficiently for many-objective expensive black-box problems.

---

## Title

**"Sample-Efficient Many-Objective Bayesian Optimization via Smooth Tchebycheff Set Scalarization"**

---

## Abstract (Revised — No Overclaiming)

> Finding a small set of complementary solutions that collectively satisfy many competing objectives is a fundamental challenge in engineering and drug design, yet existing multi-objective Bayesian optimization (MOBO) methods degrade beyond approximately five objectives. We present STCH-Set-BO, which integrates smooth Tchebycheff set (STCH-Set) scalarization—recently proposed by Lin et al. (2025) for gradient-based optimization—into the Bayesian optimization framework with Gaussian process surrogates. This enables sample-efficient black-box discovery of K complementary solutions covering m ≫ K objectives, a regime where hypervolume-based acquisitions (qEHVI) are intractable and scalarization-based methods (ParEGO) are inefficient. On synthetic many-objective benchmarks (DTLZ, m up to 100), STCH-Set-BO finds solution sets with substantially better worst-case objective coverage than baselines while using orders of magnitude fewer evaluations than evolutionary methods such as NSGA-III. We demonstrate practical value on a drug discovery task with dozens of simultaneous ADMET endpoints. Our implementation is open-source and compatible with BoTorch.

---

## Section-by-Section Outline

### 1. Introduction (1 page)

- **Problem:** Real-world design often requires finding K = 3–5 candidates covering m = 20–100+ criteria (e.g., drug candidates across ADMET endpoints). Each evaluation is expensive.
- **Existing approaches and their limits:**
  - qEHVI: hypervolume is #P-hard in m; practically limited to m ≤ 5
  - ParEGO: random weight scalarization; single-solution, inefficient for many objectives
  - NSGA-III: needs thousands of evaluations; not sample-efficient
  - Lin et al. (ICLR 2025): STCH-Set solves few-for-many but requires gradient access
  - Pires & Coelho (2025): STCH + composite BO, but single-solution only
  - Wang et al. (ICML 2024): ε-PoHVI with exact posterior, but limited to 2 objectives
- **Our contribution:** Bridge STCH-Set theory to the sample-efficient BO setting. Not a new scalarization — a new way to use an existing one for expensive black-box problems.
- **Contribution bullets:**
  1. First integration of STCH-Set into BO with GP surrogates for many-objective black-box optimization
  2. Empirical demonstration that set-based scalarization outperforms single-point methods when K > 1 and m ≫ K
  3. Scaling experiments up to m = 100 objectives in the BO setting
  4. Open-source BoTorch implementation

### 2. Background & Preliminaries (1 page)

- Multi-objective optimization, Pareto dominance, scalarization
- Bayesian optimization with GPs; MC acquisition functions
- Tchebycheff → Smooth Tchebycheff (Lin & Zhang, ICML 2024)
- STCH-Set (Lin et al., ICLR 2025): Eq. 12, key theorems (weak Pareto optimality, convergence)
- Composite BO (Astudillo & Frazier, 2019) — relevant since Pires & Coelho use it

### 3. Method: STCH-Set Bayesian Optimization (2 pages)

- **3.1 STCH-Set as MC acquisition objective:** Given GP posterior samples, compute STCH-Set utility over candidate set of K solutions; optimize via BoTorch's `optimize_acqf`.
- **3.2 Practical considerations:**
  - Weight selection: uniform λ = (1/m, ..., 1/m) for equal importance; Dirichlet sampling for exploration
  - Reference point: dynamic estimation from GP posterior or user-specified
  - μ selection: practical guidelines from ablation
- **3.3 Computational complexity:** O(Km) per acquisition evaluation vs. exponential for hypervolume. This is the key scalability argument.
- **3.4 Inherited theoretical properties:** STCH-Set-BO inherits weak Pareto optimality guarantees from Lin et al. Theorem 2 in the limit of accurate GP posteriors. We do NOT claim new theoretical results beyond this inheritance argument.

### 4. Related Work (1 page)

**Precise positioning against the 4 key competitors:**

| Method | Scalarization | Set-based (K>1) | Sample-efficient (BO) | Many-objective (m≫5) |
|--------|--------------|-----------------|----------------------|---------------------|
| qEHVI (Daulton et al., 2020/2021) | Hypervolume | No (Pareto set) | Yes | No (#P-hard) |
| ε-PoHVI (Wang et al., ICML 2024) | Hypervolume (exact) | No | Yes | No (m=2 only) |
| ParEGO / qNParEGO | Random Chebyshev | No | Yes | Degrades |
| STCH-Set (Lin et al., ICLR 2025) | STCH-Set | **Yes** | No (gradients) | **Yes** |
| Pires & Coelho (2025) | Single STCH | No | Yes (composite) | Not demonstrated |
| NSGA-III (Deb & Jain, 2013) | Reference-point EA | Population | No (1000s evals) | Yes |
| **STCH-Set-BO (ours)** | STCH-Set | **Yes** | **Yes** | **Yes** |

> **Multi-Objective Bayesian Optimization.** Hypervolume-based acquisitions (qEHVI, qNEHVI; Daulton et al., 2020, 2021) are the gold standard for 2–4 objectives but become intractable for m > 5 due to the #P-hardness of hypervolume computation. Wang et al. (ICML 2024) derive an exact posterior distribution for ε-Probability of Hypervolume Improvement, but restrict to bi-objective problems. Scalarization approaches such as ParEGO (Knowles, 2006) scale better in m but rely on random weight sampling and optimize a single solution per iteration. Trust-region methods (MORBO; Daulton et al., 2022) address high-dimensional inputs but not many objectives.
>
> **Smooth Tchebycheff Scalarization.** Lin & Zhang (ICML 2024) introduced STCH using log-sum-exp smoothing for gradient-based MOO. Lin et al. (ICLR 2025) extended this to STCH-Set for the "few-for-many" setting (K solutions, m objectives), with theoretical guarantees on weak Pareto optimality. Both works assume gradient access and unlimited evaluations. Pires & Coelho (2025) integrate single-point STCH into composite Bayesian optimization for materials design, exploiting the nested structure f ∘ g. Their work is single-solution and does not address the set-based or many-objective setting. Our work extends beyond both: we use **set** scalarization (K > 1) in BO and target many objectives (m ≫ 5).
>
> **Many-Objective Optimization.** Evolutionary methods (NSGA-III, MOEA/D) handle many objectives but require thousands of evaluations, making them unsuitable for expensive black-box problems. Dimensionality reduction approaches (PCA on objectives) lose information. The "few-for-many" formulation (Lin et al., 2025) is explicitly designed for this regime but has only been demonstrated in gradient-based settings.

### 5. Experiments (3 pages)

**Core argument to demonstrate:** STCH-Set-BO's advantage is (a) many-objective scaling and (b) complementary solution sets — NOT beating qEHVI on 2-objective problems.

#### 5.1 Sanity Check: Low-Objective Benchmarks (m = 2–5)
- ZDT1–3 (m=2), DTLZ2 (m=3,5)
- STCH-Set-BO (K=1) vs. qEHVI, qNParEGO
- **Goal:** Show competitive (not necessarily better) performance. Establish we don't break things.
- Metrics: Hypervolume, IGD
- 20 seeds, 100 evaluations after 10 initial Sobol

#### 5.2 Many-Objective Scaling (KEY EXPERIMENT)
- DTLZ2 with m = {5, 10, 20, 50, 100}
- STCH-Set-BO with K = {3, 5, ceil(√m)}
- Baselines: qNParEGO (qEHVI infeasible for m > 10), NSGA-III (with generous eval budget), random
- Budget: 200 BO evaluations; NSGA-III gets 5000 evaluations to be generous
- **Primary metric: worst-objective value** across all m objectives (the STCH-Set objective itself)
- Secondary: average objective value, coverage (fraction of objectives below threshold)
- 10 seeds
- **Key figure:** worst-objective vs. evaluations, one subplot per m. Show STCH-Set-BO at 200 evals beats NSGA-III at 5000 evals for large m.

#### 5.3 Set Advantage: K > 1 vs. K = 1
- DTLZ2, m = 20
- Compare K = {1, 2, 3, 5, 10}
- Show that K > 1 improves worst-objective coverage — the core set-based argument
- Also compare: STCH-Set-BO (K=5) vs. running single-point STCH-BO 5 times independently → show complementarity matters

#### 5.4 Ablations
- **μ sensitivity:** μ = {0.01, 0.1, 0.5, 1.0, 5.0} on DTLZ2, m=10. Provide practical default.
- **Smooth vs. non-smooth:** STCH-Set-BO vs. TCH-Set-BO (hard max/min). Show smoothness helps gradient-based acquisition optimization.
- **Weight robustness:** uniform vs. random Dirichlet weights

#### 5.5 Drug Discovery Case Study
- Use TDC ADMET benchmark: ~22 endpoints as objectives
- Molecular fingerprint space, GP surrogates per endpoint
- STCH-Set-BO (K=5) vs. single-objective (bPK-like composite) vs. qNParEGO on top-5 endpoints
- Show: K=5 candidates collectively cover more ADMET criteria than any single candidate
- Radar/spider plots for visualization

#### 5.6 Wall-Clock Time
- Acquisition optimization time vs. m for STCH-Set-BO vs. qEHVI
- Show O(Km) scaling vs. exponential

### 6. Discussion & Limitations (0.5 page)

**Honest limitations:**
- Theoretical contribution is primarily integration, not new theory — we inherit guarantees from Lin et al.
- GP surrogates for m = 100 independent outputs are expensive to fit (though linear in m)
- μ requires tuning (we provide guidelines but no adaptive scheme)
- Uniform weights assume equal objective importance; heterogeneous importance requires user input
- The method finds K complementary solutions, not a full Pareto front — this is a feature for drug discovery but a limitation for Pareto front recovery

### 7. Conclusion (0.5 page)

---

## Key Figures

| # | Figure | Purpose |
|---|--------|---------|
| 1 | Conceptual: K=3 solutions covering m=20 objectives (radar plot) | Explain the problem and our approach |
| 2 | Many-objective scaling: worst-obj vs. evals for m={5,10,20,50,100} | **Core result** — show scaling advantage |
| 3 | Set advantage: worst-obj vs. K on DTLZ2 m=20 | Show K>1 helps |
| 4 | ZDT/DTLZ Pareto fronts (m=2,3) | Sanity check — competitive with qEHVI |
| 5 | μ ablation | Practical guidance |
| 6 | Drug discovery radar plots | Application impact |
| 7 | Wall-clock time vs. m | Computational tractability |

---

## Gap Analysis: Reviewer Attack Points

### 🔴 Critical: "This is engineering, not research"
The core contribution is plugging Lin et al.'s scalarization into BoTorch. A reviewer could dismiss this as incremental.

**Defense:**
- The integration is non-trivial: MC sampling through STCH-Set, reference point estimation from GP posteriors, interaction between smoothing parameter and GP uncertainty
- The empirical contribution is substantial: first demonstration of sample-efficient many-objective BO at scale
- Analogous precedent: ParEGO (Knowles, 2006) brought Chebyshev scalarization to BO and has 2000+ citations — the "bridge" contribution is valued
- BUT: we should NOT claim new theorems we don't have. Honesty here is better than fake propositions.

### 🔴 Critical: No experiments yet
Zero benchmark results. This is the #1 blocker.

**Mitigation:** Prioritize experiments 5.1 and 5.2 immediately.

### 🟡 "How does this differ from Pires & Coelho (2025)?"
They do STCH + composite BO.

**Defense (precise):**
1. They use single-point STCH; we use STCH-**Set** (K solutions jointly optimized)
2. They use composite BO (exploiting f = g ∘ h structure); we use standard MC acquisition
3. They do not demonstrate many-objective scaling (no m > 5 experiments shown)
4. The set aspect is the key differentiator — finding complementary solutions is fundamentally different from finding one good solution

### 🟡 "How does this differ from just running ParEGO K times?"
Running single-point scalarization K times independently gives K solutions, but they are not jointly optimized for complementarity.

**Defense:** Ablation 5.3 explicitly tests this. STCH-Set jointly optimizes K solutions so they *complement* each other (different solutions cover different objectives). Independent runs may converge to similar solutions.

### 🟡 Comparison fairness: K solutions vs. 1 solution
If STCH-Set-BO returns K solutions per iteration, the eval budget must be fair.

**Defense:** Total evaluations are equal across methods. STCH-Set-BO with K=5 gets 40 iterations × 5 = 200 evals; baselines get 200 iterations × 1 = 200 evals. Also test q=K batch for qNParEGO.

### 🟡 μ sensitivity
**Defense:** Ablation study + practical default (μ = 0.1 or similar). Show results are robust across a range.

### 🟡 GP scalability for m = 100
**Defense:** Use independent GPs (standard in BoTorch MOBO, linear in m). Report wall-clock times. Note surrogate modeling is orthogonal to our contribution (acquisition function design).

### 🟢 "Wang et al. (ICML 2024) solved exact acquisition for hypervolume"
**Defense:** Their ε-PoHVI is restricted to m = 2 (bi-objective). We target m ≫ 5. Different regimes entirely.

### 🟢 "Lin et al. may scoop you"
**Defense:** Move fast. Their ICLR 2025 paper explicitly flags BO as future work. Cite generously, frame as "bringing their theory to the BO community."

---

## Experiment Priority & Timeline

```
Week 1-2:  5.1 (ZDT/DTLZ sanity) + 5.2 (many-objective scaling) — MUST HAVE
Week 3:    5.3 (set advantage K>1) + 5.4 (ablations) — MUST HAVE
Week 4-5:  5.5 (drug discovery) + 5.6 (wall-clock) — STRONG DIFFERENTIATOR
Week 6-7:  Paper writing
Week 8:    Polish, internal review, submit
```

**Minimum viable paper:** Experiments 5.1, 5.2, 5.3, 5.4 with 3+ baselines. Drug discovery elevates from "borderline" to "accept."

---

## What We Do NOT Claim

- We do NOT claim to invent STCH or STCH-Set scalarization
- We do NOT claim to be the first to combine STCH with BO (Pires & Coelho, 2025)
- We do NOT claim new theoretical results (we inherit from Lin et al.)
- We do NOT claim to recover full Pareto fronts (we find K complementary solutions)
- We DO claim: first STCH-**Set** in BO, first sample-efficient many-objective (m ≫ 5) BO finding complementary solution sets, and practical demonstration on drug discovery
