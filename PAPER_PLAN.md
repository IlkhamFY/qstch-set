# STCH-BoTorch Paper Plan

**Date:** 2026-02-17
**Authors:** Ilkham Zolotarev, Rodrigo Vargas-Hernández (McMaster University)
**Target:** NeurIPS 2026 (deadline ~May 2026)

---

## A. Paper Outline (NeurIPS Format, 9 pages + references)

### Title Options

1. **"Smooth Tchebycheff Set Scalarization for Sample-Efficient Many-Objective Bayesian Optimization"**
   — Clean, precise, highlights the BO contribution over Lin et al.

2. **"Few Candidates, Many Criteria: Bayesian Optimization with Smooth Tchebycheff Set Scalarization"**
   — More accessible, emphasizes the practical "few-for-many" framing

3. **"Beyond Hypervolume: Scalable Many-Objective Bayesian Optimization via Smooth Tchebycheff Sets"**
   — Positions against qEHVI directly, provocative

**Recommendation:** Title 1 for NeurIPS main, Title 2 for a broader venue.

---

### Abstract (Draft)

> Multi-objective Bayesian optimization (MOBO) methods based on expected hypervolume improvement scale poorly beyond 4–5 objectives, yet real-world design problems—such as selecting drug candidates across dozens of ADMET criteria—are inherently many-objective. We introduce STCH-Set-BO, which integrates smooth Tchebycheff set (STCH-Set) scalarization into the BoTorch framework, enabling sample-efficient black-box optimization that finds a small set of K complementary solutions collectively covering m ≫ K objectives. Our approach inherits the Pareto optimality guarantees of STCH-Set theory while leveraging Gaussian process surrogates for sample efficiency in expensive evaluation settings. On standard benchmarks (ZDT, DTLZ), STCH-Set-BO matches state-of-the-art MOBO methods for 2–5 objectives and dramatically outperforms them as m grows to 10, 50, and 100 objectives. We demonstrate the practical value on a drug discovery task with up to 100 ADMET endpoints as simultaneous objectives. Our implementation is open-source and compatible with the BoTorch ecosystem.

---

### Section-by-Section Outline

#### 1. Introduction (1 page)
- **Hook:** Drug discovery needs K=3–5 candidates covering m=50–100+ ADMET/efficacy/safety criteria. No existing BO method handles this.
- **Gap:** qEHVI/#P-hard beyond ~5 objectives. ParEGO uses random weights (inefficient in high-dim objective space). Lin et al. (ICLR 2025) solve "few-for-many" but require gradient access.
- **Contribution bullets:**
  1. First integration of STCH-Set scalarization into Bayesian optimization
  2. Theoretical analysis: acquisition function inherits Pareto optimality guarantees under GP posterior
  3. Empirical demonstration on standard MOO benchmarks (m=2–100) and a drug discovery case study
  4. Open-source BoTorch-compatible implementation

#### 2. Background & Preliminaries (1 page)
- Multi-objective optimization, Pareto dominance, scalarization approaches
- Bayesian optimization with GPs: acquisition functions, MC integration
- Tchebycheff scalarization → smooth Tchebycheff (Lin & Zhang, ICML 2024)
- STCH-Set scalarization (Lin et al., ICLR 2025): Eq. 5 (STCH), Eq. 12 (STCH-Set), key theorems

#### 3. Method: STCH-Set Bayesian Optimization (2 pages)
- **3.1 STCH-Set as acquisition objective:** Plug STCH-Set scalarization into MC acquisition functions. Given GP posterior samples Y ~ p(f|D), compute STCH-Set utility; optimize via gradient-based acquisition optimization (BoTorch's `optimize_acqf`).
- **3.2 Weight selection strategy:** Discuss uniform weights vs. adaptive weight selection. For "few-for-many," uniform λ=(1/m,...,1/m) is natural (all objectives equally important). Optionally: Dirichlet sampling for exploration.
- **3.3 Reference point estimation:** Dynamic ideal point from GP posterior means vs. fixed user-specified ref_point. Discuss impact of μ on effective smoothing per objective.
- **3.4 Theoretical properties:**
  - Proposition 1: STCH-Set-BO acquisition inherits weak Pareto optimality (from Lin et al. Theorem 2) in the limit of exact GP posterior
  - Proposition 2: Differentiability enables efficient gradient-based acquisition optimization (vs. hard-max Chebyshev)
  - Proposition 3: Computational complexity O(Km) per acquisition evaluation vs. O(2^m) for hypervolume

#### 4. Related Work (1 page)
- Use the draft from literature-analysis.md (already written)
- Position against: qEHVI/qNEHVI, ParEGO/qNParEGO, MORBO, Pires & Coelho (2025), NSGA-III
- Key differentiator table: method vs. #objectives scalable vs. sample-efficient vs. set-based

#### 5. Experiments (3 pages)
- **5.1 Synthetic benchmarks (ZDT, DTLZ)** — match/beat SOTA for m=2–5
- **5.2 Many-objective scaling** — m={10, 20, 50, 100}, show qEHVI fails, STCH-Set-BO succeeds
- **5.3 Ablation studies** — smooth vs. non-smooth, effect of K, effect of μ
- **5.4 Drug discovery case study** — ADMET multi-objective optimization
- **5.5 Wall-clock time comparison** — show tractability vs. hypervolume methods

#### 6. Discussion & Limitations (0.5 page)
- Limitations: weight sensitivity, μ tuning, GP scalability for very high m
- Broader impact: responsible drug discovery, reducing animal testing through better in-silico optimization

#### 7. Conclusion (0.5 page)

---

### Key Figures Needed

| # | Figure | What It Shows | Status |
|---|--------|---------------|--------|
| 1 | **Conceptual diagram** | K=3 solutions covering m=20 objectives (radar/spider plot) vs. single Pareto-optimal solution | TODO |
| 2 | **ZDT Pareto fronts** | Recovered Pareto fronts on ZDT1–3 for STCH-Set-BO vs. qEHVI vs. qNParEGO | TODO |
| 3 | **Scalability plot** | Hypervolume (or worst-objective) vs. #evaluations for m={5,10,20,50,100}. Lines: STCH-Set-BO, qEHVI, qNParEGO, random. qEHVI line should degrade/disappear at high m. | TODO |
| 4 | **Ablation: K effect** | Worst-objective-value vs. K={1,2,3,5,10} on DTLZ with m=20 | TODO |
| 5 | **Ablation: μ effect** | Performance vs. μ={0.01, 0.1, 0.5, 1.0} showing sweet spot | TODO |
| 6 | **Drug discovery** | Pareto front / radar plot for ADMET case study, comparing K=5 STCH-Set candidates vs. top-5 from single-obj bPK optimization | TODO |
| 7 | **Wall-clock time** | Log-scale time per acquisition step vs. m for STCH-Set-BO vs. qEHVI | TODO |

---

### Experiments: What's Done vs. Missing

| Experiment | Status | Notes |
|-----------|--------|-------|
| STCH scalarization library | ✅ Done | Sign bug found & fixed |
| BoTorch objective wrappers | ✅ Done | `SmoothChebyshevObjective`, `SmoothChebyshevSetObjective` |
| Math verification vs. paper | ✅ Done | Verified correct after sign fix |
| ZDT benchmarks | ❌ Missing | Priority 1 |
| DTLZ benchmarks | ❌ Missing | Priority 1 |
| qEHVI/qNParEGO baselines | ❌ Missing | Priority 1 |
| pymoo NSGA-II/III baselines | ❌ Missing | Priority 1 |
| Many-objective scaling (m>10) | ❌ Missing | Priority 1 |
| Ablation: K, μ, smooth vs. hard | ❌ Missing | Priority 2 |
| ADMET/drug discovery case study | ❌ Missing | Priority 2 |
| Wall-clock benchmarks | ❌ Missing | Priority 2 |
| Theoretical propositions/proofs | ❌ Missing | Priority 1 |

---

### Related Work Positioning

**We are the "bridge" paper:** Lin et al. (ICLR 2025) provides theory for gradient-based optimization. We bring it to the sample-efficient black-box BO setting. This is analogous to how ParEGO brought Chebyshev scalarization to BO from the evolutionary optimization community.

Key positioning:
- vs. **qEHVI/qNEHVI**: We scale to m≫5 objectives; they don't (hypervolume is #P-hard in m)
- vs. **qNParEGO**: We use principled STCH-Set instead of random weight sampling; we find complementary solution sets
- vs. **Lin et al. (ICLR 2025)**: We handle black-box expensive evaluations via GP surrogates; they need gradients
- vs. **Pires & Coelho (2025)**: We do set scalarization (K solutions for m objectives); they do single-solution STCH only
- vs. **NSGA-III**: We are sample-efficient (BO); they need thousands of evaluations

---

## B. Concrete Experiment TODO List

### Phase 1: Core Benchmarks (Weeks 1–3) — MUST HAVE

**1. ZDT Suite (m=2)**
- Problems: ZDT1 (convex), ZDT2 (non-convex), ZDT3 (disconnected), ZDT4 (multimodal), ZDT6
- Baselines: qEHVI, qNParEGO, pymoo NSGA-II, random
- STCH-Set-BO: K=1 (single solution, fair comparison)
- Budget: 100 evaluations (after 10 initial Sobol points)
- Metrics: Hypervolume, IGD, Spacing
- Repeats: 30 independent runs (different random seeds)
- **Owner: Rodrigo** (per SUPERVISOR_TODO)

**2. DTLZ Suite (m=3, 5, 10)**
- Problems: DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ7
- Baselines: same + pymoo NSGA-III (reference-point based)
- STCH-Set-BO: K=1 for m=3; K=2–3 for m=5; K=3–5 for m=10
- Budget: 200 evaluations (20 initial)
- Metrics: Hypervolume (where computable), worst-objective-value, IGD
- Repeats: 20 runs

**3. Vanilla BoTorch baselines (control)**
- Run qEHVI and qNParEGO with identical GP setup (same kernel, same initialization)
- This is critical for fair comparison — no cherry-picking

### Phase 2: Scalability & Ablations (Weeks 3–5) — MUST HAVE

**4. Many-objective scaling**
- Problem: DTLZ2 with m={10, 20, 50, 100} (scalable by design)
- Baselines: qNParEGO (qEHVI will likely fail for m>10), random, NSGA-III (with large pop)
- STCH-Set-BO: K = ceil(sqrt(m)) or K={3,5,10}
- Budget: 500 evaluations
- **Key plot:** worst-objective-value vs. evaluations for each m
- Repeats: 10 runs (expensive at high m)

**5. Ablation: smooth vs. non-smooth**
- Compare STCH-Set-BO vs. TCH-Set-BO (hard max, hard min — non-differentiable)
- On DTLZ2 m=10
- Show smooth version converges faster / to better solutions

**6. Ablation: K (number of solutions)**
- K={1, 2, 3, 5, 10} on DTLZ2 m=20
- Show that K>1 improves worst-objective coverage

**7. Ablation: μ (smoothing parameter)**
- μ={0.01, 0.05, 0.1, 0.5, 1.0, 5.0}
- On DTLZ2 m=10
- Show there's a sweet spot; too small → numerical issues; too large → poor approximation

### Phase 3: Application & Impact (Weeks 5–8) — STRONG DIFFERENTIATOR

**8. Drug discovery / ADMET benchmark**
- Option A: Use TDC (Therapeutics Data Commons) ADMET benchmark — 22 endpoints as objectives. Pick a molecular library (e.g., ZINC subset). Use GP over Morgan fingerprints.
- Option B: Use bPK's ~100 ADMET predictors as individual objectives over a ChEMBL subset
- STCH-Set-BO (K=5) vs. single-objective bPK score optimization vs. qNParEGO on top-5 ADMET endpoints
- Show: STCH-Set finds complementary candidates covering more ADMET criteria

**9. Wall-clock time**
- Plot acquisition optimization time vs. m for STCH-Set-BO vs. qEHVI
- Show O(Km) vs. exponential blowup
- Run on same hardware, report mean ± std over 10 acquisition steps

**10. Robustness to noise**
- Add Gaussian noise σ={0.01, 0.1, 0.5} to DTLZ2 m=10
- Show GP-based STCH-Set-BO degrades gracefully

### Phase 4: Nice-to-Have (If Time Permits)

**11. Comparison with dimensionality reduction**
- PCA on m=100 objectives → reduce to 5 → run qEHVI → project back
- Show STCH-Set-BO without reduction outperforms

**12. Branin-Currin (BoTorch standard)**
- 2-objective standard BoTorch benchmark
- Good for sanity check and visual Pareto front comparison

---

## C. Gap Analysis: Weaknesses & Reviewer Attack Points

### 🔴 Critical Weakness 1: NO EXPERIMENTS YET
The code exists but zero benchmark results. This is the #1 blocker. Without ZDT/DTLZ results, there is no paper.

**Mitigation:** Phase 1 experiments (Weeks 1–3). Rodrigo should start ZDT baselines immediately.

### 🔴 Critical Weakness 2: Theoretical Contribution is Thin
We're essentially plugging Lin et al.'s scalarization into BoTorch. A reviewer could say: "This is engineering, not research. What's the new theorem?"

**Mitigation options:**
- Prove a regret bound for STCH-Set-BO (e.g., cumulative regret under GP assumptions)
- Prove that STCH-Set acquisition is submodular or approximately submodular (enables greedy batch selection guarantees)
- At minimum: formal proposition that Pareto optimality guarantees transfer from the true objective to the GP posterior in the infinite-data limit

### 🟡 Weakness 3: Weight Selection
STCH-Set with uniform weights is natural for "all objectives equal." But what if objectives have different scales? Different importance? A reviewer will ask: "How do you choose λ?"

**Mitigation:** (a) Auto-normalization of objectives (standard in BoTorch); (b) Experiment showing robustness to weight perturbation; (c) Discuss adaptive weight strategies as future work.

### 🟡 Weakness 4: μ Sensitivity
The smoothing parameter μ is a hyperparameter. Too small → hard-max behavior (non-differentiable in practice). Too large → poor approximation. Reviewer: "How do you set μ? Is it sensitive?"

**Mitigation:** Ablation study (Phase 2, experiment 7). Provide practical guidelines (e.g., μ = 0.1 works across all tested problems).

### 🟡 Weakness 5: Comparison Fairness
STCH-Set-BO returns K solutions; qEHVI returns a Pareto set. How do you compare? If STCH-Set-BO gets K=5 evaluations per iteration but qEHVI gets q=1, that's unfair.

**Mitigation:** 
- Fair budget: total evaluations equal across methods (e.g., STCH-Set-BO: 100 iters × K=5 = 500 evals; qEHVI: 500 iters × q=1 = 500 evals)
- Also compare: STCH-Set-BO K=5 vs. qEHVI q=5 (same batch size)
- Report both hypervolume (favors full Pareto methods) AND worst-objective-value (favors STCH-Set)

### 🟡 Weakness 6: GP Scalability
GPs with m=100 outputs are expensive. Independent GPs per objective? Multi-output GP? Reviewer: "Does this actually run for m=100?"

**Mitigation:** 
- Use independent GPs (standard in BoTorch MOBO) — this scales linearly in m
- Report wall-clock times explicitly
- Note that the bottleneck shifts from acquisition (solved by STCH) to surrogate modeling (orthogonal problem)

### 🟢 Minor: Pires & Coelho (2025) Overlap
They do single-solution STCH in composite BO. A reviewer might ask: "How is this different?"

**Mitigation:** Clear differentiation: (a) we do set scalarization (K solutions), they don't; (b) we target many-objective (m≫5), they don't; (c) we provide scaling experiments up to m=100.

### 🟢 Minor: Lin et al. May Scoop Us
The STCH-Set authors could publish a BO extension themselves.

**Mitigation:** Move fast. Workshop paper by May 2026 establishes priority. Cite them generously and frame as "bringing their theory to the BO community."

---

## Summary: Critical Path to Submission

```
Week 1-2:  ZDT + DTLZ benchmarks (Rodrigo) + theoretical propositions (Ilkham)
Week 3:    Many-objective scaling experiments + ablations
Week 4-5:  Drug discovery case study + wall-clock benchmarks
Week 6:    Paper writing (intro + methods + experiments)
Week 7:    Related work + figures + polish
Week 8:    Internal review, submit
```

**Hard requirement for publishability:** At minimum, experiments 1–4 and 7 (ZDT, DTLZ, scaling, μ ablation) with 3+ baselines. The drug discovery case study elevates from "solid" to "strong accept."
