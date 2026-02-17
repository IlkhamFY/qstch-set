# Deep Literature Map: STCH-Set-BO

**Generated:** 2026-02-17 | **Purpose:** Comprehensive related work for NeurIPS 2026 submission

---

## 1. The Landscape: Who Did What

### A. Scalarization-Based MOBO

| Paper | Method | What They Did | #Obj Tested | Gap |
|-------|--------|--------------|-------------|-----|
| Knowles (2006) | **ParEGO** | Random Chebyshev weights + EI on scalarized GP. Sequential, single-point. | 2–4 | Non-differentiable max; single solution per iteration; degrades with m |
| Paria et al. (UAI 2019) | **MOBO-RS** | Bayesian regret framework for random scalarization BO; theoretical analysis of weight priors | 2–3 | Theory only for simple cases |
| **Daulton et al. (NeurIPS 2020)** | **qNParEGO** | Batch ParEGO in BoTorch; sequential greedy with Chebyshev scalarization + qEI | 2–4 | Still uses non-smooth max; random weights; no set-based coordination |
| **Pires & Coelho (SSRN 2025)** | **STCH + Composite BO** | Single-point STCH in BO via Astudillo-Frazier composite framework. First smooth scalarization in BO. | 2–3 (likely) | Single-solution only; no set-based; limited obj scale |

### B. Hypervolume-Based MOBO

| Paper | Method | What They Did | #Obj Tested | Gap |
|-------|--------|--------------|-------------|-----|
| Daulton et al. (NeurIPS 2020, 2021) | **qEHVI / qNEHVI** | Exact/noisy expected hypervolume improvement; batch via inclusion-exclusion | 2–6 | Hypervolume is #P-hard in m; practically ≤5 objectives |
| Wang et al. (ICML 2024) | **ε-PoHVI** | Exact posterior integration for hypervolume; avoids MC sampling | 2 | Even more restricted in #obj than qEHVI |
| Daulton et al. (ICML 2022) | **MORBO** | Multi-trust-region local BO for high-dimensional inputs (100s of params); uses local GPs + Thompson sampling for HVI | 2–4 obj, 146–222 input dims | Scales input dims, NOT objective dims |

### C. Information-Theoretic MOBO

| Paper | Method | What They Did | #Obj Tested | Gap |
|-------|--------|--------------|-------------|-----|
| Belakaria et al. (2019) | **MESMO** | Max-value entropy search for multi-obj; avoids Pareto front sampling | 2–4 | Approximations degrade with m |
| Suzuki et al. (ICML 2020) | **PFES** | Pareto frontier entropy search; closed-form info gain | 2–4 | Scales poorly beyond ~4 obj |
| Ishikura et al. (ICML 2025) | **PFES-VLB** | Variational lower bound for PFES; better for many objectives | Claims many-obj | Still approximation-heavy |
| Tu et al. (NeurIPS 2022) | **JES** | Joint entropy search over inputs and objectives | 2–4 | Same scalability wall |

### D. Smooth Tchebycheff Scalarization (Gradient-Based, NOT BO)

| Paper | Method | What They Did | #Obj Tested | Gap |
|-------|--------|--------------|-------------|-----|
| **Lin et al. (ICML 2024)** | **STCH** | Smooth Tchebycheff via log-sum-exp; O(1/ε) convergence; differentiable | 2–10+ | Requires gradient access; NOT for black-box |
| **Lin et al. (ICLR 2025)** | **STCH-Set** | Set-based STCH: K solutions collectively cover m objectives via "few-for-many" formulation | Up to 1024 obj | Requires gradient access; NOT for expensive black-box; NOT sample-efficient |

### E. Batch & Diversity Methods

| Paper | Method | What They Did | #Obj Tested | Gap |
|-------|--------|--------------|-------------|-----|
| Lyu et al. (ICML 2018) | **DGEMO** | Diversity-guided batch selection; Pareto set representation | 2–3 | Limited obj scale |
| Bradford et al. (2018) | **Kriging Believer** | Hallucinate observations for batch construction | 2–3 | No set-based coordination |
| Various (DPP-based) | **DPP batches** | Determinantal point processes for diverse batch selection | 2–4 | Diversity in input space, not objective coverage |

### F. Composite Bayesian Optimization

| Paper | Method | What They Did | #Obj Tested | Gap |
|-------|--------|--------------|-------------|-----|
| **Astudillo & Frazier (ICML 2019)** | **EI-CF** | Composite f(x)=g(h(x)); model h with multi-output GP, exploit known g for better EI | N/A (single obj) | Framework enables STCH-in-BO (Pires & Coelho use this) |

### G. Many-Objective Evolutionary Methods (Non-BO)

| Paper | Method | What They Did | #Obj Tested | Gap |
|-------|--------|--------------|-------------|-----|
| Deb & Jain (2014) | **NSGA-III** | Reference-point based evolutionary MaOO; Das-Dennis structured weights | 3–15 | Needs 1000s+ evaluations; not sample-efficient |
| Li et al. (various) | **MOEA/D** | Decomposition-based EA with Tchebycheff scalarization | 3–15 | Same: needs many evaluations |

### H. Preference & Adaptive Weight Methods

| Paper | Method | What They Did | #Obj Tested | Gap |
|-------|--------|--------------|-------------|-----|
| Lin et al. (NeurIPS 2025?) | **PUB-MOBO** | Preference-guided MOBO with GI acquisition | 2–4 | Targets user preference, not Pareto coverage |
| Various | **Adaptive weight selection** | Dynamically adjust scalarization weights based on Pareto coverage gaps | 2–8 | Not integrated with smooth scalarization or set-based approaches |

### I. Reference Materials

| Source | Key Insight |
|--------|------------|
| **Austin Tripp blog (May 2025)** | Excellent pedagogical exposition of Chebyshev scalarization, augmented variant, geometric interpretation. Notes: augmented Chebyshev adds ρΣwifi for strict monotonicity. No BO connection discussed. |
| **qNParEGO BoTorch source** | Uses `get_chebyshev_scalarization`: weights from `sample_simplex` (Dirichlet(1)), computes max_i(w_i · |f_i - z_i|) + ρ·Σw_i·f_i. Non-differentiable max operator. Sequential greedy for batches. |

---

## 2. Gap Analysis: Where We Fit

### The 2×2 Matrix

|  | Single Solution | Set of K Solutions |
|--|----------------|-------------------|
| **Gradient-based (cheap eval)** | STCH (Lin, ICML 2024) | STCH-Set (Lin, ICLR 2025) |
| **Sample-efficient BO (expensive eval)** | Pires & Coelho (2025) | **← US (STCH-Set-BO)** |

This is the clearest positioning. The bottom-right cell is empty. We fill it.

### The Objectives Scale Gap

| Method | Practical #Objectives Limit |
|--------|---------------------------|
| qEHVI/qNEHVI | ~5 (hypervolume exponential) |
| MESMO/PFES/JES | ~4 (entropy approximations) |
| ParEGO/qNParEGO | ~10 (random weights, no coordination) |
| MORBO | ~4 obj (scales input dims, not objectives) |
| NSGA-III | ~15 (but needs 1000s evals) |
| **STCH-Set-BO (ours)** | **~100 (log-sum-exp is O(m))** |

### What Nobody Has Done
1. **Set-based scalarization in BO** — finding K coordinated solutions sample-efficiently
2. **BO with >10 objectives** in a principled way (ParEGO works but poorly)
3. **Smooth differentiable scalarization for acquisition optimization** in multi-objective BO (Pires & Coelho did single-point; nobody did set-based)
4. **The "few-for-many" problem in the expensive black-box setting** — Lin et al. need gradients

---

## 3. Reviewer Attack Surface

### Attack 1: "Just an engineering contribution — plugging STCH-Set into BoTorch"
**Defense:** The integration is non-trivial: (a) composite objective formulation with multi-output GP, (b) coordinated batch acquisition where K solutions must complement each other, (c) theoretical analysis showing sample complexity advantages over random scalarization. The gap between gradient-based MOO and sample-efficient BO is real — STCH-Set assumes differentiable objectives; we show how to make it work with GP posteriors.

### Attack 2: "Why not just run qNParEGO with more random weights?"
**Defense:** Random weights have no coordination — they don't ensure K solutions cover all m objectives. STCH-Set explicitly optimizes worst-case coverage. Experiments should show STCH-Set-BO finds much better min-objective coverage than qNParEGO with same budget.

### Attack 3: "MORBO already does multi-objective BO at scale"
**Defense:** MORBO scales INPUT dimensions (100s of parameters), not OBJECTIVE dimensions. MORBO was tested with 2–4 objectives. We scale to 100 objectives. Orthogonal contributions.

### Attack 4: "Why not approximate hypervolume for many objectives?"
**Defense:** Even approximate hypervolume is exponential in m. Monte Carlo HV estimation has been tried but variance is huge for m>6. Our method is O(Km) per scalarization evaluation.

### Attack 5: "Pires & Coelho (2025) already did STCH in BO"
**Defense:** They did single-point STCH, not STCH-Set. Single-point finds ONE solution per weight vector. We find K coordinated solutions jointly. For the "few-for-many" problem (K=3, m=50), single-point methods need to somehow combine solutions post-hoc with no optimality guarantee.

### Attack 6: "Limited novelty — combining two known things"
**Defense:** The combination creates a method that addresses a problem neither component solves alone. STCH-Set needs gradients (no black-box). BO methods don't scale to many objectives. Together: sample-efficient many-objective optimization. The contribution is the bridge + the empirical demonstration that it works.

### Attack 7: "No regret bounds / theoretical guarantees"
**Defense:** Provide convergence analysis leveraging Astudillo-Frazier composite BO consistency results + STCH-Set's Pareto optimality guarantees. If formal regret bounds are too hard, be upfront and focus on empirical strength.

---

## 4. Most Convincing Experiments

### Tier 1: Must Have (Reviewers will reject without these)

1. **DTLZ suite, m ∈ {5, 10, 20, 50, 100} objectives**
   - Compare: STCH-Set-BO vs qNParEGO vs qEHVI (where feasible) vs random search
   - Metrics: worst-case objective value across K solutions, hypervolume (where computable), IGD+
   - Show: qEHVI fails/crashes for m>5; qNParEGO degrades; we scale gracefully

2. **Sample efficiency comparison with NSGA-III**
   - Same DTLZ problems, show we achieve comparable Pareto coverage in 10–50× fewer evaluations
   - Critical for "why not just use evolutionary methods" argument

3. **Ablation: STCH-Set-BO vs single-point STCH-BO (Pires-style)**
   - Isolate the value of set-based coordination
   - Show that coordinated K solutions > K independent single-point solutions

### Tier 2: Strongly Recommended

4. **Sensitivity to K (number of solutions) and μ (smoothing parameter)**
   - K ∈ {2, 3, 5, 10}; μ ∈ {0.01, 0.1, 1.0}
   - Show robustness and provide practical guidelines

5. **Real-world application: Multi-ADMET drug discovery**
   - m = 20–50 ADMET endpoints, expensive surrogate (or real) evaluations
   - Find K=3 drug candidates covering all endpoints
   - Compelling narrative for practical impact

6. **Wall-clock time comparison**
   - Show acquisition optimization is fast (gradient-based via smooth scalarization)
   - Compare to qEHVI's exponential hypervolume computation

### Tier 3: Nice to Have

7. **Comparison with MORBO** on problems where both are applicable (m ≤ 4, high input dim)
   - Show we're competitive in their regime, superior when m grows

8. **Noisy objectives** — show robustness to observation noise

9. **Constraint handling** — show STCH-Set-BO extends to constrained problems

---

## 5. Key Papers to Cite (Priority Order)

### Must Cite
1. Lin et al. (ICML 2024) — STCH scalarization [our foundation]
2. Lin et al. (ICLR 2025) — STCH-Set [our foundation]
3. Pires & Coelho (SSRN 2025) — STCH in BO [closest prior work; single-point]
4. Astudillo & Frazier (ICML 2019) — Composite BO [framework we build on]
5. Daulton et al. (NeurIPS 2020, 2021) — qEHVI/qNEHVI [main baseline]
6. Daulton et al. (NeurIPS 2020) — qNParEGO [main baseline]
7. Knowles (2006) — ParEGO [originator]
8. Daulton et al. (ICML 2022) — MORBO [must discuss]

### Should Cite
9. Belakaria et al. (2019) — MESMO
10. Suzuki et al. (ICML 2020) — PFES
11. Deb & Jain (2014) — NSGA-III [evolutionary baseline]
12. Paria et al. (UAI 2019) — Random scalarization BO theory
13. Wang et al. (ICML 2024) — ε-PoHVI

### Consider Citing
14. Lyu et al. (ICML 2018) — Batch MOBO with acquisition ensembles
15. Tripp (2025 blog) — Pedagogical Chebyshev reference (probably not in academic paper)

---

## 6. Narrative Arc for Related Work Section

**Paragraph 1: Multi-objective BO landscape.** Hypervolume-based (qEHVI) vs scalarization-based (ParEGO) vs information-theoretic (MESMO, PFES). All limited to ~5 objectives.

**Paragraph 2: The many-objective gap.** MORBO scales input dims but not objectives. NSGA-III handles many objectives but needs 1000s of evaluations. No sample-efficient method for m >> 5.

**Paragraph 3: Smooth scalarization.** Lin et al. (2024) introduced STCH for differentiable MOO. Lin et al. (2025) extended to STCH-Set for many-objective "few-for-many." Both require gradient access.

**Paragraph 4: Scalarization meets BO.** Pires & Coelho (2025) first combined STCH with composite BO but only single-point. qNParEGO uses non-smooth Chebyshev with random weights. Neither coordinates multiple solutions.

**Paragraph 5: Our position.** We bridge STCH-Set to sample-efficient BO, enabling the first method that finds K coordinated solutions for m >> 5 objectives from expensive black-box evaluations.

---

## 7. Open Questions for Further Investigation

- [ ] Get Pires & Coelho (2025) paper — verify exactly what they did, what scale, what baselines
- [ ] Check if Lin et al. have any follow-up applying STCH/STCH-Set to BO themselves
- [ ] Look for any concurrent work (2025-2026) on many-objective BO
- [ ] Verify qNParEGO source code — does it use augmented Chebyshev or plain?
- [ ] Check if ScopeBO exists as a real method or if it's a misremembering
