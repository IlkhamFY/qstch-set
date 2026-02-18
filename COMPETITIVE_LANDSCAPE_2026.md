# Competitive Landscape: Smooth Tchebycheff Scalarization for Multi-Objective Bayesian Optimization

**Last Updated:** 2026-02-17  
**Purpose:** NeurIPS 2026 submission — comprehensive prior work mapping  
**Our Method:** Smooth Tchebycheff (STch) scalarization-based MOBO acquisition function in BoTorch

---

## Executive Summary

The MOBO field has seen explosive growth in 2023–2026. Key trends:
1. **Hypervolume methods** (qNEHVI) remain the dominant baseline but scale poorly beyond m≈4–5 objectives
2. **Scalarization revival** — our STch work is part of a broader trend recognizing scalarization's scalability advantages
3. **Batch diversity** methods (PDBO, DGEMO, MOBO-OSD) address Pareto front coverage
4. **Many-objective** (m>5) remains a largely unsolved frontier — our key opportunity
5. **Coverage-based** formulations (MOCOBO, MosT) reframe the problem entirely for large m

---

## Tier 1: Direct Competitors (Must Cite & Differentiate)

### 1. qNEHVI — Noisy Expected Hypervolume Improvement (Meta/BoTorch)
- **Paper:** Daulton et al., NeurIPS 2021 (+ ongoing BoTorch updates)
- **What:** q-batch noisy EHVI with candidate batch decomposition (CBD) for polynomial batch scaling
- **Max objectives tested:** 4 (standard benchmarks); recommended by Ax for production MOBO
- **Handles m>5:** No — hypervolume computation is exponential in m (O(n^{m/2}))
- **Key limitation:** Computational cost scales exponentially with number of objectives; reference point sensitivity
- **Our differentiation:** STch scales linearly in m via scalarization; no reference point needed; recovers non-convex fronts

### 2. MORBO — Multi-Objective Regionalized BO (Meta)
- **Paper:** Daulton et al., AISTATS 2022
- **What:** Local trust regions with coordinated exploration for high-dimensional input spaces (100s of parameters)
- **Max objectives tested:** 2–4 (optical display 146-dim, vehicle 222-dim)
- **Handles m>5:** Not demonstrated; focuses on high-d inputs, not many objectives
- **Key limitation:** Designed for high-d input spaces, not many objectives; trust region overhead
- **Our differentiation:** We address many-objective scaling (high m), complementary to MORBO's high-d input focus; could combine

### 3. ParEGO — Random Scalarization Baseline
- **Paper:** Knowles, 2006 (+ qNParEGO in BoTorch)
- **What:** Random Chebyshev scalarization with EI; foundation scalarization method
- **Max objectives tested:** 2–5 typically
- **Handles m>5:** In principle yes (scalarization scales), but random weights give poor coverage at high m
- **Key limitation:** Random weight sampling → biased coverage, misses extreme Pareto regions; non-smooth max → optimization issues
- **Our differentiation:** Smooth approximation enables gradient-based optimization; principled weight selection; theoretical guarantees for Pareto recovery

### 4. Lin et al. — Smooth Tchebycheff Scalarization (ICML 2024)
- **Paper:** Lin et al., "Smooth Tchebycheff Scalarization for Multi-Objective Optimization," ICML 2024
- **What:** Smoothed Tchebycheff via log-sum-exp; theoretical Pareto recovery guarantees
- **Max objectives tested:** Multi-task learning settings
- **Handles m>5:** Theoretically yes
- **Key limitation:** Focused on multi-task/multi-objective deep learning, NOT Bayesian optimization specifically
- **Our differentiation:** We bring STch into the BO acquisition function framework with GP surrogates; batch q-acquisition; BoTorch integration; empirical BO benchmarks

### 5. MOBO-OSD — Orthogonal Search Directions (NeurIPS 2025)
- **Paper:** Ngo & Ha, NeurIPS 2025
- **What:** Orthogonal search directions relative to convex hull of individual minima (CHIM); Pareto front estimation; batch support
- **Max objectives tested:** 2–6
- **Handles m>5:** Yes, tested up to 6
- **Key limitation:** Relies on CHIM geometry which may be expensive/unstable for many objectives; subproblem decomposition
- **Our differentiation:** STch is simpler (single smooth acquisition function, no decomposition into subproblems); more natural BoTorch integration

### 6. PDBO — Pareto-front Diverse Batch MOBO (AAAI 2024)
- **Paper:** AAAI 2024
- **What:** Multi-armed bandit for acquisition function selection + DPP for batch diversity
- **Max objectives tested:** 2–4 (ZDT, DTLZ benchmarks)
- **Handles m>5:** Not demonstrated
- **Key limitation:** Complex pipeline (MAB + DPP); limited objective scaling evidence
- **Our differentiation:** Simpler, principled scalarization approach; scales to many objectives

---

## Tier 2: Important Related Methods (Should Cite)

### 7. MESMO — Max-value Entropy Search for Multi-Objective (NeurIPS 2019)
- **What:** Information-theoretic; maximizes entropy reduction about Pareto front
- **Max objectives tested:** 2–4
- **Handles m>5:** No — Monte Carlo Pareto front sampling becomes intractable
- **Key limitation:** Computational cost of Pareto front sampling; approximation quality degrades with m
- **Our differentiation:** No need for Pareto front sampling; direct scalarized acquisition

### 8. USeMO — Uncertainty-Aware Search (2020/2022)
- **What:** Two-stage: cheap MOO on surrogates → uncertainty-based selection from Pareto set
- **Max objectives tested:** 2–4
- **Handles m>5:** Not demonstrated
- **Key limitation:** Cheap MOO inner loop quality; uncertainty hyper-rectangle scales poorly
- **Our differentiation:** Single-stage acquisition; better scaling

### 9. DGEMO — Diversity-Guided EMO (NeurIPS 2020)
- **What:** Batch MOBO with diversity via performance buffer graph-cut partitioning
- **Max objectives tested:** 2–3
- **Handles m>5:** No
- **Key limitation:** Diversity mechanism specific to low-m Pareto front geometry
- **Our differentiation:** Scales to many objectives; diversity via weight vector coverage

### 10. qPOTS — Pareto Optimal Thompson Sampling (2023–2024)
- **Paper:** Renganathan & Carlson, arXiv 2023; AISTATS 2025
- **What:** Thompson sampling on GP posteriors; selects points probable to be Pareto optimal; batch via maximin distance
- **Max objectives tested:** 2–4
- **Handles m>5:** Not demonstrated
- **Key limitation:** Evolutionary MOO solver needed for each TS sample; batch diversity from posterior samples only
- **Our differentiation:** Deterministic acquisition (no sampling variance); direct gradient optimization; scales to many objectives

---

## Tier 3: Adjacent / Application-Specific (Cite Where Relevant)

### 11. GFlowNet Multi-Objective (NeurIPS 2023, ICLR 2025)
- **Papers:** Jain et al. 2023 (HN-GFN); Kim et al. 2024 (Genetic-GFN)
- **What:** Generative flow networks sampling proportional to reward; preference-conditioned for Pareto front
- **Max objectives tested:** 2–3 (molecular: GSK3β + JNK3)
- **Handles m>5:** Not demonstrated
- **Key limitation:** Requires discrete/structured search spaces; amortized training cost; not sample-efficient for expensive objectives
- **Our differentiation:** BO is for expensive black-box functions; GFlowNets for cheap-to-evaluate structured spaces. Complementary for drug discovery applications.

### 12. MOCOBO — Multi-Objective Coverage BO (arXiv 2025)
- **Paper:** Maus et al., 2025
- **What:** Find K solutions that collectively "cover" T objectives (coverage score, not hypervolume)
- **Max objectives tested:** Many (T >> K)
- **Handles m>5:** Yes — designed for it (T can be large)
- **Key limitation:** Different problem formulation (coverage, not Pareto front); reduces to linearization for K=1
- **Our differentiation:** We optimize the full Pareto front, not coverage; applicable when user wants to understand all trade-offs

### 13. MosT — Many-objective Multi-Solution Transport (ICLR 2025)
- **Paper:** Li et al., ICLR 2025
- **What:** Optimal transport matching between solutions and objectives; bi-level optimization
- **Max objectives tested:** Many (federated learning, MTL settings)
- **Handles m>5:** Yes — designed for it
- **Key limitation:** Not a BO method (assumes cheap evaluations); no GP surrogates; gradient-based on known objectives
- **Our differentiation:** We are in the BO setting with expensive black-box evaluations and GP surrogates

### 14. A-GPS — Amortized Active Generation of Pareto Sets (2025)
- **Paper:** arXiv 2025
- **What:** Generative model of Pareto set conditioned on preferences; CPE for non-dominance; amortized inference
- **Max objectives tested:** Protein design benchmarks
- **Handles m>5:** Potentially (amortized)
- **Key limitation:** Requires training generative model; not standard BO loop
- **Our differentiation:** Standard BO with GP surrogates; no generative model training

### 15. BOPE — BO with Preference Exploration (NeurIPS 2025)
- **What:** Neural network ensemble as utility surrogate; learns from pairwise comparisons
- **Key limitation:** Requires human preference feedback
- **Our differentiation:** Fully automated; no human in the loop needed

### 16. PUB-MOBO — Preference Utility Bayesian MOBO (MERL, 2025)
- **What:** Personalized MOBO via unknown utility from pairwise comparisons
- **Key limitation:** Requires preference comparisons; focuses on single preferred solution, not full Pareto front
- **Our differentiation:** Full Pareto front recovery without preference elicitation

### 17. piglot — Porto Optimization Toolbox
- **What:** Open-source BO toolbox for material/structural design (University of Porto/INEGI)
- **Relevance:** Uses BO for engineering design; potential application domain
- **Key limitation:** Application-focused, not methodological contribution
- **Our differentiation:** Methodological contribution; piglot could use our acquisition function

---

## Tier 4: Foundational / Survey References

| Reference | Year | Role in Our Paper |
|-----------|------|-------------------|
| EHVI (Emmerich et al.) | 2006 | Classic hypervolume-based MOBO |
| PESMO (Hernández-Lobato et al.) | 2016 | Information-theoretic MOBO baseline |
| NSGA-II/III | 2002/2014 | Evolutionary MOO foundations |
| Chebyshev scalarization theory | Classical | Mathematical foundation for our smoothing |
| BoTorch framework (Balandat et al.) | 2020 | Implementation platform |
| SAASBO (Eriksson & Jankowiak) | 2021 | High-d BO with sparse priors |

---

## Scaling Comparison Matrix

| Method | Max m Tested | Theoretical m Scaling | Batch Support | Non-Convex Fronts | BoTorch Native |
|--------|:---:|:---:|:---:|:---:|:---:|
| **STch (Ours)** | **TBD (target: 10+)** | **O(m) — linear** | **Yes** | **Yes** | **Yes** |
| qNEHVI | 4 | O(n^{m/2}) — exponential | Yes | Yes | Yes |
| MORBO | 4 | Unclear for m | Yes | Yes | Yes |
| ParEGO | 5 | O(m) — linear | Yes | Partial | Yes |
| MOBO-OSD | 6 | O(m²) subproblems | Yes | Yes | No |
| PDBO | 4 | Unclear | Yes | Yes | No |
| qPOTS | 4 | O(m) per TS | Yes | Yes | No |
| MESMO | 4 | Intractable | No | Yes | No |
| USeMO | 4 | O(m) hyper-rect | No | Yes | No |
| DGEMO | 3 | Poor | Yes | Limited | No |
| MOCOBO | Many | O(K·m) | Yes | N/A (coverage) | No |

---

## Key Gaps We Fill

1. **Many-objective BO (m>5):** Almost no BO method has been tested beyond 6 objectives. Hypervolume methods fail computationally. We offer the first principled scalarization-based acquisition function that scales linearly.

2. **Smooth, gradient-friendly scalarization:** ParEGO's Chebyshev max is non-smooth → poor gradient-based optimization. Our log-sum-exp smoothing enables direct use with BoTorch's MC acquisition optimization.

3. **Non-convex Pareto front recovery:** Linear scalarization misses non-convex regions. Chebyshev captures them but is non-smooth. STch gets both: non-convex recovery + smoothness.

4. **BoTorch-native integration:** Most competitors (PDBO, MOBO-OSD, DGEMO, qPOTS) have standalone implementations. STch integrates natively with BoTorch/Ax ecosystem.

5. **No reference point needed:** Unlike EHVI/qNEHVI, our method doesn't require a reference point specification.

---

## Risk Assessment: Papers That Could Scoop Us

| Risk | Paper/Group | Mitigation |
|------|-------------|------------|
| **HIGH** | Lin et al. (ICML 2024) extending STch to BO directly | We differentiate via BO-specific contributions (GP integration, batch acquisition, benchmarks). Cite extensively. |
| **MEDIUM** | MOBO-OSD scaling to m>6 with better results | Our method is simpler and BoTorch-native. Run head-to-head comparisons. |
| **MEDIUM** | BoTorch team adding STch acquisition natively | Accelerate our BoTorch PR; become the implementation they adopt. |
| **LOW** | MOCOBO reframing makes Pareto front methods obsolete | Different problem; full Pareto front still valued in most applications. |

---

## Recommended Baselines for Experiments

### Must-include (reviewers will expect these):
1. **qNEHVI** — dominant BoTorch MOBO method
2. **qNParEGO** — scalarization baseline in BoTorch
3. **Random** — sanity check

### Strongly recommended:
4. **MOBO-OSD** — strongest recent NeurIPS competitor with m=6 results
5. **PDBO** — AAAI 2024 batch diversity method
6. **qPOTS** — Thompson sampling alternative

### If space permits:
7. **MESMO** — information-theoretic alternative
8. **USeMO** — uncertainty-based alternative
9. **DGEMO** — diversity-guided batch method

---

## Citation Clusters to Track

- **Lin et al. ICML 2024** — Search Google Scholar "Cited by" monthly until submission
- **BoTorch/Ax release notes** — Check for new MOBO acquisition functions
- **NeurIPS 2025 proceedings** — Full proceedings for any late MOBO papers (MOBO-OSD confirmed)
- **ICML 2026 submissions** — OpenReview in Feb–Mar 2026
- **ICLR 2026 papers** — Check for MosT/MOCOBO extensions to BO setting

---

*This document should be updated monthly until NeurIPS 2026 submission deadline.*
