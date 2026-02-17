# Literature Analysis for stch-botorch Publication Strategy

**Date:** 2026-02-17  
**Purpose:** Critical analysis of reference papers and competitive landscape for positioning stch-botorch at a top-tier venue.

---

## 1. Lin et al. (ICLR 2025) — "Few for Many: Tchebycheff Set Scalarization for Many-Objective Optimization"

**arXiv:2405.19650 | GitHub: Xi-L/STCH-Set**

### Key Contributions
1. **TCH-Set scalarization** (Eq. 8): `max_{1≤i≤m} { λ_i (min_{1≤k≤K} f_i(x^(k)) - z_i*) }` — finds K≪m solutions to collaboratively cover m objectives
2. **STCH-Set scalarization** (Eq. 12): Smooth approximation using log-sum-exp for both max (over objectives) and min (over solutions), enabling gradient-based optimization
3. **Theoretical guarantees:**
   - Theorem 1: Existence of Pareto-optimal solutions in optimal TCH-Set
   - Theorem 2: All solutions in STCH-Set optimal set are weakly Pareto optimal (Pareto optimal with positive preferences)
   - Theorem 3: Uniform smooth approximation (STCH-Set → TCH-Set as μ→0)
   - Theorem 4: Convergence to Pareto stationary solutions

### Experiments
- **Convex many-objective optimization:** m=128, m=1024 objectives, K=3-20 solutions, 50 independent runs with random convex quadratics
- **Noisy mixed linear regression:** 1000 data points from K ground-truth linear models (K=5,10,15,20; σ=0.1,0.5,1.0)
- **Noisy mixed nonlinear regression:** Same setup with neural networks as models
- **Baselines:** LS, TCH, STCH (random preferences), MosT (Li et al. 2024), SoM (Ding et al. 2024)
- **Metrics:** Worst objective value across all m objectives, average objective value

### What They Prove
- STCH-Set consistently achieves lowest worst-case objective values
- Smoothness is critical: STCH-Set significantly outperforms non-smooth TCH-Set in all experiments
- Traditional scalarization methods (LS/TCH/STCH with random preferences) fail at the few-for-many setting
- MosT cannot scale to 1024 objectives

### Critical Limitations / What's NOT in This Paper
1. **No Bayesian optimization / black-box setting:** All experiments assume gradient access to objectives. No GP surrogates, no acquisition functions, no sample-efficient sequential design
2. **No real-world applications:** Only synthetic benchmarks (convex quadratics, mixed regression). No molecules, materials, engineering, or drug discovery
3. **No noisy/stochastic black-box objectives:** The paper assumes deterministic, differentiable objectives
4. **No integration with BoTorch or any BO framework:** Pure gradient-descent optimization
5. **No comparison with hypervolume-based methods** (qEHVI, qNEHVI) that are standard in BO
6. **No DMTA/active-learning loop:** No sequential experimental design
7. **No chemistry/pharma benchmarks:** ZDT/DTLZ suites not even tested (they use custom synthetics)
8. **Self-acknowledged limitation:** "This work only focuses on the deterministic optimization setting... One potential future research direction is to investigate how to deal with only partially observable objective values in practice."

### Significance for stch-botorch
This paper provides the **theoretical foundation** but leaves the entire **Bayesian optimization application** wide open. The gap between theory (ICLR 2025) and practice (BO with GP surrogates for real-world problems) is exactly what stch-botorch fills.

---

## 2. Nature Computational Materials 2025 — "Active learning enables generation of molecules that advance the known Pareto front"

**DOI: 10.1038/s41524-025-01924-8**

### Key Contributions
1. Active learning loop coupling JANUS genetic algorithm + MPNN property predictors + DFT validation
2. Demonstrated that **active learning (iterative retraining) is more impactful than better generative models** — adding AL to JANUS outperformed using REINVENT (SOTA generator) without AL
3. Stability-Prediction MPNN: 3.5× improvement in stable molecule generation
4. Precision for identifying top molecules improved from 7% → 86% with 3 AL iterations

### Methods
- **Generator:** JANUS genetic algorithm (exploration + exploitation populations)
- **Surrogate:** Chemprop MPNN for density (ρ) and heat of formation (ΔH_f,s)
- **Validation:** High-throughput DFT pipeline (NWChem + AiiDA)
- **Scoring:** Multi-property score = z_ΔH + z_ρ (sum of z-scores, sigmoid-capped)
- **Active learning:** 4 iterations; iterations 1-3 use diversity sampling in property space; iteration 4 selects top-500 by MPNN prediction
- **Pareto front:** 2-objective only (density vs. heat of formation)

### Results
- Without AL: No generative model advanced the Pareto front of training data
- With AL: Generated molecules with density >2 g/cc (exceeding best in 10k training set)
- MPNN prediction RMSE reduced: ΔH_f,s by 83%, ρ by 75% after 3 iterations
- Key insight: Property prediction models fail catastrophically on out-of-distribution molecules

### Critical Limitations
1. **Only 2 objectives** — not "many-objective" at all
2. **Simple scoring function** — linear sum of z-scores, no Tchebycheff, no hypervolume
3. **No formal multi-objective acquisition function** — just genetic algorithm + sigmoid-capped sum
4. **DFT as oracle is expensive** — thousands of CPU-hours per iteration
5. **No comparison with BO methods** (qEHVI, ParEGO, etc.)
6. **Domain-specific** (energetic materials: CHON molecules only)
7. **No theoretical guarantees** on Pareto optimality of generated solutions

### Comparison with Our DMTA Loop Approach
| Aspect | Nature Comp Mat 2025 | stch-botorch (proposed) |
|--------|---------------------|------------------------|
| Objectives | 2 | Many (5-100+) |
| Surrogate | MPNN (Chemprop) | GP (BoTorch) with STCH-Set acquisition |
| Generator | JANUS genetic algorithm | BO acquisition function optimization |
| Scalarization | Linear sum of z-scores | Smooth Tchebycheff set scalarization |
| AL strategy | Diversity sampling | Principled BO (EI/UCB with STCH) |
| Theory | None | Pareto optimality guarantees (Lin et al.) |
| Applications | Energetic materials (2 props) | Drug discovery (many ADMET objectives) |

**Key opportunity:** Their paper proves active learning matters, but their optimization methodology is naive. stch-botorch provides the principled multi-objective framework they lack.

---

## 3. ScopeBO — Machine Learning-Guided Scope Selection

**ChemRxiv 2025, DOI: 10.26434/chemrxiv-2025-r0sst | Roediger et al.**

### Key Contributions (from search results; PDF access blocked)
1. ML-guided substrate scope selection using Bayesian optimization
2. Balances performance metrics (yield, selectivity) with information content/diversity
3. Quantitative, objective approach replacing subjective scope selection in chemistry papers
4. Validated on multiple reaction datasets (likely Doyle/Sigman-type HTE data)

### Relevance to stch-botorch
- **Metrics:** ScopeBO optimizes for both performance AND diversity — this is inherently multi-objective
- **Benchmark potential:** Their datasets (substrate scope with yield/selectivity/diversity) could serve as chemistry benchmarks for stch-botorch
- **Philosophical alignment:** Both tools address the "few solutions for many objectives" problem — ScopeBO selects a small set of substrates to represent a broad scope, analogous to STCH-Set finding K solutions for m objectives
- **Limitation of ScopeBO:** Likely uses simple BO (single-objective) with a composite score; stch-botorch could offer a more principled multi-objective treatment

### Potential Collaboration/Benchmark
Could frame: "ScopeBO selects substrates; stch-botorch selects which reaction conditions to test across those substrates, optimizing multiple performance criteria simultaneously."

---

## 4. bPK Score — Beyond PK Score for Small-Molecule Developability

**J. Med. Chem. 2023, DOI: 10.1021/acs.jmedchem.3c01083 | Novartis**

### Key Contributions
1. Deep learning model aggregating ~100 ADMET/PK assay predictions into a single developability score
2. Trained via **rank-consistent ordinal regression** on preclinical development milestones
3. Uses MELLODDY consortium in silico predictions as input features
4. Outperforms QED, SAScore on discriminating developable vs. non-developable compounds
5. Open-source: GitHub (Novartis/beyond-PK-score)

### Methodology
- **Input:** ~100 predicted ADMET endpoints (solubility, permeability, clearance, hERG, CYP inhibition, etc.)
- **Architecture:** PyTorch neural network with ordinal regression loss
- **Output:** Single scalar score (0-1), higher = better developability
- **Training signal:** Historical milestone progression data (reached candidate? clinical? market?)
- **Interpretability:** Shapley-value analysis for endpoint importance

### Relevance to stch-botorch
This is a **critical application paper** for us:

1. **bPK as an objective:** The bPK score itself could be ONE objective in a multi-objective optimization, alongside potency, selectivity, novelty
2. **bPK's ~100 ADMET inputs as SEPARATE objectives:** Instead of collapsing 100 ADMET predictions into one score, stch-botorch could optimize them directly as a many-objective problem (K=5 compounds covering 100 ADMET criteria)
3. **This is THE motivating use case:** Drug discovery needs to find a small set of candidate molecules (K=3-5 clinical candidates) that collectively cover all ADMET/efficacy/safety objectives
4. **Benchmark:** Use bPK's 100 ADMET predictors as the 100 objectives, optimize over a molecular library

### Critical Insight
The bPK paper essentially **admits the problem we solve**: they compress 100 objectives into 1 score because existing tools can't handle 100 objectives simultaneously. STCH-Set scalarization removes this need for compression by directly finding K solutions that collaboratively cover all 100 objectives. This is a much stronger argument for drug discovery: instead of one "best average" molecule, find 5 complementary candidates.

---

## 5. Related Work: Complete Landscape

### 5.1 Smooth Tchebycheff Scalarization (Single-Solution)
- **Lin & Zhang (ICML 2024):** "Smooth Tchebycheff Scalarization for Multi-Objective Optimization" (arXiv:2402.19078) — The single-solution precursor to STCH-Set. Proves smooth log-sum-exp approximation achieves O(1/ε) convergence vs O(1/ε²) for non-smooth TCH. **Must cite.**
- **Qiu et al. (2024):** Theoretical advantages of STCH over TCH for multi-objective reinforcement learning. **Must cite.**

### 5.2 Composite Bayesian Optimization with Smooth Tchebycheff
- **Pires & Coelho (SSRN 2025):** "Composite Bayesian Optimisation for Multi-Objective Materials Design" — Combines smooth Tchebycheff with composite BO exploiting nested objective structure. **DIRECT COMPETITOR. Must cite and differentiate.** Key difference: they do single-solution STCH, not STCH-Set; likely no many-objective experiments.

### 5.3 Multi-Objective Bayesian Optimization (BoTorch ecosystem)
- **qEHVI / qNEHVI** (Daulton et al., NeurIPS 2020, 2021): Expected hypervolume improvement — standard MOBO baseline. Scales poorly beyond 4-5 objectives.
- **qNParEGO** (BoTorch): Chebyshev scalarization + noisy EI. Random weight sampling limits efficiency.
- **MORBO** (Daulton et al., NeurIPS 2022): Trust-region MOBO for high-dimensional problems.
- **MESMO / PESMO**: Entropy-based MOBO methods.

### 5.4 Many-Objective Optimization
- **MosT** (Li et al., 2024): Many-objective multi-solution transport. Bi-level optimization, uses MGDA. Cannot scale to 1024 objectives.
- **SoM** (Ding et al., 2024): Sum-of-minimum optimization, k-means++ generalization. No multi-objective guarantees.
- **NSGA-III** (Deb & Jain, 2013): Reference-point based EA. Standard evolutionary baseline for many-objective.
- **Dimensionality reduction approaches** (Deb & Saxena 2005; Brockhoff & Zitzler 2006): Reduce many objectives to few. Loses information.

### 5.5 Active Learning for Molecular Optimization
- **Gauche** (Griffiths et al., NeurIPS 2022): GP-based BO for molecules
- **Tartarus** (Nigam et al., 2022): Molecular optimization benchmark
- **Practical MOO for drug design** (Fromer & Coley, 2024): Review of multi-objective molecular optimization

### 5.6 Austin Tripp Blog Post (2025)
- Excellent tutorial on Chebyshev scalarization properties: https://austintripp.ca/blog/2025-05-12-chebyshev-scalarization/
- Discusses augmented Chebyshev, proofs of Pareto coverage. **Good pedagogical reference.**

---

## 6. Gap Analysis: What stch-botorch Uniquely Contributes

### Gap 1: STCH-Set in Bayesian Optimization (PRIMARY CONTRIBUTION)
Lin et al. (ICLR 2025) provide theory for gradient-based optimization only. **Nobody has integrated STCH-Set scalarization into a BO acquisition function with GP surrogates.** This is the core contribution.

### Gap 2: Black-Box Many-Objective Optimization
Existing MOBO methods (qEHVI, ParEGO) degrade beyond ~5 objectives. STCH-Set is designed for 100+ objectives with K≪m solutions. **No existing BO method handles this regime.**

### Gap 3: Real-World Chemistry/Pharma Benchmarks for Many-Objective BO
- Lin et al. use only synthetic benchmarks
- Nature Comp Mat paper uses only 2 objectives
- No published work demonstrates many-objective BO on realistic ADMET/drug discovery problems

### Gap 4: Principled "Few Candidates for Many Criteria" in Drug Discovery
The bPK paper compresses 100 ADMET criteria into 1 score. ScopeBO selects diverse substrates heuristically. **No principled method exists for: "find K=5 drug candidates that collectively cover 100 ADMET requirements."**

### Gap 5: Sequential Sample-Efficient Many-Objective Optimization
All Lin et al. experiments assume unlimited gradient evaluations. Real chemistry has expensive black-box evaluations (DFT, wet-lab assays). **The sample-efficient BO formulation is entirely novel.**

---

## 7. Suggested Paper Framing for Highest-Tier Venue

### Target Venues (ranked)
1. **NeurIPS 2026** (Datasets & Benchmarks or main) — ML methods paper with drug discovery application
2. **ICML 2026** — Follow-up to Lin et al.'s ICML 2024 / ICLR 2025 lineage
3. **Nature Machine Intelligence** — If pharma benchmarks are strong enough
4. **JACS Au / J. Chem. Inf. Model.** — If chemistry angle dominates

### Recommended Title
"Smooth Tchebycheff Set Scalarization for Sample-Efficient Many-Objective Bayesian Optimization with Applications to Drug Discovery"

### Framing
**Problem:** Drug discovery requires optimizing K=3-5 candidate molecules across m=50-100+ ADMET/efficacy/safety criteria. Existing MOBO methods fail beyond ~5 objectives; existing many-objective methods (STCH-Set) require gradient access.

**Solution:** We integrate STCH-Set scalarization into BoTorch as a novel acquisition function for sample-efficient, black-box, many-objective Bayesian optimization. We prove the acquisition function inherits Pareto optimality guarantees from STCH-Set theory.

**Validation:** (1) Standard MO benchmarks (ZDT, DTLZ) showing we match/beat qEHVI for 2-5 objectives; (2) Many-objective benchmarks (m=20,50,100) where qEHVI/ParEGO fail; (3) Drug discovery case study using bPK's 100 ADMET endpoints as objectives.

---

## 8. Draft Related Work Section

> **Multi-Objective Bayesian Optimization.** Bayesian optimization (BO) extends to multiple objectives through hypervolume-based acquisition functions such as expected hypervolume improvement (qEHVI) [Daulton et al., 2020] and its noisy variant qNEHVI [Daulton et al., 2021]. Scalarization-based approaches, notably ParEGO [Knowles, 2006] which uses augmented Chebyshev scalarization with random weights, offer computational simplicity but lack principled weight selection. Trust-region methods like MORBO [Daulton et al., 2022] address high-dimensional input spaces. However, all these methods scale poorly beyond approximately 5 objectives, as the hypervolume computation is #P-hard in the number of objectives [Bringmann & Friedrich, 2010], and random scalarization becomes increasingly inefficient in high-dimensional objective spaces.
>
> **Tchebycheff Scalarization.** The Tchebycheff (Chebyshev) scalarization [Bowman, 1976; Steuer & Choo, 1983] is well-known for its ability to recover all Pareto-optimal solutions, including those on non-convex regions of the Pareto front, unlike linear scalarization. Lin & Zhang [2024] introduced smooth Tchebycheff (STCH) scalarization using log-sum-exp smoothing, achieving O(1/ε) convergence for gradient-based multi-objective optimization. Building on this, Lin et al. [2025] proposed Tchebycheff set (TCH-Set) and smooth Tchebycheff set (STCH-Set) scalarization for the "few-for-many" setting, finding K≪m solutions to collaboratively cover m objectives. Pires & Coelho [2025] explore composite Bayesian optimization with smooth Tchebycheff for materials design, but consider only the single-solution case. Our work is the first to integrate STCH-Set scalarization into Bayesian optimization for sample-efficient many-objective black-box optimization.
>
> **Many-Objective Optimization.** Problems with more than 3 objectives are termed "many-objective" [Fleming et al., 2005]. Evolutionary approaches include NSGA-III [Deb & Jain, 2013] and reference-vector guided methods [Cheng et al., 2016]. Recent work addresses the complementary "few-for-many" problem: MosT [Li et al., 2024] uses bi-level optimization with MGDA, while SoM [Ding et al., 2024] generalizes k-means++ for sum-of-minimum objectives. These methods assume gradient access and unlimited evaluations, making them unsuitable for expensive black-box settings in chemistry and drug discovery.
>
> **Multi-Objective Molecular Optimization.** Active learning coupled with molecular generation has shown promise for advancing Pareto fronts [Nature Comp. Mat., 2025], though limited to 2 objectives with naive scoring. The bPK score [Novartis, J. Med. Chem. 2023] aggregates ~100 ADMET predictions into a single developability metric, implicitly acknowledging the challenge of many-objective optimization in drug discovery. ScopeBO [Roediger et al., 2025] optimizes substrate scope selection balancing performance and diversity. Our work provides a principled framework for directly optimizing many objectives without lossy compression.

---

## 9. Recommended Experiments

### Tier 1: Mandatory (Reviewers Will Reject Without These)

1. **ZDT Suite (m=2):** ZDT1-4, ZDT6. Compare stch-botorch (K=1) vs qEHVI, qNParEGO, pymoo NSGA-II. Show competitive performance. Metrics: Hypervolume, IGD, Spacing.

2. **DTLZ Suite (m=3,5,10):** DTLZ1-7. This is where STCH-Set should start winning. Show K=2-5 solutions covering m=5-10 objectives better than qEHVI (which degrades).

3. **Scalability experiment:** m={10, 20, 50, 100} objectives on synthetic problems. Plot hypervolume / worst-objective-value vs. number of evaluations. Show qEHVI fails; STCH-Set-BO succeeds.

4. **Ablation study:**
   - STCH-Set-BO vs TCH-Set-BO (smooth vs non-smooth)
   - STCH-Set-BO vs ParEGO with STCH (single-solution scalarization, K=1 at a time)
   - Effect of K (number of solutions): K=1,3,5,10
   - Effect of μ (smoothing parameter)

### Tier 2: Strong Differentiators

5. **Drug discovery benchmark using bPK's ADMET endpoints:**
   - Use public ADMET predictors as objectives (20-100 endpoints)
   - Optimize over molecular fingerprint space or latent space
   - Find K=5 candidate molecules covering all ADMET criteria
   - Compare vs single-objective optimization of bPK composite score

6. **Active learning / DMTA loop simulation:**
   - Simulate sequential design where each "experiment" reveals true objective values
   - Show sample efficiency: STCH-Set-BO needs fewer evaluations than alternatives
   - Compare with Nature Comp Mat 2025's AL approach on their density/ΔH_f data

7. **Wall-clock time comparison:** Show BoTorch + STCH-Set is practical (vs qEHVI which becomes intractable for m>10)

### Tier 3: Impact Amplifiers

8. **Real chemistry case study:** Partner with an experimentalist for wet-lab validation on ≥3 objectives (yield, ee, cost, green chemistry metrics)

9. **Comparison with dimensionality reduction:** Show STCH-Set-BO outperforms "reduce 100 objectives to 5 via PCA, then run qEHVI"

10. **Robustness to noisy objectives:** Add noise to synthetic benchmarks; show GP-based STCH-Set-BO handles noise gracefully

---

## 10. Summary & Strategic Recommendations

### The Core Story
Lin et al. (ICLR 2025) proved that STCH-Set scalarization solves the "few-for-many" problem with theoretical guarantees — but only for gradient-based optimization. **stch-botorch bridges this to Bayesian optimization**, enabling sample-efficient many-objective optimization for expensive black-box problems like drug discovery, where you need K=5 candidates covering 100 ADMET criteria.

### Positioning
- **Not a methods-only paper** (would be incremental over Lin et al.)
- **Not a drug-discovery-only paper** (too narrow for ML venues)
- **A bridge paper:** Theory (STCH-Set) → Practice (BoTorch BO) → Application (drug discovery)
- The "bridge" framing is strong because it's at the intersection of optimization theory, ML systems (BoTorch), and a high-impact application

### Key Differentiators from All Existing Work
1. First STCH-Set integration in any BO framework
2. First many-objective BO method that provably finds complementary solution sets
3. First application of principled many-objective optimization to ADMET/drug discovery
4. Open-source BoTorch integration (high adoption potential)

### Risk Assessment
- **Pires & Coelho (2025)** is the closest competitor — but they only do single-solution STCH, not STCH-Set
- **Lin et al. may extend their own work to BO** — speed matters; publish quickly
- **BoTorch team may add this natively** — pre-empt by contributing to BoTorch directly (PR strategy)

### Timeline Recommendation
- **Immediate:** Finish ZDT/DTLZ benchmarks (Tier 1 experiments)
- **2-4 weeks:** bPK/ADMET benchmark (Tier 2)
- **6-8 weeks:** Paper draft ready for NeurIPS 2026 deadline or workshop submission
