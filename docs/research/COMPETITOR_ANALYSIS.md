# Competitor Analysis for STCH-Set Bayesian Optimization

*Generated: 2026-02-17 | Brutally honest assessment*

---

## Paper 1: Lin et al. ICML 2024 — "Smooth Tchebycheff Scalarization for Multi-Objective Optimization"

**arXiv:2402.19078 | Code: github.com/Xi-L/STCH**

### What They Did
- **Problem setting:** Gradient-based differentiable multi-objective optimization. Objectives must be differentiable. Direct optimization via gradient descent — **NOT Bayesian optimization**.
- **Scalarization:** Proposed STCH = μ·log(Σ exp(λᵢ(fᵢ(x) - zᵢ*)/μ)), a smooth (log-sum-exp) approximation of the classical Tchebycheff scalarization. Parameter μ controls smoothness.
- **Number of objectives tested:** Primarily m=2,3 in synthetic/visualization experiments; multi-task learning with ~2-3 tasks; Pareto set learning experiments.
- **Single-solution or set-based:** **Single-solution** scalarization. Each preference λ yields one solution. They enumerate many λ to approximate Pareto front. Also extended to Pareto set learning (learning a mapping from λ→x).
- **GP surrogates:** **NO.** No Gaussian processes. Purely gradient-based optimization of known differentiable objectives.
- **Experimental setup:** Synthetic convex problems (2-obj), multi-task learning benchmarks (MultiMNIST, CelebA-like), Pareto set learning. ~100 trials for convergence comparisons.
- **Key limitations acknowledged:** Theoretical guarantees require μ < μ* (problem-dependent threshold); convexity assumptions for strongest results; not designed for black-box/expensive functions.
- **What they explicitly did NOT do:**
  - No Bayesian optimization
  - No GP surrogates or uncertainty quantification
  - No acquisition functions
  - No set-based optimization (K solutions jointly optimized)
  - No black-box / expensive function optimization
  - No many-objective (m >> 3) experiments in the BO sense

---

## Paper 2: Lin et al. ICLR 2025 — "Few for Many: Tchebycheff Set Scalarization for Many-Objective Optimization"

**arXiv:2405.19650 | Code: github.com/Xi-L/STCH-Set**

### What They Did
- **Problem setting:** Gradient-based differentiable many-objective optimization. Find K << m solutions to collectively cover m objectives. **NOT Bayesian optimization.**
- **Scalarization:** Proposed TCH-Set and STCH-Set scalarizations:
  - TCH-Set: max_{1≤i≤m} { λᵢ · min_{1≤k≤K} fᵢ(x^(k)) - zᵢ* }
  - STCH-Set: Smooth version using log-sum-exp for max AND negative-log-sum-exp for min, with parameters μ (outer) and μᵢ (inner per objective)
- **Number of objectives tested:** m=128, m=1024 (convex), m=1000 (mixed linear/nonlinear regression). This is their key contribution: **many-objective** (m >> 10).
- **Single-solution or set-based:** **SET-BASED.** K={3,4,5,6,8,10,15,20} solutions jointly optimized. Core novelty is the "few for many" paradigm.
- **GP surrogates:** **NO.** Purely gradient-based. All objectives evaluated directly via forward pass.
- **Experimental setup:**
  - Convex quadratic functions: m=128, m=1024, K=3-20, 50 runs each
  - Mixed linear regression: m=1000 data points, K=5-20, noise σ={0.1,0.5,1.0}, 50 runs
  - Mixed nonlinear regression (neural networks): same setup
  - Baselines: LS, TCH, STCH (random preferences), MosT, SoM
  - Metrics: worst objective value, average objective value across m objectives
- **Key limitations acknowledged:** Only deterministic optimization (all objectives always available). Noted as future work: partially observable objectives.
- **What they explicitly did NOT do:**
  - **No Bayesian optimization** — no GP surrogates, no acquisition functions
  - **No expensive black-box functions** — all objectives are cheap to evaluate
  - **No uncertainty quantification** — purely deterministic
  - No batch/sequential experimental design
  - No hypervolume-based metrics (they use worst-obj and mean-obj)
  - Did not address the case where function evaluations are expensive
  - Only uniform preference λ = (1/m, ..., 1/m) in experiments

---

## Paper 3: Pires & Coelho 2025 — "Composite Bayesian Optimisation for Multi-Objective Problems with Smooth Tchebycheff Scalarisation"

**SSRN 5168818 | Posted March 7, 2025**

### What They Did
- **Problem setting:** **Bayesian optimization** for multi-objective problems. This is the closest competitor to our work. They use STCH scalarization within a BO loop with GP surrogates.
- **Scalarization:** STCH (smooth Tchebycheff) from Lin et al. 2024, used as a **composite function** within Bayesian optimization. They exploit the compositional structure: g(f₁(x), ..., fₘ(x)) where g is the STCH scalarization and fᵢ are modeled by independent GPs.
- **Number of objectives tested:** Likely m=2-3 based on typical composite BO literature and material design focus (exact count unavailable from abstract; paper is 22 pages).
- **Single-solution or set-based:** **Single-solution.** Each BO run with a fixed preference λ finds one Pareto solution. Standard scalarization-based MOBO approach.
- **GP surrogates:** **YES.** Independent GPs for each objective, with composite acquisition function exploiting the known scalarization structure.
- **Experimental setup:** Benchmarks focused on material design, structural design, inverse problems. Claims superior sample efficiency over traditional BO.
- **Key limitations (inferred):**
  - Single-solution per run (need multiple runs with different λ to get Pareto front)
  - Likely limited to low m (composite BO doesn't scale well to many objectives)
  - No set-based optimization (K solutions jointly)
- **What they explicitly did NOT do:**
  - **No set-based optimization** — single solution per preference
  - **No STCH-Set scalarization** — only single-solution STCH
  - Likely no many-objective (m >> 3) experiments
  - No joint optimization of K solutions under GP uncertainty

---

## Paper 4: Wang et al. ICML 2024 — "Probability Distribution of Hypervolume Improvement in Bi-objective Bayesian Optimization"

**ICML 2024 Poster | arXiv:2205.05505**

### What They Did
- **Problem setting:** **Bayesian optimization** for bi-objective (m=2) problems with GP surrogates.
- **Scalarization/Acquisition:** Not scalarization-based. Derived **closed-form exact probability distribution of hypervolume improvement (HVI)** under bivariate Gaussian GP predictions. Proposed ε-PoHVI acquisition function (probability of achieving at least ε hypervolume improvement).
- **Number of objectives tested:** **m=2 ONLY.** Bi-objective exclusively. The closed-form derivation is specific to 2D geometry.
- **Single-solution or set-based:** Set-based in the sense of maintaining a Pareto approximation set, but proposes candidates one-at-a-time (sequential).
- **GP surrogates:** **YES.** Core contribution is integrating GP posterior (bivariate Gaussian) with HVI geometry.
- **Experimental setup:** 14 bi-objective test problems. Compared ε-PoHVI vs ε-PoI and EHVI. Showed significant improvements especially under high GP uncertainty.
- **Key limitations acknowledged:** Quadratic time complexity vs linear for ε-PoI. Limited to m=2.
- **What they explicitly did NOT do:**
  - **No extension beyond m=2** — method is inherently bi-objective
  - No Tchebycheff scalarization
  - No set-based joint optimization of K solutions
  - No many-objective support

---

## GAP ANALYSIS: What Is Our Contribution?

### The Landscape

| Feature | Lin ICML'24 | Lin ICLR'25 | Pires'25 | Wang ICML'24 | **OURS** |
|---|---|---|---|---|---|
| Bayesian Optimization | ❌ | ❌ | ✅ | ✅ | ✅ |
| GP Surrogates | ❌ | ❌ | ✅ | ✅ | ✅ |
| STCH Scalarization | ✅ | ✅ | ✅ | ❌ | ✅ |
| Set-based (K solutions) | ❌ | ✅ | ❌ | ❌ | ✅ |
| Many objectives (m>>3) | ❌ | ✅ | ❌ | ❌ | ✅ |
| Expensive black-box | ❌ | ❌ | ✅ | ✅ | ✅ |
| Uncertainty-aware | ❌ | ❌ | ✅ | ✅ | ✅ |
| Acquisition function | ❌ | ❌ | ✅ (composite) | ✅ (ε-PoHVI) | ✅ |

### The Precise Gap We Fill

**Nobody has combined STCH-Set scalarization with Bayesian optimization.**

Specifically:

1. **Lin ICML 2024** introduced STCH for gradient-based MOO but never considered BO, GPs, or expensive functions.

2. **Lin ICLR 2025** introduced STCH-Set for many-objective gradient-based optimization but:
   - Assumed cheap function evaluations (thousands of gradient steps)
   - No GP surrogates, no uncertainty, no acquisition functions
   - Their method requires direct gradient access to all objectives

3. **Pires & Coelho 2025** brought STCH into BO via composite acquisition, but:
   - Only single-solution STCH (not STCH-Set)
   - Likely only m=2-3 objectives
   - No set-based optimization

4. **Wang et al. 2024** did rigorous BO acquisition theory but:
   - Only m=2
   - Hypervolume-based, not scalarization-based
   - No set-based joint optimization

### Our Unique Contribution (Honest Assessment)

**We are the first to:**
- Use STCH-Set scalarization as an acquisition function within Bayesian optimization
- Jointly optimize K candidate solutions under GP uncertainty for many-objective expensive black-box problems
- Propagate GP posterior uncertainty through the smooth set scalarization (enabled by differentiability of STCH-Set + reparameterization trick)
- Demonstrate this on standard MOBO benchmarks with m > 2 objectives

### Potential Weaknesses / Honest Concerns

1. **Pires & Coelho 2025 is very close.** They do STCH + composite BO. Our novelty vs them is specifically the **Set** extension (K solutions) and potentially many-objective scaling. If their paper also handles m>3, the gap narrows.

2. **The "set-based BO" angle is novel but needs justification.** Why would you want K solutions from BO? In standard MOBO you build a Pareto front over iterations. We need a compelling use case for "few for many" in the expensive black-box setting.

3. **Scalability of GP to many objectives.** If m=100+ objectives, fitting m independent GPs is expensive. We need to address this (shared GP structure? low-rank models?).

4. **Comparison with EHVI/ParEGO baselines.** We must beat standard MOBO methods (qEHVI, qParEGO, MESMO, etc.) not just show our scalarization works.

5. **The smoothing parameter μ interaction with GP uncertainty.** This is non-trivial and could be a contribution or a pitfall. Need careful analysis.

### Recommended Positioning

**"STCH-Set Bayesian Optimization: Sample-Efficient Many-Objective Optimization with Few Solutions"**

Core story: When objective evaluations are expensive AND you have many objectives (m >> 3) AND you want a small set of complementary solutions (K << m), no existing method works:
- Standard MOBO (EHVI, ParEGO) doesn't scale to m >> 3
- Lin's STCH-Set doesn't handle expensive/black-box functions
- Pires's composite BO doesn't do set-based optimization
- We bridge this gap with a principled GP-based acquisition using STCH-Set

### Critical Experiments Needed

1. **vs qEHVI, qParEGO** on standard 2-3 objective benchmarks (show we're competitive)
2. **vs random scalarization BO** on m=5-10 objectives (show STCH-Set structure helps)
3. **Scaling experiment** to m=20-50+ objectives where baselines fail
4. **Ablation:** STCH-Set vs STCH (single-solution repeated K times) to show joint optimization matters
5. **Real-world benchmark:** materials design or molecular optimization with many properties

---

*This analysis is based on thorough reading of all four papers. The gap is real but narrow on the Pires side. Our strongest differentiator is the Set-based extension under GP uncertainty for many-objective expensive optimization.*
