# Deep Read: Lin et al. STCH Papers for BO Integration

## Paper 1: "Smooth Tchebycheff Scalarization for Multi-Objective Optimization" (ICML 2024)
- **Authors**: Xi Lin, Xiaoyuan Zhang, Zhiyuan Yang, Fei Liu, Zhenkun Wang, Qingfu Zhang
- **Venue**: ICML 2024
- **Code**: https://github.com/Xi-L/STCH
- **arXiv**: 2402.19078

## Paper 2: "Few for Many: Tchebycheff Set Scalarization for Many-Objective Optimization" (ICLR 2025)
- **Authors**: Xi Lin, Yilu Liu, Xiaoyuan Zhang, Fei Liu, Zhenkun Wang, Qingfu Zhang
- **Venue**: ICLR 2025
- **Code**: https://github.com/Xi-L/STCH-Set
- **arXiv**: 2405.19650

---

# PART I: STCH (ICML 2024) — Single-Solution Smooth Tchebycheff

## 1. Core Formulation

### Classical Tchebycheff (TCH) Scalarization
```
g^{TCH}(x|λ) = max_{1≤i≤m} { λ_i (f_i(x) - z*_i) }
```
where λ ∈ Δ^{m-1} (simplex), z* ∈ ℝ^m is the ideal point (z*_i = min f_i(x) - ε, ε > 0 small).

### Smooth Tchebycheff (STCH) Scalarization (Eq. 9)
```
g^{STCH}_μ(x|λ) = μ log( Σ_{i=1}^{m} exp( λ_i(f_i(x) - z*_i) / μ ) )
```
where μ > 0 is the smoothing parameter.

This is the **log-sum-exp** (softmax) smooth approximation of the max operator.

## 2. ALL Theorems

### Theorem 2.3 (Classical TCH — Choo & Atkins 1983)
A feasible solution x ∈ X is weakly Pareto optimal for the original MOP if and only if there exists a valid preference vector λ such that x is an optimal solution of the TCH scalarization problem. Additionally, if the optimal solution is unique for given λ, it is Pareto optimal.

### Proposition 3.3 (Smooth Approximation)
STCH g^{STCH}_μ(x|λ) is a smoothing function of classical TCH g^{TCH}(x|λ) satisfying:
1. lim_{z→x, μ↓0} g_μ(z) = g(x), ∀x ∈ X
2. Lipschitz smoothness: L_{g_μ} = K + α/μ for constants K, α > 0 independent of μ

### Proposition 3.4 (Bounded Approximation) — KEY FOR IMPLEMENTATION
```
g^{STCH}_μ(x|λ) - μ·log(m) ≤ g^{TCH}(x|λ) ≤ g^{STCH}_μ(x|λ)
```
**The approximation gap is exactly μ·log(m).** This means:
- STCH always **overestimates** the true Tchebycheff value
- The gap is uniform over all x ∈ X
- For m=100 objectives and μ=0.1: gap = 0.1 × 4.6 = 0.46

### Lemma 3.5 (Convexity)
If all objectives f_i are convex, then STCH g^{STCH}_μ(x|λ) is convex for any valid μ and λ.

### Proposition 3.6 (Iteration Complexity)
If all objectives are convex, with properly chosen μ, we achieve ε-optimal solution to nonsmooth TCH within **O(1/ε)** iterations (vs O(1/ε²) for subgradient on TCH directly).

### Theorem 3.7 (Pareto Optimality of STCH Solution)
The optimal solution of STCH is **weakly Pareto optimal**. It is **Pareto optimal** if either:
1. All preference coefficients are positive (λ_i > 0 ∀i), OR
2. The optimal solution is unique.

### Theorem 3.8 (Ability to Find All Pareto Solutions) — CRITICAL
Under mild conditions (Assumption A.1), there exists μ* such that for any 0 < μ < μ*, **every** Pareto solution of the original MOP is an optimal solution of STCH with some valid λ.

**Implication**: Too large μ may miss some Pareto solutions. For building on this: use small μ.

### Corollary 3.9
If the Pareto front is **convex**, then for **any** μ, every Pareto solution can be found by STCH with some λ. (No upper bound on μ needed.)

### Theorem 3.10 (Convergence to Pareto Stationary Solution)
If ∇g^{STCH}_μ(x̂|λ) = 0, then x̂ is a Pareto stationary solution of the original MOP.

**Proof idea**: The gradient of STCH is a convex combination of objective gradients:
```
∇g^{STCH}_μ(x|λ) = Σ_{i=1}^{m} w_i ∇f_i(x)
```
where w_i = exp(λ_i(f_i(x)-z*_i)/μ) / Σ_j exp(λ_j(f_j(x)-z*_j)/μ) are softmax weights.

When this equals zero, it satisfies Pareto stationarity (Definition 2.4) with α_i = w_i.

## 3. Assumptions About Objectives
- **Differentiable**: Yes, explicitly required. "m differentiable objective functions"
- **Convex**: NOT required for Pareto optimality guarantees (Theorems 3.7, 3.10). Convexity only needed for iteration complexity (Prop 3.6) and Corollary 3.9.
- **Lipschitz**: The gradient Lipschitz constant of STCH is L_{g_μ} = K + α/μ. Individual objectives assumed to have Lipschitz gradients for smoothness analysis.
- **No black-box assumption**: Everything is gradient-based.

## 4. Smoothing Parameter μ Relationship
- **μ → 0**: STCH → TCH exactly (Prop 3.3). Can find all Pareto solutions.
- **μ large**: STCH approaches linear scalarization behavior. May miss non-convex Pareto front regions.
- **μ too small**: Lipschitz constant L = K + α/μ → ∞, requiring smaller step sizes, slower convergence.
- **Practical values tested**: μ = 0.1 used as default in experiments.
- **Trade-off**: Small μ = better approximation but harder optimization. Large μ = easier optimization but less faithful to Tchebycheff.

## 5. Computational Complexity
| Method | Per-iteration | Notes |
|--------|--------------|-------|
| MGDA | O(m²n + QP(m)) | m backprops + QP solve |
| Linear Scalarization | O(mn) | Single backprop with weighted sum |
| TCH (subgradient) | O(mn) per iter, O(1/ε²) iters | Slow convergence |
| **STCH** | **O(mn)** per iter, **O(1/ε)** iters | Same per-iter as LS, much faster than TCH |

## 6. Experiments (ICML 2024)
- Synthetic MOO problems (2-3 objectives)
- Multi-task learning: NYUv2 (3 tasks), QM9 (11 tasks)
- Pareto set learning
- Fairness-accuracy trade-off
- Baselines: LS, TCH, EPO, MGDA, PMTL, COSMOS, and others
- Metrics: Hypervolume, distance to target preference, convergence speed

## 7. Limitations & Future Work (ICML 2024)
Not explicitly stated in a separate section. The paper focuses on deterministic, differentiable settings only.

## 8. BO/GP Mentions in ICML Paper
**NO.** The paper does not mention Bayesian optimization, Gaussian processes, or expensive/black-box optimization anywhere. It is purely about gradient-based optimization of differentiable objectives.

---

# PART II: STCH-Set (ICLR 2025) — Few Solutions for Many Objectives

## 1. Problem Formulation

### Set Optimization Problem (Eq. 6)
```
min_{X_K = {x^(k)}_{k=1}^K} f(x) = (min_{x∈X_K} f_1(x), min_{x∈X_K} f_2(x), ..., min_{x∈X_K} f_m(x))
```
Find K solutions to collectively optimize m objectives, where **1 < K ≪ m**.

### TCH-Set Scalarization (Eq. 8)
```
g^{TCH-Set}(X_K|λ) = max_{1≤i≤m} { λ_i (min_{1≤k≤K} f_i(x^(k)) - z*_i) }
```

### STCH-Set Scalarization (Eq. 12) — THE KEY EQUATION
```
g^{STCH-Set}_{μ,{μ_i}} (X_K|λ) = μ log( Σ_{i=1}^{m} exp( λ_i(smin_{1≤k≤K} f_i(x^(k)) - z*_i) / μ ) )
```
where:
```
smin_{1≤k≤K} f_i(x^(k)) = -μ_i log( Σ_{k=1}^{K} exp(-f_i(x^(k))/μ_i) )
```

**Full expanded form:**
```
g^{STCH-Set}_{μ,{μ_i}} (X_K|λ) = μ log( Σ_{i=1}^{m} exp( λ_i(-μ_i log(Σ_{k=1}^K exp(-f_i(x^(k))/μ_i)) - z*_i) / μ ) )
```

**Simplified form (same μ for all, Eq. 13):**
```
g^{STCH-Set}_μ (X_K|λ) = μ log( Σ_{i=1}^{m} exp( λ_i(-log(Σ_{k=1}^K exp(-f_i(x^(k))/μ)) - z*_i) ) )
```

**Notation:**
- X_K = {x^(1), ..., x^(K)}: set of K solutions
- m: number of objectives
- K: number of solutions (K ≪ m)
- λ = (λ_1, ..., λ_m) ∈ Δ^{m-1}: preference vector (uniform 1/m used in practice)
- z* = (z*_1, ..., z*_m): ideal point
- μ: smoothing parameter for smax (outer)
- μ_i: smoothing parameter for smin (inner, per-objective)
- smax: smooth max via log-sum-exp
- smin: smooth min via negative log-sum-exp of negatives

## 2. ALL Theorems (ICLR 2025)

### Assumption 1 (No Redundant Solution)
No solution in optimal X*_K is redundant:
```
g^{TCH-Set}(X*_K|λ) < g^{TCH-Set}(X*_K \ {x^(k)}|λ)
```
for any 1 ≤ k ≤ K ≤ m with λ > 0.

### Assumption 2 (All Positive Preference)
All preferences are strictly positive: λ_i > 0 ∀i, Σ λ_i = 1.

### Theorem 1 (Existence of Pareto Optimal Solution for TCH-Set)
There exists an optimal solution set X̄*_K for TCH-Set such that **all solutions in X̄*_K are Pareto optimal** of the original MOP. If the optimal set X*_K is unique, all solutions in X*_K are Pareto optimal.

**CAVEAT**: Without uniqueness, only an existence guarantee — some optimal sets may contain non-Pareto-optimal solutions! (Discussed in Appendix A.5)

### Theorem 2 (Pareto Optimality for STCH-Set)
**All** solutions in the optimal set X*_K for STCH-Set are **weakly Pareto optimal**. They are **Pareto optimal** if either:
1. The optimal set X*_K is unique, OR
2. All preference coefficients are positive (λ_i > 0 ∀i).

**This is STRONGER than Theorem 1 for TCH-Set!** STCH-Set guarantees all solutions are weakly Pareto optimal (not just existence).

### Theorem 3 (Uniform Smooth Approximation)
```
lim_{μ↓0, μ_i↓0 ∀i} g^{STCH-Set}_{μ,{μ_i}} (X_K|λ) = g^{TCH-Set}(X_K|λ)
```
for any valid set X_K ⊂ X.

### Theorem 4 (Convergence to Pareto Stationary Solution)
If ∇_{x^(k)} g^{STCH-Set}_{μ,{μ_i}} (X̂_K|λ) = 0 for all x^(k) ∈ X̂_K, then **all solutions in X̂_K are Pareto stationary** of the original MOP.

**When K=1, all theorems reduce to their STCH (ICML 2024) counterparts.**

## 3. Experimental Settings

### Values of K and m tested:

**Convex Many-Objective Optimization:**
- m = 128 objectives: K = {3, 4, 5, 6, 8, 10, 15, 20}
- m = 1,024 objectives: K = {3, 4, 5, 6, 8, 10, 15, 20}
- 50 independent runs each

**Noisy Mixed Linear Regression:**
- m = 1,000 data points (objectives)
- K = {5, 10, 15, 20}
- Noise levels σ = {0.1, 0.5, 1.0}
- 50 runs each

**Noisy Mixed Nonlinear Regression:**
- m = 1,000 data points
- K = {5, 10, 15, 20}
- Noise levels σ = {0.1, 0.5, 1.0}
- Neural network models

### Baselines
1. **LS** — Linear Scalarization with randomly sampled preferences
2. **TCH** — Tchebycheff Scalarization with randomly sampled preferences
3. **STCH** — Smooth Tchebycheff Scalarization with randomly sampled preferences
4. **MosT** — Many-objective Multi-solution Transport (Li et al., 2024) — bi-level optimization
5. **SoM** — Sum-of-Minimum optimization (Ding et al., 2024) — generalized k-means++/Lloyd's

### Metrics
- **Worst objective value**: max_{1≤i≤m} min_{x∈X_K} f_i(x)
- **Average objective value**: (1/m) Σ_{i=1}^m min_{x∈X_K} f_i(x)
- Wilcoxon rank-sum test at 0.05 significance level (+/=/−)

### Key Results
- **STCH-Set achieves lowest worst objective value in ALL comparisons**
- STCH-Set also achieves best/competitive average objective values
- Traditional methods (LS/TCH/STCH) fail at the few-for-many setting
- MosT better than traditional but worse than SoM and STCH-Set
- MosT cannot scale to 1024 objectives in reasonable time
- **STCH-Set significantly outperforms TCH-Set** on all comparisons (confirms importance of smoothness)
- STCH-Set outperforms SoM on average performance (explained in Appendix D.4)

## 4. Limitations Section (Verbatim)
> "This work proposes a general optimization method for multi-objective optimization which are not tied to particular applications. We do not see any specific potential societal impact of the proposed methods. This work only focuses on the deterministic optimization setting that all objectives are always available. One potential future research direction is to investigate how to deal with only partially observable objective values in practice."

## 5. Future Work Suggestions
1. **Partially observable objectives** — not all objectives available at once
2. (Implicit) Extension to stochastic settings
3. (Implicit) Application to specific domains

## 6. BO, GP, or Expensive Optimization Mentions
**NO.** Neither paper mentions:
- Bayesian optimization
- Gaussian processes
- Surrogate models
- Expensive/costly function evaluations
- Sample efficiency
- Black-box optimization (in the BO sense)

Both papers are **purely gradient-based**, assuming cheap function evaluations and differentiable objectives.

## 7. Assumptions About Objective Functions
- **Differentiable**: YES — explicitly required in both papers
- **Convex**: NOT required for Pareto optimality theorems; only for iteration complexity bounds
- **Lipschitz gradients**: Assumed implicitly for smoothness analysis
- **Continuous**: YES
- **Number of evaluations**: Unlimited (gradient-based, many iterations)
- **Deterministic**: YES — the limitation section explicitly states this

## 8. Relationship Between μ and Approximation Quality

### For smax (outer smoothing):
```
max{a_1,...,a_m} ≤ μ log(Σ exp(a_i/μ)) ≤ max{a_1,...,a_m} + μ log(m)
```

### For smin (inner smoothing):
```
min{b_1,...,b_K} - μ_i log(K) ≤ -μ_i log(Σ exp(-b_k/μ_i)) ≤ min{b_1,...,b_K}
```

### Combined for STCH-Set:
The total approximation error has contributions from both:
- Outer: gap ≤ μ·log(m)
- Inner: gap ≤ μ_i·log(K) per objective

### Practical implications for BO:
- With K=5 solutions and m=10 objectives: inner gap = μ·log(5) ≈ 1.6μ, outer gap = μ·log(10) ≈ 2.3μ
- With μ=0.1: total approximation within ~0.4 of true TCH-Set value
- **Smaller μ = tighter approximation but larger gradients (potentially numerical issues)**

## 9. How to Choose K Given m
**Not explicitly discussed in either paper.**

The papers test K from 3 to 20, with m from 128 to 1,024. The implicit guidance:
- K ≪ m (the "few for many" regime)
- K should be large enough that no solution is redundant (Assumption 1)
- When K ≥ m, problem degenerates to independent single-objective optimization
- The analogy to clustering (Ding et al., 2024): K is like number of clusters
- **No theoretical guidance on optimal K** — this is an open question

## 10. Computational Complexity

### STCH-Set per iteration:
- Evaluate all m objectives for all K solutions: O(K·m·cost_per_eval)
- Compute STCH-Set value and gradient: O(K·m) (log-sum-exp operations)
- Update all K solutions: O(K·n) where n is decision dimension

### vs MosT:
MosT requires solving bi-level optimization — much more expensive. Cannot handle m=1024 in reasonable time.

### vs MGDA:
MGDA requires O(m²n) per iteration for QP solve. STCH-Set only O(Km).

---

# PART III: CRITICAL ANALYSIS FOR BO INTEGRATION

## What Makes This Hard for BO

### 1. They assume cheap, differentiable evaluations
BO is for expensive, possibly noisy, black-box functions. The entire framework needs rethinking:
- Can't do 1000s of gradient descent iterations
- Need surrogate models (GPs)
- Need acquisition functions, not direct optimization

### 2. The scalarization is designed for direct optimization
In BO, we'd use STCH-Set as an **acquisition function scalarization**, not as the objective itself. The workflow would be:
- Fit GP surrogates to each objective
- Use STCH-Set to scalarize the GP posteriors into an acquisition function
- Optimize the acquisition function to select the next K points

### 3. The smoothness helps AND hurts in BO
- **Helps**: Smooth acquisition functions are easier to optimize with L-BFGS-B
- **Hurts**: The log-sum-exp can have numerical overflow/underflow with GP posterior samples

### 4. The "set" aspect is novel for BO
Standard MOBO methods (like qEHVI, qParEGO) optimize batch points but not in this "few-for-many" collaborative sense. STCH-Set explicitly optimizes K solutions as a team.

### 5. Key adaptation needed: from deterministic to posterior
Instead of f_i(x), we have GP posterior μ_i(x) ± σ_i(x). Options:
- **Posterior mean**: Use μ_i(x) in place of f_i(x) — simple but ignores uncertainty
- **Thompson sampling**: Sample f_i ~ GP_i, then optimize STCH-Set on samples
- **UCB-style**: Use μ_i(x) - β·σ_i(x) as optimistic estimates
- **Expected improvement variant**: More complex, need to think about set EI

### 6. The uniform preference λ = (1/m, ..., 1/m) is natural for the "cover all objectives" goal
This aligns well with BO settings where we want to ensure no objective is neglected.

## What We Can Directly Use

1. **The STCH-Set formula (Eq. 12/13)** — as acquisition function scalarization
2. **Theorem 2** — guarantees Pareto optimality of solutions
3. **Theorem 4** — convergence guarantee when optimizing acquisition
4. **The bounded approximation** — to control how close we are to true Tchebycheff
5. **The softmax weight interpretation** — w_i tells us which objectives each solution is "responsible for"

## Open Questions for Our Implementation

1. How does μ interact with GP posterior uncertainty?
2. Should μ be adapted based on the GP's predictive variance?
3. Can we use the soft-assignment weights to guide exploration?
4. What's the right batch size K for typical MOBO problems (2-10 objectives)?
5. How to handle the ideal point z* when it's unknown (use GP posterior bounds)?
6. Numerical stability of nested log-sum-exp with GP samples?

---

*Document compiled 2026-02-17. Source: arXiv HTML versions of both papers.*
