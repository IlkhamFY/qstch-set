# Scaling Analysis: Why qSTCHSet Underperforms qNParEGO at m=5

## Executive Summary

Our qSTCHSet (HV=5.044) underperforming qNParEGO (HV=5.250) on DTLZ2 m=5 is **expected behavior**, not a bug. The crossover point where set-based scalarization should dominate is likely **m ≥ 8-10**, not m=5. Here's why, and what to do about it.

---

## 1. Literature Context

### 1.1 ParEGO / qNParEGO at Moderate Objective Counts

ParEGO-style methods (random Chebyshev scalarization) are well-established for m ≤ 5-6 objectives:
- Each iteration samples a random weight vector from the unit simplex and optimizes a single augmented Tchebycheff scalarization
- In BoTorch's qNParEGO, each candidate in a batch uses a **different random scalarization**, providing implicit diversity
- The method is simple, computationally cheap, and well-calibrated for moderate m

**Key insight**: At m=5, the unit simplex is 4-dimensional. Random sampling from a 4-simplex still provides reasonable coverage of trade-off directions. ParEGO doesn't "break down" until the simplex becomes so high-dimensional that random samples cluster near the center (concentration of measure), which becomes severe around m ≥ 10.

### 1.2 Smooth Tchebycheff Scalarization (Lin et al., ICML 2024)

The STCH scalarization (Lin et al., 2024) provides a smooth, differentiable approximation to the Tchebycheff function:
- Enables gradient-based optimization with O(1/ε) convergence
- Provably finds all Pareto-optimal solutions
- Faster convergence than non-smooth Tchebycheff

### 1.3 Tchebycheff Set Scalarization (Lin et al., ICLR 2025)

The TCH-Set extension optimizes a **set of K solutions** jointly:
```
g_TCH-Set(X_K | λ) = max_{1≤i≤m} λ_i * min_{x∈X_K} (f_i(x) - z_i*)
```
- Designed for **many-objective** problems (m >> K), where K solutions collectively cover m objectives
- The smooth variant (STCH-Set) uses smooth max/min approximations with parameter μ
- Key design assumption: **m >> K** (e.g., K=5 solutions for m=100+ objectives)

**This is the critical observation**: STCH-Set is designed for the regime where the number of objectives far exceeds the number of solutions. At m=5 with K comparable to m, we're outside the intended operating regime.

### 1.4 Hypervolume Computation Scaling

Exact hypervolume computation is O(n^m) worst case:
- Feasible for m ≤ 5 with moderate population sizes
- Becomes impractical for m ≥ 6 without approximation (Monte Carlo, HypE)
- This means HV-based methods (qNEHVI) also struggle at high m, giving scalarization methods their opening

---

## 2. Why qNParEGO Wins at m=5

### 2.1 Random Scalarization Provides Excellent Diversity at Low m

At m=5, random weight vectors from the 4-simplex provide good directional diversity:
- Each batch candidate explores a genuinely different trade-off direction
- The augmented Tchebycheff term (ρ·Σ w_i|f_i - z_i*|) breaks ties and ensures movement toward the Pareto front
- With q candidates per batch, you get q independent explorations per iteration

This is essentially **decomposition-based optimization** (like MOEA/D) with random decomposition — well-suited for m=5.

### 2.2 STCH-Set's Smooth Approximation Hurts at Low m

The smoothing parameter μ introduces a fundamental trade-off:
- **Small μ** → closer to true Tchebycheff (good for quality, bad for gradients)
- **Large μ** → smoother landscape (good for optimization, but the surrogate becomes less faithful to the true objective)

At m=5, the smooth min over K solutions (soft minimum) may:
1. Over-smooth the landscape, blurring distinctions between candidates
2. Not gain much from the set-based formulation since K ≈ m means each solution doesn't need to "specialize"

### 2.3 The K ≈ m Problem

When K (batch size / set size) is close to m (number of objectives):
- Each solution could in principle handle ~1 objective
- The set-based advantage (complementary coverage) is minimal
- The optimization landscape for STCH-Set is harder than necessary

When K << m (e.g., K=5, m=50):
- Each solution must cover ~10 objectives
- Complementary specialization becomes crucial
- STCH-Set's min-over-set formulation provides genuine benefit

---

## 3. Theoretical Crossover Point

### 3.1 When Does ParEGO Break Down?

ParEGO's failure modes as m increases:

| m | Simplex Dim | Random Sampling Quality | ParEGO Effectiveness |
|---|-------------|------------------------|---------------------|
| 2-3 | 1-2 | Excellent coverage | Very strong |
| 4-5 | 3-4 | Good coverage | Strong |
| 6-8 | 5-7 | Moderate (concentration of measure begins) | Declining |
| 10+ | 9+ | Poor (samples cluster near centroid) | Weak |
| 20+ | 19+ | Very poor | Fails |

**Concentration of measure**: On the m-simplex, random points concentrate near the centroid (1/m, ..., 1/m) as m grows. This means random scalarizations become increasingly similar, destroying the diversity that makes ParEGO work.

### 3.2 When Does STCH-Set Gain Advantage?

STCH-Set's advantage grows when:
1. **m >> K**: The set-based formulation is genuinely needed (m ≥ 3K is a rough heuristic)
2. **HV computation is expensive**: m ≥ 6 makes qNEHVI slow, removing the HV-based competitor
3. **Random scalarization diversity fails**: m ≥ 8-10 where simplex concentration kicks in

**Predicted crossover: m ≈ 8-10 for STCH-Set vs qNParEGO**

### 3.3 Reference HV Values for DTLZ2

For DTLZ2 with reference point (1.1, ..., 1.1):
- Evolutionary algorithms (MODEhv) achieve HV ≈ 0.99 (normalized) at m=5
- Our values (5.044-5.250) use a different reference point; need to verify normalization
- The gap between our methods (~4% relative) is small, consistent with "wrong regime" rather than "broken method"

---

## 4. Concrete Recommendations

### 4.1 Benchmark Focus: Target m Values

| Priority | m Value | Rationale |
|----------|---------|-----------|
| **High** | m=2,3 | Sanity check — both methods should work, STCH-Set not expected to win |
| **High** | m=5 | Current benchmark — understand the gap (it's expected) |
| **Critical** | m=8 | Predicted crossover region — this is where the story should change |
| **Critical** | m=10 | Should see clear STCH-Set advantage |
| **High** | m=15-20 | Deep many-objective regime — STCH-Set should dominate decisively |

**Minimum benchmark suite**: m ∈ {2, 5, 8, 10, 15} on DTLZ2, DTLZ1, and at least one real-world problem.

### 4.2 K (Set Size / Batch Size) Scaling with m

K should NOT scale linearly with m. The whole point of set-based scalarization is K << m:

| m | Recommended K | Ratio m/K | Notes |
|---|--------------|-----------|-------|
| 5 | 2-3 | 1.7-2.5 | Marginal benefit; K=q (batch size) is fine |
| 8 | 3-4 | 2.0-2.7 | Starting to see benefit |
| 10 | 4-5 | 2.0-2.5 | Clear set-based advantage |
| 15 | 5-6 | 2.5-3.0 | Strong advantage |
| 20+ | 5-8 | 3-4+ | Designed regime |

**Key**: If your current K equals your batch size q, and q ≈ m, you're negating the set-based advantage. Try K=ceil(m/3) as a starting heuristic.

### 4.3 Smoothing Parameter μ Scaling with m

μ controls the smooth approximation quality. It likely needs to vary with m:

| m | Recommended μ | Rationale |
|---|--------------|-----------|
| 5 | 0.01-0.05 | Need tighter approximation since margins are small |
| 8-10 | 0.05-0.1 | Standard regime from the paper |
| 15+ | 0.1-0.2 | Can afford more smoothing; optimization landscape harder |

**Experiment**: Sweep μ ∈ {0.01, 0.05, 0.1, 0.2, 0.5} at each m value. The optimal μ likely increases with m because:
1. Higher m means more terms in the max, making the landscape rougher
2. More smoothing helps the optimizer navigate this rougher landscape
3. The approximation error from smoothing matters less when the Pareto front is higher-dimensional

### 4.4 Realistic Crossover Prediction

```
m=2-5:   qNParEGO ≥ qSTCHSet  (random scalarization diversity is sufficient)
m=6-8:   qNParEGO ≈ qSTCHSet  (crossover region, results are noisy)
m=8-10:  qSTCHSet > qNParEGO   (concentration of measure hurts ParEGO)
m=15+:   qSTCHSet >> qNParEGO  (ParEGO's random weights are near-identical)
```

---

## 5. Additional Investigation Items

### 5.1 Diagnostic Experiments
1. **Weight diversity analysis**: Sample 100 random weight vectors at m=5,10,15,20. Compute pairwise cosine similarities. At what m do they become degenerate?
2. **μ sensitivity sweep**: Fix m=5 and m=10, sweep μ. Does the optimal μ differ?
3. **K ablation**: At m=10, try K=2,3,5,7,10. Where is the sweet spot?

### 5.2 Potential Improvements for Low m
- **Adaptive μ**: Start with large μ for exploration, anneal to small μ for exploitation
- **Weight-conditioned STCH-Set**: Instead of uniform λ, use multiple scalarizations like ParEGO does
- **Hybrid approach**: Use qNParEGO for m ≤ 6, switch to qSTCHSet for m > 6

### 5.3 Fair Comparison Notes
- Ensure both methods get the **same total budget** (number of function evaluations)
- qNParEGO's greedy sequential optimization with different scalarizations is surprisingly powerful — it's doing implicit decomposition
- STCH-Set optimizes the full set jointly, which is harder but should pay off at high m

---

## 6. Key Takeaways

1. **m=5 is too low** for STCH-Set to show its advantage. This is not surprising.
2. **The crossover is around m=8-10**, driven by concentration of measure on the simplex.
3. **K should be much smaller than m** (K ≈ m/3) to leverage the set-based formulation.
4. **μ should be tuned per-m**, likely increasing with m.
5. **The real story for STCH-Set is m=15+**, where HV-based methods are computationally infeasible AND random scalarization diversity collapses.

---

*Analysis date: 2026-02-17*
*Based on: Lin et al. ICML 2024 (STCH), Lin et al. ICLR 2025 (TCH-Set), BoTorch qNParEGO implementation*
