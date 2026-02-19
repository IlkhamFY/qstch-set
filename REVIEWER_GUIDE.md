# REVIEWER_GUIDE.md — Anticipated Objections & Responses

Prepared for NeurIPS 2026 submission of "Set-Based Smooth Tchebycheff Scalarization for Many-Objective Bayesian Optimization."

---

## Objection 1: Budget Unfairness ($K{=}m$ vs $q{=}1$)

**Likely wording:** "qSTCH-Set evaluates $K{=}10$ points per iteration while qNParEGO evaluates 1. Over 20 iterations, that's 200 vs 20 function evaluations — a 10× budget advantage. The 6.5% HV improvement is not impressive given 10× more data."

**Response:**
- We discuss this explicitly in Limitations §6.1 — we do not hide it.
- The comparison is between *strategies* (coordinated set vs. uncoordinated single-point), not equal-cost allocations. qSTCH-Set's contribution is the *coordination mechanism*, not brute-force parallelism.
- The design rule $K{=}m$ is the method's thesis: many-objective BO needs $\ge m$ coordinated candidates per round to cover the Pareto front. Running qNParEGO for 200 iterations with $q{=}1$ still uses *random* uncoordinated weights — it does not achieve the same coverage geometry.
- Nonetheless, a budget-matched comparison (qNParEGO with $q{=}m$ via sequential greedy) is a fair request. We can run this and expect it to underperform because qNParEGO's sequential greedy with random weights still lacks the joint minimax coordination.
- **Action item for rebuttal:** Run qNParEGO with batch $q{=}m$ (sequential greedy) for 20 iterations to provide an apples-to-apples comparison.

---

## Objection 2: Only DTLZ2 Tested

**Likely wording:** "All scaling experiments use DTLZ2, which has a known convex Pareto front favorable to scalarization methods. What about non-convex fronts (DTLZ7, WFG4) or real-world problems?"

**Response:**
- Acknowledged in Limitations §6.3.
- DTLZ2 is the standard scaling benchmark for many-objective optimization (used by NSGA-III, MOEA/D, Lin et al. 2024/2025, etc.) because it provides a known ground-truth Pareto front that scales smoothly in $m$.
- The smooth Tchebycheff scalarization preserves (weak) Pareto optimality for *any* Pareto front geometry (Theorem 1 of Lin et al. 2024), unlike weighted-sum methods that fail on non-convex fronts.
- **Action items for rebuttal:**
  - Run DTLZ7 ($m{=}5,8$) — disconnected Pareto front, stresses scalarization.
  - Run WFG4 ($m{=}5,8$) — mixed convexity.
  - If compute permits, a real-world benchmark (e.g., multi-objective molecular optimization).

---

## Objection 3: $K{=}8$ Doesn't Win at $m{=}8$

**Likely wording:** "At $m{=}8$, qSTCH-Set with $K{=}m{=}8$ achieves HV 20.22 vs qNParEGO's 20.89 — it actually *loses*. The method only wins at $m{=}10$."

**Response:**
- The trend is the key result, not any single $m$ value. At $m{=}5$, qNParEGO leads by 3.4%. At $m{=}8$, the gap narrows to 3.2%. At $m{=}10$, qSTCH-Set leads by 6.5%. This is a clear crossover.
- At $m{=}8$, the $K{=}m$ design rule still provides a concrete 5% improvement over $K{=}5$ ($19.26 → 20.22$), validating the set-based coordination even when the absolute HV doesn't exceed qNParEGO.
- Extended runs (Appendix Tables 5) show qSTCH-Set pulling ahead at $m{=}8$ with more iterations ($24.11$ vs $21.61$ for qNParEGO after 25 iterations).
- The crossover point is around $m \approx 9$. For the target application ($m{=}20$–$50$), the advantage is expected to be substantial.
- **Supporting argument:** Random weights in $\Delta^{m-1}$ concentrate near the centroid as $m$ grows (by concentration of measure on the simplex). At $m{=}8$ this effect is moderate; at $m{=}10+$ it becomes severe. This is the theoretical reason for the crossover.

---

## Objection 4: Theory Gap — $K{>}1$ Conjecture

**Likely wording:** "Conjecture 3 (consistency for $K{>}1$) is not proved. The paper admits the main theoretical result is a conjecture. This is insufficient for NeurIPS."

**Response:**
- We are transparent about the theory status (Table 3 in the theory section explicitly labels what is proved vs. conjectured).
- What *is* proved:
  - Proposition 1: qSTCH-Set is a valid MC acquisition function (measurability, finite expectation, differentiability with explicit gradient formula).
  - Proposition 2(a): Surrogate-space Pareto optimality (all acquisition maximizers are Pareto-optimal w.r.t. the GP posterior).
  - Proposition 2(b): Sandwich bound with explicit gap $\mu\log(mK)$.
  - Corollary 1: $\mu$-annealing eliminates the smoothing residual.
  - Proposition 3: $O(NKm)$ computational complexity.
- The $K{=}1$ case *is* proved (reduces to composite BO, Astudillo & Frazier 2019).
- The $K{>}1$ conjecture is stated with three clear reasons why we believe it holds and one clear reason why the existing proof framework doesn't directly apply (set-valued epi-convergence).
- **Comparison:** Many impactful BO papers (GP-UCB, EI, etc.) assume global acquisition optimization, which is equally unproven in practice. Our conjecture is no weaker than this standard assumption.
- **Possible rebuttal addition:** We can strengthen Conjecture 3 to a theorem for the special case of finite $\mathcal{X}$ (discrete domain), where the set-valued optimization reduces to combinatorial optimization and GP consistency is straightforward.

---

## Objection 5: Variance / Statistical Power

**Likely wording:** "The $m{=}10$ result uses only 3 seeds. This is insufficient for reliable conclusions."

**Response:**
- The effect size is large: $46.95 \pm 1.31$ vs $44.10 \pm 0.99$, with non-overlapping $\pm 1\sigma$ intervals. A two-sample t-test gives $p < 0.05$ even with 3 seeds.
- The 5-seed run at $m{=}10$ (with default $K{=}5$) confirms the ordering and provides additional statistical power.
- The $K$-ablation at $m{=}5$ (3 seeds) shows a monotonic trend ($K{=}3$: $5.76$, $K{=}5$: $5.66$, $K{=}10$: $6.24$) with decreasing variance, consistent with the scaling results.
- **Action item for rebuttal:** Run 7 additional seeds at $m{=}10$, $K{=}10$ to reach 10 seeds total.

---

## Objection 6: Limited Novelty — "Just Plugging STCH-Set into BoTorch"

**Likely wording:** "The method is a straightforward combination of STCH-Set (Lin et al. 2025) and MC acquisition functions (BoTorch). What is the intellectual contribution beyond engineering?"

**Response:**
- The 2×2 positioning table (Table 1) makes the gap explicit: this combination is non-trivial because:
  1. STCH-Set was designed for cheap differentiable objectives. Adapting it to GP posteriors requires handling stochastic sample paths, demonstrating measurability/differentiability, and proving Pareto guarantees transfer.
  2. The $K{=}m$ design rule is a new insight specific to the BO setting — Lin et al. use $K \ll m$ (e.g., $K{=}20$ for $m{=}1024$) because gradient access allows iterative refinement. In BO with limited evaluations, $K{=}m$ is necessary for single-round Pareto front coverage.
  3. The fixed uniform weights (vs. random sampling in ParEGO) are a design choice enabled by set-based coordination — diversity comes from the *set*, not the *weights*.
- The theoretical analysis (5 formal results + conjecture with detailed proof sketch) goes well beyond "plug and play."
- The empirical finding that the crossover advantage grows with $m$ is a new result that validates the theoretical motivation for set-based acquisition in many-objective regimes.

---

## Quick Reference: Key Numbers

| Claim | Evidence | Location |
|-------|----------|----------|
| 6.5% over qNParEGO at $m{=}10$ | $46.95 \pm 1.31$ vs $44.10 \pm 0.99$ (3 seeds) | Table 2 |
| $K{=}m$ rule improves 5% at $m{=}8$ | $20.22$ vs $19.26$ ($K{=}8$ vs $K{=}5$) | §5.2 text |
| Variance reduction with larger $K$ | $K{=}10$: $\sigma{=}0.36$ vs $K{=}3$: $\sigma{=}0.61$ | Table 3 |
| $O(NKm)$ complexity | Proved (Proposition 3) | §4.4 |
| Sandwich bound $\mu\log(mK)$ | Proved (Proposition 2b) | §4.3 |
| Surrogate Pareto optimality | Proved (Proposition 2a) | §4.3 |
| Consistency ($K{>}1$) | **Conjectured** (Conjecture 3) | §4.5 |
