# Theory Section Review: What Is Proved, What Is Conjectured, What Gaps Remain

**File:** `paper/theory_section.tex`  
**Date:** 2026-02-18  
**Author:** Theory hardening subagent

---

## 1. Summary of Results

The theory section contains 3 propositions, 1 corollary, and 1 conjecture:

| # | Statement | Status | Proof Quality |
|---|-----------|--------|---------------|
| Proposition 1 | qSTCH-Set is a valid MC acquisition function (measurable, finite, differentiable) | **Proved** | Full proof; relies on standard composition arguments |
| Proposition 2(a) | STCH-Set optimizer is weakly Pareto optimal w.r.t. posterior means | **Proved (by transfer)** | Direct instantiation of Lin et al. Theorem 2 |
| Proposition 2(b) | Sandwich bound: gap ≤ μ log(m) + μ log(K) | **Proved** | Standard log-sum-exp bounds, self-contained |
| Corollary 1 | μ-annealing makes smoothing gap vanish | **Proved** | Immediate from Prop 2(b) |
| Proposition 3 | O(NKm) per-evaluation complexity | **Proved** | Direct computation, uncontroversial |
| Conjecture 1 | Consistency: qSTCH-Set BO → true Pareto front as t→∞ | **Conjecture** | Honest disclaimer with detailed gap analysis |

---

## 2. What Is Rigorously Proved

### Proposition 1: Validity as MC Acquisition Function
- **Measurability:** The integrand is a composition of measurable maps (affine reparameterization + smooth scalarization). Standard, no gaps.
- **Finite expectation:** Continuous GP sample paths on compact domain are a.s. bounded. Standard.
- **Differentiability:** log-sum-exp is C∞ with positive arguments. The gradient formula (Eq. 6-7) is derived by chain rule and verified against Lin et al. Theorem 4's gradient structure.

**Reviewer risk:** Very low. This is standard BoTorch-style MC acquisition function construction.

### Proposition 2(a): Surrogate Pareto Optimality
- **Method:** Observe that GP posterior means μ_i^(t) are smooth functions ℝ^d → ℝ, satisfying differentiability requirements of Lin et al. Since λ_i > 0 ∀i, Theorem 2 of Lin et al. (2025) applies directly.
- **What this gives:** Every solution in the optimal set X_K* is weakly Pareto optimal *w.r.t. the model*, not the true objectives.
- **Key limitation (stated explicitly):** This is a surrogate-space guarantee. It says "we're optimizing the right thing in model space" but doesn't bridge to true Pareto optimality without posterior convergence.

**Reviewer risk:** Low. The transfer argument is clean. A careful reviewer might ask about the uniqueness condition—we handle this by noting it requires λ > 0 (which we always use).

### Proposition 2(b): Sandwich Bound
- **Method:** Two independent log-sum-exp bounds composed:
  - Outer: max ≤ logsumexp ≤ max + μ log(m) — from Lin et al. 2024, Prop 3.4
  - Inner: min - μ log(K) ≤ smin ≤ min — standard
- **Composition:** The inner error of ≤ μ log(K) per objective passes through the outer scalarization. Since λ ∈ Δ^{m-1}, the total additive error is at most μ log(K).

**Reviewer risk:** Low. The bound is tight and well-understood. A reviewer might ask about the interaction between inner and outer smoothing when μ_inner ≠ μ_outer; our simplified form uses the same μ, which is what the code implements.

### Proposition 3: Complexity
- **Method:** Counting operations in the logsumexp computation.
- **Comparison:** qEHVI is #P-hard in m (well-cited), qNParEGO is O(Nm) but uncoordinated.

**Reviewer risk:** Zero. This is a counting argument.

---

## 3. What Is Conjectured (and Why)

### Conjecture 1: Consistency Under GP Posterior Convergence

**The claim:** As observations grow, qSTCH-Set solutions become ε_t-Pareto optimal for the true objectives, with ε_t → 0.

**What we CAN prove:**
1. GP posterior concentration (Srinivas et al. 2010): |f_i(x) - μ_i^(t)(x)| ≤ β_t^{1/2} σ_i^(t)(x) w.h.p.
2. STCH-Set is Lipschitz in objective values (follows from smoothness, but we acknowledge we haven't verified all corner cases for the composed bound).
3. For K=1, this reduces to composite BO (Astudillo & Frazier 2019), where consistency IS proved.

**What we CANNOT prove:**
1. **K > 1 extension:** The composite BO framework covers h(g(x)) where g is GP-modeled and h is known. For K > 1, the outer function depends on GP outputs at K *different* locations jointly. This is not covered by Astudillo-Frazier.
2. **Acquisition optimization near-global:** All BO theory assumes the acquisition is globally optimized. For K > 1, the search space is X^K (dimension Kd), making near-global optimization harder to guarantee.
3. **Set-valued convergence:** Showing minimizers of the approximate problem converge to minimizers of the true problem requires set-valued epi-convergence arguments. The continuous mapping / Berge's theorem approach works for single-point optimization but has subtleties for set-valued optimizers (permutation symmetry, possible non-uniqueness).

**Honest assessment:** We believe the conjecture is true based on:
- The K=1 case is proved
- GP concentration gives uniform convergence of the surrogate
- The STCH-Set scalarization is continuous, so limit arguments should go through
- Empirically, the method works

But a rigorous proof would be a separate theoretical contribution requiring ~5-10 pages of technical analysis.

---

## 4. Gaps and Potential Reviewer Objections

### Gap 1: No Regret Bound
We provide qualitative convergence (ε_t → 0) but NOT a regret bound of the form R_T = O(T^α) for specific α. Regret bounds exist for GP-UCB (Srinivas et al.), random scalarization BO (Paria et al. 2019), and composite BO (Astudillo-Frazier). We don't provide one for qSTCH-Set.

**Mitigation:** The paper's contribution is primarily algorithmic/empirical. State explicitly that regret analysis is future work.

### Gap 2: Acquisition Optimization Assumption
We assume L-BFGS-B with restarts finds a near-global optimum of the acquisition function. This is standard but worth acknowledging, especially because:
- The search space X^K has dimension Kd (e.g., K=5, d=10 → 50D optimization)
- The acquisition landscape may have many local optima due to the smooth-min's "soft assignment" structure

**Mitigation:** This assumption is universal in BO. Every practical method (qEHVI, qNParEGO) makes it. State it explicitly.

### Gap 3: K > 1 Consistency
The main theoretical gap. The conjecture is honest about this.

**Mitigation:** The honest disclaimer plus the K=1 reduction to known results is the strongest position. Don't oversell.

### Gap 4: Information-Directed Exploration
qSTCH-Set exploits the model (optimizes posterior mean/samples) but has no explicit exploration bonus (unlike GP-UCB). Exploration comes indirectly from posterior sampling (Thompson sampling effect). We don't analyze this.

**Mitigation:** Note that posterior sampling provides implicit exploration (as in Thompson sampling for BO), which is empirically effective.

### Gap 5: Ideal Point Estimation
We estimate z* from observed data. If z* is poorly estimated early on, the scalarization may emphasize wrong objectives. We don't analyze the effect of z* estimation error.

**Mitigation:** This is standard practice (ParEGO, qNParEGO all do this). Not a unique weakness.

---

## 5. Comparison to Theory in Competing Papers

| Paper | Theory Provided | Our Position |
|-------|----------------|-------------|
| qEHVI (Daulton et al. 2020/2021) | No convergence proofs; hypervolume monotonicity assumed | Comparable (we also lack full convergence proof) |
| ParEGO (Knowles 2006) | No theoretical analysis | We are stronger |
| Paria et al. 2019 (random scalarization) | Regret bounds for random scalarization BO | They have regret bounds; we have structure (set coordination) but no regret bounds |
| Astudillo-Frazier 2019 (composite BO) | Consistency proof for composite acquisition | We inherit this for K=1; K>1 is our gap |
| Lin et al. 2025 (STCH-Set) | Pareto optimality, convergence to stationary points | We transfer their results to BO; the transfer is clean |

**Assessment:** Our theoretical contribution is in the middle of the pack for a methods paper. The honest conjecture with clear gap analysis is stronger than claiming something we can't prove.

---

## 6. Recommendations for Paper Submission

1. **Keep the conjecture honest.** Reviewers respect intellectual honesty far more than overclaimed proofs. The gap analysis in Conjecture 1 shows we understand the limitations.

2. **Emphasize what IS proved.** Propositions 1-2 and Proposition 3 are clean, useful results. Prop 2(a) directly connects Lin et al. to BO—this IS the theoretical contribution.

3. **Don't claim "convergence proof."** In the abstract/intro, say "inherits Pareto optimality guarantees" (which is true via Prop 2) rather than "converges to the Pareto front" (which is conjectured).

4. **Consider adding:** A small numerical experiment verifying the sandwich bound (Prop 2b) on a simple problem. This would make the theory section more concrete.

5. **Future work framing:** "Establishing formal regret bounds for set-valued BO acquisition functions is an important open problem that our work motivates."

---

## 7. LaTeX Integration Notes

The theory section is in `paper/theory_section.tex`. To use it:

1. Add to preamble of `main.tex`:
```latex
\usepackage{amsthm}
\newtheorem{proposition}{Proposition}
\newtheorem{corollary}[proposition]{Corollary}
\newtheorem{conjecture}[proposition]{Conjecture}
\newtheorem{remark}[proposition]{Remark}
\theoremstyle{definition}
\newtheorem{definition}[proposition]{Definition}
\newtheorem{assumption}[proposition]{Assumption}
```

2. Replace the existing `\section{Theoretical Analysis}` in `main.tex` with:
```latex
\input{theory_section}
```

3. The section uses the following citation keys already in `references.bib`:
   - `lin2024smooth`, `lin2025few`, `astudillo2019composite`
   - `srinivas2010ucb`, `wilson2018maxvalue`, `rasmussen2006gp`
   - `balandat2020botorch`, `daulton2020qnehvi`, `daulton2021qnehvi`
   - `knowles2006parego`

All are present. No new bib entries needed.

---

*Review completed 2026-02-18.*
