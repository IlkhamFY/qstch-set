# Pires & Coelho 2025 — Deep Dive: Competitive Threat Analysis

**Paper:** "Composite Bayesian Optimisation for Multi-Objective Problems with Smooth Tchebycheff Scalarisation"  
**SSRN:** [5168818](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5168818)  
**Status:** SSRN preprint (2025), likely submitted to a journal (Composite Structures or similar)  
**Authors:** Pires, Coelho, and colleagues (CM2S group, likely University of Porto)  
**Last updated:** 2026-02-17

---

## 1. What They Did (High-Level)

Combined **smooth Tchebycheff (STCH) scalarization** (Lin et al., ICML 2024) with **composite Bayesian optimization** (Astudillo & Frazier, ICML 2019) for multi-objective optimization, specifically targeting material/structural design problems.

**Core idea:** Instead of modeling f(x) = g(h(x)) as a single black-box, they:
1. Model h(x) with a multi-output GP (the expensive simulation outputs)
2. Apply STCH scalarization as the cheap outer function g(·)
3. Use composite EI (EI-CF) to exploit this structure for better sample efficiency

This converts MOBO into single-objective BO while:
- Preserving Pareto optimality (STCH can find all Pareto solutions, convex and non-convex)
- Exploiting composite structure for better acquisition function optimization

---

## 2. The Composite BO Framework (Astudillo & Frazier 2019)

**Paper:** "Bayesian Optimization of Composite Functions" — ICML 2019  
**Key insight:** When f(x) = g(h(x)) where h is expensive (black-box) and g is cheap (known):

- **Standard BO:** Models f directly with a GP → wastes information about structure
- **Composite BO:** Models h with multi-output GP, then propagates through g → non-Gaussian posterior on f, but much more informative

**Acquisition function:** EI-CF (Expected Improvement for Composite Functions)
- Cannot be computed in closed form (non-Gaussian posterior)
- Uses stochastic gradient estimator for optimization
- Asymptotically consistent
- Demonstrated orders-of-magnitude improvement over standard BO

**Why this matters for STCH:** STCH is a known, cheap, differentiable function applied to the objective vector. This is a textbook composite structure: f(x) = STCH(F(x), w) where F(x) is the expensive multi-output simulation.

---

## 3. The piglot Package

**Repository:** [github.com/CM2S/piglot](https://github.com/CM2S/piglot)  
**Published:** JOSS (Journal of Open Source Software)  
**License:** Open source  
**Core:** Python package for derivative-free optimization of numerical solver responses

### Key Features:
- **BO backend:** Built on **BoTorch** (confirmed from README)
- **Supports:** Bayesian optimization, genetic algorithms, other optimisers
- **Focus:** Inverse problems, material/structural design, simulation optimization
- **STCH implementation:** Yes — commits from July-August 2024 show:
  - STCH scalarization implementation (Jul 26, 2024)
  - Linear scalarization (Jul 30, 2024)
  - Numerical stability fixes for STCH (Jul 31, 2024)
  - Parametric smoothing studies (Aug 2, 2024)
- **Branch name:** `composite_design_mo_stch` — confirms MO + STCH + composite design integration
- **Composite objectives:** Explicitly mentioned in README ("supports optimising stochastic and composite objectives")

### ⚠️ CRITICAL: They already have a working BoTorch-based STCH+composite BO implementation in open-source code.

---

## 4. Answers to Critical Questions

### How many objectives did they test?
**UNKNOWN from public sources.** The full PDF is behind SSRN's Cloudflare wall. Based on their application domain (material/structural design), likely **m=2 to m=5**. The STCH paper (Lin et al.) itself tested up to 100+ objectives with STCH-Set, but Pires & Coelho's composite BO formulation likely focuses on lower m since EI-CF complexity grows with output dimension.

### Did they use STCH-Set or only single-point STCH?
**Most likely single-weight STCH** (one weight vector per BO run, ParEGO-style). The composite BO framework naturally works with a single scalarization. STCH-Set (finding a representative set) is a different paradigm that doesn't map cleanly to standard BO acquisition functions.

### What benchmarks?
**UNKNOWN from abstracts.** Likely:
- Analytical test functions (ZDT, DTLZ family)
- Material/structural design problems (their domain)
- The abstract mentions "benchmarking against traditional Bayesian optimisation"

### What acquisition function exactly?
**Almost certainly EI-CF** (Expected Improvement for Composite Functions) from Astudillo & Frazier 2019. This is the natural choice when exploiting composite structure. The piglot package is built on BoTorch which supports this.

### Did they compare against qEHVI or ParEGO?
**The abstract says "benchmarking against traditional Bayesian optimisation"** — this likely includes ParEGO (augmented Chebyshev + single-objective BO) and possibly qEHVI. Specifics unknown without full paper access.

### What are their stated limitations?
**UNKNOWN from abstracts.** Likely limitations include:
- Computational cost of the "full composite" variant (mentioned as higher cost)
- Scaling to many objectives (>5)
- Single-weight-vector approach requires multiple runs for full Pareto front
- GP modeling of high-dimensional outputs

---

## 5. What They Have vs. What We're Building

| Aspect | Pires & Coelho 2025 | Our STCH-BoTorch |
|--------|---------------------|-------------------|
| **Scalarization** | STCH (Lin et al. 2024) | STCH + STCH-Set |
| **BO framework** | Composite BO (EI-CF) | Direct integration with qEHVI, qParEGO, qNEHVI |
| **Implementation** | piglot (BoTorch-based) | Native BoTorch module |
| **Weight strategy** | Single weight per run (likely) | Adaptive weight selection, set-based |
| **Objective scale** | Likely m=2-5 | Target m=2 to m=50+ |
| **Application** | Material/structural design | General-purpose MOBO |
| **Pareto front** | Multiple single-weight runs | Direct Pareto set approximation |
| **Open source** | piglot (JOSS) | BoTorch PR/standalone |
| **Smoothing param** | Fixed μ studies | Adaptive μ scheduling |

---

## 6. Novelty Differentiation — What We Can Still Claim

### ✅ SAFE claims (they likely don't do this):
1. **STCH-Set scalarization in BO** — finding representative Pareto sets, not just single points
2. **Native BoTorch integration** — as a first-class acquisition function, not wrapped in piglot
3. **Many-objective scaling (m>10)** — their composite framework likely struggles here
4. **Adaptive smoothing parameter** — μ scheduling during optimization
5. **Batch/parallel STCH-BO** — q-batch formulations
6. **Comparison with qEHVI/qNEHVI** on standard MOBO benchmarks (if they didn't)

### ⚠️ OVERLAPPING claims (they may have this):
1. **STCH + BO combination** — they did this, period
2. **Composite structure exploitation** — they explicitly do this
3. **Better than augmented Chebyshev ParEGO** — they likely show this
4. **Sample efficiency gains** — they claim this

### ❌ CANNOT claim anymore:
1. ~~"First to combine STCH with Bayesian optimization"~~ — They beat us to it
2. ~~"Novel insight that STCH creates composite structure"~~ — Explicitly their paper's thesis

---

## 7. Other Related Work

### Lin, Zhang et al. (ICML 2024) — Original STCH Paper
- **Paper:** "Smooth Tchebycheff Scalarization for Multi-Objective Optimization"
- **arXiv:** [2402.19078](https://arxiv.org/abs/2402.19078)
- **Code:** [github.com/Xi-L/STCH](https://github.com/Xi-L/STCH)
- O(1/ε) convergence for convex objectives (vs O(1/ε²) for non-smooth)
- Covers gradient-based MOO, multi-task learning
- **STCH-Set variant** for 100+ objectives
- Does NOT do Bayesian optimization

### BoTorch Native Scalarizations
- `get_chebyshev_scalarization` exists in BoTorch (augmented Chebyshev, non-smooth)
- **No smooth Chebyshev / STCH in BoTorch** as of searches
- No open PRs/issues found for STCH integration

### Austin Tripp Blog Post (2025)
- Blog on Chebyshev scalarization in BO context
- URL: https://austintripp.ca/blog/2025-05-12-chebyshev-scalarization/
- (Future date — may be a preview or misdated)

---

## 8. Recommended Strategy

1. **MUST READ the full paper.** Try:
   - University library access to SSRN
   - Email the authors directly (common in academia)
   - Check if it appears on ResearchGate or author websites

2. **Differentiate aggressively on:**
   - STCH-Set integration (multi-weight, representative set)
   - Many-objective scaling (m=10, 20, 50)
   - Native BoTorch contribution (community impact)
   - Theoretical analysis of STCH smoothing in GP posterior context
   - Adaptive μ

3. **Cite them generously** — acknowledge concurrent work, position as complementary
   - "Pires & Coelho (2025) explore composite BO with STCH for material design; we extend this to the set-based setting and provide native BoTorch integration for general MOBO"

4. **Speed matters** — if they're on SSRN, they may be targeting a journal (slow). A workshop paper or arXiv preprint from us could establish priority on the differentiated contributions.

---

## 9. Open Questions (Need Full Paper Access)

- [ ] Exact acquisition function formulation
- [ ] Number of objectives tested
- [ ] Benchmark problems used
- [ ] Comparison baselines (qEHVI? qNParEGO? NSGA-II?)
- [ ] How they handle the non-Gaussian posterior from composite structure
- [ ] Smoothing parameter μ selection strategy
- [ ] Batch/parallel formulation?
- [ ] Theoretical guarantees?
- [ ] Runtime/computational cost analysis

---

*This analysis is based on publicly available information (SSRN abstract, GitHub, web searches). Full paper access is critical for complete competitive analysis.*
