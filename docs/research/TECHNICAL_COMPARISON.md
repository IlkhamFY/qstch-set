# Technical Comparison: STCH-Set for Bayesian Optimization

## What Exists vs What We Need to Build

---

## 1. Lin et al.'s STCH / STCH-Set Code

### STCH (ICML 2024) — `github.com/Xi-L/STCH`
- **Two modules**: `STCH_MTL/` (deep multi-task learning) and `STCH_PSL/` (Pareto set learning)
- **Context**: Gradient-based MOO. NOT Bayesian optimization.
- **Formula**: Single-solution smooth Tchebycheff:
  ```
  g_μ^STCH(x|λ) = μ·log(Σᵢ exp(λᵢ(fᵢ(x) - zᵢ*) / μ))
  ```
- **Applications**: Neural network training (MTL), Pareto set learning via hypernetworks
- **No BO component whatsoever**

### STCH-Set (ICLR 2025) — `github.com/Xi-L/STCH-Set`
- **The "Few for Many" paper** — finds K solutions for m >> K objectives
- **Formula** (Eq. 12-13 in paper):
  ```
  g_μ^{STCH-Set}(X_K|λ) = μ·log(Σᵢ exp(λᵢ(smin_k fᵢ(x^(k)) - zᵢ*) / μ))
  where smin_k fᵢ(x^(k)) = -μᵢ·log(Σ_k exp(-fᵢ(x^(k))/μᵢ))
  ```
- **Experiments**:
  - Convex MOO: m=128, m=1024 objectives; K={3,4,5,6,8,10,15,20} solutions
  - Mixed linear regression: m=1000 data points, K={5,10,15,20}
  - Mixed nonlinear regression: same setup with neural networks
  - Multi-task learning benchmarks
  - **50 independent runs** per comparison, Wilcoxon rank-sum test at 0.05
- **Baselines**: LS, TCH, STCH (random preferences), MosT, SoM
- **NO comparison against BO methods** — entirely gradient-based, assumes differentiable objectives
- **Limitations stated**: "only focuses on deterministic optimization... all objectives always available. Future: partially observable objectives"

### Key Theorems
- **Theorem 1**: TCH-Set optimal set contains Pareto optimal solutions (existence guarantee, weak)
- **Theorem 2** (STCH-Set): All solutions in optimal set are **weakly Pareto optimal**. They are **Pareto optimal** if either (a) optimal set is unique, or (b) all λᵢ > 0
- **Theorem 3**: STCH-Set uniformly approximates TCH-Set as μ → 0
- **Theorem 4**: Stationary points of STCH-Set are Pareto stationary for original MOO problem

---

## 2. BoTorch's ParEGO / Chebyshev Scalarization

### `get_chebyshev_scalarization()` — Exact Source Code
```python
def get_chebyshev_scalarization(weights, Y, alpha=0.05):
    """Augmented Chebyshev: g(y) = max_i(w_i * y_i) + alpha * Σ_i(w_i * y_i)"""
    
    def chebyshev_obj(Y, X=None):
        product = weights * Y
        return product.max(dim=-1).values + alpha * product.sum(dim=-1)
    
    # Normalizes Y to [0,1], flips sign (BoTorch maximizes), handles min objectives
    Y_bounds = torch.stack([Y.min(dim=-2).values, Y.max(dim=-2).values])
    
    def obj(Y, X=None):
        Y_normalized = normalize(-Y, bounds=Y_bounds)
        Y_normalized[..., minimize] = Y_normalized[..., minimize] - 1
        return -chebyshev_obj(Y=Y_normalized)
```

### Key Details
- **Augmented Chebyshev** (not smooth): `max(w·y) + α·Σ(w·y)` with α=0.05
- **Normalization**: Scales objectives to [0,1] using observed min/max bounds
- **Sign convention**: BoTorch maximizes; internally flips to minimize, computes Chebyshev, flips back
- **Minimization support**: Negative weights → objectives shifted to [-1,0]
- **Used in q-ParEGO**: Random weights sampled per candidate, scalarized qEI acquisition
- **Reference**: Knowles 2005 (ParEGO), Daulton 2020 (q-ParEGO)

### What's Missing in BoTorch
- No smooth Tchebycheff (log-sum-exp) scalarization
- No set scalarization (K > 1 solutions)
- No STCH or STCH-Set variants
- The augmented Chebyshev uses non-smooth `max` — gradients are subgradients through argmax

---

## 3. Pires & Coelho: Composite BO with Smooth Tchebycheff

### Paper: "Composite Bayesian Optimisation for Multi-Objective Problems with Smooth Tchebycheff Scalarisation"
- **SSRN preprint** (March 2025), not yet peer-reviewed
- **Tool**: `piglot` package (`github.com/CM2S/piglot`) — derivative-free optimization for computational mechanics
- **Key idea**: Combine STCH scalarization with **composite BO** (Astudillo & Frazier, ICML 2019)

### Composite BO (Astudillo & Frazier 2019)
- For f(x) = g(h(x)) where h is expensive vector-valued, g is cheap scalar
- Model h(x) with multi-output GP → implied non-Gaussian posterior on f
- Acquisition: **EI-CF** (Expected Improvement for Composite Functions)
- Key insight: exploiting known structure of g improves sample efficiency dramatically

### Pires & Coelho's Contribution
- Use STCH as the outer function g in composite BO
- g(y) = μ·log(Σ exp(λᵢ(yᵢ - zᵢ*)/μ)) — the smooth Tchebycheff
- Model each objective with GP, apply STCH scalarization as known composite
- **Single-solution only** (K=1) — standard STCH, not STCH-Set
- Benchmarks on material design problems via piglot
- Claims better sample efficiency than standard ParEGO

---

## 4. Gap Analysis: What Nobody Has Built

### Has anyone implemented STCH-Set in BoTorch? **NO.**
- GitHub search: zero results for STCH-Set + BoTorch integration
- No BO framework has STCH-Set
- Lin et al. don't mention BO at all — their work is gradient-based
- Pires & Coelho use single-solution STCH only, not the set version

### The Opportunity Matrix

| Component | Exists? | Where? | Gap |
|-----------|---------|--------|-----|
| STCH single-solution formula | ✅ | Lin ICML'24, Pires'25 | — |
| STCH-Set formula | ✅ | Lin ICLR'25 (theory + gradient code) | Not in BO |
| BoTorch Chebyshev scalarization | ✅ | `botorch.utils.multi_objective.scalarization` | Non-smooth, no set version |
| Composite BO framework | ✅ | BoTorch (Astudillo), piglot | Not combined with STCH-Set |
| **STCH as BoTorch scalarization** | ❌ | — | **Need to build** |
| **STCH-Set as BoTorch scalarization** | ❌ | — | **Need to build** |
| **STCH-Set + composite BO** | ❌ | — | **Need to build** |
| **STCH-Set + qEI/qNEI** | ❌ | — | **Need to build** |
| Benchmarks: STCH-Set vs ParEGO in BO | ❌ | — | **Need to run** |

---

## 5. What We Need to Build — Technical Spec

### Layer 1: `get_smooth_chebyshev_scalarization()`
Drop-in replacement for BoTorch's `get_chebyshev_scalarization`:
```python
def get_smooth_chebyshev_scalarization(weights, Y, mu=0.1):
    """g(y) = μ·log(Σ exp(w_i·(y_i - z_i*) / μ))"""
    # Same normalization as BoTorch's version
    # Replace max(w·y) with log-sum-exp
    # Differentiable everywhere — better for gradient-based acq optimization
```

### Layer 2: `get_smooth_chebyshev_set_scalarization()`
New — no BoTorch equivalent exists:
```python
def get_smooth_chebyshev_set_scalarization(weights, Y, K, mu=0.1):
    """
    For K candidate solutions, m objectives:
    g(X_K) = μ·log(Σᵢ exp(λᵢ·(-μ·log(Σ_k exp(-y_ik/μ)) - zᵢ*) / μ))
    
    Input: Y tensor of shape (K, m) — K solutions, m objectives
    Output: scalar scalarization value
    """
```

### Layer 3: Acquisition Function Integration
- **q-STCH-ParEGO**: Replace Chebyshev in qExpectedImprovement with STCH
- **q-STCH-Set-ParEGO**: Use STCH-Set scalarization where q candidates form the solution set
  - Natural fit: BoTorch's q-batch = Lin's K solution set
  - Each of q candidates covers different objectives
- **Composite variant**: Use multi-output GP + STCH-Set as outer function

### Layer 4: Benchmarks to Run
Lin et al. used: convex quadratics, mixed linear/nonlinear regression (50 runs each)
We need BO-specific benchmarks:
- Standard MOO test problems (DTLZ, ZDT) with many objectives (m=10,20,50,100)
- Real-world: molecular optimization (multi-property), materials design
- Compare against: qParEGO, qEHVI, qNParEGO, MOBO baselines
- Metrics: hypervolume, worst-objective regret, coverage of Pareto front
- Budget: 50-200 function evaluations (expensive BO regime)

---

## 6. Key Technical Challenges

1. **Numerical stability**: Double log-sum-exp in STCH-Set needs careful implementation (log-sum-exp trick)
2. **μ selection**: Too small → gradient issues; too large → poor approximation. May need annealing.
3. **Weight sampling**: For BO, we sample random λ per iteration (like ParEGO). Uniform λ=(1/m,...,1/m) for set version.
4. **GP modeling**: With m >> K, need multi-output GP or independent GPs per objective. Scalability concern.
5. **Acquisition optimization**: STCH-Set scalarization is smooth but potentially multimodal — need good restart strategies.
6. **Theorem 2 in BO context**: Pareto guarantee holds for optimal solution set, but BO finds approximate optima. Need empirical validation.

---

## 7. Summary

**The core insight**: Lin et al.'s STCH-Set is a principled way to find K complementary solutions for m >> K objectives, with Pareto guarantees. Nobody has brought this into Bayesian optimization. BoTorch's ParEGO uses a crude non-smooth Chebyshev. The natural integration is:

> **Use STCH-Set scalarization within BoTorch's acquisition framework, where the q-batch of candidates IS the solution set X_K.**

This is a clean, novel contribution with strong theoretical backing (Lin's theorems) and practical relevance (expensive multi-objective optimization with many objectives).

---

*Generated: 2026-02-17 | Sources: Xi-L/STCH, Xi-L/STCH-Set, arxiv:2405.19650v3, BoTorch source, SSRN:5168818*
