# qSTCH-Set vs Baselines: Deep Code-Level Analysis

## Date: Feb 25, 2026

---

## 1. Lin et al.'s Original STCH-Set (github.com/Xi-L/STCH-Set)

Their implementation is remarkably simple. From `run_quad_func.py`:

```python
# STCH-Set (their code, verbatim)
mu = 0.1
value = mu * torch.logsumexp(-mu * torch.logsumexp(-value_set / mu, dim=0) / mu, dim=0)
```

Where `value_set` has shape `(K, m)` — K solutions, m objectives. Minimization.

Breaking it down:
- Inner: `-mu * logsumexp(-value_set / mu, dim=0)` → smooth min over K solutions per objective → shape `(m,)`
- Outer: `mu * logsumexp(inner / mu, dim=0)` → smooth max over m objectives → scalar

**Key observations about their implementation:**
1. **No weights (lambda).** They use uniform weighting implicitly.
2. **No reference point (z*).** The scalarization operates directly on raw objective values.
3. **mu=0.1 throughout.** Single temperature for both inner and outer aggregation.
4. **Direct gradient descent** on the solution set (10,000 steps, lr=1e-2).
5. They optimize K=5 solutions jointly for m=128 objectives.

Their formulation is: `max_i min_k f_i(x_k)` (smoothed).
Goal: make the *worst-covered* objective as good as possible across K solutions.

---

## 2. Our smooth_chebyshev_set Implementation

From `scalarization.py`:

```python
def smooth_chebyshev_set(Y, weights, ref_point, mu=0.1):
    # Y shape: (..., q, m) — minimization convention
    
    # Inner: smooth min over batch (dim=-2, q dimension)
    inner = -mu * torch.logsumexp(-Y / mu, dim=-2)  # shape: (..., m)
    
    # Outer: smooth max over objectives (dim=-1, m dimension)  
    S = mu * torch.logsumexp(weights * (inner - ref_point) / mu, dim=-1)
    
    utility = -S  # negate for BoTorch maximization
    return utility
```

### Differences from Lin et al.:

| Aspect | Lin et al. | Ours |
|--------|-----------|------|
| Inner aggregation | `-mu * logsumexp(-Y/mu, dim=0)` | `-mu * logsumexp(-Y/mu, dim=-2)` |
| Outer aggregation | `mu * logsumexp(inner/mu, dim=0)` | `mu * logsumexp(weights*(inner-ref)/mu, dim=-1)` |
| Weights | None (uniform) | Normalized to sum=1, multiplied in outer |
| Reference point | None | Subtracted before outer aggregation |
| Optimization | Direct GD, 10k steps | L-BFGS-B via optimize_acqf, ~200 iters |

### ⚠️ CRITICAL: Weight placement difference

Lin et al.'s outer (unweighted): `mu * logsumexp(inner / mu)`
Ours (weighted): `mu * logsumexp(weights * (inner - ref) / mu)`

The weights multiply BEFORE dividing by mu, which means:
`logsumexp(w_i * (inner_i - ref_i) / mu)`

Since weights sum to 1, and w_i = 1/m for uniform, this becomes:
`logsumexp((inner_i - ref_i) / (m * mu))`

This effectively **reduces** the temperature by factor m! For m=3 (real-world), effective temp = mu/3 = 0.033. For m=10 (DTLZ2), effective temp = mu/10 = 0.01.

Smaller effective temperature = sharper max = sparser gradients = harder optimization.

**This is likely a significant bug.** Lin et al. don't put weights inside the logsumexp.

### How BoTorch's Augmented Chebyshev differs

BoTorch's `get_chebyshev_scalarization`:
```python
# Augmented Chebyshev (Knowles 2005)
g(y) = max_i(w_i * y_i) + alpha * sum_i(w_i * y_i)
```

Key features:
1. **Augmented** with a weighted sum term (alpha=0.05) for tie-breaking
2. Uses `product.max(dim=-1).values` (hard max, not smooth)
3. Normalizes Y to [0,1] per objective using observed data bounds
4. Applied as a scalarization of a SCALAR objective → qNEI optimizes expected improvement of this scalar
5. Fresh random weights per candidate in qNParEGO

---

## 3. The Fundamental Architecture Difference

### qNParEGO pipeline:
1. Sample random weight vector w ~ Dirichlet(1,...,1) for each candidate
2. Scalarize multi-output to scalar: g(y) = max(w_i * y_i) + 0.05*sum(w_i * y_i)
3. Compute Expected Improvement of this SCALAR relative to best-so-far scalarized value
4. Optimize each candidate independently (optimize_acqf_list)
5. Each candidate targets a different region of Pareto front (due to different weights)

### qEHVI/qNEHVI pipeline:
1. Compute non-dominated partitioning of objective space
2. For each MC sample, compute hypervolume improvement
3. Optimize q candidates jointly (with sequential=True for conditioning)

### Our qSTCH-Set pipeline:
1. Draw MC posterior samples
2. Apply STCH-Set scalarization to each sample (jointly over q candidates)
3. Average over MC samples
4. Optimize q candidates jointly via L-BFGS-B

### Why qNParEGO's approach is so effective:
- **Expected Improvement** = probability of improvement × magnitude of improvement
  - Naturally balances exploration (uncertain regions) vs exploitation (good regions)
  - Well-understood, well-calibrated
- Random per-candidate weights → natural diversity without explicit set coordination
- Each candidate optimized independently → simple, stable optimization landscape

### Why our approach struggles:
- **No exploration-exploitation trade-off.** We directly optimize the scalarization value
  on posterior samples. There's no EI-like mechanism that values uncertainty.
  We're essentially doing Thompson Sampling without the diversity benefit.
- **Joint optimization in q*d dimensions** is much harder than sequential
- The STCH-Set value is a *summary statistic* — it doesn't distinguish between
  "found a known-good region" and "exploring an uncertain but potentially great region"

---

## 4. Root Causes of Underperformance (Priority Order)

### A. Weights inside logsumexp (likely a bug)
Our weights multiply the deviations BEFORE the logsumexp, effectively reducing
temperature by 1/m. Lin et al. don't use weights at all in their logsumexp.

Fix: Move weights outside or remove them (use uniform like Lin et al.).

### B. No exploration incentive
qNParEGO gets exploration from Expected Improvement.
qEHVI/qNEHVI get exploration from hypervolume improvement + uncertainty.
We get nothing — we just score the posterior mean quality.

Fix: Wrap STCH-Set in an improvement-based framework (expected improvement of STCH-Set value relative to best-observed STCH-Set value?).

### C. Missing augmentation term
BoTorch's Chebyshev adds `alpha * sum(w*y)` (5% weighted sum) for tie-breaking.
STCH-Set has no such term. When the smooth max is flat (multiple objectives equally bad), 
there's no gradient signal to break ties.

### D. Joint vs sequential optimization  
Not inherently wrong for STCH-Set (it's designed for joint optimization), but L-BFGS-B
with 10 restarts in q*d dimensions isn't enough. Lin et al. use 10,000 gradient steps.
We get ~200 L-BFGS-B iterations.

### E. Reference point sensitivity
With normalization, the ref point determines what "improvement" means.
Without careful ref point placement, the scalarization may not discriminate well.

---

## 5. Recommended Fixes

### Fix 1: Remove weights from inner logsumexp (match Lin et al.)
```python
# Current (WRONG):
S = mu * torch.logsumexp(weights * (inner - ref_point) / mu, dim=-1)

# Proposed (match paper):
S = mu * torch.logsumexp((inner - ref_point) / mu, dim=-1)
```
Or if keeping weights: `S = mu * torch.logsumexp((inner - ref_point) / mu + torch.log(weights), dim=-1)`
which keeps the same logsumexp scale.

### Fix 2: Add augmentation term (like BoTorch Chebyshev)
```python
S = mu * torch.logsumexp((inner - ref_point) / mu, dim=-1) + alpha * (inner - ref_point).sum(dim=-1)
```

### Fix 3: Consider wrapping in an EI framework
Instead of optimizing E[STCH-Set(posterior_samples)] directly,
optimize E[max(0, STCH-Set(posterior_samples) - best_observed_stch_set)]
This adds exploration incentive.

### Fix 4: Increase optimization budget for joint mode
More restarts (20-50) and more raw_samples (1024-2048) if not using sequential.

---

## 6. What This Means for the Paper

The key novelty question: is qSTCH-Set just Chebyshev-scalarized BO (i.e., qNParEGO)?

**No.** The fundamental differences are:
1. **Set-based** scalarization: candidates are aware of each other through the smooth-min.
   qNParEGO candidates are independent (different random weights).
2. **Joint optimization**: candidates are optimized together (or sequentially with conditioning).
   qNParEGO optimizes each independently with different scalarizations.
3. **Smooth approximation**: differentiable everywhere, enabling true gradient-based optimization.
   Standard Chebyshev uses hard max.

The novelty is real. The implementation needs fixing.
Adding `sequential=True` does NOT make it equivalent to qNParEGO because:
- qNParEGO uses different random weights per candidate
- We use STCH-Set which couples all candidates through the smooth-min
- Sequential just means we optimize one at a time while conditioning on selected ones,
  but the acquisition function still evaluates the full set jointly.
