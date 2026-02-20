# Mathematical Verification: stch-botorch vs. Lin et al. (ICLR 2025)

**Paper:** "Few for Many: Tchebycheff Set Scalarization for Many-Objective Optimization"  
**arXiv:** 2405.19650v3  
**Verification date:** 2025-02-17  
**Verdict: 🔴 CRITICAL BUG — Sign convention inverted in core scalarization**

---

## 1. Equation-by-Equation Comparison

### 1.1 Smooth Tchebycheff Scalarization (Paper Eq. 5 → `smooth_chebyshev`)

**Paper (Eq. 5):** *(minimization problem)*
```
g^(STCH)_μ(x|λ) = μ log( Σ_i exp( λ_i(f_i(x) - z*_i) / μ ) )
```
This approximates `max_i { λ_i(f_i(x) - z*_i) }` as μ→0.

**BoTorch utility** (maximize) should be:
```
utility = -g^(STCH) = -μ log( Σ_i exp( λ_i(Y_i - z*_i) / μ ) )
         = -μ · logsumexp( λ_i(Y_i - z*_i) / μ )
```

**Code (`scalarization.py` line 78-80):**
```python
weighted_distances = weights * (ref_point - Y)          # λ_i(z*_i - Y_i)
utility = -mu * torch.logsumexp(weighted_distances / mu, dim=-1)
```
This computes: `-μ · logsumexp( λ_i(z*_i - Y_i) / μ )` = `-μ · logsumexp( -λ_i(Y_i - z*_i) / μ )`

**🔴 BUG:** The sign inside the exponent is **flipped**. The code has `-λ_i(Y_i - z*_i)` where the paper requires `+λ_i(Y_i - z*_i)`.

**Effect:**
- Code computes `smooth_min` of weighted deviations instead of `-(smooth_max)`
- This means utility **increases** when objectives get **worse** (higher Y)
- Completely inverts the optimization direction

**Numerical proof:**
| Point Y | Paper utility | Code utility | Correct ordering? |
|---------|--------------|-------------|-------------------|
| [1, 1]  | -0.57        | 0.43        | ❌ Code: 0.43     |
| [2, 2]  | -1.07        | 0.93        | ❌ Code: 0.93     |
| [3, 1]  | -1.50        | 0.50        | ❌ Code: 0.50     |

Code says Y=[2,2] (utility=0.93) is BETTER than Y=[1,1] (utility=0.43), which is **wrong** since Y=[1,1] dominates Y=[2,2].

**Fix:** Change line 78 from:
```python
weighted_distances = weights * (ref_point - Y)
```
to:
```python
weighted_distances = weights * (Y - ref_point)
```

### 1.2 Smooth Tchebycheff Set Scalarization (Paper Eq. 12 → `smooth_chebyshev_set`)

**Paper (Eq. 12):**
```
g^(STCH-Set) = μ log( Σ_i exp( λ_i(smin_k f_i(x^(k)) - z*_i) / μ ) )
where smin_k f_i(x^(k)) = -μ log( Σ_k exp(-f_i(x^(k)) / μ) )
```

**Code (`scalarization.py` lines 143-154):**
```python
R_ik = weights * (ref_point - Y)                          # λ_i(z*_i - Y_ik)
R_i_min = -mu * torch.logsumexp(-R_ik / mu, dim=-2)       # smooth min over k
S = mu * torch.logsumexp(R_i_min / mu, dim=-1)            # smooth max over i  
utility = -S
```

**🔴 BUG (inherited):** Same sign error as `smooth_chebyshev`. `R_ik` should be `weights * (Y - ref_point)`.

**Additional discrepancy:** The code applies weights **inside** the smooth min (computing `smin_k(λ_i(Y_ik - z*_i))`) while the paper applies weights **outside** (computing `λ_i · (smin_k(Y_ik) - z*_i)`). Since λ_i is constant w.r.t. k, these are mathematically equivalent when using the **same** μ, so this is not a bug, but note that it changes the effective smoothing parameter to μ/λ_i per objective.

### 1.3 Single-Objective Reduction

**Paper:** When m=1, g^(STCH) = λ₁(f₁ - z*₁), utility = -(f₁ - z*₁) = z*₁ - f₁.

**Code:** Returns `+5.0` for Y=5, z*=0 instead of `-5.0`. **Confirms sign bug.**

### 1.4 Convergence as μ→0

**Paper (Theorem 3):** `lim_{μ→0} g^(STCH) = g^(TCH) = max_i λ_i(f_i - z*_i)`

**Code behavior:**
| μ     | Code utility | Expected utility (-max) |
|-------|-------------|------------------------|
| 1.0   | 0.399       | -2.349                 |
| 0.1   | 0.600       | -2.100                 |
| 0.01  | 0.600       | -2.100                 |
| 0.001 | 0.600       | -2.100                 |

Code converges to `min_i λ_i(Y_i - z*_i)` = 0.6, not `-max_i` = -2.1. **Wrong limit.**

---

## 2. Gradient Analysis

**Paper:** ∂g/∂f_i > 0 (scalarization increases with objectives), so ∂utility/∂Y_i < 0.

**Code:** Gradient of utility w.r.t. Y is `[+0.497, +0.003]` — **positive**, meaning code gradient pushes objectives HIGHER (worse). Should be negative.

---

## 3. Comparison with BoTorch Native Implementation

### BoTorch's `get_chebyshev_scalarization` (augmented Chebyshev)

```python
# From botorch.utils.multi_objective.scalarization
# Internally: Y_negated = -Y (converts max→min), normalizes, then:
# chebyshev_obj = max(w * Y_norm) + alpha * sum(w * Y_norm)
# Returns -chebyshev_obj (for BoTorch maximization)
```

**Key differences:**

| Feature | BoTorch Native | stch-botorch |
|---------|---------------|-------------|
| Scalarization type | Augmented Chebyshev (hard max + α·sum) | Smooth Chebyshev (LogSumExp) |
| Differentiable? | No (uses hard max) | Yes (smooth approximation) ✓ |
| Normalization | Auto-normalizes Y to [0,1] | No normalization |
| Set scalarization | Not available | Available (few-for-many) ✓ |
| Sign convention | ✅ Correct | 🔴 Inverted |
| Smoothing parameter | None (hard max) | μ controls approximation tightness |
| Theoretical guarantees | Standard Chebyshev properties | Pareto optimality (Theorems 1-4) |

**Advantages of stch-botorch (once sign is fixed):**
1. **Differentiability:** Enables gradient-based optimization unlike hard max
2. **Set scalarization:** Novel "few for many" capability not in BoTorch
3. **Controllable smoothing:** μ parameter allows trading off smoothness vs. accuracy
4. **Theoretical backing:** Proven convergence and Pareto optimality guarantees

---

## 4. Numerical Stability Assessment

### 4.1 Extreme μ values
- μ=1e-6: ⚠️ Risk of overflow in exp(x/μ) for large x. `logsumexp` handles this via max-subtraction trick, so **stable** in PyTorch.
- μ=10.0: Stable but poor approximation to max.
- All tested μ values produce finite results ✅

### 4.2 Large/small objective values
- Y ~ 1e6: Stable (logsumexp handles large values) ✅
- Y ~ 1e-10: Stable ✅

### 4.3 Edge cases
- Equal weights: Symmetry preserved ✅
- Single objective: Should reduce trivially — **fails due to sign bug** 🔴

---

## 5. Complete Bug List

### 🔴 CRITICAL: Sign Convention Inversion (scalarization.py)

**Location:** `smooth_chebyshev()` line 78, `smooth_chebyshev_set()` line 143  
**Impact:** ALL optimization using these functions optimizes in the WRONG DIRECTION  
**Fix:**
```python
# In smooth_chebyshev():
weighted_distances = weights * (Y - ref_point)  # was: weights * (ref_point - Y)

# In smooth_chebyshev_set():
R_ik = weights * (Y - ref_point)  # was: weights * (ref_point - Y)
```

### 🟡 MINOR: Objective wrapper may mask the bug

**Location:** `objectives.py` `SmoothChebyshevObjective.forward()`  
**Issue:** The objective wrapper just calls `smooth_chebyshev` without any sign correction, so the bug propagates to all acquisition functions using these objectives.

### 🟡 MINOR: Weight normalization differs from paper

**Issue:** The code normalizes weights to sum to 1. The paper uses λ ∈ Δ^{m-1} (simplex), which is the same constraint. **Not a bug**, but the code should document that user-provided weights are auto-normalized.

### 🟢 INFO: No augmentation term

**Issue:** BoTorch's native Chebyshev adds `α · Σ(w·Y)` to break ties. The STCH approach doesn't need this because the smooth approximation naturally breaks ties. This is correct per the paper.

---

## 6. Recommendations

1. **IMMEDIATE:** Fix the sign in `weighted_distances` in both `smooth_chebyshev` and `smooth_chebyshev_set`
2. **HIGH:** Add the test suite (`test_numerical.py`) to CI — it catches the sign bug immediately
3. **MEDIUM:** Add numerical stability tests for μ < 1e-4 with objectives > 100
4. **LOW:** Consider adding a normalization option (like BoTorch's auto-normalization) for user convenience
5. **LOW:** Document the relationship between μ and effective per-objective smoothing when weights are non-uniform

---

## 7. Summary

The mathematical formulations in the paper (Eq. 5, 8, 12, 13) are correctly understood in the code's documentation and comments, but the **implementation has a critical sign error** that inverts the optimization direction. The `(ref_point - Y)` should be `(Y - ref_point)` to match the paper's `(f_i(x) - z*_i)` convention. This single-character fix (`ref_point - Y` → `Y - ref_point`) resolves all identified issues.

Once fixed, the implementation correctly provides a differentiable, theoretically-grounded alternative to BoTorch's hard-max Chebyshev scalarization, with the unique addition of set scalarization for many-objective optimization.
