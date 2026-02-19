# Proof Sketch: K=1, μ→0 Reduces qSTCH-Set Exactly to qNParEGO

## Statement

**Claim:** When K=1 and μ→0, the qSTCH-Set acquisition function reduces exactly to the Chebyshev scalarization used in qNParEGO (up to the distinction between random vs. fixed weights).

## Proof

### Step 1: K=1 eliminates the smooth minimum

With K=1, there is a single candidate x. The smooth minimum over candidates becomes trivial:

```
smin_{k=1}^{1} f_i(x^(k)) = -μ log( exp(-f_i(x)/μ) ) = f_i(x)
```

This is exact for any μ > 0. The inner smooth min is identity when K=1.

### Step 2: STCH-Set reduces to single-point STCH

Substituting into the STCH-Set scalarization:

```
g_μ^{STCH-Set}(x | λ) = μ log( Σ_i exp( λ_i(f_i(x) - z_i*) / μ ) )
```

This is precisely the single-point STCH scalarization of Lin et al. (2024), Eq. (3).

### Step 3: μ→0 recovers exact Chebyshev

The log-sum-exp function satisfies:

```
lim_{μ→0+} μ log( Σ_i exp(a_i/μ) ) = max_i a_i
```

Therefore:

```
lim_{μ→0+} g_μ^{STCH-Set}(x | λ) = max_i { λ_i(f_i(x) - z_i*) }
                                    = g^{TCH}(x | λ)
```

This is exactly the augmented Chebyshev scalarization used in ParEGO/qNParEGO.

### Step 4: The acquisition function correspondence

In qSTCH-Set:
```
α^{qSTCH}(x) = E_{f~GP}[ -g_μ^{STCH-Set}(f̂(x) | λ) ]
```

In the K=1, μ→0 limit:
```
α^{qSTCH}(x) → E_{f~GP}[ -max_i { λ_i(f̂_i(x) - z_i*) } ]
             = E_{f~GP}[ min_i { -λ_i(f̂_i(x) - z_i*) } ]
```

This is the expected Chebyshev utility under the GP posterior — which is exactly what qNParEGO optimizes (via qLogNoisyExpectedImprovement with a Chebyshev objective).

### Remaining distinction: weight sampling

The only difference is that:
- **qNParEGO**: Samples a **random** λ ~ Dirichlet or uniform on Δ^{m-1} each iteration
- **qSTCH-Set (K=1)**: Uses a **fixed** λ = 1/m

This distinction vanishes over multiple iterations if one augments qSTCH-Set with random weight sampling when K=1.

## Significance

This proof establishes qSTCH-Set as a **strict generalization** of qNParEGO:
- K=1, μ→0: Recovers qNParEGO (Chebyshev scalarization)
- K=1, μ>0: Smooth Chebyshev (= Pires & Coelho, 2025)
- K>1, μ>0: Full set-based STCH (our contribution)

The generalization hierarchy is:
```
qNParEGO ⊂ STCH-BO (single-point) ⊂ qSTCH-Set (set-based)
```

## Rate of convergence

The approximation gap is bounded:
```
g^{TCH}(x | λ) ≤ g_μ^{STCH}(x | λ) ≤ g^{TCH}(x | λ) + μ log(m)
```

So with μ = 0.1 and m = 10: gap ≤ 0.1 × log(10) ≈ 0.23
With μ = 0.01 and m = 10: gap ≤ 0.01 × log(10) ≈ 0.023

This is the smoothing tax for differentiability.
