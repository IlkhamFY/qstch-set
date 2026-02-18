# Strategic Analysis: Pires & Coelho CMAME 2024 — Composite Bayesian Optimisation

**Paper:** "A composite Bayesian optimisation framework for material and structural design"
**Authors:** R.P. Cardoso Coelho, A.F. Carvalho Alves, T.M. Nogueira Pires, F.M. Andrade Pires
**Journal:** Computer Methods in Applied Mechanics and Engineering 434 (2025) 117516
**DOI:** 10.1016/j.cma.2024.117516

---

## 1. Executive Summary

This paper presents a **composite Bayesian optimisation framework** where the key insight is: don't build a GP surrogate on the scalar objective J(θ) — instead, build multi-output GPs on the raw simulation responses (stress-strain curves, force-displacement curves) and embed the known reduction function (e.g., trapezoidal integration for toughness) into the acquisition evaluation pipeline.

**Critical finding for our qSTCHSet paper:** This CMAME 2024 paper uses **EHVI (Expected Hypervolume Improvement)** exclusively for multi-objective problems. **Smooth Tchebycheff (STCH) scalarization appears NOWHERE in this paper.** STCH exists only in their unpublished development branch on GitHub (`composite_design_mo_stch`) and presumably in their SSRN preprint. This gives us a clear novelty window.

---

## 2. Their Composite BO Formulation (Section 2.3)

### Core Idea
The objective function is decomposed as:
```
J(θ) = R(θ, t(θ), Y(θ))    [Eq. 4]
```
where:
- `F[θ]` = numerical simulation yielding time grid `t(θ)` and outputs `Y(θ)`
- `R` = known reduction function (e.g., trapezoidal integration, max extraction)

### Five Steps of Composite BO:

1. **Transformation T** (Section 2.3.1): Map variable-size simulation outputs `(t(θ), Y(θ))` → fixed-size latent vector `p(θ)` via interpolation at N evenly-spaced points. For curve fitting: pointwise errors [Eq. 15]. For response design: endpoint interpolation [Eq. 16] giving `p(θ) = [y₁(θ), ..., yₙ(θ), t_min, t_max]`.

2. **Multi-output surrogate** (Section 2.3.2): Build **independent single-task GPs** for each component of `p(θ)` [Eq. 18]. They justify this via **autokrigeability** — for noiseless observations, independent GPs are equivalent to a multi-task GP. This is computationally much cheaper.

3. **Posterior of objective** (Section 2.3.3): Draw Q samples from GP posteriors → transform back via T⁻¹ → evaluate reduction function R on each sample [Eq. 21]. Key issue: independent sampling causes posterior variance to shrink as 1/√N — mitigated by PCA.

4. **MC acquisition evaluation** (Section 2.3.4): Acquisition functions evaluated as MC expectations over samples [Eq. 24]. EI naturally fits; PI and UCB rewritten as expectations [Eqs. 25-26]. Gradients via reparameterization trick + PyTorch autodiff through the entire pipeline (Fig. 5).

5. **PCA dimensionality reduction** (Section 2.3.5): Reduce latent space `p(θ)` → `p̂(θ)` via PCA [Eq. 28]. Recomputed every iteration. Number of components k chosen by unexplained variance threshold (default 10⁻⁶). This is **critical** — it reduces computational cost, decorrelates features (enabling independent GP assumption), and makes the method insensitive to interpolation resolution N.

---

## 3. Multi-Objective Formulation (Section 2.4)

### Acquisition Function: EHVI

They use **Expected Hypervolume Improvement (EHVI)** exclusively:
- Specifically cite Daulton et al. [66] for efficient exact computation
- Also mention NEHVI for noisy observations [67]
- Default acquisition in their code: `qLogExpectedHypervolumeImprovement` (qLoqEHVI)

### How Composite + MO Works

For multi-objective problems with composition, each objective can have its own reduction function. Example from Section 4.3.1:
```
[J₁(θ,β), J₂(θ,β)] = [-m(θ,β), R(θ,β, F[θ,β])]    [Eq. 46]
```
where mass m is computed analytically (not learned) and stiffness k_z uses the composite pipeline. The GP surrogates model the simulation responses, and each objective's reduction function is applied to the posterior samples.

### What They Do NOT Do
- **No Tchebycheff scalarization**
- **No ParEGO**
- **No scalarization-based MOBO at all**
- Pure hypervolume-based approach throughout

---

## 4. PCA Details

- Linear PCA via eigendecomposition of covariance matrix of standardized latent responses
- Transformation matrix W = [v₁, ..., vₖ] of first k eigenvectors [Eq. 27]
- Forward: `p̂(θ) = p(θ)W`, Reverse: `p(θ) = p̂(θ)Wᵀ` [Eqs. 28-29]
- Recomputed every BO iteration (k can change)
- Threshold study (Table 2): 10⁻⁶ gives ~8 PCs for their 32-point interpolation, good balance of quality vs. cost
- **Without PCA**: 146.5 min median time; **With PCA (10⁻⁶)**: 26.0 min — 5.6× speedup
- PCA also **improves** optimization quality (higher objective values) by preventing over-exploitation from artificially low posterior variance

---

## 5. Experimental Results

### Problems (4 total):

| Problem | Dims | Objectives | Type | Budget |
|---------|------|------------|------|--------|
| Analytical response [Eq. 31] | 8 (2n, n=4) | 1 | Integral of cosine+Gaussian sum | 128 |
| 3D particle-matrix RVE toughness | 2 (a, b) | 1 | Max toughness via stress-strain integral | 159 |
| CTC metamaterial (3 sub-examples) | 2-3 (θ, β, σ_y) | 1 or 2 | Stiffness, energy, mass | 64-128 |
| Metallic foam beam (2 sub-examples) | 16 (ξ₁...ξ₁₆) | 1 | Specific bending stiffness / energy | 128-256 |

### Baselines:
1. **Classical BO** (scalar GP on J(θ), same BoTorch backend)
2. **Random sampling** (Sobol sequences)
3. **Bessa surrogate** (build GP on random DoE, then optimize surrogate — no adaptive sampling)

### Key Results:
- Composite BO consistently best, especially for nonlinear problems
- For analytical problem: composite needs ~10× fewer evaluations than Bessa surrogate
- For CTC metamaterial MO: composite achieves highest hypervolume with least dispersion
- For 16D foam beam: composite converges faster and with much less variance than classical BO
- Bessa surrogate competitive only when given much larger budgets (5-8× more evaluations)

---

## 6. Limitations They Acknowledge

1. **Transformation design is problem-dependent** — endpoint interpolation may fail for responses with severe length differences or varying length scales (Remark 1)
2. **Independent GPs lose correlation info** when sampling — mitigated by PCA but not eliminated
3. **Hyperparameters**: DoE size heuristic (max(8, 2n_dim)) is problem-specific
4. **No theoretical convergence guarantees** for the composite approach
5. **Computational overhead** of composite approach vs. classical BO (justified only when function evaluations are expensive)
6. **Limited to time-dependent responses** (though they note reparameterization can generalize this)

---

## 7. piglot Package and BoTorch Relationship

**piglot** is their open-source Python package ([github.com/CM2S/piglot](https://github.com/CM2S/piglot), JOSS paper) that:
- Wraps BoTorch for Bayesian optimization
- Adds the composite BO pipeline (transformation T, PCA, multi-output GPs, composite acquisition evaluation)
- Provides solver interfaces (FEM, analytical curves, etc.)
- Handles YAML config-driven optimization

**Key architectural point**: piglot builds ON TOP of BoTorch. They use BoTorch's:
- `SingleTaskGP` models
- `fit_gpytorch_mll` for GP fitting
- Standard acquisition functions (qLogEI, qLogEHVI, etc.)
- `optimize_acqf` for acquisition optimization
- `SobolQMCNormalSampler` for MC sampling

The composite pipeline is implemented as a `GenericMCObjective` / `GenericMCMultiOutputObjective` that takes GP posterior samples → applies T⁻¹ → applies reduction R → returns scalar/vector objectives. This is how they get gradients through the whole chain via PyTorch autograd.

---

## 8. The `composite_design_mo_stch` Branch — CRITICAL INTELLIGENCE

### What It Contains

This branch (last commit: Sep 3, 2024, by Tiago Pires) adds **Smooth Tchebycheff scalarization** to piglot. Key changes:

**In `objective.py` → `ObjectiveResult.scalarise()`:**
```python
if self.scalarisation == 'stch':
    ideal_point = np.where(types, 1.0, 0.0)
    u = 0.006  # smoothing parameter
    tch_numerator = np.abs((norm_funcs - ideal_point) * costs) * weights
    tch_values = tch_numerator / u
    # Numerical stability loop: increment u until exp doesn't overflow
    u_increment = 0.001
    max_u = 0.2
    while u <= max_u:
        exp_sum = np.sum(np.exp(tch_values))
        if not np.isinf(exp_sum):
            break
        u += u_increment
        tch_values = tch_numerator / u
    return np.log(exp_sum) * u
```

### Key Observations:
1. **Naive STCH implementation** — uses `log(Σ exp(w|f-f*|/u))·u` which is the standard smooth Tchebycheff
2. **No BoTorch integration** — this is computed in NumPy on `ObjectiveResult`, NOT as a differentiable acquisition function
3. **Hacky numerical stability** — they just increment u until exp doesn't overflow (u from 0.006 to 0.2 in 0.001 steps)
4. **Not a q-batch acquisition** — this is scalarization of observed values, not an acquisition function for candidate selection
5. **The branch adds `scalarisation` field** to `ObjectiveResult` with options: 'mean', 'stch', 'linear'
6. **50 commits ahead of main, 19 behind** — significant development, not merged

### What This Means Strategically:
- They're working on STCH but it's **primitive** — just scalarizing observed objectives, not building a proper STCH acquisition function
- They don't have a **differentiable, batch-compatible** STCH scalarization integrated into BoTorch's acquisition framework
- They don't have **qSTCHSet** (set-valued STCH for true multi-point Pareto optimization)
- Their approach is more like "scalarize then optimize" rather than "optimize the scalarization as an acquisition"

---

## 9. Strategic Positioning for Our qSTCHSet Paper

### Our Advantages Over Pires & Coelho:

| Aspect | Pires & Coelho (CMAME 2024) | Our qSTCHSet |
|--------|---------------------------|--------------|
| MO acquisition | EHVI (hypervolume-based) | STCH scalarization (Tchebycheff-based) |
| Scalarization | Not used (EHVI only) | Core contribution — smooth, differentiable |
| Batch optimization | Standard q-batch EHVI | q-batch STCH with set-valued formulation |
| BoTorch integration | External wrapper (piglot) | Native BoTorch acquisition function |
| Composite BO | Yes (their main contribution) | Orthogonal — our STCH works with any GP setup |
| Theoretical grounding | Empirical only | Convergence properties of STCH smoothing |
| STCH implementation | Branch-only, NumPy, naive | PyTorch, differentiable, numerically stable |

### How to Frame the Relationship:
1. **Complementary, not competing**: Their composite BO (multi-output GPs + PCA + embedded reduction) is about HOW to build surrogates. Our STCH is about WHICH acquisition to optimize. The two could be combined.
2. **They validate EHVI works with composite BO** — we can cite this as baseline and show STCH scalarization offers advantages (different Pareto coverage, better for many objectives).
3. **Their STCH branch proves demand** — the Porto group clearly sees value in STCH for MO problems but hasn't cracked the proper BoTorch-native implementation.
4. **Our contribution is the acquisition function theory** — differentiable smooth Tchebycheff as a proper BoTorch acquisition, with q-batch support and convergence analysis.

### Suggested Citations:
- Cite their CMAME 2024 paper for composite BO framework + EHVI results
- Note that their approach is "hypervolume-based" and ours provides an "alternative scalarization-based" approach
- If their SSRN paper on STCH is published, cite it as concurrent/prior work on STCH in engineering design, noting our contribution is the rigorous BoTorch-native formulation

---

## 10. Open Questions

1. **What's in their SSRN paper?** — Need to find and analyze it. It likely contains the STCH formulation that the GitHub branch implements.
2. **Is their STCH actually used as an acquisition function or just for post-hoc scalarization?** — From the code, it appears to be scalarization of objective values, not an acquisition function. This is a crucial distinction.
3. **Do they have any theoretical analysis of STCH properties?** — Unlikely given the empirical nature of their CMAME paper.
4. **Could composite BO + STCH acquisition be combined?** — Yes, and this could be a powerful direction. Their multi-output GP surrogates + our STCH acquisition = potential collaboration or extension.
