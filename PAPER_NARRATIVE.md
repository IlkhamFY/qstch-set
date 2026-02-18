# Paper Narrative: STCH-Set Bayesian Optimization

**Working Title:** "Few for Many in Bayesian Optimization: Smooth Tchebycheff Set Scalarization for Many-Objective Expensive Black-Box Problems"

**Authors:** Ilkham Yabbarov, Rodrigo Vargas-Hernández

**Target:** NeurIPS 2026 (deadline ~May 2026)

---

## One-Sentence Summary

We bring STCH-Set scalarization — which guarantees Pareto-optimal coverage of m objectives by K coordinated solutions — from the gradient-based world to sample-efficient Bayesian optimization, enabling the first BO method that scales gracefully to m >> 5 objectives.

## The Story

### The Problem (Why)

Multi-objective Bayesian optimization (MOBO) enables sample-efficient optimization of expensive black-box functions with conflicting objectives. However, existing methods hit a wall:

- **Hypervolume-based methods** (qEHVI, qNEHVI) are state-of-the-art for 2-5 objectives but hypervolume computation is #P-hard in m — exponential cost makes them impractical for m > 5.
- **Scalarization methods** (ParEGO, qNParEGO) use random Chebyshev weights to decompose into single-objective problems. They scale better in m but: (a) each weight vector finds only one solution, (b) random weights don't coordinate to ensure coverage, (c) the non-smooth max operator makes acquisition optimization harder.
- **Information-theoretic methods** (MESMO, PFES) approximate entropy computations that also degrade beyond m ≈ 4.

Meanwhile, real applications need many objectives:
- Drug discovery: 8-50+ ADMET properties (potency, selectivity, permeability, stability, toxicity, CYP inhibition...)
- Materials design: strength, weight, cost, thermal conductivity, corrosion resistance...
- Pharma currently *collapses* these into lossy MPO scores because MOBO can't handle them natively.

### The Key Insight (What)

Lin et al. (ICML 2024, ICLR 2025) introduced Smooth Tchebycheff (STCH) scalarization and its Set variant (STCH-Set). STCH-Set jointly optimizes K solutions to collectively cover m objectives, with:
- Guaranteed weak Pareto optimality for ALL K solutions (Theorem 2, ICLR 2025)
- O(Km) computation — linear in both K and m
- Fully differentiable via log-sum-exp smoothing

**But their framework assumes cheap, differentiable objectives.** It cannot be directly applied to expensive black-box functions.

Pires & Coelho (SSRN 2025) took the first step, combining single-point STCH with composite BO. But they only find ONE solution per optimization — no set-based coordination.

We fill the remaining empty cell:

|                       | Single Solution | Set of K Solutions |
|-----------------------|-----------------|-------------------|
| **Gradient-based**    | STCH (Lin ICML24) | STCH-Set (Lin ICLR25) |
| **Bayesian Optimization** | Pires & Coelho (2025) | **Ours: qSTCH-Set** |

### The Method (How)

**qSTCH-Set**: A Monte Carlo acquisition function that applies STCH-Set scalarization to GP posterior samples.

Given multi-output GP posteriors for m objectives:
1. Draw MC samples from the joint posterior at candidate points
2. Apply STCH-Set scalarization: smooth min over K candidates, smooth max over m objectives
3. Average over MC samples to get acquisition value
4. Jointly optimize K candidates via L-BFGS-B (differentiable through log-sum-exp)

Key properties:
- **Coordinated batch**: K candidates are optimized as a team, not independently
- **Scales to m >> 5**: O(Km) per acquisition evaluation
- **Differentiable**: Smooth log-sum-exp enables gradient-based acquisition optimization
- **Pareto guarantee**: Inherits STCH-Set's Theorem 2 under GP posterior concentration
- **Drop-in BoTorch integration**: Standard `MCAcquisitionFunction` interface

### The Evidence (So What)

**Phase 1 (in progress):**
- ZDT suite (m=2): STCH-NParEGO beats Vanilla-NParEGO by 3.7 HV points
- DTLZ2 (m=3): First benchmark running
- DTLZ2 (m=5,8,10): Where we expect dominance over qEHVI

**Phase 2 (planned):**
- DTLZ2 m=15,20: Unprecedented for BO — competitors can't even run here
- Wall-clock scaling: Show O(Km) vs exponential
- Drug discovery: Multi-ADMET optimization with K=3 lead candidates

## Reviewer Defenses

1. **"Just engineering"** → Non-trivial adaptation from gradient to posterior; includes theoretical Pareto guarantee transfer analysis
2. **"Pires & Coelho did it"** → They did single-point, not set-based. Single-point finds one solution per weight; we find K coordinated solutions
3. **"MORBO scales"** → MORBO scales input dims (d=222), not objectives (tested m=2-4). Orthogonal
4. **"Why not approximate HV?"** → Even MC HV estimation has huge variance for m>6. Our O(Km) is fundamentally cheaper
5. **"No regret bounds"** → Provide convergence analysis under GP posterior concentration + empirical superiority

## Key Figures

1. **2×2 positioning matrix** (Table 1) — immediately shows the gap
2. **HV convergence on DTLZ2** for m=3,5,8,10,15 — show graceful scaling
3. **Wall-clock time vs m** — exponential (qEHVI) vs linear (ours)
4. **Pareto front visualization** — parallel coordinates for m>3
5. **Drug discovery case study** — real ADMET objectives, K=3 lead candidates
6. **Ablation: K, μ, MC samples**
7. **STCH-Set vs independent single-point STCH** — shows coordination value

---

*Last updated: 2026-02-17*
