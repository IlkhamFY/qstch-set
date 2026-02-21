# PAPERS.md — Living Paper Knowledge Base

> **For Jarvis**: Read this file when reasoning about positioning, related work, or claims.
> Update it every time a new paper is read or a new insight is formed.
> This is the single source of truth for what every paper says and how it relates to ours.

---

## Our Paper

**qSTCH-Set** (Yabbarov et al., in prep)
- Method: K candidates jointly optimized via sum of smooth Chebyshev (STCH) scalarizations over GP posterior samples
- Key insight: K=m design rule — one candidate per objective direction
- One-stage, fully differentiable, proper MC acquisition function
- Results: m=5 wins (6.65 vs 6.43), m=10 wins by 6.5% (46.95 vs 44.10), m=8 K=5 loses (K=8 closes gap)
- Repo: https://github.com/IlkhamFY/stch-botorch
- Position in 2x2: (BO) × (Set-based) — the empty cell

---

## Directly Related (Cite + Contrast)

### Lin et al. 2024 — STCH for Multi-Objective Optimization (ICML)
- arXiv: not on arXiv, ICML 2024
- **What**: Introduces smooth Tchebycheff (STCH) scalarization replacing max with log-sum-exp. Applied to gradient-based multi-objective optimization (not BO). Single-point scalarization.
- **Key result**: STCH smoother than exact Chebyshev, gradient-friendly
- **Overlap**: We use their STCH formulation directly
- **Gap**: No BO, no set-based, no GP posterior sampling
- **Position in 2x2**: (Gradient) × (Single-point)
- **Threat level**: NONE — foundational building block, we cite and extend

### Lin et al. 2025 — Few for Many: Tchebycheff SET Scalarization (ICLR)
- **What**: Extends STCH to set-based optimization — jointly scalarizes K solutions covering the Pareto front. Still gradient-based, not BO.
- **Key result**: K solutions efficiently cover m objectives, m up to 1024
- **Overlap**: We use their set-based STCH formulation in BO
- **Gap**: No Bayesian optimization, no GP, no uncertainty
- **Position in 2x2**: (Gradient) × (Set-based)
- **Threat level**: NONE — the other foundational building block, we extend to BO

### Pires & Coelho 2025 — Smooth Tchebycheff for Composite BO (SSRN preprint)
- **What**: Uses STCH scalarization inside Bayesian optimization. Single-point (K=1). Builds on Astudillo & Frazier 2019 composite BO framework.
- **Key result**: STCH scalarization improves BO acquisition over exact Chebyshev
- **Overlap**: Combines STCH + BO — same domain as us
- **Gap**: Single-point only (K=1), no set structure, no K=m rule
- **Position in 2x2**: (BO) × (Single-point)
- **Threat level**: LOW — closest prior work but we extend to set-based

### Paulson et al. 2025 — Generative MOBO with qPMHI (arXiv:2512.17659, Dec 2025)
- **What**: Two-stage "generate-then-optimize" for discrete molecular design. Stage 1: generate pool with STCH scalarization (optimizing GP mean per weight). Stage 2: select batch via qPMHI (P(max HVI)), which decomposes additively.
- **Key result**: Outperforms latent-space BO methods on molecular benchmarks. Quinone cathode materials for flow batteries.
- **Implementation**: We have `src/stch_botorch/acquisition/qpmhi.py` implementing their qPMHI, and `stch_qpmhi.py` implementing the two-stage pipeline. This was our STARTING POINT before we pivoted.
- **Why we pivoted away from STCH-qPMHI:**
  1. Stage 1 uses GP MEAN (not uncertainty) — not proper BO
  2. Candidates generated independently — no coupling, no guarantee of complementarity
  3. Two objectives: generate by STCH, select by qPMHI — misaligned
  4. Designed for DISCRETE molecular spaces where gradients over inputs don't exist
  5. In CONTINUOUS spaces (our target), end-to-end gradient optimization is strictly better
- **Overlap**: We implement qPMHI in our repo. We use STCH for candidate generation similarly.
- **Gap**: No joint optimization, no gradient flow, discrete not continuous
- **Threat level**: LOW — complementary (discrete vs continuous). Should cite as related work.
- **Narrative**: "Our method handles continuous BO; Paulson et al. handle discrete molecular spaces."

---

## Baselines (Cite + Benchmark Against)

### Daulton et al. 2020 — qEHVI (NeurIPS)
- **What**: q Expected Hypervolume Improvement. Differentiable HV with box decomposition.
- **Complexity**: O(2^m) in objectives — exponential, #P-hard
- **Limit**: Practical only for m ≤ 5-6. BoTorch says FastNondominatedPartitioning "very slow" for >5 objectives.
- **Our result at m=5**: Still running after 30 iters (never finished in our interactive session)
- **Cite for**: Gold standard comparison, scaling wall motivation

### Daulton et al. 2021 — qNEHVI (NeurIPS)
- **What**: Noisy version of qEHVI. Mathematically equivalent in noiseless setting.
- **Same complexity issue** for large m.

### qNParEGO (BoTorch implementation of ParEGO)
- **What**: q-batch version of ParEGO. Random Chebyshev scalarization per candidate, independent.
- **Our results**: Wins at m=8 K=5 (20.89 vs 19.26). Basically ties at m=10 K=5 (44.10 vs 45.45). We beat it at m=5 K=5 and m=10 K=10.
- **Key weakness**: Uncoordinated weights — no guarantee K candidates cover objective space
- **Cite for**: Primary scalable baseline, shows uncoordinated scalarization limits

### Daulton et al. 2022 — MORBO (UAI)
- **What**: Multi-Objective BO over high-dimensional SEARCH spaces (not objective space). Uses local models.
- **IMPORTANT**: MORBO scales in INPUT dimension, not objective dimension. It is NOT a many-objective method.
- **Cite for**: Distinguishing input-dim scaling vs objective-dim scaling

### MOBO-OSD 2025 (NeurIPS, anonymous)
- **What**: Batch MOBO via orthogonal search directions.
- **Limit**: m ≤ 6 in their experiments.
- **Threat level**: NONE for m > 6

---

## Background / Theory

### Astudillo & Frazier 2019 — Composite BO (ICML)
- **What**: Framework for BO with composite objectives h(f(x)). Theory for when composed functions maintain properties of acquisition functions.
- **Why we cite**: Our Proposition 2 (Pareto optimality transfer) uses their Theorem 2. K=1 case of our method reduces to their framework.
- **Key theorem**: Preserves consistency and convergence for K=1. We conjecture extension to K>1.

### Knowles 2006 — ParEGO
- **What**: Original scalarization-based MOBO. Random Chebyshev weights per iteration.
- **Cite for**: Historical context of scalarization in MOBO

### Deb & Jain 2014 — NSGA-III
- **What**: Reference-point based NSGA for many objectives
- **Cite for**: Evolutionary baseline context, needs thousands of evals

---

## To Read / Verify

- [ ] Lin et al. ICML 2024 — need exact theorem numbers for consistency proof
- [ ] Pires & Coelho — SSRN, not peer-reviewed yet; check if published version exists
- [ ] MOBO-OSD 2025 — anonymous, check if deanonymized/published
- [ ] Paulson et al. 2512.17659 — ADD TO REFERENCES.BIB (missing!)

---

## BibTeX Entries Needed (not yet in references.bib)

```bibtex
@article{paulson2025qpmhi,
  title={Generative Multi-Objective {B}ayesian Optimization with Scalable Batch Evaluations for Sample-Efficient {De Novo} Molecular Design},
  author={Sorourifar, Farshud and Tan, Tianhong and Peng, You and Paulson, Joel A.},
  journal={arXiv preprint arXiv:2512.17659},
  year={2025}
}
```

---

## System: How to Keep This Updated

**Every time I read a new paper:**
1. Add entry to this file under the right section
2. Add BibTeX entry to `references.bib`
3. Update MEMORY.md with one-line summary if it's a key paper

**Every time a claim in the paper changes:**
- Update the "Our Paper" section
- Update "Threat level" if competitive landscape shifts

**On session restart:**
- `memory_search("stch-botorch papers")` will surface this file
- Read it before any discussion of related work or positioning

---

## [Informing Acquisition Functions via Foundation Models for Molecular Discovery](https://arxiv.org/abs/2512.13935)
**Added:** 2026-02-21
**Authors:** Qi Chen, Fabio Ramos, Alan Aspuru-Guzik, Florian Shkurti
**Venue:** arXiv preprint (2025)
**Citations:** 0 (new)
**Threat Level:** 2 � LLM priors + MCTS for single-objective molecular BO; no multi-objective component

### Method
Likelihood-free BO that bypasses GP surrogate modeling entirely. Uses LLM/chemistry foundation model priors directly to inform acquisition functions. Learns a tree-structured partition of molecular space (MCTS) with local acquisition functions. LLM-based coarse clustering restricts candidate evaluation to high-value regions.

### Gap They Claim
Standard BO fails in low-data, large candidate-space molecular discovery. GP surrogates lack chemistry priors; Bayesian deep learning is expensive.

### Key Results
Improved scalability, robustness, sample efficiency vs standard BO in LLM-guided molecular discovery. No multi-objective results.

### Overlap With My Work
Not a direct competitor -- no MOBO, no Pareto front, no scalarization. Complementary: their MCTS acquisition search could pair with qSTCH-Set for multi-objective molecular BO. Aspuru-Guzik group -- influential, worth monitoring for MOBO follow-up. Future work framing: qSTCH-Set + LLM priors for DMTA loops.

### BibTeX
```bibtex
@article{Chen2025llmbo,
  title   = {Informing Acquisition Functions via Foundation Models for Molecular Discovery},
  author  = {Chen, Qi and Ramos, Fabio and Aspuru-Guzik, Al\'{a}n and Shkurti, Florian},
  journal = {arXiv preprint arXiv:2512.13935},
  year    = {2025},
  url     = {https://arxiv.org/abs/2512.13935}
}
```

---

## [A Sober Look at LLMs for Material Discovery](https://arxiv.org/abs/2402.05015)
**Added:** 2026-02-21
**Authors:** Agustinus Kristiadi, Felix Strieth-Kalthoff, Marta Skreta, Pascal Poupart, Alan Aspuru-Guzik, Geoff Pleiss
**Venue:** arXiv preprint (2024)
**Citations:** 47
**Threat Level:** 1 � Single-objective BO with LLM features; orthogonal to multi-objective scalarization

### Method
Treats LLMs as fixed feature extractors for standard GP surrogate models. Uses parameter-efficient finetuning + Bayesian neural networks to get proper posterior uncertainty. Rigorous Bayesian treatment -- no heuristics.

### Gap They Claim
Existing LLM-for-BO uses non-Bayesian, point-estimated LLMs with unreliable uncertainty. LLMs only useful for molecular BO if domain-pretrained or finetuned.

### Key Results
Domain-specific LLMs outperform general LLMs as BO surrogates. General LLMs (GPT-4 etc.) are NOT reliably useful for principled BO. 47 citations in ~1 year -- well-received reality check.

### Overlap With My Work
Completely orthogonal -- better GP surrogate vs our better acquisition strategy. Key insight for our motivation: domain-pretrained LLM features + qSTCH-Set acquisition = natural DMTA stack. We sidestep their "only useful with domain pretraining" concern entirely since qSTCH-Set doesn't rely on LLM priors at all. Cite when discussing surrogate models for molecular BO in intro.

### BibTeX
```bibtex
@article{Kristiadi2024soberlook,
  title   = {A Sober Look at {LLMs} for Material Discovery: Are They Actually Good for {Bayesian} Optimization Over Molecules?},
  author  = {Kristiadi, Agustinus and Strieth-Kalthoff, Felix and Skreta, Marta and Poupart, Pascal and Aspuru-Guzik, Al\'{a}n and Pleiss, Geoff},
  journal = {arXiv preprint arXiv:2402.05015},
  year    = {2024},
  url     = {https://arxiv.org/abs/2402.05015}
}
```
