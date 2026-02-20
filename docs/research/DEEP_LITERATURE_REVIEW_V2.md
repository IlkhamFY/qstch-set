# Deep Literature Review: Smooth Tchebycheff Scalarization & Many-Objective BO
**Date:** 2026-02-19
**Focus:** Set-based scalarization, Smooth Tchebycheff, Many-Objective Bayesian Optimization (MOBO)

## 1. Core Theoretical Foundation: Smooth Tchebycheff Scalarization

The foundation of our method rests on the **Smooth Tchebycheff (STCH)** scalarization, which relaxes the non-smooth `max` operator of the classical Tchebycheff scalarization into a differentiable Log-Sum-Exp form.

### **Lin et al. (ICML 2024 / ICLR 2025)**
- **Paper:** "Smooth Tchebycheff Scalarization for Multi-Objective Optimization" (Lin et al., ICML 2024) and "Few-Shot Many-Objective Optimization via Smooth Tchebycheff Set Scalarization" (Lin et al., ICLR 2025).
- **Key Contribution (Single Point):** Proposed $g^{STCH}_\mu(x|\lambda) = \mu \log \sum_i \exp(\lambda_i (f_i(x) - z^*_i)/\mu)$.
  - **Differentiability:** Makes the scalarization gradients Lipschitz continuous, enabling standard gradient-based optimizers (GD, L-BFGS) to converge to Pareto stationary points with $O(1/\epsilon)$ rates.
  - **Pareto Guarantee:** Proved that minimizers of $g^{STCH}$ are weakly Pareto optimal for any $\mu > 0$, and Pareto optimal if weights are strictly positive.
- **Key Contribution (Set-Based):** The "Few-for-Many" setup (ICLR 2025).
  - **Problem:** How to cover a high-dimensional Pareto front ($m \gg 5$) with a small set of $K$ solutions?
  - **Solution:** **STCH-Set**. Instead of finding one point, optimize a set $X = \{x^{(1)}, ..., x^{(K)}\}$ to minimize the aggregate scalarization:
    $$ g^{STCH-Set}_\mu(X) = \mu \log \sum_{i=1}^m \exp \left( \frac{\lambda_i (\text{smin}_k f_i(x^{(k)}) - z^*_i)}{\mu} \right) $$
  - **Mechanism:** The inner `smin` (smooth minimum) assigns the best candidate in the batch to each objective. The outer `smax` (smooth maximum) ensures the worst-satisfied objective is improved. This forces the set $X$ to coordinate: different $x^{(k)}$ specialize in different $f_i$ to minimize the global worst-case deviation.

## 2. Bayesian Optimization Context

### **Pires & Coelho (2025)**
- **Paper:** "MOBO-OSD: Many-Objective Bayesian Optimization with Orthogonal Search Directions" (and related work on Composite BO).
- **Approach:** They integrated the single-point STCH scalarization into **Composite Bayesian Optimization**.
  - **Method:** $g^{STCH}$ is treated as a composite function $h(f(x))$. They use the chain rule (Asturdillo & Frazier, 2019) to compute gradients of the acquisition function.
  - **Limitation:** This is a **single-point** approach ($q=1$). It finds one Pareto point at a time. To approximate the front, they must sample random $\lambda$ vectors (like ParEGO).
  - **Scaling Gap:** For $m \gg 5$, random weight sampling is inefficient. Most weights cluster in the center of the simplex (the "corners problem"), failing to cover the extent of the high-dimensional front.

### **The Gap: Set-Based Acquisition**
- **Existing:** 
  - **qNParEGO:** Batch selection ($q > 1$) but effectively $q$ independent searches with random weights. No coordination.
  - **qEHVI:** Theoretically sound set-based utility (hypervolume), but computationally intractable for $m > 5$ ($2^m$ scaling).
- **Our Contribution (qSTCH-Set):**
  - We bring **Lin et al.'s STCH-Set** concept into **Bayesian Optimization**.
  - We define a **joint acquisition function** $\alpha(X_K)$ over a batch of size $K=m$.
  - **Mechanism:** The acquisition function value depends on the *joint* predictive distribution of the batch. The `smin` operator inside the scalarization naturally handles the "who covers what" assignment problem effectively during acquisition optimization.
  - **Result:** We acquire $m$ points that *collectively* push the Pareto front outward in all $m$ directions simultaneously.

## 3. Related Concepts

### **Many-Objective BO (MaOBO)**
- **Definition:** Optimization with $m > 4$ objectives.
- **Challenges:** 
  - **Visualization:** Impossible to see trade-offs.
  - **Hypervolume:** Calculation is \#P-hard.
  - **Selection:** "Knee points" become hard to define.
- **State of the Art:**
  - **Random Scalarization (ParEGO):** The default fallback. Slow convergence.
  - **Trust Region Methods (MORBO):** Scale in $d$ (inputs), not necessarily $m$ (objectives).
  - **Entropy Search (MESMO/JES):** suffer from sampling costs in high $m$.

### **Set-Based Acquisition Functions**
- **Concept:** Optimizing utility $U(\{x_1, ..., x_q\})$ rather than $\sum U(x_i)$.
- **Examples:** qEHVI (joint hypervolume), qKG (joint information gain).
- **qSTCH-Set vs. qEHVI:** qSTCH-Set is a *scalarization-based* set acquisition. It approximates the hypervolume-like behavior (corner coverage) via a smooth, differentiable proxy that scales linearly ($O(Km)$) instead of exponentially.

## 4. Synthesis for the Paper
The narrative is clear:
1. **Lin et al.** solved the gradient-based many-objective problem with STCH-Set.
2. **Pires & Coelho** brought single-point STCH to BO.
3. **qSTCH-Set** completes the picture: **Set-based STCH for BO**. It is the scalable answer to qEHVI for $m > 5$.
