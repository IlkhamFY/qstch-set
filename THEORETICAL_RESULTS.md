# Theoretical Results: STCH-Set Scalarization for Bayesian Optimization

**Context:** These propositions formalize the theoretical properties of using Smooth Tchebycheff Set (STCH-Set) scalarization (Lin et al., ICLR 2025) as an acquisition function within Bayesian optimization (BO). They connect the Pareto optimality guarantees from the original paper to the GP-based BO setting, establish computational complexity advantages, and characterize the smoothness properties that enable gradient-based acquisition optimization.

---

## Notation

- $\mathcal{X} \subseteq \mathbb{R}^d$: input domain
- $\boldsymbol{f}(\boldsymbol{x}) = (f_1(\boldsymbol{x}), \ldots, f_m(\boldsymbol{x}))$: $m$ black-box objective functions (minimization)
- $\mathcal{GP}_i$: independent Gaussian process surrogate for $f_i$, $i = 1, \ldots, m$
- $\mu_i^{(n)}(\boldsymbol{x}), \sigma_i^{(n)}(\boldsymbol{x})$: GP posterior mean and standard deviation for objective $i$ after $n$ observations
- $\boldsymbol{X}_K = \{\boldsymbol{x}^{(k)}\}_{k=1}^K$: candidate batch of $K$ points
- $\boldsymbol{\lambda} \in \Delta^{m-1}_{++} = \{\boldsymbol{\lambda} \mid \sum_i \lambda_i = 1, \lambda_i > 0 \; \forall i\}$: preference vector
- $\boldsymbol{z}^* \in \mathbb{R}^m$: ideal (reference) point
- $\mu > 0$: smoothing parameter

The STCH-Set scalarization (Lin et al., Eq. 12, simplified with uniform $\mu$):
$$g_\mu^{(\text{STCH-Set})}(\boldsymbol{X}_K \mid \boldsymbol{\lambda}) = \mu \log\left(\sum_{i=1}^m \exp\left(\frac{\lambda_i\left(-\mu \log\left(\sum_{k=1}^K \exp\left(-f_i(\boldsymbol{x}^{(k)})/\mu\right)\right) - z_i^*\right)}{\mu}\right)\right)$$

The corresponding BO acquisition (to be maximized):
$$\alpha_{\text{STCH-Set}}(\boldsymbol{X}_K) = -g_\mu^{(\text{STCH-Set})}(\boldsymbol{X}_K \mid \boldsymbol{\lambda})$$

where $f_i$ is replaced by posterior samples or the posterior mean $\mu_i^{(n)}$.

---

## Proposition 1: Pareto Optimality of STCH-Set Acquisition Maximizers Under the GP Posterior Mean

**Proposition 1.** *Let $\boldsymbol{\mu}^{(n)}(\boldsymbol{x}) = (\mu_1^{(n)}(\boldsymbol{x}), \ldots, \mu_m^{(n)}(\boldsymbol{x}))$ denote the GP posterior mean vector after $n$ observations, and define the posterior-mean STCH-Set scalarization:*
$$\hat{g}_\mu^{(n)}(\boldsymbol{X}_K \mid \boldsymbol{\lambda}) := g_\mu^{(\text{STCH-Set})}(\boldsymbol{X}_K \mid \boldsymbol{\lambda})\big|_{f_i = \mu_i^{(n)}}$$

*Suppose $\boldsymbol{\lambda} \in \Delta^{m-1}_{++}$ and Assumptions 1--2 of Lin et al. hold for $\hat{g}_\mu^{(n)}$. Then:*

*(a) Every solution $\boldsymbol{x}^{(k)}$ in any optimal batch $\boldsymbol{X}_K^* = \arg\min_{\boldsymbol{X}_K} \hat{g}_\mu^{(n)}(\boldsymbol{X}_K \mid \boldsymbol{\lambda})$ is Pareto optimal with respect to the surrogate objectives $\boldsymbol{\mu}^{(n)}$.*

*(b) If the GP posteriors are consistent—i.e., $\mu_i^{(n)}(\boldsymbol{x}) \xrightarrow{a.s.} f_i(\boldsymbol{x})$ uniformly on $\mathcal{X}$ as $n \to \infty$—then for any $\varepsilon > 0$, there exists $N$ such that for all $n \geq N$, the solutions in $\boldsymbol{X}_K^*$ are $\varepsilon$-Pareto optimal with respect to the true objectives $\boldsymbol{f}$.*

**Proof Sketch.**

*(a)* The posterior mean $\mu_i^{(n)}$ is a smooth (in fact, $C^\infty$ for standard kernels) function $\mathcal{X} \to \mathbb{R}$. The STCH-Set scalarization treats these as the objective functions. Since $\lambda_i > 0$ for all $i$, the hypotheses of Theorem 2 of Lin et al. (2025) are satisfied directly. The conclusion follows: all solutions in $\boldsymbol{X}_K^*$ are Pareto optimal with respect to $(\mu_1^{(n)}, \ldots, \mu_m^{(n)})$. $\square$

*(b)* Under GP posterior consistency (which holds, e.g., for Matérn kernels on compact $\mathcal{X}$ with a dense observation sequence; see Chowdhury & Gopalan, 2017; Srinivas et al., 2010), we have $\|\mu_i^{(n)} - f_i\|_\infty \to 0$ a.s. for each $i$. Let $\boldsymbol{x}^{(k)} \in \boldsymbol{X}_K^*$ be Pareto optimal for $\boldsymbol{\mu}^{(n)}$, meaning there is no $\boldsymbol{x} \in \mathcal{X}$ with $\mu_i^{(n)}(\boldsymbol{x}) \leq \mu_i^{(n)}(\boldsymbol{x}^{(k)})$ for all $i$ and strict inequality for some $i$. If $\boldsymbol{x}^{(k)}$ were $\varepsilon$-dominated by some $\boldsymbol{x}'$ under $\boldsymbol{f}$ (i.e., $f_i(\boldsymbol{x}') \leq f_i(\boldsymbol{x}^{(k)}) - \varepsilon$ for all $i$), then for $n$ large enough that $\|\mu_i^{(n)} - f_i\|_\infty < \varepsilon/2$ for all $i$, we would have $\mu_i^{(n)}(\boldsymbol{x}') < \mu_i^{(n)}(\boldsymbol{x}^{(k)})$ for all $i$, contradicting Pareto optimality under $\boldsymbol{\mu}^{(n)}$. $\square$

**Remark.** Part (a) is a direct instantiation of Lin et al. Theorem 2 and requires no new proof machinery. Part (b) connects it to the BO setting via standard GP consistency results. Note that this does *not* provide regret bounds—it is a qualitative guarantee that the acquisition function searches in the right region of the Pareto front as the model improves.

---

## Proposition 2: Computational Complexity of STCH-Set vs. Hypervolume-Based Acquisition

**Proposition 2.** *Consider a multi-objective BO problem with $m$ objectives, a candidate batch of size $K$ ($=q$ in BoTorch notation), and $N_{\text{MC}}$ Monte Carlo posterior samples.*

*(a) The STCH-Set acquisition function $\alpha_{\text{STCH-Set}}$ can be evaluated in $O(N_{\text{MC}} \cdot K \cdot m)$ time and $O(K \cdot m)$ space (per MC sample).*

*(b) The expected hypervolume improvement (EHVI) acquisition function requires $O(N_{\text{MC}} \cdot m \cdot P^{m-1})$ time in the worst case for computing the hypervolume contribution, where $P$ is the size of the current Pareto set approximation. For fixed $P$, the hypervolume computation is $\#P$-hard in $m$ (Bringmann & Friedrich, 2010).*

*(c) Hence, for fixed $K$, the per-evaluation complexity of STCH-Set scales as $\Theta(m)$ in the number of objectives, while hypervolume-based methods scale super-polynomially (exponentially in the worst case) in $m$.*

**Proof.**

*(a)* The STCH-Set computation (Eq. 12/13 of Lin et al.) consists of:
1. **Weighted deviations:** $R_{ik} = \lambda_i(f_i(\boldsymbol{x}^{(k)}) - z_i^*) / \mu$ for each $i \in [m], k \in [K]$: $O(Km)$ operations.
2. **Smooth min over $k$:** For each objective $i$, compute $\text{logsumexp}(-R_{i,1:K})$: $O(K)$ per objective, $O(Km)$ total.
3. **Smooth max over $i$:** Compute $\text{logsumexp}$ over the $m$ smooth-min values: $O(m)$ operations.

Total: $O(Km)$ per MC sample. With $N_{\text{MC}}$ samples: $O(N_{\text{MC}} \cdot Km)$. Space per sample: $O(Km)$ for the $R_{ik}$ matrix.

*(b)* Computing the exact hypervolume of a point set in $\mathbb{R}^m$ is $\#P$-hard for general $m$ (Bringmann & Friedrich, 2010). The best known exact algorithms (e.g., the inclusion-exclusion or WFG algorithm by While et al., 2012) run in $O(2^m)$ or $O(n^{m/2})$ time. Even the fast $m=2,3$ cases require $O(P \log P)$ and $O(P^2)$ respectively.

*(c)* The ratio of complexities is $O(Km) / O(2^m) \to 0$ as $m \to \infty$. $\square$

**Remark.** In the many-objective regime ($m \geq 10$), hypervolume-based methods become computationally intractable while STCH-Set remains linear in $m$. This is the primary practical motivation for scalarization-based approaches in many-objective BO. Note that ParEGO (Knowles, 2006) also achieves $O(m)$ per-evaluation cost via random scalarization, but uses a single scalarization weight per iteration and does not jointly optimize a batch for coverage.

---

## Proposition 3: Smoothness and Gradient-Based Acquisition Optimization

**Proposition 3 (Smoothness of STCH-Set).** *Let all objective functions $f_i: \mathcal{X} \to \mathbb{R}$ be $L$-smooth (i.e., $L$-Lipschitz continuous gradients). Then:*

*(a) The STCH-Set scalarization $g_\mu^{(\text{STCH-Set})}(\boldsymbol{X}_K \mid \boldsymbol{\lambda})$ is $C^\infty$ with respect to all decision variables $\boldsymbol{X}_K = (\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(K)}) \in \mathcal{X}^K$, for any $\mu > 0$.*

*(b) The gradient of $g_\mu^{(\text{STCH-Set})}$ with respect to $\boldsymbol{x}^{(k)}$ is:*
$$\nabla_{\boldsymbol{x}^{(k)}} g_\mu^{(\text{STCH-Set})} = \sum_{i=1}^m w_i \cdot p_{ik} \cdot \nabla f_i(\boldsymbol{x}^{(k)})$$
*where the weights are:*
$$w_i = \frac{\exp\left(\lambda_i(R_i^{\min} - z_i^*)/\mu\right)}{\sum_{j=1}^m \exp\left(\lambda_j(R_j^{\min} - z_j^*)/\mu\right)}, \quad p_{ik} = \frac{\exp\left(-f_i(\boldsymbol{x}^{(k)})/\mu\right)}{\sum_{\ell=1}^K \exp\left(-f_i(\boldsymbol{x}^{(\ell)})/\mu\right)}$$
*with $R_i^{\min} = -\mu \log\sum_k \exp(-f_i(\boldsymbol{x}^{(k)})/\mu)$. Here $w_i \in (0,1)$ is the softmax attention weight over objectives, and $p_{ik} \in (0,1)$ is the softmin attention weight of solution $k$ for objective $i$.*

*(c) In contrast, the (non-smooth) TCH-Set scalarization $g^{(\text{TCH-Set})}$ is non-differentiable at any $\boldsymbol{X}_K$ where either (i) the maximum over objectives is attained by multiple objectives, or (ii) the minimum over solutions for some objective is attained by multiple solutions. At such points, only subgradients exist, and subgradient methods converge at rate $O(1/\sqrt{T})$ vs. $O(1/T)$ for gradient descent on smooth functions (Nesterov, 2005).*

*(d) When used as a BO acquisition function with GP posterior samples, $g_\mu^{(\text{STCH-Set})}$ composed with GP sample paths inherits $C^\infty$ smoothness (for standard kernels such as SE or Matérn with $\nu > 1$), enabling the use of L-BFGS or Adam for acquisition optimization. This is in contrast to standard Chebyshev scalarizations used in ParEGO, which require restart-based optimization strategies to handle non-differentiability.*

**Proof Sketch.**

*(a)* $g_\mu^{(\text{STCH-Set})}$ is a composition of $\exp$, $\log$, and $\sum$ applied to smooth functions $f_i$. Since $\mu > 0$, no division by zero occurs, and the $\log$ arguments are strictly positive (as they are sums of exponentials). By the chain rule, the composition is $C^\infty$. $\square$

*(b)* Direct computation via the chain rule. The outer $\text{logsumexp}$ contributes the softmax weights $w_i$. The inner $(-\mu \cdot \text{logsumexp})$ contributes the softmin weights $p_{ik}$. The gradient with respect to $\boldsymbol{x}^{(k)}$ picks up only terms involving $f_i(\boldsymbol{x}^{(k)})$, yielding the stated formula. $\square$

*(c)* The $\max$ function $\max(a_1, \ldots, a_m)$ has a kink whenever two or more arguments are equal. Similarly for $\min$. Standard results in non-smooth analysis (Clarke, 1990) give that the subdifferential is the convex hull of the gradients of the active components. Subgradient descent on Lipschitz convex functions converges at $O(1/\sqrt{T})$ (Nesterov, 2004, §3.2), whereas gradient descent on $L$-smooth convex functions converges at $O(1/T)$ (or $O(1/T^2)$ with acceleration). $\square$

*(d)* GP sample paths with SE kernel are $C^\infty$ a.s.; with Matérn-$\nu$ kernel they are $C^{\lceil \nu \rceil - 1}$ a.s. (Rasmussen & Williams, 2006, §4.2). Composing with the $C^\infty$ STCH-Set scalarization preserves differentiability up to the smoothness of the GP samples. Standard BO implementations (BoTorch) use the reparameterization trick for MC acquisition functions, and the STCH-Set gradient formula in (b) is fully compatible with automatic differentiation. $\square$

**Remark.** The gradient formula in (b) provides useful intuition: solution $\boldsymbol{x}^{(k)}$ is updated as a *weighted combination of objective gradients*, where the weights $w_i \cdot p_{ik}$ are large when (i) objective $i$ is close to being the worst-case bottleneck ($w_i$ large), and (ii) solution $k$ is the best solution for objective $i$ ($p_{ik}$ large). As $\mu \to 0$, $w_i \to \mathbf{1}[i = \arg\max]$ and $p_{ik} \to \mathbf{1}[k = \arg\min]$, recovering the hard TCH-Set subgradient.

---

## Summary Table

| Property | STCH-Set (Ours) | EHVI/NEHVI | ParEGO |
|---|---|---|---|
| Pareto optimality guarantee | ✅ Thm 2 (Lin et al.) + Prop 1 | ✅ (hypervolume monotonicity) | ✅ (Chebyshev theory) |
| Per-evaluation complexity in $m$ | $O(Km)$ — **linear** | $O(2^m)$ — **exponential** | $O(m)$ — linear |
| Batch joint optimization | ✅ (inherent in set formulation) | ✅ (q-EHVI) | ❌ (one point per weight) |
| Differentiable | ✅ ($C^\infty$ for $\mu > 0$) | ✅ (differentiable formulations exist) | ❌ (hard max) |
| Many-objective scalability ($m > 10$) | ✅ | ❌ | ✅ (but no batch coverage) |

---

## References

- Lin, X., Liu, Y., Zhang, X., Liu, F., Wang, Z., & Zhang, Q. (2025). Few for Many: Tchebycheff Set Scalarization for Many-Objective Optimization. *ICLR 2025*. arXiv:2405.19650v3.
- Bringmann, K., & Friedrich, T. (2010). Approximating the volume of unions and intersections of high-dimensional geometric objects. *Computational Geometry*, 43(6-7), 601–610.
- Chowdhury, S. R., & Gopalan, A. (2017). On kernelized multi-armed bandits. *ICML 2017*.
- Srinivas, N., Krause, A., Kakade, S., & Seeger, M. (2010). Gaussian process optimization in the bandit setting: No regret and experimental design. *ICML 2010*.
- Knowles, J. (2006). ParEGO: A hybrid algorithm with on-line landscape approximation for expensive multiobjective optimization problems. *IEEE Trans. Evol. Comput.*, 10(1), 50–66.
- Nesterov, Y. (2005). Smooth minimization of non-smooth functions. *Math. Program.*, 103(1), 127–152.
- Clarke, F. H. (1990). *Optimization and Nonsmooth Analysis*. SIAM.
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- While, L., Bradstreet, L., & Barone, L. (2012). A fast way of calculating exact hypervolumes. *IEEE Trans. Evol. Comput.*, 16(1), 86–95.
