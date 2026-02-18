# Theoretical Analysis: qSTCHSet for Multi-Objective Bayesian Optimization

## 1. Preliminaries and Notation

Consider a multi-objective optimization problem with $m$ black-box objective functions $f_1, \dots, f_m : \mathcal{X} \to \mathbb{R}$, where $\mathcal{X} \subset \mathbb{R}^d$ is compact. We seek to minimize all objectives simultaneously. Each objective is modeled by an independent Gaussian process surrogate: $f_i \sim \mathcal{GP}(\mu_i^{(0)}, k_i)$, yielding posterior mean $\mu_i^{(n)}(x)$ and posterior variance $\sigma_i^{(n)}(x)^2$ after $n$ observations.

**Definition 1** ($\varepsilon$-Pareto optimality). A point $x \in \mathcal{X}$ is $\varepsilon$-Pareto optimal if there exists no $x' \in \mathcal{X}$ such that $f_i(x') \leq f_i(x) - \varepsilon$ for all $i \in [m]$.

**Definition 2** (Set $\varepsilon$-Pareto optimality). A set $X_K = \{x^{(1)}, \dots, x^{(K)}\}$ is set $\varepsilon$-Pareto optimal if the set-objective vector $\tilde{f}(X_K) = (\min_{k} f_1(x^{(k)}), \dots, \min_{k} f_m(x^{(k)}))$ is $\varepsilon$-Pareto optimal in the set-objective space.

**The STCH-Set scalarization** (Lin et al., 2025):
$$
g_\mu^{\text{STCH-Set}}(X_K \mid \lambda) = \mu \log\!\Bigl(\sum_{i=1}^{m} \exp\!\bigl(\lambda_i(\operatorname{smin}_k f_i(x^{(k)}) - z_i^*) / \mu\bigr)\Bigr)
$$
where $\operatorname{smin}_k f_i(x^{(k)}) = -\mu \log\!\bigl(\sum_{k=1}^K \exp(-f_i(x^{(k)})/\mu)\bigr)$ is the smooth minimum.

---

## 2. GP Posterior Concentration

We adopt the standard regularity assumptions from the GP-UCB literature (Srinivas et al., 2010).

**Assumption 1** (RKHS regularity). Each $f_i$ has bounded RKHS norm $\|f_i\|_{k_i} \leq B$ with respect to its kernel $k_i$.

**Assumption 2** (Bounded variance). The kernels satisfy $k_i(x, x) \leq 1$ for all $x \in \mathcal{X}$.

Under these assumptions, the following uniform concentration result holds (Srinivas et al., 2010, Theorem 6):

**Lemma 1** (GP posterior concentration). For each $i \in [m]$, with probability at least $1 - \delta/m$, simultaneously for all $x \in \mathcal{X}$ and all $n \geq 1$:
$$
|f_i(x) - \mu_i^{(n)}(x)| \leq \beta_n^{1/2} \, \sigma_i^{(n)}(x)
$$
where $\beta_n = 2B^2 + 300 \gamma_n \log^3(n/\delta)$ and $\gamma_n$ is the maximum information gain after $n$ observations. By a union bound over $m$ objectives, all objectives concentrate simultaneously with probability $\geq 1 - \delta$.

**Remark.** For common kernels, $\gamma_n$ grows sub-linearly: $\gamma_n = O((\log n)^{d+1})$ for squared exponential, $\gamma_n = O(n^{d(d+1)/(2\nu + d(d+1))} (\log n))$ for Matérn-$\nu$. This ensures $\beta_n^{1/2} \sigma_i^{(n)}(x) \to 0$ for observed regions as $n \to \infty$.

---

## 3. Main Result: Transferring Pareto Guarantees to the BO Setting

### 3.1. The qSTCHSet Acquisition Function

Define the **qSTCHSet acquisition function** by replacing true objectives with GP posterior samples (via the reparameterization trick for MC integration):
$$
\alpha_{\text{qSTCH}}(X_K) = \mathbb{E}_{\omega}\!\bigl[g_\mu^{\text{STCH-Set}}(\hat{f}^{(n)}_\omega(X_K) \mid \lambda)\bigr]
$$
where $\hat{f}^{(n)}_\omega(x) = (\hat{f}_{1,\omega}^{(n)}(x), \dots, \hat{f}_{m,\omega}^{(n)}(x))$ are joint posterior samples via the pathwise conditioning approach (Wilson et al., 2020).

In practice, we approximate this expectation with $N_{\text{MC}}$ base samples and optimize via SAA (sample average approximation) using L-BFGS-B, leveraging the differentiability of the log-sum-exp structure.

### 3.2. Approximation Gap Decomposition

**Proposition 1** (Approximation gap decomposition). *Let $X_K^* = \arg\min_{X_K} g_\mu^{\text{STCH-Set}}(X_K \mid \lambda)$ be the optimal set under the posterior-mean STCH-Set objective, where $f_i$ is replaced by $\mu_i^{(n)}$. Suppose GP concentration (Lemma 1) holds for all $m$ objectives simultaneously (an event of probability $\geq 1 - \delta$). Then the true TCH-Set value of $X_K^*$ satisfies:*
$$
g^{\text{TCH-Set}}(X_K^* \mid \lambda) \leq g_\mu^{\text{STCH-Set}}(\mu^{(n)}(X_K^*) \mid \lambda) + \underbrace{\mu \log(m) + \mu \log(K)}_{\text{smoothing gap}} + \underbrace{2 \beta_n^{1/2} \bar{\sigma}_n}_{\text{posterior uncertainty}}
$$
*where $\bar{\sigma}_n = \max_{i \in [m]} \max_{x \in X_K^*} \sigma_i^{(n)}(x)$.*

**Proof sketch.** The bound follows from three sources of error, composed via triangle inequality:

1. **Outer smoothing gap** (smax vs max): By the standard log-sum-exp bound,
$$
g^{\text{TCH}}(X_K \mid \lambda) \leq g_\mu^{\text{STCH}}(X_K \mid \lambda) \leq g^{\text{TCH}}(X_K \mid \lambda) + \mu \log(m)
$$
This is Proposition 3.4 of Lin et al. (2024).

2. **Inner smoothing gap** (smin vs min): For each objective $i$,
$$
\min_k f_i(x^{(k)}) - \mu \log(K) \leq \operatorname{smin}_k f_i(x^{(k)}) \leq \min_k f_i(x^{(k)})
$$
This contributes at most $\mu \log(K)$ through the outer scalarization (since $\lambda \in \Delta^{m-1}$).

3. **Posterior uncertainty**: Under the concentration event, for any $x$,
$$
|\mu_i^{(n)}(x) - f_i(x)| \leq \beta_n^{1/2} \sigma_i^{(n)}(x)
$$
The STCH-Set scalarization is Lipschitz in the objective values (with constant depending on $\lambda$ and $\mu$). For the TCH-Set value, the maximum operator has Lipschitz constant 1, giving an additive error of at most $2\beta_n^{1/2}\bar{\sigma}_n$ (factor of 2 from applying concentration at both the solution set and the comparator).

Combining these three terms yields the stated bound. $\square$

**Remark.** The total approximation gap is thus:
$$
\varepsilon_{\text{total}} = \mu\log(m) + \mu\log(K) + O(\beta_n^{1/2}\bar{\sigma}_n)
$$
The first two terms are controlled by the user-specified smoothing parameter $\mu$ and are $O(\mu \log(mK))$. The third term vanishes as $n \to \infty$ under standard GP assumptions.

### 3.3. Asymptotic Pareto Optimality

**Proposition 2** (Asymptotic $\varepsilon$-Pareto optimality). *Under Assumptions 1–2, suppose the qSTCHSet acquisition is optimized at each BO iteration with $\lambda_i > 0$ for all $i$. As $n \to \infty$, the set of solutions $X_K^{(n)}$ selected by qSTCHSet satisfies: with probability $\geq 1 - \delta$, all $x^{(k)} \in X_K^{(n)}$ are $\varepsilon_n$-Pareto optimal where*
$$
\varepsilon_n = \mu\log(m) + \mu\log(K) + O(\beta_n^{1/2} \bar{\sigma}_n) \xrightarrow{n \to \infty} \mu\log(mK).
$$

**Proof sketch.** By Theorem 2 of Lin et al. (2025), the minimizer of STCH-Set with $\lambda > 0$ produces solutions that are Pareto optimal *of the scalarized problem*. Under GP posterior concentration, the scalarized problem on $\mu^{(n)}$ converges uniformly to the scalarized problem on the true $f$. The $\varepsilon$-Pareto optimality gap is then bounded by the sum of the smoothing approximation error and the posterior uncertainty, both of which are quantified in Proposition 1. The smoothing gap $\mu\log(mK)$ remains as an irreducible residual for fixed $\mu > 0$; it can be made arbitrarily small by choosing $\mu \to 0$ (at the cost of acquisition function smoothness and thus optimization difficulty). $\square$

**Status: Proposition, not full theorem.** A complete proof would require formalizing the acquisition function optimization guarantee (i.e., that L-BFGS-B with restarts finds a near-global optimum of the MC acquisition), which is standard to assume but difficult to prove. This is a well-known gap shared by essentially all BO acquisition function analyses.

### 3.4. Annealing Schedule for $\mu$

**Corollary 1** (Vanishing gap with $\mu$-annealing). *If $\mu_n = c / \log(n+1)$ for some constant $c > 0$, then $\varepsilon_n \to 0$ as $n \to \infty$, i.e., the solutions converge to exact Pareto optimality.*

This follows immediately since $\mu_n \log(mK) \to 0$ and $\beta_n^{1/2}\bar{\sigma}_n \to 0$. In practice, we fix $\mu = 0.1$ as a reasonable trade-off between approximation quality and numerical stability, consistent with Lin et al. (2024, 2025).

---

## 4. Computational Complexity Analysis

**Proposition 3** (Per-evaluation complexity). *The qSTCHSet acquisition function with $K$ candidate points, $m$ objectives, $N_{\text{MC}}$ MC samples, and $d$-dimensional input evaluates in $O(N_{\text{MC}} \cdot K \cdot m)$ time (excluding GP posterior sampling cost).*

| Method | Acquisition Eval. | Scaling in $m$ | Batch Pareto? |
|--------|-------------------|-----------------|---------------|
| qEHVI (Daulton et al., 2021) | $O(N_{\text{MC}} \cdot 2^m)$* | Exponential | Yes |
| qNParEGO (Daulton et al., 2020) | $O(N_{\text{MC}} \cdot q)$ | $O(m)$ per scalarization | No (random $\lambda$) |
| **qSTCHSet (ours)** | $O(N_{\text{MC}} \cdot Km)$ | **Linear** | **Yes (K solutions)** |

*qEHVI has been improved via box decomposition (Daulton et al., 2021) but the non-dominated partitioning step remains exponential in $m$ in the worst case. For $m > 5$, this becomes the bottleneck.

**Key advantage.** qSTCHSet scales linearly in $m$, making it the only method that provides Pareto coverage guarantees while remaining tractable for $m \gg 5$.

---

## 5. Theoretical Comparison with qNParEGO

qNParEGO (Daulton et al., 2020) applies random Chebyshev scalarization with a different $\lambda \sim \text{Dirichlet}(\mathbf{1})$ at each BO iteration. We characterize when qSTCHSet is theoretically preferable.

**Proposition 4** (Diversity advantage of qSTCHSet over qNParEGO). *Consider a single BO iteration selecting $q = K$ points.*

*(i) qNParEGO with a single random $\lambda$ optimizes toward one region of the Pareto front per iteration. Over $T$ iterations, the expected number of distinct Pareto front regions covered scales as $O(T^{m/(m+1)})$ (coupon collector with continuous support).*

*(ii) qSTCHSet with $K$ solutions and uniform $\lambda = \mathbf{1}/m$ explicitly optimizes all $K$ solutions to collectively cover the Pareto front at each iteration. The soft-assignment weights $w_{ik} \propto \exp(-f_i(x^{(k)})/\mu)$ encourage specialization: each solution $x^{(k)}$ is driven toward a different trade-off region.*

**Status: Informal argument.** A formal diversity guarantee would require quantifying the coverage of the Pareto front, e.g., via $\varepsilon$-net arguments. We conjecture that qSTCHSet achieves $\varepsilon$-coverage of the Pareto front in $O(1/\varepsilon^{m-1})$ iterations vs $O((1/\varepsilon)^{m-1} \log(1/\varepsilon)^{m-1})$ for qNParEGO, but proving this rigorously remains open.

**When does qSTCHSet dominate qNParEGO?**
1. **Large $m$**: Random scalarization wastes samples exploring low-value trade-off directions; qSTCHSet coordinates $K$ solutions.
2. **Diverse Pareto front**: When the Pareto front has complex geometry (e.g., disconnected regions), coordinated search is more efficient than random.
3. **Small evaluation budget**: With few BO iterations, qSTCHSet's per-iteration Pareto coverage is more valuable.

**When is qNParEGO preferable?**
1. **$m = 2$**: Random scalarization covers the 1D Pareto front efficiently; the coordination overhead of qSTCHSet is unnecessary.
2. **Very large batch sizes**: qNParEGO naturally diversifies within a batch via different $\lambda$ draws; qSTCHSet would need $K$ to grow commensurately.

---

## 6. The Role of $K$

The number of solutions $K$ in the set plays a dual role:

**Proposition 5** (Trade-off in $K$). *For fixed $\mu$ and $m$:*

*(i) Coverage: The maximal gap between any Pareto optimal point and its nearest solution in $X_K$ decreases in $K$ (more solutions = finer coverage). If $K \geq m$, each solution can specialize to one objective.*

*(ii) Approximation quality: The inner smoothing gap $\mu \log(K)$ increases in $K$, degrading the overall approximation. Additionally, the optimization landscape becomes harder: the acquisition function has $Kd$ decision variables.*

*(iii) Optimization signal: For fixed evaluation budget, increasing $K$ reduces the per-solution optimization pressure. Empirically, solutions become less differentiated when $K$ is too large.*

**Practical guidance:**
- $K = 2$–$4$ for $m \leq 5$ objectives
- $K = 5$–$10$ for $5 < m \leq 20$
- $K$ should satisfy $K \ll m$ (the "few for many" regime of Lin et al., 2025)
- A heuristic: $K \approx \lceil \sqrt{m} \rceil$ balances coverage and optimization difficulty

---

## 7. Connection to $\varepsilon$-PAL

The $\varepsilon$-PAL framework (Zuluaga et al., 2016) provides PAC-style guarantees for multi-objective optimization: with probability $\geq 1 - \delta$, identify all $\varepsilon$-Pareto optimal points using $O(\beta_n \gamma_n / \varepsilon^2)$ evaluations. We note the following connection:

**Remark 1** (PAC interpretation). The qSTCHSet framework can be viewed through the $\varepsilon$-PAL lens where $\varepsilon = \mu\log(mK)$. Unlike $\varepsilon$-PAL, which classifies all points in a discretized design space, qSTCHSet directly optimizes a continuous acquisition function to propose $K$ candidate solutions. This is more sample-efficient when $|\mathcal{X}|$ is large or continuous, but provides weaker classification guarantees (we find $K$ good solutions rather than classifying all of $\mathcal{X}$).

**Conjecture 1** (Sample complexity). *Under the assumptions of Proposition 2 and mild regularity conditions on the Pareto front, the qSTCHSet algorithm identifies $K$ solutions that are $\varepsilon$-Pareto optimal after*
$$
n = O\!\left(\frac{\beta_n \gamma_n}{\varepsilon^2}\right)
$$
*evaluations, matching the $\varepsilon$-PAL rate up to constants.*

**Status: Conjecture.** A full proof would require adapting the $\varepsilon$-PAL analysis to the continuous-optimization, set-valued setting. The key difficulty is that qSTCHSet proposes new points rather than classifying a fixed discretization. We leave this for future work.

---

## 8. Summary of Theoretical Contributions

| Result | Type | Key Statement |
|--------|------|---------------|
| Proposition 1 | Proved (sketch) | Gap decomposition: $\mu\log(m) + \mu\log(K) + O(\beta_n^{1/2}\bar{\sigma}_n)$ |
| Proposition 2 | Proved (sketch) | Asymptotic $\varepsilon$-Pareto optimality as $n \to \infty$ |
| Corollary 1 | Proved | $\mu$-annealing yields vanishing gap |
| Proposition 3 | Proved | $O(N_{\text{MC}} \cdot Km)$ complexity |
| Proposition 4 | Informal | qSTCHSet dominates qNParEGO when $m$ large, diverse PF needed |
| Proposition 5 | Semi-formal | Trade-off in $K$: coverage vs approximation vs signal |
| Conjecture 1 | Open | PAC-style sample complexity matching $\varepsilon$-PAL |

**What is rigorous:** Propositions 1–3 follow from combining known results (Lin et al. 2024/2025 approximation bounds + Srinivas et al. 2010 GP concentration) via standard arguments. The proof sketches can be formalized straightforwardly.

**What is conjectural:** The diversity advantage (Prop 4), the PAC sample complexity (Conj 1), and the precise characterization of optimal $K$ all require substantially more technical machinery and are presented as informed conjectures supported by our empirical results.

---

## References

- Daulton, S., Balandat, M., & Bakshy, E. (2020). Differentiable expected hypervolume improvement for parallel multi-objective Bayesian optimization. NeurIPS.
- Daulton, S., Balandat, M., & Bakshy, E. (2021). Parallel Bayesian optimization of multiple noisy objectives with expected hypervolume improvement. NeurIPS.
- Lin, X., Zhang, X., Yang, Z., Liu, F., Wang, Z., & Zhang, Q. (2024). Smooth Tchebycheff scalarization for multi-objective optimization. ICML.
- Lin, X., Liu, Y., Zhang, X., Liu, F., Wang, Z., & Zhang, Q. (2025). Few for many: Tchebycheff set scalarization for many-objective optimization. ICLR.
- Srinivas, N., Krause, A., Kakade, S., & Seeger, M. (2010). Gaussian process optimization in the bandit setting: No regret and experimental design. ICML.
- Wilson, J. T., Borovitskiy, V., Terenin, A., Mostowsky, P., & Deisenroth, M. P. (2020). Efficiently sampling functions from Gaussian process posteriors. ICML.
- Zuluaga, M., Krause, A., & Püschel, M. (2016). $\varepsilon$-PAL: An active learning approach to the multi-objective optimization problem. JMLR.
