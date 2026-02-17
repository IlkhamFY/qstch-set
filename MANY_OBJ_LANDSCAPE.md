# Many-Objective Bayesian Optimization: Landscape Analysis

*Research conducted 2026-02-17. Purpose: Assess whether our StochBO paper addresses a real gap.*

---

## 1. The REAL Frontier of Many-Objective BO

### Maximum objectives tested in BO literature

| Method | Max m tested | Input dim | Notes |
|--------|-------------|-----------|-------|
| **MORBO** (Daulton et al., 2022) | ~4 objectives | 146-222 params | Focus was high-D *input* space, not high-D objective space |
| **qNEHVI** (Daulton et al., 2021) | ~4-6 | moderate | Polynomial in batch size, but HV is exponential in m |
| **Kalai-Smorodinsky BO** | up to 9 | - | Single compromise point, not Pareto front |
| **MaO-BO** (Konakovic Lukovic et al., 2021) | ~6-10 | - | Objective reduction via GP posterior similarity |
| **ParEGO** | ~5-8 practical | - | Random scalarization; simplex sampling degrades with m |
| **MESMO/PFES/JES** | ~4-6 typical | - | Information-theoretic; no explicit m scaling results |

**Key finding: NO BO paper handles m > 10 objectives.** The practical ceiling is around 4-6 objectives for most methods, with a few pushing to ~10 via specialized tricks.

### Why the ceiling exists

1. **Hypervolume is #P-hard** in number of objectives — exponential computation
2. **Pareto front size** explodes: with m objectives, almost every point becomes non-dominated
3. **GP modeling**: need one GP per objective → linear scaling in modeling but exponential in acquisition
4. **Simplex coverage**: for ParEGO-style methods, uniform sampling on (m-1)-simplex becomes hopeless

### MORBO specifics
- MORBO's innovation is **high-dimensional input spaces** (d=146-222), NOT many objectives
- Uses local trust regions with independent GPs
- Tested on problems with a handful of objectives (likely 2-4 in experiments)
- Does NOT solve the many-objective problem

## 2. How People Handle Many Objectives Today

### In BO specifically:
- **Random scalarization (ParEGO-style)**: Convert to single-objective via random Chebyshev weights. Scales poorly beyond ~5-8 objectives due to simplex coverage.
- **Hypervolume-based (EHVI, qNEHVI)**: State of the art for 2-4 objectives. Exponential in m.
- **Information-theoretic (MESMO, PFES, JES)**: Invariant to objective reparameterization. JES is best; still tested only on 2-4 objectives.
- **Objective reduction (MaO-BO)**: Detect redundant objectives via GP posterior similarity, remove them. Only approach explicitly targeting many-objective BO.
- **Decomposition**: Break into single-objective subproblems along search directions.

### What's NOT done in BO (but done in EMO):
- PCA on objective space (common in NSGA-III variants)
- Objective clustering
- Reference point-based methods (well-developed in evolutionary, barely in BO)

### Practical limits
- **m ≤ 4**: Standard MOBO works well (qNEHVI, EHVI)
- **m = 5-8**: Requires tricks (scalarization, decomposition, objective reduction)
- **m > 10**: **Nobody does this in BO. Full stop.**

## 3. Information-Theoretic Methods (MESMO, PFES, JES)

- **MESMO**: Max-value entropy search for multi-objective; special case of PFES with crude box decomposition approximation
- **PFES**: Pareto Front Entropy Search; broader entropy-based approach
- **JES**: Joint Entropy Search; upper bound on convex combination of PES and MES; best performer

JES advantages:
- Invariant to objective reparameterization (unlike HV-based methods)
- Comparable cost to NEHVI per evaluation
- Batch extension with e⁻¹ regret bound (submodular)

**Scaling with m**: None of these papers demonstrate scaling beyond ~4-6 objectives. The entropy computation involves Pareto front sampling which suffers from same curse of dimensionality.

## 4. ScopeBO

**NOT related to many-objective optimization.** ScopeBO is a tool from the Doyle lab (UCLA) for automating **reaction scope selection** in organic chemistry — it uses BO to decide which substrates/conditions to test. Unrelated to our work.

## 5. Drug Discovery: The Real-World Motivation

### How many properties matter?

| Scope | Number of endpoints | Source |
|-------|-------------------|--------|
| Core ADMET for lead optimization | **20-30** | Industry standard |
| ADMET-AI platform | **41 ADMET + 8 physicochemical** | Valence Labs |
| ADMETlab 3.0 (comprehensive) | **119 endpoints** | Academic tool |
| ADMET-score (practical) | **18 predicted properties** | admetSAR |
| Early screening suite | **50+** properties | Various pharma |

### The Novartis bPK Score
- **"Beyond PK" (bPK) score**: Deep learning model that takes predicted ADMET profile as input and predicts balanced potency-PK relationship
- Trained to optimize the tradeoff between drug potency and pharmacokinetic properties
- Represents the industry approach: **collapse many objectives into composite scores**
- Published on GitHub (Novartis/beyond-PK-score)

### How industry actually handles many-objective drug design

1. **MPO scores**: Collapse 5-20 properties into a single weighted score (Pfizer CNS MPO, Novartis bPK, etc.)
2. **Desirability functions**: Sigmoidal transformations per property, then geometric mean
3. **Rules/filters**: Lipinski Ro5, Veber rules, PAINS — hard constraints, not objectives
4. **Expert judgment**: Medicinal chemists mentally weight properties based on project needs
5. **Sequential optimization**: Optimize potency first, then fix ADMET issues, then iterate

**Critical insight**: Industry does NOT do multi-objective optimization with 20+ objectives simultaneously. They always reduce to:
- A single composite score (MPO), OR
- 2-4 key objectives + constraints, OR
- Sequential single-objective campaigns

### Is there REAL demand for m > 10 objective BO?

**The honest answer: Yes for the problem, but current practice avoids it through reduction.**

The need exists: a real drug candidate must satisfy 20-50 property criteria. But:
- MPO scores are lossy — they encode human assumptions about tradeoffs
- Different projects weight properties differently
- Composite scores miss Pareto-optimal solutions that a chemist might prefer

**The gap we could fill**: If you could do true multi-objective optimization with m=20-50 and return K=3-5 diverse candidates covering the Pareto front, that's strictly better than collapsing to a single MPO score. The question is whether the computational cost is justified.

## 6. "Few Solutions for Many Objectives" — Real-World Demand

### Drug discovery: K=3-5 candidates from 50+ properties
- **YES, this is exactly what happens**: Drug projects advance 3-5 lead candidates to preclinical
- Each must satisfy ~50 property criteria
- Currently done via MPO scores + expert judgment
- A method that says "here are your 3 best diverse candidates, each optimal on different property tradeoffs" is **extremely valuable**
- This is literally what medchem teams do manually over months

### Materials science
- Battery materials: optimize ionic conductivity, stability, processability, cost, energy density (~5-10 objectives)
- Alloys: strength, ductility, corrosion resistance, density, cost (~5-8 objectives)
- Less extreme than drug discovery, but still relevant

### Other domains
- LLM fine-tuning: accuracy vs. cost vs. latency vs. safety vs. fairness (growing)
- Engineering design: already using MORBO for multi-objective with high-D inputs

## 7. Assessment: Does Our Paper Have a Real Audience?

### ✅ Strong arguments FOR:

1. **Clear gap**: Nobody does BO with m > 10. The need exists (drug discovery, 20-50 properties).
2. **Industry pain point**: MPO scores are known to be lossy; medicinal chemists want better tools.
3. **"K solutions" framing is natural**: Drug projects literally advance K=3-5 candidates.
4. **Computational barrier is well-known**: HV is #P-hard; any method that bypasses this has value.
5. **Unique positioning**: We'd be the FIRST BO paper to credibly operate at m > 10.

### ⚠️ Concerns:

1. **"Nobody does it" could mean "nobody needs it"** — maybe MPO scores are good enough?
2. **Benchmark problem**: No standard benchmarks exist for m > 10 objective BO. We'd need to create them.
3. **Evaluation metric**: Can't use hypervolume at m > 10 (too expensive). Need alternative metrics.
4. **Surrogate accuracy**: With m=50 objectives, you need 50 GPs. Are they accurate enough with limited data?
5. **Pareto front is meaningless at m=50**: Almost everything is Pareto-optimal. The "K best" framing must be well-defined.

### 🎯 Recommendation:

The paper has a real audience IF:
- We demonstrate on a **realistic drug discovery benchmark** (not just synthetic functions)
- We clearly articulate WHY MPO scores are insufficient
- We show that our K=3-5 solutions are meaningfully better than top-K from an MPO ranking
- We handle the "what does Pareto-optimal even mean at m=50" question head-on
- We compare against the actual industry approach (composite scores) not just other BO methods

**The strongest pitch**: "For the first time, Bayesian optimization that handles 20-50 objectives simultaneously, returning a small diverse set of candidates — demonstrated on real drug discovery data."

---

## Summary Table: State of the Art

| Aspect | Current SOTA | Our target |
|--------|-------------|------------|
| Max objectives in BO | ~4-6 (up to 10 with tricks) | 20-50 |
| Acquisition function | qNEHVI, JES | StochBO (scalarization-based?) |
| HV computation | Exact up to ~8, approximate beyond | Bypass entirely |
| Drug discovery MPO | Single composite score | True multi-objective |
| Solutions returned | Pareto front (grows exponentially) | K=3-5 diverse candidates |
| Benchmarks | DTLZ, ZDT (≤10 obj) | Need drug discovery benchmarks |

---

*This analysis suggests a genuine gap with real-world motivation, but execution must be drug-discovery-grounded to be compelling.*
