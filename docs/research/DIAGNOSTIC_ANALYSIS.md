# Diagnostic Analysis: STCH-Set Underperformance at m=5

## The Problem
DTLZ2 m=5, K=5 (q=5), 2 seeds, 20 iters:
- STCH-Set: HV = 5.044 ± 0.236
- qNParEGO: HV = 5.250 ± 0.045

## Root Cause Hypotheses (Ranked by Likelihood)

### 1. K = m is the WRONG regime for STCH-Set [HIGH CONFIDENCE]
Lin et al.'s whole premise is **K << m** ("5 solutions for 100+ objectives").
When K=5 and m=5, each solution covers ~1 objective — there's no collaborative
advantage. STCH-Set degenerates to something similar to single-point scalarization.

The set-based advantage comes from the **min over K** in the inner aggregation:
for each objective i, the best candidate covers it. When K=m, this "assignment"
is trivially 1-to-1. When K << m, multiple objectives get assigned to the same
candidate, and the smooth min encourages **specialization** — that's the magic.

**Prediction**: At m=10 with K=5, STCH-Set should win because:
- Each candidate must cover ~2 objectives
- The collaborative allocation matters
- qNParEGO's random scalarizations become increasingly unlikely to cover all objectives

### 2. mu=0.1 may be wrong for BO setting
Lin et al. use mu with gradient descent where they can take many steps.
In BO, we only evaluate the acquisition function — the mu controls how
"crisp" the min-max assignment is. 

mu=0.1 with m=5: smoothing gap ≤ 0.1*log(5) + 0.1*log(5) ≈ 0.32
mu=0.01 with m=5: gap ≤ 0.01*log(5) + 0.01*log(5) ≈ 0.032

But TOO small mu causes numerical issues in logsumexp. Need ablation.

### 3. Joint optimization of q=5 candidates is HARD
qNParEGO optimizes ONE candidate at a time with random scalarization (sequential greedy).
qSTCHSet optimizes ALL 5 JOINTLY. This is a much harder optimization problem:
- 5*d dimensional instead of d dimensional
- The acquisition landscape is more complex
- L-BFGS-B with limited restarts may not find good solutions

**Fix**: Increase num_restarts and raw_samples significantly for joint optimization.

### 4. qNParEGO's random scalarizations provide accidental diversity
At m=5, random Chebyshev weights on the 4-simplex actually provide decent 
coverage. The sequential greedy with different random weights naturally
produces diverse candidates. This is a feature, not a bug.

### 5. Hypervolume metric may not be the right metric for STCH-Set
STCH-Set optimizes worst-case objective coverage, not hypervolume.
It could be that our candidates have BETTER worst-case coverage but 
WORSE hypervolume. We should also report:
- Worst-case objective value (max_i min_k f_i(x_k))
- Coverage metric from Lin et al.
- IGD+ (inverted generational distance)

## Experimental Plan

### A. Scaling experiment (CRITICAL)
- m = {3, 5, 8, 10, 15, 20} on DTLZ2
- K = 5 for all (the "few for many" setting)
- Compare: STCH-Set, qNParEGO, random
- Skip qEHVI for m > 6 (exponential cost)
- Report: HV, worst-case objective, wall-clock time

### B. K ablation
- m = 10, K = {3, 5, 10}
- Hypothesis: K=5 optimal for m=10, K=3 for m≥20

### C. mu ablation  
- m = 10, K = 5, mu = {0.01, 0.05, 0.1, 0.5, 1.0}
- Find sweet spot between numerical stability and approximation quality

### D. Alternative metric
Report STCH-Set scalarization value directly alongside HV.
If our candidates have better STCH-Set values but worse HV, that's
an important insight about what we're actually optimizing.

## Reframing the Paper Narrative

If STCH-Set doesn't dominate qNParEGO at m=5, that's FINE. The story is:

**"For m ≤ 5, existing methods work. For m > 5, they don't. We do."**

qEHVI: O(2^m) — impractical for m > 6
qNParEGO: random scalarizations lose coverage guarantees as m grows
STCH-Set: O(Km) with Pareto guarantees for all K solutions

The crossover point is likely m ∈ {8, 10}. This is still a strong paper
because NO ONE does BO beyond m≈5 (confirmed in MANY_OBJ_LANDSCAPE.md).

## Action Items
1. Run m=8,10 ASAP — these are the headline results
2. Implement worst-case objective metric
3. mu ablation at m=10
4. Consider sequential STCH-Set (optimize one at a time like ParEGO but with STCH-Set objective)
