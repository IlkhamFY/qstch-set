# Claims Verification Report
**Date:** 2026-02-19
**Target:** `paper/main.tex` Claims vs. Benchmark Results

## 1. Primary Claims (Abstract & Intro)

### **Claim 1: "qSTCH-Set outperforms qNParEGO on DTLZ2 (m=5)"**
- **Text:** "On DTLZ2 with $m=5$ objectives (5 seeds, 30 iterations), qSTCH-Set achieves a hypervolume of $6.646 \pm 0.066$, outperforming both qNParEGO ($6.429 \pm 0.254$) and single-point STCH ($6.117 \pm 0.156$)."
- **Data Source:** `projects/stch-botorch/benchmarks/results/dtlz2_m5_results.json`
- **Verification:**
  - **qSTCH-Set:** Mean HV = 6.646, Std = 0.066. (Matches)
  - **qNParEGO:** Mean HV = 6.429, Std = 0.254. (Matches)
  - **STCH-NParEGO:** Mean HV = 6.117, Std = 0.156. (Matches)
- **Status:** **VERIFIED**. The claim is accurate to the third decimal place.

### **Claim 2: "Set-based coordination improves Pareto front coverage"**
- **Text:** "demonstrating that set-based coordination improves Pareto front coverage in the sample-efficient regime."
- **Logic:** The comparison between **qSTCH-Set** (set-based optimization, $K=5$) and **STCH-NParEGO** (single-point, $q=1$) isolates the effect of set coordination. Both use the same smooth scalarization logic, but qSTCH-Set optimizes the *joint* utility of 5 points.
- **Data:** The gap is $6.646 - 6.117 = 0.529$ HV units (approx 8.6% improvement).
- **Status:** **VERIFIED**. The experimental design (including STCH-NParEGO as an ablation) supports this causal claim.

## 2. Scaling Claims (Section 5)

### **Claim 3: "Scaling to m=8" (New Results)**
- **Text:** (To be added/updated in Section 5)
- **Data Source:** `projects/stch-botorch/benchmarks/results/dtlz2_m8_results.json`
- **Verification:**
  - **qSTCH-Set (K=8):** Mean HV = 24.108 ± 0.314
  - **qNParEGO (q=1):** Mean HV = 21.606 ± 0.826
  - **STCH-NParEGO (q=1):** Mean HV = 20.620 ± 1.459
  - **Random:** Mean HV = 18.026 ± 0.196
- **Observation:** The gap widens significantly at $m=8$. qSTCH-Set is ~2.5 HV units better than qNParEGO.
- **Status:** **PENDING UPDATE**. The paper currently has placeholders for m=8. These numbers should be inserted.

## 3. Comparison with Competitors

### **Claim 4: "qEHVI is intractable for m > 5"**
- **Context:** The paper positions qSTCH-Set against qEHVI.
- **Verification:** Standard literature (Daulton et al., 2020) confirms qEHVI complexity is exponential in $m$. Computation for $m=8$ involves $2^8 = 256$ terms per integration step, making it extremely slow or requiring approximation.
- **Status:** **VERIFIED** by literature consensus.

## 4. Conclusion
All quantitative claims in the current draft for $m=5$ are perfectly supported by the JSON logs. The qualitative claims about "coordination" are supported by the ablation study. The $m=8$ results are ready to be integrated and show an even stronger advantage for the proposed method.
