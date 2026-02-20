# Paper Execution Plan — Honest Assessment

## What We Have (Solid)
- ✅ Deep literature map (30+ papers categorized, gap analysis, 2x2 matrix)
- ✅ Deep reads of Lin ICML24, Lin ICLR25 (all theorems, equations)
- ✅ Pires & Coelho analysis (competitive positioning)
- ✅ Working code: qSTCHSet acquisition function, 82 tests passing
- ✅ Paper draft (main.tex) — structure exists but needs major rewrite
- ✅ references.bib — 23 entries
- ✅ 7 reviewer attacks identified with defenses
- ✅ Benchmark m=5 done (STCH-Set wins), m=8 running

## What's Missing / Risky

### 1. PAPERS NOT ACTUALLY READ (only summarized from abstracts/search)
- [ ] **Pires & Coelho full paper** — we analyzed from code, not from reading the paper
- [ ] **Astudillo & Frazier 2019** — cited as framework but never deeply read
- [ ] **Ishikura et al. ICML 2025 (PFES-VLB)** — claims many-obj, could be competitor
- [ ] **Daulton et al. 2022 MORBO** — need to verify exactly what m they tested
- [ ] **Wang et al. ICML 2024 (ε-PoHVI)** — need to verify claims
- [ ] Need to check Lin et al. 2025 **supplementary material** for BO experiments

### 2. THEORY GAPS
- [ ] Our "theoretical results" (theory_notes.md) — are these rigorous or handwavy?
- [ ] No formal convergence guarantee yet
- [ ] Need to verify: does STCH-Set with GP posterior samples preserve Pareto optimality?
- [ ] Approximation bound: μ·log(m) + μ·log(K) — is this tight? Can we improve?

### 3. EXPERIMENTAL GAPS
- [ ] m=8, m=10 still running (need final numbers)
- [ ] No mu ablation yet
- [ ] No K ablation yet
- [ ] No wall-clock comparison
- [ ] No comparison with PFES-VLB or other recent many-obj methods
- [ ] No real-world application demo (drug discovery ADMET)

### 4. WRITING GAPS
- [ ] main.tex intro/abstract need sharpening
- [ ] Related work section needs careful positioning
- [ ] Theory section needs formal proofs or honest disclaimers
- [ ] Conclusion needs future work that's honest about limitations

---

## Execution Plan (Priority Order)

### Phase 1: Deep Paper Analysis (TODAY)
**Goal: Read ALL key papers properly, not from summaries. Catch any misinterpretation.**

Papers to deep-read (in order):
1. **Lin et al. ICLR 2025 supplementary** — check if they mention BO at all
2. **Astudillo & Frazier 2019** — composite BO framework (our theoretical backbone)
3. **Pires & Coelho 2025** — verify our positioning is accurate
4. **Ishikura et al. 2025 PFES-VLB** — potential competitor for many-obj
5. **Daulton et al. 2022 MORBO** — verify m limitation claims

**Method: Download PDFs, read methods sections, extract exact claims/limitations.**
**Verification: For each paper, write "what they claim" vs "what they actually show" vs "our advantage".**

### Phase 2: Theory Hardening (NEXT)
- Formalize Proposition: STCH-Set-BO consistency (convergence as data grows)
- Verify approximation bounds with GP posterior
- Write clean proofs or cite composability of existing results

### Phase 3: Experiments Completion (PARALLEL WITH BENCHMARKS)
- Wait for m=8, m=10 results
- Run mu ablation at m=5 (quick)
- Run K ablation at m=5 (quick)
- Wall-clock timing table

### Phase 4: Paper Rewrite (AFTER 1-3)
- Sharpen narrative based on actual deep reading
- Ensure every claim is backed by theorem or experiment
- Related work that's fair and accurate

---

## Sub-Agent Strategy

### Good use of sub-agents:
- Downloading and extracting paper PDFs (parallel)
- Running ablation experiments (parallel, independent)
- Generating LaTeX tables/figures from results

### BAD use of sub-agents (avoid):
- Writing theory sections (needs coherent reasoning across sections)
- Interpreting papers (risk of hallucination — main agent must read)
- Positioning against competitors (needs holistic view)

### Error prevention:
- Sub-agents produce ARTIFACTS (JSON, data, figures), not PROSE
- Main agent synthesizes artifacts into paper
- Every claim gets a "source" tag: [theorem X from paper Y] or [experiment Z]
- Cross-check: if sub-agent says "paper X claims Y", verify by reading paper X

---

## Key Risk: Misinterpretation
The biggest risk is claiming something about a competitor that's wrong, or claiming a theoretical property we don't have. Prevention:
1. Read the actual paper, not just the abstract
2. For every "they don't do X" claim, search the paper for X
3. For every "we prove X" claim, write the full proof or say "we conjecture"
4. Have a "claims verification checklist" before submission
