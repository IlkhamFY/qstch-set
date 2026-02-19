# Competitive Intelligence Update — 2026-02-18

## NEW: MOBO-OSD (NeurIPS 2025)
- **Paper**: "MOBO-OSD: Batch Multi-Objective Bayesian Optimization via Orthogonal Search Directions"
- **OpenReview**: https://openreview.net/forum?id=oRMfTkP6kC
- **What it does**: Batch MOBO using orthogonal search directions on convex hull of individual minima + Pareto front estimation
- **Objectives tested**: 2-6 only
- **Our advantage**: We go to m=10+ (and theoretically 100+). They still use hypervolume-based reasoning. Complementary, not competing.
- **Action**: Must cite and discuss. Positioning: "scales batch size, not objective count"
- **Add to references.bib**

## CONFIRMED: Pires & Coelho (SSRN 2025)
- **Paper**: "Composite Bayesian Optimisation for Multi-Objective Material Design"
- **SSRN**: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5168818
- **What it does**: Single-point STCH in composite BO framework (Astudillo-Frazier)
- **Our advantage**: Set-based (K solutions jointly), not single-point. This is the key differentiator.
- **Status**: Still behind Cloudflare paywall for full paper

## CONFIRMED: PFES-VLB does NOT exist
- Searched ICML 2025 paper list — no paper matching this description
- Was a hallucination in our literature map
- **Action**: Remove from DEEP_LITERATURE_MAP.md

## NO concurrent work found on:
- STCH-Set in BO (our exact contribution)
- Set-based scalarization in BO
- BO with >10 objectives using scalarization

## Our competitive position remains strong:
- 2x2 matrix is accurate
- Bottom-right cell (BO + Set) is truly empty
- MOBO-OSD is the strongest recent MOBO paper but doesn't go beyond m=6
