# Competitive Analysis: arXiv:2412.05265

**Analyzed:** 2026-02-18  
**Shared by:** Rodrigo Vargas (Slack, Feb 16)  
**Analyst:** Automated competitive intelligence  

---

## Paper Identification

| Field | Value |
|-------|-------|
| **Title** | Reinforcement Learning: An Overview |
| **Authors** | Kevin P. Murphy |
| **arXiv ID** | 2412.05265 |
| **First submitted** | December 6, 2024 |
| **Latest version** | v5 (December 1, 2025) |
| **Type** | Survey / Textbook monograph (~9 MB) |
| **Subjects** | cs.AI, cs.LG |
| **Venue** | arXiv preprint (not a conference paper) |

## Content Summary

This is a **massive RL survey/monograph** by Kevin Murphy (Google), expanding on chapters 34–35 of his textbook *Probabilistic Machine Learning: Advanced Topics* (2023). It covers:

- Value-based RL (DQN, Q-learning, Rainbow, etc.)
- Policy-based RL (REINFORCE, PPO, SAC, TD3, etc.)
- Model-based RL (MPC, MCTS, AlphaZero, Dreamer, etc.)
- Other topics: offline RL, hierarchical RL, imitation learning, distributional RL
- LLMs and RL (RLHF, LLMs-for-RL)
- General RL / AIXI / universal AGI

It is a **pedagogical overview of the entire RL field**, not a research contribution proposing a new method.

---

## Threat Assessment: Specific Questions

| Question | Answer | Threat? |
|----------|--------|---------|
| Does it do Set-based optimization? | **No.** It covers single-agent sequential decision making. No multi-objective set-based formulations. | ❌ No |
| Does it use STCH scalarization? | **No.** Tchebycheff / smooth Tchebycheff scalarization is not mentioned. The paper focuses on single-objective reward maximization. | ❌ No |
| Does it handle m>6 objectives? | **No.** Multi-objective optimization is essentially absent. The paper discusses single-reward MDPs. | ❌ No |
| Does it use Bayesian Optimization? | **Barely.** BO is mentioned in passing (Section 1.2.6.2) as one canonical example of an optimization problem, alongside best-arm identification, active learning, and SGD. No new BO methods are proposed. | ❌ No |
| Is it from a top venue? | **No.** It is an arXiv preprint / textbook supplement, not a NeurIPS/ICML/ICLR submission. | ❌ No |

---

## Relevance to Our Work (qSTCHSet)

**Relevance: NONE.**

This paper operates in a completely different domain:

| Dimension | Murphy (2412.05265) | Our Work (qSTCHSet) |
|-----------|---------------------|---------------------|
| **Problem** | Sequential decision making (MDPs) | Multi-objective black-box optimization |
| **Method** | RL algorithms (value/policy/model-based) | Bayesian optimization with GP surrogates |
| **Objectives** | Single scalar reward | Many (m>6) simultaneous objectives |
| **Key innovation** | None (survey paper) | Smooth Tchebycheff set-based acquisition function |
| **Evaluation budget** | Millions of interactions | ~100–500 expensive evaluations |
| **Framework** | Custom RL libraries | BoTorch / GPyTorch |

There is zero methodological overlap. The paper does not address multi-objective optimization, scalarization methods, Pareto front approximation, or sample-efficient optimization of expensive black-box functions.

---

## Verdict

### 🟢 IRRELEVANT — Not a threat, not an opportunity

**3-sentence summary:**

This paper is a comprehensive RL survey by Kevin Murphy covering value-based, policy-based, and model-based reinforcement learning for sequential decision making. It is **not a threat** because it operates in an entirely different domain (single-objective RL / MDPs) with zero overlap to our work on set-based smooth Tchebycheff scalarization for many-objective Bayesian optimization. We should **ignore it** — no need to cite, position against, or respond to this paper.

---

## Action Items

- [ ] **No action required** for our NeurIPS 2026 submission
- [ ] Confirm with Rodrigo whether this was the intended paper — it's possible the arXiv ID was mistyped or a different paper was meant (perhaps something on multi-objective RL or BO?)
- [ ] If Rodrigo meant a different paper, re-run this analysis on the correct ID

---

## Possible Misidentification

Given that this paper is completely unrelated to our work, it's worth considering whether the arXiv ID `2412.05265` was correct. Some papers that *would* be relevant and were posted around the same time:

- Papers on multi-objective BO with scalarization
- Papers on many-objective optimization
- Papers extending Lin et al.'s STCH work

**Recommendation:** Ask Rodrigo to confirm the paper ID or share the title/first author.
