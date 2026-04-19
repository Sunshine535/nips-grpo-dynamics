# Literature Landscape — RLVR/GRPO Dynamics (snapshot 2026-04-19)

Context: building on our CSD framework (binary-reward GRPO ≡ estimator-level KL-difference between in-batch τ⁺/τ⁻ + closed-form ρ* + ADQ controller + Q_CSD collapse predictor). This file surveys published work to identify the **unoccupied territory** and concurrent threats.

---

## Concurrent / near-concurrent works to differentiate from

| Paper | Framing | Overlap with our CSD | Differentiation |
|-------|---------|----------------------|------------------|
| [It Takes Two: GRPO Is Secretly DPO (2510.00977)](https://arxiv.org/abs/2510.00977) | GRPO as implicit contrastive + control-variate link to DPO | **"GRPO is contrastive"** claim (abstract level) | They: no τ⁺/τ⁻ KL decomposition, no asymmetric ρ, no closed-form ρ*, no adaptive controller. Their finding: 2-GRPO ≈ 16-GRPO empirically. |
| [GRPO's Effective Loss + Success Amplification (2503.06639)](https://arxiv.org/abs/2503.06639) | Weighted contrastive loss with *temporal* synthetic samples, fixed-point analysis | Touches contrastive structure | Their samples = previous-policy outputs (temporal). Ours = within-batch τ⁺/τ⁻ (structural). They have fixed-point on success probability; we have ρ*. Different math, different controllers. |
| [GRPO Is Secretly a PRM (2509.21154)](https://arxiv.org/abs/2509.21154) | GRPO ⇔ Process Reward Model via prefix sharing + MC | "GRPO is secretly X" pattern | Step-level credit via prefix sharing, not distribution-level KL. Orthogonal to CSD. |
| [RL via Self-Distillation / SDPO (2601.20802)](https://arxiv.org/abs/2601.20802) | Method: feedback-conditioned next-token distillation | Name overlap only | Method not theory. No GRPO-gradient decomposition, no ρ analysis. |
| [GBMPO: Flexible Bregman Divergences for GRPO (2602.04380)](https://arxiv.org/abs/2602.04380) | Replaces KL with Bregman | Generalizes our KL-centric theory | **Risk:** subsumes KL as a special case. We must argue why binary-reward case is uniquely KL-decomposable. |
| [Group-Relative REINFORCE is Secretly Off-Policy (2509.24203)](https://arxiv.org/abs/2509.24203) | Demystifies clipping + importance-sampling role in GRPO | Frames GRPO as off-policy | Clipping focus, not contrastive/distillation focus. |
| [Demystifying GRPO: U-Statistic (2603.01162)](https://arxiv.org/abs/2603.01162) | GRPO gradient as U-statistic | Statistical structure | Does not decompose into τ⁺/τ⁻ form. MSE analysis. |
| [BPR: Does RL Really Incentivize Reasoning Beyond Base? (2504.13837)](https://arxiv.org/abs/2504.13837) | Empirical: acc ≤ pass@k of base (NeurIPS'25) | Our capacity prediction aligns | Empirical paper, our framework explains it theoretically. No conflict. |
| [Transform-Augmented GRPO (2601.22478)](https://arxiv.org/abs/2601.22478) | Improves pass@k via reward mixing + Bernoulli variance | Different mechanism | Reward engineering, not gradient-view. |

---

## Crowded territory (DO NOT propose here without major novelty)

- **Token-level credit assignment**: GTPO, GRPO-λ (multiple papers), TEMPO, λ-GRPO, T-SPMO, TR-GRPO, Beyond Token-Level PG, Execution-Grounded Credit Assignment — **highly saturated**.
- **Curriculum learning for LLM RL**: E2H Reasoner, VCRL (variance-based curriculum), Omni-Thinker, Actor-Curator, VL-Cogito — **saturated**.
- **Experience replay for LLM RL**: [Efficient RL with Experience Replay (2604.08706)](https://arxiv.org/abs/2604.08706), RLEP, Prioritized Replay (2601.02648), Retrospective Replay, difficulty-targeted selection — **saturated**.
- **"GRPO is secretly X" reframings**: DPO, PRM, off-policy REINFORCE — **saturating fast**. Any new "secretly X" claim needs very strong evidence.
- **Mode-collapse mitigation**: DiverseGRPO (image), S-GRPO, failure-mode-of-max-entropy RLHF — growing.
- **Length control / length bias**: GR3 (length rescaling), Dr. GRPO, many others — saturated.

---

## Underexplored structural gaps (candidate novel territory)

### Gap 1. Bayesian/uncertainty-aware ρ controller
Our ADQ estimates ρ* = Cov(g⁺,g⁻)/Var(g⁺) pointwise from noisy EMA statistics. A posterior over ρ* enables principled exploration (Thompson sampling on ρ, UCB bounds) vs. point estimate. Nobody has done this for any GRPO hyperparameter.

### Gap 2. CSD-view extension to continuous / partial-credit rewards
Our Theorem 1 is binary-reward-specific. Code partial credit, multi-step math, agentic tasks increasingly have continuous rewards. The natural generalization is a reward-weighted τ_r(y) ∝ f(r(y)) with ρ(r) as reward-dependent weighting. **No published work has the continuous analog of KL(τ⁻‖π) − ρ·KL(τ⁺‖π).**

### Gap 3. Batch-level CSD diversity certificate + generalization gap theorem
Q_CSD is per-group. A batch-level diversity metric across *different* prompt groups could certify generalization: "more diverse τ⁺ distribution across prompts → smaller generalization gap." Nobody connects GRPO's in-batch structure to generalization theory.

### Gap 4. CSD-guided online rollout curation
Q_CSD predicts collapse risk. USE it at rollout time: reject a batch with low Q_CSD and resample. Unlike existing adaptive-rollout (budget-based, success-rate-based), this uses a *theoretically-derived collapse signal*. Plausible 30-50% compute win.

### Gap 5. Information-geometric CSD: natural-gradient ρ
View ∇L = √(p(1-p))·[g⁻ − ρ·g⁺] on the Fisher-Rao statistical manifold. Natural-gradient preconditioning yields ρ*_natural ≠ ρ*_euclidean; dimension-independent. Complements FR-PPO but for GRPO-with-CSD framing.

### Gap 6. Pareto-ρ frontier for multi-objective RLHF
Multi-reward RLHF: each reward has its own (τ⁺_r, τ⁻_r, ρ_r). Pareto-optimal ρ vector ≠ scalarized weighted sum. Gives new theory + new method.

### Gap 7. Optimal-G theory from CSD (closing "2-GRPO ≈ 16-GRPO" gap)
2510.00977 empirically showed 2-GRPO matches 16-GRPO. CSD gives a clean theoretical explanation: with G=2 and balanced outcomes, the τ⁺/τ⁻ KL-difference is maximally contrastive. Derivable lower bound on G for stable training vs. prompt difficulty.

### Gap 8. CSD-aware reward-model training
Reward models today are trained to match human preferences; they do NOT know their use in CSD. Joint (reward model, policy) optimization with CSD signal-quality as auxiliary loss → reward models that make CSD work better.

### Gap 9. CSD at inference time (self-consistency ↔ training equivalence)
k-sample self-consistency at inference = same τ⁺/τ⁻ structure as training. CSD framework predicts optimal k for self-consistency as function of base-model confidence. Unifies training and inference adaptation.

### Gap 10. τ⁺ / τ⁻ diversity as RLHF-exploration objective (beyond max-entropy)
Standard RL exploration: maximize policy entropy. CSD view says: entropy of τ⁺ matters more than entropy of π. Optimizing H(τ⁺) as explicit exploration objective → new method.

---

## Strategic assessment for NeurIPS best-paper bar

**Best-paper criteria:** deep theory × broad applicability × clean empirical × hard to concurrent-duplicate.

**Top 3 candidates ranked:**
1. **Gap 2 (Continuous-reward CSD)** — broad applicability, theoretical, strong empirical upside on partial-credit code/math. Risk: moderate (concurrent DPO/IPO variants for continuous rewards exist but not with CSD structure).
2. **Gap 4 (CSD-guided online rollout curation)** — immediately practical, theoretically grounded, compute-efficient. Risk: lower-bar novelty than Gap 2.
3. **Gap 7 (Optimal-G theory)** — crisp and surprising, directly fills the gap left by 2510.00977. Risk: narrow scope may hurt "best-paper" prestige.

**Synergy:** A paper combining Gap 2 (continuous-reward CSD theory) + Gap 4 (CSD rollout curation for efficiency) + Gap 7 (optimal-G corollary) would hit three pillars: theory, practice, and surprising finding. That's our strongest shot.

**Direct threats to monitor (concurrent):** 2510.00977 (Oct'25) and 2503.06639 (ongoing v4) are the closest; we must cite both and differentiate in Abstract + §1.
