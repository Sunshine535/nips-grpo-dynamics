---
version: v3
timestamp: 2026-04-27T10:00:00Z
pipeline: research-pipeline / idea-discovery
status: gate-1-pending
target_venue: NeurIPS 2027 or ICML 2027
---

# Idea Report: Post-TASA Research Directions

## Context

After comprehensive audit of the nips-grpo-dynamics project:
- **Current results** (TASA + Verified CE Replay, 55.0% vs 27.8% on GSM8K) are real but **not novel enough for NeurIPS main** — binary degeneracy is known (DAPO, 2025.03), experience replay has 6+ competing papers (RePO, RLEP, ExGRPO, R3, FreshPER, DyJR), and the field has 30+ GRPO variant papers.
- **The one promising signal**: D (positive CE, 55.0%) > C (contrastive, 49.0%) on full GSM8K, suggesting positive-only imitation outperforms contrastive learning under sparse binary rewards.
- **Key anchor paper**: "Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?" (NeurIPS 2025 Best Paper Runner-Up) showed RLVR narrows the base model's reasoning distribution — pass@k decreases during training.

---

## Idea 1 (Top Pick): Can Supervised Replay Break the RLVR Ceiling?

### Core Question

The NeurIPS 2025 Best Paper Runner-Up proved that RLVR (GRPO, PPO, Reinforce++) does NOT create new reasoning capabilities — it only redistributes probability mass from the base model's existing distribution, narrowing pass@k while improving pass@1. **Can supervised experience replay preserve the base model's reasoning diversity (pass@k) while still gaining the exploitation benefits (pass@1)?**

### Hypothesis

GRPO + CE Replay maintains higher pass@k than pure GRPO because:
1. The replay bank accumulates diverse verified solutions across training (per-prompt, hash-deduplicated)
2. CE loss directly reinforces these diverse solutions, counteracting GRPO's distribution-narrowing effect
3. The SFT signal targets SPECIFIC verified solutions (not just the current policy mode), preserving exploration paths

### Why This Is Novel

| Aspect | Prior Work | Our Contribution |
|--------|-----------|-----------------|
| RLVR ceiling | Demonstrated (NeurIPS 2025 BPR-Up) | Test whether replay can break/mitigate it |
| Experience replay in GRPO | RePO, RLEP, R3 focus on convergence speed/efficiency | We measure pass@k (capability preservation), not just pass@1 |
| Positive > contrastive | Known generally (RAFT literature) | First to connect to pass@k ceiling via mechanistic analysis |
| Replay bank diversity | Not studied | Hash-deduplicated per-prompt banks as diversity reservoir |

**No existing paper connects experience replay to the RLVR ceiling question.** This positions us as a direct follow-up to the NeurIPS 2025 Best Paper Runner-Up.

### Pilot Experiment (Immediate, ~8 GPU-hours)

Using EXISTING checkpoints from sage_minimal_abc:

1. Take 3 models: Base Qwen3.5-9B, B (TASA-only), D (TASA+CE Replay)
2. Sample k=50 completions per question on 200 GSM8K test questions (temperature=1.0)
3. Compute pass@k for k in {1, 5, 10, 25, 50}
4. Compare curves:
   - If Base pass@50 > B pass@50: confirms NeurIPS 2025 (RLVR narrows)
   - If D pass@50 > B pass@50: **CE Replay preserves diversity** -> GREEN LIGHT
   - If D pass@50 approx Base pass@50: **CE Replay fully preserves ceiling** -> STRONG SIGNAL

### Full Experimental Plan (2-3 months)

**Phase A: Multi-scale pass@k analysis**
- Models: Qwen3-1.7B, Qwen2.5-7B, Qwen3.5-9B, Qwen3.5-27B
- Benchmarks: GSM8K, MATH-500, MBPP
- Methods: Base, GRPO, GRPO+CE Replay, GRPO+Contrastive, RePO, RLEP
- Metrics: pass@k (k=1,5,10,25,50,100), solution diversity (unique correct solutions)
- Seeds: 5 per condition

**Phase B: Replay bank dynamics**
- Track bank size, diversity (unique solutions per prompt), age distribution across training
- Correlation: bank diversity vs pass@k preservation
- Ablation: bank size limit (K=1,2,5,10), freshness weighting

**Phase C: Mechanism analysis**
- Token-level: which tokens are reinforced differently by CE vs RL?
- Distribution analysis: KL(pi_trained || pi_base) trajectory for GRPO vs GRPO+CE
- Gradient decomposition: how does CE signal interact with RL signal?

**Phase D: Theoretical framework**
- Formalize: replay bank as "anchor distribution" that prevents excessive KL drift
- Prove: under what conditions replay maintains pass@k lower bound
- Connect: to catastrophic forgetting and continual learning literature

### Risk Assessment

- **If pilot is positive** (D preserves pass@k): strong paper, 7-8/10 potential
- **If pilot is negative** (D also narrows): pivot to "even supervised signals can't break the ceiling" — still publishable as an important negative result extending the NeurIPS 2025 BPR-Up
- **If pilot is ambiguous**: need more seeds/models before committing

### Novelty Verification

- Searched: "experience replay pass@k RLVR ceiling" — **0 results**
- Searched: "replay break reasoning boundary base model" — **0 results**
- Closest work: RePO measures pass@1 improvement, not pass@k preservation
- The NeurIPS 2025 paper itself calls for "improved RL paradigms to unlock potential" — we test whether replay is one

**Novelty: CONFIRMED as of 2026-04-27**

---

## Idea 2: The Imitation-Contrast Phase Diagram

### Core Question

When does positive-only learning (SFT/CE) outperform contrastive learning (DPO/pairwise) in RL post-training? Can we characterize this as a function of measurable properties of the task and model?

### Hypothesis

There exists a phase boundary in (reward_density x model_capability x group_size) space:
- Below the boundary: contrastive helps (failures are informative)
- Above the boundary: contrastive hurts (failures contain useful partial reasoning)
- The boundary shifts with task difficulty and model capability

### Evidence from Current Results

D (positive CE, 55.0%) > C (contrastive, 49.0%) at:
- Reward density: binary sparse (approx 40% success rate per group at G=4)
- Model capability: Qwen3.5-9B base (25.5% GSM8K)
- Group size: G=4

### Why This Is Novel

- Individual observations exist (RAFT literature, NSR paper, our D>C)
- But no systematic phase diagram mapping the boundary
- Connects to ongoing SFT vs DPO debate with controlled experiments
- Practical value: tells practitioners which strategy to use

### Experimental Plan (2-3 months)

- Sweep: reward density (continuous -> 5-level -> binary), model pass rate (easy->hard tasks), G in {2,4,8,16}
- Measure: accuracy gap (positive_only - contrastive)
- Models: 3 sizes, 3 benchmarks, 5 seeds per condition
- Analysis: fit phase boundary, theoretical explanation

### Risk Assessment

- **Medium novelty**: individual pieces known, synthesis is new
- **High feasibility**: controlled experiments
- **Medium impact**: useful but not paradigm-shifting
- Estimated score: **5-6/10** for NeurIPS main

---

## Idea 3: Unifying GRPO Training Instabilities via Effective Horizon Theory

### Core Question

Can binary degeneracy, mode collapse, grokking phase transitions, and catastrophic training collapse in GRPO be unified as manifestations of a single underlying mechanism — the effective number of gradient-contributing samples collapsing to zero?

### Hypothesis

When the effective number of informative gradient-contributing samples per batch drops below a critical threshold N*, all known GRPO instabilities emerge:
- N* approx 0: binary degeneracy (all-correct/all-incorrect groups)
- N* < G: mode collapse (insufficient diversity pressure)
- N* oscillates around N*_crit: grokking (sudden phase transition)
- N* -> 0 permanently: catastrophic collapse

### Why This Would Be Impactful

- Unifies 4+ separate phenomena into one framework
- Provides a single diagnostic metric (N_eff) for practitioners
- Explains why different fixes (DAPO, TASA, larger G) all help: they increase N_eff

### Risk Assessment

- **High novelty**: no unified framework exists
- **Low feasibility**: needs deep theory + massive experiments
- **High impact if successful, but very risky**
- Estimated score: **8/10 if proven, 3/10 if hand-wavy**

---

## Ranking

| Rank | Idea | Novelty | Feasibility | Impact | Expected Score |
|------|------|---------|-------------|--------|---------------|
| **1** | **RLVR Ceiling + Replay** | **HIGH** | **HIGH** | **HIGH** | **7-8/10** |
| 2 | Imitation-Contrast Phase Diagram | MEDIUM | HIGH | MEDIUM | 5-6/10 |
| 3 | Unified Instability Theory | HIGH | LOW | HIGH | 3-8/10 (bimodal) |

## Recommendation

**Pursue Idea 1** as the primary direction. It has the best risk-reward profile:
- Directly extends the NeurIPS 2025 Best Paper Runner-Up
- The pilot experiment can be run IMMEDIATELY on existing checkpoints (~8 GPU-hours)
- Positive or negative, the result is publishable
- The pass@k measurement framework is standard and unambiguous
- The story is clean: "RLVR narrows capabilities -> can replay fix this?"

**Use Idea 2** as a secondary analysis that strengthens the paper (the D>C finding provides additional evidence).

**Defer Idea 3** unless theoretical breakthrough emerges during Phase D.

## Immediate Next Step

Run the pilot experiment: measure pass@k on existing base/B/D checkpoints.
If pass@k curves diverge between B and D, we have a green light for the full study.
