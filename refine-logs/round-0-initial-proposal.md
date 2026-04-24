# Research Proposal: Signed-Measure Contrastive Decomposition for Continuous-Reward GRPO with Training Phase Diagrams

## Problem Anchor
- **Bottom-line problem**: GRPO fails catastrophically with continuous/partial-credit rewards because group-mean baselines give positive advantage to incorrect solutions that happen to be "less wrong" than average. This misleads the policy gradient into reinforcing wrong reasoning.
- **Must-solve bottleneck**: Standard GRPO treats rewards as scalar offsets from group mean. With continuous rewards (e.g., partial credit on MATH), a solution scoring 0.3 in a group averaging 0.2 gets positive advantage despite being wrong. The gradient pushes the model toward "less wrong" rather than "correct."
- **Non-goals**: (1) Token-level credit assignment (saturated). (2) Experience replay/SFT hybrid (saturated). (3) "GRPO is secretly X" reframings. (4) Curriculum learning.
- **Constraints**: 8x A800-80GB, Qwen3.5-9B + LoRA r=64, TRL 0.14, 2-week implementation, NeurIPS 2026.
- **Success condition**: SM-CSD demonstrably outperforms standard GRPO, Dr. GRPO, CoRPO, and RLOO on MATH with partial-credit rewards, and the phase diagram provides actionable stability insights that predict training outcomes.

## Technical Gap

### Why current methods fail
1. **Standard GRPO**: Advantage = r - mean(r_group). With continuous rewards, "less wrong" gets reinforced.
2. **Dr. GRPO**: Removes std normalization but still uses group-mean baseline. Same "less wrong" problem.
3. **CoRPO**: Adds a correctness-threshold baseline (r - c instead of r - mean), which helps but still treats the advantage as a scalar. It does not decompose the reward signal into distinct positive and negative teacher distributions.
4. **DAPO**: Focuses on entropy collapse and degenerate groups, not on continuous reward structure.

### What is missing
No method constructs two separate normalized probability distributions from reward excess and deficit. The key insight: continuous rewards carry directional information -- not just "how good" but "in which direction to move." A signed-measure decomposition extracts this directional signal.

### Core technical claim
Decomposing continuous rewards into positive (tau+) and negative (tau-) contrastive teacher distributions, then optimizing the signed KL difference, provides a principled and provably better-directed gradient than scalar-advantage GRPO methods.

## Method Thesis
- **One-sentence thesis**: SM-CSD converts continuous rewards into two contrastive teacher distributions (tau+ from reward excess, tau- from reward deficit) and optimizes their signed KL difference, providing directionally correct gradients where scalar-advantage methods mislead.
- **Why smallest adequate intervention**: Only changes the advantage computation -- no new modules, no reward models, no replay buffers.
- **Why timely**: MATH and code-generation increasingly use partial-credit rewards; existing GRPO methods were designed for binary rewards.

## Contribution Focus
- **Dominant contribution**: Signed-Measure CSD -- a new advantage computation for continuous-reward GRPO that decomposes rewards into contrastive teacher distributions.
- **Supporting contribution**: (alpha,beta) Training Phase Diagram -- systematic stability map of GRPO training dynamics.
- **Explicit non-contributions**: No new architectures, no reward models, no replay mechanisms, no curriculum.

## Proposed Method

### Complexity Budget
- **Frozen / reused**: Qwen3.5-9B base model, LoRA, TRL GRPOTrainer pipeline, tokenizer, dataset loading
- **New trainable components**: ZERO -- SM-CSD only changes the advantage computation (no new parameters)
- **Tempting additions intentionally not used**: Replay bank (saturated), process reward model (too expensive), curriculum scheduler

### System Overview
```
Prompt -> Generate G completions -> Compute continuous rewards r1...rG
    -> Partition: S+ = {y : r(y) > c}, S- = {y : r(y) <= c}
    -> Compute signed-measure weights:
        w+_i = max(r_i - c, 0) / sum_j max(r_j - c, 0)    (reward-excess distribution)
        w-_i = max(c - r_i, 0) / sum_j max(c - r_j, 0)    (reward-deficit distribution)
    -> Loss = sum_i w-_i * log pi(y_i|x) - rho * sum_i w+_i * log pi(y_i|x) + beta * KL(pi || pi_ref)
    -> Backprop through LoRA parameters
```

### Core Mechanism: Signed-Measure CSD

**Input**: G completions {y1,...,yG} with continuous rewards {r1,...,rG}, correctness threshold c.

**Step 1 -- Partition**: Split completions into positive set S+ = {i : r_i > c} and negative set S- = {i : r_i <= c}.

**Step 2 -- Construct teacher distributions**:
- tau+: over S+, weight w+_i = (r_i - c) / sum_{j in S+} (r_j - c)
- tau-: over S-, weight w-_i = (c - r_i) / sum_{j in S-} (c - r_j)

**Step 3 -- Signed KL loss**:
```
L_SM = -sum_{i in S-} w-_i * log pi(y_i|x)  +  rho * sum_{i in S+} w+_i * log pi(y_i|x)
```
First term: push away from tau- (weighted by "how wrong"). Second term: pull toward tau+ (weighted by "how right"). rho controls the balance.

**Step 4 -- Degenerate group handling**:
- If S+ = empty (all wrong): L_SM = -sum_i w-_i * log pi(y_i|x) (pure avoidance, weighted by severity)
- If S- = empty (all correct): L_SM = rho * sum_i w+_i * log pi(y_i|x) (pure reinforcement, weighted by quality)

**Step 5 -- rho setting**:
- Default: rho = |S-| / |S+| (balance positive and negative teacher influence)
- Alternative: rho from CSD theory: rho* = Cov(g+,g-) / Var(g+) estimated via EMA

**Relation to standard GRPO**: When rewards are binary {0,1} and c=0.5, SM-CSD reduces to: w+ = 1/|S+|, w- = 1/|S-|, which is equivalent to Dr. GRPO (mean-centered, no-std advantage). SM-CSD is a strict generalization.

### Supporting Component: (alpha,beta) Phase Diagram

Parameterize the SM-CSD loss with explicit positive weight alpha and negative weight beta:
```
L(alpha,beta) = -beta * sum_{i in S-} w-_i * log pi(y_i|x) + alpha * sum_{i in S+} w+_i * log pi(y_i|x)
```

Sweep (alpha,beta) in [0,1] x [0,2] on a grid (e.g., 5x4 = 20 points) and measure:
- Final accuracy, entropy trajectory, KL divergence, reward trajectory

Label each region: Stable, Entropy Collapse, Gradient Starvation, Reward Hacking.

### Partial-Credit Reward for MATH

Implement a 3-level continuous reward:
- r = 1.0: final answer matches gold (extracted via boxed{} or ####)
- r = 0.5: numeric answer is within 10% of gold, or correct intermediate equation detected
- r = 0.0: completely wrong answer or no extractable answer

Correctness threshold c = 0.5.

### Training Plan
1. G=8, batch_size=1, grad_accum=4 -> effective 4 prompts x 8 generations = 32 completions per update
2. lr = 2e-5, warmup 5%, cosine decay
3. KL penalty beta = 0.04, increase to 0.1 if collapse detected
4. 200 training steps
5. LoRA r=64, same target modules

### Trust-Region Rescue
Monitor: KL velocity (d(KL)/d(step)), reward drop (>20% over 10 steps). When collapse detected, pulse lower LR + higher beta for 20 steps.

### Failure Modes and Diagnostics
- All-positive group: degrades to pure reinforcement (acceptable)
- All-negative group: degrades to pure avoidance (better than zero gradient)
- Reward noise: partial-credit reward has extraction noise -> add confidence filter
- rho instability: fall back to rho = |S-|/|S+|

### Novelty and Elegance Argument
- Closest work: CoRPO (65% overlap) -- changes baseline from group-mean to threshold, but still scalar advantages. SM-CSD constructs two normalized distributions.
- Key difference: distribution-level method vs scalar-level method.
- Why focused: Only changes advantage computation. Zero new parameters.

## Claim-Driven Validation Sketch

### Claim 1: SM-CSD outperforms scalar-advantage methods on continuous-reward tasks
- Experiment: SM-CSD vs {Standard GRPO, Dr. GRPO, CoRPO, RLOO} on MATH with partial credit, 4 seeds x 200 steps
- Metric: MATH accuracy (full test set), GSM8K accuracy (transfer)
- Expected: SM-CSD > CoRPO > Dr. GRPO > Standard GRPO on MATH

### Claim 2: Phase diagram predicts training stability
- Experiment: 20-point (alpha,beta) grid on MATH
- Metric: accuracy + entropy + KL at convergence per point
- Expected: Clear phase boundaries; SM-CSD default sits in stable region

### Ablation: SM-CSD reduces to Dr. GRPO on binary rewards
- Run SM-CSD on GSM8K (binary) and show equivalent performance to Dr. GRPO

## Compute and Timeline Estimate
- Phase diagram: 20 points x 1 seed x ~2h = 40 GPU-hours
- SM-CSD main: 4 seeds x ~2h = 8 GPU-hours
- Baselines: 4 methods x 4 seeds x ~2h = 32 GPU-hours
- Ablations: ~16 GPU-hours
- Total: ~96 GPU-hours = ~12 hours on 8x A800
- Timeline: Implementation 3 days, experiments 2 days, analysis 2 days = 1 week
