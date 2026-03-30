# Refinement Report

**Problem**: GRPO signal balance stability under binary verifiable rewards
**Initial Approach**: Phase diagrams of (α,β) space + zero-score gradient reshaping + curriculum
**Date**: 2026-03-29
**Rounds**: 4 / 5
**Final Score**: 9.1 / 10
**Final Verdict**: READY

## Problem Anchor
GRPO is the dominant RL post-training algorithm for LLM reasoning, yet practitioners lack principled guidance for the positive/negative signal balance. Training has hidden stability regimes. No theory maps or navigates these.

## Output Files
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Final proposal: `refine-logs/FINAL_PROPOSAL.md`
- Score evolution: `refine-logs/score-history.md`
- Per-round files: `refine-logs/round-{1,2,3}-review.md`, `round-{1,2,3}-refinement.md`, `round-4-review.md`

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|-------|------------------|--------------------|----------------------|-------------------|-------------|------------------|-----------------|---------|---------|
| 1     | 8                | 5                  | 6                    | 8                 | 6           | 6                | 6               | 6.4     | REVISE  |
| 2     | 9.1              | 8.0                | 8.8                  | 7.2               | 7.8         | 8.7              | 7.9             | 8.2     | REVISE  |
| 3     | 9.1              | 8.5                | 7.7                  | 8.3               | 9.0         | 9.3              | 8.2             | 8.7     | REVISE  |
| 4     | 9.3              | 8.9                | 9.1                  | 9.3               | 8.9         | 9.0              | 9.5             | 9.1     | READY   |

## Round-by-Round Review Record

| Round | Main Reviewer Concerns | What Was Changed | Result |
|-------|-------------------------|------------------|--------|
| 1 | (α,β) collapse; theory needs m/G; "phase transition" too strong; V_pos/V_zero not operational; AdaBalance second headline; budget | Reparameterized to ρ; m/G theory; "stability map"; telemetry; demoted AdaBalance; two-stage sweep | resolved |
| 2 | P(m) insufficient; upper bound needs λ_KL/ε; p_0 alone weak; E[∇L] not proved | Assumptions A1-A3; Theorem/Proposition split; joint collapse conditions; binary scope | resolved |
| 3 | ρ* wrong for original GRPO; overclaimed; no robustness test | L_ρ family interpretation; "stability analysis"; Exp 3 added; K/τ as hyperparams | resolved |
| 4 | None blocking | — | READY |

## Final Proposal Snapshot
- Title: "Stability Analysis of GRPO Signal Balance Under Binary Verifiable Rewards"
- One scalar control variable ρ (effective balance ratio)
- Stability analysis: Theorem 1-3 (sharp lower bound), Proposition 1 (approximate upper bound)
- AdaBalance: minimum-variance controller derived as corollary, 2 hyperparameters
- Under explicit assumptions A1-A3 (binary rewards, group i.i.d., Bernoulli sufficient statistic)
- 4 experiments: stability prediction, AdaBalance comparison, i.i.d. robustness, 27B transfer

## Method Evolution Highlights
1. **Most important simplification**: (α,β) 2D → ρ 1D parameterization
2. **Most important mechanism upgrade**: Theory conditioned on group outcome m/G with explicit assumptions
3. **Most important honesty move**: Corrected ρ* claim to L_ρ family (not original GRPO); Proposition 1 labeled approximate; "stability analysis" not "stability law"

## Pushback / Drift Log
| Round | Reviewer Said | Author Response | Outcome |
|-------|---------------|-----------------|---------|
| 1 | Consider collapsing (α,β) to 1D | Accepted — derived ρ as effective ratio | accepted |
| 2 | Upper bound may need extra variables | Accepted — split into Theorem + Proposition | accepted |
| 3 | ρ* claim wrong for original GRPO | Accepted — corrected to L_ρ family | accepted |
| All | No drift suggestions | — | clean |

## Remaining Weaknesses
1. Proposition 1 (upper bound) is approximate and empirically calibrated — not as clean as Theorem 3
2. Theory limited to binary rewards — continuous reward generalization is future work
3. Assumptions A2 (group i.i.d.) may not hold perfectly in practice — Exp 3 addresses this
4. 27B check is a sanity test, not full scaling evidence

## Next Steps
- **Proceed to `/experiment-plan`** for a full experiment roadmap
- Then implement the refined method in code (fix existing codebase)
- Then `/run-experiment` to execute
- Then `/auto-review-loop` for paper polish
