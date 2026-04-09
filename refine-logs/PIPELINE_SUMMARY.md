# Pipeline Summary

**Problem**: GRPO training collapse + no principled hyperparameter guidance
**Final Method Thesis**: Under binary rewards, the GRPO gradient IS a contrastive self-distillation objective. This theorem yields closed-form optimal ρ, a collapse predictor, and CSDPO — a principled algorithm that eliminates collapse.
**Final Verdict**: READY (theory proven, nightmare weaknesses addressed, experiments planned)
**Date**: 2026-04-09

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Refinement report: `refine-logs/REFINEMENT_REPORT.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Contribution Snapshot
- **Dominant**: CSD Equivalence Theorem + 3 quantitative predictions
- **Supporting**: CSDPO algorithm (theory-derived, zero overhead)
- **Rejected complexity**: Process rewards, external teachers, architecture changes

## Must-Prove Claims
1. GRPO gradient = CSD gradient (R² > 0.95)
2. CSD capacity bound holds (accuracy ≤ pass@G·T_eff)
3. Closed-form ρ* eliminates collapse (0% vs 50%)
4. Q_CSD predicts collapse (AUROC > 0.85)
5. CSDPO beats SRPO on 3 models

## First Runs to Launch
1. `train_csd_verification.py --model Qwen2.5-7B --rho 1.0 --seed 42`
2. `train_grpo_sweep.py --model Qwen2.5-7B --rho 1.0 --seeds 10`
3. `eval_passk.py --model Qwen2.5-7B --k 1,4,8,16,32,64,128,256`

## Main Risks
- **Risk**: CSD equivalence too obvious → **Mitigation**: 3 surprising quantitative predictions
- **Risk**: CSDPO ≤ SRPO → **Mitigation**: Theory contribution stands independently
- **Risk**: Q_CSD AUROC low → **Mitigation**: Combined feature predictor

## Next Action
- Implement CSD logging in GRPOTrainer
- Implement CSDPO components (EA, QW, ADQ, GCR)
- Proceed to `/run-experiment`
