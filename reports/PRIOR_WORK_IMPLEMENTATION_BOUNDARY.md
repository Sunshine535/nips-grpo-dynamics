# Prior Work Implementation Boundary

## Our code (original implementation)
- SageGRPOTrainer: TASA signed PG + contrastive pair loss + evidence bank
- ContrastiveEvidenceBank: prompt-local pos/neg storage + pair sampling
- PromptCreditState: Beta-posterior per prompt (auxiliary)

## External baselines (to be run as separate comparison, NOT our code)
- DPO: offline pairwise preference optimization (cite Rafailov et al. 2023)
- RePO: replay-enhanced policy optimization (cite Li et al. 2025)
- RLEP: success trajectory replay (cite Kwai 2025)
- ExGRPO: experience replay by correctness (cite 2025)
- RE-GRPO: hard negative pool (cite Neurocomputing 2026)

## Boundary rule
Official baseline code must be isolated in separate scripts/dirs with
clear attribution. Never merge baseline code into our main method.
