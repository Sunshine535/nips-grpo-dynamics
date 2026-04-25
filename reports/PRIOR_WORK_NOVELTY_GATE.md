# Prior Work Novelty Gate

Novelty Gate Status: NOVELTY_RISK

Paper-facing method name: COVER-GRPO (Contrastive Online Verifier Evidence Replay)
Source filenames remain sage_* temporarily for continuity.

## Required citations/baselines
- DPO: pairwise preferred/rejected objective (Rafailov et al. 2023)
- RePO: replay-enhanced policy optimization for GRPO (Li et al. 2025)
- RLEP: verified success replay for LLM reasoning (Kwai 2025)
- ExGRPO: replay/experience management by correctness/value (2025)
- RE-GRPO: hard negative cases in GRPO (Neurocomputing 2026)
- GRPO effective-loss analysis: binary-reward GRPO as weighted contrastive loss
- Self-Hinting SAGE: SAGE name collision in sparse-reward GRPO (Liao 2026)
- Tencent-Hunyuan SAGE-GRPO: SAGE-GRPO name collision in video GRPO

## Mechanism differentiation
COVER-GRPO differs from DPO/RePO/RLEP/ExGRPO via:
1. Online verifier-generated (not offline/human) pos/neg pairs
2. Prompt-local evidence memory (not global replay buffer)
3. Coupled with on-policy signed threshold PG (TASA), not standalone
4. Binary sparse reward setting with small group size (G=4)

## What cannot be claimed as novel
- Pairwise log-sigmoid loss form (= DPO family)
- Replay in GRPO/RLVR (= RePO/RLEP/ExGRPO)
- Using failures in RL (= RE-GRPO, negative reinforcement work)

## What CAN be claimed if experiments pass
- Combining online verifier contrast with signed threshold PG
- Prompt-local evidence memory for sparse binary GRPO
- Showing contrastive evidence beats positive-only replay in this setting
