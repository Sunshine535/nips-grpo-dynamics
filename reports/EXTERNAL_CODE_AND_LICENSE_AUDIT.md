# External Code and License Audit

## Main method files
- src/sage_grpo_trainer.py: written from scratch, no external code
- src/contrastive_evidence_bank.py: written from scratch
- src/prompt_credit_state.py: written from scratch
- scripts/run_sage_grpo.py: written from scratch

## Dependencies used (not copied)
- TRL GRPOTrainer: MIT license, used as base class (not modified)
- PyTorch: BSD license, standard usage
- Transformers: Apache 2.0, standard usage

## No external method code copied
DPO, RePO, RLEP, ExGRPO code was NOT copied. Pairwise loss is a standard
log-sigmoid formulation independently implemented.

AUDIT_STATUS: CLEAN
