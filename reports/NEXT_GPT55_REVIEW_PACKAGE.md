# Package for GPT-5.5 Pro Review

## Summary of Changes
1. Implemented TRACE-GRPO trainer (src/trace_grpo_trainer.py) with 3 modes: full/constant_gate/no_replay
2. Implemented PromptCreditState (src/prompt_credit_state.py) — Beta-posterior per prompt with frontier/uncertainty/saturation
3. Implemented TrustGatedReplayBank (src/trust_gated_replay_bank.py) — weighted sampling by trust/age/diversity
4. Fixed eval protocol: added --selection {first_n,full,random}, saves eval_question_ids
5. Created TRACE launcher (scripts/run_trace_grpo.py) and config (configs/trace_grpo_minimal.yaml)
6. Created all required reports per execution protocol

## What Was NOT Changed
- Legacy ASER trainer (src/aser_trainer_v14.py) — kept as variant A baseline
- Legacy replay bank / prompt stats — kept for backward compatibility
- RETRACTIONS.md — frozen
- All raw result JSONs — untouched
- Reward function semantics — unchanged
- attention_mask fix — PENDING (identified but not yet applied to avoid breaking running experiments)

## Tests Run
- PromptCreditState: frontier ordering (unseen=0, uncertain=1.0, easy=0.31, hopeless=0.31) — PASS
- TrustGatedReplayBank: trust weights (frontier > easy, fresh > stale, saturation penalty) — PASS
- TASA advantage: 6 property tests (sign, monotonicity, zero-sum, binary equiv, multi-batch) — PASS
- New modules compile: all 4 new .py files — PASS
- TRACE trainer smoke test: PENDING (requires GPU)

## Currently Running on Server
- TASA vs Dr.GRPO G=4 experiment (8 GPUs, seeds 42-45, 200 steps) — started ~02:04 UTC Apr 24

## What GPT-5.5 Pro Should Review Next
1. Is the TRACE trainer implementation faithful to the diagnosis?
2. Does PromptCreditState correctly implement the Beta-posterior frontier?
3. Does TrustGatedReplayBank correctly gate replay by trust?
4. Is the lambda_eff computation (frontier * trust * drift_budget) correct?
5. Are the A/B/C ablation modes properly isolated?
6. Should the attention_mask fix be applied before or after first TRACE experiments?
7. What is the minimum experiment to validate TRACE before full comparison?

## Unresolved Risks
- attention_mask not passed in logprob computation (both ASER and TRACE)
- EOS-after-decode for reward (both trainers)
- G=4 with batch=2 nearly OOMs (78GB on 80GB cards) — may need batch=1
- TASA experiment running concurrently — results not yet available
- No full-set 200-step evaluation yet (would validate or invalidate existing positive signal)
