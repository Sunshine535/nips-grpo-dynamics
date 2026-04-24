# Bug Fix Log

## Bug Fix: eval_stratified.py first-N prefix default

Files changed: scripts/eval_stratified.py
Reason: GPT-5.5 P0 — all positive results based on deterministic first-200 test questions
Evidence: Code uses `.select(range(n))` — always takes first N questions
Change: Added --selection {first_n,full,random}; default kept as first_n for backward compat; added eval_question_ids to output JSON; fixed accuracy denominator to use actual evaluated count
Verification command: grep "selection" scripts/eval_stratified.py
Before: Only first-N possible, no selection tracking, accuracy = correct/args.n
After: Explicit selection mode, eval ids saved, accuracy = correct/n_evaluated
Remaining risk: Existing launch scripts still pass --n 1319 without --selection full; need to update

## Bug Fix: attention_mask in get_per_token_logps (PENDING)

Files changed: NOT YET — identified in GPT-5.5 diagnosis P0
Reason: Model forward calls for logprobs do not pass attention_mask, potentially polluting logprobs with pad token context
Evidence: src/aser_trainer_v14.py line ~155: `mdl(input_ids, num_logits_to_keep=...)` without attention_mask
Change: PENDING — will fix in TRACE trainer (trace_grpo_trainer.py uses same pattern)
Remaining risk: Both ASER and TRACE trainers affected
