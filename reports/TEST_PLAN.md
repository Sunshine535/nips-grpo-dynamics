# Test Plan

| Test | Purpose | Command | Expected | Status |
|------|---------|---------|----------|--------|
| PromptCreditState frontier | Verify frontier ordering: uncertain > easy/hopeless | `python3 -c "from src.prompt_credit_state import ..."` | Ordering correct | PASS |
| TrustGatedReplayBank weights | Verify frontier items preferred, age decay works, saturation penalizes | `python3 -c "from src.trust_gated_replay_bank import ..."` | Weights ordered correctly | PASS |
| TASA advantage signs | Verify P1-P4 properties | Unit test in session | All 6 tests pass | PASS |
| Eval --selection flag | Verify new CLI flag works | `python3 scripts/eval_stratified.py --help` | Shows --selection choices | PASS (compile) |
| New modules compile | All .py files valid syntax | `python3 -m py_compile src/*.py` | No errors | PASS |
| TRACE trainer smoke test | 2-step train on remote server | `python3 scripts/run_trace_grpo.py --max-steps 2` | Runs, saves lambda_eff logs | PENDING (needs GPU) |
| A/B/C comparison | 200 steps, 3 seeds, full GSM8K eval | Launch script on server | C > A and B on full-set | PENDING (needs GPU) |
