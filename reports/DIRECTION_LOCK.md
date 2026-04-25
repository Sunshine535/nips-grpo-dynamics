PROJECT_DIRECTION_LOCK:
Develop a mechanism-level, reproducible, positive-result NeurIPS main-track method for stable and effective LLM post-training under sparse / binary verifiable rewards, especially small-group GRPO/RLVR-style math reasoning.

Allowed changes:
- credit assignment
- replay mechanism
- loss/objective
- prompt-conditioned evidence state
- trainer implementation
- configs
- mechanism logging
- ablations
- baseline coverage

Forbidden pivots:
- negative-result paper
- failure-analysis-only paper
- dataset-specific preprocessing trick
- GSM8K-only benchmark hack
- TASA-only renamed as final method
- current TRACE stabilized-low-accuracy claim
- weakening baselines
- changing metric/split/preprocessing to create gains

Scope Compliance Status:
PASS for SAGE-GRPO successor path, because it preserves the project direction and introduces a mechanism-level successor rather than downgrading the project.
