# Gap Map

## G1: No proof GRPO gradient = contrastive self-distillation
- Status: **UNRESOLVED** (our core contribution)
- Linked: idea:csd-equivalence
- Evidence: 12 searches, 0 prior proofs found

## G2: No theoretical explanation for RLVR capacity bound
- Status: **UNRESOLVED**
- Linked: idea:csd-equivalence (Theorem 2)
- NeurIPS 2025 BPR showed empirically but no theory

## G3: No unified failure taxonomy for GRPO
- Status: **UNRESOLVED**
- LLD, entropy collapse, seed variance treated as separate phenomena
- Linked: idea:csd-equivalence (CSD unifies as distillation failures)

## G4: No principled adaptive ρ with closed-form solution
- Status: **UNRESOLVED**
- AdaBalance (our v1) was variance-based, not CSD-derived
- Linked: idea:csd-equivalence (Theorem 3)

## G5: No collapse predictor from step-0 data
- Status: **PARTIALLY ADDRESSED**
- GAC (2603.01501) does gradient cosine during training (not step-0)
- Our step-0 Q_CSD predictor is novel
- Linked: idea:csd-equivalence (Proposition 1)

## G6: GRPO variant unification is descriptive, not predictive
- Status: **UNRESOLVED**
- SRPO combines GRPO+SDPO heuristically
- No framework predicts which variant works WHERE
- Linked: idea:csd-equivalence (predictive unification table)
