# Query Pack — nips-grpo-dynamics
Generated: 2026-04-09 | Papers: 12 | Ideas: 3 | Claims: 5

## Direction
GRPO training dynamics for LLM reasoning. Proving GRPO = contrastive self-distillation → CSDPO method.

## Top Gaps (unresolved)
- G1: No proof GRPO gradient = CSD (OUR CORE)
- G2: No theory for RLVR capacity bound
- G3: No unified GRPO failure taxonomy
- G4: No closed-form adaptive ρ from first principles
- G5: No step-0 collapse predictor
- G6: No predictive variant unification

## Paper Clusters
**Cluster A — GRPO+Distillation fusion**: SRPO(2604.02288), SDPO(2601.20802), RLSD(2604.03128), HDPO(2603.23871). Heuristically combine RL+distillation. CSD proves they're the same.
**Cluster B — GRPO stability fixes**: DAPO(2503.14476), LLD(2512.04220), DaGRPO(2512.06337), GTPO(2508.03772), TR-GRPO. Each patches one failure mode. CSD unifies.
**Cluster C — GRPO theory**: Vojnovic(2502.18548), U-Stat(2603.01162), CoPG(2406.19185). Analyze different aspects. None proves CSD.
**Cluster D — Contrastive GRPO**: CLIPO(2603.10101). Adds contrastive head. CSD proves GRPO already IS contrastive.

## Failed Ideas (BANLIST)
- idea:stability-v1: ρ-variance decomposition → theory too shallow, AdaBalance broke on TRL
- idea:metagrpo-v1: Basin analysis + transient rescue → 3/10 nightmare, analysis-only, 1040 GPU-hrs

## Active Chains
G1(no CSD proof) → idea:csd-equivalence → claim:C1(verify R²>0.95) → if confirmed → G2-G6 follow
paper:rlvr_limit2025(capacity=empirical) → G2(no theory) → claim:C2(tight bound via CSD)
paper:lld2025(collapse mechanism) → G3(no unified taxonomy) → CSD explains as teacher degradation

## Open Unknowns
- Is CSD equivalence too obvious for best paper? (need surprising quantitative prediction)
- Does CSDPO actually beat SRPO? (need experiments)
- Does Q_CSD predictor generalize across models? (need cross-model validation)
