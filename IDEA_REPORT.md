# Idea Discovery Report — GRPO Dynamics

**Direction**: nips-grpo-dynamics
**Date**: 2026-04-07
**Pipeline**: research-lit → novelty-check → nightmare-review × 多轮 pivot
**Settings**: AUTO_PROCEED=false, HUMAN_CHECKPOINT=true, DIFFICULTY=nightmare

---

## Executive Summary

经过完整 pipeline（文献调研 34+ 篇、3 轮方向探索、2 次 nightmare 审稿），最终选定方向：

**"GRPO Training as a Metastable Dynamical System"**
- Basin geometry（统计力学诊断）
- Step-0 trainability prediction（训练前预测 seed 命运）
- Kramers rescue（最小干预恢复坏 seed）
- 跨模型验证：Qwen2.5-7B, Qwen3-8B, Qwen3.5-9B

---

## Pipeline 历程

### Round 1: 原方向 — Unified Stability Map
- **文献调研**: 34+ 篇 GRPO 变体论文 (2025.03–2026.04)
- **新颖性**: 4/10 — "variant unification" 是最拥挤子方向
- **Nightmare 审稿**: 2/10 — 5 FATAL 缺陷（定理证明错误、理论-实现不匹配、ρ=1.0 矛盾）
- **结论**: ❌ 方向不可行

### Round 2: Pivot — Predict + Rescue + Control
- **方向**: Early-warning signals + Checkpoint rescue + Closed-loop ρ/λ_KL 控制
- **新颖性查新**: LOW
  - GAC (2603.01501) 已做 gradient cosine precursor
  - SAFE (2602.04651) 已实现 PID-controlled adaptive KL
  - Rollback-Augmented RL (2510.14503) 已形式化 checkpoint rollback
- **结论**: ❌ 全部被 scooped

### Round 3: 跨学科方向探索
- **GPT-5.4 brainstorm**: 10 个跨学科 idea（统计力学、动力系统、最优传输）
- **最优合并**: Basin Geometry + Step-0 Prediction + Kramers Rescue
- **新颖性查新**: MODERATE-HIGH
  - Claim 1 (Basin): MODERATE — Grokking FSS paper (2603.24746) 用同工具但不同现象
  - Claim 2 (Step-0): HIGH — 无直接先例
  - Claim 3 (Kramers): MODERATE — CL paper (2604.04154) 用同理论但不同领域
- **用户确认**: ✅ 选定此方向 + 加入 Qwen3

### Round 4: Nightmare 审稿 — MetaGRPO
- **Score**: 3/10
- **关键批评**:
  1. 三 claim 强行缝合，缺乏定量串联
  2. n=3-6 seeds 做 Binder cumulant 不 rigorous
  3. Step-0 中 g⁺ 在 sparse reward 下可能 undefined
  4. Kramers 形式主义是比喻（Adam ≠ Langevin）
  5. "三模型族" 过度宣传（都是 Qwen）
  6. **FATAL**: 从未真正证明 metastability（bimodal ≠ multi-attractor）
- **审稿人指出的 killer experiment**:
  > Transient-rescue: 对 collapsing seeds 施加短暂 ρ/KL 脉冲后恢复原参数，证明永久逃逸。rescue 概率在 t_irrev 后急剧下降。如果与 holdout step-0 predictor 吻合，论文难以被 dismiss。

---

## 最终方向（修正后）

### Title
"Metastable Training Dynamics in GRPO: Seed-Resolved Basin Analysis and Transient Rescue"

### 修正要点（基于 nightmare review）
1. **放弃 "phase transition" / "universality" claim** — 改为 "structured seed variance"
2. **增加 seeds**: ≥20 at critical ρ, ≥10 at other ρ values
3. **核心实验: Transient rescue** — 脉冲干预 → 恢复原参数 → 证明永久逃逸
4. **Step-0 predictor 必须 beat trivial baselines**（不能只是 ρ 或初始 acc 的 proxy）
5. **Kramers 语言降级** — 用 "barrier-crossing analogy" 而非声称是 Kramers 理论
6. **加入真正不同的模型族**：Qwen3-8B + 如果计算允许加 Llama

### 实验计划

| 实验 | 模型 | Runs | Seeds | GPU-hours |
|------|------|------|-------|-----------|
| Dense ρ-sweep | Qwen2.5-7B | 180 | 20 × 9ρ | ~360 |
| Dense ρ-sweep | Qwen3-8B | 180 | 20 × 9ρ | ~360 |
| Dense ρ-sweep | Qwen3.5-9B | 90 | 10 × 9ρ | ~180 |
| Step-0 probes | All models | 450 | from sweeps | ~20 |
| **Transient rescue** | Qwen2.5-7B | 120 | 3 seeds × 8t × 5Δ | ~60 |
| **Transient rescue** | Qwen3-8B | 120 | 3 seeds × 8t × 5Δ | ~60 |
| **Total** | | | | **~1040** |

### 预计 NeurIPS 分数
- 当前: 3/10
- 修正后 + transient rescue 成功: 6-7/10
- 修正后 + rescue + step-0 predictor works: 7-8/10

---

## 竞争者速查表

| 论文 | ArXiv | 威胁 | 我们的差异 |
|------|-------|------|-----------|
| Mroueh dynamics | 2503.06639 | HIGH | 我们做 seed-level basin analysis，不是 aggregate dynamics |
| ICLR 2026 off-policy | 2509.24203 | MED | 不同问题（what GRPO is vs when it fails） |
| lambda-GRPO | 2510.06870 | MED | 我们不做 variant unification |
| Grokking FSS | 2603.24746 | MED | 同工具，不同现象（grokking vs GRPO collapse） |
| Kramers CL | 2604.04154 | MED | 同理论，不同 landscape（CL vs GRPO） |
| GAC gradient cosine | 2603.01501 | LOW | 我们做 step-0 预测，不是 during-training 检测 |
| SAFE PID control | 2602.04651 | LOW | 我们不做 closed-loop control |

---

## Pipeline 状态

| Stage | Status |
|-------|--------|
| 1. Idea Discovery | ✅ 完成（4 轮迭代） |
| 1.5 Gate 1 | ✅ 用户确认 MetaGRPO + Qwen3 |
| **2. Implementation** | **⏳ 就绪** |
| 3. Deploy Experiments | ⏸ blocked on implementation |
| 4. Auto Review Loop | ⏸ blocked on experiments |
| 5. Final Summary | ⏸ blocked |
