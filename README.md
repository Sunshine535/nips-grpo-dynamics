# GRPO Dynamics — 相图与零分梯度重塑

## 项目简介

研究 GRPO (Group Relative Policy Optimization) 训练动力学。在 (α, β) 参数网格上构建训练相图，揭示稳定训练区域；提出 HalluZero 零分梯度重塑策略（clip/temperature/curriculum/relabel），解决零分样本导致的梯度不稳定问题。

**Review 状态**: Round 4, Score 6.0/10, completed

## 环境安装

```bash
cd /workspace/nips-grpo-dynamics
python3 -m venv .venv
source .venv/bin/activate
# 注意：需要匹配 CUDA driver 版本
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers datasets accelerate trl peft wandb numpy scipy matplotlib pandas pyyaml
```

## 快速开始

```bash
source .venv/bin/activate

# 单点训练 (α=0.5, β=1.0)
python3 scripts/train_grpo_sweep.py --positive_ratio 0.5 --negative_weight 1.0 --seed 42

# 评估
python3 scripts/eval_phase_point.py --checkpoint_dir results/phase_diagram/a0.5_b1.0_s42
```

## 完整实验流程（Phase 0-8）

```bash
# 一键全流程
bash run.sh

# Quick 模式（缩小网格）
QUICK=1 bash run.sh

# 从特定阶段恢复
bash scripts/run_all_experiments.sh --from-phase 3

# 多卡训练
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate_2gpu.yaml \
    scripts/train_grpo_sweep.py --positive_ratio 0.5 --negative_weight 1.0
```

| Phase | 内容 | 说明 |
|-------|------|------|
| 0 | 模型下载 | HF 模型缓存 |
| 1 | Baseline GRPO | 基准训练 |
| 2 | (α,β) 相图扫描 | 参数网格 |
| 3 | HalluZero 扫描 | 4 种零分策略 |
| 4 | 构建相图 | 聚合结果 |
| 5 | 梯度分析 | 零分梯度统计 |
| 6 | 课程策略 | 课程学习对比 |
| 7 | 诊断分析 | 健康区诊断 |
| 8 | 27B 验证 | Scaling 验证 |

## 断点续训

- Pipeline 使用 `results/.phase_markers/` 标记已完成阶段
- 重新运行会自动跳过已完成的 phase
- 强制重跑：`FORCE_RERUN=1 bash run.sh`
- 训练 checkpoint: `--resume_from_checkpoint`

## 项目结构

```
src/
  balanced_grpo.py        # BalancedGRPOCallback, BalancedGRPOConfig
  zero_score_handler.py   # ZeroScoreHandler (clip/temperature/curriculum/relabel)
scripts/
  train_grpo_sweep.py     # 单点 GRPO 训练
  eval_phase_point.py     # 评估
  build_phase_diagram.py  # 构建相图
  train_grpo_halluzero.py # HalluZero 训练
  analyze_gradients.py    # 梯度分析
  plot_phase_diagram.py   # 可视化
configs/
  sweep_grid.yaml         # 参数网格
  grpo_9b.yaml            # 9B 训练配置
results/                  # 全部实验结果
```

## 下一步

1. 完善相图可视化，标注稳定/不稳定区域边界
2. 27B scaling 验证（Phase 8）
3. 论文撰写
