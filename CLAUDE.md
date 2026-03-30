# Project: nips-grpo-dynamics

## Project goal

GRPO Dynamics: Phase Diagrams and Zero-Score Gradient Reshaping for Stable RL Post-Training — 基于 TRL GRPOTrainer，在 (α,β) 参数网格上构建相图，研究 HalluZero 零分梯度重塑、课程学习与健康区诊断。

## Key models

- `Qwen/Qwen3.5-9B` — 主实验模型
- `Qwen/Qwen3.5-27B` — scaling 验证

## Key datasets

- GSM8K (`openai/gsm8k`) — 训练与评测
- MATH (`lighteval/MATH`) — 评测（部分脚本子采样）

## Repo map

- `scripts/` — 实验脚本
  - `run_all_experiments.sh` — 全阶段编排（Phase 0–8）
  - `train_grpo_sweep.py` — (α,β)×seed 相图扫描
  - `eval_phase_point.py` — 评估单个相点
  - `build_phase_diagram.py` — 构建相图
  - `train_grpo_halluzero.py` — HalluZero 训练
  - `eval_halluzero.py` — HalluZero 评估
  - `analyze_gradients.py` — 梯度分析
  - `run_curriculum_strategies.py` — 课程策略
  - `run_diagnostic_analysis.py` — 诊断分析
  - `plot_phase_diagram.py` — 相图可视化
  - `gpu_utils.sh` — GPU 分配工具
- `src/` — 核心模块
  - `balanced_grpo.py` — 均衡 GRPO
  - `zero_score_handler.py` — 零分处理器
- `configs/`
  - `sweep_grid.yaml` — 参数扫描网格
  - `grpo_9b.yaml` — 9B 训练配置

## Common commands

```bash
bash setup.sh
source .venv/bin/activate

# 一键全流程
bash run.sh

# Quick 模式（缩小网格，快速验证）
QUICK=1 bash run.sh

# 从特定阶段恢复
bash scripts/run_all_experiments.sh --from-phase 3

# 强制重跑
FORCE_RERUN=1 bash run.sh

# 单独评测（含 MATH）
python scripts/eval_phase_point.py --eval_math --benchmarks gsm8k math
```

## Experiment phases

| Phase | 内容 |
|-------|------|
| 0 | HF 模型下载 |
| 1 | Baseline GRPO 训练 |
| 2 | (α,β)×seed 相图扫描 |
| 3 | HalluZero 零分策略扫描 |
| 4 | 构建相图 |
| 5 | 梯度分析 |
| 6 | 课程策略 |
| 7 | 诊断分析 |
| 8 | 27B 验证（非 quick 模式） |

## Data and outputs

- Checkpoints: `checkpoints/`
- 相图: `results/phase_diagram/`
- HalluZero: `results/zero_score_sweep/`
- 梯度分析: `results/gradient_analysis/`
- 课程策略: `results/curriculum/`
- 27B 验证: `results/validation_27b_base/`
- 日志: `results/logs/`

## Environment

- Python 3.10, PyTorch 2.10 (CUDA 12.8)
- 关键依赖: transformers, datasets, accelerate, trl, peft, wandb
- **使用 wandb**（`requirements.txt` 已声明）
- 可选: flash-attn
- 训练使用 `accelerate launch` 和 `torchrun`

## Project-specific rules

- Phase 2/3 使用 `CUDA_VISIBLE_DEVICES=$(get_gpu_id ...)` 做 GPU 并行扫描
- `QUICK=1` 或 `--quick` 可切换 quick 模式（缩小网格）
- Phase 8 (27B) 仅在非 quick 模式下运行

## Remote server

<!-- TODO: 请补充此项目的主服务器信息 -->

- SSH: `ssh YOUR_SERVER`
- GPU: 待确认
- Activate: `source .venv/bin/activate`
- Code dir: 待确认
- Background: `screen -dmS grpo bash -c '...'`
