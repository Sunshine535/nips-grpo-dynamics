#!/bin/bash
# Wave 14 — Fusion: 500-step SPO+Replay (highest-payoff) + Phase diagram sweep
#
# GPU 0-2: SPO+Replay 500 steps, seeds 42/43/44 (improve 69.4% → target 78%+)
# GPU 3-7: Phase diagram α×β grid on dr_grpo backbone (5 points per wave)
#
# Phase diagram grid: α ∈ {0.1, 0.3, 0.5, 0.7, 0.9} × β ∈ {0.0, 0.5, 1.0, 2.0}
# = 20 points total. This wave does the first 5, next waves do 5 each.
set -e
cd "$(dirname "$0")"

if [ -d /openbayes/input/input0/hub ]; then export HF_HOME=/openbayes/input/input0; fi
export HF_HUB_CACHE=$HF_HOME/hub HF_DATASETS_CACHE=$HF_HOME/datasets TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

MODEL="Qwen/Qwen3.5-9B"

# === Part A: 500-step SPO+Replay (GPU 0-2) ===
mkdir -p results/wave14_500step/logs
for pair in "0 42" "1 43" "2 44"; do
    read gpu seed <<< "$pair"
    tag="spo_replay_500step_seed$seed"
    CUDA_VISIBLE_DEVICES=$gpu nohup python3 scripts/run_aser_mvp.py \
        --seed $seed --backbone spo --config configs/aser_mvp.yaml \
        --output-dir results/wave14_500step --max-steps 500 --model $MODEL \
        --run-name "$tag" \
        > results/wave14_500step/logs/$tag.log 2>&1 &
    echo $! > results/wave14_500step/logs/$tag.pid
    echo "[launch] $tag on GPU $gpu pid=$(cat results/wave14_500step/logs/$tag.pid)"
done

# === Part B: Phase diagram (GPU 3-7, dr_grpo backbone, no replay, no SPO) ===
# First wave: α=0.1,0.3,0.5,0.7,0.9 all at β=1.0 (standard negative weight)
mkdir -p results/wave14_phase_diagram/logs
gpu=3
for alpha in 0.1 0.3 0.5 0.7 0.9; do
    tag="phase_a${alpha}_b1.0_seed42"
    CUDA_VISIBLE_DEVICES=$gpu nohup python3 scripts/run_aser_mvp.py \
        --seed 42 --backbone dr_grpo --config configs/aser_mvp.yaml \
        --output-dir results/wave14_phase_diagram --max-steps 200 --model $MODEL \
        --lambda-rep 0 --no-dup \
        --alpha-pos $alpha --beta-neg 1.0 \
        --run-name "$tag" \
        > results/wave14_phase_diagram/logs/$tag.log 2>&1 &
    echo $! > results/wave14_phase_diagram/logs/$tag.pid
    echo "[launch] $tag on GPU $gpu pid=$(cat results/wave14_phase_diagram/logs/$tag.pid)"
    gpu=$((gpu + 1))
done

echo "[done] Wave 14: 3 × 500-step + 5 × phase-diagram launched on 8 GPUs"
