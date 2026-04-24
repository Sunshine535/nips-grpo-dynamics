#!/bin/bash
# Wave 16 — The correct path: G=8, lr=2e-5, n=1319 eval
#
# Root cause fixes:
#   1. G=2→8: dramatically better advantage estimates (8 completions per prompt)
#   2. lr=1e-4→2e-5: prevent overfitting/collapse at longer training
#   3. n=1319 eval: honest full-set evaluation
#
# GPU 0-3: SPO+Replay with G=8, lr=2e-5, 200 steps (4 seeds)
# GPU 4-7: Standard GRPO (fixed rho=1.0) with G=8, lr=2e-5, 200 steps (4 seeds)
set -e
cd "$(dirname "$0")"

if [ -d /openbayes/input/input0/hub ]; then export HF_HOME=/openbayes/input/input0; fi
export HF_HUB_CACHE=$HF_HOME/hub HF_DATASETS_CACHE=$HF_HOME/datasets TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

MODEL="Qwen/Qwen3.5-9B"

# Need a modified config with G=8 and lr=2e-5
cat > configs/aser_g8_lr2e5.yaml << 'YAMLEOF'
dataset:
  name: "openai/gsm8k"
  split: "train"

model:
  name: "Qwen/Qwen3.5-9B"

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-5
  warmup_ratio: 0.05
  weight_decay: 0.01
  max_grad_norm: 1.0
  bf16: true
  logging_steps: 1
  num_generations: 8
  max_completion_length: 256
  gradient_checkpointing: false

lora:
  r: 64
  lora_alpha: 128
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  lora_dropout: 0.05
  task_type: "CAUSAL_LM"

aser:
  backbone: "spo"
  dup_frac: 0.0
  hardness_temp: 2.0
  dup_warmup_steps: 100
  alpha_baseline: 0.1
  alpha_success: 0.1
  lambda_rep: 0.05
  replay_batch_size: 2
  replay_max_per_prompt: 2
  replay_warmup_steps: 50
  success_threshold: 0.5
YAMLEOF

N_EVAL=1319

# === Part A: SPO+Replay G=8 (GPU 0-3) ===
OUTDIR_SPO="results/wave16_spo_g8"
mkdir -p $OUTDIR_SPO/logs
PIDS_A=()
for pair in "0 42" "1 43" "2 44" "3 45"; do
    read gpu seed <<< "$pair"
    tag="spo_replay_g8_seed$seed"
    if [ -d "$OUTDIR_SPO/$tag/checkpoint-final" ]; then echo "[skip] $tag"; continue; fi
    CUDA_VISIBLE_DEVICES=$gpu nohup python3 scripts/run_aser_mvp.py \
        --seed $seed --backbone spo --config configs/aser_g8_lr2e5.yaml \
        --output-dir $OUTDIR_SPO --max-steps 200 --model $MODEL \
        --run-name "$tag" \
        > $OUTDIR_SPO/logs/$tag.log 2>&1 &
    PIDS_A+=($!)
    echo "[train] $tag on GPU $gpu pid=$!"
done

# === Part B: Standard GRPO G=8 baseline (GPU 4-7) ===
OUTDIR_GRPO="results/wave16_grpo_g8"
mkdir -p $OUTDIR_GRPO/logs
PIDS_B=()
for pair in "4 42" "5 43" "6 44" "7 45"; do
    read gpu seed <<< "$pair"
    tag="grpo_g8_seed$seed"
    if [ -d "$OUTDIR_GRPO/$tag/checkpoint-final" ]; then echo "[skip] $tag"; continue; fi
    CUDA_VISIBLE_DEVICES=$gpu nohup python3 scripts/run_aser_mvp.py \
        --seed $seed --backbone dr_grpo --config configs/aser_g8_lr2e5.yaml \
        --output-dir $OUTDIR_GRPO --max-steps 200 --model $MODEL \
        --lambda-rep 0 --no-dup \
        --run-name "$tag" \
        > $OUTDIR_GRPO/logs/$tag.log 2>&1 &
    PIDS_B+=($!)
    echo "[train] $tag on GPU $gpu pid=$!"
done

# Wait for all training
echo "[info] Waiting for 8 training runs..."
for pid in "${PIDS_A[@]}" "${PIDS_B[@]}"; do wait $pid; done
echo "[train] All training complete"

# === Evals (n=1319 full test set) ===
mkdir -p $OUTDIR_SPO/evals $OUTDIR_GRPO/evals
gpu=0
PIDS_E=()

# SPO+Replay evals
for seed in 42 43 44 45; do
    tag="spo_replay_g8_seed$seed"
    adapter="$OUTDIR_SPO/$tag/checkpoint-final"
    out="$OUTDIR_SPO/evals/eval_seed${seed}.json"
    if [ -f "$out" ] || [ ! -d "$adapter" ]; then continue; fi
    CUDA_VISIBLE_DEVICES=$gpu nohup python3 scripts/eval_stratified.py \
        --adapter "$adapter" --n $N_EVAL --out "$out" \
        > $OUTDIR_SPO/evals/eval_seed${seed}.log 2>&1 &
    PIDS_E+=($!)
    echo "[eval] $tag on GPU $gpu"
    gpu=$((gpu + 1))
done

# GRPO evals
for seed in 42 43 44 45; do
    tag="grpo_g8_seed$seed"
    adapter="$OUTDIR_GRPO/$tag/checkpoint-final"
    out="$OUTDIR_GRPO/evals/eval_seed${seed}.json"
    if [ -f "$out" ] || [ ! -d "$adapter" ]; then continue; fi
    CUDA_VISIBLE_DEVICES=$gpu nohup python3 scripts/eval_stratified.py \
        --adapter "$adapter" --n $N_EVAL --out "$out" \
        > $OUTDIR_GRPO/evals/eval_seed${seed}.log 2>&1 &
    PIDS_E+=($!)
    echo "[eval] $tag on GPU $gpu"
    gpu=$((gpu + 1))
done

for pid in "${PIDS_E[@]}"; do wait $pid; done

# Print results
echo ""
echo "=== WAVE 16 RESULTS (G=8, lr=2e-5, n=$N_EVAL) ==="
echo "--- SPO+Replay ---"
for f in $OUTDIR_SPO/evals/eval_seed*.json; do
    [ -f "$f" ] && python3 -c "import json,os; d=json.load(open('$f')); print(f'  {os.path.basename(\"$f\"):30s} acc={d[\"accuracy\"]:.4f} ({d[\"correct\"]}/{d[\"n\"]})')"
done
echo "--- Standard GRPO ---"
for f in $OUTDIR_GRPO/evals/eval_seed*.json; do
    [ -f "$f" ] && python3 -c "import json,os; d=json.load(open('$f')); print(f'  {os.path.basename(\"$f\"):30s} acc={d[\"accuracy\"]:.4f} ({d[\"correct\"]}/{d[\"n\"]})')"
done

echo "[done] Wave 16 complete"
