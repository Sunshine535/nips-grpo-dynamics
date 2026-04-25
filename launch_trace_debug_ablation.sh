#!/bin/bash
# GPT-5.5 review Task 3: clean ablation isolating trust gate from infrastructure.
#
# Variants (3 modes × 2 seeds = 6 runs, fits 6 GPUs):
#   B0_uniform_constant        : TRACE trainer + uniform replay sampler + lambda=lambda_max (NO trust mechanism)
#   B1_trust_sampler_constant  : TRACE trainer + trust replay sampler + lambda=lambda_max (sampler only, no adaptive lambda)
#   C_full                     : TRACE trainer + trust sampler + adaptive lambda_eff (weighted drift budget)
#
# Hypothesis:
# - If B0 ≈ A (legacy), then TRACE trainer/storage is fine; sampler is the discriminator.
# - If B0 < A, then TraceGRPOTrainer/PromptCreditState has bug or behavioural diff.
# - If C > B1, then adaptive lambda mechanism adds value over trust sampling alone.
set -euo pipefail

cd "$(dirname "$0")"

if [ -d /openbayes/input/input0/hub ]; then export HF_HOME=/openbayes/input/input0; fi
export HF_HUB_CACHE=${HF_HOME:-}/hub
export HF_DATASETS_CACHE=${HF_HOME:-}/datasets
export TRANSFORMERS_CACHE=${HF_HOME:-}/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="Qwen/Qwen3.5-9B"
MAX_STEPS=${MAX_STEPS:-200}
SEEDS=${SEEDS:-"42 43"}
OUTDIR=${OUTDIR:-results/trace_debug_ablation}

mkdir -p "$OUTDIR/logs"

GPU=0
for seed in $SEEDS; do
    tag="B0_uniform_constant_seed${seed}"
    CUDA_VISIBLE_DEVICES=$GPU nohup python3 scripts/run_trace_grpo.py \
        --seed $seed --config configs/trace_grpo_minimal.yaml \
        --output-dir "$OUTDIR/B0_uniform" --max-steps $MAX_STEPS --model $MODEL \
        --trace-mode uniform_constant \
        --run-name "$tag" \
        > "$OUTDIR/logs/$tag.log" 2>&1 &
    echo "[B0] $tag GPU $GPU pid=$!"
    GPU=$((GPU+1))
done

for seed in $SEEDS; do
    tag="B1_trust_sampler_constant_seed${seed}"
    CUDA_VISIBLE_DEVICES=$GPU nohup python3 scripts/run_trace_grpo.py \
        --seed $seed --config configs/trace_grpo_minimal.yaml \
        --output-dir "$OUTDIR/B1_trust_sampler" --max-steps $MAX_STEPS --model $MODEL \
        --trace-mode trust_sampler_constant \
        --run-name "$tag" \
        > "$OUTDIR/logs/$tag.log" 2>&1 &
    echo "[B1] $tag GPU $GPU pid=$!"
    GPU=$((GPU+1))
done

for seed in $SEEDS; do
    tag="C_full_seed${seed}"
    CUDA_VISIBLE_DEVICES=$GPU nohup python3 scripts/run_trace_grpo.py \
        --seed $seed --config configs/trace_grpo_minimal.yaml \
        --output-dir "$OUTDIR/C_full" --max-steps $MAX_STEPS --model $MODEL \
        --trace-mode full \
        --run-name "$tag" \
        > "$OUTDIR/logs/$tag.log" 2>&1 &
    echo "[C] $tag GPU $GPU pid=$!"
    GPU=$((GPU+1))
done

echo "[launched] 6 runs (B0/B1/C × seeds: $SEEDS), $MAX_STEPS steps"
wait
echo "[done] Training complete"

# Full-set eval on all 6 runs
N=1319
GPU=0
for variant in B0_uniform B1_trust_sampler C_full; do
    mkdir -p "$OUTDIR/$variant/evals"
    for seed in $SEEDS; do
        case "$variant" in
            B0_uniform) tag="B0_uniform_constant_seed${seed}" ;;
            B1_trust_sampler) tag="B1_trust_sampler_constant_seed${seed}" ;;
            C_full) tag="C_full_seed${seed}" ;;
        esac
        adapter="$OUTDIR/$variant/$tag/checkpoint-final"
        out="$OUTDIR/$variant/evals/eval_${tag}.json"
        [ -f "$out" ] && continue
        [ -d "$adapter" ] || { echo "[skip] $tag no checkpoint"; continue; }
        CUDA_VISIBLE_DEVICES=$GPU nohup python3 scripts/eval_stratified.py \
            --adapter "$adapter" --n $N --selection full --out "$out" \
            > "$OUTDIR/$variant/evals/${tag}.log" 2>&1 &
        echo "[eval] $tag GPU $GPU"
        GPU=$(( (GPU+1) % 8 ))
    done
done
wait
echo "[done] Evals complete"

echo "=== RESULTS ==="
for f in "$OUTDIR"/*/evals/eval_*.json; do
    [ -f "$f" ] && python3 -c "
import json, os
d = json.load(open('$f'))
print(f'{os.path.basename(\"$f\"):60s} acc={d[\"accuracy\"]:.4f}')
"
done
