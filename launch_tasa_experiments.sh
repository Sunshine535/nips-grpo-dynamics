#!/bin/bash
# TASA-GRPO experiments: main comparison on GSM8K (binary) + MATH (partial credit)
#
# GPU 0-1: TASA on GSM8K (binary, c=0.5) — sanity: should match Dr. GRPO
# GPU 2-3: Dr. GRPO on GSM8K (binary) — baseline comparison
# GPU 4-5: TASA on MATH (partial credit, c=0.5) — main claim
# GPU 6-7: Dr. GRPO on MATH (partial credit) — baseline for MATH
set -e
cd "$(dirname "$0")"

if [ -d /openbayes/input/input0/hub ]; then export HF_HOME=/openbayes/input/input0; fi
export HF_HUB_CACHE=$HF_HOME/hub HF_DATASETS_CACHE=$HF_HOME/datasets TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

MODEL="Qwen/Qwen3.5-9B"
CONFIG="configs/aser_g8_lr2e5.yaml"

# === Phase 1: GSM8K sanity check (TASA vs Dr. GRPO, binary reward) ===
echo "=== Phase 1: GSM8K binary reward ==="

OUTDIR_TASA_GSM="results/tasa_gsm8k"
OUTDIR_DRGRPO_GSM="results/tasa_drgrpo_gsm8k"
mkdir -p $OUTDIR_TASA_GSM/logs $OUTDIR_DRGRPO_GSM/logs

PIDS=()

# TASA on GSM8K (GPU 0-1)
for pair in "0 42" "1 43"; do
    read gpu seed <<< "$pair"
    tag="tasa_gsm8k_seed$seed"
    CUDA_VISIBLE_DEVICES=$gpu nohup python3 scripts/run_aser_mvp.py \
        --seed $seed --backbone tasa --config $CONFIG \
        --output-dir $OUTDIR_TASA_GSM --max-steps 200 --model $MODEL \
        --lambda-rep 0 --no-dup --tasa-threshold 0.5 \
        --run-name "$tag" \
        > $OUTDIR_TASA_GSM/logs/$tag.log 2>&1 &
    PIDS+=($!)
    echo "[train] $tag on GPU $gpu pid=$!"
done

# Dr. GRPO on GSM8K (GPU 2-3)
for pair in "2 42" "3 43"; do
    read gpu seed <<< "$pair"
    tag="drgrpo_gsm8k_seed$seed"
    CUDA_VISIBLE_DEVICES=$gpu nohup python3 scripts/run_aser_mvp.py \
        --seed $seed --backbone dr_grpo --config $CONFIG \
        --output-dir $OUTDIR_DRGRPO_GSM --max-steps 200 --model $MODEL \
        --lambda-rep 0 --no-dup \
        --run-name "$tag" \
        > $OUTDIR_DRGRPO_GSM/logs/$tag.log 2>&1 &
    PIDS+=($!)
    echo "[train] $tag on GPU $gpu pid=$!"
done

# TASA on GSM8K with additional seeds (GPU 4-5) for more data points
for pair in "4 44" "5 45"; do
    read gpu seed <<< "$pair"
    tag="tasa_gsm8k_seed$seed"
    CUDA_VISIBLE_DEVICES=$gpu nohup python3 scripts/run_aser_mvp.py \
        --seed $seed --backbone tasa --config $CONFIG \
        --output-dir $OUTDIR_TASA_GSM --max-steps 200 --model $MODEL \
        --lambda-rep 0 --no-dup --tasa-threshold 0.5 \
        --run-name "$tag" \
        > $OUTDIR_TASA_GSM/logs/$tag.log 2>&1 &
    PIDS+=($!)
    echo "[train] $tag on GPU $gpu pid=$!"
done

# Dr. GRPO additional seeds (GPU 6-7)
for pair in "6 44" "7 45"; do
    read gpu seed <<< "$pair"
    tag="drgrpo_gsm8k_seed$seed"
    CUDA_VISIBLE_DEVICES=$gpu nohup python3 scripts/run_aser_mvp.py \
        --seed $seed --backbone dr_grpo --config $CONFIG \
        --output-dir $OUTDIR_DRGRPO_GSM --max-steps 200 --model $MODEL \
        --lambda-rep 0 --no-dup \
        --run-name "$tag" \
        > $OUTDIR_DRGRPO_GSM/logs/$tag.log 2>&1 &
    PIDS+=($!)
    echo "[train] $tag on GPU $gpu pid=$!"
done

echo "[info] Waiting for Phase 1 (8 training runs)..."
for pid in "${PIDS[@]}"; do wait $pid; done
echo "[Phase 1] Training complete"

# === Phase 1 eval (n=1319 full GSM8K test) ===
N=1319
mkdir -p $OUTDIR_TASA_GSM/evals $OUTDIR_DRGRPO_GSM/evals
gpu=0
PIDS=()

for seed in 42 43 44 45; do
    for dir_tag in "$OUTDIR_TASA_GSM tasa_gsm8k_seed${seed}" "$OUTDIR_DRGRPO_GSM drgrpo_gsm8k_seed${seed}"; do
        read outdir tag <<< "$dir_tag"
        adapter="$outdir/$tag/checkpoint-final"
        out="$outdir/evals/eval_seed${seed}.json"
        if [ -f "$out" ] || [ ! -d "$adapter" ]; then continue; fi
        CUDA_VISIBLE_DEVICES=$gpu nohup python3 scripts/eval_stratified.py \
            --adapter "$adapter" --n $N --out "$out" \
            > $outdir/evals/eval_seed${seed}.log 2>&1 &
        PIDS+=($!)
        echo "[eval] $tag on GPU $gpu"
        gpu=$(( (gpu + 1) % 8 ))
    done
done

for pid in "${PIDS[@]}"; do wait $pid; done
echo "[Phase 1] Eval complete"

# === Print Phase 1 results ===
echo ""
echo "=== PHASE 1 RESULTS: GSM8K binary (n=$N) ==="
echo "--- TASA ---"
for f in $OUTDIR_TASA_GSM/evals/eval_seed*.json; do
    [ -f "$f" ] && python3 -c "import json,os; d=json.load(open('$f')); print(f'  {os.path.basename(\"$f\"):30s} acc={d[\"accuracy\"]:.4f}')"
done
echo "--- Dr. GRPO ---"
for f in $OUTDIR_DRGRPO_GSM/evals/eval_seed*.json; do
    [ -f "$f" ] && python3 -c "import json,os; d=json.load(open('$f')); print(f'  {os.path.basename(\"$f\"):30s} acc={d[\"accuracy\"]:.4f}')"
done

echo ""
echo "[done] TASA experiments Phase 1 complete"
