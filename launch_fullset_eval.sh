#!/bin/bash
# CRITICAL: Full-set (n=1319) evaluation of ALL 200-step adapters
# This resolves whether the n=200 positive results hold on the complete test set.
# Must run AFTER Wave 14b/15 finish (or kill them).
set -e
cd "$(dirname "$0")"

if [ -d /openbayes/input/input0/hub ]; then export HF_HOME=/openbayes/input/input0; fi
export HF_HUB_CACHE=$HF_HOME/hub HF_DATASETS_CACHE=$HF_HOME/datasets TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

N=1319  # full GSM8K test set
OUTDIR="results/fullset_eval"
mkdir -p $OUTDIR/logs

# Define all adapters to evaluate (most important first)
declare -a ADAPTERS=(
    # SPO+Replay 200-step (our method — 9 seeds)
    "results/wave10_aser/spo_full_seed42/checkpoint-final spo_full_seed42"
    "results/wave10_aser/spo_full_seed43/checkpoint-final spo_full_seed43"
    "results/wave10_aser/spo_full_seed44/checkpoint-final spo_full_seed44"
    "results/wave10_aser/spo_full_seed46/checkpoint-final spo_full_seed46"
    "results/wave10_aser/spo_full_seed47/checkpoint-final spo_full_seed47"
    "results/wave10_aser/spo_full_seed48/checkpoint-final spo_full_seed48"
    "results/wave10_aser/spo_full_seed49/checkpoint-final spo_full_seed49"
    "results/wave10_aser/spo_full_seed50/checkpoint-final spo_full_seed50"
    "results/wave10_aser/spo_full_seed51/checkpoint-final spo_full_seed51"
    # Fixed-ρ=0.70 baseline (6 seeds from wave11, 4 from gates)
    "results/gates_1_2/rho0.70_seed42/checkpoint-final rho070_seed42"
    "results/gates_1_2/rho0.70_seed43/checkpoint-final rho070_seed43"
    "results/gates_1_2/rho0.70_seed44/checkpoint-final rho070_seed44"
    "results/gates_1_2/rho0.70_seed45/checkpoint-final rho070_seed45"
    "results/wave11_rho070_boost/rho0.70_seed46/checkpoint-final rho070_seed46"
    "results/wave11_rho070_boost/rho0.70_seed47/checkpoint-final rho070_seed47"
    "results/wave11_rho070_boost/rho0.70_seed48/checkpoint-final rho070_seed48"
    "results/wave11_rho070_boost/rho0.70_seed49/checkpoint-final rho070_seed49"
    "results/wave11_rho070_boost/rho0.70_seed50/checkpoint-final rho070_seed50"
    "results/wave11_rho070_boost/rho0.70_seed51/checkpoint-final rho070_seed51"
    # RFT-only control (4 seeds)
    "results/wave11_rft_control/aser_spo_seed42_nopg/checkpoint-final rft_seed42"
    "results/wave11_rft_control/aser_spo_seed43_nopg/checkpoint-final rft_seed43"
    "results/wave11_rft_control/aser_spo_seed44_nopg/checkpoint-final rft_seed44"
    "results/wave11_rft_control/aser_spo_seed45_nopg/checkpoint-final rft_seed45"
    # SFT-gold control (4 seeds)
    "results/wave13_sft_gold_control/sft_gold_seed42/checkpoint-final sft_gold_seed42"
    "results/wave13_sft_gold_control/sft_gold_seed43/checkpoint-final sft_gold_seed43"
    "results/wave13_sft_gold_control/sft_gold_seed44/checkpoint-final sft_gold_seed44"
    "results/wave13_sft_gold_control/sft_gold_seed45/checkpoint-final sft_gold_seed45"
    # Dr. GRPO (3 seeds)
    "results/wave9_dr_grpo/rho1.00_seed42_drgrpo/checkpoint-final drgrpo_seed42"
    "results/wave9_dr_grpo/rho1.00_seed43_drgrpo/checkpoint-final drgrpo_seed43"
    "results/wave9_dr_grpo/rho1.00_seed44_drgrpo/checkpoint-final drgrpo_seed44"
    # SPO-only (3 seeds)
    "results/wave10_aser/spo_only_seed42/checkpoint-final spo_only_seed42"
    "results/wave10_aser/spo_only_seed43/checkpoint-final spo_only_seed43"
    "results/wave10_aser/spo_only_seed44/checkpoint-final spo_only_seed44"
)

# Also evaluate base model (no adapter)
echo "[info] ${#ADAPTERS[@]} adapters + 1 base model to evaluate"

# Process in batches of 8 (one per GPU)
batch=0
total=${#ADAPTERS[@]}
i=0

while [ $i -lt $total ]; do
    batch=$((batch + 1))
    echo "=== Batch $batch (adapters $i to $((i+7 < total ? i+7 : total-1))) ==="
    PIDS=()
    gpu=0
    for j in $(seq $i $((i+7 < total ? i+7 : total-1))); do
        read adapter tag <<< "${ADAPTERS[$j]}"
        out="$OUTDIR/${tag}.json"
        if [ -f "$out" ]; then echo "[skip] $tag exists"; gpu=$((gpu+1)); continue; fi
        if [ ! -d "$adapter" ]; then echo "[WARN] $adapter missing, skip"; gpu=$((gpu+1)); continue; fi
        CUDA_VISIBLE_DEVICES=$gpu nohup python3 scripts/eval_stratified.py \
            --adapter "$adapter" --n $N --out "$out" \
            > $OUTDIR/logs/${tag}.log 2>&1 &
        PIDS+=($!)
        echo "[eval] $tag on GPU $gpu pid=$!"
        gpu=$((gpu + 1))
    done
    # Wait for this batch
    for pid in "${PIDS[@]}"; do wait $pid; done
    echo "[batch $batch] complete"
    i=$((i + 8))
done

# Base model eval
out="$OUTDIR/base_model.json"
if [ ! -f "$out" ]; then
    echo "[eval] base model on GPU 0"
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_stratified.py \
        --n $N --out "$out" \
        > $OUTDIR/logs/base_model.log 2>&1
fi

# Print results
echo ""
echo "=== FULL-SET EVALUATION RESULTS (n=$N) ==="
for f in $OUTDIR/*.json; do
    [ -f "$f" ] && python3 -c "import json,os; d=json.load(open('$f')); print(f'{os.path.basename(\"$f\"):40s} acc={d[\"accuracy\"]:.4f} ({d[\"correct\"]}/{d[\"n\"]})')"
done

echo "[done] Full-set evaluation complete"
