#!/bin/bash
# Supplementary experiments for "Don't Normalize" paper reframing
# Priority: Sign+CE > TASA-lr-confound > G-sweep
set -e
cd /root/nips-grpo-dynamics
R=results/supplementary_r1
mkdir -p $R

export HF_HOME=/openbayes/input/input0
export HF_HUB_CACHE=/openbayes/input/input0/hub
export HF_DATASETS_CACHE=/openbayes/input/input0/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_DISABLED=true

# === Wave 1: Sign+CE (7 seeds) + TASA-4xlr (1 seed) ===
echo "[$(date -u)] === Wave 1: Sign+CE x7 + TASA-4xlr seed42 ==="
for i in 0 1 2 3 4 5 6; do
    seed=$((42 + i))
    echo "  GPU $i: Sign+CE seed=$seed"
    CUDA_VISIBLE_DEVICES=$i nohup python3 scripts/run_sage_grpo.py \
        --sage-mode positive_ce_only --advantage-mode sign \
        --lambda-pos 0.05 --replay-warmup-steps 50 \
        --seed $seed --config configs/sage_grpo_minimal.yaml \
        --output-dir $R/SC_sign_ce_seed${seed} \
        --max-steps 200 \
        > $R/log_SC_s${seed}.txt 2>&1 &
done
# GPU 7: TASA with 4x lr (8e-5 vs 2e-5)
echo "  GPU 7: TASA-4xlr seed=42"
CUDA_VISIBLE_DEVICES=7 nohup python3 scripts/run_sage_grpo.py \
    --sage-mode tasa_only --advantage-mode tasa \
    --lr 8e-5 \
    --seed 42 --config configs/sage_grpo_minimal.yaml \
    --output-dir $R/BLR4_tasa_4xlr_seed42 \
    --max-steps 200 \
    > $R/log_BLR4_s42.txt 2>&1 &
echo "[$(date -u)] Wave 1 launched. Waiting..."
wait
echo "[$(date -u)] Wave 1 DONE."

# === Wave 2: TASA-4xlr x6 + TASA-2xlr seed42 + Sign G=2 seed42 ===
echo "[$(date -u)] === Wave 2: TASA-lr confound + G-sweep pilot ==="
for i in 0 1 2 3 4 5; do
    seed=$((43 + i))
    echo "  GPU $i: TASA-4xlr seed=$seed"
    CUDA_VISIBLE_DEVICES=$i nohup python3 scripts/run_sage_grpo.py \
        --sage-mode tasa_only --advantage-mode tasa \
        --lr 8e-5 \
        --seed $seed --config configs/sage_grpo_minimal.yaml \
        --output-dir $R/BLR4_tasa_4xlr_seed${seed} \
        --max-steps 200 \
        > $R/log_BLR4_s${seed}.txt 2>&1 &
done
echo "  GPU 6: TASA-2xlr seed=42"
CUDA_VISIBLE_DEVICES=6 nohup python3 scripts/run_sage_grpo.py \
    --sage-mode tasa_only --advantage-mode tasa \
    --lr 4e-5 \
    --seed 42 --config configs/sage_grpo_minimal.yaml \
    --output-dir $R/BLR2_tasa_2xlr_seed42 \
    --max-steps 200 \
    > $R/log_BLR2_s42.txt 2>&1 &
echo "  GPU 7: Sign G=2 seed=42"
CUDA_VISIBLE_DEVICES=7 nohup python3 scripts/run_sage_grpo.py \
    --sage-mode tasa_only --advantage-mode sign \
    --num-generations 2 \
    --seed 42 --config configs/sage_grpo_minimal.yaml \
    --output-dir $R/SG2_sign_g2_seed42 \
    --max-steps 200 \
    > $R/log_SG2_s42.txt 2>&1 &
echo "[$(date -u)] Wave 2 launched. Waiting..."
wait
echo "[$(date -u)] Wave 2 DONE."

# === Wave 3: G-sweep (Sign G=2 x2, Sign G=8 x3, TASA G=2 x3) ===
echo "[$(date -u)] === Wave 3: G-sweep ==="
for i in 0 1; do
    seed=$((43 + i))
    echo "  GPU $i: Sign G=2 seed=$seed"
    CUDA_VISIBLE_DEVICES=$i nohup python3 scripts/run_sage_grpo.py \
        --sage-mode tasa_only --advantage-mode sign \
        --num-generations 2 \
        --seed $seed --config configs/sage_grpo_minimal.yaml \
        --output-dir $R/SG2_sign_g2_seed${seed} \
        --max-steps 200 \
        > $R/log_SG2_s${seed}.txt 2>&1 &
done
for i in 2 3 4; do
    seed=$((42 + i - 2))
    echo "  GPU $i: Sign G=8 seed=$seed"
    CUDA_VISIBLE_DEVICES=$i nohup python3 scripts/run_sage_grpo.py \
        --sage-mode tasa_only --advantage-mode sign \
        --num-generations 8 \
        --seed $seed --config configs/sage_grpo_minimal.yaml \
        --output-dir $R/SG8_sign_g8_seed${seed} \
        --max-steps 200 \
        > $R/log_SG8_s${seed}.txt 2>&1 &
done
for i in 5 6 7; do
    seed=$((42 + i - 5))
    echo "  GPU $i: TASA G=2 seed=$seed"
    CUDA_VISIBLE_DEVICES=$i nohup python3 scripts/run_sage_grpo.py \
        --sage-mode tasa_only --advantage-mode tasa \
        --num-generations 2 \
        --seed $seed --config configs/sage_grpo_minimal.yaml \
        --output-dir $R/BG2_tasa_g2_seed${seed} \
        --max-steps 200 \
        > $R/log_BG2_s${seed}.txt 2>&1 &
done
echo "[$(date -u)] Wave 3 launched. Waiting..."
wait
echo "[$(date -u)] Wave 3 DONE."

# === Wave 4: Eval all supplementary checkpoints ===
echo "[$(date -u)] === Wave 4: Eval ==="
GPU=0
for d in $R/*/; do
    name=$(basename $d)
    [[ "$name" == log_* ]] && continue
    out="$R/evals/eval_${name}.json"
    [ -f "$out" ] && echo "SKIP $name" && continue
    adapter=$(find "$d" -name adapter_config.json -path '*/checkpoint-final/*' 2>/dev/null | head -1)
    [ -z "$adapter" ] && continue
    adir=$(dirname "$adapter")
    mkdir -p $R/evals
    echo "  GPU $GPU: eval $name"
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval_stratified.py --adapter "$adir" --n 1319 --selection full --out "$out" > $R/log_eval_${name}.txt 2>&1 &
    GPU=$((GPU+1))
    if [ $GPU -ge 4 ]; then
        wait; GPU=0
    fi
done
wait
echo "[$(date -u)] === ALL SUPPLEMENTARY DONE ==="
for f in $R/evals/eval_*.json; do
    [ ! -f "$f" ] && continue
    n=$(basename $f .json | sed 's/eval_//')
    a=$(python3 -c "import json;print(round(json.load(open('$f'))['accuracy']*100,2))" 2>/dev/null)
    echo "$n: ${a}%"
done
