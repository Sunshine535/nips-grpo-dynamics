#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG="configs/rho_sweep.yaml"
QUICK="${QUICK:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
FROM_PHASE="${1:---from-phase}"
START_PHASE=0

if [[ "$FROM_PHASE" == "--from-phase" ]] && [[ -n "${2:-}" ]]; then
    START_PHASE="$2"
fi

MARKER_DIR="results/.phase_markers"
mkdir -p "$MARKER_DIR" results/logs

phase_done() {
    [[ -f "$MARKER_DIR/phase_${1}_done" ]] && [[ "$FORCE_RERUN" != "1" ]]
}

mark_done() {
    date > "$MARKER_DIR/phase_${1}_done"
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a results/logs/pipeline_v2.log
}

source scripts/gpu_utils.sh 2>/dev/null || true

RHO_VALUES=(0.1 0.3 0.5 0.7 1.0 1.5 2.0 3.0 5.0)
SEEDS=(42 43 44)

if [[ "$QUICK" == "1" ]]; then
    RHO_VALUES=(0.3 1.0 3.0)
    SEEDS=(42)
    log "QUICK mode: reduced grid"
fi

# ── Phase 0: Model download ──
if [[ "$START_PHASE" -le 0 ]] && ! phase_done 0; then
    log "Phase 0: Downloading models"
    python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
for model in ['Qwen/Qwen3.5-9B']:
    AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    print(f'Tokenizer ready: {model}')
" 2>&1 | tee results/logs/phase0_download.log
    mark_done 0
fi

# ── Phase 1: Coarse rho sweep ──
if [[ "$START_PHASE" -le 1 ]] && ! phase_done 1; then
    log "Phase 1: Coarse rho sweep (200 steps)"
    for rho in "${RHO_VALUES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            OUT_DIR="results/sweep_coarse/rho${rho}_seed${seed}"
            if [[ -f "$OUT_DIR/training_metrics.json" ]] && [[ "$FORCE_RERUN" != "1" ]]; then
                log "  Skip: rho=$rho seed=$seed (already done)"
                continue
            fi
            log "  Training: rho=$rho seed=$seed (coarse)"
            python scripts/train_rho_sweep.py \
                --rho "$rho" \
                --seed "$seed" \
                --max_steps 200 \
                --config "$CONFIG" \
                --output_dir "$OUT_DIR" \
                2>&1 | tee "results/logs/coarse_rho${rho}_seed${seed}.log"
        done
    done
    mark_done 1
fi

# ── Phase 2: Stability analysis ──
if [[ "$START_PHASE" -le 2 ]] && ! phase_done 2; then
    log "Phase 2: Computing stability map"
    python scripts/compute_stability_map.py \
        --output_dir results/stability_analysis \
        --group_size 4 \
        --kl_coef 0.05 \
        --clip_range 0.2 \
        2>&1 | tee results/logs/phase2_stability.log
    mark_done 2
fi

# ── Phase 3: Eval coarse sweep + fine sweep ──
if [[ "$START_PHASE" -le 3 ]] && ! phase_done 3; then
    log "Phase 3: Evaluating coarse sweep checkpoints"
    for rho in "${RHO_VALUES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            CKPT_DIR="results/sweep_coarse/rho${rho}_seed${seed}"
            if [[ ! -d "$CKPT_DIR" ]]; then
                continue
            fi
            python scripts/eval_phase_point.py \
                --checkpoint_dir "$CKPT_DIR" \
                --positive_ratio 1.0 \
                --negative_weight 1.0 \
                --seed "$seed" \
                --output_dir results/sweep_coarse \
                2>&1 | tee "results/logs/eval_rho${rho}_seed${seed}.log" || true
        done
    done
    mark_done 3
fi

# ── Phase 4: AdaBalance + baselines ──
if [[ "$START_PHASE" -le 4 ]] && ! phase_done 4; then
    log "Phase 4: AdaBalance and baseline comparison"

    for seed in "${SEEDS[@]}"; do
        log "  Vanilla GRPO (rho=1.0) seed=$seed"
        python scripts/train_rho_sweep.py \
            --rho 1.0 --seed "$seed" \
            --config "$CONFIG" \
            --output_dir "results/adabalance/vanilla_seed${seed}" \
            2>&1 | tee "results/logs/adabalance_vanilla_seed${seed}.log"

        log "  AdaBalance K=50 tau=0.1 seed=$seed"
        python scripts/train_adabalance.py \
            --K 50 --tau 0.1 --seed "$seed" \
            --config "$CONFIG" \
            --output_dir "results/adabalance/adabalance_K50_tau0.1_seed${seed}" \
            2>&1 | tee "results/logs/adabalance_K50_seed${seed}.log"
    done

    for K in 10 50 100; do
        for tau in 0.05 0.1 0.2; do
            log "  AdaBalance ablation K=$K tau=$tau"
            python scripts/train_adabalance.py \
                --K "$K" --tau "$tau" --seed 42 \
                --config "$CONFIG" \
                --output_dir "results/adabalance/ablation_K${K}_tau${tau}" \
                2>&1 | tee "results/logs/adabalance_ablation_K${K}_tau${tau}.log"
        done
    done

    mark_done 4
fi

# ── Phase 5: Robustness test ──
if [[ "$START_PHASE" -le 5 ]] && ! phase_done 5; then
    log "Phase 5: i.i.d. violation robustness test"
    python scripts/run_robustness_test.py \
        --output_dir results/robustness \
        --group_size 4 \
        2>&1 | tee results/logs/phase5_robustness.log
    mark_done 5
fi

# ── Phase 6: 27B transfer (skip in QUICK) ──
if [[ "$START_PHASE" -le 6 ]] && ! phase_done 6 && [[ "$QUICK" != "1" ]]; then
    log "Phase 6: 27B transfer sanity check"
    for rho in 0.3 1.0 3.0; do
        python scripts/train_rho_sweep.py \
            --rho "$rho" --seed 42 \
            --model_name "Qwen/Qwen3.5-27B" \
            --config "$CONFIG" \
            --output_dir "results/validation_27b/rho${rho}_seed42" \
            2>&1 | tee "results/logs/27b_rho${rho}.log"
    done
    mark_done 6
fi

# ── Phase 7: Build all figures ──
if [[ "$START_PHASE" -le 7 ]] && ! phase_done 7; then
    log "Phase 7: Building figures and analysis"
    mkdir -p results/figures

    python scripts/compute_stability_map.py \
        --output_dir results/figures \
        --resolution 300 \
        2>&1 | tee results/logs/phase7_figures.log

    mark_done 7
fi

date > results/.pipeline_v2_done
log "Pipeline complete!"
