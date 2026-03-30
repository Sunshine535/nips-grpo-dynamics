#!/usr/bin/env bash
# =============================================================================
# Master pipeline: RLBalance phase diagram + HalluZero zero-score reshaping
# =============================================================================
# Usage:
#   bash scripts/run_all_experiments.sh           # full grid (long-running)
#   QUICK=1 bash scripts/run_all_experiments.sh   # (dev only: reduced grids)
#
# Environment:
#   PROJ_DIR_ROOT  — repository root (set automatically if unset)
#   QUICK=1        — same as --quick
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJ_DIR_ROOT="${PROJ_DIR_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJ_DIR_ROOT"

QUICK="${QUICK:-0}"
if [[ "${1:-}" == "--quick" ]]; then
  QUICK=1
  shift
fi

# --- Virtualenv (setup.sh creates .venv under project root) ---
if [[ -f "$PROJ_DIR_ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$PROJ_DIR_ROOT/.venv/bin/activate"
  echo "[env] Activated venv: $PROJ_DIR_ROOT/.venv"
else
  echo "[warn] No .venv found at $PROJ_DIR_ROOT/.venv — using current Python"
fi

# --- GPU utilities (monorepo shared copy, then local fallback) ---
# shellcheck source=/dev/null
source "$SCRIPT_DIR/../../_shared/gpu_utils.sh" 2>/dev/null \
  || source "$SCRIPT_DIR/gpu_utils.sh"

auto_setup

PHASE_MARKER_DIR="$PROJ_DIR_ROOT/results/.phase_markers"
mkdir -p "$PHASE_MARKER_DIR"
FORCE_RERUN="${FORCE_RERUN:-0}"

phase_done() { touch "$PHASE_MARKER_DIR/phase_${1}.done"; echo "[PHASE $1] Completed at $(date)"; }
is_phase_done() {
    [[ "$FORCE_RERUN" == "1" ]] && return 1
    [[ -f "$PHASE_MARKER_DIR/phase_${1}.done" ]] && echo "[PHASE $1] Already completed. Skipping. (FORCE_RERUN=1 to override)" && return 0
    return 1
}

TORCHRUN="$(get_torchrun_cmd "$NUM_GPUS")"
export TORCHRUN

PYTHON="${PYTHON:-python3}"

# Multi-GPU accelerate config for DDP training (Phase 1 baseline)
ACCEL_CONFIG="$PROJ_DIR_ROOT/configs/accelerate_multi_gpu.yaml"
generate_accelerate_config "$ACCEL_CONFIG" "$NUM_GPUS"
ACCEL_CMD="accelerate launch --config_file $ACCEL_CONFIG"
CONFIG_SWEEP="$PROJ_DIR_ROOT/configs/sweep_grid.yaml"
CONFIG_HALLU="$PROJ_DIR_ROOT/configs/grpo_9b.yaml"
CKPT_ROOT="$PROJ_DIR_ROOT/checkpoints"
PHASE_EVAL_DIR="$PROJ_DIR_ROOT/results/phase_diagram"
ZERO_SWEEP_ROOT="$PROJ_DIR_ROOT/results/zero_score_sweep"
ANALYSIS_DIR="$PROJ_DIR_ROOT/results/analysis"
MODEL_9B="${MODEL_9B:-Qwen/Qwen3.5-9B}"
MODEL_27B="${MODEL_27B:-Qwen/Qwen3.5-27B}"

# Training / eval shortcuts
if [[ "$QUICK" == "1" ]]; then
  echo "============================================"
  echo " QUICK MODE: reduced steps, grids, and eval"
  echo "============================================"
  SWEEP_MAX_STEPS=(--max_steps 40)
  BASELINE_MAX_STEPS=(--max_steps 40)
  HALLU_EPOCHS=(--num_epochs 1)
  EVAL_NUM_SAMPLES=(--num_samples 64)
  EVAL_GSM8K=(--gsm8k_samples 64)
  EVAL_MATH=(--math_samples 32)
  PHASE2_ALPHAS=(0.3 0.5 0.7)
  PHASE2_BETAS=(0.0 1.0 2.0)
  PHASE2_SEEDS=(42)
  PHASE3_SEEDS=(42)
  # One hyperparameter value per strategy
  CLIP_FACTORS=(0.1)
  TEMP_BOOSTS=(1.5)
  CURR_WARMS=(500)
  RELABEL_EPS=(0.01)
  PHASE8_EVAL_GSM8K=(--gsm8k_samples 32)
  PHASE8_EVAL_MATH=(--math_samples 16)
  SKIP_27B_VALIDATION=1
else
  SWEEP_MAX_STEPS=()
  BASELINE_MAX_STEPS=()
  HALLU_EPOCHS=()
  EVAL_NUM_SAMPLES=(--num_samples 500)
  EVAL_GSM8K=()
  EVAL_MATH=(--math_samples 500)
  PHASE2_ALPHAS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
  PHASE2_BETAS=(0.0 0.25 0.5 1.0 2.0)
  PHASE2_SEEDS=(42 43 44)
  PHASE3_SEEDS=(42 43)
  CLIP_FACTORS=(0.05 0.1 0.2)
  TEMP_BOOSTS=(1.2 1.5 2.0)
  CURR_WARMS=(200 500 1000)
  RELABEL_EPS=(0.005 0.01 0.02)
  PHASE8_EVAL_GSM8K=()
  PHASE8_EVAL_MATH=(--math_samples 500)
  SKIP_27B_VALIDATION=0
fi

echo "============================================"
echo " PROJ_DIR_ROOT = $PROJ_DIR_ROOT"
echo " QUICK         = $QUICK"
echo " NUM_GPUS      = $NUM_GPUS"
echo " ACCEL_CMD     = $ACCEL_CMD"
echo " TORCHRUN      = $TORCHRUN"
echo "============================================"

# -----------------------------------------------------------------------------
# Phase 0 — Model download (Hugging Face Hub)
# -----------------------------------------------------------------------------
if ! is_phase_done 0; then
echo ""
if [[ "$QUICK" == "1" ]]; then
  echo ">>> Phase 0: Model download ($MODEL_9B only; --quick skips 27B prefetch)"
  MODELS_TO_FETCH=("$MODEL_9B")
else
  echo ">>> Phase 0: Model download ($MODEL_9B + $MODEL_27B)"
  MODELS_TO_FETCH=("$MODEL_9B" "$MODEL_27B")
fi
for mid in "${MODELS_TO_FETCH[@]}"; do
  $PYTHON -c "from huggingface_hub import snapshot_download; print('[snapshot_download]', '${mid}'); snapshot_download(repo_id='${mid}')"
done
phase_done 0; fi

# -----------------------------------------------------------------------------
# Phase 1 — Baseline GRPO (α,β parameterization, reference point α=0.5, β=1.0)
# Standard reference in this codebase: symmetric reward scaling without an
# explicit sweep — same trainer as the diagram study, fixed at the center of
# the (α,β) plane.
# -----------------------------------------------------------------------------
if ! is_phase_done 1; then
echo ""
echo ">>> Phase 1: Baseline GRPO (train_grpo_sweep.py, α=0.5, β=1.0, seed 42)"
BASELINE_DIR="$CKPT_ROOT/baseline_grpo_alpha0.50_beta1.00_seed42"
if [[ ! -f "$BASELINE_DIR/training_metrics.json" ]]; then
  if [[ "$NUM_GPUS" -gt 1 ]]; then
    echo "    Using ${NUM_GPUS}-GPU DDP via accelerate launch"
    $ACCEL_CMD "$SCRIPT_DIR/train_grpo_sweep.py" \
      --positive_ratio 0.5 \
      --negative_weight 1.0 \
      --seed 42 \
      --config "$CONFIG_SWEEP" \
      --output_dir "$BASELINE_DIR" \
      --per_device_train_batch_size 2 \
      --gradient_accumulation_steps 1 \
      "${BASELINE_MAX_STEPS[@]}"
  else
    echo "    Using single-GPU training"
    $PYTHON "$SCRIPT_DIR/train_grpo_sweep.py" \
      --positive_ratio 0.5 \
      --negative_weight 1.0 \
      --seed 42 \
      --config "$CONFIG_SWEEP" \
      --output_dir "$BASELINE_DIR" \
      "${BASELINE_MAX_STEPS[@]}"
  fi
else
  echo "    (skip) $BASELINE_DIR already has training_metrics.json"
fi

$PYTHON "$SCRIPT_DIR/eval_phase_point.py" \
  --checkpoint_dir "$BASELINE_DIR" \
  --positive_ratio 0.5 \
  --negative_weight 1.0 \
  --seed 42 \
  --output_dir "$PHASE_EVAL_DIR" \
  "${EVAL_NUM_SAMPLES[@]}" \
  --eval_math
phase_done 1; fi

# Round-robin GPU pool: wait when the last-launched job completes a full cycle of NUM_GPUS workers.
wait_if_gpu_batch_full() {
  if [ $((GPU_IDX % NUM_GPUS)) -eq 0 ] && [ ${#PIDS[@]} -ge "$NUM_GPUS" ]; then
    for pid in "${PIDS[@]}"; do wait "$pid" || exit 1; done
    PIDS=()
  fi
}

# -----------------------------------------------------------------------------
# Phase 2 — Phase diagram sweep (α × β × seeds)
# -----------------------------------------------------------------------------
if ! is_phase_done 2; then
echo ""
echo ">>> Phase 2: Phase diagram sweep (parallel, ${NUM_GPUS} GPUs)"

run_phase2_combo() {
  local A="$1" B="$2" S="$3"
  local TAG RUN_DIR EVAL_JSON
  TAG="$(printf 'alpha%.2f_beta%.2f_seed%d' "$A" "$B" "$S")"
  RUN_DIR="$CKPT_ROOT/$TAG"
  if [[ ! -f "$RUN_DIR/training_metrics.json" ]]; then
    echo "    train  $TAG (GPU ${CUDA_VISIBLE_DEVICES:-?})"
    $PYTHON "$SCRIPT_DIR/train_grpo_sweep.py" \
      --positive_ratio "$A" \
      --negative_weight "$B" \
      --seed "$S" \
      --config "$CONFIG_SWEEP" \
      --output_dir "$RUN_DIR" \
      "${SWEEP_MAX_STEPS[@]}"
  else
    echo "    (skip train) $TAG"
  fi
  EVAL_JSON="$PHASE_EVAL_DIR/eval_${TAG}.json"
  if [[ ! -f "$EVAL_JSON" ]]; then
    echo "    eval   $TAG (GPU ${CUDA_VISIBLE_DEVICES:-?})"
    $PYTHON "$SCRIPT_DIR/eval_phase_point.py" \
      --checkpoint_dir "$RUN_DIR" \
      --positive_ratio "$A" \
      --negative_weight "$B" \
      --seed "$S" \
      --output_dir "$PHASE_EVAL_DIR" \
      "${EVAL_NUM_SAMPLES[@]}" \
      --eval_math
  else
    echo "    (skip eval) $TAG"
  fi
}

PHASE2_LOG_DIR="$PROJ_DIR_ROOT/results/logs/phase2"
mkdir -p "$PHASE2_LOG_DIR"

GPU_IDX=0
PIDS=()
for A in "${PHASE2_ALPHAS[@]}"; do
  for B in "${PHASE2_BETAS[@]}"; do
    for S in "${PHASE2_SEEDS[@]}"; do
      TAG="$(printf 'alpha%.2f_beta%.2f_seed%d' "$A" "$B" "$S")"
      LOG_FILE="$PHASE2_LOG_DIR/${TAG}.log"
      CUDA_VISIBLE_DEVICES=$(get_gpu_id $GPU_IDX) run_phase2_combo "$A" "$B" "$S" >"$LOG_FILE" 2>&1 &
      PIDS+=($!)
      GPU_IDX=$((GPU_IDX + 1))
      wait_if_gpu_batch_full
    done
  done
done
for pid in "${PIDS[@]}"; do wait "$pid" || exit 1; done
echo ">>> Phase 2: all jobs finished OK"
phase_done 2; fi

# -----------------------------------------------------------------------------
# Phase 3 — Zero-score strategy sweep (4 strategies × hyperparams × seeds)
# Path layout includes the substring 'sweep' for run_diagnostic_analysis.py
# -----------------------------------------------------------------------------
if ! is_phase_done 3; then
echo ""
echo ">>> Phase 3: Zero-score strategy sweep (HalluZero, parallel, ${NUM_GPUS} GPUs)"

# Args: gpu_index strategy output_dir [train_grpo_halluzero.py args...]
run_hallu() {
  local gpu="$1"
  local strategy="$2"
  local out_dir="$3"
  shift 3
  export CUDA_VISIBLE_DEVICES="$gpu"
  if [[ -f "$out_dir/training_metrics.json" ]]; then
    echo "    (skip train) $out_dir (GPU $gpu)"
  else
    echo "    train  $out_dir (GPU $gpu)"
    $PYTHON "$SCRIPT_DIR/train_grpo_halluzero.py" \
      --config_path "$CONFIG_HALLU" \
      --output_dir "$out_dir" \
      --zero_score_strategy "$strategy" \
      "$@" \
      "${HALLU_EPOCHS[@]}"
  fi
  if [[ ! -f "$out_dir/eval/summary.json" ]]; then
    echo "    eval   $out_dir (GPU $gpu)"
    $PYTHON "$SCRIPT_DIR/eval_halluzero.py" \
      --model_path "$out_dir" \
      --output_dir "$out_dir/eval" \
      "${EVAL_GSM8K[@]}" \
      "${EVAL_MATH[@]}"
  else
    echo "    (skip eval) $out_dir (GPU $gpu)"
  fi
}

PHASE3_LOG_DIR="$PROJ_DIR_ROOT/results/logs/phase3"
mkdir -p "$PHASE3_LOG_DIR"

GPU_IDX=0
PIDS=()
for S in "${PHASE3_SEEDS[@]}"; do
  for CF in "${CLIP_FACTORS[@]}"; do
    TAG="cf_$(printf '%.2f' "$CF")"
    LOG_FILE="$PHASE3_LOG_DIR/clip_${TAG}_seed${S}.log"
    run_hallu "$(get_gpu_id $GPU_IDX)" clip "$ZERO_SWEEP_ROOT/sweep/clip/${TAG}/seed_${S}" \
      --seed "$S" \
      --clip_factor "$CF" >"$LOG_FILE" 2>&1 &
    PIDS+=($!)
    GPU_IDX=$((GPU_IDX + 1))
    wait_if_gpu_batch_full
  done
  for TB in "${TEMP_BOOSTS[@]}"; do
    TAG="tb_$(printf '%.2f' "$TB")"
    LOG_FILE="$PHASE3_LOG_DIR/temperature_${TAG}_seed${S}.log"
    run_hallu "$(get_gpu_id $GPU_IDX)" temperature "$ZERO_SWEEP_ROOT/sweep/temperature/${TAG}/seed_${S}" \
      --seed "$S" \
      --temperature_boost "$TB" >"$LOG_FILE" 2>&1 &
    PIDS+=($!)
    GPU_IDX=$((GPU_IDX + 1))
    wait_if_gpu_batch_full
  done
  for CW in "${CURR_WARMS[@]}"; do
    TAG="warm_${CW}"
    LOG_FILE="$PHASE3_LOG_DIR/curriculum_${TAG}_seed${S}.log"
    run_hallu "$(get_gpu_id $GPU_IDX)" curriculum "$ZERO_SWEEP_ROOT/sweep/curriculum/${TAG}/seed_${S}" \
      --seed "$S" \
      --curriculum_warmup_steps "$CW" >"$LOG_FILE" 2>&1 &
    PIDS+=($!)
    GPU_IDX=$((GPU_IDX + 1))
    wait_if_gpu_batch_full
  done
  for RE in "${RELABEL_EPS[@]}"; do
    TAG="eps_$(printf '%.4f' "$RE" | tr '.' 'p')"
    LOG_FILE="$PHASE3_LOG_DIR/relabel_${TAG}_seed${S}.log"
    run_hallu "$(get_gpu_id $GPU_IDX)" relabel "$ZERO_SWEEP_ROOT/sweep/relabel/${TAG}/seed_${S}" \
      --seed "$S" \
      --relabel_epsilon "$RE" >"$LOG_FILE" 2>&1 &
    PIDS+=($!)
    GPU_IDX=$((GPU_IDX + 1))
    wait_if_gpu_batch_full
  done
done
for pid in "${PIDS[@]}"; do wait "$pid" || exit 1; done
echo ">>> Phase 3: all jobs finished OK"
phase_done 3; fi

# -----------------------------------------------------------------------------
# Phase 4 — rho-GRPO sweep (uses RhoGRPOTrainer with actual rho-weighting)
# -----------------------------------------------------------------------------
if ! is_phase_done 4; then
echo ""
echo ">>> Phase 4: rho-GRPO sweep (parallel, ${NUM_GPUS} GPUs)"

CONFIG_RHO="$PROJ_DIR_ROOT/configs/rho_sweep.yaml"
RHO_SWEEP_DIR="$PROJ_DIR_ROOT/results/sweep_coarse"
RHO_EVAL_DIR="$PROJ_DIR_ROOT/results/rho_eval"
mkdir -p "$RHO_EVAL_DIR"

if [[ "$QUICK" == "1" ]]; then
  RHO_VALUES=(0.3 1.0 3.0)
  RHO_SEEDS=(42)
  RHO_MAX_STEPS=(--max_steps 40)
else
  RHO_VALUES=(0.1 0.3 0.5 0.7 1.0 1.5 2.0 3.0 5.0)
  RHO_SEEDS=(42 43 44)
  RHO_MAX_STEPS=()
fi

GPU_IDX=0
PIDS=()
for R in "${RHO_VALUES[@]}"; do
  for S in "${RHO_SEEDS[@]}"; do
    TAG="$(printf 'rho%.2f_seed%d' "$R" "$S")"
    RUN_DIR="$RHO_SWEEP_DIR/$TAG"
    LOG_FILE="$PROJ_DIR_ROOT/results/logs/phase4/${TAG}.log"
    mkdir -p "$(dirname "$LOG_FILE")"
    (
      export CUDA_VISIBLE_DEVICES="$(get_gpu_id $GPU_IDX)"
      if [[ ! -f "$RUN_DIR/training_metrics.json" ]]; then
        $PYTHON "$SCRIPT_DIR/train_rho_sweep.py" \
          --rho "$R" --seed "$S" \
          --config "$CONFIG_RHO" \
          --output_dir "$RUN_DIR" \
          "${RHO_MAX_STEPS[@]}"
      fi
      $PYTHON "$SCRIPT_DIR/eval_phase_point.py" \
        --checkpoint_dir "$RUN_DIR" \
        --rho "$R" --seed "$S" \
        --output_dir "$RHO_EVAL_DIR" \
        "${EVAL_NUM_SAMPLES[@]}" \
        --eval_math
    ) >"$LOG_FILE" 2>&1 &
    PIDS+=($!)
    GPU_IDX=$((GPU_IDX + 1))
    wait_if_gpu_batch_full
  done
done
for pid in "${PIDS[@]}"; do wait "$pid" || exit 1; done
echo ">>> Phase 4: rho sweep finished OK"
phase_done 4; fi

# -----------------------------------------------------------------------------
# Phase 5 — AdaBalance training (uses RhoGRPOTrainer with adaptive rho)
# -----------------------------------------------------------------------------
if ! is_phase_done 5; then
echo ""
echo ">>> Phase 5: AdaBalance training"

if [[ "$QUICK" == "1" ]]; then
  ADA_MAX_STEPS=(--max_steps 40)
  ADA_SEEDS=(42)
else
  ADA_MAX_STEPS=()
  ADA_SEEDS=(42 43)
fi

GPU_IDX=0
PIDS=()
for S in "${ADA_SEEDS[@]}"; do
  TAG="adabalance_K50_tau0.1_seed${S}"
  ADA_DIR="$PROJ_DIR_ROOT/results/adabalance/$TAG"
  LOG_FILE="$PROJ_DIR_ROOT/results/logs/phase5/${TAG}.log"
  mkdir -p "$(dirname "$LOG_FILE")"
  (
    export CUDA_VISIBLE_DEVICES="$(get_gpu_id $GPU_IDX)"
    if [[ ! -f "$ADA_DIR/training_metrics.json" ]]; then
      $PYTHON "$SCRIPT_DIR/train_adabalance.py" \
        --seed "$S" \
        --config "$PROJ_DIR_ROOT/configs/rho_sweep.yaml" \
        --output_dir "$ADA_DIR" \
        "${ADA_MAX_STEPS[@]}"
    fi
    $PYTHON "$SCRIPT_DIR/eval_phase_point.py" \
      --checkpoint_dir "$ADA_DIR" \
      --seed "$S" \
      --output_dir "$PROJ_DIR_ROOT/results/rho_eval" \
      "${EVAL_NUM_SAMPLES[@]}" \
      --eval_math
  ) >"$LOG_FILE" 2>&1 &
  PIDS+=($!)
  GPU_IDX=$((GPU_IDX + 1))
  wait_if_gpu_batch_full
done
for pid in "${PIDS[@]}"; do wait "$pid" || exit 1; done
echo ">>> Phase 5: AdaBalance finished OK"
phase_done 5; fi

# -----------------------------------------------------------------------------
# Phase 6 — Gradient analysis (zero vs nonzero)
# -----------------------------------------------------------------------------
if ! is_phase_done 6; then
echo ""
echo ">>> Phase 6: analyze_gradients.py"
$PYTHON "$SCRIPT_DIR/analyze_gradients.py" \
  --model_path "$MODEL_9B" \
  --output_dir "$PROJ_DIR_ROOT/results/gradient_analysis" \
  --num_samples "$( [[ "$QUICK" == "1" ]] && echo 48 || echo 200 )"
phase_done 6; fi

# -----------------------------------------------------------------------------
# Phase 7 — Curriculum strategies (α/β schedules)
# -----------------------------------------------------------------------------
if ! is_phase_done 7; then
echo ""
echo ">>> Phase 7: run_curriculum_strategies.py"
$PYTHON "$SCRIPT_DIR/run_curriculum_strategies.py" \
  --config "$CONFIG_SWEEP" \
  --output_dir "$PROJ_DIR_ROOT/results/curriculum" \
  --best_alpha 0.5 \
  --best_beta 1.0 \
  --total_steps_estimate "$( [[ "$QUICK" == "1" ]] && echo 120 || echo 500 )"
phase_done 7; fi

# -----------------------------------------------------------------------------
# Phase 8 — Phase diagram + collapse-zone analysis (AFTER curriculum in Phase 7)
# -----------------------------------------------------------------------------
if ! is_phase_done 8; then
echo ""
echo ">>> Phase 8: build_phase_diagram.py"
$PYTHON "$SCRIPT_DIR/build_phase_diagram.py" \
  --results_dir "$PHASE_EVAL_DIR" \
  --checkpoint_dir "$CKPT_ROOT" \
  --curriculum_dir "$PROJ_DIR_ROOT/results/curriculum" \
  --output_dir "$ANALYSIS_DIR"
phase_done 8; fi

# -----------------------------------------------------------------------------
# Phase 9 — Diagnostic figures / tables
# -----------------------------------------------------------------------------
if ! is_phase_done 9; then
echo ""
echo ">>> Phase 9: run_diagnostic_analysis.py"
$PYTHON "$SCRIPT_DIR/run_diagnostic_analysis.py" \
  --results_dir "$ZERO_SWEEP_ROOT" \
  --output_dir "$ANALYSIS_DIR/diagnostics"
phase_done 9; fi

# -----------------------------------------------------------------------------
# Phase 10 — 27B validation (weights prefetched in Phase 0 when not --quick)
# -----------------------------------------------------------------------------
if ! is_phase_done 10; then
echo ""
if [[ "${SKIP_27B_VALIDATION:-0}" == "1" ]]; then
  echo ">>> Phase 10: skipped (--quick; avoids large 27B download / eval)"
else
  echo ">>> Phase 10: 27B validation (eval_halluzero on base $MODEL_27B)"
  $PYTHON "$SCRIPT_DIR/eval_halluzero.py" \
    --model_path "$MODEL_27B" \
    --output_dir "$PROJ_DIR_ROOT/results/validation_27b_base" \
    "${PHASE8_EVAL_GSM8K[@]}" \
    "${PHASE8_EVAL_MATH[@]}"
fi
phase_done 10; fi

echo ""
echo "============================================"
echo " All phases finished."
echo "  Phase diagram eval : $PHASE_EVAL_DIR"
echo "  Rho sweep eval     : ${RHO_EVAL_DIR:-results/rho_eval}"
echo "  Aggregated analysis: $ANALYSIS_DIR"
echo "  Zero-score sweep   : $ZERO_SWEEP_ROOT"
echo "============================================"

# --- Pipeline completion marker ---
DONE_FILE="$(dirname "$(dirname "${BASH_SOURCE[0]}")")/results/.pipeline_done"
mkdir -p "$(dirname "$DONE_FILE")"
cat > "$DONE_FILE" << DONEEOF
{
  "project": "$(basename "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")",
  "completed_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "hostname": "$(hostname)",
  "gpus": "${NUM_GPUS:-unknown}",
  "status": "PIPELINE_COMPLETE"
}
DONEEOF
echo ""
echo "[PIPELINE_COMPLETE] All experiments finished successfully."
echo "  Marker: $DONE_FILE"
echo "  Run 'bash collect_results.sh' to package results."
