#!/bin/bash
# CSD Pilot — 多卡自适应一键启动脚本
# 用法:
#   bash run_csd_pilot.sh           # 运行全部 3 个 pilot (自动检测 GPU)
#   bash run_csd_pilot.sh 2         # 只运行 Pilot 2 (ADQ collapse 消除)
#   QUICK=1 bash run_csd_pilot.sh   # 快速模式 (50 steps)
#
# 多卡模式:
#   自动检测 GPU 数量，Pilot 2/3 的多 seed 实验会并行分配到不同 GPU
#   例：8 GPU → Pilot 2 的 10 个 run 同时跑 8 个，wall time 缩短 ~8x
#
# 预估 wall time (Qwen2.5-7B):
#   单 GPU:  Pilot 1 ~2h, Pilot 2 ~5h, Pilot 3 ~5h
#   8 GPU:   Pilot 1 ~2h, Pilot 2 ~1h, Pilot 3 ~1h
#   QUICK:   各 ~15min (单 GPU) / ~5min (8 GPU)

set -e
cd "$(dirname "$0")"

# Source GPU utilities
source scripts/gpu_utils.sh
auto_setup

# Activate venv
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    echo "ERROR: .venv not found. Run: bash setup.sh"
    exit 1
fi

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PILOT="${1:-all}"
MAX_STEPS="${MAX_STEPS:-200}"
OUTPUT="results/csd_pilot"

if [ "${QUICK:-0}" = "1" ]; then
    MAX_STEPS=50
    echo "[QUICK MODE] max_steps=50"
fi

# Detect vLLM
VLLM_FLAG=""
if python3 -c "import vllm" 2>/dev/null; then
    VLLM_FLAG="--use_vllm"
    echo "vLLM detected — using accelerated generation"
fi

echo ""
echo "============================================"
echo " CSD Pilot Experiments"
echo " Model:  $MODEL"
echo " Pilot:  $PILOT"
echo " Steps:  $MAX_STEPS"
echo " GPUs:   $NUM_GPUS"
echo " Output: $OUTPUT"
echo "============================================"
echo ""

# ─────────────────────────────────────────────
# Helper: run a single training on a specific GPU
# ─────────────────────────────────────────────
run_on_gpu() {
    local gpu_id=$1
    shift
    CUDA_VISIBLE_DEVICES=$(get_gpu_id "$gpu_id") python3 scripts/run_csd_pilot.py "$@"
}

# ─────────────────────────────────────────────
# Pilot 1: CSD Verification (4 runs, sequential — each is short)
# Uses single GPU per run, runs 4 ρ values sequentially
# ─────────────────────────────────────────────
run_pilot1() {
    echo "===== PILOT 1: CSD Equivalence Verification ====="
    local P1_DIR="$OUTPUT/pilot1_csd_verification"
    mkdir -p "$P1_DIR"

    local rho_values=(0.5 1.0 2.0 3.0)
    local pids=()
    local gpu_idx=0

    for rho in "${rho_values[@]}"; do
        local run_dir="$P1_DIR/rho${rho}_seed42"
        if [ -f "$run_dir/pilot_results.json" ]; then
            echo "  [skip] rho=$rho seed=42 already done"
            continue
        fi
        echo "  [launch] rho=$rho seed=42 → GPU $(get_gpu_id $gpu_idx)"
        CUDA_VISIBLE_DEVICES=$(get_gpu_id $gpu_idx) python3 scripts/run_csd_pilot.py \
            --pilot 1_single --model "$MODEL" --max_steps "$MAX_STEPS" \
            --output_dir "$P1_DIR" --rho "$rho" --seeds 1 $VLLM_FLAG \
            > "$P1_DIR/log_rho${rho}.txt" 2>&1 &
        pids+=($!)
        gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))

        # If all GPUs busy, wait for a slot
        if [ ${#pids[@]} -ge "$NUM_GPUS" ]; then
            wait "${pids[0]}"
            pids=("${pids[@]:1}")
        fi
    done

    # Wait for remaining
    for pid in "${pids[@]}"; do wait "$pid"; done
    echo "  Pilot 1 complete."
}

# ─────────────────────────────────────────────
# Pilot 2: ADQ Collapse (多 seed 并行)
# 最关键的 pilot — constant ρ=1.0 vs ADQ
# ─────────────────────────────────────────────
run_pilot2() {
    echo "===== PILOT 2: ADQ Collapse Elimination ====="
    local P2_DIR="$OUTPUT/pilot2_adq_collapse"
    mkdir -p "$P2_DIR"

    local n_seeds="${SEEDS_P2:-5}"
    [ "${QUICK:-0}" = "1" ] && n_seeds=3
    local pids=()
    local gpu_idx=0

    # Phase A: constant ρ=1.0 runs (parallel across GPUs)
    echo "  --- Phase A: constant ρ=1.0 ($n_seeds seeds) ---"
    for ((s=42; s<42+n_seeds; s++)); do
        local run_dir="$P2_DIR/rho1.00_seed${s}"
        if [ -f "$run_dir/pilot_results.json" ]; then
            echo "  [skip] const ρ=1.0 seed=$s already done"
            continue
        fi
        echo "  [launch] const ρ=1.0 seed=$s → GPU $(get_gpu_id $gpu_idx)"
        CUDA_VISIBLE_DEVICES=$(get_gpu_id $gpu_idx) python3 scripts/run_csd_pilot.py \
            --pilot 2_single --model "$MODEL" --max_steps "$MAX_STEPS" \
            --output_dir "$P2_DIR" --rho 1.0 --seeds 1 --seed_start "$s" \
            $VLLM_FLAG > "$P2_DIR/log_const_seed${s}.txt" 2>&1 &
        pids+=($!)
        gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))

        if [ ${#pids[@]} -ge "$NUM_GPUS" ]; then
            wait "${pids[0]}"
            pids=("${pids[@]:1}")
        fi
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    pids=()
    gpu_idx=0

    # Phase B: ADQ runs (parallel across GPUs)
    echo "  --- Phase B: ADQ ($n_seeds seeds) ---"
    for ((s=42; s<42+n_seeds; s++)); do
        local run_dir="$P2_DIR/rho1.00_seed${s}_adq"
        if [ -f "$run_dir/pilot_results.json" ]; then
            echo "  [skip] ADQ seed=$s already done"
            continue
        fi
        echo "  [launch] ADQ seed=$s → GPU $(get_gpu_id $gpu_idx)"
        CUDA_VISIBLE_DEVICES=$(get_gpu_id $gpu_idx) python3 scripts/run_csd_pilot.py \
            --pilot 2_single --model "$MODEL" --max_steps "$MAX_STEPS" \
            --output_dir "$P2_DIR" --rho 1.0 --seeds 1 --seed_start "$s" --use_adq \
            $VLLM_FLAG > "$P2_DIR/log_adq_seed${s}.txt" 2>&1 &
        pids+=($!)
        gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))

        if [ ${#pids[@]} -ge "$NUM_GPUS" ]; then
            wait "${pids[0]}"
            pids=("${pids[@]:1}")
        fi
    done
    for pid in "${pids[@]}"; do wait "$pid"; done

    echo "  Pilot 2 complete. Generating summary..."
    python3 -c "
import json, glob, os, numpy as np
d = '$P2_DIR'
const_results, adq_results = [], []
for f in sorted(glob.glob(os.path.join(d, 'rho1.00_seed*/pilot_results.json'))):
    with open(f) as fh:
        r = json.load(fh)
    if r.get('use_adq'):
        adq_results.append(r)
    else:
        const_results.append(r)
n = max(len(const_results), len(adq_results))
cc = sum(1 for r in const_results if r.get('collapsed', False))
ac = sum(1 for r in adq_results if r.get('collapsed', False))
summary = {
    'n_seeds': n, 'rho': 1.0,
    'constant_rho': {'collapse_count': cc, 'collapse_rate': round(cc/max(n,1),2),
        'mean_final_reward': round(np.mean([r.get('final_reward_mean',0) for r in const_results]),4) if const_results else 0},
    'adq': {'collapse_count': ac, 'collapse_rate': round(ac/max(n,1),2),
        'mean_final_reward': round(np.mean([r.get('final_reward_mean',0) for r in adq_results]),4) if adq_results else 0},
    'killer_result': cc > 0 and ac == 0,
}
with open(os.path.join(d, 'pilot2_summary.json'), 'w') as fh:
    json.dump(summary, fh, indent=2)
print(f'Constant: {cc}/{n} collapsed ({100*cc//max(n,1)}%)')
print(f'ADQ:      {ac}/{n} collapsed ({100*ac//max(n,1)}%)')
if summary['killer_result']: print('KILLER: ADQ eliminates ALL collapse!')
"
}

# ─────────────────────────────────────────────
# Pilot 3: Q_CSD Predictor (多 seed 并行)
# ─────────────────────────────────────────────
run_pilot3() {
    echo "===== PILOT 3: Q_CSD Collapse Predictor ====="
    local P3_DIR="$OUTPUT/pilot3_qcsd_predictor"
    mkdir -p "$P3_DIR"

    local n_seeds="${SEEDS_P3:-10}"
    [ "${QUICK:-0}" = "1" ] && n_seeds=5
    local pids=()
    local gpu_idx=0

    for ((s=42; s<42+n_seeds; s++)); do
        local run_dir="$P3_DIR/rho1.00_seed${s}"
        if [ -f "$run_dir/pilot_results.json" ]; then
            echo "  [skip] seed=$s already done"
            continue
        fi
        echo "  [launch] ρ=1.0 seed=$s → GPU $(get_gpu_id $gpu_idx)"
        CUDA_VISIBLE_DEVICES=$(get_gpu_id $gpu_idx) python3 scripts/run_csd_pilot.py \
            --pilot 3_single --model "$MODEL" --max_steps "$MAX_STEPS" \
            --output_dir "$P3_DIR" --rho 1.0 --seeds 1 --seed_start "$s" \
            $VLLM_FLAG > "$P3_DIR/log_seed${s}.txt" 2>&1 &
        pids+=($!)
        gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))

        if [ ${#pids[@]} -ge "$NUM_GPUS" ]; then
            wait "${pids[0]}"
            pids=("${pids[@]:1}")
        fi
    done
    for pid in "${pids[@]}"; do wait "$pid"; done

    echo "  Pilot 3 complete. Computing AUROC..."
    python3 scripts/run_csd_pilot.py --pilot 3_analyze --output_dir "$P3_DIR" 2>/dev/null || \
    python3 -c "
import json, glob, os, numpy as np
d = '$P3_DIR'
early_q, collapsed = [], []
for f in sorted(glob.glob(os.path.join(d, 'rho1.00_seed*/csd_logs.json'))):
    with open(f) as fh:
        csd = json.load(fh)
    if len(csd) >= 3:
        early_q.append(np.mean([x.get('q_csd',0) for x in csd[:3]]))
        collapsed.append(int(csd[-1].get('is_collapsed', False)))
auroc = None
if len(set(collapsed)) > 1:
    pos = [q for q,c in zip(early_q, collapsed) if c==1]
    neg = [q for q,c in zip(early_q, collapsed) if c==0]
    if pos and neg:
        auroc = sum(1 for p in pos for n in neg if n>p) / (len(pos)*len(neg))
summary = {'n_seeds': len(early_q), 'n_collapsed': sum(collapsed),
           'qcsd_auroc': round(auroc,4) if auroc else None,
           'early_qcsd': [round(q,6) for q in early_q], 'collapsed': collapsed}
with open(os.path.join(d, 'pilot3_summary.json'), 'w') as fh:
    json.dump(summary, fh, indent=2)
print(f'Collapsed: {sum(collapsed)}/{len(collapsed)}')
if auroc: print(f'Q_CSD AUROC: {auroc:.4f}')
"
}

# ─────────────────────────────────────────────
# Main dispatch
# ─────────────────────────────────────────────
case "$PILOT" in
    1)   run_pilot1 ;;
    2)   run_pilot2 ;;
    3)   run_pilot3 ;;
    all) run_pilot1; run_pilot2; run_pilot3 ;;
    *)   echo "Unknown pilot: $PILOT (use 1, 2, 3, or all)"; exit 1 ;;
esac

echo ""
echo "============================================"
echo " All pilots complete! Results in: $OUTPUT"
echo "============================================"

# Final summary
for summary_file in "$OUTPUT"/pilot*/pilot*_summary.json; do
    [ -f "$summary_file" ] || continue
    echo ""
    echo "--- $(basename "$(dirname "$summary_file")") ---"
    python3 -c "
import json
with open('$summary_file') as f:
    d = json.load(f)
if 'constant_rho' in d:
    print(f'  Constant ρ=1.0: {d[\"constant_rho\"][\"collapse_rate\"]*100:.0f}% collapse')
    print(f'  ADQ (CSD ρ*):   {d[\"adq\"][\"collapse_rate\"]*100:.0f}% collapse')
    if d.get('killer_result'): print('  KILLER: ADQ eliminates ALL collapse!')
elif 'qcsd_auroc' in d:
    auroc = d.get('qcsd_auroc')
    if auroc: print(f'  Q_CSD AUROC: {auroc:.4f}')
    print(f'  Collapsed: {d.get(\"n_collapsed\",0)}/{d.get(\"n_seeds\",0)}')
" 2>/dev/null || true
done
