#!/bin/bash
# CSD Pilot — 多卡自适应一键启动脚本
# 用法:
#   bash run_csd_pilot.sh           # 运行全部 3 个 pilot (自动检测 GPU)
#   bash run_csd_pilot.sh 2         # 只运行 Pilot 2 (ADQ collapse 消除)
#   QUICK=1 bash run_csd_pilot.sh   # 快速模式 (50 steps)
#
# 环境变量:
#   MODEL=Qwen/Qwen3.5-9B       # 默认模型
#   CRITICAL_RHO=0.7             # Pilot 2 使用的 ρ (Qwen3.5 崩溃点)
#   MAX_STEPS=200                # 训练步数
#   SEEDS_P2=5 SEEDS_P3=10      # seed 数量

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

# --- HF cache: use shared cache directory ---
export HF_HOME="/ytech_m2v4_hdd/mengzijie/.cache/hf"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME"
echo "HF_HOME: $HF_HOME"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
PILOT="${1:-all}"
MAX_STEPS="${MAX_STEPS:-200}"
CRITICAL_RHO="${CRITICAL_RHO:-0.7}"
OUTPUT="results/csd_pilot"

if [ "${QUICK:-0}" = "1" ]; then
    MAX_STEPS=50
    echo "[QUICK MODE] max_steps=50"
fi

# ─── 前置验证：确保代码能 import ───
echo ""
echo "[preflight] Verifying imports..."
python3 -c "
from src.torch_compat import apply_torch_compat_patch
apply_torch_compat_patch()
from src.rho_grpo_trainer import RhoGRPOTrainer, AdaBalanceGRPOTrainer
from src.csd_logging import CSDLoggingCallback
from trl import GRPOConfig
print('  All imports OK')
" || { echo "ERROR: Import check failed. Fix dependencies first."; exit 1; }

echo ""
echo "============================================"
echo " CSD Pilot Experiments"
echo " Model:       $MODEL"
echo " Pilot:       $PILOT"
echo " Steps:       $MAX_STEPS"
echo " Critical ρ:  $CRITICAL_RHO"
echo " GPUs:        $NUM_GPUS"
echo " Output:      $OUTPUT"
echo "============================================"
echo ""

# ─── Helper: run one training, capture exit code ───
run_one() {
    local gpu_idx=$1; shift
    local log_file=$1; shift
    CUDA_VISIBLE_DEVICES=$(get_gpu_id "$gpu_idx") python3 scripts/run_csd_pilot.py "$@" \
        > "$log_file" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "  [FAIL] $(basename "$log_file") — see log for details"
        tail -5 "$log_file" 2>/dev/null
    fi
    return $rc
}

# ─── Pilot 1: CSD Verification ───
run_pilot1() {
    echo "===== PILOT 1: CSD Equivalence Verification ====="
    local P1_DIR="$OUTPUT/pilot1_csd_verification"
    mkdir -p "$P1_DIR"

    local rho_values=(0.5 1.0 2.0 3.0)
    local pids=()
    local gpu_idx=0

    for rho in "${rho_values[@]}"; do
        local rho_fmt=$(printf "%.2f" "$rho")
        local run_dir="$P1_DIR/rho${rho_fmt}_seed42"
        if [ -f "$run_dir/pilot_results.json" ]; then
            echo "  [skip] rho=$rho seed=42 already done"
            continue
        fi
        echo "  [launch] rho=$rho seed=42 → GPU $(get_gpu_id $gpu_idx)"
        run_one $gpu_idx "$P1_DIR/log_rho${rho_fmt}.txt" \
            --pilot 1_single --model "$MODEL" --max_steps "$MAX_STEPS" \
            --output_dir "$P1_DIR" --rho "$rho" --seed_start 42 $VLLM_FLAG &
        pids+=($!)
        gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))
        if [ ${#pids[@]} -ge "$NUM_GPUS" ]; then
            wait "${pids[0]}" || true
            pids=("${pids[@]:1}")
        fi
    done
    for pid in "${pids[@]}"; do wait "$pid" || true; done
    echo "  Pilot 1 done."
}

# ─── Pilot 2: ADQ Collapse Elimination ───
run_pilot2() {
    echo "===== PILOT 2: ADQ Collapse Elimination ====="
    local P2_DIR="$OUTPUT/pilot2_adq_collapse"
    mkdir -p "$P2_DIR"

    local n_seeds="${SEEDS_P2:-5}"
    [ "${QUICK:-0}" = "1" ] && n_seeds=3

    local rho_fmt=$(printf "%.2f" "$CRITICAL_RHO")
    echo "  Critical ρ=$CRITICAL_RHO, seeds=$n_seeds"

    # Phase A: constant ρ
    echo "  --- Phase A: constant ρ=$CRITICAL_RHO ---"
    local pids=()
    local gpu_idx=0
    for ((s=42; s<42+n_seeds; s++)); do
        local run_dir="$P2_DIR/rho${rho_fmt}_seed${s}"
        if [ -f "$run_dir/pilot_results.json" ]; then
            echo "  [skip] const seed=$s already done"
            continue
        fi
        echo "  [launch] const ρ=$CRITICAL_RHO seed=$s → GPU $(get_gpu_id $gpu_idx)"
        run_one $gpu_idx "$P2_DIR/log_const_seed${s}.txt" \
            --pilot 2_single --model "$MODEL" --max_steps "$MAX_STEPS" \
            --output_dir "$P2_DIR" --rho "$CRITICAL_RHO" --seed_start "$s" \
            $VLLM_FLAG &
        pids+=($!)
        gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))
        if [ ${#pids[@]} -ge "$NUM_GPUS" ]; then
            wait "${pids[0]}" || true
            pids=("${pids[@]:1}")
        fi
    done
    for pid in "${pids[@]}"; do wait "$pid" || true; done

    # Phase B: ADQ
    echo "  --- Phase B: ADQ ---"
    pids=()
    gpu_idx=0
    for ((s=42; s<42+n_seeds; s++)); do
        local run_dir="$P2_DIR/rho${rho_fmt}_seed${s}_adq"
        if [ -f "$run_dir/pilot_results.json" ]; then
            echo "  [skip] ADQ seed=$s already done"
            continue
        fi
        echo "  [launch] ADQ seed=$s → GPU $(get_gpu_id $gpu_idx)"
        run_one $gpu_idx "$P2_DIR/log_adq_seed${s}.txt" \
            --pilot 2_single --model "$MODEL" --max_steps "$MAX_STEPS" \
            --output_dir "$P2_DIR" --rho "$CRITICAL_RHO" --seed_start "$s" --use_adq \
            $VLLM_FLAG &
        pids+=($!)
        gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))
        if [ ${#pids[@]} -ge "$NUM_GPUS" ]; then
            wait "${pids[0]}" || true
            pids=("${pids[@]:1}")
        fi
    done
    for pid in "${pids[@]}"; do wait "$pid" || true; done

    # Summary: glob all pilot_results.json, regardless of rho format
    echo "  Generating summary..."
    python3 -c "
import json, glob, os, numpy as np
d = '$P2_DIR'
const_results, adq_results = [], []
for f in sorted(glob.glob(os.path.join(d, '*/pilot_results.json'))):
    with open(f) as fh:
        r = json.load(fh)
    if r.get('use_adq'):
        adq_results.append(r)
    else:
        const_results.append(r)
n_const = len(const_results)
n_adq = len(adq_results)
cc = sum(1 for r in const_results if r.get('collapsed', False))
ac = sum(1 for r in adq_results if r.get('collapsed', False))
summary = {
    'n_seeds_const': n_const, 'n_seeds_adq': n_adq,
    'rho': $CRITICAL_RHO,
    'constant_rho': {
        'collapse_count': cc,
        'collapse_rate': round(cc/max(n_const,1), 2),
        'mean_final_reward': round(np.mean([r.get('final_reward_mean',0) for r in const_results]), 4) if const_results else 0,
    },
    'adq': {
        'collapse_count': ac,
        'collapse_rate': round(ac/max(n_adq,1), 2),
        'mean_final_reward': round(np.mean([r.get('final_reward_mean',0) for r in adq_results]), 4) if adq_results else 0,
    },
    'killer_result': cc > 0 and ac == 0,
}
with open(os.path.join(d, 'pilot2_summary.json'), 'w') as fh:
    json.dump(summary, fh, indent=2)
print(f'Constant rho={$CRITICAL_RHO}: {cc}/{n_const} collapsed')
print(f'ADQ:                          {ac}/{n_adq} collapsed')
if summary['killer_result']: print('KILLER: ADQ eliminates ALL collapse!')
elif n_const == 0 and n_adq == 0: print('WARNING: No results found. Check log files.')
"
}

# ─── Pilot 3: Q_CSD Predictor ───
run_pilot3() {
    echo "===== PILOT 3: Q_CSD Collapse Predictor ====="
    local P3_DIR="$OUTPUT/pilot3_qcsd_predictor"
    mkdir -p "$P3_DIR"

    local n_seeds="${SEEDS_P3:-10}"
    [ "${QUICK:-0}" = "1" ] && n_seeds=5

    local pids=()
    local gpu_idx=0
    for ((s=42; s<42+n_seeds; s++)); do
        local rho_fmt=$(printf "%.2f" "$CRITICAL_RHO")
        local run_dir="$P3_DIR/rho${rho_fmt}_seed${s}"
        if [ -f "$run_dir/pilot_results.json" ]; then
            echo "  [skip] seed=$s already done"
            continue
        fi
        echo "  [launch] ρ=$CRITICAL_RHO seed=$s → GPU $(get_gpu_id $gpu_idx)"
        run_one $gpu_idx "$P3_DIR/log_seed${s}.txt" \
            --pilot 3_single --model "$MODEL" --max_steps "$MAX_STEPS" \
            --output_dir "$P3_DIR" --rho "$CRITICAL_RHO" --seed_start "$s" \
            $VLLM_FLAG &
        pids+=($!)
        gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))
        if [ ${#pids[@]} -ge "$NUM_GPUS" ]; then
            wait "${pids[0]}" || true
            pids=("${pids[@]:1}")
        fi
    done
    for pid in "${pids[@]}"; do wait "$pid" || true; done

    echo "  Computing AUROC..."
    python3 -c "
import json, glob, os, numpy as np
d = '$P3_DIR'
early_q, collapsed = [], []
for f in sorted(glob.glob(os.path.join(d, '*/csd_logs.json'))):
    with open(f) as fh:
        csd = json.load(fh)
    if len(csd) >= 3:
        early_q.append(np.mean([x.get('q_csd',0) for x in csd[:3]]))
        collapsed.append(int(csd[-1].get('is_collapsed', False)))
auroc = None
if len(set(collapsed)) > 1 and early_q:
    pos = [q for q,c in zip(early_q, collapsed) if c==1]
    neg = [q for q,c in zip(early_q, collapsed) if c==0]
    if pos and neg:
        auroc = sum(1 for p in pos for n in neg if n>p) / (len(pos)*len(neg))
summary = {'n_seeds': len(early_q), 'n_collapsed': sum(collapsed) if collapsed else 0,
           'qcsd_auroc': round(auroc,4) if auroc else None,
           'early_qcsd': [round(q,6) for q in early_q], 'collapsed': collapsed}
with open(os.path.join(d, 'pilot3_summary.json'), 'w') as fh:
    json.dump(summary, fh, indent=2)
n = len(collapsed)
print(f'Collapsed: {sum(collapsed)}/{n}')
if auroc: print(f'Q_CSD AUROC: {auroc:.4f}')
elif n == 0: print('WARNING: No results found. Check log files.')
"
}

# ─── Main dispatch ───
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
for sf in "$OUTPUT"/pilot*/pilot*_summary.json; do
    [ -f "$sf" ] || continue
    echo ""
    echo "--- $(basename "$(dirname "$sf")") ---"
    python3 -c "
import json
with open('$sf') as f: d = json.load(f)
if 'constant_rho' in d:
    cr = d['constant_rho']; ar = d['adq']
    print(f'  Constant: {cr[\"collapse_count\"]}/{d.get(\"n_seeds_const\",\"?\")} collapsed ({cr[\"collapse_rate\"]*100:.0f}%)')
    print(f'  ADQ:      {ar[\"collapse_count\"]}/{d.get(\"n_seeds_adq\",\"?\")} collapsed ({ar[\"collapse_rate\"]*100:.0f}%)')
    if d.get('killer_result'): print('  KILLER: ADQ eliminates ALL collapse!')
elif 'qcsd_auroc' in d:
    auroc = d.get('qcsd_auroc')
    if auroc: print(f'  Q_CSD AUROC: {auroc:.4f}')
    print(f'  Collapsed: {d.get(\"n_collapsed\",0)}/{d.get(\"n_seeds\",0)}')
" 2>/dev/null || true
done
