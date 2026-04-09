#!/bin/bash
# CSD Pilot — 一键启动脚本
# 用法:
#   bash run_csd_pilot.sh           # 运行全部 3 个 pilot
#   bash run_csd_pilot.sh 2         # 只运行 Pilot 2 (ADQ collapse 消除)
#   QUICK=1 bash run_csd_pilot.sh   # 快速模式 (50 steps)
#
# 预估时间 (单 GPU, Qwen2.5-7B):
#   Pilot 1: ~2h (4 runs × 200 steps)
#   Pilot 2: ~5h (10 runs × 200 steps)
#   Pilot 3: ~5h (10 runs × 200 steps)
#   QUICK 模式: 各 ~15min

set -e
cd "$(dirname "$0")"

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

echo "============================================"
echo " CSD Pilot Experiments"
echo " Model: $MODEL"
echo " Pilot: $PILOT"
echo " Steps: $MAX_STEPS"
echo " Output: $OUTPUT"
echo "============================================"

# Detect vLLM
VLLM_FLAG=""
if python3 -c "import vllm" 2>/dev/null; then
    VLLM_FLAG="--use_vllm"
    echo "vLLM detected — using accelerated generation"
fi

python3 scripts/run_csd_pilot.py \
    --pilot "$PILOT" \
    --model "$MODEL" \
    --max_steps "$MAX_STEPS" \
    --output_dir "$OUTPUT" \
    $VLLM_FLAG

echo ""
echo "============================================"
echo " Pilot complete! Results in: $OUTPUT"
echo "============================================"

# Quick summary
if [ -f "$OUTPUT/pilot2_adq_collapse/pilot2_summary.json" ]; then
    echo ""
    echo "--- Pilot 2 (ADQ Collapse) ---"
    python3 -c "
import json
with open('$OUTPUT/pilot2_adq_collapse/pilot2_summary.json') as f:
    d = json.load(f)
print(f'Constant ρ=1.0: {d[\"constant_rho\"][\"collapse_rate\"]*100:.0f}% collapse')
print(f'ADQ (CSD ρ*):   {d[\"adq\"][\"collapse_rate\"]*100:.0f}% collapse')
if d.get('killer_result'):
    print('🎯 KILLER: ADQ eliminates ALL collapse!')
"
fi

if [ -f "$OUTPUT/pilot3_qcsd_predictor/pilot3_summary.json" ]; then
    echo ""
    echo "--- Pilot 3 (Q_CSD Predictor) ---"
    python3 -c "
import json
with open('$OUTPUT/pilot3_qcsd_predictor/pilot3_summary.json') as f:
    d = json.load(f)
auroc = d.get('qcsd_auroc')
if auroc:
    print(f'Q_CSD AUROC: {auroc:.4f}')
else:
    print('Q_CSD AUROC: N/A (need both collapsed and non-collapsed runs)')
"
fi
