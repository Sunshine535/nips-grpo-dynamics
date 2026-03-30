#!/bin/bash
# =============================================================================
# SenseCore ACP launch script for nips-grpo-dynamics
#
# Docker image: PyTorch 2.10, TRL 0.29.1, PEFT, DeepSpeed, Accelerate
#
# Usage:
#   bash run_acp.sh                      # full pipeline
#   QUICK=1 bash run_acp.sh              # quick mode (reduced grids)
#   FORCE_RERUN=1 bash run_acp.sh        # ignore phase markers
#   bash run_acp.sh --quick              # same as QUICK=1
# =============================================================================
set -euo pipefail

# === Paths ===
PROJECT_DIR="/data/szs/250010072/nwh/nips-grpo-dynamics"
DATA_DIR="/data/szs/share/grpo-dynamics"

export MODEL_9B="/data/szs/share/Qwen3.5-9B"
export MODEL_27B="${MODEL_27B:-Qwen/Qwen3.5-27B}"

# === Environment ===
export HF_HOME="${DATA_DIR}/hf_cache"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=8

mkdir -p "$HF_HOME"

# === GPU detection ===
if ! command -v nvidia-smi &>/dev/null; then
    echo "[ERROR] nvidia-smi not found"
    exit 1
fi
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "[ERROR] No GPUs detected"
    exit 1
fi

GPU_IDS=""
for ((i=0; i<NUM_GPUS; i++)); do
    [ -n "$GPU_IDS" ] && GPU_IDS="${GPU_IDS},"
    GPU_IDS="${GPU_IDS}${i}"
done
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')

# Validate via torch (use total_memory, not total_mem)
python3 -c "
import torch
n = torch.cuda.device_count()
print(f'PyTorch: {torch.__version__}')
print(f'CUDA:    {torch.version.cuda}')
print(f'GPUs:    {n}')
for i in range(n):
    props = torch.cuda.get_device_properties(i)
    mem_gib = props.total_memory / (1024**3)
    print(f'  GPU {i}: {props.name}  {mem_gib:.1f} GiB')
"

echo "============================================================"
echo " SenseCore ACP — nips-grpo-dynamics"
echo " GPUs: ${NUM_GPUS} × ${GPU_MEM_MIB} MiB"
echo " PROJECT_DIR: ${PROJECT_DIR}"
echo " DATA_DIR:    ${DATA_DIR}"
echo " MODEL_9B:    ${MODEL_9B}"
echo " HF_HOME:     ${HF_HOME}"
echo "============================================================"

# === Validate local model ===
if [ ! -d "$MODEL_9B" ]; then
    echo "[ERROR] Model dir not found: $MODEL_9B"
    exit 1
fi
echo "[OK] Model found: $MODEL_9B"

# === Project setup ===
cd "$PROJECT_DIR"

# Symlink results/ and logs/ to shared storage for persistence
SHARED_RESULTS="${DATA_DIR}/results"
SHARED_LOGS="${DATA_DIR}/logs"
SHARED_CKPTS="${DATA_DIR}/checkpoints"
mkdir -p "$SHARED_RESULTS" "$SHARED_LOGS" "$SHARED_CKPTS"

for LINK_NAME in results checkpoints; do
    SHARED_TARGET="${DATA_DIR}/${LINK_NAME}"
    LOCAL_PATH="${PROJECT_DIR}/${LINK_NAME}"
    if [ ! -L "$LOCAL_PATH" ]; then
        if [ -d "$LOCAL_PATH" ]; then
            cp -rn "$LOCAL_PATH/"* "$SHARED_TARGET/" 2>/dev/null || true
            rm -rf "$LOCAL_PATH"
        fi
        ln -sf "$SHARED_TARGET" "$LOCAL_PATH"
        echo "[SYMLINK] ${LINK_NAME}/ -> $SHARED_TARGET"
    fi
done

LOG_DIR="${PROJECT_DIR}/results/logs"
if [ ! -L "$LOG_DIR" ] && [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$SHARED_LOGS"
    ln -sf "$SHARED_LOGS" "$LOG_DIR"
    echo "[SYMLINK] results/logs/ -> $SHARED_LOGS"
fi
mkdir -p "$PROJECT_DIR/results/logs"

# === Install missing deps (Docker should have most) ===
pip install --quiet datasets scipy matplotlib pandas huggingface_hub tqdm pyyaml wandb 2>/dev/null || true

# === Patch YAML configs to use local model paths ===
echo "[CONFIG] Patching model paths to local: $MODEL_9B"
python3 -c "
import yaml, sys

patches = {
    'configs/sweep_grid.yaml': [('model', 'name')],
    'configs/grpo_9b.yaml':    [('model', 'name_or_path')],
    'configs/rho_sweep.yaml':  [('model', 'name')],
}

model_path = '${MODEL_9B}'

for cfg_path, keys_list in patches.items():
    try:
        with open(cfg_path) as f:
            data = yaml.safe_load(f)
        changed = False
        for section, key in keys_list:
            if section in data and key in data[section]:
                old = data[section][key]
                if old != model_path:
                    data[section][key] = model_path
                    changed = True
                    print(f'  {cfg_path}: {section}.{key} = {model_path} (was {old})')
        if changed:
            with open(cfg_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        print(f'  [WARN] {cfg_path}: {e}', file=sys.stderr)
"

# === Mark Phase 0 done (models pre-installed) ===
PHASE_MARKERS="${PROJECT_DIR}/results/.phase_markers"
mkdir -p "$PHASE_MARKERS"
echo "{\"phase\":0,\"completed\":\"$(date -u '+%Y-%m-%dT%H:%M:%SZ')\",\"note\":\"models pre-installed at $MODEL_9B\"}" \
    > "$PHASE_MARKERS/phase_0.done"
echo "[SKIP] Phase 0: Model pre-installed at $MODEL_9B"

# === Run pipeline ===
echo ""
echo "============================================================"
echo " Pipeline starting: $(date)"
echo " QUICK=${QUICK:-0}  FORCE_RERUN=${FORCE_RERUN:-0}"
echo "============================================================"

export PROJ_DIR_ROOT="$PROJECT_DIR"

bash scripts/run_all_experiments.sh "$@" 2>&1 | tee "${PROJECT_DIR}/results/logs/run_acp_$(date +%Y%m%d_%H%M%S).log"
EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "============================================================"
    echo " GRPO Dynamics — Pipeline Complete [$(date)]"
    echo " Results: results/ -> ${SHARED_RESULTS}"
    echo " Logs:    results/logs/"
    echo "============================================================"
else
    echo "============================================================"
    echo " Pipeline FAILED (exit code: $EXIT_CODE)"
    echo " Check log: results/logs/"
    echo " To resume: bash run_acp.sh (completed phases auto-skipped)"
    echo "============================================================"
    exit $EXIT_CODE
fi
