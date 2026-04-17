#!/bin/bash
# 一键环境安装 + 运行实验（conda 版本）
# 在服务器 /openbayes/input/input0/nips-grpo-dynamics 下运行

set -e
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

# ─── Step 0: 检查资产 ───
HF_HOME="${HF_HOME:-/openbayes/input/input0}"
export HF_HOME
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "============================================"
echo " [0/4] Check downloaded assets"
echo "============================================"
echo "HF_HOME: $HF_HOME"
echo ""

MODELS_FOUND=0
for m in "Qwen--Qwen2.5-7B-Instruct" "Qwen--Qwen3-8B" "Qwen--Qwen3.5-9B"; do
    path="$HF_HUB_CACHE/models--$m"
    if [ -d "$path" ] && find "$path" -name "*.safetensors" -o -name "*.bin" 2>/dev/null | head -1 | grep -q .; then
        size=$(du -sh "$path" 2>/dev/null | cut -f1)
        echo "  ✓ $m ($size)"
        MODELS_FOUND=$((MODELS_FOUND + 1))
    else
        echo "  ✗ $m (missing)"
    fi
done

GSM8K_PATH="$HF_DATASETS_CACHE/openai___gsm8k"
if [ -d "$GSM8K_PATH" ] || [ -d "$HF_HUB_CACHE/datasets--openai--gsm8k" ]; then
    echo "  ✓ GSM8K dataset"
else
    echo "  ✗ GSM8K dataset (missing, will download)"
fi

if [ $MODELS_FOUND -eq 0 ]; then
    echo ""
    echo "ERROR: No models found in $HF_HUB_CACHE"
    echo "Download first: bash download_assets.sh"
    exit 1
fi
echo "  → $MODELS_FOUND model(s) ready"

# ─── Step 1: Conda 环境 ───
echo ""
echo "============================================"
echo " [1/4] Conda environment"
echo "============================================"

ENV_NAME="${ENV_NAME:-csd}"
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install miniconda first."
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "Creating conda env '$ENV_NAME' with Python 3.11 ..."
    conda create -y -n "$ENV_NAME" python=3.11
else
    echo "Conda env '$ENV_NAME' exists"
fi
conda activate "$ENV_NAME"
echo "Active: $(which python)"

# ─── Step 2: 依赖 ───
echo ""
echo "============================================"
echo " [2/4] Install dependencies"
echo "============================================"

PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
pip config set global.index-url "$PIP_MIRROR" 2>/dev/null || true

# 检测 CUDA 版本
TORCH_INDEX="https://download.pytorch.org/whl/cu121"
if command -v nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || echo "")
    if [ -n "$CUDA_VER" ]; then
        CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
        echo "CUDA: $CUDA_VER"
        if [ "$CUDA_MAJOR" -ge 13 ] || { [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 8 ]; }; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu128"
        elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        fi
    fi
fi

if ! python -c "import torch" 2>/dev/null; then
    echo "Installing torch from $TORCH_INDEX ..."
    pip install "torch>=2.4.0,<2.6.0" "torchvision" "torchaudio" --index-url "$TORCH_INDEX"
fi

echo "Installing requirements.txt ..."
pip install -r requirements.txt

# flash-attn 可选，失败不阻塞
pip install flash-attn --no-build-isolation 2>/dev/null || echo "  flash-attn skipped (optional)"

# 验证
python -c "
import torch, transformers, datasets, trl, accelerate, peft
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA:    {torch.version.cuda}')
print(f'  GPUs:    {torch.cuda.device_count()}')
print(f'  TRL:     {trl.__version__}')
print(f'  transformers: {transformers.__version__}')
"

# ─── Step 3: 运行 Pilot ───
echo ""
echo "============================================"
echo " [3/4] Run pilot experiment"
echo "============================================"

# 默认 QUICK 模式，先快速验证
PILOT="${PILOT:-2}"
QUICK="${QUICK:-1}"

export PILOT QUICK

echo "Running: QUICK=$QUICK bash run_csd_pilot.sh $PILOT"
QUICK="$QUICK" bash run_csd_pilot.sh "$PILOT"

# ─── Step 4: 显示结果 ───
echo ""
echo "============================================"
echo " [4/4] Results"
echo "============================================"

find results/csd_pilot -name "pilot*_summary.json" -exec echo "--- {} ---" \; -exec cat {} \; 2>/dev/null

echo ""
echo "Done. Full logs in: results/csd_pilot/"
