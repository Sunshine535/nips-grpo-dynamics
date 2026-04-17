#!/bin/bash
# 下载 CSD 实验所需的模型和数据集（使用镜像代理）
#
# 代理配置:
#   GitHub: https://ghfast.top/ (前缀到原 github URL)
#   HuggingFace: https://hf-mirror.com
#
# 用法（在下载容器 /openbayes/input/input0 执行）:
#   # git clone 用代理:
#   git clone https://ghfast.top/https://github.com/Sunshine535/nips-grpo-dynamics.git
#   cd nips-grpo-dynamics
#   bash download_assets.sh
#
# 下载完成后:
#   scp -r downloads/ tju-hpc:/ytech_m2v4_hdd/mengzijie/.cache/hf/
#
# 环境变量:
#   NO_PROXY=1         # 禁用 HF 镜像
#   MODELS_ONLY=1      # 仅下载模型
#   DATASETS_ONLY=1    # 仅下载数据集
#   TEST_CONN=1        # 测试到 tju-hpc 的连通性

set -e
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$PROJ_DIR/downloads}"
mkdir -p "$DOWNLOAD_DIR"

# ─── 代理配置 ───
if [ "${NO_PROXY:-0}" != "1" ]; then
    export HF_ENDPOINT="https://hf-mirror.com"
    echo "[proxy] HF_ENDPOINT=$HF_ENDPOINT"
    echo "[proxy] GitHub: use https://ghfast.top/<original-url> for git clone"
fi

echo "============================================"
echo " CSD Asset Downloader (mirrored)"
echo " Project:    $PROJ_DIR"
echo " Download:   $DOWNLOAD_DIR"
echo " HF Mirror:  ${HF_ENDPOINT:-default}"
echo "============================================"

# ─── 0. 连通性测试（可选，需要先配置 SSH） ───
test_connectivity() {
    local target="${TJU_HPC_HOST:-tju-hpc}"
    echo ""
    echo "[connectivity] Testing connection to $target ..."
    if command -v ssh &>/dev/null; then
        if ssh -o ConnectTimeout=10 -o BatchMode=yes "$target" "echo 'OK from \$(hostname)'" 2>&1; then
            echo "  ✓ SSH OK"
        else
            echo "  ✗ SSH failed (check ~/.ssh/config or set TJU_HPC_HOST)"
        fi
    else
        echo "  ! ssh not available"
    fi
    if command -v ping &>/dev/null; then
        ping -c 2 -W 3 "$target" 2>/dev/null && echo "  ✓ ping OK" || echo "  ✗ ping failed"
    fi
}
[ "${TEST_CONN:-0}" = "1" ] && test_connectivity

# ─── 1. 安装 huggingface_hub（仅下载工具） ───
echo ""
echo "[setup] Installing huggingface_hub ..."
# 使用清华 pip 镜像
PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
pip install -q -U "huggingface_hub[cli]" -i "$PIP_MIRROR" 2>&1 | tail -3 || {
    pip install -q -U "huggingface_hub[cli]" 2>&1 | tail -3 || {
        echo "ERROR: failed to install huggingface_hub"; exit 1;
    }
}

# hf_transfer 在镜像下可能不兼容，默认关闭
export HF_HUB_ENABLE_HF_TRANSFER=0
pip install -q hf_transfer -i "$PIP_MIRROR" 2>/dev/null || true

# ─── 2. 配置缓存目标（与 tju-hpc 上的 HF_HOME 保持一致） ───
export HF_HOME="$DOWNLOAD_DIR"
export HF_HUB_CACHE="$DOWNLOAD_DIR/hub"
mkdir -p "$HF_HUB_CACHE"

# ─── 3. 模型清单（按优先级） ───
MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"   # primary — known collapse at ρ=1.0
    "Qwen/Qwen3-8B"               # secondary — newer instruct model
    "Qwen/Qwen3.5-9B"             # latest — for scaling verification
)

# ─── 4. 数据集清单 ───
DATASETS=(
    "openai/gsm8k:main"           # primary training/eval
)

# ─── 5. 下载模型 ───
download_model() {
    local model="$1"
    local local_name="${model//\//--}"
    local target="$HF_HUB_CACHE/models--$local_name"
    if [ -d "$target" ] && [ "$(find "$target" -name "*.safetensors" 2>/dev/null | head -1)" ]; then
        echo "  [skip] $model (already exists)"
        return 0
    fi
    echo "  [download] $model (via $HF_ENDPOINT)"
    HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}" \
    huggingface-cli download "$model" \
        --resume-download 2>&1 | tail -5 || {
        echo "  ✗ failed: $model"
        return 1
    }
    echo "  ✓ $model done"
}

download_dataset() {
    local spec="$1"
    local name="${spec%:*}"
    local config="${spec#*:}"
    local local_name="${name//\//--}"
    local target="$HF_HUB_CACHE/datasets--$local_name"
    if [ -d "$target" ]; then
        echo "  [skip] $name (already exists)"
        return 0
    fi
    echo "  [download] dataset $name (config=$config, via $HF_ENDPOINT)"
    # Ensure datasets library is installed
    pip install -q datasets -i "$PIP_MIRROR" 2>/dev/null || pip install -q datasets 2>/dev/null || true
    HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}" \
    python3 -c "
import os
os.environ['HF_ENDPOINT'] = '${HF_ENDPOINT:-https://huggingface.co}'
from datasets import load_dataset
ds = load_dataset('$name', '$config' if '$config' != '$name' else None, trust_remote_code=True)
print(f'  ✓ loaded: {list(ds.keys())}, total_rows={sum(len(s) for s in ds.values())}')
" 2>&1 | tail -3 || echo "  ✗ failed: $name"
}

if [ "${DATASETS_ONLY:-0}" != "1" ]; then
    echo ""
    echo "[models] Downloading ${#MODELS[@]} model(s) ..."
    for m in "${MODELS[@]}"; do download_model "$m" || true; done
fi

if [ "${MODELS_ONLY:-0}" != "1" ]; then
    echo ""
    echo "[datasets] Downloading ${#DATASETS[@]} dataset(s) ..."
    for d in "${DATASETS[@]}"; do download_dataset "$d" || true; done
fi

# ─── 6. 总结 + 下一步指令 ───
echo ""
echo "============================================"
echo " Download complete"
echo "============================================"
du -sh "$DOWNLOAD_DIR" 2>/dev/null
echo ""
echo "Total assets in: $DOWNLOAD_DIR"
ls -la "$HF_HUB_CACHE" 2>/dev/null | head -20
echo ""
echo "============================================"
echo " Next steps (transfer to tju-hpc):"
echo "============================================"
cat <<EOF

# 在下载容器执行（推送到 tju-hpc）:
rsync -avzP --partial $DOWNLOAD_DIR/hub/ \\
    tju-hpc:/ytech_m2v4_hdd/mengzijie/.cache/hf/hub/

# 或者打包后 scp（适合大文件断点续传困难时）:
tar -czf /tmp/csd_assets.tar.gz -C $DOWNLOAD_DIR hub/
scp /tmp/csd_assets.tar.gz tju-hpc:/tmp/

# 在 tju-hpc 解压:
mkdir -p /ytech_m2v4_hdd/mengzijie/.cache/hf/
tar -xzf /tmp/csd_assets.tar.gz -C /ytech_m2v4_hdd/mengzijie/.cache/hf/

# 验证（在 tju-hpc）:
export HF_HOME=/ytech_m2v4_hdd/mengzijie/.cache/hf
python3 -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct'); print('OK')"

EOF
