#!/bin/bash
set -e
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="nips-grpo-dynamics"

echo "============================================"
echo " Environment Setup (venv + pip + PyTorch 2.10 + CUDA 12.8)"
echo "============================================"

# --- Pick Python (prefer 3.12 > 3.11 > 3.10) ---
PYTHON_CMD=""
for try in python3.12 python3.11 python3.10 python3; do
    if command -v "$try" &>/dev/null; then
        PYTHON_CMD="$try"
        break
    fi
done
if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Need python3.10+."
    exit 1
fi

# Validate Python version >= 3.10
PY_VERSION="$("$PYTHON_CMD" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)"
PY_MAJOR="$("$PYTHON_CMD" -c "import sys; print(sys.version_info.major)" 2>/dev/null)"
PY_MINOR="$("$PYTHON_CMD" -c "import sys; print(sys.version_info.minor)" 2>/dev/null)"
if [ -z "$PY_MAJOR" ] || [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "${PY_MINOR:-0}" -lt 10 ]; }; then
    echo "ERROR: Python >= 3.10 required, found $PY_VERSION ($PYTHON_CMD)"
    exit 1
fi
echo "[1/5] Using: $PYTHON_CMD ($PY_VERSION)"

# --- Create venv ---
VENV_DIR="$PROJ_DIR/.venv"
if [ -d "$VENV_DIR" ] && { [ ! -f "$VENV_DIR/bin/activate" ] || [ ! -x "$VENV_DIR/bin/python" ]; }; then
    echo "[2/5] Removing incomplete .venv (missing bin/activate or python) ..."
    rm -rf "$VENV_DIR"
fi
if [ ! -d "$VENV_DIR" ]; then
    echo "[2/5] Creating venv ..."
    if ! "$PYTHON_CMD" -m venv "$VENV_DIR"; then
        echo ""
        echo "ERROR: python -m venv failed (ensurepip). On Debian/Ubuntu:"
        ver="$("$PYTHON_CMD" -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))")"
        echo "  sudo apt install python${ver}-venv"
        rm -rf "$VENV_DIR" 2>/dev/null || true
        exit 1
    fi
else
    echo "[2/5] Venv exists: $VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-600}"

echo "[3/5] Upgrading pip ..."
python -m pip install -U pip setuptools wheel

echo "[4/5] Installing PyTorch + project deps ..."
if python -c "import torch; print(torch.__version__)" 2>/dev/null; then
    echo "  [info] System PyTorch detected — skipping torch install to prevent conflicts"
    grep -v '^torch' "$PROJ_DIR/requirements.txt" | python -m pip install -r /dev/stdin
else
    python -m pip install \
        "torch>=2.4.0" "torchvision" "torchaudio" \
        -r "$PROJ_DIR/requirements.txt" \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple
fi

# --- Optional: flash-attention ---
_FA_MARKER="$VENV_DIR/.flash_attn_attempted"
if [ ! -f "$_FA_MARKER" ]; then
    echo "[5/5] Installing flash-attn (optional, first time only) ..."
    python -m pip install flash-attn --no-build-isolation 2>/dev/null || echo "  flash-attn skipped (optional)"
    touch "$_FA_MARKER"
else
    echo "[5/5] Flash-attn already attempted (skip rebuild)"
fi

# --- Verify ---
echo ""
echo "============================================"
python -c "
import torch, transformers, datasets, trl, accelerate, peft
print(f'  PyTorch       : {torch.__version__}')
print(f'  Transformers  : {transformers.__version__}')
print(f'  TRL           : {trl.__version__}')
print(f'  Accelerate    : {accelerate.__version__}')
print(f'  PEFT          : {peft.__version__}')
print(f'  Datasets      : {datasets.__version__}')
print(f'  CUDA          : {torch.version.cuda}')
print(f'  GPUs          : {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
"
echo "============================================"
echo ""
echo "Setup complete!"
echo "  Activate:  source $VENV_DIR/bin/activate"
echo "  Run:       bash scripts/run_all_experiments.sh"
