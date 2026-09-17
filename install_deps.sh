#!/usr/bin/env bash
#
# Installs project dependencies, picking the correct torch/torchvision/
# torchaudio build for the current machine:
#   - macOS            -> plain PyPI wheels (CPU/MPS build)
#   - Linux + CUDA 12.x -> pinned cu128 build from download.pytorch.org
#   - Linux, no usable NVIDIA driver -> plain CPU wheels, with a warning
#
# Run from the repo root: ./install_deps.sh

set -euo pipefail

REQ_FILE="requirements.txt"
CU128_TORCH="torch==2.11.0+cu128"
CU128_TORCHVISION="torchvision==0.26.0+cu128"
CU128_TORCHAUDIO="torchaudio==2.11.0+cu128"
CU128_INDEX="https://download.pytorch.org/whl/cu128"

if [ ! -f "$REQ_FILE" ]; then
    echo "ERROR: $REQ_FILE not found in $(pwd). Run this from the repo root." >&2
    exit 1
fi

echo "== Installing base requirements from $REQ_FILE =="
pip install -r "$REQ_FILE"

OS_NAME="$(uname -s)"

install_mac() {
    echo "== Detected macOS: installing plain torch/torchvision/torchaudio =="
    pip install torch torchvision torchaudio
}

install_linux_cpu_fallback() {
    echo "WARNING: no usable NVIDIA driver detected; installing CPU-only torch." >&2
    echo "         GPU-accelerated code will not run. If this is wrong, check" >&2
    echo "         that 'nvidia-smi' works and the driver is loaded, then re-run." >&2
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

install_linux() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        install_linux_cpu_fallback
        return
    fi

    # nvidia-smi's header reports the *maximum* CUDA version the installed
    # driver supports (not necessarily what's currently loaded) -- e.g.
    # "CUDA Version: 12.8". Parse that out.
    DRIVER_CUDA="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | head -n1 | awk '{print $3}')"

    if [ -z "$DRIVER_CUDA" ]; then
        echo "WARNING: nvidia-smi ran but CUDA version couldn't be parsed from its output." >&2
        install_linux_cpu_fallback
        return
    fi

    DRIVER_MAJOR="${DRIVER_CUDA%%.*}"

    echo "== Detected NVIDIA driver supporting CUDA $DRIVER_CUDA =="

    if [ "$DRIVER_MAJOR" = "12" ]; then
        echo "== Installing torch 2.11.0 pinned to CUDA 12.8 build =="
        echo "   (PyPI's own default wheel for torch>=2.11 targets CUDA 13.0," >&2
        echo "   which will not run on a 12.x-only driver -- hence the explicit" >&2
        echo "   +cu128 pin and extra index below.)" >&2
        pip install "$CU128_TORCH" "$CU128_TORCHVISION" "$CU128_TORCHAUDIO" \
            --extra-index-url "$CU128_INDEX"
    elif [ "$DRIVER_MAJOR" -ge "13" ] 2>/dev/null; then
        echo "== Driver supports CUDA $DRIVER_CUDA (>=13); installing plain PyPI torch =="
        echo "   (PyPI's default wheel now targets CUDA 13.0, which matches.)" >&2
        pip install torch torchvision torchaudio
    else
        echo "WARNING: driver reports CUDA $DRIVER_CUDA, which this script doesn't" >&2
        echo "         have a known-good pin for. Falling back to CPU-only torch." >&2
        echo "         Install manually if you need GPU support at this CUDA version." >&2
        install_linux_cpu_fallback
    fi
}

case "$OS_NAME" in
    Darwin)
        install_mac
        ;;
    Linux)
        install_linux
        ;;
    *)
        echo "ERROR: unsupported OS '$OS_NAME'. Install torch manually." >&2
        exit 1
        ;;
esac

echo "== Installing pytorch-lightning =="
pip install pytorch-lightning

echo "== Done. Verifying torch install =="
python3 -c "
import torch
print(f'torch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"