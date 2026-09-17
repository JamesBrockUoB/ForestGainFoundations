#!/usr/bin/env bash
#
# Installs project dependencies, picking the correct torch/torchvision/
# torchaudio build for the current machine:
#   - macOS -> plain PyPI wheels (CPU/MPS build)
#   - Linux -> pinned cu128 build from download.pytorch.org
#
# Run from the repo root: ./install_deps.sh

set -eu

REQ_FILE="requirements.txt"

CU128_TORCH="torch==2.11.0+cu128"
CU128_TORCHVISION="torchvision==0.26.0+cu128"
CU128_TORCHAUDIO="torchaudio==2.11.0+cu128"
CU128_INDEX="https://download.pytorch.org/whl/cu128"

PIP="python3 -m pip"
PIP_FLAGS="--no-cache-dir"

if [ ! -f "$REQ_FILE" ]; then
    echo "ERROR: $REQ_FILE not found in $(pwd). Run this from the repo root." >&2
    exit 1
fi

OS_NAME="$(uname -s)"

install_mac() {
    echo "== Detected macOS: installing plain torch/torchvision/torchaudio =="
    $PIP install $PIP_FLAGS torch torchvision torchaudio
}

install_linux() {
    echo "== Installing torch 2.11.0 pinned to CUDA 12.8 build =="
    $PIP install $PIP_FLAGS \
        "$CU128_TORCH" "$CU128_TORCHVISION" "$CU128_TORCHAUDIO" \
        --extra-index-url "$CU128_INDEX"
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

echo "== Verifying PyTorch installation =="

python3 -c "
import torch
print(f'torch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"

echo "== Installing project requirements from $REQ_FILE =="
$PIP install $PIP_FLAGS -r "$REQ_FILE"

echo "== Installing pytorch-lightning =="
$PIP install $PIP_FLAGS pytorch-lightning

echo "== Verifying PyTorch installation =="

python3 -c "
import torch
print(f'torch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"