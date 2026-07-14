#!/usr/bin/env bash
# VibeWarp installer (Linux/macOS).
# Creates ./env, installs PyTorch + VibeWarp. Re-run safe.
# Requires: python3.11+ and ffmpeg available via your package manager.
# (The pinned deps require >=3.11; the project is validated on 3.14.)
set -euo pipefail
cd "$(dirname "$0")"

# PyTorch wheel index — change to match your CUDA driver
# (see https://pytorch.org/get-started/locally/). macOS ignores this.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

PY=python3
if ! $PY -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' 2>/dev/null; then
    echo "python3 >= 3.11 required (found: $($PY --version 2>&1))." >&2
    echo "Install it via your package manager, e.g. 'sudo apt install python3.11-venv'." >&2
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "WARNING: ffmpeg not found on PATH."
    echo "  Linux:  sudo apt install ffmpeg"
    echo "  macOS:  brew install ffmpeg"
    echo "Continuing install; rendering will need ffmpeg."
fi

if [ ! -d env ]; then
    echo "Creating virtual environment..."
    $PY -m venv env
fi
# shellcheck disable=SC1091
source env/bin/activate

if [ "$(uname)" = "Darwin" ]; then
    python -m pip install 'torch==2.11.0' 'torchvision==0.26.0'
else
    python -m pip install 'torch==2.11.0' 'torchvision==0.26.0' --index-url "$TORCH_INDEX"
fi

python -m pip install -e .

cat <<'EOF'
-----------------------------------------------------------------
Install complete.

Next steps:
  1. Download models for your settings file:
       ./run.sh --download-models --warpfusion-settings path/to/settings.txt
  2. Render:
       ./run.sh --warpfusion-settings path/to/settings.txt --video input.mp4
-----------------------------------------------------------------
EOF
