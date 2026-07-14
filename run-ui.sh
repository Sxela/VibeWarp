#!/usr/bin/env bash
# VibeWarp Svelte web UI launcher — run ./install.sh once first.
# Serves on http://localhost:7860.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f env/bin/activate ]; then
    echo "Environment not found. Run ./install.sh first." >&2
    exit 1
fi
# shellcheck disable=SC1091
source env/bin/activate

# Install the optional local API server on first run.
if ! python -c 'import fastapi, uvicorn' 2>/dev/null; then
    echo "Installing UI dependencies..."
    pip install -e '.[ui]'
fi

exec python -m vibewarp.web "$@"
