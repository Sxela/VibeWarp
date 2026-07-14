#!/usr/bin/env bash
# VibeWarp launcher — run ./install.sh once first.
# All arguments are forwarded to `python -m vibewarp`.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f env/bin/activate ]; then
    echo "Environment not found. Run ./install.sh first." >&2
    exit 1
fi
# shellcheck disable=SC1091
source env/bin/activate
exec python -m vibewarp "$@"
