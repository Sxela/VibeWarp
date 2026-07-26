#!/usr/bin/env bash
#
# Backward-compatible entry point. The complete Qwen Image Edit downloader now
# has a name that reflects everything it fetches, not only ConvRot checkpoints.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${script_dir}/download-qwen-image-edit.sh" "$@"
