#!/usr/bin/env bash
#
# Download every model file required by VibeWarp's Mage-Flow Edit and
# Mage-Flow Edit Turbo presets into the directory from which this script is
# invoked.
#
# Completed files are skipped. Interrupted downloads remain as NAME.part and
# resume from the existing byte count the next time the script is run.
#
# Optional environment variables:
#   HF_TOKEN     Hugging Face token, if authentication is required.
#   HF_REPO      Override the source repository.
#   HF_REVISION  Override the source revision.

set -euo pipefail

repo="${HF_REPO:-Comfy-Org/Mage-Flow}"
revision="${HF_REVISION:-main}"
destination="$PWD"

command -v curl >/dev/null 2>&1 || {
    echo "Error: curl is required." >&2
    exit 1
}

auth_args=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    auth_args=(-H "Authorization: Bearer ${HF_TOKEN}")
fi

download_file() {
    local remote_path="$1"
    local filename="${remote_path##*/}"
    local target="${destination}/${filename}"
    local partial="${target}.part"
    local download_url
    download_url="https://huggingface.co/${repo}/resolve/${revision}/${remote_path}?download=true"

    if [[ -f "$target" ]]; then
        echo "Skipping existing ${filename}"
        return
    fi

    if [[ -f "$partial" ]]; then
        echo "Resuming ${filename} from $(wc -c < "$partial" | tr -d ' ') bytes"
    else
        echo "Downloading ${filename}"
    fi

    curl --fail --location --show-error \
        --retry 5 --retry-delay 3 --retry-all-errors \
        --continue-at - \
        "${auth_args[@]}" \
        --output "$partial" \
        "$download_url"
    mv "$partial" "$target"
}

required_files=(
    "diffusion_models/mage_flow_edit_int8_convrot.safetensors"
    "diffusion_models/mage_flow_edit_turbo_int8_convrot.safetensors"
    "text_encoders/qwen3vl_4b_bf16.safetensors"
    "vae/mage_flow_vae_bf16.safetensors"
)

echo "Downloading ${#required_files[@]} Mage-Flow file(s) to ${destination}"
for remote_path in "${required_files[@]}"; do
    download_file "$remote_path"
done

echo "Mage-Flow model downloads complete."
