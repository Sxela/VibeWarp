#!/usr/bin/env bash
#
# Download the complete Qwen Image Edit 2511 model set used by VibeWarp into the
# directory from which this script is invoked:
#   - every Qwen ConvRot safetensors diffusion model in the ComfyUI repository
#   - Qwen 2.5 VL text/vision encoder
#   - Qwen Image VAE
#   - LightX2V 8-step Lightning LoRA
#   - Q5_K_M GGUF comparison checkpoint
#
# Completed files are skipped. Interrupted downloads remain as NAME.part and
# resume from the existing byte count the next time the script is run.
#
# Optional environment variables:
#   HF_TOKEN     Hugging Face token, if authentication is required.
#   HF_REPO      Override the source repository.
#   HF_REVISION  Override the source revision.

set -euo pipefail

repo="${HF_REPO:-Comfy-Org/Qwen-Image-Edit_ComfyUI}"
revision="${HF_REVISION:-main}"
destination="$PWD"
api_url="https://huggingface.co/api/models/${repo}/revision/${revision}"

command -v curl >/dev/null 2>&1 || {
    echo "Error: curl is required." >&2
    exit 1
}

auth_args=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    auth_args=(-H "Authorization: Bearer ${HF_TOKEN}")
fi

download_file() {
    local source_repo="$1"
    local source_revision="$2"
    local remote_path="$3"
    local filename="${remote_path##*/}"
    local target="${destination}/${filename}"
    local partial="${target}.part"
    local download_url
    download_url="https://huggingface.co/${source_repo}/resolve/${source_revision}/${remote_path}?download=true"

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

metadata_file="$(mktemp)"
cleanup() {
    rm -f "$metadata_file"
}
trap cleanup EXIT

echo "Reading file list from https://huggingface.co/${repo} ..."
curl --fail --silent --show-error --location \
    "${auth_args[@]}" \
    --output "$metadata_file" \
    "$api_url"

files=()
if command -v jq >/dev/null 2>&1; then
    while IFS= read -r remote_path; do
        [[ -n "$remote_path" ]] && files+=("$remote_path")
    done < <(
        jq -r '
            .siblings[].rfilename
            | select(test("(^|/)qwen[^/]*convrot[^/]*[.]safetensors$"; "i"))
        ' "$metadata_file"
    )
else
    python_cmd=""
    if command -v python3 >/dev/null 2>&1; then
        python_cmd="python3"
    elif command -v python >/dev/null 2>&1; then
        python_cmd="python"
    else
        echo "Error: jq, python3, or python is required to read Hugging Face metadata." >&2
        exit 1
    fi
    while IFS= read -r remote_path; do
        [[ -n "$remote_path" ]] && files+=("$remote_path")
    done < <(
        "$python_cmd" -c '
import json
import re
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    metadata = json.load(handle)
pattern = re.compile(r"(^|/)qwen[^/]*convrot[^/]*[.]safetensors$", re.I)
for sibling in metadata.get("siblings", []):
    path = sibling.get("rfilename", "")
    if pattern.search(path):
        print(path)
' "$metadata_file"
    )
fi

if (( ${#files[@]} == 0 )); then
    echo "Error: no Qwen ConvRot safetensors files were found in ${repo}@${revision}." >&2
    exit 1
fi

echo "Found ${#files[@]} ConvRot file(s); downloading complete Qwen set to ${destination}"
for remote_path in "${files[@]}"; do
    download_file "$repo" "$revision" "$remote_path"
done

required_files=(
    "Comfy-Org/HunyuanVideo_1.5_repackaged|main|split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
    "Comfy-Org/Qwen-Image_ComfyUI|main|split_files/vae/qwen_image_vae.safetensors"
    "lightx2v/Qwen-Image-Edit-2511-Lightning|main|Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"
    "vantagewithai/Qwen-Image-Edit-2511-GGUF|main|Qwen-Image-Edit-2511-Q5_K_M.gguf"
)

for spec in "${required_files[@]}"; do
    IFS='|' read -r source_repo source_revision remote_path <<< "$spec"
    download_file "$source_repo" "$source_revision" "$remote_path"
done

echo "Qwen model downloads complete."
