#!/usr/bin/env bash
#
# Download all regular/Plus IP-Adapter checkpoints supported by VibeWarp and
# both CLIP vision encoders. Completed files are skipped. Interrupted files
# remain as *.part and resume on the next run.
#
# Optional environment variables:
#   HF_TOKEN                    Hugging Face token
#   VIBEWARP_MODELS_DIR         Model root (default: <script>/models)
#   VIBEWARP_DOWNLOAD_DRY_RUN=1 Print targets without downloading

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
models_root="${VIBEWARP_MODELS_DIR:-${script_dir}/models}"
hf_base="https://huggingface.co/h94/IP-Adapter/resolve/main"

command -v curl >/dev/null 2>&1 || {
    echo "Error: curl is required." >&2
    exit 1
}

mkdir -p "${models_root}/controlnet/clip_vision"

auth_args=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    auth_args=(-H "Authorization: Bearer ${HF_TOKEN}")
fi

targets=(
    "models/ip-adapter_sd15.safetensors|controlnet/ip-adapter_sd15.safetensors"
    "models/ip-adapter_sd15_light.safetensors|controlnet/ip-adapter_sd15_light.safetensors"
    "models/ip-adapter-plus_sd15.safetensors|controlnet/ip-adapter-plus_sd15.safetensors"
    "models/ip-adapter-plus-face_sd15.safetensors|controlnet/ip-adapter-plus-face_sd15.safetensors"
    "models/ip-adapter-full-face_sd15.safetensors|controlnet/ip-adapter-full-face_sd15.safetensors"
    "models/ip-adapter_sd15_vit-G.safetensors|controlnet/ip-adapter_sd15_vit-G.safetensors"
    "sdxl_models/ip-adapter_sdxl.bin|controlnet/ip-adapter_sdxl.bin"
    "sdxl_models/ip-adapter_sdxl_vit-h.bin|controlnet/ip-adapter_sdxl_vit-h.bin"
    "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin|controlnet/ip-adapter-plus_sdxl_vit-h.bin"
    "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin|controlnet/ip-adapter-plus-face_sdxl_vit-h.bin"
    "models/image_encoder/model.safetensors|controlnet/clip_vision/clip_vision_vit_h.safetensors"
    "sdxl_models/image_encoder/model.safetensors|controlnet/clip_vision/clip_vision_vit_bigg.safetensors"
)

echo
echo "IP-Adapter destination:"
echo "  ${models_root}/controlnet"
echo "CLIP vision destination:"
echo "  ${models_root}/controlnet/clip_vision"
echo

for spec in "${targets[@]}"; do
    IFS='|' read -r remote_path relative_path <<< "$spec"
    target="${models_root}/${relative_path}"
    partial="${target}.part"
    filename="${target##*/}"
    download_url="${hf_base}/${remote_path}?download=true"

    if [[ -f "$target" ]]; then
        echo "Skipping existing ${filename}"
        continue
    fi

    if [[ -n "${VIBEWARP_DOWNLOAD_DRY_RUN:-}" ]]; then
        echo "Would download ${filename}"
        echo "  ${download_url}"
        continue
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
    echo "Saved ${target}"
done

echo
echo "IP-Adapter downloads complete."
echo "Set the UI CLIP vision path to:"
echo "  ${models_root}/controlnet/clip_vision"
