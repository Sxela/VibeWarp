#!/usr/bin/env bash
set -euo pipefail

# Remote Linux VM bootstrap for Qwen-Image-Edit-2511 temporal LoRA training.
# Override paths/settings with environment variables, for example:
#   WORKSPACE=/workspace/qwen-temporal \
#   DATASET_DIR=/workspace/datasets/ditto-qwen \
#   bash scripts/setup_qwen_temporal_training.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIBEWARP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE="${WORKSPACE:-/workspace/qwen-temporal}"
DATASET_DIR="${DATASET_DIR:-${WORKSPACE}/dataset}"
AI_TOOLKIT_DIR="${AI_TOOLKIT_DIR:-${WORKSPACE}/ai-toolkit}"
AI_TOOLKIT_REF="${AI_TOOLKIT_REF:-main}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE}/models/Qwen-Image-Edit-2511}"
ARA_DIR="${ARA_DIR:-${WORKSPACE}/models/accuracy-recovery-adapters}"
ARA_PATH="${ARA_DIR}/qwen_image_edit_2511_torchao_uint3.safetensors"
TRAINING_DIR="${TRAINING_DIR:-${WORKSPACE}/output}"
RUN_NAME="${RUN_NAME:-qwen_temporal_ditto_v1}"
CONFIG_PATH="${CONFIG_PATH:-${AI_TOOLKIT_DIR}/config/${RUN_NAME}.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v "${PYTHON_BIN}" >/dev/null || {
  echo "${PYTHON_BIN} is required" >&2; exit 1;
}

mkdir -p "${WORKSPACE}" "${WORKSPACE}/models" "${TRAINING_DIR}"

if [[ ! -d "${AI_TOOLKIT_DIR}/.git" ]]; then
  git clone https://github.com/ostris/ai-toolkit.git "${AI_TOOLKIT_DIR}"
fi
git -C "${AI_TOOLKIT_DIR}" checkout "${AI_TOOLKIT_REF}"
git -C "${AI_TOOLKIT_DIR}" submodule update --init --recursive

if [[ ! -x "${AI_TOOLKIT_DIR}/venv/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${AI_TOOLKIT_DIR}/venv"
fi
VENV_PYTHON="${AI_TOOLKIT_DIR}/venv/bin/python"
"${VENV_PYTHON}" -m pip install --upgrade pip wheel

# These are AI Toolkit's current documented Linux pins. Override the index for
# a VM image whose driver expects another CUDA wheel channel.
"${VENV_PYTHON}" -m pip install --no-cache-dir \
  torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
  --index-url "${TORCH_INDEX_URL}"
"${VENV_PYTHON}" -m pip install -r "${AI_TOOLKIT_DIR}/requirements.txt"
"${VENV_PYTHON}" -m pip install opencv-python-headless huggingface_hub

HF=("${AI_TOOLKIT_DIR}/venv/bin/hf")
if [[ ! -x "${HF[0]}" ]]; then
  HF=("${VENV_PYTHON}" -m huggingface_hub.commands.huggingface_cli)
fi

"${HF[@]}" download Qwen/Qwen-Image-Edit-2511 \
  --local-dir "${MODEL_DIR}"
"${HF[@]}" download ostris/accuracy_recovery_adapters \
  qwen_image_edit_2511_torchao_uint3.safetensors \
  --local-dir "${ARA_DIR}"

"${VENV_PYTHON}" "${VIBEWARP_ROOT}/tools/render_qwen_temporal_config.py" \
  --template "${VIBEWARP_ROOT}/configs/qwen_temporal_ai_toolkit.example.yaml" \
  --output "${CONFIG_PATH}" \
  --run-name "${RUN_NAME}" \
  --training-dir "${TRAINING_DIR}" \
  --dataset-dir "${DATASET_DIR}" \
  --model-dir "${MODEL_DIR}" \
  --ara-path "${ARA_PATH}"

cat <<EOF

Setup complete.

Prepare data:
  ${VENV_PYTHON} ${VIBEWARP_ROOT}/tools/prepare_qwen_temporal_dataset.py \\
    --metadata /path/to/Ditto-1M/training_metadata \\
    --dataset-root /path/to/Ditto-1M \\
    --output ${DATASET_DIR} --resume

Validate data:
  ${VENV_PYTHON} ${VIBEWARP_ROOT}/tools/validate_qwen_temporal_dataset.py ${DATASET_DIR}

Train:
  cd ${AI_TOOLKIT_DIR}
  ${VENV_PYTHON} run.py ${CONFIG_PATH}
EOF
