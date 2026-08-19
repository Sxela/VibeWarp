# Qwen temporal multi-reference LoRA training

## Goal

Train Qwen-Image-Edit-2511 to edit one raw frame at a time while using the two
previous edited frames as temporal evidence. This remains an image-edit model:
there are no video latents, fixed clip length, or 360p checkpoint.

The native reference order is deliberately:

1. current raw frame (`Image 1`, also the starting latent);
2. edited frame `t-2` (`Image 2`);
3. edited frame `t-1` (`Image 3`); and
4. edited frame `t` as the training target.

AI Toolkit already supports Qwen 2511 and three matched control-image folders,
so the first experiment needs no custom trainer. It does need a dataset compiler
and, later, a two-frame output history in VibeWarp's ordinary multi-reference
frame loop.

## License warning

Ditto-1M is CC BY-NC-SA 4.0 and the Ditto repository describes the release as
academic/research use. Treat the resulting dataset and weights as non-commercial
unless legal review establishes a different permitted use.

## Remote VM setup

From a VibeWarp checkout on a Linux GPU VM:

```bash
WORKSPACE=/workspace/qwen-temporal \
DATASET_DIR=/workspace/qwen-temporal/dataset \
bash scripts/setup_qwen_temporal_training.sh
```

The script installs AI Toolkit in its own virtual environment, downloads
`Qwen/Qwen-Image-Edit-2511` and the matching 2511 uint3 accuracy-recovery
adapter, and writes an absolute-path training config into AI Toolkit's `config`
directory. `AI_TOOLKIT_REF` can pin a tested commit. `TORCH_INDEX_URL` can select
the CUDA wheel channel required by the VM image.

## Prepare Ditto

Download a selected Ditto subset rather than the full approximately 2.3 TB
release. Keep its `training_metadata`, source videos, and corresponding edited
videos under one dataset root, then run:

```bash
/workspace/qwen-temporal/ai-toolkit/venv/bin/python \
  tools/prepare_qwen_temporal_dataset.py \
  --metadata /datasets/Ditto-1M/training_metadata \
  --dataset-root /datasets/Ditto-1M \
  --output /workspace/qwen-temporal/dataset \
  --sample-every 4 \
  --max-edge 1024 \
  --resume
```

Useful smoke-test limits are `--max-videos 10 --max-samples 500`. Splits are
deterministic at the source-video level, so frames from one video cannot leak
between train and validation. The compiler writes a JSONL manifest recording
the original video pair and absolute target frame for every sample.

Validate the compiled data before renting a long training window:

```bash
/workspace/qwen-temporal/ai-toolkit/venv/bin/python \
  tools/validate_qwen_temporal_dataset.py \
  /workspace/qwen-temporal/dataset
```

## Train

The generated config uses batch size 1 plus gradient accumulation because
current Qwen 2511 multi-control training has reported failures above batch size
1. It disables AI Toolkit's independent sample generation; recursive validation
is more representative for this LoRA.

```bash
cd /workspace/qwen-temporal/ai-toolkit
venv/bin/python run.py config/qwen_temporal_ditto_v1.yaml
```

Start with 10k-50k samples from the global-style subsets. Split by video, not
frame. After the basic mapping works, add anchor degradation or model-generated
anchors to reduce teacher-forcing exposure bias.

## Validate the trained LoRA

Copy a checkpoint into ComfyUI's LoRA directory, start ComfyUI with the normal
Qwen 2511 model, and run both modes on a held-out Ditto pair:

```bash
python tools/validate_qwen_temporal_lora.py \
  --source-video source.mp4 \
  --target-video edited.mp4 \
  --instruction "Turn the video into an ink drawing." \
  --output validation/run-10000-recursive \
  --mode recursive \
  --unet qwen_image_edit_2511_int8_convrot.safetensors \
  --clip qwen_2.5_vl_7b_fp8_scaled.safetensors \
  --vae qwen_image_vae.safetensors \
  --lora qwen_temporal_ditto_v1_000010000.safetensors
```

Repeat with `--mode teacher_forced`. A large gap between teacher-forced and
recursive output indicates exposure bias: the LoRA works with clean Ditto
anchors but cannot recover from its own small errors. The script saves PNGs,
an MP4, per-frame reconstruction values, and a temporal-delta error. These
numeric values are diagnostics rather than perceptual-quality scores; visual
review of identity, details, motion preservation, and accumulated drift remains
required.

## First experiment matrix

- Base Qwen, current raw frame only.
- Base Qwen, raw frame plus two preceding generated frames.
- Temporal LoRA with ground-truth preceding frames.
- Temporal LoRA recursively using its own preceding outputs.
- Recursive temporal LoRA with mildly degraded anchors.

Keep the held-out video list, seed, instruction, model quantization, LoRA
strength, steps, and CFG fixed across checkpoints.
