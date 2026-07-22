# CLI reference

All invocations use `python -m vibewarp`. CLI flags override any field loaded
from a settings or config file.

## Flags

| Flag | Default | Description |
|---|---|---|
| `--video` | | Input video path |
| `--checkpoint` | | SD checkpoint path |
| `--prompt` | `"a beautiful painting"` | Text prompt |
| `--negative-prompt` | `""` | Negative prompt |
| `--config` | | JSON config file |
| `--warpfusion-settings` | | WarpFusion settings .txt file |
| `--models-root` | | Root dir for model paths (auto-inferred from checkpoint path if omitted) |
| `--steps` | `25` | Diffusion steps |
| `--cfg-scale` | `7.5` | Classifier-free guidance scale |
| `--style-strength` | `0.65` | Style strength (0=keep init, 1=full regen) |
| `--seed` | `-1` | Random seed (-1 = random) |
| `--sampler` | `sample_euler_ancestral` | k-diffusion sampler |
| `--width` | `512` | Output width |
| `--height` | `512` | Output height |
| `--max-size` | `0` | Maximum source-video edge; preserves aspect ratio and rounds dimensions to multiples of 8 |
| `--nth-frame` | `1` | Extract every Nth video frame |
| `--start-frame` | `0` | Start frame |
| `--end-frame` | `0` | End frame (0 = all) |
| `--resume-from` | `0` | Resume from frame number |
| `--no-flow` | | Disable optical flow warping |
| `--output-dir` | `images_out` | Output directory |
| `--batch-name` | `warpfusion` | Batch name |
| `--print-config` | | Print resolved config as JSON and exit |
| `--download-models` | | Download checkpoint + ControlNets from HuggingFace, then exit |
| `--download-ckpt-only` | | Download checkpoint only |
| `--download-cn-only` | | Download ControlNets only (requires `--warpfusion-settings` or downloads all) |

## Usage examples

Plain CLI render (no settings file):

```bash
python -m vibewarp \
  --video input.mp4 \
  --checkpoint /models/checkpoints/model.safetensors \
  --prompt "a watercolor painting" \
  --steps 25 \
  --width 576 --height 1024
```

Load WarpFusion settings, then override resolution and frame range:

```bash
python -m vibewarp \
  --warpfusion-settings settings.txt \
  --video input.mp4 \
  --max-size 768 \
  --end-frame 30
```

## Model downloads

Download the checkpoint and all ControlNets referenced in a settings file:

```bash
python -m vibewarp --download-models --warpfusion-settings path/to/settings.txt
```

Checkpoint only / ControlNets only:

```bash
python -m vibewarp --download-ckpt-only
python -m vibewarp --download-cn-only --warpfusion-settings path/to/settings.txt
```

By default models are saved under a `models/` folder in the current working
directory (`models/checkpoints/` and `models/ControlNet/`) — no hardcoded
path. Override the location with `--models-root`:

```bash
python -m vibewarp --download-models --models-root /data/models
```

When rendering, model paths resolve in this order: an absolute path in the
settings file that exists on disk (used as-is), then `--models-root`, then
`./models` under the current directory, then the root inferred from the
settings file's checkpoint path. So a settings file exported on another
machine (e.g. pointing at `C:\models\...`) still works: if those files are
absent, VibeWarp finds the ones you downloaded into `./models`.

### Manual download

Any SD 1.5 checkpoint (`.safetensors` or `.ckpt`) works:

- [Realistic Vision](https://civitai.com/models/4201)
- [RevAnimated](https://civitai.com/models/7371)
- [Absolute Reality](https://huggingface.co/digiplay/AbsoluteReality_v1.8.1)
- [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)

ControlNet v1.1 models from https://huggingface.co/lllyasviel/ControlNet-v1-1:

| File | Use |
|---|---|
| `control_v11f1p_sd15_depth.pth` | Depth-based structure |
| `control_v11p_sd15_softedge.pth` | Soft edge detection |
| `control_v11p_sd15_canny.pth` | Canny edge detection |
| `control_v11f1e_sd15_tile.pth` | Tile/detail preservation |
| `control_v11p_sd15_inpaint.pth` | Inpainting guidance |
| `control_v11e_sd15_ip2p.pth` | InstructPix2Pix style |

## Python API

```python
from vibewarp.config import RunConfig, VideoConfig, DiffusionConfig
from vibewarp.pipeline import run

config = RunConfig(
    sd_checkpoint_path="path/to/model.safetensors",
    video=VideoConfig(video_init_path="input.mp4", width=576, height=1024),
    text_prompts={0: "a watercolor painting"},
    diffusion=DiffusionConfig(steps=25, cfg_scale=7.5),
)
output_frames = run(config)
```

## Web UI endpoints

`run-ui.bat` / `./run-ui.sh` (or `vibewarp-ui`) serves the Svelte app plus a small JSON API:

| Endpoint | Purpose |
|---|---|
| `GET /api/config` | `RunConfig` defaults + generated form schema |
| `POST /api/config/import` | Translate a `.json`/`.txt` settings file into a config |
| `GET /api/controlnet/catalog` | ControlNets available for a base model, and which checkpoints exist on disk |
| `GET /api/preview/runs` | Past runs under `<output_dir>/<batch_name>/`, newest first |
| `GET /api/preview/runs/{run}` | Layers in a run and the frames each one has |
| `GET /api/preview/runs/{run}/image` | One layer's image for a frame (`?layer=&frame=`) |
| `POST /api/jobs` | Submit a render |
| `GET /api/jobs/{id}/events` | Server-sent progress/log stream |
| `POST /api/jobs/{id}/cancel` | Cancel a running render |

`GET /api/controlnet/catalog` takes `model_version` (filters to `sd15`/`sd21`/`sdxl`)
and `model_dir` (scanned non-recursively for `.pth`/`.safetensors`/`.ckpt`/`.bin`).
Each net reports a `resolved_path`; `null` means its checkpoint is missing, which the
UI surfaces before the render starts rather than as a mid-run crash.

The `/api/preview/*` endpoints back the **Preview & Debug** tab, which puts a run's
layers side by side on a frame slider: the init video frame, the warped init, each
ControlNet's source image and detected map, the diffusion input, and the output.
Frame numbers in the API are always **render** frame numbers (0-based). Extracted
video frames are stored 1-based on disk (render frame N is `video_frames/{N+1}.jpg`),
and the API applies that offset for you — so `layer=init&frame=0` returns the init
image that actually produced output frame 0.

## Environment variables

| Variable | Effect |
|---|---|
| `VIBEWARP_DEBUG=1` | Restores full `transformers` verbosity. By default the noisy CLIP "LOAD REPORT" (a large table of `UNEXPECTED` vision-tower keys, harmless — the text encoder ignores the vision weights) is suppressed. |
| `VIBEWARP_DEBUG_DIFFUSION=1` | Logs the sigma schedule, CFG, and initial-noise values per frame — used for notebook parity debugging. |

## Output layout

Each run creates a numbered subdirectory under `<output-dir>/<batch-name>/`:

```
images_out/warpfusion/0/
  warpfusion(0)_000000.png   # rendered frames
  warpfusion(0)_000001.png
  ...
  warpfusion.mp4             # assembled video
  settings/                  # saved run config
  video_frames/              # extracted input frames
  flow/                      # cached optical flow + consistency maps
  debug/                     # CN maps, diffusion inputs, exact edit references
  _warped_NNNNNN.png         # per-frame warped init images (temp, used as CN source)
```
