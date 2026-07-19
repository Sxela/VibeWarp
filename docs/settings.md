# Settings reference

Details on WarpFusion settings mapping and every configurable subsystem.
For CLI flags and usage, see [cli.md](cli.md).

## WarpFusion settings mapping

When loading a WarpFusion settings file (`--warpfusion-settings`), the
following fields are mapped automatically. CLI flags override any field from
the settings file.

**Key mappings:**

| Settings field | Config field | Notes |
|---|---|---|
| `model_path` | `sd_checkpoint_path` | Resolved relative to `models_root` |
| `model_version` | `model_version` | `control_multi` → `control_multi_v15` |
| `noise_mode` | `diffusion.noise_mode` + `reconstruction_noise.enabled` | `'default'` = randn, `'fixed'` = persistent start_code, `'reconstructed'` = DDIM inversion |
| `code_randomness` | `diffusion.code_randomness` | Blend ratio for `noise_mode='fixed'` (0 = fully fixed, 1 = fully random) |
| `guidance_use_start_code` | `diffusion.guidance_use_start_code` | Kept for RNG parity with the notebook (burns one CUDA draw per frame) |
| `steps_schedule` | `diffusion.steps` + `steps_schedule` | Dict `{frame: steps}` |
| `cfg_scale_schedule` | `diffusion.cfg_scale` + `cfg_scale_schedule` | Supports `[[v1, v2, v3]]` ramp format |
| `style_strength_schedule` | `diffusion.style_strength` | List or dict schedule |
| `flow_blend_schedule` | `warp.flow_blend_schedule` | Resolved per-frame during render |
| `controlnet_multimodel` | `controlnet.models` | JSON string → dict of CN configs |
| `cond_image_src` | `controlnet.cond_image_src` | Global CN source: `init` or `stylized` |
| `detect_resolution` | per-CN `detect_resolution` | Global fallback for per-CN `-1`/`'global'` values |
| `dynamic_thresh` | `diffusion.dynamic_thresh` | Static thresholding clamp (default 30) |
| `lora_dir` | `lora_dir` | Directory containing LORA `.safetensors` files |
| `controlnet_model_dir` | `controlnet.model_dir` | Base directory for ControlNet checkpoints; relative or omitted per-model paths resolve here |
| `soften_consistency_mask` | `flow.soften_consistency_mask` | Floor clip for CC weights (default 0) |
| `use_tiled_vae` | `vae.use_tiled_vae` | Enable tiled VAE for high-res renders |
| `num_tiles` | `vae.num_tiles` | `[rows, cols]` tile grid |
| `tile_size` | `vae.tile_size` | Tile edge in latent space (default 128) |
| `vae_stride` | `vae.vae_stride` | Stride in latent space (default 96) |
| `vae_padding` | `vae.vae_padding` | Fractional overlap `[h_frac, w_frac]` |
| `keep_audio` | `video_assembly.keep_audio` | Reattach audio from source video |
| `use_deflicker` | `video_assembly.use_deflicker` | ffmpeg deflicker filter |
| `upscale_ratio` | `video_assembly.upscale_ratio` | RealESRGAN upscale factor (1=off) |
| `upscale_model` | `video_assembly.upscale_model` | RealESRGAN model name |
| `upscale_model_path` | `video_assembly.upscale_model_path` | Path to .pth weights |
| `use_background_mask_video` | `video_assembly.use_background_mask_video` | Composite onto background |
| `invert_mask_video` | `video_assembly.invert_mask_video` | Invert alpha mask |
| `background_video` | `video_assembly.background_video` | Background type |
| `background_source_video` | `video_assembly.background_source_video` | Color or image path |
| `mask_clip_low` | `video_assembly.mask_clip_low` | Alpha mask floor threshold |
| `mask_clip_high` | `video_assembly.mask_clip_high` | Alpha mask ceiling threshold |
| `missed_consistency_schedule` | `flow.missed_consistency_schedule` | Per-frame schedule for missed CC weight |
| `overshoot_consistency_schedule` | `flow.overshoot_consistency_schedule` | Per-frame schedule for overshoot CC weight |
| `edges_consistency_schedule` | `flow.edges_consistency_schedule` | Per-frame schedule for edges CC weight |
| `consistency_blur_schedule` | `flow.consistency_blur_schedule` | Per-frame schedule for blur sigma |
| `consistency_dilate_schedule` | `flow.consistency_dilate_schedule` | Per-frame schedule for dilation radius |
| `soften_consistency_schedule` | `flow.soften_consistency_schedule` | Per-frame schedule for soften value |

**ControlNet source selection:**

Each CN in `controlnet_multimodel` has a `source` field:
- `'stylized'` → annotates the warped previous rendered frame (temporal coherence)
- `'init'` → annotates the raw video frame (structural fidelity)
- `'global'` → uses the global `cond_image_src` setting

On frame 0, `'stylized'` falls back to the raw video frame since no previous render exists.

The `preprocess` field maps to annotator type: `'PIDI'` → `softedge`, `'dw_pose'` → `dwpose`, `'depth_anything'` → `depth`, `'global'` → auto-detect from model key.

**Schedule formats supported:**

- `[value]` — constant for all frames
- `[v0, v1, v2, ...]` — one value per frame (indexed)
- `{frame: value}` — keyframe dict (with linear interpolation when `blend=True`)
- `[[v0, v1, v2]]` — WarpFusion ramp format: interpolate across diffusion steps (CFG only)

## ControlNet modes and layer weights

Each ControlNet entry has a `mode`. It is the user-facing knob; `layer_weights`
(13 values, one per UNet block) and `zero_uncond` are **derived from it**, not
set independently:

| `mode` | Layer weights | `zero_uncond` | Meaning |
|---|---|---|---|
| `balanced` | `[1.0] × 13` | off | Default — every block weighted equally |
| `controlnet` | `0.825 ** (12 - i)` | on | "ControlNet is more important" |
| `prompt` | `0.825 ** (12 - i)` | off | "My prompt is more important" |
| `custom` | your `layer_weights` | your `zero_uncond` | Hand-authored, nothing derived |

The soft curve damps the early (low-index) blocks, which is what shifts the
balance between the conditioning and the prompt. `custom` is a VibeWarp addition
— the notebook has only the first three.

This derivation runs when models are loaded, so it applies equally to WarpFusion
settings files, VibeWarp JSON, the Python API, and the web UI.

## Available ControlNets

`vibewarp/controlnet_catalog.py` is the single source of truth for which nets
exist: their keys, labels, expected checkpoint filenames, download URLs, base
model (SD1.5 / SD2.1 / SDXL), annotator, and which global detector settings apply
to each. The web UI, the settings importer, path inference, and
`--download-models` all read from it, so adding a net is a one-line change there.

Checkpoint paths resolve in this order: an explicit `path` on the entry, then the
net's expected filename inside `model_dir`. Nets whose upstream checkpoint has no
canonical name (several publish as `diffusion_pytorch_model.safetensors`) are
renamed on download so they cannot overwrite each other.

**SDXL ControlNets** ship in diffusers format, which shares no key names with the
SD1.5 (`cldm`) layout. VibeWarp converts them on load
(`core/controlnet_diffusers.py`) — no ComfyUI install is needed, unlike the original
notebook. The format is detected from the checkpoint itself, so an SDXL net works
whether you point at it explicitly or let `model_dir` resolve it. A ControlNet from
the wrong base model (an SD1.5 net on an SDXL checkpoint) is rejected up front
rather than failing with a shape mismatch mid-render.

## Consistency mask

When optical flow warping is enabled (`flow_warp=True`), a 3-channel
consistency map (CC map) is generated alongside the flow field. Each channel
marks a different type of warp unreliability:

| Channel | Layer | Meaning |
|---|---|---|
| R | `missed` | Areas the forward flow couldn't reach (occlusions, motion out-of-frame) |
| G | `overshoot` | Areas where backward flow disagrees (ambiguous matches) |
| B | `edges` | Flow discontinuities near moving object boundaries |

`edge_consistency_width` is a Sobel kernel size. OpenCV requires an odd value
between 1 and 31; imported values outside that range are safely normalized
(for example, `20` becomes `21`) instead of failing during flow generation.

The current non-legacy generator follows WarpFusion v0.37's RAFT behavior,
including its magnitude-dependent flow coloring for the edge channel, 20 update
iterations for backward flow and torchvision's default 12 for forward flow, and
its distinction between the clamped flow returned immediately and the unclamped
flow cached for resumed runs. The matching defaults are `flow_lq=True` and
`missed_consistency_dilation=2`.

**Post-processing pipeline** (matches notebook `load_cc()`):

1. Weight each channel: `clip(1 - weight, 1)` — weight=0 ignores that layer, weight=1 = full effect
2. Multiply all weighted channels together
3. Binary threshold at 0.5 → hard mask
4. Morphological dilation (`consistency_dilate`) — expands unreliable regions
5. Gaussian blur (`consistency_blur`) — softens mask edges
6. Clip the floor with `soften_consistency_mask` — raises the minimum weight so some stylized content bleeds through even in inconsistent regions

**All parameters are per-frame schedulable** using the `*_schedule` variants:

```python
from vibewarp.config import RunConfig, FlowConfig

config = RunConfig(
    flow=FlowConfig(
        # Static defaults
        missed_consistency_weight=1.0,   # 0 = ignore R channel
        overshoot_consistency_weight=1.0,
        edges_consistency_weight=1.0,
        consistency_blur=1,              # gaussian sigma
        consistency_dilate=3,            # dilation radius in pixels
        soften_consistency_mask=0.0,     # 0 = hard mask, 1 = fully ignore CC

        # Per-frame schedules (override static values when set)
        soften_consistency_schedule={0: 0.0, 30: 0.3},  # ramp up softening
        missed_consistency_schedule={0: 1.0, 60: 0.5},  # reduce missed effect
    )
)
```

`soften_consistency_mask` is equivalent to WarpFusion's `forward_weights_clip`.
Values above 0 allow the warped previous render to partially "bleed through"
in inconsistent regions, which can reduce flickering at the cost of some
structural fidelity.

The effective post-processed mask used for each blend is saved as
`debug/consistency_mask_NNNNNN.png` and appears in History under **Optical
flow**. White keeps the warped stylized frame; black falls back toward the raw
current frame. Frame 0 has no mask because it has no preceding frame to warp.

## Video assembly

Post-processing options applied during video creation (configured via `VideoAssemblyConfig`):

```python
from vibewarp.config import RunConfig, VideoAssemblyConfig

config = RunConfig(
    video_assembly=VideoAssemblyConfig(
        keep_audio=True,           # attach audio from source video (ffmpeg)
        use_deflicker=True,        # ffmpeg deflicker=mode=pm:size=10 filter
        upscale_ratio=4,           # RealESRGAN upscale (1 = disabled)
        upscale_model='realesr-animevideov3',
        upscale_model_path='/models/realesr-animevideov3.pth',
        use_background_mask_video=True,   # composite onto background
        background_video='color',         # 'color', 'image', or 'init_video'
        background_source_video='black',
        invert_mask_video=False,
    )
)
```

### RealESRGAN upscaling

Upscale each output frame before encoding. Requires the `realesrgan` and `basicsr` pip packages:

```bash
pip install realesrgan basicsr
```

Download weights manually from the GitHub releases for the model you want, then set `upscale_model_path`.

| Model | Scale | Notes |
|---|---|---|
| `realesr-animevideov3` | 4× | Best for animation/illustration |
| `realesr-general-x4v3` | 4× | General purpose |
| `RealESRGAN_x4plus` | 4× | Photorealistic content |
| `RealESRGAN_x4plus_anime_6B` | 4× | Anime, lightweight |
| `RealESRGAN_x2plus` | 2× | 2× upscale |

### Audio preservation

When `keep_audio=True` and a source video is set, the audio track is remuxed
into the output without re-encoding (`-c:v copy`). If the source has no audio
stream, the error is silently suppressed and the silent video is kept.

### Deflicker

Adds a `deflicker=mode=pm:size=10` ffmpeg filter to the encode pass. Useful
for reducing frame-to-frame brightness flicker in generated videos.

### Background mask compositing

When `use_background_mask_video=True`, each output frame is composited onto a
background using pre-computed alpha masks. The masks must be in
`{video_frames_folder}_alpha/` named `{frame_num+1:06d}.jpg` (grayscale,
0=transparent 255=opaque).

Background types:
- `'color'` — solid color (CSS color string or name, e.g. `'black'`, `'#ff0000'`)
- `'image'` — path to a background image
- `'init_video'` — raw video frame for the same index

`mask_clip_low` / `mask_clip_high` threshold the mask before compositing.

## Tiled sampler

The tiled sampler reduces UNet/ControlNet peak VRAM at high resolutions. It
splits the latent spatially on every denoiser evaluation, runs each overlapping
tile independently, then combines the predictions with normalized trapezoidal
blend windows before the sampler takes its next step.

```python
from vibewarp.config import DiffusionConfig, RunConfig

config = RunConfig(
    diffusion=DiffusionConfig(
        tiled_sampler=True,
        sampler_tile_size=512,
        sampler_tile_overlap=35,
        sampler_tile_split_threshold=20,
    )
)
```

These controls are in **System → Performance** in the web UI. VibeWarp derives
the row and column counts from the render dimensions and selected pixel-space
tile edge. The UI recommends 512px for SD1.5 and
AnimateDiff, or 1024px for vanilla SDXL, matching their training resolutions.

ControlNet hints and spatial image conditioning are cropped with every tile.
AnimateDiff is supported: tiling only splits height and width, never its temporal
frame batch. This is separate from tiled VAE, which only affects image
encoding/decoding.

| Field | Default | Description |
|---|---:|---|
| `tiled_sampler` | `False` | Enable per-evaluation spatial tiling |
| `sampler_tile_size` | `512` | Target tile edge in output-image pixels; grid is automatic |
| `sampler_tile_overlap` | `35.0` | Overlap between neighbouring tiles, as a percentage of the tile edge. This is the cross-fade that hides the seams. `0` = hard cuts (fastest, seams likely). |
| `sampler_tile_split_threshold` | `20.0` | Leftover a tile may absorb before another is added, as a percentage of the tile edge: `n = ceil(total/tile - threshold)`. Tiles are sized *from* the count, so without slack a 1080px side against a 1024px target splits into two ~640px tiles. `0` = the plain `ceil()` behaviour. Independent of the overlap. |

## Tiled VAE

For high-resolution renders where the full image doesn't fit in VRAM for VAE
encode/decode, enable tiled VAE:

```python
from vibewarp.config import RunConfig, VaeConfig

config = RunConfig(
    vae=VaeConfig(
        use_tiled_vae=True,
        tile_size=128,       # latent-space tile edge (128 → 1024px tiles)
        vae_stride=96,       # latent-space stride between tiles
        # OR: fractional overlap
        # vae_padding=[0.5, 0.5],  # 50% overlap
        # OR: explicit grid
        # num_tiles=[2, 2],   # 2×2 grid, tile size derived from image dims
    )
)
```

| Field | Default | Description |
|---|---|---|
| `use_tiled_vae` | `False` | Enable tiled VAE |
| `tile_size` | `128` | Tile edge in latent space (×8 = pixel size, so 128→1024px) |
| `vae_stride` | `96` | Stride between tiles in latent space |
| `vae_padding` | `None` | Fractional overlap `[h_frac, w_frac]` of tile_size (overrides `vae_stride`) |
| `num_tiles` | `None` | `[rows, cols]` grid — tile size derived from image dims (overrides `tile_size`) |

`tile_size` and `vae_stride` are in **latent space** (divide pixel dims by 8).
The default `tile_size=128` gives 1024px tiles with `vae_stride=96` (768px
stride, 256px overlap on each side).

## Noise modes

Three noise modes control how the initial noise is constructed for each frame:

| `noise_mode` | Behavior |
|---|---|
| `default` | Fresh `randn` each frame — maximum variation |
| `fixed` | Persistent `start_code` tensor mixed with per-frame randn via `code_randomness`. Lower randomness = more temporally coherent noise structure. |
| `reconstructed` | DDIM inversion of the init image — noise that "knows" the frame content. Best temporal consistency; slower (runs an extra inversion pass per frame). |

For `noise_mode='fixed'`, `code_randomness` controls the blend:
- `0.0` = fully fixed (same start_code every frame)
- `1.0` = fully random (same as `default`)
- Typical value: `0.1`–`0.3`

## Prompt features

### Multi-prompt blending

Separate prompts with `|` and optionally add a `:weight` suffix to each:

```
"impressionist painting:0.7 | anime style:0.3"
```

Weights are normalized automatically. Each sub-prompt is encoded
independently; the conditionings are blended before sampling.

### LORA scheduling

Embed LORA references directly in prompt keyframes using `<lora:name:weight>` syntax:

```python
config = RunConfig(
    lora_dir="/models/loras",
    text_prompts={
        0:  "a painting <lora:impressionist:0.8>",
        60: "a portrait <lora:impressionist:0.4> <lora:face_detail:0.6>",
    },
)
```

LORAs are loaded from `lora_dir` by filename stem (`.safetensors`, `.pt`, or
`.ckpt`). Multiple LORAs stack additively. Weights are resolved per-frame from
the keyframe schedule; a weight of 0 skips loading that LORA. Supports
A1111-format LORA files.

LORA tags are stripped from the prompt before text encoding — the clean
prompt is what gets encoded.

**Merge precision** (`lora_merge_precision`, default `'fp16'`):

- `'fp16'` — replicates the WarpFusion notebook's A1111 merge arithmetic
  exactly (up/down cast to the weight dtype before the matmul). Required for
  notebook output parity.
- `'fp32'` — computes deltas in float32 and rounds once at apply time. More
  accurate merged weights, but renders deviate slightly from the notebook.

```python
config = RunConfig(lora_dir="/models/loras", lora_merge_precision="fp32")
```

## Vendored libraries

VibeWarp vendors everything it needs in `vibewarp/vendor/` — no external git
clones required.

| Vendor | Source | Used for |
|---|---|---|
| `k_diffusion` | [Sxela/k-diffusion](https://github.com/Sxela/k-diffusion) (fork of crowsonkb/k-diffusion) | Samplers and sigma schedules (+ `sample_lcm` from the ComfyUI fork, WarpFusion callback semantics) |
| `ldm` | [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion) | SD model architecture (pytorch-lightning stripped) |
| `cldm` | [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet) | ControlNet model loading |
| `sgm` | [Sxela/generative-models](https://github.com/Sxela/generative-models) (fork of Stability-AI/generative-models) | SDXL model stack (DiffusionEngine, conditioner, VAE — full inference subset, pytorch-lightning stripped) |
| `resize_right` | [assafshocher/ResizeRight](https://github.com/assafshocher/ResizeRight) | Antialiased image resizing |
| `python_color_transfer` | [pengbo-learn/python-color-transfer](https://github.com/pengbo-learn/python-color-transfer) | Color matching between frames |
| `controlmodel_ipadapter` | [Sxela/sxela-stablediffusion](https://github.com/Sxela/sxela-stablediffusion) | IP-Adapter UNet cross-attention hooks (+ bundled uncond CLIP embeds) |
| `animatediff_mm` | [ArtVentureX/comfyui-animatediff](https://github.com/ArtVentureX/comfyui-animatediff) + notebook extensions | AnimateDiff motion modules (SD1.5 v1/v2/v3, SDXL/HotshotXL), UNet injection, beta-schedule registration |
