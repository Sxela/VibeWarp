# Disco Diffusion and flow turbo

Select `disco_512` for the classic 512px unconditional ImageNet fine-tune or
`disco_256` for OpenAI's unconditional 256px model. This is pixel diffusion:
the model follows prompts through gradients from image/text encoders.

CLI example: `run.bat --model-version disco_512 --video input.mp4 --sampler ddim --steps 50 --turbo`.
For download-only preparation use `run.bat --download-models --model-version disco_512`,
or `run.bat --download-models --config settings.json` for precisely the configured assets.

Model selection shows the required model files and their local presence without
starting downloads. Click **Download required models** to prepare them, or
**Start render** to acquire missing weights as part of rendering. The monitor
shows file byte bars, speed, verification, and errors while transfers run.
Downloads resume from `.part` files and publish the final filename only after
completion. Selecting a model does not load it into GPU memory.
Disco checkpoint, secondary-denoiser and distilled-LoRA paths are under
**Project → Model**, directly below the model selector, like the SD checkpoint.
These paths now belong to project settings rather than the System tab.

## Models and guidance

[Optional quality LoRA](disco-quality-lora.md) adds a separately adjustable
aesthetic adapter, alone or stacked with the existing turbo/distilled adapter.

Weights live under `<model_path>/disco`, or `<root_dir>/models/disco` when the
model path is blank. Existing custom checkpoint and secondary paths take
precedence. The secondary and 512px weights have SHA256 checks; the historical
256px download currently has no pinned checksum. Original CLIP archives are
checked against the hash embedded in OpenAI's download URL.

The default guidance ensemble is ViT-B/32, ViT-B/16, and RN50. The encoder
checkboxes also offer ViT-L/14, RN101, and experimental SigLIP 2 base/large.
Modern encoders are loaded separately from the pinned OpenCLIP encoders used
by SD2/SDXL. Changing Disco guidance does not change those text encoders.

[SigLIP 2 setup and real-checkpoint validation](disco-siglip2.md) includes an
importable settings example. Base weights are installed in this checkout.
Text preprocessing explicitly uses lowercase and fixed-length padding.

The smaller `SecondaryDiffusionImageNet2` predicts a clean image from the noisy
sample. With it enabled, CLIP guidance differentiates through that network
instead of the large diffusion U-Net. Disable it to use the main denoiser for
guidance. Intermediate clean-image estimates appear in the render monitor;
`preview_interval=0` disables these previews.

The backend supports DDIM and PLMS, weighted text and image prompts, global
and inner cutouts, TV/range/saturation losses, LPIPS init guidance, and trusted
region inpainting. Each text prompt can contain one `text:weight` per line;
negative prompts supply repulsive targets. Image prompts use frame schedules
with local `path:weight` entries. Cutout schedule keys describe denoising
progress from 0 to 100, not video frame numbers.

The shared **Gradient guidance source** selector also works for Disco:
`raw frame` guides toward the current source frame, `prev stylized` toward
the previous output, `prev warped` toward its flow warp, and `prev warped + cc`
toward the prepared diffusion init (the consistency-composited warp, including
background masking). This default follows the v0.8 notebook's LPIPS target.
The other choices select an independent target. On the first frame, or after a scene reset, Disco falls back
to its prepared init if no previous target exists. Without flow, temporal
targets fall back to the previous stylized frame.

`disco.init_scale` controls perceptual LPIPS guidance and `disco.image_scale`
controls the existing spherical-distance loss applied directly to RGB tensors.
The latter is the pixel-space counterpart of the SD latent-distance guidance;
it does not load a VAE. The v0.8 notebook has LPIPS init guidance; this direct
RGB loss is a VibeWarp extension. Both scales accept zero to disable the loss.
By default both losses guide the **whole image**, matching the notebook's
unmasked LPIPS. Enable **Render → Disco Diffusion → Mask frame guidance to
consistent areas** (`disco.mask_frame_guidance=true`) to exclude inconsistent
pixels from LPIPS and image-space guidance. This uses the consistency mask's
existing blur and threshold controls and works independently of the inpainting
toggle. If no consistency mask is available, guidance covers the whole image.
Frame losses continue on every step
when `guidance_every` reduces CLIP evaluation frequency.

**Render → Disco Diffusion → Steps Schedule / Style Strength Schedule** use the
same frame scheduling controls as SD: constant values, per-frame lists, or
linearly interpolated frame keyframes. `disco.steps_schedule={0:30,20:50}` and
`disco.style_strength_schedule={0:0.65,20:0.8}` are examples. Null schedules
retain the legacy scalar values; the UI displays those scalar fallbacks.
Steps interpolate then truncate to an integer. Schedules respect the shared
range-relative keyframe setting and persist in saved settings. Changing steps
rebuilds the diffusion respacing without reloading weights. Turbo's optional
intermediate strength overrides the scheduled strength on intermediate frames.
Guided distilled sampling supports the same frame schedules, guidance and video
controls. Its adapter is evaluated on the original 1000-step noise schedule,
resampled to the requested count. The separate unconditional reference mode
(`disco.distilled_guided=false`) retains its fixed trained-grid controls.
See [guided distilled LoRA](disco-four-step-lora.md).

Step counts describe the full respaced sampler; strength controls the fraction
actually executed (30 steps at 0.65 executes 20). The render log records the
resolved steps, strength, executed count and seed for each diffused frame.
Progressive resolution schedules must reach full size inside that executed
budget, including when frame schedules reduce it.

Changes in flow consistency can change how much detail is regenerated: trusted
pixels are pinned to a re-noised init until the inpainting stop, while other
pixels are free to diffuse. A sudden loss of trusted area can therefore expose
more of the frame to CLIP-driven texture. Full-image LPIPS remains active in
both regions by default; the optional frame-guidance mask excludes inconsistent
regions when enabled. Changing frame
seeds and random cutouts can add variation. These mechanisms do not establish
the quality of a particular set of guidance scales without a render comparison.

An optional [noise-relative guidance update cap](disco-guidance-noise-cap.md)
limits the combined guidance displacement per step rather than just its raw
gradient. `guidance_noise_cap_percent=100` means one current-noise standard
deviation in image RMS; 0 disables it. The existing absolute gradient clamp
remains active. In the controlled example, 100% did not bind, while 1% did.

Guided Disco rounds width and height **up** to the next multiple of 64, including
source-aspect video sizes. DDIM, PLMS and guided distilled renders use the input
video's aspect with **Max size** as their longest-edge budget. A legacy
`max_size=0` now uses the saved width/height's longest edge as that budget,
rather than forcing the old square canvas. The UI preview uses the same rule.
The pixel grid can change the aspect after rounding: for example, 768×432
becomes 768×448. Only the standalone unconditional distilled reference requires
fixed 512×512. A small cutout count is useful when
checking settings; more cutouts improve the breadth of guidance at extra cost.

## Tiled and multiscale sampling

The shared **Tiled Sampler** controls now work for Disco as well as SD.
Enable `diffusion.tiled_sampler`, choose a target tile size (512 or 768 pixels
is a useful starting point), and set the overlap. Each denoiser evaluation
blends overlapping predictions into one full-frame field before DDIM/PLMS
updates the global sample. The secondary denoiser is tiled too. CLIP cutouts,
frame guidance, inpainting, and previews still operate on the assembled frame.

This mode follows the per-step approach described in
[Mago's WAN tiled sampler](https://github.com/MagoStudio/ComfyUI-MagoNodes/blob/main/wan_tiled_sampler.py),
adapted through VibeWarp's shared sampler wrapper rather than adding ComfyUI
as a runtime dependency. It changes attention context and normalization;
it is **not lossless operation chunking**, and it does not promise pixel parity.

For Disco, `diffusion.sampler_scale_schedule` now controls **persistent sampling
grids**. At 2048px with `{0:25, 10:50, 20:100}`, steps 0–9 sample at 512px,
10–19 at 1024px, and 20–29 at 2048px. The sample is resized only at those two
boundaries. Denoising, cutouts, CLIP gradients, and regularization run directly
on the current grid; there is no per-step upscale for guidance. Clean reference
images and masks are resized to that same grid. SD retains its existing
prediction-resizing behavior.

Cutout windows preserve their relative footprint. A 50%-wide window at a 50%
sampling scale is 25% of the final image width. CLIP's encoder input remains
its fixed trained size. At boundaries, clean predictions are interpolated;
noise keeps its coarse component and receives new fine-scale Gaussian detail.
PLMS history resets because old epsilon tensors belong to a different grid.
At full resolution, tiling takes over if enabled. Multiscale also works with
tiling disabled.

Keys count actual executed sampling steps after strength-based skipping.
Schedules must not decrease and must reach full resolution before the last
executed step. For the 30-step example above, use `disco.style_strength=1`;
weaker strength executes fewer steps and needs earlier transition keys.
The earlier prediction-only coarse implementation failed visual review. The
new persistent-grid mode is also experimental; removing the per-step resize
has not established a fix for saturation. Start with `{0:100}` for pure tiling.
See the [progressive-grid checks](disco-progressive-sampling.md).

Tile bounds and stage sizes align to 64 pixels. Learned variance is blended
separately from epsilon. Gradient-enabled tiled
predictions recompute tile activations on backward to reduce retained memory.
Overlap and grid alignment mean actual tiles can be larger than the requested
target. More overlap costs extra work, and the remaining global guidance and
current-resolution sample still consume memory.

Example settings fragment:

```json
{
  "model_version": "disco_512",
  "disco": {"steps": 30, "attention": "sdpa", "use_secondary_model": true},
  "diffusion": {
    "tiled_sampler": true,
    "sampler_tile_size": 768,
    "sampler_tile_overlap": 35,
    "sampler_scale_schedule": {"0": 100}
  }
}
```

Tiling and multiscale remain disabled by default for existing settings.
See the [2048px benchmark and visual review](disco-tiled-sampling-benchmark.md)
for memory savings, extra compute, and the failed aggressive multiscale example.

## Attention

For the optional unconditional distilled adapter and its exact four-step
sampling contract, see [four-step LoRA setup](disco-four-step-lora.md). This is
a separate sampler mode; it does not accelerate the CLIP/SigLIP-guided path.

`sdpa` is the default and lets PyTorch select a compatible fused kernel.
`manual` preserves the reference QKV calculation for comparisons. Legacy head
packing is preserved, and forward values and input gradients are tested
against it. Gradient checkpointing uses PyTorch's non-reentrant implementation,
so frozen parameters do not prevent image gradients.

Small-tensor parity does not imply identical full FP16 guided renders. The
[measured manual/SDPA comparison](disco-attention-benchmark.md) found no clear
speed or overall peak-memory advantage on the tested Windows build, and
visible detail differences across attention backends. That Torch build has
no compiled FlashAttention kernel.

The [1024/2048 measurements](disco-attention-large-sizes.md) show a substantial
attention-memory saving at 2048px. Full guided-frame speed must be measured
separately from the network's forward-pass speed.

`flash` optionally uses the separately installed FlashAttention package on
compatible CUDA tensors and otherwise falls back to SDPA. `sage` optionally
uses SageAttention for forward-only denoising. Attention requiring image
gradients uses SDPA under the Sage setting. No optional CUDA package is installed
automatically. Benchmark the complete guided step, including encoder backward,
rather than extrapolating from attention kernel benchmarks.

[SageAttention Windows installation and measured results](sage-attention-windows.md)
records the validated RTX 5090 environment, install commands, real-kernel
dispatch checks, and image comparisons. Sage was installed as an optional
backend; it did not improve the paired 512px benchmark's overall speed or peak.

## Turbo

Turbo is shared by sequential frame rendering. It fully diffuses preroll
frames, then diffuses anchors at the configured interval. Intermediate frames
advance two flow-warped histories and blend them, preserving separate feedback
and displayed images. Set `intermediate_strength` above zero for a light SD or
Disco diffusion pass on intermediate frames; zero means warp only.

Scene cuts detected by the thumbnail difference threshold reset the histories
and force a diffusion pass. `consistency_soften` controls the older turbo
history's minimum consistency weight. Turbo requires flow, and cannot be
combined with AnimateDiff or contact-sheet batching.

Lossless history checkpoints are saved under each run's `temp/turbo` directory.
Resume restores both histories from the previous completed frame, including
when that frame was an intermediate. Changing settings before resuming fails
clearly; start a new run to change settings. Retain this directory to preserve
both histories on resume. GPU interpolation/gradient reductions can still
produce small floating-point differences in newly diffused frames.

## Shared acquisition

Known SD1.5 default weights, selected ControlNets and IP-Adapters, their vision
encoders, and RAFT participate in preparation. Zero-weight models are skipped.
Custom paths remain authoritative. Third-party Hugging Face and Torch loaders
also report their downloads during rendering, including annotators and LPIPS.

The collapsible **Model downloads** queue covers explicit preparation and
Render. Each model has its own progress and Pause, Resume, and Cancel controls;
two managed models can transfer at once. Pausing a managed model preserves its
partial download and frees its slot for the next queued model. Other downloads
continue when one model is paused or cancelled. Render waits for paused required
models; cancelling a required model prevents that render from completing.
The **Required models** list can also be collapsed independently.

For a local ComfyUI server, set **System → Paths → Comfy models dir** to that
installation's `models` directory. Known FLUX.2, HiDream, Qwen, and Mage files
then download into their required subfolders. Gated downloads use the token
configured for Hugging Face. A remote server's filesystem cannot be inferred
from its HTTP URL; install weights on that server, or provide a mounted path.

## Guidance experiments and research

`lock_cutouts_across_frames` repeats crop geometry across frames to reduce one
source of temporal randomness. It is not flow tracking: a moving subject can
leave the fixed crop. `guidance_every` reduces encoder evaluations;
`balance_guidance` normalizes individual semantic gradient magnitudes before
combining them. These change the sampling trajectory and are opt-in.

Optional trusted-region frame guidance uses the consistency mask to avoid penalizing
newly revealed regions. Future experiments worth evaluating separately are
flow-transported crops, regional prompt losses, noise-dependent encoder
budgets, and dense feature guidance for structure.

SigLIP 2 is an implemented comparison, not a claim of better Disco aesthetics.
Retrieval and recognition improvements do not establish better gradients for
iterative image optimization. Compare fixed seeds, timing, VRAM, temporal
stability, and the desired artistic character.

Further encoder candidates researched:

- [SigLIP 2 paper](https://arxiv.org/abs/2502.14786): improved semantic,
  localization, and dense-feature capabilities.
- [Meta Perception Encoder](https://github.com/facebookresearch/perception_models/blob/main/apps/pe/README.md):
  PE Core B/L are potential image/video guidance encoders.
- [OpenCLIP](https://github.com/mlfoundations/open_clip): DFN ViT-L/14 is a
  conservative CLIP-family comparison; its integration must preserve the
  existing legacy dependency compatibility.
- [MobileCLIP2](https://machinelearning.apple.com/research/mobileclip2): a
  possible lower-cost cutout encoder; backward cost needs measurement.

## Reference and validation

[Non-LoRA 0.7 notebook parity audit](disco-notebook-parity-072.md) records exact
sampler/secondary/loss checks and the remaining settings, crop and importer
differences. It does not claim full guided-image parity.

The supplied `warpfusion_diffuse_inpaint_v0_7_2 (1).ipynb` was inspected before
implementation. The inference sampler/U-Net comes from
[guided-diffusion-sxela](https://github.com/Sxela/guided-diffusion-sxela) at
`97304f0be3cac2fd63060aba0bbcea5177ef2244`, including masked DDIM/PLMS.
Provenance and the MIT license accompany the vendored inference modules.

The secondary architecture comes from the supplied notebook. Notebook globals,
shared temporary filenames, setup cells, unsafe schedule evaluation, and
unrelated animation modes are not ported. Dango video cutouts now use the
notebook's zero padding, ResizeRight, grayscale prefix and full augmentation
stack. CPU cutout outputs/RNG and CUDA CLIP embeddings match the controlled
reference checks; exact whole-notebook pixel parity is not claimed.

Tests cover attention forward/backward parity, frozen-model checkpointing,
guided DDIM/PLMS, previews, download resume/integrity/deduplication, model
explicit download progress, settings round trips, and uninterrupted versus resumed turbo.

GPU checks on an RTX 5090 with PyTorch 2.11/CUDA 12.8 exercised the real 512px
checkpoint, OpenAI ViT-B/32, secondary and main-U-Net guidance, DDIM and PLMS,
512×512 rendering, RAFT, consistency inpainting, and a four-frame turbo video.
The secondary-guided 512×512 smoke run peaked at about 2.9 GiB allocated.
On real-run resume the skipped frame was identical; the following diffused
frame differed in 31 of 49,152 channels by at most one 8-bit level (MAE 0.000631).
The 256px checkpoint and optional external FlashAttention package were not GPU
benchmarked. Sage's separate measurements are linked above. SigLIP Base has
real-checkpoint CUDA gradient and 512px render checks; better guided-image
quality has not been established.
