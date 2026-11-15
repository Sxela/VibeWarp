# Disco notebook video settings and trusted temporal guidance

Disco's CLIP/SigLIP gradient controls remain exclusive to the Disco backend.
Stable Diffusion continues using its normal text conditioning. No compilation
or legacy CC implementation is introduced by this integration.

## Disco controls

Selected CLIP/SigLIP guidance encoders load once per run and remain on the GPU
between frames. Per-frame allocator cache clearing does not unload their weights.

The UI places encoder selection, guidance scale, image prompts and secondary
guidance in **CLIP guidance**, alongside the init/image scale schedules.
Its separate **Cutouts** section precedes **Advanced guidance**; progress
schedules are directly editable. **Fixed seed**
is beside Seed in **Prompts & sampling**. Empty workflows/groups disappear for
the selected model and settings.

- Secondary-model precision is automatic: native BF16 on compatible CUDA devices, FP16 on other CUDA devices, and BF16 on CPU. Weights and inference use the selected half precision with differentiable autocast. Returned predictions and the existing guidance/loss calculations remain FP32. The hidden compatibility field `disco.secondary_precision` is always `auto`; older saved precision choices migrate to this policy.
- All classic CLIP and SigLIP guidance encoders load in FP16, including both SigLIP 2 sizes. There is no encoder precision selector or CPU precision fallback. Legacy `disco.clip_precision` values are migrated to `fp16`; the compatibility field is hidden. Normalized vision inputs use FP16, while returned embeddings, guidance losses, and sampler calculations retain their existing FP32 precision. Classic CLIP keeps its standard internal normalization precision.
- `disco.cut_batch_group`: positive number of cut batches concatenated per CLIP encoder call; default 1. The saved six-step example uses 2. Cutouts keep their original generation/RNG order, including a partial final group, and loss weights account for each group's share of the total budget. Different encoder batch sizes can change floating-point rounding.
- `disco.cut_max_size`: maximum longest side of the working image used to build
  CLIP/SigLIP cutouts, before square padding and cropping. Default 0 preserves
  notebook/native construction; try 1024 to reduce memory. Values must be 0 or
  integers of at least 32. Final encoder inputs retain the encoder's required
  resolution. Resizing is differentiable and preserves crop footprints relative
  to the original working image, but an enabled cap changes sampled image detail
  and can change the render. It does not reduce denoiser or LPIPS resolution.
  Overview construction is skipped entirely when the overview count is zero.
- `disco.pad_to_multiple_64`: resize content to `video.visible_width` / `visible_height`, then center it in an edge-padded /64 canvas. With visible dimensions zero, derive them from source aspect and `video.max_size`, or use explicit width/height if no size budget is set. This allows 1080x1920 content inside a 1088x1920 canvas without stretching.
- `video_assembly.crop_to_visible`: remove the canvas padding during export, after blending/masking and before upscaling. Saved render PNGs and blend feedback keep the full working canvas; cropped or upscaled export images never feed the next blend. Default false.
- Frame range is the only render limit. Notebook `max_frames` and retired
  `video.max_frames` values migrate into the range when no explicit range was
  selected. A one-frame profile uses `[0]`, preserving `[0,0]` as the full-range
  sentinel. Editing the range overrides the imported selection; no separate cap
  is stored or exported.

Raw extracted JPEGs are retained in `video_frames_raw` for padded runs. The separate padded cache uses the notebook's Lanczos resize, edge padding, and JPEG quality 95, with raw/output hashes to avoid applying the transform twice on resume. Geometry changes always start from the raw cache. Existing runs and unpadded modes retain their original behavior.

- `disco.clip_grad_steps`: sequential fixed-noise corrections before one DDIM advance; default 1.
- `disco.clip_divide_cuts`: distribute `cutn_batches` across passes instead of repeating the full budget; default false. Four passes and four batches means `[4,4,4,4]`; divided means `[1,1,1,1]`.
- Corrections are weighted by `1/passes`, or by each pass's fraction of the split budget. Four passes do not multiply the nominal guidance scale by four.
- `disco.timestep_grid`: optional descending original VP timesteps plus terminal -1. Its length must be `steps+1`. Example: `[699,582,466,349,233,116,-1]`. Skipping two executes 466, 349, 233, 116.
- `disco.use_secondary_model` and DDIM `eta` work on the guided quality/turbo LoRA path. The unconditional distilled reference remains restricted to its original trained setup.
- `disco.init_scale_schedule` and `image_scale_schedule` distinguish the first frame from subsequent temporal frames.
- `disco.cut_overview`, `cut_innercut`, and `cut_icgray_schedule` use **Inference
  progress (%)**: `100 * step_index / max(executed_steps - 1, 1)`. Progress starts
  at 0 after skipped steps, reaches 100 on the last executed step, and remains 0
  for a single-step run. Repeated guidance passes share a step's progress.
  Values hold until the next key, without interpolation or additional sampling
  steps. Overview/innercut values are integer cut counts per batch; grayscale
  values are probabilities in [0,1]. Null grayscale uses `cut_icgray_p`.
- `disco.cut_schedule_version=2` records this representation. Old notebook and
  native noise-timeline schedules migrate once by evaluating the configured
  first frame's actual grid and skip count, then placing its selected values at
  executed-step percentages. Fractional boundaries are supported and retained
  through JSON/UI round trips. Later changes to steps/strength use the new axis.
  The original-timestep mode is no longer a user control or an exported setting.
- `disco.mask_frame_guidance` uses soft trust for spatial LPIPS separately from the binary sampler inpaint mask. `inpaint_mask_sigma` makes blur semantics explicit.
- `disco.notebook_rng` preserves the notebook's shared noise/cutout RNG ordering.
- `disco.merge_loras` optionally merges fixed strengths into FP32 base weights. Reload after changing strengths. No compilation is involved.
- The noise-relative cap covers the complete repeated correction. With repeated passes and a cap enabled, a baseline denoiser evaluation is needed; cap-off sampling has no extra baseline evaluation.

Classic CLIP and SigLIP use the same controls; their existing encoders and preprocessing remain backend-specific. PLMS repeated corrections are not supported and are rejected explicitly.

## Loading a notebook export

The importer detects Disco by `diffusion_model` or the guided-LoRA marker. It reads render settings from the export, not similarly named training hyperparameters in the adapter metadata. It maps paths, strengths, the inference grid, cutout schedules, scales, eta, secondary, brightness and frame turbo. Cutout expressions support only bounded numeric list literals, repetition and concatenation, never executable Python.

Both original lowercase fields and v0.9.1 uppercase controls are accepted, including `CLIP_GRAD_STEPS`, `CLIP_DIVIDE_CUTS`, `SPEED_MERGE_LORAS`, `SPEED_SECONDARY_PRECISION`, `SPEED_CUT_BATCH_GROUP`, `PAD_TO_MULTIPLE_64`, `FRAME_LOSS_CC_TRUSTED_ONLY`, adapter enable/path/strength controls, and export crop/audio controls. Explicit lowercase values take precedence; false/zero values are preserved. `_effective_grid` takes precedence over adapter training metadata. `_visible_size` supplies the padded content dimensions.

The 1920px quality-20k 100% / turbo-4200 50% preset imports as six scheduled steps, skip two (original timesteps 665, 499, 332, 165), four sequential CLIP passes, four cut batches per pass, groups of two, merged LoRAs, secondary auto precision, eta 0.55 and clamp 0.04. This closes the known configuration/execution gaps; it does not establish bit-exact equality between OpenAI CLIP and OpenCLIP or between different hardware/library versions.

Old exports containing `"inpainting_stop": "inpainting_stop"` are ambiguous: import emits a warning and uses 1. Save a new native settings file to record the evaluated value. An export without an approved first-frame path cannot reconstruct that unstated asset: provide it using `video.first_frame_path`.

Native settings save the enabled LoRA file hashes. Mismatches are rejected. This permits asset relocation while detecting substituted weights. Frame-zero references must match the output dimensions exactly.

## Trusted temporal guidance in SD/SDXL

Enable `diffusion.temporal_guidance_cc_trusted_only` to restrict existing pixel (`init_scale`) and latent (`init_latent_scale`) temporal guidance. Default false preserves prior behavior.

The detached soft CC reliability map is current-frame aligned and captured before any minimum-trust floor used for warp blending. It is kept separate from binary sampler inpainting masks, and turbo history warps cannot overwrite it.

- Pixel loss uses spatial LPIPS. Untrusted predicted pixels remain perceptual context with gradients detached, and the loss map is weighted by trust using full-image normalization.
- Latent loss retains the historical width-vector spherical-distance definition, using trust-weighted vectors and row-coverage weighting.
- The final temporal latent correction is trust-gated before the existing full-tensor RMS clamp, accounting for spatial gradient spread through the VAE decoder.
- No trusted coverage means zero temporal correction. Full trust takes the original loss path. Missing CC preserves unmasked init guidance; first frames do not invent a temporal mask.
- Use warped guidance targets. `prev stylized` is unwarped and is rejected with this flag because the CC map is in current-frame coordinates.

The mask affects temporal guidance only, not the denoiser's text conditioning. It does not renormalize by trusted area or amplify small trusted regions.

## Shared video behavior

Current VibeWarp CC is retained; the unused legacy-CC compatibility field is hidden. No legacy algorithm is copied.

Frame turbo blends diffused frames after frame zero by `1/interval`, including preroll, while retaining the unblended model output as feedback. Existing lossless turbo checkpoints preserve resume histories.

`video.first_frame_path` is a reproducible approved anchor through the normal pipeline, without an external harness copying output files. `video.first_frame_sha256` records and verifies its identity.

`video_assembly.fps=0` derives source FPS divided by `extract_nth_frame`; an explicit positive value overrides it. If the system ffmpeg lacks libx264, the already-installed imageio-ffmpeg build is used for encoding. Video export behavior remains separate from saved render frames.

## Validation

`tests/test_video_guidance_scope.py` covers real settings round-trips, data-only schedules, grid/skip correctness, repeated correction ordering and cut budgets, cap/RNG behavior, masked gradient support, full/empty masks, fixed LoRA merge outputs and input gradients, turbo preroll feedback, approved first frames and output FPS.

Real-model validation artifacts and effective settings are stored separately in the discodiffusion-warp workspace under `dist/vibewarp-scope-validation`; source changes alone do not establish exact full-video pixel parity.
