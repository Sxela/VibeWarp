# Persistent coarse-to-fine Disco sampling

Implemented and tested on 2026-11-05. This replaces Disco's experimental
prediction-only scaling path for actual renders. SD's existing multiscale
behavior is unchanged, and pure 100%-resolution Disco tiling is unchanged.

## Sampling semantics

For a 2048px square output with 30 executed steps and
`sampler_scale_schedule={0:25, 10:50, 20:100}`:

| Executed steps | Persistent sample | Guidance grid |
|---|---|---|
| 0–9 | 512 x 512 | 512 x 512 |
| 10–19 | 1024 x 1024 | 1024 x 1024 |
| 20–29 | 2048 x 2048 | 2048 x 2048 |

There are two state transitions, not an upscale/downscale every step. Coarse
stages use whole-frame denoising. At the final size, optional overlapping
tiling applies to primary and secondary predictions. CLIP, regularization,
frame targets and masks use the current sampling grid. Guidance gradients are
never resized back to a separate full-size sample. Preview images cover the
current whole frame and may have its native pixel dimensions.

Cutout windows scale in pixels with the current grid. A window spanning 50%
of a 1024px coarse frame is 512px wide: 25% of the final 2048px width. CLIP's
encoder input stays at its trained dimensions. Minimum inner-window size is
scaled proportionally too; overview cutouts still cover the entire current
frame. Crop rounding can move boundaries by a pixel.

At each boundary, the clean estimate is interpolated once. Noise is refined
by retaining the coarse component and adding Gaussian detail orthogonal to
it, preserving unit marginal variance for unit Gaussian coarse noise. This
avoids averaging away variance or simply repeating identical noise blocks.
It is an approximate progressive sampling method, not pixel-equivalent to
full-resolution denoising. PLMS history resets at boundaries; the final-step
restart uses first-order sampling to avoid evaluating a negative timestep.

Schedules must grow monotonically and reach full size within the actually
executed steps. Strength-based skipping reduces that count. The example needs
`disco.style_strength=1`, or earlier transition keys when fewer steps execute.
Frame schedules `disco.steps_schedule` and `disco.style_strength_schedule`
also change this budget. Resolution keys remain executed **sampling steps**;
the two Disco schedules use **video frame** coordinates. Each frame resolves
its own sampler respacing and progressive plan without reloading model weights.
This also applies to **guided distilled** sampling: the LoRA remains active in
primary predictions and guidance backward, while the original 1000-step VP
coefficients are retained. The exact unconditional reference mode stays separate.

## Render checks and limits

RTX 5090, Torch 2.11.0+cu128, CUDA async allocator, SDPA, Disco 512 model,
secondary guidance, ViT-B/32, 30 DDIM steps, full regeneration, seed 42.
Tile target 768px, overlap 35%; final grid resolves to nine 896px tiles.
Eight CLIP cutouts per step; no flow, LPIPS, preview I/O, or turbo in timings.

The repeated run with 2 overview + 6 inner crops throughout completed warmup
and two timed frames. Median time was **47.696 s/frame**, peak allocated
**6.236 GiB**, reserved **8.719 GiB**. Both output images showed severe
saturation and poor structure. This confirms the revised data flow runs, but
does **not** establish that per-step interpolation caused the earlier failure
or that progressive sampling fixes quality.

A separate quality-only ablation used eight overview cutouts during the
coarse stages, then 2 overview + 6 inner cutouts at full size. Its denoising
progress schedules were `cut_overview={0:8,67:2}` and
`cut_innercut={0:0,67:6}`. It also showed severe saturation and lost structure.
Those historical schedules used the pre-v2 noise-timeline percentage axis;
new configurations use executed inference progress, with old settings migrated
to preserve their configured run's selected values.
This was one image without warmup, not a performance measurement. The early
inner-crop schedule is therefore not a sufficient explanation for this tested
failure either. Guidance strength, sampling budget and transitions remain
unisolated factors; no root-cause claim or quality preset is made.

- [Repeated progressive render results](../.cache/disco-progressive-2048/results.json)
- [Progressive output](../.cache/disco-progressive-2048/tiled_0.png)
- [Overview-only early-stage ablation](../.cache/disco-progressive-2048-overview/results.json)
- [Overview-only output](../.cache/disco-progressive-2048-overview/tiled_0.png)

Tests cover persistent per-step shapes, actual native-resolution guidance,
relative cutout footprints, integer/noninteger noise refinement, noise
statistics and inherited components, DDIM/PLMS, PLMS history resets, primary
versus secondary guidance, masks/previews, and invalid schedules. Existing
settings with `{0:100}` retain the prior full-size path.

The original [tiling benchmark](disco-tiled-sampling-benchmark.md) remains
historical evidence for the earlier prediction-resizing experiment and for
pure full-resolution tiling. Do not compare these separate sessions as an
isolated speed attribution study.

Guided runs now also offer an optional
[noise-relative update cap](disco-guidance-noise-cap.md). It uses each stage's
actual marginal noise level and bounds the final conditioning displacement,
including PLMS history. The controlled 512px check does not establish that it
fixes this progressive example's saturation.
