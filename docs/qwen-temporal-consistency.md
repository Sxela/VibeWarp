# Qwen temporal-consistency research

Status: contact-sheet infrastructure implemented experimentally; live ComfyUI
quality/VRAM validation remains to be run. Attention and latent-injection ideas
remain research plans.

This note records promising ways to improve frame-to-frame consistency when
VibeWarp uses Qwen-Image-Edit-2511. The first implementation target is joint
contact-sheet rendering. Attention and latent-injection experiments should
remain follow-up work until the contact-sheet baselines have been measured.

## Current VibeWarp behavior

The Qwen backend submits a separate ComfyUI graph for every frame. Each graph:

- loads up to three ordered reference images;
- uses the first reference to determine the editable latent and output size;
- uses a fixed seed by default;
- runs `TextEncodeQwenImageEditPlus`, the reference-latent method, and a normal
  `KSampler`; and
- returns one decoded image.

The outer render loop already computes an optical-flow-warped previous output
and a forward/backward consistency mask. That image can be selected as a Qwen
reference, but the default first reference is the raw current frame. No Qwen
transformer state or intermediate denoising latent currently survives between
Comfy submissions.

Relevant code:

- `vibewarp/core/qwen_comfy.py`: Comfy graph construction and submission
- `vibewarp/core/diffusion.py`: sequential frame loop, warping, reference
  resolution, output persistence, and resume behavior
- `vibewarp/core/edit_references.py`: ordered raw/previous/warped/fixed inputs
- `vibewarp/config.py`: `QwenEditConfig`

## Recommended order

1. Implement and measure three-frame contact sheets.
2. Compare raw-only sheets with stylized-boundary reinjection patterns.
3. Expose lower-risk controls such as Qwen denoise and stable sheet scaling.
4. Prototype Qwen-specific K/V injection using a warped stylized source.
5. Only then investigate saved denoising latents or motion-aligned noise.
6. Evaluate the full Qwen-Video-Edit backend separately; it is not a small
   extension of the existing per-frame Comfy backend.

## Milestone 1: three-frame contact sheets

### Why this is first

The Qwen-Video-Edit authors found that an unmodified Qwen image editor can edit
multiple video frames arranged as a single image grid with useful cross-frame
consistency. Joint attention sees all frames during the same denoising run, so
this is materially different from sending adjacent frames as separate image
references. Their zero-training demonstrations use both pixel-space contact
sheets and independently encoded latent tiles with grid positional encoding.

Sources:

- Paper: <https://arxiv.org/abs/2608.14790>
- Reference implementation: <https://github.com/yunpeng1998/Qwen-Video-Edit>

The first prototype should use the existing Qwen-Image-Edit-2511 Comfy graph
and one pixel-space sheet. It should not initially depend on the Qwen-Video-Edit
checkpoint, Wan VAE, or its training code.

### Experiment A: vanilla raw three-frame sheet

For source frames `0, 1, 2`:

```text
input sheet A:   [ raw 0 | raw 1 | raw 2 ]
Qwen output:     [ out 0 | out 1 | out 2 ]
saved frames:      out 0   out 1   out 2
```

Then process the next independent group:

```text
input sheet B:   [ raw 3 | raw 4 | raw 5 ]
Qwen output:     [ out 3 | out 4 | out 5 ]
```

This is the control condition. It tests within-sheet consistency and makes the
chunk-boundary discontinuity visible. It must use the same seed, prompt, tile
dimensions, gutters, and layout as the reinjection experiments.

Initial prompt direction:

> Treat the three panels as consecutive frames from one video. Apply the same
> visual transformation to every panel. Keep character identity, materials,
> palette, lighting, and fine details consistent while preserving the motion
> and composition shown in each panel.

### Experiment B: reinject two previous stylized frames

After rendering raw frames `0, 1, 2`, use the last two stylized results as
chronological references for one new raw target:

```text
input sheet A:   [ raw 0 | raw 1 | raw 2 ]
Qwen output A:   [ out 0 | out 1 | out 2 ]

input sheet B:   [ out 1 | out 2 | raw 3 ]
Qwen output B:   [ echo 1| echo 2| out 3 ]
saved from B:                       out 3
```

The first two tiles are temporal style/identity anchors and are not saved over
the already accepted frames. Their generated echo tiles remain in debug output
to measure whether Qwen silently restyles the anchors.

Initial prompt direction:

> Panels 1 and 2 are already-stylized consecutive preceding frames and are
> temporal references only. Use their change to understand the ongoing motion.
> Panel 3 is the different raw next frame and is the only panel to transform.
> Transfer persistent appearance, but preserve panel 3's pose, camera,
> geometry, and composition. Never paste either reference into panel 3.

This mode advances one new frame per Qwen call after the initial three. The
next sheet is `[out 2 | out 3 | raw 4]`.

### Reinjection subvariants to test

The wording and target-panel policy may matter as much as the sheet layout.
Keep these as named experiment variants rather than hiding them behind one
prompt:

1. **One anchor, two targets** (tested; collapsed): `[out N | raw N+1 |
   raw N+2]`. In the first live Qwen test the model repeated the anchor across
   every cell even after stronger positional prompting.
2. **Two anchors, one target** (current): `[out N-1 | out N | raw N+1]`; accept only the
   last output. This costs one Qwen call per new frame but gives the model two
   stylized temporal examples and makes "stylize the last panel using the
   previous panels" unambiguous.
3. **Warped anchor**: replace `out N` with the existing flow warp of `out N`
   toward raw `N+1`. This may improve local correspondence but may also teach
   the sheet to reproduce warp artifacts.
4. **Anchor plus raw duplicate**: include both `out N` and `raw N` before the
   new target. This tests whether paired before/after context communicates the
   intended edit more clearly than a stylized anchor alone.

The two-anchor/one-target mode is now the active `reinject` behavior. It trades
throughput for two examples of actual motion and a single unambiguous target.

### Layout and preprocessing matrix

The existing Comfy graph uses `FluxKontextImageScale`, whose preferred canvases
range only from roughly `0.43:1` to `2.33:1`. A horizontal strip of three
landscape video frames is much wider than that and would be center-cropped.
The initial implementation must therefore select a layout adaptively and build
the sheet directly at a preferred backend canvas size so the Comfy scaler is a
no-op:

| Variant | Purpose |
|---|---|
| Vertical `3 x 1` | Default for landscape frames |
| Horizontal `1 x 3` | Default for portrait frames |
| Row-major `2 x 2` with one unused cell | Default for near-square frames |
| Thin fixed gutters | Prevent content from bleeding across tile boundaries |
| No gutters | Maximize effective pixels per frame |
| Small panel labels/arrows | Test whether explicit ordering helps the VLM branch |

The layout helper should search the official preferred canvas list and a small
gutter range for equal-sized cells whose aspect ratio most closely matches the
source frames. Every tile must use the same transform. Record both the exact
input sheet and the image received back from Comfy. If Comfy changes the canvas
dimensions despite receiving a preferred size, fail clearly instead of
silently guessing new split boundaries.

The main limitation is pixel budget. Qwen's normal approximately one-megapixel
resize is shared by the whole sheet, so three full-width frames may become too
small. Test effective per-tile resolution explicitly rather than silently
upscaling low-resolution tiles back to the configured video size.

### Prompt experiments

For each sheet pattern compare:

1. The normal VibeWarp style/edit prompt with no temporal suffix.
2. A short temporal suffix (same video, consecutive frames, consistent style).
3. The explicit panel-role prompt shown above.
4. Target-only wording for reinjection (do not edit panels 1-2; edit panel 3).
5. A compact wording with `@Image`-style labels only if visible panel labels or
   the encoder actually establish those labels; do not assume tile positions
   receive native `@Image1` semantics inside a single bitmap.

Prompt changes must be isolated from layout changes in the test matrix.

The first live 2x2 reinjection test exposed a concrete failure mode: Qwen copied
the stylized anchor into every cell, including the unused gray cell. A stricter
layout-aware prompt still collapsed. The active reinjection pattern therefore
uses two distinct stylized anchors and one raw target, giving Qwen both observed
motion and an unambiguous single edit cell. The prompt still fixes every panel's
position, forbids copying either reference, and protects unused gray cells.

## Implementation plan for contact sheets

### 1. Configuration surface

Expose one shared experimental `contact_sheet` section for all ComfyUI
image-edit model families, with:

- mode: `off`, `raw_triplets`, or `reinject`;
- layout: `adaptive`, `row`, `column`, or `grid`;
- sheet size fixed to three for the first implementation;
- gutter width, initially four pixels where the selected canvas permits it;
- an optional role-instruction override (the grid-preservation guard remains);
  and
- debug-sheet retention.

The chunk planning, prompt roles, persistence, and UI are model-independent.
Canvas selection remains backend-aware: Qwen, Flux, and Mage start with the
roughly one-megapixel Kontext canvas list, while HiDream uses its approximately
four-megapixel trained presets. Qwen is the first live-quality benchmark, but
the shared path should remain callable for every Comfy edit backend so each can
be evaluated without another UI redesign.

### 2. Pure sheet helpers

Implement model-independent, unit-testable helpers that:

- assemble ordered equal-sized RGB tiles;
- return a manifest mapping tile positions to source frame numbers and roles;
- split an output sheet using the exact manifest geometry;
- reject unexpected output dimensions rather than cropping approximately; and
- save input/output sheets plus the manifest under the run's debug directory.

The manifest is important for resume behavior and for distinguishing accepted
frames from anchor/echo tiles.

### 3. Image-edit backend entry point

Submit the composite as one image through the existing Qwen render method and
split its single decoded result afterward. Reuse the existing model loading,
prompt graph, progress monitor, and Comfy upload/download handling. Do not
emulate a sheet by passing three native Qwen references: that produces a single
output latent and is a different test.

### 4. Chunk-aware orchestration

The current `run_frames()` path calls `_render_single_frame()` sequentially.
Add an edit-model chunk branch before that loop that:

- chooses raw and stylized inputs according to the selected pattern;
- submits one sheet;
- saves accepted tiles with normal absolute frame filenames;
- reports progress once for every accepted source frame;
- updates `state.prev_frame` to the last accepted output;
- applies post-color matching deliberately (per tile, not to the composite);
- handles a short final chunk without duplicating frames unless duplication is
  an explicit experiment; and
- leaves the existing per-frame path unchanged when temporal mode is off.

For the first implementation, contact-sheet mode owns the edit backend's first
image and rejects active optional reference slots. Supporting uploaded style
images in later slots is follow-up work; dynamic per-frame references are
ambiguous at chunk scope.

For reinjection, the saved prior output is the authoritative anchor. Never use
the regenerated echo tile as the next anchor unless a separate recursive-echo
experiment explicitly requests it.

### 5. Resume semantics

Resume must operate at accepted-frame boundaries, not merely sheet indices.
On resume:

- find the last completed normal output frame;
- reconstruct the next sheet from that persisted stylized output and raw source
  frames;
- never require a previous debug sheet to continue; and
- overwrite no completed output unless an explicit rerender range includes it.

Persisting the manifest makes incomplete Comfy results detectable and avoids
mistaking an anchor tile for a newly rendered frame.

### 6. Tests

Unit tests should cover:

- pixel-exact assemble/split round trips with and without gutters;
- manifest roles and absolute frame numbering;
- raw groups `0-2`, `3-5`, including a short final group;
- reinjection groups `raw 0-2`, then `out 1 + out 2 + raw 3`;
- rolling two-anchor/one-target advancement;
- discarding the echo tile while retaining its debug image;
- resume at the first and second accepted frame after a boundary;
- cancellation during a sheet submission;
- prompt selection when prompt schedules change inside a sheet; and
- output dimension mismatch from Comfy.

Prompt schedules need an explicit policy. The safe initial rule is to split a
sheet whenever the fully resolved positive or negative prompt changes, including
`{caption}` substitution, so all accepted targets in one call share the same
prompt. A later experiment can deliberately allow per-panel prompt text encoded
into the sheet prompt.

## Evaluation protocol

Use a short clip containing camera motion, articulated subject motion,
occlusion/disocclusion, and fine recurring details. Keep model, seed, sampler,
resolution, source frames, and style prompt fixed.

At minimum compare:

- current per-frame Qwen baseline;
- raw three-frame sheets;
- one-anchor/two-target reinjection;
- two-anchor/one-target reinjection; and
- the best sheet mode with Lightning disabled at the base sampling settings.

Measure separately:

- within-sheet temporal error;
- error at sheet boundaries;
- flow-warped LPIPS or perceptual difference in valid regions;
- identity/detail similarity using DINO or an appropriate face metric;
- raw-frame structural fidelity;
- anchor echo drift (`echo N` versus saved `out N`);
- runtime and peak VRAM; and
- effective output resolution/sharpness.

Always retain side-by-side and difference videos. A single whole-video score can
hide the exact boundary failure this feature is intended to fix.

## Follow-up research

### Qwen source-token K/V injection

AttnRouter's `KVInject` operates specifically on Qwen-Image-Edit-2511's MMDiT.
It alpha-blends the source-image K/V projections into corresponding noisy-output
K/V projections for selected transformer layers and early denoising steps. The
paper reports a useful alpha range around `0.3-0.5`, middle transformer blocks,
and early steps, but those values were not established for VibeWarp's 8-step
Lightning LoRA.

For VibeWarp, the most plausible source is the flow-warped,
consistency-composited previous stylized frame. This obtains temporal source
features during the current forward pass and is preferable to blindly reusing
stale K/V tensors from the previous Comfy request.

Source: <https://arxiv.org/abs/2605.01480>

### Flow-warped intermediate latents

LatentWarp saves the previous frame's intermediate denoising latents, warps them
with optical flow, masks invalid/occluded regions, and injects them into the
current denoising trajectory. This also constrains query features, which plain
cross-frame K/V sharing does not. VibeWarp has the necessary flow and
consistency maps but would need a custom Qwen sampler and storage policy for
per-step latents.

Source: <https://arxiv.org/abs/2311.00353>

### Motion-aligned starting noise

A fixed seed gives the same noise in screen coordinates, not object/motion
coordinates. Flow-warped Gaussian noise could carry texture decisions along
motion while filling disocclusions with fresh noise. This requires an explicit
noise input rather than the current `KSampler` seed interface and should be
treated as experimental because Qwen-Image-Edit was not trained for this use.

Sources:

- <https://arxiv.org/abs/2501.08331>
- <https://github.com/yitongdeng-projects/infinite_resolution_integral_noise_warping_code>

### Full Qwen-Video-Edit backend

The complete Qwen-Video-Edit system uses Wan 2.1 video-VAE latents, grid RoPE,
warm-started input/output projections, a trained Qwen checkpoint, and optional
Wan 2.2 denoising enhancement. It is the strongest directly relevant approach,
but its weights, runtime, resolution assumptions, and chunk-oriented inference
make it a separate backend project rather than the first contact-sheet patch.

Do not conflate ordinary diffusion feature caches such as EasyCache, TeaCache,
or first-block caches with temporal memory. They normally reuse work across
denoising steps for speed; they do not automatically align identities or
details across moving video frames.

## Open decisions after the first benchmark

- Whether two-anchor/one-target prompting avoids anchor duplication in live
  Qwen output.
- Whether the anchor should be the exact saved output or its flow-warped form.
- Whether contact sheets should cross scene cuts or force a fresh raw group.
- How to resolve prompt schedules that change more often than sheet boundaries.
- Whether panel labels/gutters help more than the pixels they consume.
- Whether loss of spatial resolution outweighs the temporal gain.
- Whether the best result justifies a custom KVInject node or a full
  Qwen-Video-Edit backend.

The frame-at-a-time temporal multi-reference LoRA experiment, including Ditto
dataset compilation and recursive validation, is specified in
[`qwen-temporal-lora-training.md`](qwen-temporal-lora-training.md).
