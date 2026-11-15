# Non-LoRA notebook parity audit

Checked on 2026-11-06 against the approved local
`C:\code\my\discodiffusion-warp\warpfusion_diffuse_inpaint_v0_7_2 (1).ipynb`.
Its saved batch name contains `0.7.0`. No LoRA was loaded or evaluated.

**Result: component parity passes; settings and full guided-image parity do not.**
The original audit left render behavior unchanged. The cutout follow-up below
was implemented on 2026-11-06; other findings remain as documented.

## Cutout follow-up: resolved

`make_cutouts` now uses the existing vendored ResizeRight implementation, zero
padding, the inclusive grayscale prefix and the full notebook video augmentation
stack. The notebook's symmetric floor padding for odd dimensions is preserved.
Progressive sampling still scales the minimum crop size with the current grid;
explicit cutout generators isolate the augmentation RNG from the sampler.

Against the actual notebook definitions:

- **8/8 CPU cases:** bit-exact cutout values and identical CPU RNG consumption.
  Includes square, landscape, portrait and odd dimensions; 0/1 grayscale
  probabilities; overview counts 0–16; inner crops; augmentations on/off.
- Squared-cutout diagnostic image gradients: maximum difference **2.91e-11**.
- Real **ViT-B/32 on CUDA**, using one shared encoder to isolate cutout behavior:
  cutouts and embeddings are bit-exact with augmentations off and on. CLIP image
  gradients are exact without augmentation and differ by at most **1.42e-10**
  with augmentation; gradient cosine is 1 in both cases.

[Follow-up results](../.cache/disco-cutout-parity-072/results.json).
This resolves the cutout implementation finding. It does not establish complete
notebook pipeline parity or separately compare OpenAI CLIP and OpenCLIP loaders.
Reproduce with the command below, adding
`--clip-root models/disco --output .cache/disco-cutout-parity-072`.

## Measured checks

The reference is the clean local `guided-diffusion-sxela` checkout at
`97304f0be3cac2fd63060aba0bbcea5177ef2244`. Selected notebook definitions are
extracted with AST; installation cells, shell commands and the notebook's global
render loop are not executed. Original source hashes are recorded in the report.

| Check | Result | Scope |
|---|---|---|
| DDIM/PLMS sampling | **24/24 bit-exact**, max absolute error 0 | Every sample and clean prediction; 30/50 steps; skip 10 or 70%; masks on/off; DDIM eta 0/0.25; PLMS. Same small synthetic denoiser, explicit gradient, init, noise and RNG seed. |
| Respacing and betas | **Exact** | Same original timestep maps and respaced beta arrays in those cases. |
| Spherical, TV and range loss functions | **Exact outputs and image gradients** | Directly extracted notebook functions versus application functions. This does not establish identical integration of the losses. |
| Secondary denoiser | **Exact outputs and input gradients** | Notebook architecture versus application architecture at 64×64, both loaded strictly from the real secondary checkpoint. SHA256 `983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a`. |
| Dango cutouts, original audit | **Initially not equivalent; now resolved above** | Original four-case measurements remain preserved in the initial report. |
| Legacy settings import | **Fails Disco identification** | A representative notebook-export subset imports as `control_multi_v15`, with SD steps 25 and no Disco section. |

[Machine-readable results](../.cache/disco-notebook-parity-072/results.json).
These are CPU component checks. There is **no full-video, primary-checkpoint,
OpenAI-CLIP-versus-OpenCLIP, GPU image or visual-quality parity claim**.

## Settings differences

These are the literal settings saved inside the supplied notebook, compared
with current application defaults. Existing runs may override application defaults.

| Setting | Supplied notebook | Application default / mapping |
|---|---|---|
| Model / sampler / secondary | Crowson 512 / DDIM / enabled | Same when selecting Disco 512 + DDIM. |
| CLIP ensemble | ViT-B/32, ViT-B/16, RN50 | Same selected ensemble; encoder implementation parity was not measured here. |
| Steps | 30 | 50; set `disco.steps_schedule={0:30}` to match. |
| First-frame skip | 10 of 30 → 20 executed | Default strength 0.65 also executes 20 at 30 steps. |
| Later-frame skip | 70% → 21 skipped, **9 executed** | Default strength 0.65 still executes 20. A frame strength schedule `[2/3,0.3]` expresses the notebook's first/later budgets. |
| CLIP scale | **35000** | **5000**. |
| Cutout batches | **4** | **1**. |
| Overview / inner counts | 16/0 → 12/4 → 8/8 → 4/12 at 40%, 60%, 80% | 4/12 → 2/14 → 0/16 at 40%, 60%. |
| Inner grayscale control | 0.2 for first 10%, then 0 | Scalar 0.2, with different crop-selection semantics described below. |
| Eta | **0.25** | **0**. |
| Gradient RMS clamp | **0.25** | **0.05**. |
| Range scale | **0** | **150**. |
| LPIPS | `init_scale=0`; `frames_scale=15000` has a wiring problem | Default `disco.init_scale=1000`. See below before mapping `frames_scale`. |
| Flow blend | **0.999** | **0.5**. |
| Warp padding | 20%, reflect | 10%, reflect. |
| Consistency blur | 2 | 1, plus modern mask processing; flow/mask parity was not measured. |
| Color strength | 0.4 | Color matching disabled; placement/algorithm parity was not measured. |
| Turbo | Enabled, interval 3, preroll 10, warp-only intermediate frames | Disabled by default. Shared turbo supports these controls, but adds scene resets by default. |
| Inpainting | On; stop 6; blur 15; threshold 220/255 | Same values by default. Actual masks still depend on the flow/consistency pipeline. |
| Output canvas | Manual 530×908, rounded **down** to **512×896** | Source aspect and upward /64 rounding, as requested for the application. This intentionally differs. |

Matching the scalar settings alone cannot remove the cutout and conditioning
differences. No “exact notebook preset” is supplied, because that label would
overstate what the current implementation can reproduce.

## Original findings and current status

1. **Resolved in follow-up: cutout padding, resizing, grayscale and augmentation.**
   Notebook cell 13, lines 227–268 uses zero padding, ResizeRight, a fixed
   grayscale crop prefix (`i <= int(probability * crop_count)`, including one
   grayscale crop when the probability is zero), and the longer video augmentation
   stack. The initial `make_cutouts` used replicated borders, antialiased bicubic resize,
   Bernoulli grayscale selection and a shorter augmentation stack. The RNG
   consumption also differed. On a seeded 64×96 synthetic input, 16 overview
   cutouts with augmentation disabled differed by mean absolute **0.1679**
   on the float image scale. Gradient cosine for the squared-cutout diagnostic
   loss was **0.4168**; this is not a measured CLIP-loss gradient cosine.

2. **Denoising-progress indexing differs at boundaries.** Notebook cell 13,
   lines 601–610 uses `999-int(t)` into 1000-entry arrays. The application uses
   `100*(1-t/1000)` against percentage keys. For `t=600`, the notebook selects
   entry 399, while a percentage key at 40 has already switched in the app.

3. **The notebook's `frames_scale` does not control the LPIPS loss as written.**
   Cell 13 assigns local `init_scale=args.frames_scale` on temporal frames, but
   the loss reads `args.init_scale` at lines 627–629. The supplied setting is
   zero, so the source does not activate the advertised 15000 temporal LPIPS
   scale. Faithfully reproducing that behavior means LPIPS off, not 15000.
   Current application guidance uses its explicit configured scale correctly.

4. **Range-loss integration is a deliberate correction, not literal parity.**
   The notebook includes `range_loss(out)` in a loss differentiated first with
   respect to downstream `x_in`; that separate range branch contributes no
   gradient there. The app differentiates the prediction-range branch directly
   with respect to the sample. A controlled graph check gives zero notebook
   range gradient versus app RMS **2.1844** at scale 150. Both are identical
   when range scale is zero, as in the supplied notebook.

5. **PLMS conditioning and zero-gradient handling are more robust in the app.**
   The app resolves each actual mapped timestep, including bootstrap evaluations;
   the notebook's denoiser/blend uses outer `cur_t`. Its clamp can divide by zero;
   the app guards the RMS and nonfinite gradients. The measured sampler equality
   above uses an explicit common conditioning function, not the notebook's entire
   `cond_fn`, and does not erase these differences.

6. **Legacy Disco settings exports need a dedicated importer.** The current
   importer recognizes WarpFusion SD settings but does not map `diffusion_model`,
   CLIP toggles, cutout arrays, temporal skip controls and Disco loss scales into
   `config.disco`. The notebook exporter also writes the literal string
   `'inpainting_stop'` for that field (cell 13, line 896), rather than its value.
   The import probe uses a generated representative subset, not a captured export.

Optional guidance masking, noise-relative caps, tiling, progressive resizing,
locked cutouts and encoder balancing must also be disabled for a literal legacy
comparison. Flow generation, consistency construction, color matching, turbo
video feedback and full prompt/image-prompt behavior remain outside the measured
parity scope here.

## Reproduce

```powershell
.venv/Scripts/python.exe tools/check_disco_notebook_parity.py --notebook 'C:/code/my/discodiffusion-warp/warpfusion_diffuse_inpaint_v0_7_2 (1).ipynb' --source-root C:/code/my/discodiffusion-warp --secondary-checkpoint models/disco/secondary_model_imagenet_2.pth
```

The tool produces source provenance, settings extraction, importer diagnostics,
per-case sampler errors, loss/secondary errors and cutout diagnostics under
`.cache/disco-notebook-parity-072`. It downloads no models and changes no notebook.
