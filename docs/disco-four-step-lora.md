# Optional distilled Disco LoRA

[Optional quality LoRA](disco-quality-lora.md) adds an independently adjustable
aesthetic branch alongside this turbo adapter, or alone with DDIM.

The **distilled** sampler runs the Crowson/OpenAI 512px adapter inside the
normal guided DDIM/video pipeline. CLIP/SigLIP, image prompts, LPIPS/image
guidance, consistency masking, inpainting, flow, turbo, tiling and progressive
sampling remain available. An exact unconditional reference route is retained.
This integration follows the sibling project's
[LoRA inference handoff](../../discodiffusion-warp/LORA_INTEGRATION_HANDOFF.md).
It is a custom adapter format; SD, Diffusers and PEFT LoRA files are not accepted.

**Render → Disco Diffusion → LoRA strength** (`disco.lora_strength`, default 1)
multiplies the adapter's trained alpha/rank contribution. **0** disables the
adapter, **1** uses its trained strength, and values above 1 amplify it. The
setting is saved with the run and works in both guided and standalone modes,
including checkpoint recomputation during guidance backward. It changes no
checkpoint weights. Standalone parity comparisons use strength 1.

## Use it

1. Select **Disco 512** and choose **distilled** under **Sampler**.
2. In **Project → Model**, below the model selector, set **Distilled LoRA file or run directory**
   to an exported inference `.safetensors` file, or a supervised training run
   containing `best.json`.
3. Choose the exact base checkpoint under the existing custom checkpoint path,
   or leave that path blank to prepare the catalog's checksum-verified base.
4. Keep **Guided distilled sampling** enabled, then set **Steps Schedule** and
   **Style Strength Schedule** as for regular Disco. A four-step-trained adapter
   can run six, twelve or other counts between 2 and 1000. More steps do not
   imply a better result or a newly trained adapter.

Selecting distilled preserves prompts, guidance settings, dimensions, flow,
turbo, schedules and the secondary-model choice. The UI changes only the sampler
and keeps the secondary-model control available. Use `eta=0` explicitly for
deterministic DDIM, or retain a chosen stochastic eta. Sampling uses FP32 parameters/head/update arithmetic and
the adapter's precision setting for UNet autocast. The adapter stays active
during the entire render, including checkpoint recomputation in guidance backward.

The original 1000-step linear VP schedule stays fixed. At the trained step count
we use the saved original indices exactly; at another count we linearly resample
the same span from 999 to the clean endpoint -1, rounding intermediate indices
toward zero. Four-step weights at six steps use `[999,832,665,499,332,165,-1]`.
Frame schedules change this respacing without reloading weights. Strength skips
the earliest scheduled transitions, and inpainting stop is clamped to the executed
budget. The final output uses the actual last DDIM state, including any guidance
cap, rather than an uncapped prediction.

The local v0.8 notebook implements guided LoRA but still enforces its trained
step count. Expanded step schedules are VibeWarp's extension, based on the
previous six-step resampling experiment below. The existing optional
**Guidance update cap** works here; 10% matches the notebook's starting preset.
Its quality and an appropriate guidance scale still require render comparisons.

For the original unconditional parity route, import one of the reference examples
below (`disco.distilled_guided=false`). That route retains fixed 512×512,
the adapter's trained grid, full-noise starts and disabled guidance/flow/masks.
It is separate from the guided path and preserves the earlier parity contract.

[Importable guided six-step example](../examples/disco_guided_lora.json).
This is a functional configuration, not a validated quality preset; select an
input video, verify the adapter path and tune guidance for that content.

An [importable four-step example](../examples/disco_four_step_lora.json) is
included. Adapter/checkpoint paths are project settings and appear below the
model selector. Check those paths after importing settings from another machine.

The local supervised 32-seed export was copied to:

```text
models/disco/lora/crowson-uncond512-4step-r16-supervised32-ema.safetensors
```

Its SHA256 is
`4c5440462a40e376d7ef78b69bf78bc70642457872d32dda41809590b7b9e4db`.
This is the locally available exported adapter, **not a claim that it is the
latest selected best from the Colab training run**. To follow the current
selected best, point the field at that run directory. The loader reads
`best.json`, resolves `best/<checkpoint>/best-ema.safetensors`, requires its
`COMPLETE.json` manifest, and verifies the checksum. It never loads `resume.pt`.

### User-provided training-update 1680 checkpoint

`student-1680step.safetensors` is now installed alongside the earlier EMA
export at `models/disco/lora/student-1680step.safetensors`. Metadata identifies
**raw student weights at training update 1680**, not EMA/selected-best weights.
Its inference grid still uses **four** steps, rank 16 and alpha 16. Source and
installed SHA256 match:
`9c0ab40499f20cc0377d62fb15a7c78cd4e5a933093a246ef0038c515ee5a244`.

[Importable 1680 profile](../examples/disco_four_step_1680.json).
Set the adapter path below the model selector to this file, or import this
profile. The older EMA export is preserved.

The same real CUDA parity check passed for this checkpoint: seeds 12345 and
12346, with adapter enabled and disabled, produced exact reference tensor
and pixel equality (maximum error zero). The public render path also passed.
[1680 parity report](../.cache/disco-lora-1680-parity/results.json) ·
[1680 example output](../.cache/disco-lora-1680-parity/render.png).

### Comparison with the normally sampled base/teacher

The comparison now includes the **48-step teacher trajectory used to generate
the distillation targets** (12 substeps per saved interval) and a **50-step
unadapted base run**, in addition to both four-step adapters. Each uses the
same initial noise for seeds 12345 and 12346, original linear VP schedule,
FP32 parameters/output head/update arithmetic, and FP16 autocast. There are no
prompts, init blending or intermediate clipping. These seeds do not overlap
the adapter's training/held-out seed ranges.

[48-step teacher / 50-step base / earlier EMA / update 1680 comparison](../.cache/disco-lora-teacher-comparison/comparison.png).
The 48-step grid follows the training code's per-interval integer linspace;
the 50-step baseline follows `tdm.core.timestep_grid(50)`, from 999 to the clean
endpoint. This is not the legacy guided application's respaced/init-image loop.

Mean reconstruction proxy against the 48-step teacher, lower is better:

| Candidate | Model evaluations | Mean paired multiscale L1 + saturation penalty |
|---|---:|---:|
| Base, four-step control only | 4 | 0.151708 |
| Earlier EMA | 4 | 0.070528 |
| Raw update 1680 | 4 | 0.063341 |

Update 1680 reduces this proxy by approximately 10.2% versus the earlier EMA
on these two seeds. It is still visibly grainier and changes detail compared
with the 48/50-step teacher/base. The two longer base runs are visually close.
The metric follows `tdm.evaluation.paired_score` against unclipped teacher
endpoints. It is a reconstruction proxy, not a human-preference metric; two
examples do not establish general quality superiority. Step-count reduction
is not a measured wall-time speedup.

[Full metrics and exact grids](../.cache/disco-lora-teacher-comparison/results.json) ·
[Unclipped output tensors](../.cache/disco-lora-teacher-comparison/raw_tensors.pt).
The reusable `tools/compare_disco_lora_teacher.py` script generates this study.

### Experimental six-step grid with the four-step-trained 1680 weights

Tested the same adapter at `[999,832,665,499,332,165,-1]`, without changing
the original 1000-step VP schedule, metadata, or alpha/rank scale. This is
resampling four-step-trained weights, not the separately trained six-step
adapter. All results were finite. Both four-step controls reproduced the
previous cached raw tensors exactly, validating the cached teacher comparison.

| Candidate | Mean reconstruction proxy to 48-step teacher, lower is better |
|---|---:|
| 1680 LoRA, trained four-step grid | 0.063341 |
| 1680 LoRA, experimental six-step grid | 0.100846 |
| Unadapted base, six-step grid | 0.122502 |

The six-step adapter result had approximately **59.2% higher reconstruction
error** than its four-step result, worse on both seeds. It produced stronger
contrast and amplified texture; SSIM improved on seed 12345 and worsened on
12346. It remains better than the six-step unadapted base on the aggregate
proxy, but this does not establish a general visual-quality gain. These two
seeds are illustrative and there was no warmed latency benchmark.

[Teacher / base-six / LoRA-four / LoRA-six images](../.cache/disco-lora-six-step-experiment/comparison.png) ·
[Metrics and exact grids](../.cache/disco-lora-six-step-experiment/results.json).
The experiment is reproducible with `tools/experiment_disco_lora_steps.py`.
The application's trained-grid validation remains intact; adapter files and
normal sampling defaults were not changed.

## Supported behavior

The validated mode is **unconditional 512 x 512 generation**. Prompts,
CLIP/SigLIP, image/frame guidance, image-strength blending, masking, flow,
larger dimensions, stochastic sampling and progressive resizing are rejected.
The video pipeline can emit independent unconditional frames using its seed
sequence; the input video does not condition these generated images. This
does not establish a four-step video stylization workflow.

The supervised adapter retains noise/artifacts. Exact agreement with its
reference implementation is not a claim of teacher-quality images. Rejected
TDM v1/v2/v3 formats are not substituted for the supervised recipe. Four-step
weights are not treated as six-step weights. Six-step metadata/math is supported
for its matching grid; the real-checkpoint GPU parity check here used four steps.

## Inference contract

- Base SHA256:
  `9c111ab89e214862b76e1fa6a1b3f1d329b1a88281885943d2cdbe357ad57648`.
- Fresh FP32 base parameters, with no half conversion. FP16 CUDA autocast
  surrounds inference; the output head and VP arithmetic stay FP32.
- All Linear, Conv1d and Conv2d targets receive the `student` branch, including
  selected EMA exports whose keys still use that name. Exact keys, shapes,
  finite weights, saved rank and alpha/rank scale are enforced. Injection
  cannot nest wrappers. The optional UI strength multiplies the trained scale;
  its default of 1 preserves the original calculation.
- Original 1000-step linear VP schedule and original integer timesteps.
  Four-step grid: `[999,749,499,249,-1]`. The clean endpoint `-1` is never a
  model call. Six-step grid: `[999,832,665,499,332,165,-1]`.
- Epsilon uses the first three of six output channels. Learned variance is
  ignored in this deterministic route. Eta is zero; no intermediate clean
  clipping. Only display/export images are clamped.
- The adapter context remains active for the whole rollout and restores the
  prior active branch on exit. No training package or globally patched
  `guided_diffusion` namespace is imported by the host backend.

## Acceptance evidence, 2026-11-05

Guided runtime smoke checks used the real `student-1680step.safetensors` at
512×512, seed 42, ViT-B/32, CLIP scale 1000, LPIPS 10, RGB loss 1 and a 10%
guidance update cap. Both the trained four-step grid and the rescaled six-step
grid completed, respected the cap and restored adapter state after guidance
backward. Peak allocated memory was about 7.62 GiB. The warm six-step frame
took 7.76 seconds; the first four-step timing includes lazy LPIPS setup and
is not a comparable speed measurement. These are functional checks, not a
quality comparison or a recommended guidance preset.
[Guided smoke report](../.cache/disco-guided-lora-smoke-1105/results.json).

The following parity evidence applies to the standalone reference route:

The standalone reference used the provided `tdm.core`, its vendor commit
`22e0df8183507e13a7813f8d38d51b072ca1e67c`, and the same local exported adapter.
On RTX 5090 / Torch 2.11.0+cu128:

- Seeds **12345 and 12346**, adapter active and disabled: **raw tensor equality
  and exported pixel equality in all four comparisons**. Maximum error: zero.
- Every rollout called exactly **999,749,499,249**, with the expected active
  branch at every call. All **207 adapter layers** were present.
- Stored parameters and output-head tensors were FP32.
- Disabling the adapter matched the standalone frozen-base control. The
  adapter had a nonzero measured effect when enabled.
- The public image-render entry point also matched the reference pixels.
- Backend regression: **1551 passed, 20 skipped, 5 deselected**. Frontend:
  **120 passed**. The packaged UI build passed.

[Raw parity report](../.cache/disco-lora-parity/results.json) ·
[Example output](../.cache/disco-lora-parity/render.png).

Reproduce reference parity from the checkout root:

```powershell
.venv/Scripts/python.exe tools/check_disco_lora_parity.py --source-root 'C:\code\my\discodiffusion-warp' --teacher 'C:\code\my\discodiffusion-warp\models\512x512_diffusion_uncond_finetune_008100.pt' --adapter 'models/disco/lora/crowson-uncond512-4step-r16-supervised32-ema.safetensors'
```
