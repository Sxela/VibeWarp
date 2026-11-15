# Noise-relative guidance update cap

`disco.guidance_noise_cap_percent` is optional and defaults to **0 (off)**.
The UI field is **Guidance update cap (% of step noise RMS, 0 off)** under
the advanced Disco controls.

It caps the **applied sampler displacement**, not the raw CLIP gradient:

```text
budget = percent / 100 × sigma_t
delta  = guided_proposal - unguided_proposal
factor = min(1, budget / RMS(delta))
result = unguided_proposal + factor × delta
```

RMS and factors are calculated independently for each image. Direction is
preserved. `sigma_t = sqrt(1 - alpha_bar_t)` is the current marginal VP noise
standard deviation in RGB [-1,1] units. It is not DDIM's stochastic eta noise,
which is zero for deterministic DDIM. Gaussian noise has no finite maximum.

- **100%:** displacement RMS cannot exceed one current-noise standard deviation.
- **10%:** at most one tenth of that standard deviation.
- **1%:** at most one hundredth.
- **0:** disabled, retaining the previous sampler path.

This is an image-RMS limit, **not a per-pixel absolute bound**. Individual
pixels can exceed the RMS budget. Floating-point cancellation can produce
tiny numerical differences at the bound.

## What it caps

The cap applies to the combined semantic guidance (CLIP or SigLIP), frame
guidance, and TV/range/saturation regularization. These losses share the
sampler's conditioning field, so leaving the other terms uncapped would not
guarantee a total guidance-update budget. It does not cap the base denoiser's
ordinary update or the separately sampled DDIM eta noise.

The existing `disco.clamp_max` raw-gradient RMS clamp remains active. The new
cap is an additional ceiling, not an automatic gain or replacement guidance
scale. A permissive percentage may never activate if the gradient clamp or
loss magnitudes already keep applied updates below that budget.

For DDIM, the guided and unguided proposals share the same input/model
prediction; the cap runs before adding the same stochastic noise. It requires
no extra U-Net evaluation.

For PLMS, the cap runs after bootstrap/history extrapolation and final-step
handling. Parallel unguided epsilon history is recorded at the same actual
states. Bootstrap computes its unguided Euler state explicitly, requiring one
extra unguided model evaluation whenever PLMS history starts or restarts.
This baseline removes the explicit guidance contribution for the current
proposal; it is not a separate unguided trajectory from the original seed.
Tests confirm that returning the baseline at every step reproduces a fully
unguided rollout, including history and DDIM eta noise.

Progressive sampling uses the native grid and noise level of each stage;
PLMS resets both histories at scale transitions. The hook is scoped to one
render and restored on exit or cancellation. Step events record the applied
RMS, budget and limiting factor. Distilled unconditional LoRA mode rejects a
nonzero cap because that route has no validated guidance path.

Merge this into existing guided Disco settings to try a 1% ceiling:

```json
{"disco": {"guidance_noise_cap_percent": 1.0}}
```

## Controlled check, 2026-11-05

RTX 5090, Disco 512, 30 DDIM steps, seed 42, secondary denoiser, ViT-B/32,
eight cutouts, no LPIPS/flow/turbo, full-resolution sampling. Each condition
was repeated twice with deterministic spatial backward controls. Those CPU
spatial-gradient controls are for correctness; **these are not production
timing measurements**.

| Setting | Limited steps | Output repeatable | Difference from disabled |
|---|---:|---|---|
| Off | 0 | Byte-identical repeats | Reference |
| 100% | 0 | Byte-identical repeats | Pixel-identical |
| 1% | 13/30 | Byte-identical repeats | Substantial composition/color change |

With the original 0.05 gradient clamp, the maximum uncapped update RMS was
only **1.888% of sigma_t**, explaining why a 100% ceiling did not bind. The
1% condition reached its budget to floating-point tolerance. It retained
more scene structure in this example but remained strongly saturated; no
general quality fix or tuned preset is claimed from one prompt/seed.

![Disabled, 100%, and 1% guidance caps](../.cache/disco-guidance-noise-cap-comparison/comparison.png)

[Combined metrics](../.cache/disco-guidance-noise-cap-comparison/results.json) ·
[100% step records](../.cache/disco-guidance-cap-100/results.json) ·
[1% step records](../.cache/disco-guidance-cap-1/results.json).

Validation: **1579 backend tests passed, 20 skipped, 5 deselected**;
**120 frontend tests passed**; UI build passed. Targeted tests cover
per-image budgets, stochastic DDIM, PLMS bootstrap/history/final steps,
uncapped-value parity, native progressive guidance, model reuse and hook
restoration, masks/previews, and invalid configuration values.
