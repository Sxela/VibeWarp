# Optional quality LoRA

The optional quality adapter is an independent additive branch on the Crowson
Disco 512 primary denoiser. It works alone with DDIM or together with the
existing turbo/distilled LoRA. It is **disabled by default**.

These are custom Linear/Conv1d/Conv2d adapters, not SD/PEFT LoRAs.

## UI setup

1. Under **Render → Disco Diffusion**, enable **Enable quality LoRA**.
2. Under **Project → Model**, set **Quality LoRA file** to the aesthetic export.
3. Set **Quality LoRA strength** independently of the existing turbo **LoRA strength**.
   Both are fractions: 1 means 100%, 0 removes that branch's contribution.
4. For quality-only inference, choose **DDIM**. No turbo file is required or read.
   For a stack, choose **distilled**, keep **Guided distilled sampling** on, and
   supply the existing distilled/turbo path as well.

Enabling quality changes only the adapter's enabled state. It preserves the selected
sampler, DDIM eta, secondary denoiser, guidance, prompts, video dimensions,
frame schedules, flow, masks and turbo frame interpolation. Choose Disco 512 and
DDIM or guided distilled sampling explicitly; validation reports incompatible choices.
The secondary-model control remains available. Strength and path
settings are saved with the run. Models load once per run; strengths are applied
from the original scales rather than multiplying already scaled weights.
The custom quality file has no automatic download URL.

## Supplied files

| Role | File | Actual metadata | SHA256 |
|---|---|---|---|
| Quality | `models/loras/student-aesthetic-3908step-r128.safetensors` | **Iteration 8908**, raw; rank 128, alpha 128; aesthetic real-image epsilon MSE | `8899f61c69755ddebe6fe0226b581d26f73b638033b615c13988d0cdd7ea65ad` |
| Turbo | `models/loras/student-r64-3200step.safetensors` | Iteration 3200, raw; rank 64, alpha 64; supervised DDIM endpoint regression | `c2d6f107eb0a0e1dfb609d5553c8d58df88cb9065c387282a76067ccfbf528a7` |

The quality filename's `3908` is not its training iteration. Metadata and hash
are logged on load. The loader validates the base checksum, quality method and
objective, architecture/size, rank/alpha, precision, FP32 head, epsilon training,
complete adapter key set, tensor shapes and finite floating-point weights.

For every wrapped layer, both branches receive the same original input:

```text
base(x)
  + quality_strength * (quality_alpha / quality_rank) * Q_up(Q_down(x))
  + turbo_strength   * (turbo_alpha   / turbo_rank)   * T_up(T_down(x))
```

Quality's exported `student` keys are remapped into `adapters.quality`; the turbo
branch remains `adapters.student`. The ranks and scales are independent. Quality
continues working when turbo strength is zero. Both branches remain active for
guidance backward and checkpoint recomputation, and their activation/strength
state is restored on exit, including cancellation/failure.

## Schedule and guidance

The stack retains the **original 1000-step linear VP schedule**, FP32 parameters,
FP32 output head/update arithmetic and the existing precision-aware UNet autocast
wrapper. Explicit frame step schedules already provide the rescaled-grid override:
enabling quality does not silently change the selected count or strength.
Rescaling is an inference experiment, not a different trained turbo adapter.

| Scheduled steps | Style strength | Executed calls | First original timestep |
|---:|---:|---:|---:|
| 6 | 0.5 | 3 | 499 |
| 10 | 0.5 | 5 | 499 |
| 30 | 0.5 | 15 | 499 |

All modes use the same explicit integer-grid rule, `1000*(steps-i)//steps-1`.
No LoRA-specific scaling is applied to the user's nominal CLIP guidance scale.

When quality is enabled and the noise-relative cap is nonzero, that cap
**replaces the absolute gradient clamp** on this guided DDIM path, matching the
handoff. Setting the cap to zero returns to the existing absolute clamp.
The cap also applies in quality-only mode and when either strength is zero.
It bounds the guidance-induced next-state correction, excluding denoising and
inpainting updates. Trace events include original timestep, spaced index, sigma,
raw/capped update RMS, cap budget/engagement and counted nonfinite gradient values.
Nonfinite gradients are logged before sanitation; sanitation is not evidence
that upstream computations were finite. The final image uses the actual final
DDIM sample. Existing CLIP/SigLIP, image prompts, LPIPS/image losses, optional
guidance masking, inpainting, flow and frame turbo remain in the shared path.

[Importable stacked example](../examples/disco_quality_turbo.json) contains the
handoff's nominal 3/6-call, CLIP 70000, cap 15%, seed 100500 settings and supplied
Windows file paths. Set an input video and check paths before rendering. The
video aspect is derived from the source, subject to the /64 render grid.
This is a candidate profile, not a general quality recommendation.

## Verification and scope

- Independent Linear, Conv1d and Conv2d tests check different ranks, additive
  outputs/input gradients, exact zero/zero vanilla output and quality without a
  turbo branch. Full tensor-key/shape validation remains in the load path.
- Five-frame CPU integration checks run both quality-only and stacked backends
  through the shared frame pipeline, with synthetic identity warps, consistency
  masks and changing independent strengths. They verify sampler persistence,
  inpainting ranges and restoration without cumulative scaling. They do not
  measure temporal quality on real optical flow.
- Real supplied weights completed 512×512 first-frame CUDA checks for vanilla,
  quality-only, turbo-only and stacked modes, plus quality-only with no turbo
  file loaded. All used three calls at original timesteps 499, 332, 165, one
  ViT-B/32 encoder, four overview cuts, CLIP 70000 and cap 15%. No nonfinite
  guidance was observed; capped updates stayed within budget. This is a smaller
  functional check, not a reproduction of the full 576×832 multi-encoder handoff
  candidate or its five-frame video.

The strict deterministic check (with random augmentations disabled) produced
**exact pixel equality** between quality-only loading without a turbo file and
the stacked loader with turbo strength zero. The augmented CUDA path also runs,
but its affine-grid backward is nondeterministic; strict deterministic checking
therefore uses `--deterministic`, which disables those augmentations. Stacked
peak allocated memory was approximately **8.25 GiB** in the deterministic check.
Reports are generated locally under `.cache/disco-quality-stack-deterministic`
and `.cache/disco-quality-stack-smoke`; caches are not included in releases.

Reproduce the GPU checks:

```powershell
.venv/Scripts/python.exe tools/check_disco_quality_stack.py --quality models/loras/student-aesthetic-3908step-r128.safetensors --turbo models/loras/student-r64-3200step.safetensors --input init_images/000001.jpg
```

Add `--deterministic --output .cache/disco-quality-stack-deterministic` for the
controlled loading-path comparison.

The speed adapter was trained against the original teacher, not re-distilled
against the quality-tuned teacher. Additive stacking and valid execution do not
establish exact distillation fidelity, better temporal coherence or general
aesthetic improvement.
