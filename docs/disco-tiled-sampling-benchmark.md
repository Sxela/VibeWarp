# Disco tiled sampling: implementation and measured tradeoffs

**Historical note:** the coarse/multiscale results below used the original
full-size-state, prediction-resizing implementation. Disco has since switched
to [persistent native sampling grids](disco-progressive-sampling.md). Pure
100%-resolution tiling is unchanged. The earlier coarse timings and images
must not be presented as measurements of the new implementation.

Tested 2026-11-05 on the RTX 5090, Windows, Torch 2.11.0+cu128.
The optional mode uses the shared per-model-evaluation tiling wrapper, adapted
for pixel diffusion after reviewing
[MagoStudio's WAN tiled sampler](https://github.com/MagoStudio/ComfyUI-MagoNodes/blob/main/wan_tiled_sampler.py).
It requires no ComfyUI runtime dependency. This is approximate tiled sampling,
not lossless feed-forward chunking.

## Measurements

2048 x 2048, Disco 512 checkpoint, SDPA, 30 DDIM steps, secondary denoiser,
ViT-B/32, eight cutouts, full regeneration, seed 42, two warmed timed repeats.
No flow, LPIPS frame guidance, turbo, inpainting, or preview I/O in the timing.
Model load and PNG writing are excluded. CUDA async allocation was used for
these complete comparisons to fit the untiled control without the native
allocator's excessive reservation growth. Application/launcher defaults remain
unchanged. Memory columns are PyTorch counters, not whole-machine GPU usage.

| Mode | Median seconds/frame | Peak allocated | Peak reserved |
|---|---:|---:|---:|
| Untiled, full resolution | 47.273 | 19.011 GiB | 23.375 GiB |
| Tiled, 100% resolution throughout | 83.256 | 5.867 GiB | 6.875 GiB |
| Coarse 25% → 50% → tiled 100% | 32.415 | 6.810 GiB | 7.688 GiB |

**Pure tiling saved 69.1% of peak allocated memory but took 1.76x as long.**
Target tile size was 768px, overlap 35%, split threshold 20%. The resolved
grid is 3 x 3, each tile 896 x 896 after alignment. Overlap therefore evaluates
more total pixels than one full-frame pass. These settings prioritize memory,
not speed.

The aggressive coarse schedule used `{0:25, 10:50, 20:100}`. Its speedup is
from both reduced early resolution and tiling, and it **failed visual review**:
severe saturation and lost structure. It is not a recommended quality preset.
Pure tiling retained recognizable village/flower motifs, but composition and
detail changed. Its full-frame comparison SSIM was 0.734; this is a descriptive
image-difference measure, not an aesthetic score or a parity guarantee.

The untiled/coarse comparison alternated modes within one session. Pure tiling
was measured in a separate nearby session with the same protocol, not
interleaved with that control. Only one prompt/seed was tested. Production
same-mode repeats also drift, so full-image changes cannot all be attributed
to tiling. No claim of deterministic pixel parity is made.

![Untiled, pure tiling, and failed aggressive multiscale output](../.cache/disco-tiling-2048-fullscale/full-vs-tiled-vs-multiscale.png)

- [Combined summary](../.cache/disco-tiling-2048-fullscale/comparison-summary.json)
- [Untiled/coarse paired results](../.cache/disco-tiling-2048-multiscale/results.json)
- [Pure tiling results](../.cache/disco-tiling-2048-fullscale/results.json)

A separate native-allocator compatibility smoke test exercised all three
scales at 2048px, using six DDIM steps and `{0:25, 2:50, 4:100}`. Warmup and
two measured frames completed, at 6.835 GiB allocated / 11.072 GiB reserved.
This verifies the mode can run without an allocator environment override;
its timing and image quality are not comparable to the 30-step tests.
[Smoke report](../.cache/disco-tiling-2048-native-smoke/results.json).

## Behavior and controls

Enable **Tiled Sampler** in the existing sampler controls. Begin with
`sampler_scale_schedule={0:100}`. Tile size, overlap, split threshold, minimum
multiscale size, and scale schedule now appear for Disco as well as SD.
Tiling/multiscale remain off by default. The [Disco guide](disco-diffusion.md#tiled-and-multiscale-sampling)
contains a settings example.

The global noise state, DDIM/PLMS trajectory, masks, guidance targets, cutouts,
and previews stay at full resolution. Primary epsilon/learned-variance
predictions and secondary clean-image predictions are tiled per evaluation,
with 64-pixel-aligned bounds and normalized positive overlap weights. Secondary
and main-model guidance backward recompute tile activations to bound retained
neural-network memory. Coarse epsilon restoration goes through a clean-image
estimate, preserving the full-resolution noise state. PLMS bootstrap calls
resolve the schedule from mapped timestep, including strength-based skipping.

Validation: 64 targeted tests passed, followed by **1502 backend tests passed,
20 skipped, 5 deselected**, and **118 frontend tests passed**. Coverage includes
head/variance shapes, alignment, overlapping gradients, noisy downscaling,
clean prediction restoration, mapped schedules, DDIM/PLMS, inpainting masks,
previews, settings round trips, and model-family UI visibility. Real CUDA
sampling additionally exercised the secondary-denoiser backward path.

Reproduce pure tiling:

```powershell
$env:PYTORCH_ALLOC_CONF = 'backend:cudaMallocAsync'
$env:TRITON_CACHE_DIR = Join-Path (Get-Location) '.cache\triton'
$env:CUDA_CACHE_PATH = Join-Path (Get-Location) '.cache\cuda-cache'
.venv/Scripts/python.exe tools/benchmark_disco_tiling.py --modes tiled --size 2048 --steps 30 --tile-size 768 --scale-schedule '0:100' --checkpoint 'C:\code\my\discodiffusion-warp\models\512x512_diffusion_uncond_finetune_008100.pt' --output .cache/disco-tiling-2048-fullscale
```

Omit `--modes tiled` to alternate a fresh untiled control against tiling. Use
the native allocator for a production-environment comparison when that control
fits; do not infer absolute timings across different allocation settings.
