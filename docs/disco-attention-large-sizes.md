# Disco attention at 1024 and 2048 pixels

Measured on 2026-11-05 using the existing Disco 512 checkpoint, RTX 5090,
Torch 2.11.0+cu128 and the installed SageAttention 2.2 Windows package.
Rendering dimensions were increased directly; these are pixel-space network
evaluations, not upscaled 512-pixel results. Defaults were not changed.

## 1024 x 1024: full guided renders

30 DDIM steps, secondary denoiser, ViT-B/32, eight cutouts, seed 42,
two warmed repeats per backend in alternating order. No flow, LPIPS,
inpainting, turbo, or preview I/O. Each candidate has a separate paired manual
baseline; compare within a row rather than across sessions.

| Candidate | Manual frame median | Candidate frame median | Speedup | Peak allocated, both |
|---|---:|---:|---:|---:|
| SDPA | 12.318 s | 12.014 s | 1.025x | 6.153 GiB |
| Sage | 12.845 s | 12.377 s | 1.038x | 6.153 GiB |

These small differences overlap individual-run variation. Sage dispatched
480 actual Sage calls per frame. Production image repeats also drifted, so
the generated comparison images do not establish attention-only visual parity.
There was no deterministic large-image correctness pass.

Artifacts, including images and complete reports:

- [SDPA results](../.cache/disco-attention-1024-sdpa/results.json)
- [SDPA images](../.cache/disco-attention-1024-sdpa/comparison.png)
- [Sage results](../.cache/disco-attention-1024-sage/results.json)
- [Sage images](../.cache/disco-attention-1024-sage/comparison.png)

## Initial 2048 x 2048 test: U-Net forward passes only

| Candidate | Paired manual median | Candidate median | Speedup | Manual peak | Candidate peak |
|---|---:|---:|---:|---:|---:|
| SDPA | 8.029 s | 6.641 s | 1.209x | 26.756 GiB | 18.834 GiB |
| Sage | 7.389 s | 6.082 s | 1.215x | 26.756 GiB | 18.834 GiB |

Two warmed single-forward measurements per backend, alternating order.
Each candidate used the same weights and input as its manual baseline.
Peak memory is PyTorch allocated memory, excluding other applications and
unused allocator reservations. The reduction is **7.922 GiB, or 29.6%**.
Real Sage calls were verified in the forward dispatch probe.

The GPU already had approximately 10.5 GiB occupied before benchmarking.
Physical GPU memory filled during the 2048 tests. Windows GPU memory paging
can therefore affect the timing results; these are measurements under memory
pressure, not clean isolated kernel speed estimates. No other applications
were closed or modified.

The first long 2048 benchmark was interrupted during repeated U-Net timings.
A shorter attempt in that initial session completed the paired SDPA U-Net measurements, but its full
guided warmup remained impractically slow under memory pressure and was also
interrupted. It had been configured for 10 DDIM steps. That session produced no completed 2048 guided
frame timings or image parity claims. Sage was then measured
using the explicit U-Net-only mode.

- [SDPA partial results, with completed U-Net samples](../.cache/disco-attention-2048-sdpa-short/results.json)
- [SDPA log](../.cache/disco-attention-2048-sdpa-short.log)
- [Sage U-Net results](../.cache/disco-attention-2048-sage-unet/results.json)

Reproduce a 2048 forward comparison from the repository root:

```powershell
$env:TRITON_CACHE_DIR = Join-Path (Get-Location) '.cache\triton'
$env:CUDA_CACHE_PATH = Join-Path (Get-Location) '.cache\cuda-cache'
.venv/Scripts/python.exe tools/benchmark_disco_attention.py --candidate sage --size 2048 --unet-only --unet-repeats 2 --unet-iterations 1 --checkpoint 'C:\code\my\discodiffusion-warp\models\512x512_diffusion_uncond_finetune_008100.pt' --output .cache/disco-attention-2048-sage-unet
```

Use `--candidate sdpa` for SDPA. Omit `--unet-only` for full renders when
sufficient GPU headroom is available. The memory saving becomes substantial
at 2048 even though the measured benefit at 512 and 1024 was small.

## 2048 rerun with background GPU memory cleared

The user cleared other GPU allocations. Starting usage was 2362 MiB. The native
allocator still developed excessive reservations within our test: a manual
U-Net pass peaked at 26.756 GiB allocated but 35.904 GiB reserved, versus
18.834 GiB allocated / 32.906 GiB reserved for SDPA. Unused cache was released
between backends, outside timing. Limiting large-block splitting to 128 MiB
failed with CUDA OOM and was discarded. Expandable segments were reported
unsupported by this Windows Torch build.

CUDA's async allocator (`PYTORCH_ALLOC_CONF=backend:cudaMallocAsync`) supported
the completed paired 30-step guided-frame tests below. This was a benchmark
process setting; launchers and application defaults were not modified. Both
backends in each pair used the same allocator. Its memory counters should not
be treated as identical to the native allocator's reservation statistics.
Cache release and allocation bookkeeping can also affect these timings.

| Candidate | Manual frame median | Candidate frame median | Speedup | Manual allocated peak | Candidate allocated peak |
|---|---:|---:|---:|---:|---:|
| SDPA | 70.823 s | 76.228 s | 0.929x | 26.933 GiB | 19.011 GiB |
| Sage | 67.824 s | 65.305 s | 1.039x | 26.933 GiB | 19.011 GiB |

Reserved peaks were 30.156 GiB manual and 23.375 GiB optimized. Each test used
one seed (42), two timed repeats per backend, alternating order, plus warmups.
The Sage frame range was 52.746–77.864 s, illustrating substantial timing
variation. A CPU-only validation suite also ran during part of the Sage session;
these results must not be described as isolated GPU-kernel performance.
The substantial allocation reduction is repeatable, but end-to-end speed
improvement is not consistent. No deterministic full-image parity pass was
performed at 2048px.

Forward-only U-Net medians in the same sessions:

| Candidate | Manual | Candidate | Samples per backend |
|---|---:|---:|---:|
| SDPA | 1.506 s | 1.248 s | 4 |
| Sage | 1.462 s | 1.169 s | 4 |

- [Completed SDPA report](../.cache/disco-attention-2048-sdpa-free-async/results.json)
- [SDPA image comparison](../.cache/disco-attention-2048-sdpa-free-async/comparison.png)
- [Completed Sage report](../.cache/disco-attention-2048-sage-free-async/results.json)
- [Sage image comparison](../.cache/disco-attention-2048-sage-free-async/comparison.png)

Repeat with the same allocator setting and the benchmark's `--size 2048
--steps 30 --repeats 2 --seeds 42 --unet-repeats 4 --unet-iterations 1` options.
For lower-memory sampling, see the optional tiled/multiscale mode in the
[Disco guide](disco-diffusion.md#tiled-and-multiscale-sampling).
