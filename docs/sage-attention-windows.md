# SageAttention on the Windows RTX 5090 environment

Installed and tested on 2026-11-05 in this checkout's `.venv`:

| Component | Validated version |
|---|---|
| Python | 3.13.15 |
| Torch | 2.11.0+cu128 (unchanged) |
| GPU | RTX 5090, sm120 |
| SageAttention | 2.2.0+cu128torch2.10.0andhigher.post6 |
| Triton Windows | 3.6.0.post26 |

The Sage package is the [woct0rdho Windows fork](https://github.com/woct0rdho/SageAttention),
using its CUDA 12.8 / stable-Torch-ABI wheel. It includes sm120 kernels and
uses the Python stable ABI. Triton Windows comes from PyPI. The CUDA toolkit
installed system-wide was not changed, and no Torch package was replaced.

FA3's published implementation targets Hopper GPUs such as the H100, rather
than this 5090 ([upstream requirements](https://github.com/Dao-AILab/flash-attention#flashattention-3-beta-release)).
Sage was the selected Windows-compatible installation; FA2 and FA3 were not installed.

## Reproduce the installation

Run these commands from the repository root with the intended environment's
Python. The standard `.bat` launchers use `env`, while the benchmark environment
used here is `.venv`; substitute `env` below if that is where the UI runs.

```powershell
$sageWheel = 'https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post6/sageattention-2.2.0%2Bcu128torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl#sha256=103e06df49486daa87c3338bf2490ada2b074da1422f2b90566a8f2f81a6b76e'
.venv/Scripts/python.exe -m pip install --no-deps triton-windows==3.6.0.post26 $sageWheel

.venv/Scripts/python.exe -c "import torch, sageattention, triton; print(torch.__version__, triton.__version__)"
```

The wheel URL pins the SHA256 published with the release asset. `--no-deps`
avoids changing the existing Torch installation. These are the tested versions,
not a universal installation recipe for other Torch/CUDA combinations.

To run the UI in the environment already prepared here:

```powershell
$env:TRITON_CACHE_DIR = Join-Path $PWD '.cache/triton'
$env:CUDA_CACHE_PATH = Join-Path $PWD '.cache/cuda-cache'
.venv/Scripts/python.exe -m vibewarp.web
```

Select `sage` under Disco's attention setting, or set
`disco.attention="sage"` in JSON. The default backend was not changed.
Input-gradient paths continue to use SDPA. A real CUDA backward smoke test
confirmed finite input gradients and zero Sage calls in that path.

## Measured result

Same protocol as the [manual/SDPA benchmark](disco-attention-benchmark.md):
512×512, 30 DDIM steps, secondary denoiser, ViT-B/32, eight cutouts, two seeds,
three alternating-order timed repeats per backend, warmed before measurement.
No LPIPS, flow, turbo or preview I/O. Shared GPU; compare paired measurements
from this run, not absolute timings against earlier runs under different load.

| Production measurement | Manual | Sage 2.2 |
|---|---:|---:|
| Median U-Net forward | 78.79 ms | 79.52 ms |
| Median full guided frame | 3.740 s | 3.745 s |
| Peak allocated GPU memory | 3.049 GiB | 3.049 GiB |

No meaningful end-to-end speedup or overall allocated-memory reduction was
measured. The benchmark counted **480 actual Sage calls per Sage frame**,
and zero in manual frames; it did not benchmark a silent SDPA fallback.
Profiling recorded INT8 quantization and FP8 attention CUDA kernels.

The ordinary CUDA runs had same-backend variation, so image parity was checked
separately with deterministic spatial gradients on CPU, as in the prior report.
Both same-backend repeat controls were byte-identical in that correctness run:

| Seed | Manual–Sage MAE, 8-bit levels | SSIM | PSNR |
|---|---:|---:|---:|
| 42 | 14.1514 / 255 | 0.83920 | 15.60 dB |
| 1234 | 14.2255 / 255 | 0.82705 | 15.61 dB |

The broad arrangement remains similar, with visible local-detail changes.
Sage is available for further experiments, but this case does not justify
switching away from manual attention for reference fidelity. Results do not
establish performance at larger resolutions or on other architectures.

Artifacts are in `.cache/disco-sage-benchmark` and `.cache/disco-sage-parity`:
raw JSON, timing samples, observed package versions, dispatch events, repeat
controls, source PNGs, and side-by-side comparisons. Reproduce with:

```powershell
.venv/Scripts/python.exe tools/benchmark_disco_attention.py --candidate sage `
  --checkpoint 'path/to/512x512_diffusion_uncond_finetune_008100.pt' `
  --output .cache/disco-sage-benchmark

.venv/Scripts/python.exe tools/benchmark_disco_attention.py --candidate sage `
  --checkpoint 'path/to/512x512_diffusion_uncond_finetune_008100.pt' `
  --deterministic-parity --repeats 2 --output .cache/disco-sage-parity
```
