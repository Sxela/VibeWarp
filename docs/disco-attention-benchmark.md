# Disco manual attention versus SDPA

Measured on 2026-11-05 with an RTX 5090, Windows, Python 3.13.15,
PyTorch 2.11.0+cu128. This GPU was shared with other workloads; the small
timing difference below is not evidence of a reliable speedup.

## Production timing

One resident 512px checkpoint, 512×512 output, 30 DDIM steps, full regeneration,
secondary denoiser, OpenAI ViT-B/32, two global and six inner cutouts, normal
augmentations, seeds 42 and 1234, three repeats per seed/backend. Both paths
were warmed with a complete guided render. Measurement order alternated.
Model loading, PNG writing, preview I/O, flow, turbo, LPIPS and frame guidance
were excluded. This is a one-encoder test, not the default three-encoder ensemble.

| Measurement | Manual | SDPA |
|---|---:|---:|
| U-Net forward, median | 171.67 ms | 177.54 ms |
| Guided frame, median | 9.132 s | 8.974 s |
| Guided frame range | 8.025–9.267 s | 8.607–9.127 s |
| Peak allocated GPU memory, guided frame | 3.049 GiB | 3.049 GiB |
| Peak allocated GPU memory, U-Net forward | 2.859 GiB | 2.859 GiB |

The observed full-frame difference is 1.8%; it is inside the variability of
these shared-GPU measurements. No reduction in the whole-render allocated
memory peak was measured. This does not rule out lower memory use inside an
individual attention block: a different operation can still set the overall peak.

All 16 U-Net attention modules were switched and verified. Profiling showed
the original einsum/softmax path for manual attention and
`aten::_scaled_dot_product_efficient_attention` for SDPA. A separate runtime
probe confirmed `torch.backends.cuda.is_flash_attention_available() == False`;
forcing FlashAttention reported that this Torch build was compiled without it.
This measures the available memory-efficient SDPA kernel, not FlashAttention.

## Image parity

A single real U-Net prediction differed by mean absolute error 0.00018755,
maximum 0.00167441, relative L2 0.00024253 between attention implementations.
Small per-evaluation differences can grow across a guided sampling trajectory.

Ordinary CUDA sampling also produced substantial differences between repeated
renders of the same seed and backend. Those production images cannot isolate
the effect of changing attention. The repeat control is part of the benchmark,
so these differences were not mistakenly classified as attention-only drift.

A second, **correctness-only** run enabled deterministic algorithms and moved
the differentiable interpolation, affine grid sampling and padding operations
to CPU. All model networks stayed on GPU. This preserves the selected spatial
operations while avoiding their nondeterministic CUDA backward implementations.
The modified spatial execution means its timings are not production timings.

Both same-backend repeats were byte-identical for each seed in that control.
Cross-backend results were:

| Seed | Manual–SDPA MAE, 8-bit levels | SSIM | PSNR |
|---|---:|---:|---:|
| 42 | 13.6244 / 255 | 0.84135 | 16.09 dB |
| 1234 | 12.3277 / 255 | 0.84811 | 16.65 dB |

The broad composition and color arrangement remain recognizable, but local
details differ visibly. Full-render pixel parity is not established by the
small-tensor forward/backward tests. For this build, manual attention remains
the reference choice when matching its rendering behavior matters; there is
no demonstrated performance advantage sufficient to offset that difference.

These are implementation comparisons, not a quality assessment of either model.
The test uses a small guidance ensemble and full regeneration from a neutral init.

## Reproduce

Use existing weights; the benchmark never downloads them. Adjust the checkpoint
path and models root for the installation:

```powershell
.venv/Scripts/python.exe tools/benchmark_disco_attention.py `
  --checkpoint 'path/to/512x512_diffusion_uncond_finetune_008100.pt' `
  --models-root models --output .cache/disco-attention-benchmark

.venv/Scripts/python.exe tools/benchmark_disco_attention.py `
  --checkpoint 'path/to/512x512_diffusion_uncond_finetune_008100.pt' `
  --models-root models --deterministic-parity --repeats 2 `
  --output .cache/disco-attention-parity
```

Each output directory contains raw timing/memory samples, kernel dispatch,
tensor and image metrics in `results.json`, all seed/repeat PNGs, `comparison.png`,
and `report.md`. The comparison columns are manual, SDPA, and absolute RGB
difference amplified four times. Production attention defaults were not changed
by this benchmark.
