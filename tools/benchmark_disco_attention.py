"""Warm, paired manual/SDPA Disco benchmarks with visual parity artifacts.

Uses existing weights only. Run from the repository root, for example:
python tools/benchmark_disco_attention.py --checkpoint path/to/512_model.pt
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import datetime as dt
import gc
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from skimage.metrics import structural_similarity

from vibewarp.config import RunConfig
from vibewarp.core.disco import DiscoBackend
from vibewarp.core.disco_attention import PixelAttention, configure_attention
from vibewarp.vendor.guided_diffusion.unet import QKVAttention, QKVAttentionLegacy


@contextmanager
def deterministic_spatial_gradients(enabled):
    """Correctness-only control for CUDA's nondeterministic spatial backward.

    Preserve the same interpolation, padding and affine math, but evaluate
    those differentiable operations on CPU. Never use these timings to infer
    production speed. All other model computations remain on GPU.
    """
    if not enabled:
        yield
        return
    import torch.nn.functional as functional
    import torchvision.transforms._functional_tensor as vision_functional
    saved = {name: getattr(functional, name) for name in ('interpolate', 'grid_sample', 'pad')}
    old_vision_grid = vision_functional.grid_sample
    previous = torch.are_deterministic_algorithms_enabled()
    old_cudnn = torch.backends.cudnn.deterministic
    def wrap(name, function):
        def cpu_spatial(input, *args, **kwargs):
            if input.is_cuda and torch.is_grad_enabled() and input.requires_grad:
                if name == 'grid_sample':
                    if args:
                        args = (args[0].cpu(), *args[1:])
                    else:
                        kwargs['grid'] = kwargs['grid'].cpu()
                return function(input.cpu(), *args, **kwargs).to(input.device)
            return function(input, *args, **kwargs)
        return cpu_spatial
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        for name, function in saved.items():
            setattr(functional, name, wrap(name, function))
        vision_functional.grid_sample = functional.grid_sample
        yield
    finally:
        for name, function in saved.items():
            setattr(functional, name, function)
        vision_functional.grid_sample = old_vision_grid
        torch.use_deterministic_algorithms(previous)
        torch.backends.cudnn.deterministic = old_cudnn


def image_metrics(first, second):
    a, b = np.asarray(first), np.asarray(second)
    delta = a.astype(np.float64)-b.astype(np.float64)
    mse = float(np.square(delta).mean())
    return {
        'mae_8bit': float(np.abs(delta).mean()),
        'max_8bit': int(np.abs(delta).max()),
        'psnr_db': 10*math.log10(255**2/mse) if mse else None,
        'ssim': float(structural_similarity(a, b, channel_axis=-1, data_range=255)),
        'changed_channels_pct': float(np.count_nonzero(delta)/delta.size*100),
    }


def gpu_state():
    try:
        return subprocess.check_output([
            'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm',
            '--format=csv,noheader'], text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return 'unavailable'


def contact_sheet(output, seeds, comparisons, repeats, candidate='sdpa'):
    first = Image.open(output/f'manual_seed{seeds[0]}_repeat0.png')
    width, height = first.size
    sheet = Image.new('RGB', (width*3, (height+66)*len(seeds)), '#171b22')
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    except OSError:
        font = ImageFont.load_default()
    for row, seed in enumerate(seeds):
        y = row*(height+66)
        manual = Image.open(output/f'manual_seed{seed}_repeat0.png').convert('RGB')
        sdpa = Image.open(output/f'{candidate}_seed{seed}_repeat0.png').convert('RGB')
        diff = np.abs(np.asarray(manual).astype(np.int16)-np.asarray(sdpa).astype(np.int16))
        difference = Image.fromarray(np.minimum(diff*4, 255).astype(np.uint8))
        for col, (label, picture) in enumerate([
            (f'Manual | seed {seed}', manual), (f'{candidate.upper()} | same seed and settings', sdpa),
            ('Absolute RGB difference x4 (clipped)', difference)]):
            draw.text((col*width+12, y+8), label, font=font, fill='white')
            sheet.paste(picture, (col*width, y+34))
        metrics = comparisons[str(seed)][f'manual_vs_{candidate}']
        psnr = f"{metrics['psnr_db']:.2f}" if metrics['psnr_db'] is not None else 'inf'
        draw.text((12, y+height+38),
            f"MAE {metrics['mae_8bit']:.4f}/255 | SSIM {metrics['ssim']:.5f} | "
            f"PSNR {psnr} dB | {repeats} repeats per backend",
            font=font, fill='#bdc9d9')
    sheet.save(output/'comparison.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--models-root', type=Path, default=Path('models'))
    parser.add_argument('--output', type=Path, default=Path('.cache/disco-attention-benchmark'))
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--unet-repeats', type=int, default=8)
    parser.add_argument('--unet-iterations', type=int, default=5)
    parser.add_argument('--unet-only', action='store_true', help='Measure network forward passes without guided renders')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 1234])
    parser.add_argument('--candidate', choices=['sdpa', 'sage', 'flash'], default='sdpa')
    parser.add_argument('--deterministic-parity', action='store_true',
                        help='Correctness-only CPU spatial gradients; timings are not production speed')
    parser.add_argument('--prompt', default='an impressionist oil painting of a quiet riverside village, '
                        'flower gardens, reflections in the water, vivid colors, expressive brush strokes')
    args = parser.parse_args()
    if args.deterministic_parity:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if args.repeats < 2:
        parser.error('Use at least two repeats to estimate same-backend variability')
    if args.unet_repeats < 1 or args.unet_iterations < 1:
        parser.error('U-Net repeat and iteration counts must be positive')
    if not torch.cuda.is_available():
        parser.error('CUDA is required')
    package = {'sage':'sageattention', 'flash':'flash_attn'}.get(args.candidate)
    if package and importlib.util.find_spec(package) is None:
        parser.error(f'{package} must be installed; fallback timings are not valid benchmarks')
    for path in [args.checkpoint, args.models_root/'disco/secondary_model_imagenet_2.pth',
                 args.models_root/'disco/clip/ViT-B-32.pt']:
        if not path.is_file():
            parser.error(f'Missing weight file: {path}')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    config = RunConfig(model_version='disco_512', model_path=str(args.models_root.resolve()))
    config.flow.flow_warp = False
    config.disco.checkpoint_path = str(args.checkpoint.resolve())
    d = config.disco
    d.attention, d.steps, d.sampler, d.style_strength = 'manual', args.steps, 'ddim', 1.0
    d.encoders, d.init_scale, d.image_scale = ['ViT-B-32'], 0, 0
    d.cut_overview, d.cut_innercut = {0: 2}, {0: 6}
    d.skip_augs, d.preview_interval, d.inpaint = False, 0, False
    report = {
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'python': platform.python_version(),
        'torch_flash_compiled': torch.backends.cuda.is_flash_attention_available(),
        'config': asdict(config), 'size': args.size, 'seeds': args.seeds,
        'repeats': args.repeats, 'samples': [], 'unet_samples': [],
        'unet_repeats': args.unet_repeats, 'unet_iterations': args.unet_iterations,
        'unet_only': args.unet_only,
        'deterministic_parity': args.deterministic_parity,
        'candidate': args.candidate,
        'allocator_config': os.environ.get('PYTORCH_ALLOC_CONF', os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default')),
        'note': 'Shared GPU; utilization snapshots do not prove exclusive access. '
                'Wall time includes guidance and RGB conversion, excludes model load, warmup and PNG writing. '
                'No LPIPS, flow, turbo or preview I/O. One ViT-B/32 encoder, eight cutouts, secondary denoiser.',
    }
    candidate_calls = [0]
    if args.candidate == 'sage':
        import sageattention
        original_sage = sageattention.sageattn
        def counted_sage(*inputs, **kwargs):
            candidate_calls[0] += 1
            return original_sage(*inputs, **kwargs)
        sageattention.sageattn = counted_sage
        report['packages'] = {}
        for name in ('sageattention', 'triton-windows', 'triton'):
            try:
                report['packages'][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
    if args.deterministic_parity:
        report['note'] += ' DETERMINISTIC CORRECTNESS PASS: spatial gradient operations run on CPU. '
        report['note'] += 'These render timings are NOT representative of production performance.'
    def save():
        (output/'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    save()
    print('Loading one shared model; no weights are downloaded.', flush=True)
    backend = DiscoBackend(config)
    # Keep long, high-resolution runs observable without changing sampling math.
    sampler_name = 'ddim_sample_loop_progressive'
    sampler = getattr(backend.diffusion, sampler_name)
    def logged_sampler(*inputs, **kwargs):
        started = time.perf_counter()
        for index, result in enumerate(sampler(*inputs, **kwargs), 1):
            if index == 1 or index % 5 == 0:
                print(f'DDIM step {index}/{args.steps}: {time.perf_counter()-started:.1f}s', flush=True)
            yield result
    setattr(backend.diffusion, sampler_name, logged_sampler)
    original = [(parent, name, child) for parent in backend.model.modules()
                for name, child in parent.named_children()
                if isinstance(child, (QKVAttention, QKVAttentionLegacy))]
    assert original, 'No reference attention layers found'
    report['attention_layers'] = len(original)

    def select(mode):
        # configure_attention('manual') is intentionally a no-op in production.
        # Restore the actual reference modules before every benchmark switch.
        for parent, name, child in original:
            setattr(parent, name, child)
        configure_attention(backend.model, mode)
        count = sum(isinstance(m, PixelAttention) for m in backend.model.modules())
        assert count == (len(original) if mode != 'manual' else 0), (mode, count)

    generator = torch.Generator(device='cuda').manual_seed(912)
    x = torch.randn((1, 3, args.size, args.size), generator=generator, device='cuda')
    timestep = torch.tensor([500.0], device='cuda')
    tensors = {}
    for mode in ('manual', args.candidate):
        select(mode)
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            backend.model(x, timestep)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
                tensors[mode] = backend.model(x, timestep).float().cpu()
            report.setdefault('dispatch', {})[mode] = [event.key for event in profiler.key_averages()
                if any(key in event.key.lower() for key in ('attention', 'einsum', 'softmax'))]
            if mode == 'sage':
                assert candidate_calls[0] > 0, 'Sage was selected but never called'
                report['dispatch'][mode].append('sageattention.sageattn: direct calls observed')
    difference = tensors['manual']-tensors[args.candidate]
    report['unet_tensor_parity'] = {
        'mae': float(difference.abs().mean()), 'max': float(difference.abs().max()),
        'relative_l2': float(difference.norm()/tensors['manual'].norm().clamp_min(1e-12))}
    del tensors, difference
    for repeat in range(args.unet_repeats):
        for mode in (('manual', args.candidate) if repeat%2 == 0 else (args.candidate, 'manual')):
            select(mode)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.no_grad():
                for _ in range(args.unet_iterations):
                    prediction = backend.model(x, timestep)
                del prediction
            torch.cuda.synchronize()
            report['unet_samples'].append({'backend': mode, 'seconds': (time.perf_counter()-started)/args.unet_iterations,
                'peak_allocated_gib': torch.cuda.max_memory_allocated()/1024**3,
                'peak_reserved_gib': torch.cuda.max_memory_reserved()/1024**3})
            print('U-Net: '+json.dumps(report['unet_samples'][-1]), flush=True)
            save()
    if args.unet_only:
        report['unet_summary'] = {}
        for mode in ('manual', args.candidate):
            samples = [s for s in report['unet_samples'] if s['backend'] == mode]
            report['unet_summary'][mode] = {
                'median_seconds': statistics.median(s['seconds'] for s in samples),
                'max_allocated_gib': max(s['peak_allocated_gib'] for s in samples)}
        report['unet_summary']['speedup'] = (report['unet_summary']['manual']['median_seconds'] /
                                             report['unet_summary'][args.candidate]['median_seconds'])
        report['note'] += ' U-NET ONLY: no full guided renders or image parity measurements.'
        save()
        print(json.dumps(report['unet_summary'], indent=2), flush=True)
        return
    del x, timestep
    init = Image.new('RGB', (args.size, args.size), (127, 127, 127))
    for mode in ('manual', args.candidate):
        select(mode)
        gc.collect()
        torch.cuda.empty_cache()
        print(f'Full-guidance warmup: {mode}', flush=True)
        with deterministic_spatial_gradients(args.deterministic_parity):
            backend.render(config, init, args.prompt, '', 999)
    for repeat in range(args.repeats):
        for seed in args.seeds:
            for mode in (('manual', args.candidate) if repeat%2 == 0 else (args.candidate, 'manual')):
                select(mode)
                gc.collect()
                torch.cuda.empty_cache()
                before = gpu_state()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                resident = torch.cuda.memory_allocated()/1024**3
                calls_before = candidate_calls[0]
                started = time.perf_counter()
                with deterministic_spatial_gradients(args.deterministic_parity):
                    image = backend.render(config, init, args.prompt, '', seed)
                torch.cuda.synchronize()
                seconds = time.perf_counter()-started
                peak = torch.cuda.max_memory_allocated()/1024**3
                image.save(output/f'{mode}_seed{seed}_repeat{repeat}.png')
                sample = {'backend': mode, 'seed': seed, 'repeat': repeat, 'seconds': seconds,
                    'peak_allocated_gib': peak, 'incremental_peak_gib': peak-resident,
                    'peak_reserved_gib': torch.cuda.max_memory_reserved()/1024**3,
                    'gpu_before': before, 'gpu_after': gpu_state()}
                if args.candidate == 'sage':
                    sample['sage_calls'] = candidate_calls[0]-calls_before
                    assert sample['sage_calls'] == (len(original)*args.steps if mode == 'sage' else 0), sample
                report['samples'].append(sample)
                print(json.dumps(sample), flush=True)
                save()
    comparisons = {}
    for seed in args.seeds:
        load = lambda mode, repeat: Image.open(output/f'{mode}_seed{seed}_repeat{repeat}.png').convert('RGB')
        comparisons[str(seed)] = {f'manual_vs_{args.candidate}': image_metrics(load('manual', 0), load(args.candidate, 0)),
            'manual_repeat': image_metrics(load('manual', 0), load('manual', 1)),
            f'{args.candidate}_repeat': image_metrics(load(args.candidate, 0), load(args.candidate, 1))}
    report['image_parity'] = comparisons
    repeatable = all(metrics[key]['mae_8bit'] == 0
                     for metrics in comparisons.values() for key in ('manual_repeat', f'{args.candidate}_repeat'))
    report['repeatability_control_passed'] = repeatable
    if args.deterministic_parity and not repeatable:
        save()
        raise RuntimeError('Deterministic same-backend repeat control failed; do not infer attention parity')
    for collection, key in [('samples', 'render_summary'), ('unet_samples', 'unet_summary')]:
        report[key] = {}
        for mode in ('manual', args.candidate):
            samples = [s for s in report[collection] if s['backend'] == mode]
            seconds = [s['seconds'] for s in samples]
            report[key][mode] = {'median_seconds': statistics.median(seconds),
                'min_seconds': min(seconds), 'max_seconds': max(seconds),
                'max_allocated_gib': max(s['peak_allocated_gib'] for s in samples)}
        report[key]['speedup'] = report[key]['manual']['median_seconds']/report[key][args.candidate]['median_seconds']
    save()
    contact_sheet(output, args.seeds, comparisons, args.repeats, args.candidate)
    lines = ['# Disco attention benchmark', '', f"GPU: {report['device']}; PyTorch {report['torch']}.",
        '', f"{args.size}x{args.size}, {args.steps} DDIM steps, secondary guidance, ViT-B/32, 8 cutouts, "
        f"{len(args.seeds)} seeds x {args.repeats} repeats; warmed and alternating order.", '',
        f'| Test | Manual median | {args.candidate.upper()} median | Speedup |', '|---|---:|---:|---:|']
    if args.deterministic_parity:
        lines[0] = '# Disco attention: deterministic correctness run'
        lines[2:2] = ['**CPU spatial gradients enabled: render timings below are diagnostic, not production performance.**', '']
    for key, label in [('unet_summary', 'U-Net forward'), ('render_summary', 'Full guided render')]:
        summary = report[key]
        lines.append(f"| {label} | {summary['manual']['median_seconds']:.4f}s | "
                     f"{summary[args.candidate]['median_seconds']:.4f}s | {summary['speedup']:.3f}x |")
    lines += ['', 'Peak allocated GPU memory includes resident models; external processes are excluded from this allocator metric.', '',
        f'| Seed | Manual vs {args.candidate.upper()} MAE / 255 | SSIM | PSNR | Manual repeat MAE | {args.candidate.upper()} repeat MAE |',
        '|---|---:|---:|---:|---:|---:|']
    for seed, metrics in comparisons.items():
        pair = metrics[f'manual_vs_{args.candidate}']
        lines.append(f"| {seed} | {pair['mae_8bit']:.6f} | {pair['ssim']:.6f} | {pair['psnr_db']} | "
                     f"{metrics['manual_repeat']['mae_8bit']:.6f} | {metrics[f'{args.candidate}_repeat']['mae_8bit']:.6f} |")
    lines += ['', f'![Manual, {args.candidate.upper()}, and amplified absolute difference](comparison.png)', '', report['note'], '',
        'Raw timings, peaks, kernel dispatch and tensor parity are in [results.json](results.json).', '']
    if not repeatable:
        lines += ['**Same-backend repeats also drift. These production image differences cannot be attributed solely to attention.**', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'render': report['render_summary'], 'unet': report['unet_summary'],
                      'parity': comparisons, 'output': str(output)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
