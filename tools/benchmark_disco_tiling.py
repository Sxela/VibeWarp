"""Paired full-frame versus tiled Disco render benchmark using local weights."""
import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw
import torch

from benchmark_disco_attention import gpu_state, image_metrics, deterministic_spatial_gradients
from vibewarp.config import RunConfig
from vibewarp.core.disco import DiscoBackend
from vibewarp.render_events import render_event_scope
from vibewarp.disco_catalog import CLASSIC_ENCODERS, MODERN_ENCODERS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--models-root', type=Path, default=Path('models'))
    parser.add_argument('--size', type=int, default=2048)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--tile-size', type=int, default=768)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--modes', choices=['full', 'tiled'], nargs='+', default=['full', 'tiled'])
    parser.add_argument('--quality-only', action='store_true', help='Skip warmup and permit one repeat; not a performance benchmark')
    parser.add_argument('--noise-cap-percent',type=float,default=0.)
    parser.add_argument('--deterministic-parity',action='store_true',help='Correctness-only CPU spatial backward, not production timing')
    parser.add_argument('--overview-schedule', default='0:2', help='Denoising progress percent:overview count pairs')
    parser.add_argument('--inner-schedule', default='0:6', help='Denoising progress percent:inner count pairs')
    parser.add_argument('--attention', choices=['manual', 'sdpa', 'sage'], default='sdpa')
    parser.add_argument('--encoders', nargs='+', choices=[*CLASSIC_ENCODERS, *MODERN_ENCODERS], default=['ViT-B-32'])
    parser.add_argument('--scale-schedule', default='0:100', help='Step:percent pairs, e.g. 0:25,10:50,20:100')
    parser.add_argument('--output', type=Path, default=Path('.cache/disco-tiling-2048'))
    args = parser.parse_args()
    if args.deterministic_parity:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    modes = tuple(dict.fromkeys(args.modes))
    if args.repeats < (1 if args.quality_only else 2):
        parser.error('At least two repeats are required')
    args.output.mkdir(parents=True, exist_ok=True)
    config = RunConfig(model_version='disco_512', model_path=str(args.models_root.resolve()))
    d = config.disco
    d.checkpoint_path = str(args.checkpoint.resolve())
    d.steps, d.attention, d.style_strength = args.steps, args.attention, 1.
    d.guidance_noise_cap_percent = args.noise_cap_percent
    d.encoders, d.init_scale, d.image_scale = args.encoders, 0, 0
    cut_schedule = lambda text: {int(k):int(v) for k,v in (pair.split(':') for pair in text.split(','))}
    d.cut_overview, d.cut_innercut = cut_schedule(args.overview_schedule), cut_schedule(args.inner_schedule)
    d.preview_interval, d.inpaint = 0, False
    config.diffusion.sampler_tile_size = args.tile_size
    if args.scale_schedule.lstrip().startswith('{'):
        pairs = json.loads(args.scale_schedule).items()
    else:
        pairs = (pair.split(':') for pair in args.scale_schedule.split(','))
    schedule = {int(k):float(v) for k, v in pairs}
    from vibewarp.config_io import validate_config
    config.diffusion.sampler_scale_schedule = schedule
    errors = validate_config(config, require_inputs=False)
    if errors:
        parser.error(str(errors))
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    report = {'device':torch.cuda.get_device_name(), 'torch':torch.__version__, 'config':asdict(config),
        'quality_only':args.quality_only, 'sampling_state':'persistent native-resolution Disco stages',
        'deterministic_parity':args.deterministic_parity,
        'size':args.size, 'allocator_config':os.environ.get('PYTORCH_ALLOC_CONF', 'default'),
        'gpu_at_start':gpu_state(), 'samples':[], 'note':
        'Same seed 42 and weights; timings exclude model loading, warmup and PNG writing. '
        'Tiling changes model context; no lossless image parity claim. '
        'Production spatial backward is nondeterministic, so repeated images may drift. '
        'CUDA async allocator memory counters describe that allocator and are not directly comparable to native reservations.'}
    def save():
        (args.output/'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    save()
    backend = DiscoBackend(config)
    image = Image.new('RGB', (args.size, args.size), (127,127,127))
    prompt = ('an impressionist oil painting of a quiet riverside village, '
              'flower gardens, reflections in the water, vivid colors, expressive brush strokes')
    def render(mode, seed):
        config.diffusion.tiled_sampler = mode == 'tiled'
        config.diffusion.sampler_scale_schedule = schedule if mode == 'tiled' else {0:100.}
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        cap_events=[]
        def progress(event):
            if 'guidance_cap_scale' in event:
                cap_events.append({key:event[key] for key in ('step','guidance_cap_scale','guidance_update_rms','guidance_noise_budget_rms')})
            step = event.get('step', 0)
            if step == 1 or step % 5 == 0:
                print(f'{mode} step {step}/{args.steps}: {time.perf_counter()-started:.1f}s', flush=True)
        with render_event_scope(progress),deterministic_spatial_gradients(args.deterministic_parity):
            result = backend.render(config, image, prompt, '', seed)
        torch.cuda.synchronize()
        return result, {'mode':mode, 'seconds':time.perf_counter()-started, 'guidance_cap_events':cap_events,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
            'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3}
    for mode in (() if args.quality_only else modes):
        print(f'Warmup: {mode}', flush=True)
        _, sample = render(mode, 999)
        print(json.dumps(sample), flush=True)
    for repeat in range(args.repeats):
        for mode in (modes if repeat % 2 == 0 else tuple(reversed(modes))):
            print(f'Measuring {mode}, repeat {repeat}', flush=True)
            result, sample = render(mode, 42)
            result.save(args.output/f'{mode}_{repeat}.png')
            sample.update(repeat=repeat, gpu_after=gpu_state())
            report['samples'].append(sample)
            print(json.dumps(sample), flush=True)
            save()
    report['summary'] = {}
    for mode in modes:
        samples = [s for s in report['samples'] if s['mode'] == mode]
        report['summary'][mode] = {
            'median_seconds':statistics.median(s['seconds'] for s in samples),
            'max_allocated_gib':max(s['peak_allocated_gib'] for s in samples),
            'max_reserved_gib':max(s['peak_reserved_gib'] for s in samples)}
    load = lambda mode, repeat: Image.open(args.output/f'{mode}_{repeat}.png').convert('RGB')
    if 'full' in modes and 'tiled' in modes:
        report['image_comparison'] = image_metrics(load('full',0),load('tiled',0))
    if args.repeats > 1:
        report['repeat_variability'] = {mode:image_metrics(load(mode,0),load(mode,1)) for mode in modes}
    sheet = Image.new('RGB', (1024*len(modes),1056), '#171b22')
    draw = ImageDraw.Draw(sheet)
    for column, mode in enumerate(modes):
        draw.text((column*1024+12,8), mode, fill='white')
        sheet.paste(load(mode,0).resize((1024,1024)), (column*1024,32))
    sheet.save(args.output/'comparison.png')
    save()
    print(json.dumps(report['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
