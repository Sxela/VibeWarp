"""Run a trained temporal Qwen LoRA through VibeWarp's native Comfy graph.

The validation is frame-at-a-time and supports either recursive generated
anchors or teacher-forced ground-truth anchors. It saves frames, a preview MP4,
and simple reconstruction/temporal-change metrics when a target video is given.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from vibewarp.config import RunConfig
from vibewarp.core.qwen_comfy import ComfyQwenEditClient


def read_video(path: Path, max_frames: int = 0) -> tuple[list[Image.Image], float]:
    import cv2
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f'Could not open video: {path}')
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
    frames: list[Image.Image] = []
    try:
        while not max_frames or len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f'No frames decoded from {path}')
    return frames, fps


def temporal_prompt(instruction: str, reference_count: int) -> str:
    if reference_count == 1:
        return instruction
    if reference_count == 2:
        history = 'Image 2 is the preceding already-edited video frame.'
    else:
        history = (
            'Images 2 and 3 are the two preceding already-edited video frames '
            'in chronological order.')
    return (
        f'Edit Image 1 only: {instruction} {history} Use the preceding images '
        'only to preserve persistent identity, style, materials, palette, '
        'lighting, and fine details across motion. Preserve Image 1\'s pose, '
        'action, camera view, geometry, and composition. Do not copy a preceding '
        'frame into Image 1.'
    )


def metric_row(output: Image.Image, target: Image.Image,
               previous_output: Image.Image | None,
               previous_target: Image.Image | None) -> dict:
    size = target.size
    out = np.asarray(output.convert('RGB').resize(size), dtype=np.float32) / 255.0
    tgt = np.asarray(target.convert('RGB'), dtype=np.float32) / 255.0
    mse = float(np.mean((out - tgt) ** 2))
    row = {
        'mae': float(np.mean(np.abs(out - tgt))),
        'psnr': float(-10 * math.log10(max(mse, 1e-12))),
    }
    if previous_output is not None and previous_target is not None:
        prev_out = np.asarray(
            previous_output.convert('RGB').resize(size), dtype=np.float32) / 255.0
        prev_tgt = np.asarray(
            previous_target.convert('RGB').resize(size), dtype=np.float32) / 255.0
        row['temporal_delta_mae'] = float(np.mean(np.abs(
            (out - prev_out) - (tgt - prev_tgt))))
    return row


def write_video(frames: list[Image.Image], path: Path, fps: float) -> None:
    import cv2
    width, height = frames[0].size
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f'Could not create preview video: {path}')
    try:
        for image in frames:
            rgb = np.asarray(image.convert('RGB').resize((width, height)))
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def validate(args: argparse.Namespace) -> dict:
    source_frames, fps = read_video(Path(args.source_video), args.max_frames)
    target_frames = None
    if args.target_video:
        target_frames, _ = read_video(Path(args.target_video), args.max_frames)
        frame_count = min(len(source_frames), len(target_frames))
        source_frames = source_frames[:frame_count]
        target_frames = target_frames[:frame_count]
    if args.mode == 'teacher_forced' and target_frames is None:
        raise ValueError('--mode teacher_forced requires --target-video')

    config = RunConfig(model_version='qwen_image_edit_2511')
    qwen = config.qwen
    qwen.comfy_server_url = args.comfy_url
    qwen.comfy_unet_name = args.unet
    qwen.comfy_clip_name = args.clip
    qwen.comfy_vae_name = args.vae
    qwen.comfy_lora_name = args.lora
    qwen.comfy_timeout = args.timeout
    qwen.use_lightning_lora = True
    qwen.lora_strength = args.lora_strength
    qwen.steps = args.steps
    qwen.guidance_scale = args.guidance
    qwen.multi_reference_instruction = ''
    client = ComfyQwenEditClient(config)

    output_dir = Path(args.output).resolve()
    frames_dir = output_dir / args.mode
    frames_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Image.Image] = []
    metrics: list[dict] = []
    for index, raw in enumerate(source_frames):
        references = [raw]
        if index >= 1:
            history = target_frames if args.mode == 'teacher_forced' else outputs
            if index >= 2:
                references.append(history[index - 2])
            references.append(history[index - 1])
        prompt = temporal_prompt(args.instruction, len(references))
        output = client.render(
            config, raw, prompt, args.negative_prompt,
            args.seed if args.fixed_seed else args.seed + index,
            raw.width, raw.height, reference_images=references,
            debug_dir=str(output_dir / 'debug'), frame_num=index,
        ).convert('RGB').resize(raw.size, Image.Resampling.LANCZOS)
        output.save(frames_dir / f'{index:06d}.png')
        if target_frames is not None:
            row = metric_row(
                output, target_frames[index],
                outputs[-1] if outputs else None,
                target_frames[index - 1] if index else None)
            row['frame'] = index
            metrics.append(row)
        outputs.append(output)
        print(f'Validated frame {index + 1}/{len(source_frames)}')

    write_video(outputs, output_dir / f'{args.mode}.mp4', fps)
    aggregate = {}
    if metrics:
        for key in ('mae', 'psnr', 'temporal_delta_mae'):
            values = [row[key] for row in metrics if key in row]
            if values:
                aggregate[key] = float(np.mean(values))
    summary = {
        'mode': args.mode,
        'frames': len(outputs),
        'instruction': args.instruction,
        'lora': args.lora,
        'aggregate': aggregate,
        'per_frame': metrics,
    }
    (output_dir / f'{args.mode}_metrics.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-video', required=True)
    parser.add_argument('--target-video')
    parser.add_argument('--instruction', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('recursive', 'teacher_forced'),
                        default='recursive')
    parser.add_argument('--max-frames', type=int, default=24)
    parser.add_argument('--comfy-url', default='http://127.0.0.1:8188')
    parser.add_argument('--unet', required=True)
    parser.add_argument('--clip', required=True)
    parser.add_argument('--vae', required=True)
    parser.add_argument('--lora', required=True)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--guidance', type=float, default=4.0)
    parser.add_argument('--lora-strength', type=float, default=1.0)
    parser.add_argument('--negative-prompt', default=' ')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fixed-seed', action='store_true')
    parser.add_argument('--timeout', type=float, default=1200.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = validate(args)
    print(json.dumps(summary['aggregate'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
