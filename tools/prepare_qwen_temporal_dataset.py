"""Compile aligned Ditto video pairs into AI Toolkit Qwen edit samples.

Each output target uses three native Qwen reference images:
  Image 1: current raw frame
  Image 2: edited frame t-2
  Image 3: edited frame t-1
  target:  edited frame t

Ditto metadata entries must provide ``source_path``, ``edited_path``, and
``instruction``. JSON arrays, JSONL files, and directories containing either
format are accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import deque
from pathlib import Path
from typing import Iterable, Iterator


IMAGE_DIRS = ('target', 'raw_current', 'edited_prev2', 'edited_prev1')


def load_metadata(path: Path) -> list[dict]:
    """Read Ditto triplets from one JSON/JSONL file or a directory of them."""
    files = sorted(path.rglob('*.json')) + sorted(path.rglob('*.jsonl')) \
        if path.is_dir() else [path]
    items: list[dict] = []
    for file_path in files:
        if file_path.suffix.lower() == '.jsonl':
            with file_path.open(encoding='utf-8') as handle:
                data = [json.loads(line) for line in handle if line.strip()]
        else:
            with file_path.open(encoding='utf-8') as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                for key in ('data', 'items', 'annotations', 'samples'):
                    if isinstance(data.get(key), list):
                        data = data[key]
                        break
                else:
                    data = [data]
        if not isinstance(data, list):
            raise ValueError(f'Metadata must contain a list: {file_path}')
        for item in data:
            if not isinstance(item, dict):
                raise ValueError(f'Non-object metadata entry in {file_path}')
            missing = {
                key for key in ('source_path', 'edited_path', 'instruction')
                if not item.get(key)
            }
            if missing:
                raise ValueError(
                    f'Metadata entry in {file_path} is missing {sorted(missing)}')
            items.append(item)
    return items


def resolve_media_path(value: str, dataset_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else dataset_root / path


def split_for(source_path: str, validation_fraction: float, seed: int) -> str:
    """Assign every edit of a source video to the same deterministic split."""
    digest = hashlib.sha256(f'{seed}:{source_path}'.encode()).digest()
    value = int.from_bytes(digest[:8], 'big') / float(1 << 64)
    return 'validation' if value < validation_fraction else 'train'


def sample_name(source_path: str, edited_path: str, frame: int) -> str:
    identity = hashlib.sha256(
        f'{source_path}\0{edited_path}'.encode()).hexdigest()[:16]
    return f'{identity}_f{frame:06d}'


def temporal_caption(instruction: str) -> str:
    clean = ' '.join(str(instruction).split())
    return (
        f'Edit Image 1 only: {clean} Images 2 and 3 are the two preceding '
        'already-edited video frames in chronological order. Use them only to '
        'preserve persistent identity, style, materials, palette, lighting, and '
        'fine details across motion. Preserve Image 1\'s pose, action, camera '
        'view, geometry, and composition. Do not copy either preceding frame.'
    )


def resize_frame(frame, max_edge: int):
    if max_edge <= 0 or max(frame.shape[:2]) <= max_edge:
        return frame
    import cv2
    height, width = frame.shape[:2]
    scale = max_edge / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def ensure_layout(output: Path) -> None:
    for split in ('train', 'validation'):
        for name in IMAGE_DIRS:
            (output / split / name).mkdir(parents=True, exist_ok=True)


def write_sample(
    output: Path,
    split: str,
    name: str,
    frames: tuple,
    caption: str,
    jpeg_quality: int,
    resume: bool,
) -> bool:
    import cv2
    paths = [output / split / folder / f'{name}.jpg' for folder in IMAGE_DIRS]
    caption_path = output / split / 'target' / f'{name}.txt'
    if resume and caption_path.is_file() and all(path.is_file() for path in paths):
        return False
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    for path, frame in zip(paths, frames):
        if not cv2.imwrite(str(path), frame, params):
            raise RuntimeError(f'Could not write {path}')
    caption_path.write_text(caption + '\n', encoding='utf-8')
    return True


def compile_video_pair(
    item: dict,
    *,
    dataset_root: Path,
    output: Path,
    temporal_gap: int,
    sample_every: int,
    max_edge: int,
    jpeg_quality: int,
    validation_fraction: float,
    seed: int,
    resume: bool,
    remaining: int | None,
) -> Iterator[dict]:
    import cv2
    source = resolve_media_path(str(item['source_path']), dataset_root)
    edited = resolve_media_path(str(item['edited_path']), dataset_root)
    if not source.is_file() or not edited.is_file():
        raise FileNotFoundError(f'Missing video pair: {source} / {edited}')
    raw_capture = cv2.VideoCapture(str(source))
    edited_capture = cv2.VideoCapture(str(edited))
    if not raw_capture.isOpened() or not edited_capture.isOpened():
        raise RuntimeError(f'Could not open video pair: {source} / {edited}')
    history = deque(maxlen=2 * temporal_gap + 1)
    split = split_for(str(item['source_path']), validation_fraction, seed)
    caption = temporal_caption(str(item['instruction']))
    frame_index = 0
    emitted = 0
    try:
        while True:
            raw_ok, raw = raw_capture.read()
            edit_ok, edit = edited_capture.read()
            if not raw_ok or not edit_ok:
                break
            raw = resize_frame(raw, max_edge)
            edit = resize_frame(edit, max_edge)
            history.append((frame_index, edit.copy()))
            eligible = (
                frame_index >= 2 * temporal_gap
                and (frame_index - 2 * temporal_gap) % sample_every == 0
            )
            if eligible:
                prev2 = history[-1 - 2 * temporal_gap][1]
                prev1 = history[-1 - temporal_gap][1]
                name = sample_name(
                    str(item['source_path']), str(item['edited_path']), frame_index)
                frames = (edit, raw, prev2, prev1)
                wrote = write_sample(
                    output, split, name, frames, caption, jpeg_quality, resume)
                emitted += 1
                yield {
                    'name': name,
                    'split': split,
                    'source_path': str(source),
                    'edited_path': str(edited),
                    'source_key': str(item['source_path']),
                    'frame': frame_index,
                    'temporal_gap': temporal_gap,
                    'instruction': str(item['instruction']),
                    'written': wrote,
                }
                if remaining is not None and emitted >= remaining:
                    break
            frame_index += 1
    finally:
        raw_capture.release()
        edited_capture.release()


def prepare(args: argparse.Namespace) -> dict:
    metadata_path = Path(args.metadata).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    output = Path(args.output).resolve()
    if not 0 <= args.validation_fraction < 1:
        raise ValueError('--validation-fraction must be in [0, 1)')
    if args.temporal_gap < 1 or args.sample_every < 1:
        raise ValueError('--temporal-gap and --sample-every must be positive')
    ensure_layout(output)
    items = load_metadata(metadata_path)
    if args.max_videos:
        items = items[:args.max_videos]
    manifest: list[dict] = []
    failures: list[dict] = []
    for index, item in enumerate(items, start=1):
        remaining = (
            args.max_samples - len(manifest)
            if args.max_samples else None)
        if remaining is not None and remaining <= 0:
            break
        try:
            rows = compile_video_pair(
                item, dataset_root=dataset_root, output=output,
                temporal_gap=args.temporal_gap, sample_every=args.sample_every,
                max_edge=args.max_edge, jpeg_quality=args.jpeg_quality,
                validation_fraction=args.validation_fraction, seed=args.seed,
                resume=args.resume, remaining=remaining)
            manifest.extend(rows)
        except Exception as exc:
            failure = {'index': index, 'item': item, 'error': str(exc)}
            failures.append(failure)
            print(f"Skipping metadata item {index}: {exc}")
            if args.fail_fast:
                raise
    manifest_path = output / 'manifest.jsonl'
    with manifest_path.open('w', encoding='utf-8') as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    summary = {
        'metadata_items': len(items),
        'samples': len(manifest),
        'written': sum(bool(row['written']) for row in manifest),
        'train': sum(row['split'] == 'train' for row in manifest),
        'validation': sum(row['split'] == 'validation' for row in manifest),
        'failures': failures,
        'control_order': ['raw_current', 'edited_prev2', 'edited_prev1'],
    }
    (output / 'summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', required=True,
                        help='Ditto metadata JSON/JSONL file or directory')
    parser.add_argument('--dataset-root', required=True,
                        help='Root used to resolve metadata video paths')
    parser.add_argument('--output', required=True,
                        help='Output AI Toolkit dataset directory')
    parser.add_argument('--temporal-gap', type=int, default=1,
                        help='Frame gap between references (default: 1)')
    parser.add_argument('--sample-every', type=int, default=4,
                        help='Emit every Nth eligible target (default: 4)')
    parser.add_argument('--max-edge', type=int, default=1024,
                        help='Downscale frames to this maximum edge; 0 keeps source')
    parser.add_argument('--jpeg-quality', type=int, default=95)
    parser.add_argument('--validation-fraction', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-videos', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--resume', action='store_true',
                        help='Keep already-complete matching samples')
    parser.add_argument('--fail-fast', action='store_true')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = prepare(args)
    print(json.dumps(summary, indent=2))
    if not summary['samples']:
        print('No training samples were produced.')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
