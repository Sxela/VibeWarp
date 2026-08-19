"""Validate an AI Toolkit temporal multi-reference dataset before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


CONTROL_DIRS = ('raw_current', 'edited_prev2', 'edited_prev1')


def image_info(path: Path) -> tuple[tuple[int, int], str]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.size, image.mode


def validate_dataset(root: Path, max_errors: int = 100) -> dict:
    errors: list[str] = []
    counts: dict[str, int] = {}
    names_by_split: dict[str, set[str]] = {}
    for split in ('train', 'validation'):
        split_root = root / split
        target_dir = split_root / 'target'
        targets = sorted(target_dir.glob('*.jpg'))
        names_by_split[split] = {path.stem for path in targets}
        counts[split] = len(targets)
        for target in targets:
            caption = target.with_suffix('.txt')
            if not caption.is_file() or not caption.read_text(
                    encoding='utf-8').strip():
                errors.append(f'Missing/empty caption: {caption}')
                if len(errors) >= max_errors:
                    break
            paths = [target] + [
                split_root / folder / target.name for folder in CONTROL_DIRS]
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                errors.append(f'Missing matched images for {target.name}: {missing}')
                if len(errors) >= max_errors:
                    break
                continue
            try:
                infos = [image_info(path) for path in paths]
            except Exception as exc:
                errors.append(f'Unreadable sample {target.name}: {exc}')
                if len(errors) >= max_errors:
                    break
                continue
            if len({size for size, _ in infos}) != 1:
                errors.append(
                    f'Image-size mismatch for {target.name}: '
                    f'{[size for size, _ in infos]}')
            if any(mode != 'RGB' for _, mode in infos):
                errors.append(
                    f'Non-RGB sample {target.name}: {[mode for _, mode in infos]}')
            if len(errors) >= max_errors:
                break

    overlap = names_by_split['train'] & names_by_split['validation']
    if overlap:
        errors.append(f'{len(overlap)} sample names occur in both splits')

    manifest_path = root / 'manifest.jsonl'
    source_splits: dict[str, set[str]] = {}
    manifest_rows = 0
    if not manifest_path.is_file():
        errors.append(f'Missing manifest: {manifest_path}')
    else:
        with manifest_path.open(encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(
                        f'Invalid manifest line {line_number}: {exc}')
                    continue
                manifest_rows += 1
                source_splits.setdefault(row['source_key'], set()).add(row['split'])
        leaked = [key for key, splits in source_splits.items() if len(splits) > 1]
        if leaked:
            errors.append(
                f'{len(leaked)} source videos leak across train/validation splits')

    return {
        'root': str(root),
        'counts': counts,
        'manifest_rows': manifest_rows,
        'source_videos': len(source_splits),
        'errors': errors[:max_errors],
        'valid': not errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', help='Prepared dataset root')
    parser.add_argument('--max-errors', type=int, default=100)
    args = parser.parse_args(argv)
    result = validate_dataset(Path(args.dataset).resolve(), args.max_errors)
    print(json.dumps(result, indent=2))
    return 0 if result['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
