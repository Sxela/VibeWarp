"""Render the checked-in AI Toolkit config template with absolute paths."""

from __future__ import annotations

import argparse
from pathlib import Path


TOKENS = {
    '__RUN_NAME__': 'run_name',
    '__TRAINING_DIR__': 'training_dir',
    '__DATASET_DIR__': 'dataset_dir',
    '__MODEL_DIR__': 'model_dir',
    '__ARA_PATH__': 'ara_path',
}


def render(template: Path, output: Path, values: dict[str, str]) -> None:
    text = template.read_text(encoding='utf-8')
    for token, argument in TOKENS.items():
        value = values[argument]
        text = text.replace(token, value.replace('\\', '/'))
    leftovers = [token for token in TOKENS if token in text]
    if leftovers:
        raise ValueError(f'Unresolved template tokens: {leftovers}')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--run-name', default='qwen_temporal_ditto_v1')
    parser.add_argument('--training-dir', required=True)
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--ara-path', required=True)
    args = parser.parse_args(argv)
    render(
        Path(args.template), Path(args.output),
        {key: str(getattr(args, key)) for key in TOKENS.values()},
    )
    print(Path(args.output).resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
