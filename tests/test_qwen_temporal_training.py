import json
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image

from tools.prepare_qwen_temporal_dataset import (
    load_metadata, prepare, sample_name, split_for, temporal_caption)
from tools.render_qwen_temporal_config import render
from tools.validate_qwen_temporal_dataset import validate_dataset
from tools.validate_qwen_temporal_lora import temporal_prompt


def test_load_ditto_metadata_accepts_json_and_jsonl(tmp_path):
    folder = tmp_path / 'metadata'
    folder.mkdir()
    first = {
        'source_path': 'source/a.mp4',
        'edited_path': 'edited/a.mp4',
        'instruction': 'Make it watercolor.',
    }
    second = {
        'source_path': 'source/b.mp4',
        'edited_path': 'edited/b.mp4',
        'instruction': 'Make it ink.',
    }
    (folder / 'a.json').write_text(
        json.dumps({'items': [first]}), encoding='utf-8')
    (folder / 'b.jsonl').write_text(
        json.dumps(second) + '\n', encoding='utf-8')

    assert load_metadata(folder) == [first, second]


def test_split_is_stable_per_source_video_and_names_include_edit_identity():
    assert split_for('source/a.mp4', 0.2, 42) == \
        split_for('source/a.mp4', 0.2, 42)
    assert sample_name('source/a.mp4', 'edited/a.mp4', 9) != \
        sample_name('source/a.mp4', 'edited/b.mp4', 9)


def test_temporal_caption_assigns_current_raw_image_first():
    caption = temporal_caption('Turn it into a painting.\nKeep the person.')

    assert caption.startswith(
        'Edit Image 1 only: Turn it into a painting. Keep the person.')
    assert 'Images 2 and 3 are the two preceding already-edited' in caption
    assert 'Do not copy either preceding frame' in caption


def _write_video(path: Path, colors: list[tuple[int, int, int]]):
    import cv2
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*'MJPG'), 8, (32, 24))
    assert writer.isOpened()
    try:
        for color in colors:
            frame = np.full((24, 32, 3), color, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_prepare_compiles_aligned_rolling_video_samples(tmp_path):
    source = tmp_path / 'source.avi'
    edited = tmp_path / 'edited.avi'
    _write_video(source, [(index * 10, 0, 0) for index in range(6)])
    _write_video(edited, [(0, index * 10, 0) for index in range(6)])
    metadata = tmp_path / 'metadata.json'
    metadata.write_text(json.dumps([{
        'source_path': source.name,
        'edited_path': edited.name,
        'instruction': 'Make it green.',
    }]), encoding='utf-8')
    output = tmp_path / 'dataset'
    args = Namespace(
        metadata=str(metadata), dataset_root=str(tmp_path), output=str(output),
        temporal_gap=1, sample_every=1, max_edge=0, jpeg_quality=95,
        validation_fraction=0.0, seed=42, max_videos=0, max_samples=0,
        resume=False, fail_fast=True,
    )

    summary = prepare(args)

    assert summary['samples'] == 4
    assert summary['train'] == 4
    for folder in ('target', 'raw_current', 'edited_prev2', 'edited_prev1'):
        assert len(list((output / 'train' / folder).glob('*.jpg'))) == 4
    assert validate_dataset(output)['valid'] is True


def _write_sample(root: Path, split: str, name: str):
    for index, folder in enumerate(
            ('target', 'raw_current', 'edited_prev2', 'edited_prev1')):
        path = root / split / folder
        path.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (32, 24), (index * 20, 10, 10)).save(
            path / f'{name}.jpg')
    (root / split / 'target' / f'{name}.txt').write_text(
        'Edit Image 1 only.\n', encoding='utf-8')


def test_dataset_validator_checks_matching_controls_and_video_leakage(tmp_path):
    _write_sample(tmp_path, 'train', 'train_sample')
    _write_sample(tmp_path, 'validation', 'validation_sample')
    rows = [
        {'name': 'train_sample', 'split': 'train', 'source_key': 'a'},
        {'name': 'validation_sample', 'split': 'validation', 'source_key': 'b'},
    ]
    (tmp_path / 'manifest.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')

    result = validate_dataset(tmp_path)

    assert result['valid'] is True
    assert result['counts'] == {'train': 1, 'validation': 1}


def test_dataset_validator_rejects_source_video_split_leak(tmp_path):
    _write_sample(tmp_path, 'train', 'train_sample')
    _write_sample(tmp_path, 'validation', 'validation_sample')
    rows = [
        {'name': 'train_sample', 'split': 'train', 'source_key': 'same'},
        {'name': 'validation_sample', 'split': 'validation', 'source_key': 'same'},
    ]
    (tmp_path / 'manifest.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')

    result = validate_dataset(tmp_path)

    assert result['valid'] is False
    assert any('leak' in error for error in result['errors'])


def test_config_renderer_replaces_all_training_paths(tmp_path):
    template = tmp_path / 'template.yaml'
    output = tmp_path / 'config.yaml'
    template.write_text(
        '__RUN_NAME__|__TRAINING_DIR__|__DATASET_DIR__|__MODEL_DIR__|__ARA_PATH__',
        encoding='utf-8')

    render(template, output, {
        'run_name': 'temporal',
        'training_dir': '/work/output',
        'dataset_dir': '/work/data',
        'model_dir': '/work/model',
        'ara_path': '/work/ara.safetensors',
    })

    assert output.read_text(encoding='utf-8') == \
        'temporal|/work/output|/work/data|/work/model|/work/ara.safetensors'


def test_recursive_validation_prompt_assigns_previous_frames_as_images_2_and_3():
    prompt = temporal_prompt('Make it watercolor.', 3)

    assert prompt.startswith('Edit Image 1 only: Make it watercolor.')
    assert 'Images 2 and 3 are the two preceding' in prompt
    assert 'Do not copy a preceding frame into Image 1' in prompt
