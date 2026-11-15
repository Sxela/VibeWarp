"""Background mask sources, shared by render-time and assembly-time masking.

Whatever the source, a render materializes one mask per extracted frame into
``<run>/video_frames_alpha/{frame+1:06d}.png`` at the render canvas size. Render
compositing (before/after diffusion) and video assembly both read that folder, so the
two stages can never disagree about which matte they use, and re-assembling a run from
History needs neither the RVM cache nor a GPU.

Sources:

- ``RVM matting``  — RobustVideoMatting on the input video (CUDA, shared RVM cache).
- ``mask video``   — a user matte video: one frame per *source* video frame.
- ``image folder`` — user matte images, sorted by name: one per *source* video frame.

User sources follow the input video frame for frame, so ``extract_nth_frame`` applies to
them exactly as it does to the input video. Light pixels are foreground.
"""
from __future__ import annotations

import os
import re
import shutil
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

RVM = 'RVM matting'
VIDEO = 'mask video'
FOLDER = 'image folder'
MASK_SOURCES = (RVM, VIDEO, FOLDER)
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')


def wanted(config) -> bool:
    """Any stage composites with a background mask."""
    return config.mask.use_background_mask or config.video_assembly.use_background_mask_video


def uses_rvm(config) -> bool:
    return config.mask.mask_source == RVM


def run_mask_folder(video_frames_folder) -> str:
    return str(video_frames_folder) + '_alpha'


def mask_file(folder, frame) -> str | None:
    """The run mask for zero-based `frame`: PNG as written by `materialize`, or the
    notebook's JPEG naming for masks placed there by hand."""
    for extension in ('.png', '.jpg'):
        path = os.path.join(folder, f'{frame + 1:06d}{extension}')
        if os.path.exists(path):
            return path
    return None


def _natural_key(name):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', name)]


def folder_images(folder) -> list:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f'Mask folder does not exist: {folder}')
    return sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS),
                  key=lambda p: _natural_key(p.name))


def _extracted_indices(video_frames_folder) -> list:
    indices = []
    for name in os.listdir(video_frames_folder):
        stem, extension = os.path.splitext(name)
        if extension.lower() in ('.jpg', '.jpeg', '.png') and stem.isdigit():
            indices.append(int(stem) - 1)
    return sorted(indices)


def _extract_mask_video(config, indices, folder):
    """Extract the matte video with the input video's stride and numbering."""
    from vibewarp.video.input import extract_frames
    stride = config.video.extract_nth_frame
    shutil.rmtree(folder, ignore_errors=True)
    extract_frames(video_path=config.mask.mask_path, output_dir=str(folder), nth_frame=stride,
                   start_frame=indices[0] * stride, end_frame=indices[-1] * stride,
                   start_number=indices[0] + 1)
    missing = [i for i in indices if not (Path(folder) / f'{i + 1:06d}.jpg').is_file()]
    if missing:
        raise ValueError(
            f'Mask video {config.mask.mask_path} has no frame for render frame {missing[0] + 1} '
            '(it must have one frame per input video frame)')
    return lambda index: Path(folder) / f'{index + 1:06d}.jpg'


def _folder_source(config, last_index):
    images = folder_images(config.mask.mask_path)
    stride = config.video.extract_nth_frame
    needed = last_index * stride
    if needed >= len(images):
        raise ValueError(
            f'Mask folder {config.mask.mask_path} has {len(images)} images, but render frame '
            f'{last_index + 1} needs image {needed + 1} (one image per input video frame)')
    return lambda index: images[index * stride]


def _alpha_from_file(path, config, size):
    import torch
    from vibewarp.video.raw_processing import fit_alpha
    with Image.open(path) as opened:
        values = np.asarray(opened.convert('L'), dtype=np.float32) / 255
    return fit_alpha(torch.from_numpy(values)[None, None], config, size)


def _to_image(alpha):
    # Same 8-bit quantization the render used when it read RVM alpha directly.
    return Image.fromarray(alpha[0, 0].mul(255).round().byte().numpy())


def materialize(config, video_frames_folder, batch_folder, cache=None) -> str | None:
    """Write the run's masks for every extracted frame. Returns the folder, or None."""
    if not wanted(config):
        return None
    indices = _extracted_indices(video_frames_folder)
    if not indices:
        return None
    size = (config.video.height, config.video.width)
    source = config.mask.mask_source
    if source == RVM:
        from vibewarp.video.raw_processing import load_alpha
        if cache is None or cache.alpha is None:
            raise RuntimeError('RVM mask source requires the prepared RVM cache')
        alpha_for = lambda index: load_alpha(config, cache, index, size, device='cpu')
    else:
        if source == VIDEO:
            path_for = _extract_mask_video(config, indices, Path(batch_folder) / 'mask_video_frames')
        else:
            path_for = _folder_source(config, indices[-1])
        alpha_for = lambda index: _alpha_from_file(path_for(index), config, size)
    target = Path(run_mask_folder(video_frames_folder))
    target.mkdir(parents=True, exist_ok=True)
    for index in indices:
        _to_image(alpha_for(index)).save(target / f'{index + 1:06d}.png')
    print(f'  Background masks ({source}): {len(indices)} frames -> {target}')
    return str(target)


def preview_mask(config, frame, size, scratch):
    """One user-source mask at `size` (height, width), for the raw-input preview."""
    if config.mask.mask_source == FOLDER:
        path = _folder_source(config, frame)(frame)
        return _to_image(_alpha_from_file(path, _unpadded(config), size))
    folder = Path(scratch) / ('mask-' + uuid.uuid4().hex)
    try:
        path = _extract_mask_video(config, [frame], folder)(frame)
        return _to_image(_alpha_from_file(path, _unpadded(config), size))
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def _unpadded(config):
    from dataclasses import replace
    return replace(config, model_version='control_multi_v15')
