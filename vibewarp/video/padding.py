"""Disco's visible-content geometry and reproducible padded JPEG cache."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def resolve_padded_geometry(config):
    """Resolve visible dimensions before allocating the aligned working canvas."""
    video = config.video
    if video.visible_width > 0 and video.visible_height > 0:
        visible = (video.visible_width, video.visible_height)
    elif video.max_size > 0:
        from vibewarp.video.input import probe_video
        source = probe_video(video.video_init_path)
        scale = video.max_size / max(source['width'], source['height'])
        visible = tuple(max(2, round(source[key]*scale/2)*2) for key in ('width', 'height'))
    else:
        visible = (video.width, video.height)
    canvas = tuple(max(64, math.ceil(value/64)*64) for value in visible)
    video.visible_width, video.visible_height = visible
    video.width, video.height = canvas
    return visible, canvas


def prepare_padded_frames(source_folder, output_folder, visible, canvas):
    """Resize raw JPEGs, edge-pad and save at quality 95, as the notebook does.

    Raw files stay separate so resume and geometry changes never resize an
    already padded JPEG. Content hashes also detect replaced source frames.
    """
    source, output = Path(source_folder), Path(output_folder)
    if source.resolve() == output.resolve():
        raise ValueError('Raw and padded frame folders must be separate')
    if (len(visible) != 2 or len(canvas) != 2 or min(*visible) <= 0
            or any(v > c for v, c in zip(visible, canvas))):
        raise ValueError('Visible dimensions must fit inside the padded canvas')
    output.mkdir(parents=True, exist_ok=True)
    marker = Path(str(output)+'.padding.json')
    previous = json.loads(marker.read_text()) if marker.is_file() else {}
    geometry = [list(visible), list(canvas)]
    records = {}
    left, top = ((c-v)//2 for v, c in zip(visible, canvas))
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(source.glob('*.jpg')):
        target = output/path.name
        raw_hash = digest(path)
        old = previous.get('frames', {}).get(path.name, {})
        if (previous.get('geometry') == geometry and old.get('raw') == raw_hash
                and target.is_file() and old.get('padded') == digest(target)):
            records[path.name] = old
            continue
        with Image.open(path) as image:
            pixels = np.asarray(image.convert('RGB').resize(tuple(visible), Image.Resampling.LANCZOS))
        padded = np.pad(pixels, ((top, canvas[1]-visible[1]-top),
            (left, canvas[0]-visible[0]-left), (0, 0)), mode='edge')
        Image.fromarray(padded).save(target, quality=95)
        records[path.name] = {'raw': raw_hash, 'padded': digest(target)}
    if not records:
        raise ValueError(f'No raw JPEG frames found in {source}')
    # Exclude stale frames if a reused raw cache has been shortened.
    for path in output.glob('*.jpg'):
        if path.name not in records:
            path.unlink()
    marker.write_text(json.dumps({'geometry': geometry, 'frames': records}, indent=2))


def crop_visible(image, visible):
    """Remove centered canvas padding; leave the saved working frames intact."""
    width, height = visible
    if width <= 0 or height <= 0 or width > image.width or height > image.height:
        raise ValueError('Visible crop must fit inside the rendered frame')
    left, top = (image.width-width)//2, (image.height-height)//2
    return image.crop((left, top, left+width, top+height))
