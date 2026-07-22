"""Shared visual-role helpers for multi-reference edit backends."""

from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont


def add_reference_label(
    image: Image.Image, text: str
) -> Image.Image:
    """Overlay a high-contrast role label without changing image dimensions."""
    labeled = image.convert('RGB').copy()
    width, height = labeled.size
    band_height = max(24, round(height * 0.075))
    font_size = max(12, round(band_height * 0.48))
    try:
        font = ImageFont.truetype('DejaVuSans-Bold.ttf', font_size)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(labeled)
    top = height - band_height
    draw.rectangle((0, top, width, height), fill='black')
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    draw.text(
        ((width - text_width) / 2, top + (band_height - text_height) / 2 - bbox[1]),
        text,
        fill='white',
        font=font,
    )
    return labeled


def add_style_reference_label(
    image: Image.Image, text: str = 'style reference'
) -> Image.Image:
    """Backward-compatible convenience wrapper for the style role label."""
    return add_reference_label(image, text)


def save_debug_reference(
    image: Image.Image,
    debug_dir: str | None,
    backend: str,
    reference_number: int,
    frame_num: int | None,
) -> None:
    """Persist the exact image supplied to an edit model reference input."""
    if not debug_dir or frame_num is None:
        return
    os.makedirs(debug_dir, exist_ok=True)
    image.convert('RGB').save(os.path.join(
        debug_dir,
        f'{backend}_reference_{reference_number}_{frame_num:06d}.png',
    ))
