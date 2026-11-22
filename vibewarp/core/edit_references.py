"""Shared visual-role helpers for multi-reference edit backends."""

from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

from vibewarp.config import EditReferenceConfig


REFERENCE_SOURCE_LABELS = {
    'raw': 'raw frame',
    'previous': 'previous stylized frame',
    'warped': 'previous stylized + warp + consistency',
    'upload': 'uploaded image',
    'none': 'none',
}
MAX_EDIT_REFERENCES = {'flux': 10, 'hidream': 10, 'qwen': 3, 'mage': 16, 'comfy_image': 5}


def add_reference_label(
    image: Image.Image, text: str, opacity: float = 0.7,
    font_scale: float = 1.0, badge_height: float = 0.12,
) -> Image.Image:
    """Bake a resolution-scaled tag with adjustable band height into model input."""
    base = image.convert('RGBA')
    width, height = base.size
    band_height = min(height, max(32, round(height * badge_height)))
    font_size = max(1, round(min(width, height) * 0.08 * font_scale))
    def load_font(size):
        for name in ('DejaVuSans-Bold.ttf', 'segoeuib.ttf', 'arialbd.ttf'):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                pass
        # Pillow's embedded scalable font: never silently fall back to 10px.
        return ImageFont.load_default(size=size)
    alpha = round(max(0.0, min(1.0, opacity)) * 255)
    overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    padding = max(1, round(min(width, band_height) * 0.1))
    while True:
        font = load_font(font_size)
        stroke = max(0, round(font_size * 0.02))
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
        text_width, text_height = bbox[2]-bbox[0], bbox[3]-bbox[1]
        fit = min(1.0, max(1, width-2*padding)/max(1, text_width),
                  max(1, band_height-2*padding)/max(1, text_height))
        if fit >= 1 or font_size <= 1:
            break
        font_size = max(1, min(font_size-1, int(font_size*fit)))
    top = height - band_height
    draw.rectangle((0, top, width, height), fill=(0, 0, 0, alpha))
    draw.text(
        ((width - text_width) / 2 - bbox[0], top + (band_height - text_height) / 2 - bbox[1]),
        text,
        fill=(255, 255, 255, alpha),
        font=font,
        stroke_width=stroke,
        stroke_fill=(255, 255, 255, alpha),
    )
    return Image.alpha_composite(base, overlay).convert('RGB')


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


def resolve_reference_images(
    specs: list[EditReferenceConfig],
    *,
    raw_image: Image.Image,
    previous_image: Image.Image | None,
    warped_image: Image.Image | None,
    size: tuple[int, int],
    max_references: int = 10,
    label_opacity: float = 0.7,
    label_font_scale: float = 1.0,
    label_badge_height: float = 0.12,
) -> list[Image.Image]:
    """Resolve an ordered reference configuration for one render frame.

    A temporal source in required slot 1 falls back to the current raw frame.
    Unavailable temporal sources in optional slots are omitted instead of
    sending duplicate raw-frame placeholders. Later available slots compact
    into the ordered model input list.
    """
    if not specs:
        raise ValueError('At least one edit reference is required')
    resolved: list[Image.Image] = []
    for index, spec in enumerate(specs[:max_references]):
        if index > 0 and spec.source == 'none':
            break
        if spec.source == 'raw':
            image = raw_image
        elif spec.source == 'previous':
            if previous_image is None and index > 0:
                continue
            image = previous_image or raw_image
        elif spec.source == 'warped':
            if warped_image is None and index > 0:
                continue
            image = warped_image or raw_image
        elif spec.source == 'upload':
            if index == 0:
                raise ValueError('Reference image 1 cannot be an upload')
            if not spec.image_path or not os.path.isfile(spec.image_path):
                raise FileNotFoundError(
                    f'Uploaded reference image does not exist: {spec.image_path}')
            with Image.open(spec.image_path) as opened:
                image = opened.convert('RGB')
        else:
            raise ValueError(f'Unsupported edit reference source: {spec.source!r}')
        image = image.convert('RGB')
        if image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        if spec.label:
            image = add_reference_label(
                image, f'@Image{len(resolved) + 1}', opacity=label_opacity,
                font_scale=label_font_scale, badge_height=label_badge_height)
        resolved.append(image)
    if not resolved:
        raise ValueError('At least one edit reference is required')
    return resolved


def reference_prompt(prompt: str, instruction: str, count: int) -> str:
    """Add the configured role instruction only for multi-reference renders."""
    instruction = instruction.strip()
    if count <= 1 or not instruction:
        return prompt
    return f'{instruction}\n\n{prompt}' if prompt else instruction
