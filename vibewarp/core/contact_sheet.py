"""Pure layout helpers for multi-frame image-edit contact sheets."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Iterable, Sequence

from PIL import Image


# Kept in sync with ComfyUI's FluxKontextImageScale preferred resolutions.
# Building at one of these sizes prevents that node from cropping a wide/tall
# multi-frame sheet before Qwen sees it.
PREFERRED_KONTEXT_RESOLUTIONS: tuple[tuple[int, int], ...] = (
    (672, 1568), (688, 1504), (720, 1456), (752, 1392),
    (800, 1328), (832, 1248), (880, 1184), (944, 1104),
    (1024, 1024),
    (1104, 944), (1184, 880), (1248, 832), (1328, 800),
    (1392, 752), (1456, 720), (1504, 688), (1568, 672),
)


@dataclass(frozen=True)
class SheetLayout:
    """Exact Qwen canvas geometry for one contact sheet."""

    canvas_size: tuple[int, int]
    rows: int
    columns: int
    gutter: int
    cell_size: tuple[int, int]
    boxes: tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True)
class SheetCell:
    """One temporal role placed in a contact-sheet cell."""

    frame_num: int | None
    role: str
    accept: bool


def _shape_for(layout: str, count: int, source_ratio: float) -> tuple[int, int]:
    if count < 1 or count > 3:
        raise ValueError('Contact sheets support one to three images')
    if layout == 'adaptive':
        if count == 1:
            return 1, 1
        if count == 2:
            return ((2, 1) if source_ratio >= 1 else (1, 2))
        if source_ratio > 1.2:
            return 3, 1
        if source_ratio < (1 / 1.2):
            return 1, 3
        return 2, 2
    if layout == 'row':
        return 1, count
    if layout == 'column':
        return count, 1
    if layout == 'grid':
        return (1, 1) if count == 1 else (2, 2)
    raise ValueError(f'Unknown Qwen contact-sheet layout: {layout!r}')


def choose_sheet_layout(
    source_size: tuple[int, int],
    count: int,
    layout: str = 'adaptive',
    gutter: int = 4,
    preferred_resolutions: Sequence[tuple[int, int]] = PREFERRED_KONTEXT_RESOLUTIONS,
) -> SheetLayout:
    """Choose an exact backend-preferred canvas with equal-sized cells."""
    source_width, source_height = source_size
    if source_width < 1 or source_height < 1:
        raise ValueError('Source dimensions must be positive')
    if gutter < 0:
        raise ValueError('Contact-sheet gutter cannot be negative')
    source_ratio = source_width / source_height
    rows, columns = _shape_for(layout, count, source_ratio)

    candidates: list[tuple[float, int, int, int, int]] = []
    for canvas_width, canvas_height in preferred_resolutions:
        usable_width = canvas_width - gutter * (columns - 1)
        usable_height = canvas_height - gutter * (rows - 1)
        if usable_width < columns or usable_height < rows:
            continue
        if usable_width % columns or usable_height % rows:
            continue
        cell_width = usable_width // columns
        cell_height = usable_height // rows
        aspect_error = abs(log((cell_width / cell_height) / source_ratio))
        candidates.append((
            aspect_error, -cell_width * cell_height,
            canvas_width, canvas_height, cell_width,
        ))
    if not candidates:
        raise ValueError(
            f'No preferred model canvas supports {rows}x{columns} cells '
            f'with a {gutter}px gutter')

    _, _, canvas_width, canvas_height, cell_width = min(candidates)
    cell_height = (
        canvas_height - gutter * (rows - 1)) // rows
    boxes = []
    for row in range(rows):
        for column in range(columns):
            left = column * (cell_width + gutter)
            top = row * (cell_height + gutter)
            boxes.append((left, top, left + cell_width, top + cell_height))
    return SheetLayout(
        canvas_size=(canvas_width, canvas_height),
        rows=rows,
        columns=columns,
        gutter=gutter,
        cell_size=(cell_width, cell_height),
        boxes=tuple(boxes),
    )


def assemble_contact_sheet(
    images: Sequence[Image.Image],
    layout: SheetLayout,
    *,
    background: tuple[int, int, int] = (127, 127, 127),
) -> Image.Image:
    """Resize images into the ordered cells of an exact-size RGB canvas."""
    if not images or len(images) > len(layout.boxes):
        raise ValueError('Image count does not fit the contact-sheet layout')
    sheet = Image.new('RGB', layout.canvas_size, background)
    for image, box in zip(images, layout.boxes):
        tile = image.convert('RGB').resize(
            layout.cell_size, Image.Resampling.LANCZOS)
        sheet.paste(tile, box[:2])
    return sheet


def split_contact_sheet(
    image: Image.Image,
    layout: SheetLayout,
    count: int,
) -> list[Image.Image]:
    """Split a decoded sheet using its exact input geometry."""
    if image.size != layout.canvas_size:
        raise ValueError(
            f'The edit backend changed contact-sheet size from {layout.canvas_size} to '
            f'{image.size}; cannot split tiles safely')
    if count < 1 or count > len(layout.boxes):
        raise ValueError('Invalid contact-sheet output count')
    rgb = image.convert('RGB')
    return [rgb.crop(box) for box in layout.boxes[:count]]


def sheet_manifest(
    layout: SheetLayout,
    cells: Iterable[SheetCell],
) -> dict:
    """Return a serializable description of one sheet submission."""
    cell_list = list(cells)
    if len(cell_list) > len(layout.boxes):
        raise ValueError('Cell manifest does not fit its layout')
    return {
        'canvas_size': list(layout.canvas_size),
        'rows': layout.rows,
        'columns': layout.columns,
        'gutter': layout.gutter,
        'cell_size': list(layout.cell_size),
        'cells': [
            {
                'frame_num': cell.frame_num,
                'role': cell.role,
                'accept': cell.accept,
                'box': list(layout.boxes[index]),
            }
            for index, cell in enumerate(cell_list)
        ],
    }
