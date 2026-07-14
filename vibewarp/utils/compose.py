"""Image composition utilities (hstack, vstack)."""

from PIL import Image, ImageDraw


def hstack(images):
    """Horizontally stack a list of PIL images (or file paths).

    Each image gets a thin black border. All images are placed
    in a row at y=0 (top-aligned).

    Args:
        images: list of PIL Images or file path strings

    Returns:
        Combined PIL Image.
    """
    if isinstance(images[0], str):
        images = [Image.open(p).convert('RGB') for p in images]

    widths, heights = zip(*(im.size for im in images))
    for im in images:
        draw = ImageDraw.Draw(im)
        draw.rectangle(((0, 0), (im.size[0], im.size[1])), outline="black", width=3)

    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    return new_im


def vstack(images):
    """Vertically stack a list of PIL images (or file paths).

    Args:
        images: list of PIL Images or file path strings

    Returns:
        Combined PIL Image.
    """
    if isinstance(next(iter(images)), str):
        images = [Image.open(p).convert('RGB') for p in images]

    widths, heights = zip(*(im.size for im in images))
    total_height = sum(heights)
    max_width = max(widths)
    new_im = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]

    return new_im
