"""Image utility functions: perlin noise, resampling, transforms."""

import io
import math

import cv2
import numpy as np
import requests
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps


def interp(t):
    """Smooth interpolation function for Perlin noise."""
    return 3 * t**2 - 2 * t**3


def perlin(width, height, scale=10, device=None):
    """Generate a single octave of Perlin noise."""
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)
    wx = 1 - interp(xs)
    wy = 1 - interp(ys)
    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))
    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)


def perlin_ms(octaves, width, height, grayscale, device=None):
    """Generate multi-scale Perlin noise."""
    out_array = [0.5] if grayscale else [0.5, 0.5, 0.5]
    for i in range(1 if grayscale else 3):
        scale = 2 ** len(octaves)
        oct_width = width
        oct_height = height
        for oct in octaves:
            p = perlin(oct_width, oct_height, scale, device)
            out_array[i] += p * oct
            scale //= 2
            oct_width *= 2
            oct_height *= 2
    return torch.cat(out_array)


def create_perlin_noise(octaves, width, height, grayscale, target_h, target_w, device=None):
    """Create Perlin noise image resized to target dimensions.

    Args:
        octaves: list of octave weights
        width: base grid width
        height: base grid height
        grayscale: if True, produce single-channel noise
        target_h: output height in pixels
        target_w: output width in pixels
        device: torch device
    """
    out = perlin_ms(octaves, width, height, grayscale, device=device)
    if grayscale:
        out = TF.resize(size=(target_h, target_w), img=out.unsqueeze(0))
        out = TF.to_pil_image(out.clamp(0, 1)).convert('RGB')
    else:
        out = out.reshape(-1, 3, out.shape[0] // 3, out.shape[1])
        out = TF.resize(size=(target_h, target_w), img=out)
        out = TF.to_pil_image(out.clamp(0, 1).squeeze())
    out = ImageOps.autocontrast(out)
    return out


def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    """Lanczos resampling of a 4D tensor to a target (h, w) size."""
    n, c, h, w = input.shape
    dh, dw = size

    input = input.reshape([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.reshape([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


def fetch(url_or_path):
    """Fetch content from a URL or local file path, returning a file-like object."""
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


def read_image_workaround(path):
    """Read image via OpenCV and convert BGR to RGB."""
    im_tmp = cv2.imread(path)
    return cv2.cvtColor(im_tmp, cv2.COLOR_BGR2RGB)


def img2tensor(img, size=None, interp=Image.LANCZOS):
    """Convert PIL image to CUDA float tensor [1, C, H, W]."""
    img = img.convert('RGB')
    if size:
        img = img.resize(size, interp)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float()[None, ...]


def make_even(x):
    return x if (x % 2 == 0) else x + 1


def fit(img, maxsize=512, interp=Image.LANCZOS):
    """Resize image so its largest dimension is at most maxsize."""
    maxdim = max(*img.size)
    if maxdim > maxsize:
        ratio = maxsize / maxdim
        x, y = img.size
        size = (make_even(int(x * ratio)), make_even(int(y * ratio)))
        img = img.resize(size, interp)
    return img
