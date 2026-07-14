"""Tests for vibewarp.utils.image."""

import math

import numpy as np
import pytest
import torch
from PIL import Image

from vibewarp.utils.image import (
    create_perlin_noise,
    fit,
    img2tensor,
    interp,
    lanczos,
    make_even,
    perlin,
    perlin_ms,
    ramp,
    resample,
    sinc,
)


class TestInterp:
    def test_boundaries(self):
        assert interp(0) == 0.0
        assert interp(1) == 1.0

    def test_midpoint(self):
        assert interp(0.5) == 0.5

    def test_monotonic(self):
        vals = [interp(t / 10) for t in range(11)]
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1]


class TestSinc:
    def test_zero(self):
        x = torch.tensor([0.0])
        assert torch.allclose(sinc(x), torch.ones(1))

    def test_integer(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = sinc(x)
        assert torch.allclose(result, torch.zeros(3), atol=1e-6)


class TestLanczos:
    def test_sums_to_one(self):
        x = ramp(0.5, 2)
        kernel = lanczos(x, 2)
        assert abs(kernel.sum().item() - 1.0) < 1e-5


class TestRamp:
    def test_basic_shape(self):
        r = ramp(0.5, 2)
        assert r.ndim == 1
        assert len(r) > 0


class TestResample:
    def test_downsample(self, dummy_tensor_4d):
        result = resample(dummy_tensor_4d, (32, 32))
        assert result.shape == (1, 3, 32, 32)

    def test_upsample(self, dummy_tensor_4d):
        result = resample(dummy_tensor_4d, (128, 128))
        assert result.shape == (1, 3, 128, 128)

    def test_same_size(self, dummy_tensor_4d):
        result = resample(dummy_tensor_4d, (64, 64))
        assert result.shape == (1, 3, 64, 64)


class TestPerlin:
    def test_output_shape(self):
        result = perlin(2, 3, scale=4, device='cpu')
        assert result.shape == (2 * 4, 3 * 4)

    def test_deterministic_with_seed(self):
        torch.manual_seed(42)
        a = perlin(2, 2, scale=5, device='cpu')
        torch.manual_seed(42)
        b = perlin(2, 2, scale=5, device='cpu')
        assert torch.allclose(a, b)


class TestPerlinMs:
    def test_grayscale(self):
        result = perlin_ms([1, 0.5], 2, 2, grayscale=True, device='cpu')
        # Returns a list of tensors concatenated; result is 2D for grayscale
        assert result.ndim == 2

    def test_color(self):
        result = perlin_ms([1, 0.5], 2, 2, grayscale=False, device='cpu')
        # 3 channels concatenated along dim 0
        assert result.ndim == 2


class TestCreatePerlinNoise:
    def test_returns_pil(self):
        img = create_perlin_noise([1, 0.5], 2, 2, True, target_h=32, target_w=32, device='cpu')
        assert isinstance(img, Image.Image)
        assert img.size == (32, 32)

    def test_color_mode(self):
        img = create_perlin_noise([1, 0.5], 2, 2, False, target_h=32, target_w=32, device='cpu')
        assert isinstance(img, Image.Image)


class TestMakeEven:
    def test_even_stays(self):
        assert make_even(4) == 4

    def test_odd_incremented(self):
        assert make_even(5) == 6

    def test_zero(self):
        assert make_even(0) == 0


class TestFit:
    def test_no_resize_needed(self):
        img = Image.new('RGB', (100, 100))
        result = fit(img, maxsize=512)
        assert result.size == (100, 100)

    def test_resize_large(self):
        img = Image.new('RGB', (1000, 500))
        result = fit(img, maxsize=512)
        assert max(result.size) <= 512
        # Dimensions should be even
        assert result.size[0] % 2 == 0
        assert result.size[1] % 2 == 0


class TestImg2Tensor:
    def test_shape(self, dummy_rgb_image):
        t = img2tensor(dummy_rgb_image)
        assert t.shape == (1, 3, 64, 64)
        assert t.dtype == torch.float32

    def test_with_resize(self, dummy_rgb_image):
        t = img2tensor(dummy_rgb_image, size=(32, 32))
        assert t.shape == (1, 3, 32, 32)
