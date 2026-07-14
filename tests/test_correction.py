"""Tests for vibewarp.color.correction."""

import numpy as np
import pytest
from PIL import Image

from vibewarp.color.correction import adjust_brightness, get_brightness_contrast


class TestGetBrightnessContrast:
    def test_white_image(self):
        img = Image.new('RGB', (32, 32), (255, 255, 255))
        brightness, contrast = get_brightness_contrast(img)
        assert brightness == 255.0
        assert contrast == 0.0

    def test_black_image(self):
        img = Image.new('RGB', (32, 32), (0, 0, 0))
        brightness, contrast = get_brightness_contrast(img)
        assert brightness == 0.0

    def test_mid_gray(self):
        img = Image.new('RGB', (32, 32), (128, 128, 128))
        brightness, _ = get_brightness_contrast(img)
        assert abs(brightness - 128) < 1


class TestAdjustBrightness:
    def test_returns_pil(self, dummy_rgb_image):
        result = adjust_brightness(dummy_rgb_image)
        assert isinstance(result, Image.Image)

    def test_bright_image_dimmed(self):
        img = Image.new('RGB', (32, 32), (230, 230, 230))
        result = adjust_brightness(img, high_brightness_threshold=180)
        result_brightness, _ = get_brightness_contrast(result)
        orig_brightness, _ = get_brightness_contrast(img)
        assert result_brightness < orig_brightness

    def test_dark_image_brightened(self):
        img = Image.new('RGB', (32, 32), (10, 10, 10))
        result = adjust_brightness(img, low_brightness_threshold=40)
        result_brightness, _ = get_brightness_contrast(result)
        orig_brightness, _ = get_brightness_contrast(img)
        assert result_brightness > orig_brightness

    def test_normal_image_unchanged(self):
        img = Image.new('RGB', (32, 32), (128, 128, 128))
        result = adjust_brightness(img)
        arr1 = np.array(img)
        arr2 = np.array(result)
        np.testing.assert_array_equal(arr1, arr2)
