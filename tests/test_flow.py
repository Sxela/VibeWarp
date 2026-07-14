"""Tests for vibewarp.flow.flow_utils."""

import os
import tempfile

import numpy as np
import pytest
import torch

from vibewarp.flow.flow_utils import (
    InputPadder,
    flow_to_image,
    make_colorwheel,
    read_flow,
    warp_flow,
    write_flow,
)


class TestWriteReadFlow:
    def test_roundtrip(self, dummy_flow):
        with tempfile.NamedTemporaryFile(suffix='.flo', delete=False) as f:
            path = f.name
        try:
            write_flow(path, dummy_flow)
            loaded = read_flow(path)
            np.testing.assert_allclose(loaded, dummy_flow, atol=1e-5)
        finally:
            os.unlink(path)

    def test_roundtrip_separate_uv(self):
        u = np.random.randn(16, 16).astype(np.float32)
        v = np.random.randn(16, 16).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix='.flo', delete=False) as f:
            path = f.name
        try:
            write_flow(path, u, v)
            loaded = read_flow(path)
            np.testing.assert_allclose(loaded[:, :, 0], u, atol=1e-5)
            np.testing.assert_allclose(loaded[:, :, 1], v, atol=1e-5)
        finally:
            os.unlink(path)


class TestWarpFlow:
    def test_zero_flow_is_identity(self):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        flow = np.zeros((32, 32, 2), dtype=np.float32)
        result = warp_flow(img, flow)
        # With zero flow, output should match input closely
        np.testing.assert_allclose(result, img, atol=1)

    def test_output_shape(self, dummy_rgb_array):
        h, w = dummy_rgb_array.shape[:2]
        flow = np.zeros((h, w, 2), dtype=np.float32)
        result = warp_flow(dummy_rgb_array, flow)
        assert result.shape == dummy_rgb_array.shape


class TestMakeColorwheel:
    def test_shape(self):
        cw = make_colorwheel()
        assert cw.ndim == 2
        assert cw.shape[1] == 3
        assert cw.shape[0] == 55  # RY+YG+GC+CB+BM+MR = 15+6+4+11+13+6

    def test_values_range(self):
        cw = make_colorwheel()
        assert cw.min() >= 0
        assert cw.max() <= 255


class TestFlowToImage:
    def test_output_shape(self, dummy_flow):
        img = flow_to_image(dummy_flow)
        assert img.shape == (32, 32, 3)
        assert img.dtype == np.uint8

    def test_zero_flow(self):
        flow = np.zeros((16, 16, 2), dtype=np.float32)
        img = flow_to_image(flow)
        assert img.shape == (16, 16, 3)


class TestInputPadder:
    def test_pad_unpad_roundtrip(self):
        x = torch.randn(1, 3, 55, 47)  # non-divisible by 8
        padder = InputPadder(x.shape)
        [padded] = padder.pad(x)
        assert padded.shape[-2] % 8 == 0
        assert padded.shape[-1] % 8 == 0
        unpadded = padder.unpad(padded)
        assert unpadded.shape == x.shape
        torch.testing.assert_close(unpadded, x)

    def test_already_divisible(self):
        x = torch.randn(1, 3, 64, 64)
        padder = InputPadder(x.shape)
        [padded] = padder.pad(x)
        assert padded.shape == x.shape
