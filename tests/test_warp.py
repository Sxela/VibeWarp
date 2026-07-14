"""Tests for vibewarp.flow.warp."""

import numpy as np
import pytest
from PIL import Image

from vibewarp.flow.warp import (
    blended_roll,
    get_k_means_clusters,
    k_means_warp,
    warp_frame,
)


class TestBlendedRoll:
    def test_integer_shift(self):
        arr = np.arange(10).astype(float)
        result = blended_roll(arr, 2, 0)
        expected = np.roll(arr, 2, axis=0)
        np.testing.assert_array_equal(result, expected)

    def test_fractional_shift(self):
        arr = np.arange(10).astype(float)
        result = blended_roll(arr, 1.5, 0)
        # Should be blend of roll-by-1 and roll-by-2
        r1 = np.roll(arr, 1, axis=0)
        r2 = np.roll(arr, 2, axis=0)
        expected = r1 * 0.5 + r2 * 0.5
        np.testing.assert_allclose(result, expected)

    def test_zero_shift(self):
        arr = np.arange(5).astype(float)
        result = blended_roll(arr, 0, 0)
        np.testing.assert_array_equal(result, arr)


class TestGetKMeansClusters:
    def test_basic(self, dummy_flow):
        clustered, centers = get_k_means_clusters(dummy_flow, 3)
        assert clustered.shape == dummy_flow.shape
        assert centers.shape == (3, 2)

    def test_single_cluster(self):
        flow = np.ones((16, 16, 2), dtype=np.float32)
        clustered, centers = get_k_means_clusters(flow, 1)
        assert centers.shape == (1, 2)
        np.testing.assert_allclose(centers[0], [1.0, 1.0], atol=1e-5)


class TestKMeansWarp:
    def test_returns_pil(self):
        flow = np.random.randn(32, 32, 2).astype(np.float32)
        img = Image.new('RGB', (32, 32), color=(128, 128, 128))
        result = k_means_warp(flow, img, num_k=3)
        assert isinstance(result, Image.Image)


class TestWarpFrame:
    def test_zero_flow_blend_zero(self):
        """With zero flow and zero blend, result should match frame2."""
        frame1 = Image.new('RGB', (32, 32), color=(255, 0, 0))
        frame2 = Image.new('RGB', (32, 32), color=(0, 255, 0))
        flow = np.zeros((32, 32, 2), dtype=np.float32)
        result = warp_frame(frame1, frame2, flow, blend=0.0, pad_pct=0.0)
        arr = np.array(result)
        # With blend=0, output is purely frame2
        np.testing.assert_allclose(arr[:, :, 1], 255, atol=2)  # green channel

    def test_output_is_pil(self):
        frame1 = Image.new('RGB', (32, 32))
        frame2 = Image.new('RGB', (32, 32))
        flow = np.zeros((32, 32, 2), dtype=np.float32)
        result = warp_frame(frame1, frame2, flow, blend=0.5)
        assert isinstance(result, Image.Image)
