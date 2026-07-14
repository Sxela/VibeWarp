"""Tests for vibewarp.flow.consistency."""

import numpy as np
import pytest
import torch

from vibewarp.flow.consistency import (
    edge_detector,
    extract_occlusion_mask,
    filter_unreliable,
    get_unreliable,
    make_cc_map,
    remove_small_holes,
)


class TestEdgeDetector:
    def test_uniform_no_edges(self):
        img = np.ones((32, 32, 3), dtype=np.uint8) * 128
        edges = edge_detector(img, threshold=0.5, edge_width=3)
        assert edges.max() == 0

    def test_has_edges(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[:, 16:, :] = 255  # sharp vertical edge
        edges = edge_detector(img, threshold=0.1, edge_width=3)
        assert edges.max() == 255

    def test_output_shape(self):
        img = np.random.randint(0, 255, (64, 48, 3), dtype=np.uint8)
        edges = edge_detector(img)
        assert edges.shape == (64, 48)


class TestGetUnreliable:
    def test_zero_flow(self):
        flow = np.zeros((16, 16, 2), dtype=np.float32)
        unreliable, valid = get_unreliable(flow)
        assert unreliable.shape == (16, 16, 3)
        # Zero flow means all pixels map to themselves - should be mostly reliable
        assert valid.all()

    def test_output_shapes(self, dummy_flow):
        unreliable, valid = get_unreliable(dummy_flow)
        h, w = dummy_flow.shape[:2]
        assert unreliable.shape == (h, w, 3)
        assert valid.shape == (h, w)


class TestRemoveSmallHoles:
    def test_removes_small_holes(self):
        mask = np.ones((32, 32), dtype=np.uint8) * 255
        mask[15, 15] = 0  # tiny hole
        result = remove_small_holes(mask, min_size=50)
        assert result[15, 15] == 255  # hole filled


class TestExtractOcclusionMask:
    def test_zero_flow(self):
        flow = torch.zeros(1, 2, 16, 16)
        mask, mag = extract_occlusion_mask(flow, threshold=10)
        assert mask.shape == (16, 16)
        assert mask.max() == 0  # no occlusion with zero flow


class TestMakeCcMap:
    def test_output_shape(self):
        flow_fwd = np.random.randn(32, 32, 2).astype(np.float32) * 0.1
        flow_bwd = np.random.randn(32, 32, 2).astype(np.float32) * 0.1
        cc = make_cc_map(flow_fwd, flow_bwd)
        assert cc.shape == (32, 32, 3)
        assert cc.dtype == np.uint8
