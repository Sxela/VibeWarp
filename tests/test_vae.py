"""Tests for vibewarp.core.vae (tiled VAE utilities)."""

import torch
import pytest

from vibewarp.core.vae import (
    get_fold_unfold,
    get_weighting,
    _delta_border,
    _meshgrid,
)


class TestMeshgrid:
    def test_shapes(self):
        grid = _meshgrid(4, 6)
        assert grid.shape == (4, 6, 2)  # (h, w, 2) with [y, x]

    def test_values(self):
        grid = _meshgrid(3, 3)
        assert grid[0, 0, 0] == 0  # y at top-left
        assert grid[2, 0, 0] == 2  # y at bottom-left
        assert grid[0, 0, 1] == 0  # x at top-left
        assert grid[0, 2, 1] == 2  # x at top-right


class TestDeltaBorder:
    def test_shape(self):
        w = _delta_border(8, 8)
        assert w.shape == (8, 8)

    def test_corners_are_zero(self):
        w = _delta_border(16, 16)
        # Corners should have minimum weighting (zero at border)
        assert w[0, 0] <= w[8, 8]
        assert w[0, 0] == 0.0


class TestGetWeighting:
    def test_shape(self):
        # weighting shape: (1, h*w, Ly*Lx)
        w = get_weighting(64, 64, 2, 2, 'cpu')
        assert w.shape == (1, 64 * 64, 2 * 2)


class TestGetFoldUnfold:
    def test_basic(self):
        x = torch.randn(1, 4, 64, 64)
        fold, unfold, norm, w = get_fold_unfold(x, (32, 32), (16, 16))
        # Unfold should produce patches
        patches = unfold(x)
        assert patches.ndim == 3

    def test_weighted_reconstruct(self):
        """Weighted fold/unfold should produce a valid output."""
        x = torch.randn(1, 4, 64, 64)
        ks = (32, 32)
        stride = (16, 16)
        fold, unfold, norm, w = get_fold_unfold(x, ks, stride)
        patches = unfold(x)
        # Apply weighting (as done in tiled VAE)
        p = patches.view(1, -1, ks[0], ks[1], patches.shape[-1])
        p = p * w
        p = p.view(1, -1, p.shape[-1])
        recon = fold(p) / norm
        # Normalization should produce finite values
        assert torch.isfinite(recon).all()
        assert recon.shape == x.shape

    def test_with_downscale(self):
        x = torch.randn(1, 4, 64, 64)
        fold, unfold, norm, w = get_fold_unfold(x, (32, 32), (16, 16), df=2)
        patches = unfold(x)
        assert patches.ndim == 3


class TestGetFoldUnfoldUpscale:
    def test_with_upscale(self):
        x = torch.randn(1, 4, 32, 32)
        fold, unfold, norm, w = get_fold_unfold(x, (16, 16), (8, 8), uf=2)
        patches = unfold(x)
        assert patches.ndim == 3
