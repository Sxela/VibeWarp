"""Tests for GPU PDF color transfer — numerical parity vs numpy reference."""

import numpy as np
import pytest
import torch

from vibewarp.color.pdf_gpu import pdf_transfer_gpu
from vibewarp.vendor.python_color_transfer.color_transfer import ColorTransfer


def _requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("GPU PDF transfer needs CUDA")


class TestPdfGpuParity:
    """GPU PDF output must match the numpy reference within histogram quantization."""

    def test_matches_numpy_small(self):
        _requires_cuda()
        rng = np.random.default_rng(0)
        img_in = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        img_ref = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)

        ref = ColorTransfer().pdf_transfer(img_arr_in=img_in, img_arr_ref=img_ref)
        gpu = pdf_transfer_gpu(img_in, img_ref)

        assert ref.shape == gpu.shape == (64, 64, 3)
        # n=300 bins → 1 bin ≈ 0.85 uint8 units, per-rotation quantization accumulates.
        # 6 rotations → bounded drift. Most pixels should be within 8 levels.
        diff = np.abs(ref.astype(np.int16) - gpu.astype(np.int16))
        assert diff.mean() < 3.0, f"mean diff {diff.mean():.2f} exceeds tolerance"
        assert np.percentile(diff, 95) < 10.0

    def test_identical_images_roughly_idempotent(self):
        _requires_cuda()
        rng = np.random.default_rng(42)
        img = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)

        out = pdf_transfer_gpu(img, img)
        # Transferring an image onto itself should be near-identity.
        diff = np.abs(img.astype(np.int16) - out.astype(np.int16))
        assert diff.mean() < 2.0

    def test_handles_non_square(self):
        _requires_cuda()
        rng = np.random.default_rng(7)
        img_in = rng.integers(0, 255, size=(48, 96, 3), dtype=np.uint8)
        img_ref = rng.integers(0, 255, size=(48, 96, 3), dtype=np.uint8)

        out = pdf_transfer_gpu(img_in, img_ref)
        assert out.shape == (48, 96, 3)
        assert out.dtype == np.uint8

    def test_cpu_fallback(self):
        # device='cpu' should work without CUDA for CI portability
        rng = np.random.default_rng(3)
        img_in = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
        img_ref = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
        out = pdf_transfer_gpu(img_in, img_ref, device='cpu')
        assert out.shape == (32, 32, 3)


class TestMatchColorVarDefault:
    """match_color_var with no transfer_fn should use the GPU PDF by default."""

    def test_default_applies_pdf_transfer(self):
        _requires_cuda()
        from vibewarp.color.matching import match_color_var

        rng = np.random.default_rng(0)
        stylized = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        raw = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)

        # With opacity=1, full color transfer should happen (not identity)
        out = match_color_var(stylized, raw, opacity=1.0)
        diff = np.abs(out.astype(np.int16) - raw.astype(np.int16))
        # Should differ noticeably from raw when colors are transferred
        assert diff.mean() > 1.0
