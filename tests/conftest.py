"""Shared test fixtures."""

import os
import sys

# Put vibewarp/vendor on sys.path so bare `from ldm...` / `from sgm...` imports
# resolve against the vendored copies (mirrors what _ensure_vendor_on_path does
# at model-load time in production).
_vendor_dir = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'vibewarp', 'vendor')
)
if os.path.isdir(_vendor_dir) and _vendor_dir not in sys.path:
    sys.path.insert(0, _vendor_dir)

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def dummy_rgb_image():
    """Create a small 64x64 RGB PIL image."""
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def dummy_rgb_array():
    """Create a small 64x64 RGB numpy array."""
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def dummy_tensor_4d():
    """Create a [1, 3, 64, 64] float tensor."""
    return torch.randn(1, 3, 64, 64)


@pytest.fixture
def dummy_flow():
    """Create a small 32x32 optical flow field."""
    return np.random.randn(32, 32, 2).astype(np.float32)
