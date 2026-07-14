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


# --- CPU-only environments (CI) -----------------------------------------------------
#
# VibeWarp renders on CUDA: the diffusion, warp and reconstruction paths call
# `torch.amp.autocast('cuda')` and allocate on `device='cuda'` unconditionally. On the
# CPU-only torch that CI installs (a GPU runner is not free) those raise:
#
#     AssertionError: Torch not compiled with CUDA enabled
#
# Those tests exercise real render paths, so skipping them on CPU is the honest outcome —
# but a hand-maintained list of which ones would drift the moment someone added another,
# and it would only blow up on CI.
#
# So: recognise that specific torch error and turn it into a skip. Deliberately narrow —
# it matches only torch's own "no CUDA" message, so a genuine failure still fails.
_NO_CUDA = (
    'Torch not compiled with CUDA enabled',   # CPU-only build (what CI installs)
    'Found no NVIDIA driver',                 # CUDA build, no device
)


def is_missing_cuda(exc_type, exc_value) -> bool:
    """Is this torch complaining it has no CUDA, rather than a real failure?

    Kept narrow on purpose: a bare AssertionError from an actual assert must still fail.
    """
    if not (isinstance(exc_type, type)
            and issubclass(exc_type, (AssertionError, RuntimeError))):
        return False
    return any(marker in str(exc_value) for marker in _NO_CUDA)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Turn "torch has no CUDA" into a skip on CPU-only machines.

    Must be a HOOKWRAPPER: `pytest_runtest_call` is not a replace-hook, so calling
    `item.runtest()` from a plain implementation runs the test a SECOND time — which
    duplicates every fixture and broke ~55 unrelated tests with FileExistsError when I
    first wrote this.
    """
    outcome = yield
    if torch.cuda.is_available() or not outcome.excinfo:
        return
    exc_type, exc_value, _ = outcome.excinfo
    if is_missing_cuda(exc_type, exc_value):
        outcome.force_exception(pytest.skip.Exception(
            "needs a CUDA build of torch (the render path uses autocast('cuda')); "
            'this environment has no GPU', _use_item_location=True))
