"""The CPU-only skip must be narrow: skip "no CUDA", but never a real failure.

CI installs CPU-only torch, so the ~27 tests that exercise real render paths raise
"Torch not compiled with CUDA enabled". Those get skipped. Anything else must still fail —
a rule that swallows genuine errors is worse than a red CI.
"""

import pytest

from tests.conftest import is_missing_cuda


@pytest.mark.parametrize('exc_type, message', [
    (AssertionError, 'Torch not compiled with CUDA enabled'),
    (RuntimeError, 'Found no NVIDIA driver on your system'),
    (AssertionError, 'blah Torch not compiled with CUDA enabled blah'),
])
def test_recognises_a_missing_cuda(exc_type, message):
    assert is_missing_cuda(exc_type, message) is True


@pytest.mark.parametrize('exc_type, message', [
    (AssertionError, 'assert 3 == 4'),                    # a real assertion failure
    (RuntimeError, 'CUDA out of memory'),                 # a REAL gpu error — must fail
    (RuntimeError, 'shape mismatch'),
    (ValueError, 'Torch not compiled with CUDA enabled'),  # right words, wrong type
    (FileNotFoundError, 'missing.ckpt'),
])
def test_does_not_swallow_a_real_failure(exc_type, message):
    assert is_missing_cuda(exc_type, message) is False
