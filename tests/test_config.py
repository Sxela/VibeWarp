"""Tests for vibewarp.utils.misc (settings, paths, seeding)."""

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from vibewarp.utils.misc import create_path, seed_everything


class TestCreatePath:
    def test_creates_directory(self, tmp_path):
        target = os.path.join(str(tmp_path), "subdir", "nested")
        create_path(target)
        assert os.path.isdir(target)

    def test_existing_dir_no_error(self, tmp_path):
        create_path(str(tmp_path))  # should not raise


class TestSeedEverything:
    def test_reproducibility(self):
        seed_everything(42)
        a = torch.randn(10)
        r1 = np.random.rand(5)

        seed_everything(42)
        b = torch.randn(10)
        r2 = np.random.rand(5)

        torch.testing.assert_close(a, b)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_differ(self):
        seed_everything(42)
        a = torch.randn(10)
        seed_everything(123)
        b = torch.randn(10)
        assert not torch.allclose(a, b)
