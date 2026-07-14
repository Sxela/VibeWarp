"""Tests for vibewarp.color.matching."""

import numpy as np

from vibewarp.color.matching import get_stats, match_color_simple, match_color_var


class TestMatchColorSimple:
    def test_identity_match(self, dummy_rgb_array):
        result = match_color_simple(dummy_rgb_array, dummy_rgb_array, opacity=1.0)
        assert result.shape == dummy_rgb_array.shape
        assert result.dtype == np.uint8

    def test_zero_opacity_is_noop(self, dummy_rgb_array):
        ref = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = match_color_simple(ref, dummy_rgb_array, opacity=0.0)
        np.testing.assert_array_equal(result, dummy_rgb_array)

    def test_output_range(self, dummy_rgb_array):
        ref = np.ones((64, 64, 3), dtype=np.uint8) * 200
        result = match_color_simple(ref, dummy_rgb_array, opacity=1.0)
        assert result.min() >= 0
        assert result.max() <= 255


class TestMatchColorVar:
    def test_without_transfer_fn(self, dummy_rgb_array):
        ref = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = match_color_var(ref, dummy_rgb_array, opacity=1.0)
        assert result.shape == dummy_rgb_array.shape
        assert result.dtype == np.uint8

    def test_with_identity_transfer(self, dummy_rgb_array):
        ref = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        def identity_transfer(img_arr_in, img_arr_ref):
            return img_arr_in.astype(np.float64)

        result = match_color_var(ref, dummy_rgb_array, transfer_fn=identity_transfer)
        assert result.shape == dummy_rgb_array.shape


class TestGetStats:
    def test_basic(self, dummy_rgb_array):
        stats = get_stats(dummy_rgb_array)
        assert 'r' in stats and 'g' in stats and 'b' in stats
        for ch in ['r', 'g', 'b']:
            assert 'mean' in stats[ch]
            assert 'std' in stats[ch]
            assert 0 <= stats[ch]['mean'] <= 255
            assert stats[ch]['std'] >= 0
