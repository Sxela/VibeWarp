"""Tests for sliding context reconstruction noise and batched DDIM inversion."""

import os
import pytest
import torch

from vibewarp.config import ReconstructionNoiseConfig
from vibewarp.core.reconstruction import (
    _apply_model_sliding_ctx,
    _build_cond_dict,
    find_noise_for_image,
    get_predicted_noise_adiff_batched,
)


# ---- Mock models ----

class MockSD:
    parameterization = 'eps'
    def get_learned_conditioning(self, prompts):
        return torch.randn(len(prompts), 77, 768)
    def apply_model(self, x, t, cond=None):
        return torch.randn_like(x)


class MockWrap:
    def get_sigmas(self, steps):
        return torch.linspace(14.6, 0.029, steps + 1)
    def sigma_to_t(self, sigma):
        return sigma * 100
    def get_scalings(self, sigma):
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1) ** 0.5
        return c_out, c_in


# ---- Sliding Context Tests ----

class TestApplyModelSlidingCtx:
    def test_single_window_covers_all(self):
        sd = MockSD()
        x = torch.randn(4, 4, 8, 8)
        t = torch.ones(4) * 100
        cond = torch.randn(4, 77, 768)
        windows = [[0, 1, 2, 3]]
        result = _apply_model_sliding_ctx(sd, x, t, cond, None, windows)
        assert result.shape == (4, 4, 8, 8)

    def test_multiple_windows_average_overlap(self):
        """Overlapping windows should average outputs for shared frames."""
        call_count = [0]
        class CountSD(MockSD):
            def apply_model(self, x, t, cond=None):
                call_count[0] += 1
                return torch.ones_like(x)

        sd = CountSD()
        x = torch.randn(4, 4, 8, 8)
        t = torch.ones(4) * 100
        cond = torch.randn(4, 77, 768)
        # Two overlapping windows: [0,1,2] and [1,2,3]
        windows = [[0, 1, 2], [1, 2, 3]]
        result = _apply_model_sliding_ctx(sd, x, t, cond, None, windows)
        assert result.shape == (4, 4, 8, 8)
        assert call_count[0] == 2
        # All outputs are 1.0 (ones averaged = ones)
        assert torch.allclose(result, torch.ones_like(result))

    def test_empty_windows_returns_zeros(self):
        sd = MockSD()
        x = torch.randn(2, 4, 8, 8)
        t = torch.ones(2) * 100
        cond = torch.randn(2, 77, 768)
        result = _apply_model_sliding_ctx(sd, x, t, cond, None, [])
        # No windows processed, but out_count clamped to 1 so result is zeros
        assert torch.allclose(result, torch.zeros_like(result))

    def test_window_indices_beyond_batch_skipped(self):
        sd = MockSD()
        x = torch.randn(2, 4, 8, 8)
        t = torch.ones(2) * 100
        cond = torch.randn(2, 77, 768)
        # Window has indices beyond batch size
        windows = [[0, 1, 5, 10]]
        result = _apply_model_sliding_ctx(sd, x, t, cond, None, windows)
        assert result.shape == (2, 4, 8, 8)

    def test_with_dict_image_conditioning(self):
        sd = MockSD()
        x = torch.randn(4, 4, 8, 8)
        t = torch.ones(4) * 100
        cond = torch.randn(4, 77, 768)
        img_cond = {'depth': torch.randn(4, 1, 64, 64)}
        windows = [[0, 1], [2, 3]]
        result = _apply_model_sliding_ctx(sd, x, t, cond, img_cond, windows)
        assert result.shape == (4, 4, 8, 8)

    def test_uc_flag_passed_through(self):
        """is_uc flag should be forwarded to _build_cond_dict."""
        sd = MockSD()
        x = torch.randn(2, 4, 8, 8)
        t = torch.ones(2) * 100
        cond = torch.randn(2, 77, 768)
        img = torch.randn(2, 3, 64, 64)
        # Should not crash with is_uc=True
        result = _apply_model_sliding_ctx(sd, x, t, cond, img, [[0, 1]], is_uc=True)
        assert result.shape == (2, 4, 8, 8)


class TestFindNoiseWithSlidingContext:
    def test_sliding_context_basic(self):
        sd, wrap = MockSD(), MockWrap()
        latent = torch.randn(4, 4, 8, 8)
        # Build a simple context schedule: 3 steps, each with one window
        ctx_sched = [[[0, 1, 2, 3]]] * 4  # enough for 3 steps
        result = find_noise_for_image(
            sd_model=sd, model_wrap=wrap,
            init_latent=latent, prompt='test',
            steps=3, sliding_context=True,
            context_schedule=ctx_sched,
        )
        assert result.shape == (4, 4, 8, 8)

    def test_sliding_context_with_cfg(self):
        sd, wrap = MockSD(), MockWrap()
        latent = torch.randn(4, 4, 8, 8)
        ctx_sched = [[[0, 1, 2, 3]]] * 4
        result = find_noise_for_image(
            sd_model=sd, model_wrap=wrap,
            init_latent=latent, prompt='test', neg_prompt='bad',
            cfg_scale=7.5, steps=3,
            sliding_context=True, context_schedule=ctx_sched,
        )
        assert result.shape == (4, 4, 8, 8)

    def test_sliding_context_disabled_without_schedule(self):
        """sliding_context=True but no schedule should fall back to standard."""
        sd, wrap = MockSD(), MockWrap()
        latent = torch.randn(1, 4, 8, 8)
        result = find_noise_for_image(
            sd_model=sd, model_wrap=wrap,
            init_latent=latent, prompt='test',
            steps=3, sliding_context=True,
            context_schedule=None,
        )
        assert result.shape == (1, 4, 8, 8)


# ---- Batched DDIM Inversion Tests ----

class TestGetPredictedNoiseAdiffBatched:
    def test_missing_frames_uses_random_latent(self, tmp_path):
        """Non-existent frame paths should use random latents."""
        config = ReconstructionNoiseConfig(enabled=True, batched=True)
        # image_size defaults to (512, 512), so latents are 64x64
        rand = torch.randn(2, 4, 64, 64)
        result = get_predicted_noise_adiff_batched(
            sd_model=MockSD(),
            model_wrap=MockWrap(),
            frame_paths=['/nonexistent/a.png', '/nonexistent/b.png'],
            frame_idxs=[0, 1],
            random_noise=rand,
            config=config,
            prompt='test',
        )
        assert result.shape == (2, 4, 64, 64)

    def test_batched_returns_correct_shape(self, tmp_path):
        config = ReconstructionNoiseConfig(enabled=True, batched=True)
        rand = torch.randn(3, 4, 64, 64)
        result = get_predicted_noise_adiff_batched(
            sd_model=MockSD(),
            model_wrap=MockWrap(),
            frame_paths=['/no/a.png', '/no/b.png', '/no/c.png'],
            frame_idxs=[0, 1, 2],
            random_noise=rand,
            config=config,
            prompt='test',
        )
        assert result.shape == (3, 4, 64, 64)

    def test_batched_with_sliding_context(self, tmp_path):
        config = ReconstructionNoiseConfig(
            enabled=True, batched=True, sliding_context=True,
        )
        rand = torch.randn(2, 4, 64, 64)
        result = get_predicted_noise_adiff_batched(
            sd_model=MockSD(),
            model_wrap=MockWrap(),
            frame_paths=['/no/a.png', '/no/b.png'],
            frame_idxs=[0, 1],
            random_noise=rand,
            config=config,
            prompt='test',
            context_length=16,
            context_overlap=4,
        )
        assert result.shape == (2, 4, 64, 64)

    def test_batched_randomness_blend(self):
        """With randomness > 0, output should differ from pure rec noise."""
        config = ReconstructionNoiseConfig(
            enabled=True, batched=True, randomness=0.5,
        )
        rand = torch.ones(1, 4, 64, 64) * 100  # distinctive noise
        result = get_predicted_noise_adiff_batched(
            sd_model=MockSD(),
            model_wrap=MockWrap(),
            frame_paths=['/no/a.png'],
            frame_idxs=[0],
            random_noise=rand,
            config=config,
            prompt='test',
        )
        assert result.shape == (1, 4, 64, 64)
        # With randomness=0.5 and rand=100, the blend should be nonzero
        assert result.abs().mean() > 0


class TestReconstructionNoiseConfigNewFields:
    def test_sliding_context_default(self):
        config = ReconstructionNoiseConfig()
        assert config.sliding_context is False

    def test_batched_default(self):
        config = ReconstructionNoiseConfig()
        assert config.batched is False

    def test_sliding_context_enabled(self):
        config = ReconstructionNoiseConfig(sliding_context=True)
        assert config.sliding_context is True

    def test_batched_enabled(self):
        config = ReconstructionNoiseConfig(batched=True)
        assert config.batched is True
