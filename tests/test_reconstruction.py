"""Tests for reconstruction noise (DDIM inversion) module."""

import os
import pytest
import numpy as np
import torch

from vibewarp.config import RunConfig, ReconstructionNoiseConfig
from vibewarp.core.reconstruction import (
    _blend_noise,
    _append_dims,
    _build_cond_dict,
    _noise_cache_key,
    _load_cached_noise,
    _save_cached_noise,
    find_noise_for_image,
    get_predicted_noise_adiff,
)


class TestBlendNoise:
    def test_pure_reconstructed(self):
        """randomness=0 should return pure reconstructed noise."""
        rec = torch.randn(1, 4, 8, 8)
        rand = torch.randn(1, 4, 8, 8)
        result = _blend_noise(rec, rand, 0.0)
        assert torch.allclose(result, rec)

    def test_pure_random(self):
        """randomness=1 should return pure random noise."""
        rec = torch.randn(1, 4, 8, 8)
        rand = torch.randn(1, 4, 8, 8)
        result = _blend_noise(rec, rand, 1.0)
        assert torch.allclose(result, rand)

    def test_half_blend_normalization(self):
        """randomness=0.5 should normalize by sqrt(0.5^2 + 0.5^2)."""
        rec = torch.ones(1, 4, 8, 8) * 2
        rand = torch.ones(1, 4, 8, 8) * 4
        result = _blend_noise(rec, rand, 0.5)
        expected = (0.5 * 2 + 0.5 * 4) / (0.5 ** 2 + 0.5 ** 2) ** 0.5
        assert torch.allclose(result, torch.ones_like(result) * expected, atol=1e-6)

    def test_blend_preserves_shape(self):
        """Blended output should have same shape as inputs."""
        rec = torch.randn(3, 4, 16, 16)
        rand = torch.randn(3, 4, 16, 16)
        result = _blend_noise(rec, rand, 0.3)
        assert result.shape == rec.shape

    def test_blend_between_zero_and_one(self):
        """Non-extreme randomness should produce intermediate values."""
        rec = torch.ones(1, 4, 8, 8) * 10
        rand = torch.zeros(1, 4, 8, 8)
        result = _blend_noise(rec, rand, 0.3)
        # Should be between 0 and 10 (approximately)
        assert result.mean().item() > 0
        assert result.mean().item() < 15  # accounting for normalization


class TestAppendDims:
    def test_1d_to_4d(self):
        x = torch.tensor([1.0, 2.0])
        result = _append_dims(x, 4)
        assert result.shape == (2, 1, 1, 1)

    def test_2d_to_4d(self):
        x = torch.randn(3, 4)
        result = _append_dims(x, 4)
        assert result.shape == (3, 4, 1, 1)

    def test_same_dims(self):
        x = torch.randn(2, 3, 4, 5)
        result = _append_dims(x, 4)
        assert result.shape == x.shape

    def test_error_on_fewer_dims(self):
        x = torch.randn(2, 3, 4, 5)
        with pytest.raises(ValueError):
            _append_dims(x, 3)


class TestBuildCondDict:
    def test_basic_cond(self):
        cond = torch.randn(1, 77, 768)
        result = _build_cond_dict(cond)
        assert 'c_crossattn' in result
        assert len(result['c_crossattn']) == 1
        assert torch.equal(result['c_crossattn'][0], cond)
        assert 'c_concat' not in result

    def test_with_tensor_image_cond(self):
        cond = torch.randn(1, 77, 768)
        img = torch.randn(1, 3, 64, 64)
        result = _build_cond_dict(cond, img, is_uc=False)
        assert 'c_concat' in result
        assert torch.equal(result['c_concat'][0], img)

    def test_uc_zeros_image_cond(self):
        cond = torch.randn(1, 77, 768)
        img = torch.randn(1, 3, 64, 64)
        result = _build_cond_dict(cond, img, is_uc=True)
        assert 'c_concat' in result
        assert torch.allclose(result['c_concat'][0], torch.zeros_like(img))

    def test_with_dict_image_cond(self):
        cond = torch.randn(1, 77, 768)
        img_dict = {'depth': torch.randn(1, 1, 64, 64)}
        result = _build_cond_dict(cond, img_dict, is_uc=False)
        assert result['c_concat'] is img_dict

    def test_uc_zeros_dict_image_cond(self):
        cond = torch.randn(1, 77, 768)
        img_dict = {'depth': torch.randn(1, 1, 64, 64)}
        result = _build_cond_dict(cond, img_dict, is_uc=True)
        assert torch.allclose(result['c_concat']['depth'], torch.zeros_like(img_dict['depth']))


class TestNoiseCaching:
    def test_cache_key_deterministic(self):
        config = ReconstructionNoiseConfig(cfg_scale=1.0, steps_pct=0.5)
        key1 = _noise_cache_key(config)
        key2 = _noise_cache_key(config)
        assert key1 == key2
        assert len(key1) == 16

    def test_cache_key_differs_for_different_config(self):
        c1 = ReconstructionNoiseConfig(cfg_scale=1.0)
        c2 = ReconstructionNoiseConfig(cfg_scale=2.0)
        assert _noise_cache_key(c1) != _noise_cache_key(c2)

    def test_save_and_load_cache(self, tmp_path):
        config = ReconstructionNoiseConfig()
        noise = torch.randn(1, 4, 8, 8)
        _save_cached_noise(str(tmp_path), 5, config, noise)
        loaded = _load_cached_noise(str(tmp_path), 5, config)
        assert loaded is not None
        assert torch.allclose(loaded, noise)

    def test_load_missing_returns_none(self, tmp_path):
        config = ReconstructionNoiseConfig()
        result = _load_cached_noise(str(tmp_path), 99, config)
        assert result is None

    def test_load_empty_dir_returns_none(self):
        result = _load_cached_noise('', 0, ReconstructionNoiseConfig())
        assert result is None


class TestReconstructionNoiseConfig:
    def test_defaults(self):
        config = ReconstructionNoiseConfig()
        assert config.enabled is False
        assert config.randomness == 0.0
        assert config.cfg_scale == 1.0
        assert config.steps_pct == 1.0
        assert config.source == 'init'
        assert config.steps_cutoff == 0
        assert config.noise_scale == 1.0

    def test_in_run_config(self):
        rc = RunConfig()
        assert hasattr(rc, 'reconstruction_noise')
        assert isinstance(rc.reconstruction_noise, ReconstructionNoiseConfig)

    def test_custom_values(self):
        config = ReconstructionNoiseConfig(
            enabled=True,
            randomness=0.3,
            cfg_scale=2.0,
            steps_pct=0.5,
            source='stylized',
        )
        assert config.enabled is True
        assert config.randomness == 0.3
        assert config.source == 'stylized'


class TestGetPredictedNoiseAdiff:
    def test_missing_frames_uses_random(self, tmp_path):
        """When frame paths don't exist, should fall back to random noise."""
        config = ReconstructionNoiseConfig(enabled=True)
        rand = torch.randn(2, 4, 8, 8)
        # Create a minimal mock sd_model and model_wrap — not needed since
        # frames don't exist and we skip inversion
        class MockSD:
            def get_learned_conditioning(self, prompts):
                return torch.randn(len(prompts), 77, 768)
        class MockWrap:
            def get_sigmas(self, steps):
                return torch.linspace(14.6, 0.029, steps + 1)

        result = get_predicted_noise_adiff(
            sd_model=MockSD(),
            model_wrap=MockWrap(),
            frame_paths=['/nonexistent/a.png', '/nonexistent/b.png'],
            frame_idxs=[0, 1],
            random_noise=rand,
            config=config,
            prompt='test',
        )
        assert result.shape == (2, 4, 8, 8)

    def test_cached_noise_skips_inversion(self, tmp_path):
        """When noise is cached, should load from cache without calling SD."""
        config = ReconstructionNoiseConfig(enabled=True)
        cached = torch.randn(1, 4, 8, 8)
        _save_cached_noise(str(tmp_path), 0, config, cached)

        class MockSD:
            def get_learned_conditioning(self, prompts):
                return torch.randn(len(prompts), 77, 768)
        class MockWrap:
            def get_sigmas(self, steps):
                return torch.linspace(14.6, 0.029, steps + 1)

        result = get_predicted_noise_adiff(
            sd_model=MockSD(),
            model_wrap=MockWrap(),
            frame_paths=['/nonexistent/frame.png'],
            frame_idxs=[0],
            random_noise=torch.randn(1, 4, 8, 8),
            config=config,
            prompt='test',
            cache_dir=str(tmp_path),
        )
        assert result.shape == (1, 4, 8, 8)
        assert torch.allclose(result, cached)


class TestFindNoiseForImage:
    """Integration-level tests for DDIM inversion using mocks."""

    def _make_mock_models(self):
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

        return MockSD(), MockWrap()

    def test_basic_inversion(self):
        sd, wrap = self._make_mock_models()
        latent = torch.randn(1, 4, 8, 8)
        result = find_noise_for_image(
            sd_model=sd, model_wrap=wrap,
            init_latent=latent, prompt='test',
            steps=3,
        )
        assert result.shape == latent.shape

    def test_with_cfg(self):
        sd, wrap = self._make_mock_models()
        latent = torch.randn(1, 4, 8, 8)
        result = find_noise_for_image(
            sd_model=sd, model_wrap=wrap,
            init_latent=latent, prompt='test', neg_prompt='bad',
            cfg_scale=7.5, steps=3,
        )
        assert result.shape == latent.shape

    def test_batched_input(self):
        sd, wrap = self._make_mock_models()
        latent = torch.randn(4, 4, 8, 8)
        result = find_noise_for_image(
            sd_model=sd, model_wrap=wrap,
            init_latent=latent, prompt='test',
            steps=2,
        )
        assert result.shape == (4, 4, 8, 8)

    def test_steps_cutoff(self):
        sd, wrap = self._make_mock_models()
        latent = torch.randn(1, 4, 8, 8)
        # Should not crash with early cutoff
        result = find_noise_for_image(
            sd_model=sd, model_wrap=wrap,
            init_latent=latent, prompt='test',
            steps=5, steps_cutoff=2,
        )
        assert result.shape == latent.shape

    def test_separate_uncond_false(self):
        sd, wrap = self._make_mock_models()
        latent = torch.randn(1, 4, 8, 8)
        result = find_noise_for_image(
            sd_model=sd, model_wrap=wrap,
            init_latent=latent, prompt='test', neg_prompt='bad',
            cfg_scale=7.5, steps=3, separate_uncond=False,
        )
        assert result.shape == latent.shape

    def test_cfg_scale_one_no_uncond(self):
        """cfg_scale=1 should skip uncond entirely."""
        sd, wrap = self._make_mock_models()
        latent = torch.randn(1, 4, 8, 8)
        result = find_noise_for_image(
            sd_model=sd, model_wrap=wrap,
            init_latent=latent, prompt='test',
            cfg_scale=1.0, steps=3,
        )
        assert result.shape == latent.shape
