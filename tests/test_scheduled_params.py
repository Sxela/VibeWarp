"""Tests for scheduled parameters and updated CFGDenoiser."""

import torch
import pytest

from vibewarp.config import DiffusionConfig, RunConfig, WarpConfig
from vibewarp.core.diffusion import get_frame_schedule
from vibewarp.core.model_loader import CFGDenoiser


class TestGetFrameScheduleWithSchedules:
    def test_constant_values(self):
        cfg = RunConfig(diffusion=DiffusionConfig(steps=30, cfg_scale=8.0, seed=42))
        sched = get_frame_schedule(0, cfg)
        assert sched['steps'] == 30
        assert sched['cfg_scale'] == 8.0
        assert sched['seed'] == 42
        assert sched['style_strength'] == 0.65  # default

    def test_steps_schedule_dict(self):
        cfg = RunConfig(diffusion=DiffusionConfig(
            steps=25,
            steps_schedule={0: 30, 10: 50},
        ))
        assert get_frame_schedule(0, cfg)['steps'] == 30
        # Frame 5 is halfway between keys 0→10, blends to 40
        assert get_frame_schedule(5, cfg)['steps'] == 40
        assert get_frame_schedule(10, cfg)['steps'] == 50
        assert get_frame_schedule(20, cfg)['steps'] == 50

    def test_cfg_scale_schedule_list(self):
        cfg = RunConfig(diffusion=DiffusionConfig(
            cfg_scale=7.5,
            cfg_scale_schedule=[5.0, 7.0, 9.0, 11.0],
        ))
        assert get_frame_schedule(0, cfg)['cfg_scale'] == 5.0
        assert get_frame_schedule(2, cfg)['cfg_scale'] == 9.0
        # Past end of list: uses last value
        assert get_frame_schedule(100, cfg)['cfg_scale'] == 11.0

    def test_style_strength_schedule(self):
        cfg = RunConfig(diffusion=DiffusionConfig(
            style_strength=0.65,
            style_strength_schedule={0: 0.8, 20: 0.5},
        ))
        assert get_frame_schedule(0, cfg)['style_strength'] == 0.8
        assert get_frame_schedule(20, cfg)['style_strength'] == 0.5
        # Between: blend (since blend_json_schedules=True)
        sched = get_frame_schedule(10, cfg)
        assert 0.5 < sched['style_strength'] < 0.8

    def test_flow_blend_schedule(self):
        cfg = RunConfig(warp=WarpConfig(
            flow_blend=0.5,
            flow_blend_schedule={0: 0.3, 10: 0.7},
        ))
        assert get_frame_schedule(0, cfg)['flow_blend'] == 0.3
        assert get_frame_schedule(10, cfg)['flow_blend'] == 0.7

    def test_none_schedule_uses_default(self):
        cfg = RunConfig(diffusion=DiffusionConfig(
            steps=25, steps_schedule=None,
        ))
        assert get_frame_schedule(0, cfg)['steps'] == 25

    def test_returns_all_keys(self):
        cfg = RunConfig()
        sched = get_frame_schedule(0, cfg)
        assert 'steps' in sched
        assert 'cfg_scale' in sched
        assert 'seed' in sched
        assert 'style_strength' in sched
        assert 'flow_blend' in sched


class TestCFGDenoiserImageCond:
    def test_passes_image_cond_through(self):
        """CFGDenoiser should forward image_cond to inner model."""
        received_kwargs = {}

        class MockModel(torch.nn.Module):
            def forward(self, x, sigma, cond=None, **kwargs):
                received_kwargs.update(kwargs)
                return x

        inner = MockModel()
        cfg = CFGDenoiser(inner)

        x = torch.randn(1, 4, 8, 8)
        sigma = torch.ones(1)
        uncond = torch.randn(1, 1, 768)
        cond = torch.randn(1, 1, 768)
        image_cond = torch.randn(1, 3, 64, 64)

        cfg(x, sigma, uncond=uncond, cond=cond, cfg_scale=7.5,
            image_cond=image_cond)

        assert 'image_cond' in received_kwargs
        # Should be doubled (uncond + cond batch)
        assert received_kwargs['image_cond'].shape[0] == 2

    def test_works_without_image_cond(self):
        """CFGDenoiser should still work when no image_cond is provided."""
        class MockModel(torch.nn.Module):
            def forward(self, x, sigma, cond=None, **kwargs):
                return x

        inner = MockModel()
        cfg = CFGDenoiser(inner)

        x = torch.randn(1, 4, 8, 8)
        sigma = torch.ones(1)
        uncond = torch.randn(1, 1, 768)
        cond = torch.randn(1, 1, 768)

        result = cfg(x, sigma, uncond=uncond, cond=cond, cfg_scale=7.5)
        assert result.shape == x.shape

    def test_image_cond_not_in_outer_kwargs(self):
        """image_cond should be popped from kwargs, not passed as-is."""
        received_kwargs = {}

        class MockModel(torch.nn.Module):
            def forward(self, x, sigma, cond=None, **kwargs):
                received_kwargs.update(kwargs)
                return x

        inner = MockModel()
        cfg = CFGDenoiser(inner)

        x = torch.randn(1, 4, 8, 8)
        sigma = torch.ones(1)
        uncond = torch.randn(1, 1, 768)
        cond = torch.randn(1, 1, 768)
        image_cond = torch.randn(1, 3, 64, 64)

        cfg(x, sigma, uncond=uncond, cond=cond, cfg_scale=7.5,
            image_cond=image_cond)

        # image_cond should be duplicated (batch of 2)
        ic = received_kwargs['image_cond']
        assert ic.shape[0] == 2
        # Both halves should be the same
        assert torch.allclose(ic[0], ic[1])
