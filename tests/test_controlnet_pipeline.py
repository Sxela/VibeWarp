"""Tests for ControlNet pipeline wiring — config + load_models integration."""

import os
import pytest

from vibewarp.config import (
    ControlNetConfig,
    ControlNetEntry,
    AnimateDiffConfig,
    RunConfig,
)


class TestControlNetConfig:
    def test_default_disabled(self):
        cfg = ControlNetConfig()
        assert cfg.enabled is False
        assert cfg.models == {}

    def test_with_models(self):
        cfg = ControlNetConfig(
            enabled=True,
            models={
                'control_sd15_canny': ControlNetEntry(
                    path='/models/canny.pth',
                    weight=0.8,
                ),
            },
        )
        assert cfg.enabled is True
        assert 'control_sd15_canny' in cfg.models
        assert cfg.models['control_sd15_canny'].weight == 0.8

    def test_entry_defaults(self):
        entry = ControlNetEntry()
        assert entry.weight == 1.0
        assert entry.start == 0.0
        assert entry.end == 1.0
        assert entry.annotator == ''
        assert entry.source == ''

    def test_multiple_models(self):
        cfg = ControlNetConfig(
            enabled=True,
            models={
                'control_sd15_canny': ControlNetEntry(path='a.pth', weight=0.5),
                'control_sd15_depth': ControlNetEntry(path='b.pth', weight=0.7),
                'control_sd15_tile': ControlNetEntry(path='c.pth', weight=1.0),
            },
        )
        assert len(cfg.models) == 3

    def test_mode_default(self):
        cfg = ControlNetConfig()
        assert cfg.mode == 'internal'


class TestAnimateDiffConfig:
    def test_default_disabled(self):
        cfg = AnimateDiffConfig()
        assert cfg.enabled is False
        assert cfg.motion_module_path == ''

    def test_with_values(self):
        cfg = AnimateDiffConfig(
            enabled=True,
            motion_module_path='/models/mm.safetensors',
            batch_length=64,
            context_length=24,
        )
        assert cfg.enabled is True
        assert cfg.batch_length == 64
        assert cfg.context_length == 24

    def test_defaults(self):
        cfg = AnimateDiffConfig()
        assert cfg.batch_length == 32
        assert cfg.batch_overlap == 8
        assert cfg.context_length == 16
        assert cfg.context_overlap == 10
        assert cfg.stop_early == 0


class TestRunConfigWithNewConfigs:
    def test_has_controlnet(self):
        cfg = RunConfig()
        assert hasattr(cfg, 'controlnet')
        assert isinstance(cfg.controlnet, ControlNetConfig)

    def test_has_animatediff(self):
        cfg = RunConfig()
        assert hasattr(cfg, 'animatediff')
        assert isinstance(cfg.animatediff, AnimateDiffConfig)

    def test_controlnet_disabled_by_default(self):
        cfg = RunConfig()
        assert cfg.controlnet.enabled is False

    def test_animatediff_disabled_by_default(self):
        cfg = RunConfig()
        assert cfg.animatediff.enabled is False

    def test_full_config_with_both(self):
        cfg = RunConfig(
            controlnet=ControlNetConfig(
                enabled=True,
                models={'control_sd15_canny': ControlNetEntry(path='canny.pth')},
            ),
            animatediff=AnimateDiffConfig(
                enabled=True,
                motion_module_path='mm.safetensors',
            ),
        )
        assert cfg.controlnet.enabled is True
        assert cfg.animatediff.enabled is True
