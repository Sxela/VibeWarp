"""Tests for vibewarp.config dataclasses."""

from vibewarp.config import (
    BrightnessConfig,
    ColorConfig,
    DiffusionConfig,
    FlowConfig,
    RunConfig,
    VideoConfig,
    WarpConfig,
    MODEL_URLS,
    CONTROLNET_MODEL_URLS,
)


class TestRunConfig:
    def test_defaults(self):
        cfg = RunConfig()
        assert cfg.batch_name == 'warpfusion'
        assert cfg.animation_mode == 'Video Input'
        assert isinstance(cfg.flow, FlowConfig)
        assert isinstance(cfg.diffusion, DiffusionConfig)

    def test_nested_override(self):
        cfg = RunConfig(
            flow=FlowConfig(flow_warp=False, flow_threads=8),
            diffusion=DiffusionConfig(steps=50, cfg_scale=12.0),
        )
        assert cfg.flow.flow_warp is False
        assert cfg.flow.flow_threads == 8
        assert cfg.diffusion.steps == 50

    def test_model_urls_present(self):
        assert 'sd_v1_5' in MODEL_URLS
        assert len(CONTROLNET_MODEL_URLS) > 0


class TestFlowConfig:
    def test_defaults(self):
        cfg = FlowConfig()
        assert cfg.flow_warp is True
        assert cfg.num_flow_updates == 20


class TestBrightnessConfig:
    def test_defaults(self):
        cfg = BrightnessConfig()
        assert cfg.enable is False
        assert cfg.high_brightness_threshold == 180
