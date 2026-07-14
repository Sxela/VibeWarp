"""Smoke tests for vibewarp.pipeline and vibewarp.core.diffusion."""

import os

import pytest

from vibewarp.config import FreeUConfig, RunConfig
from vibewarp.core.diffusion import FrameState, RenderContext, get_frame_schedule
from vibewarp.pipeline import _configure_unified_forward, _resolve_controlnet_path, setup_directories


class TestSetupDirectories:
    def test_creates_dirs(self, tmp_path):
        cfg = RunConfig(root_dir=str(tmp_path), batch_name='test_batch')
        dirs = setup_directories(cfg)
        assert os.path.isdir(dirs['output'])
        assert os.path.isdir(dirs['batch'])
        assert os.path.isdir(dirs['settings'])
        assert os.path.isdir(dirs['temp'])
        assert os.path.isdir(dirs['models'])
        assert os.path.isdir(dirs['init_images'])

    def test_idempotent(self, tmp_path):
        cfg = RunConfig(root_dir=str(tmp_path))
        setup_directories(cfg)
        setup_directories(cfg)  # should not raise


class TestControlNetDirectory:
    def test_absolute_checkpoint_path_is_unchanged(self):
        path = os.path.abspath(os.path.join("models", "canny.pth"))
        assert _resolve_controlnet_path("control_sd15_canny", path, "other") == path

    def test_relative_checkpoint_path_uses_model_directory(self):
        assert _resolve_controlnet_path("custom", "custom.pth", "controlnet") == \
            os.path.join("controlnet", "custom.pth")

    def test_known_model_filename_is_inferred_from_directory(self):
        assert _resolve_controlnet_path("control_sd15_canny", "", "controlnet") == \
            os.path.join("controlnet", "control_v11p_sd15_canny.pth")


class TestFrameState:
    def test_defaults(self):
        s = FrameState()
        assert s.frame_num == 0
        assert s.init_image is None

    def test_update(self):
        s = FrameState(frame_num=10, seed=42)
        assert s.frame_num == 10
        assert s.seed == 42


class TestRenderContext:
    def test_defaults(self):
        ctx = RenderContext()
        assert ctx.sd_model is None
        assert ctx.loaded_controlnets == {}

    def test_with_config(self):
        cfg = RunConfig(batch_name='mytest')
        ctx = RenderContext(config=cfg)
        assert ctx.config.batch_name == 'mytest'


class TestUnifiedForwardSetup:
    def test_empty_controlnet_dict_is_still_patched(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            'vibewarp.core.controlnet.patch_model_for_controlnets',
            lambda model, controlnets, **kwargs: calls.append((model, controlnets, kwargs)))
        model = object()
        _configure_unified_forward(model, {}, FreeUConfig())
        assert calls == [(model, {}, {'normalize_weights': False})]

    def test_freeu_applies_without_controlnets(self, monkeypatch):
        class Model:
            class Inner:
                diffusion_model = object()
            model = Inner()

        calls = []
        monkeypatch.setattr(
            'vibewarp.core.controlnet.patch_model_for_controlnets',
            lambda model, controlnets, **kwargs: calls.append(('patch', controlnets, kwargs)))
        monkeypatch.setattr(
            'vibewarp.core.freeu.set_freeu',
            lambda unet, config: calls.append(('freeu', unet, config)))
        config = FreeUConfig(do_freeunet=True)
        model = Model()
        _configure_unified_forward(model, {}, config)
        assert calls == [('patch', {}, {'normalize_weights': False}),
                         ('freeu', model.model.diffusion_model, config)]

    def test_stock_validation_rejects_freeu(self):
        with pytest.raises(RuntimeError, match='VALIDATE_STOCK_FORWARD'):
            _configure_unified_forward(
                object(), {}, FreeUConfig(do_freeunet=True),
                validate_stock_forward=True)


class TestGetFrameSchedule:
    def test_returns_dict(self):
        cfg = RunConfig()
        result = get_frame_schedule(0, cfg)
        assert isinstance(result, dict)
        assert 'steps' in result
        assert 'cfg_scale' in result

    def test_uses_config_values(self):
        from vibewarp.config import DiffusionConfig
        cfg = RunConfig(diffusion=DiffusionConfig(steps=50, cfg_scale=12.0))
        result = get_frame_schedule(0, cfg)
        assert result['steps'] == 50
        assert result['cfg_scale'] == 12.0
