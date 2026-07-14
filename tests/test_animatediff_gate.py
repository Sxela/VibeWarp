"""AnimateDiff must not silently half-engage.

`enabled` alone switches the render into batch mode (diffusion.run_frames) and
wraps a context sampler — but the motion module was only loaded when a path was
also set. Enabling without a module therefore paid the batching cost, injected no
temporal attention, and looked like AnimateDiff did nothing at all.
"""

import pytest

from vibewarp.config import RunConfig
from vibewarp.config_io import validate_config


def _config(tmp_path, **animatediff):
    checkpoint = tmp_path / "model.safetensors"
    video = tmp_path / "input.mp4"
    checkpoint.write_bytes(b"m")
    video.write_bytes(b"v")
    config = RunConfig(sd_checkpoint_path=str(checkpoint))
    config.video.video_init_path = str(video)
    for key, value in animatediff.items():
        setattr(config.animatediff, key, value)
    return config


class TestValidation:
    def test_enabled_without_motion_module_is_an_error(self, tmp_path):
        errors = validate_config(_config(tmp_path, enabled=True))
        paths = [e["path"] for e in errors]
        assert "animatediff.motion_module_path" in paths

    def test_enabled_with_missing_motion_module_is_an_error(self, tmp_path):
        config = _config(tmp_path, enabled=True,
                         motion_module_path=str(tmp_path / "nope.ckpt"))
        errors = validate_config(config, check_paths=True)
        assert any("does not exist" in e["message"] for e in errors
                   if e["path"] == "animatediff.motion_module_path")

    def test_enabled_with_real_motion_module_is_valid(self, tmp_path):
        module = tmp_path / "mm_sd_v15_v2.ckpt"
        module.write_bytes(b"mm")
        config = _config(tmp_path, enabled=True, motion_module_path=str(module))
        errors = validate_config(config, check_paths=True)
        assert not [e for e in errors if e["path"].startswith("animatediff")]

    def test_disabled_without_motion_module_is_fine(self, tmp_path):
        errors = validate_config(_config(tmp_path, enabled=False), check_paths=True)
        assert not [e for e in errors if e["path"].startswith("animatediff")]


class TestLoaderGate:
    def test_load_models_refuses_to_half_engage(self, monkeypatch, tmp_path):
        from vibewarp import pipeline

        config = _config(tmp_path, enabled=True, motion_module_path="")
        monkeypatch.setattr(pipeline, 'load_sd_model', lambda *a, **k: object())
        monkeypatch.setattr(pipeline, 'wrap_model_for_kdiffusion', lambda *a, **k: (None, None))

        with pytest.raises(ValueError, match="motion_module_path"):
            pipeline.load_models(config, device='cpu')
