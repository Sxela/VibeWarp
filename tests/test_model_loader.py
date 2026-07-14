"""Tests for vibewarp.core.model_loader — unit tests (no GPU/model files required)."""

import os

import pytest
import torch

from vibewarp.core.model_loader import (
    CFGDenoiser,
    MODEL_CONFIGS,
    configure_clip_skip,
    load_controlnet,
    load_sd_model,
    wrap_model_for_kdiffusion,
)


def test_configure_clip_skip_selects_normalized_hidden_layer():
    class Encoder:
        layer = 'last'
        layer_idx = None
    class Model:
        cond_stage_model = Encoder()
    model = Model()
    configure_clip_skip(model, 2)
    assert model.cond_stage_model.layer == 'hidden'
    assert model.cond_stage_model.layer_idx == -2
    configure_clip_skip(model, 1)
    assert model.cond_stage_model.layer == 'last'
    assert model.cond_stage_model.layer_idx is None


def test_sd15_controlnet_head_layout_matches_yaml():
    from vibewarp.vendor.cldm.cldm import SD15_CONTROLNET_PARAMS
    assert SD15_CONTROLNET_PARAMS['num_heads'] == 8
    assert SD15_CONTROLNET_PARAMS['num_head_channels'] == -1
    assert SD15_CONTROLNET_PARAMS['legacy'] is False


class TestModelConfigs:
    def test_has_v15(self):
        assert 'control_multi_v15' in MODEL_CONFIGS
        assert 'v1_5' in MODEL_CONFIGS

    def test_has_v21(self):
        assert 'control_multi_v21' in MODEL_CONFIGS
        assert 'v2_1' in MODEL_CONFIGS

    def test_values_are_yaml(self):
        for key, val in MODEL_CONFIGS.items():
            assert val.endswith('.yaml'), f"{key} config should be a .yaml file"


class TestCFGDenoiser:
    def test_init(self):
        inner = torch.nn.Identity()
        cfg = CFGDenoiser(inner)
        assert cfg.inner_model is inner

    def test_is_module(self):
        cfg = CFGDenoiser(torch.nn.Identity())
        assert isinstance(cfg, torch.nn.Module)

    def test_forward_signature(self):
        """CFGDenoiser.forward should accept x, sigma, uncond, cond, cfg_scale."""
        import inspect
        sig = inspect.signature(CFGDenoiser.forward)
        params = list(sig.parameters.keys())
        assert 'x' in params
        assert 'sigma' in params
        assert 'uncond' in params
        assert 'cond' in params
        assert 'cfg_scale' in params

    def test_forward_with_mock_model(self):
        """Test CFGDenoiser forward pass with a simple mock inner model."""
        class MockModel(torch.nn.Module):
            def forward(self, x, sigma, cond=None, **kwargs):
                return x  # identity

        inner = MockModel()
        cfg = CFGDenoiser(inner)

        batch = 1
        x = torch.randn(batch, 4, 8, 8)
        sigma = torch.ones(batch)
        uncond = torch.randn(batch, 1, 768)
        cond = torch.randn(batch, 1, 768)

        result = cfg(x, sigma, uncond=uncond, cond=cond, cfg_scale=7.5)
        assert result.shape == x.shape

    def test_cfg_scale_effect(self):
        """Higher cfg_scale should amplify difference between cond and uncond."""
        class MockModel(torch.nn.Module):
            def forward(self, x, sigma, cond=None, **kwargs):
                # Return different outputs for uncond vs cond half
                batch_half = x.shape[0] // 2
                uncond_out = torch.zeros_like(x[:batch_half])
                cond_out = torch.ones_like(x[batch_half:])
                return torch.cat([uncond_out, cond_out])

        inner = MockModel()
        cfg = CFGDenoiser(inner)

        x = torch.randn(1, 4, 8, 8)
        sigma = torch.ones(1)
        uncond = torch.zeros(1, 1, 768)
        cond = torch.ones(1, 1, 768)

        result_low = cfg(x, sigma, uncond=uncond, cond=cond, cfg_scale=1.0)
        result_high = cfg(x, sigma, uncond=uncond, cond=cond, cfg_scale=10.0)

        # Higher cfg should produce larger magnitude output
        assert result_high.abs().mean() > result_low.abs().mean()

    def test_sd_batch_size_one_uses_separate_unet_calls(self):
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.batch_sizes = []
            def forward(self, x, sigma, cond=None, **kwargs):
                self.batch_sizes.append(x.shape[0])
                return x
        inner = MockModel()
        cfg = CFGDenoiser(inner)
        cfg._sd_batch_size = 1
        x = torch.randn(1, 4, 8, 8)
        cfg(x, torch.ones(1), uncond=torch.zeros(1, 1, 768),
            cond=torch.ones(1, 1, 768), cfg_scale=7.0)
        assert inner.batch_sizes == [1, 1]


class TestLoadSdModel:
    def test_raises_on_missing_file(self):
        """load_sd_model should fail gracefully with bad paths."""
        with pytest.raises(Exception):
            load_sd_model(
                ckpt_path='/nonexistent/model.safetensors',
                config_path='/nonexistent/config.yaml',
            )


class TestWrapModelForKdiffusion:
    def test_raises_without_model(self):
        """wrap_model_for_kdiffusion needs a real model object."""
        with pytest.raises(Exception):
            wrap_model_for_kdiffusion(None)


class TestLoadControlnet:
    def test_raises_on_missing_file(self):
        with pytest.raises(Exception):
            load_controlnet('/nonexistent/controlnet.pth')


class TestCacheFingerprint:
    """The pickle bakes in classes from torch/transformers/open_clip, so a version
    change can invalidate it even when the vendored source is untouched — this bit
    us for real when open_clip 2.24 vs 3.3 swapped the text encoder's tensor layout
    and a stale pickle would have silently reused the wrong one."""

    def test_tracks_library_versions(self, monkeypatch):
        import transformers

        from vibewarp.core.model_loader import _code_fingerprint

        before = _code_fingerprint()
        monkeypatch.setattr(transformers, '__version__', '9.9.9-not-a-real-version')
        assert _code_fingerprint() != before

    def test_is_stable_across_calls(self):
        from vibewarp.core.model_loader import _code_fingerprint
        assert _code_fingerprint() == _code_fingerprint()

    def test_fingerprint_reaches_the_cache_key(self, monkeypatch):
        from vibewarp.core import model_loader as ml

        monkeypatch.setattr(ml, '_code_fingerprint', lambda: 'aaaaaaaa')
        first = ml._cache_path('model.safetensors', None, True)
        monkeypatch.setattr(ml, '_code_fingerprint', lambda: 'bbbbbbbb')
        assert ml._cache_path('model.safetensors', None, True) != first


class TestSkipPretrainedTextEncoders:
    """The ldm/sgm configs download pretrained CLIP-L (~1.7GB) and OpenCLIP bigG
    (~10GB), then load_state_dict overwrites every one of those tensors with the
    checkpoint's own encoders. Skipping the download is a numeric no-op that also
    removes a hidden network dependency at render time."""

    def test_patches_and_restores_cleanly(self):
        import inspect
        import open_clip
        import transformers
        from vibewarp.core.model_loader import _skip_pretrained_text_encoders

        original_open_clip = open_clip.create_model_and_transforms
        assert 'from_pretrained' not in transformers.CLIPTextModel.__dict__

        with _skip_pretrained_text_encoders():
            assert 'from_pretrained' in transformers.CLIPTextModel.__dict__
            assert open_clip.create_model_and_transforms is not original_open_clip

        # from_pretrained is INHERITED from PreTrainedModel: the patch shadows it,
        # so the restore must delete the shadow rather than setattr a bound method
        # back onto the class (which would break the classmethod for subclasses).
        assert 'from_pretrained' not in transformers.CLIPTextModel.__dict__
        assert inspect.ismethod(transformers.CLIPTextModel.from_pretrained)
        assert open_clip.create_model_and_transforms is original_open_clip

    def test_restores_even_if_the_body_raises(self):
        import open_clip
        import transformers
        from vibewarp.core.model_loader import _skip_pretrained_text_encoders

        original_open_clip = open_clip.create_model_and_transforms
        with pytest.raises(RuntimeError):
            with _skip_pretrained_text_encoders():
                raise RuntimeError("instantiate failed")
        assert 'from_pretrained' not in transformers.CLIPTextModel.__dict__
        assert open_clip.create_model_and_transforms is original_open_clip

    def test_open_clip_pretrained_is_forced_off(self, monkeypatch):
        import open_clip
        from vibewarp.core.model_loader import _skip_pretrained_text_encoders

        seen = {}
        monkeypatch.setattr(open_clip, 'create_model_and_transforms',
                            lambda *a, **kw: seen.update(kw) or ('m', None, None))
        with _skip_pretrained_text_encoders():
            open_clip.create_model_and_transforms('ViT-bigG-14',
                                                  pretrained='laion2b_s39b_b160k')
        assert seen['pretrained'] is None

    def test_missing_text_encoder_keys_filters_by_prefix(self):
        from vibewarp.core.model_loader import _missing_text_encoder_keys

        missing = [
            'model.diffusion_model.input_blocks.0.0.weight',   # UNet — not our concern
            'cond_stage_model.transformer.text_model.x.weight',  # SD1.5 encoder
            'conditioner.embedders.1.model.ln_final.weight',     # SDXL bigG
        ]
        found = _missing_text_encoder_keys(missing)
        assert found == [
            'cond_stage_model.transformer.text_model.x.weight',
            'conditioner.embedders.1.model.ln_final.weight',
        ]

    def test_no_missing_keys_when_checkpoint_is_complete(self):
        from vibewarp.core.model_loader import _missing_text_encoder_keys
        assert _missing_text_encoder_keys(['model.diffusion_model.out.weight']) == []
