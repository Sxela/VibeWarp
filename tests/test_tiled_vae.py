"""Tests for tiled VAE patch/unpatch and settings mapping."""

import pytest
import torch
import types


# ---- patch/unpatch round-trip ----

class _MockPosterior:
    """Fake DiagonalGaussianDistribution that returns a known tensor on sample()."""
    def __init__(self, val):
        self._val = val

    def sample(self):
        return self._val


class _MockSdModel:
    """Minimal fake SD model for testing VAE patches."""
    scale_factor = 0.18215

    def encode_first_stage(self, x):
        # Returns a fake posterior (not a tensor)
        return _MockPosterior(x * 0.5)

    def decode_first_stage(self, z):
        return z * 2.0

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, torch.Tensor):
            return self.scale_factor * encoder_posterior
        return self.scale_factor * encoder_posterior.sample()


class TestTiledDecodeScaleFactor:
    """tiled_decode must rescale by 1/scale_factor before the raw decoder.

    Regression: the missing rescale compressed every tiled render's contrast
    by exactly scale_factor (~0.18) — washed-out gray output (found 2026-07-11
    during GPU parity work).
    """

    def test_single_tile_matches_rescaled_decode(self):
        from vibewarp.core.vae import tiled_decode

        class _Decoder:
            def decode(self, z):
                # fake VAE decoder: 8x upsample, linear transform
                return (z * 3.0).repeat_interleave(8, dim=-2).repeat_interleave(8, dim=-1)

        class _Model:
            scale_factor = 0.5
            first_stage_model = _Decoder()

        z = torch.arange(4 * 8 * 8, dtype=torch.float32).reshape(1, 4, 8, 8) / 100.0
        out = tiled_decode(_Model(), z, num_tiles=(1, 1), df=8)
        expected = _Decoder().decode(z / _Model.scale_factor)
        assert out.shape == expected.shape
        assert torch.allclose(out, expected, atol=1e-4)


class TestTiledEncodeScaleFactor:
    """tiled_encode must return a latent scaled exactly like the untiled path.

    Regression (found 2026-07-13): it called `first_stage_model.encode` directly.
    For ldm (SD1.5) that is fine — get_first_stage_encoding applies scale_factor.
    But sgm (SDXL) puts the scaling inside encode_first_stage and makes
    get_first_stage_encoding an identity, so the tiler bypassed it and SDXL latents
    came out 1/scale_factor (~7.7x) too large. Invisible for txt2img, which never
    encodes; it destroyed every SDXL img2img render (style_strength < 1).
    This is the mirror of the tiled_decode bug above.
    """

    def _sgm_model(self):
        """SDXL-shaped: the scaling lives in encode_first_stage."""
        class _Encoder:
            def encode(self, x):
                return x[:, :4, ::8, ::8]          # raw, unscaled

        class _Model:
            scale_factor = 0.13025
            first_stage_model = _Encoder()

            def encode_first_stage(self, x):        # sgm: scales here
                return self.scale_factor * self.first_stage_model.encode(x)

            def get_first_stage_encoding(self, z):  # sgm: identity
                return z

        return _Model()

    def _ldm_model(self):
        """SD1.5-shaped: the scaling lives in get_first_stage_encoding."""
        class _Encoder:
            def encode(self, x):
                return x[:, :4, ::8, ::8]

        class _Model:
            scale_factor = 0.18215
            first_stage_model = _Encoder()

            def encode_first_stage(self, x):        # ldm: no scaling here
                return self.first_stage_model.encode(x)

            def get_first_stage_encoding(self, z):  # ldm: scales here
                return self.scale_factor * z

        return _Model()

    def test_sgm_latent_is_scaled(self):
        from vibewarp.core.vae import tiled_encode
        model = self._sgm_model()
        x = torch.ones(1, 4, 64, 64)
        out = tiled_encode(model, x, num_tiles=(1, 1), df=8)
        expected = model.encode_first_stage(x)
        assert torch.allclose(out, expected, atol=1e-4), (
            "SDXL tiled latent must be scaled — an unscaled one is ~7.7x too large "
            "and destroys img2img")

    def test_ldm_latent_is_scaled_and_not_double_scaled(self):
        from vibewarp.core.vae import tiled_encode
        model = self._ldm_model()
        x = torch.ones(1, 4, 64, 64)
        out = tiled_encode(model, x, num_tiles=(1, 1), df=8)
        expected = model.get_first_stage_encoding(model.encode_first_stage(x))
        assert torch.allclose(out, expected, atol=1e-4)

    def test_both_architectures_agree_on_magnitude(self):
        """Whatever the architecture, a tiled latent has the same magnitude as the
        untiled one — that is the invariant both bugs broke."""
        from vibewarp.core.vae import tiled_encode
        x = torch.ones(1, 4, 64, 64)
        for model in (self._sgm_model(), self._ldm_model()):
            untiled = model.get_first_stage_encoding(model.encode_first_stage(x))
            tiled = tiled_encode(model, x, num_tiles=(1, 1), df=8)
            assert torch.allclose(tiled.std(), untiled.std(), atol=1e-4)


class TestPatchUnpatch:
    def _model(self):
        return _MockSdModel()

    def _vae_config(self, **kw):
        from vibewarp.config import VaeConfig
        return VaeConfig(use_tiled_vae=True, **kw)

    def test_methods_replaced_after_patch(self):
        from vibewarp.core.vae import patch_model_for_tiled_vae
        m = self._model()
        orig_enc = m.encode_first_stage
        patch_model_for_tiled_vae(m, self._vae_config())
        assert m.encode_first_stage is not orig_enc

    def test_originals_saved(self):
        from vibewarp.core.vae import patch_model_for_tiled_vae
        m = self._model()
        patch_model_for_tiled_vae(m, self._vae_config())
        assert hasattr(m, '_orig_encode_first_stage')
        assert hasattr(m, '_orig_decode_first_stage')
        assert hasattr(m, '_orig_get_first_stage_encoding')

    def test_unpatch_restores_methods(self):
        from vibewarp.core.vae import patch_model_for_tiled_vae, unpatch_model_tiled_vae
        m = self._model()
        orig_enc = m.encode_first_stage
        orig_dec = m.decode_first_stage
        patch_model_for_tiled_vae(m, self._vae_config())
        unpatch_model_tiled_vae(m)
        assert m.encode_first_stage == orig_enc
        assert m.decode_first_stage == orig_dec
        assert not hasattr(m, '_orig_encode_first_stage')

    def test_unpatch_noop_when_not_patched(self):
        from vibewarp.core.vae import unpatch_model_tiled_vae
        m = self._model()
        unpatch_model_tiled_vae(m)  # should not raise


class TestPatchedGetFirstStageEncoding:
    """Patched get_first_stage_encoding returns tensors as-is (no double-scaling)."""

    def test_tensor_input_returned_as_is(self):
        from vibewarp.core.vae import patch_model_for_tiled_vae
        from vibewarp.config import VaeConfig
        m = _MockSdModel()
        patch_model_for_tiled_vae(m, VaeConfig(use_tiled_vae=True))
        t = torch.ones(1, 4, 8, 8) * 3.0
        result = m.get_first_stage_encoding(t)
        assert torch.allclose(result, t)

    def test_posterior_input_still_sampled_and_scaled(self):
        from vibewarp.core.vae import patch_model_for_tiled_vae
        from vibewarp.config import VaeConfig
        m = _MockSdModel()
        patch_model_for_tiled_vae(m, VaeConfig(use_tiled_vae=True))
        raw = torch.ones(1, 4, 8, 8)
        posterior = _MockPosterior(raw)
        result = m.get_first_stage_encoding(posterior)
        expected = m.scale_factor * raw
        assert torch.allclose(result, expected)


# ---- VaeConfig defaults ----

class TestVaeConfigDefaults:
    def test_defaults(self):
        from vibewarp.config import VaeConfig
        cfg = VaeConfig()
        assert cfg.use_tiled_vae is False
        assert cfg.num_tiles is None
        assert cfg.tile_size == 128
        assert cfg.vae_stride == 96
        assert cfg.vae_padding is None

    def test_in_run_config(self):
        from vibewarp.config import RunConfig
        rc = RunConfig()
        assert hasattr(rc, 'vae')
        assert not rc.vae.use_tiled_vae


# ---- settings mapping ----

class TestVaeSettingsMapping:
    def _write_settings(self, tmp_path, extra=None):
        import json
        d = {
            'model_path': 'm.safetensors',
            'text_prompts': {'0': ['a painting']},
            'negative_prompts': {'0': ['']},
        }
        if extra:
            d.update(extra)
        p = str(tmp_path / 's.txt')
        with open(p, 'w') as f:
            json.dump(d, f)
        return p

    def test_defaults_when_absent(self, tmp_path):
        from vibewarp.settings import load_warpfusion_settings
        p = self._write_settings(tmp_path)
        cfg = load_warpfusion_settings(p)
        assert cfg['vae']['use_tiled_vae'] is False
        assert cfg['vae']['tile_size'] == 128
        assert cfg['vae']['vae_stride'] == 96
        assert cfg['vae']['num_tiles'] is None
        assert cfg['vae']['vae_padding'] is None

    def test_values_mapped(self, tmp_path):
        from vibewarp.settings import load_warpfusion_settings
        p = self._write_settings(tmp_path, {
            'use_tiled_vae': True,
            'num_tiles': [2, 2],
            'tile_size': 64,
            'vae_stride': 48,
            'vae_padding': [0.5, 0.5],
        })
        cfg = load_warpfusion_settings(p)
        assert cfg['vae']['use_tiled_vae'] is True
        assert cfg['vae']['num_tiles'] == [2, 2]
        assert cfg['vae']['tile_size'] == 64
        assert cfg['vae']['vae_stride'] == 48
        assert cfg['vae']['vae_padding'] == [0.5, 0.5]
