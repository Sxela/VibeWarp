"""Tests for SDXL basic support."""

import pytest
import torch
import torch.nn as nn

from vibewarp.core.model_loader import (
    CFGDenoiser,
    MODEL_CONFIGS,
    is_sdxl_model,
    setup_sdxl_model,
    split_sdxl_cond,
)


class TestIsSDXLModel:
    def test_sdxl_base(self):
        assert is_sdxl_model('sdxl_base') is True

    def test_sdxl_refiner(self):
        assert is_sdxl_model('sdxl_refiner') is True

    def test_control_multi_sdxl(self):
        assert is_sdxl_model('control_multi_sdxl') is True

    def test_control_multi_animatediff_sdxl(self):
        assert is_sdxl_model('control_multi_animatediff_sdxl') is True

    def test_sd15_not_sdxl(self):
        assert is_sdxl_model('control_multi_v15') is False

    def test_v21_not_sdxl(self):
        assert is_sdxl_model('control_multi_v21') is False

    def test_case_insensitive(self):
        assert is_sdxl_model('SDXL_BASE') is True
        assert is_sdxl_model('Sdxl_Refiner') is True


class TestModelConfigs:
    def test_sdxl_configs_present(self):
        assert 'sdxl_base' in MODEL_CONFIGS
        assert 'sdxl_refiner' in MODEL_CONFIGS
        assert 'control_multi_sdxl' in MODEL_CONFIGS
        assert 'control_multi_animatediff_sdxl' in MODEL_CONFIGS

    def test_sdxl_base_config_path(self):
        assert MODEL_CONFIGS['sdxl_base'] == 'sd_xl_base.yaml'

    def test_sdxl_refiner_config_path(self):
        assert MODEL_CONFIGS['sdxl_refiner'] == 'sd_xl_refiner.yaml'


class TestCFGDenoiserSDXL:
    def _make_mock_inner(self):
        class MockInner(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner_model = None

            def forward(self, x, sigma, cond=None, **kwargs):
                # Return cond-weighted x for testing
                return x * 0.5

        return MockInner()

    def test_dict_cond_extracts_crossattn(self):
        """SDXL dict cond should extract crossattn for the inner model."""
        inner = self._make_mock_inner()
        denoiser = CFGDenoiser(inner)

        x = torch.randn(1, 4, 8, 8)
        sigma = torch.tensor([1.0])
        cond = {
            'crossattn': torch.randn(1, 77, 768),
            'vector': torch.randn(1, 1, 256),
        }
        uncond = {
            'crossattn': torch.randn(1, 77, 768),
            'vector': torch.randn(1, 1, 256),
        }
        result = denoiser(x, sigma, uncond=uncond, cond=cond, cfg_scale=7.5)
        assert result.shape == (1, 4, 8, 8)

    def test_tensor_cond_still_works(self):
        """Standard SD1.5 tensor cond should still work."""
        inner = self._make_mock_inner()
        denoiser = CFGDenoiser(inner)

        x = torch.randn(1, 4, 8, 8)
        sigma = torch.tensor([1.0])
        cond = torch.randn(1, 77, 768)
        uncond = torch.randn(1, 77, 768)
        result = denoiser(x, sigma, uncond=uncond, cond=cond, cfg_scale=7.5)
        assert result.shape == (1, 4, 8, 8)

    def test_image_cond_forwarded(self):
        inner = self._make_mock_inner()
        denoiser = CFGDenoiser(inner)

        x = torch.randn(1, 4, 8, 8)
        sigma = torch.tensor([1.0])
        cond = torch.randn(1, 77, 768)
        uncond = torch.randn(1, 77, 768)
        ic = torch.randn(1, 3, 64, 64)
        result = denoiser(
            x, sigma, uncond=uncond, cond=cond,
            cfg_scale=7.5, image_cond=ic,
        )
        assert result.shape == (1, 4, 8, 8)


class _MockConditioner:
    """Minimal mock of SDXL's conditioner."""
    def __init__(self):
        self.vector_in = None
        self.embedders = [_MockEmbedder('prompt'), _MockEmbedder('original_size_as_tuple')]

    def __call__(self, batch):
        return {
            'crossattn': torch.randn(1, 77, 1024),
            'vector': torch.randn(1, 1, 256),
        }


class _MockEmbedder:
    def __init__(self, input_key):
        self.input_key = input_key
        self.ucg_rate = 0.0


class _MockUNet(torch.nn.Module):
    """Stands in for sgm's UNetModel, which deliberately has no `.dtype`."""
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 1)


class _MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = _MockUNet()

    def forward(self, x, t, cond, **kwargs):
        return torch.randn_like(x)


class _MockSDXLModel(torch.nn.Module):
    """Minimal mock of an SDXL sd_model.

    An nn.Module (not a bare object) so setup_sdxl_model exercises the real
    register_buffer / parameters() paths it depends on.
    """
    def __init__(self):
        super().__init__()
        self.conditioner = _MockConditioner()
        self.model = _MockModel()
        self.parameterization = 'eps'

    def get_learned_conditioning(self, prompts):
        return torch.randn(len(prompts), 77, 768)

    def apply_model(self, x, t, cond=None, **kwargs):
        return torch.randn_like(x)

    def get_first_stage_encoding(self, z):
        return z * 0.18215


class TestSetupSDXLModel:
    def test_patches_apply_model(self):
        model = _MockSDXLModel()
        original = model.apply_model
        setup_sdxl_model(model, 1024, 1024)
        assert model.apply_model is not original

    def test_patches_get_first_stage_encoding(self):
        model = _MockSDXLModel()
        setup_sdxl_model(model, 1024, 1024)
        z = torch.randn(1, 4, 128, 128)
        result = model.get_first_stage_encoding(z)
        # SDXL identity encoding
        assert torch.equal(result, z)

    def test_patches_get_learned_conditioning(self):
        model = _MockSDXLModel()
        setup_sdxl_model(model, 1024, 1024)
        result = model.get_learned_conditioning(['a test prompt'])
        assert isinstance(result, dict)
        assert 'crossattn' in result
        assert 'vector' in result

    def test_cond_stage_model_alias(self):
        model = _MockSDXLModel()
        setup_sdxl_model(model, 1024, 1024)
        assert model.cond_stage_model is model.conditioner

    def test_no_conditioner_is_noop(self):
        """Model without conditioner should not crash."""
        class PlainModel:
            pass
        model = PlainModel()
        setup_sdxl_model(model, 512, 512)  # Should not raise

    def test_apply_model_with_tensor_cond(self):
        model = _MockSDXLModel()
        setup_sdxl_model(model, 1024, 1024)
        model.conditioner.vector_in = torch.randn(1, 1, 256)
        x = torch.randn(1, 4, 128, 128)
        t = torch.tensor([100.0])
        cond = torch.randn(1, 77, 1024)
        result = model.apply_model(x, t, cond)
        assert result.shape == x.shape

    def test_apply_model_with_dict_cond(self):
        model = _MockSDXLModel()
        setup_sdxl_model(model, 1024, 1024)
        model.conditioner.vector_in = torch.randn(1, 1, 256)
        x = torch.randn(1, 4, 128, 128)
        t = torch.tensor([100.0])
        cond = {
            'c_crossattn': [torch.randn(1, 77, 1024)],
        }
        result = model.apply_model(x, t, cond)
        assert result.shape == x.shape

    def test_apply_model_with_y_key(self):
        model = _MockSDXLModel()
        setup_sdxl_model(model, 1024, 1024)
        model.conditioner.vector_in = torch.randn(1, 1, 256)
        y = torch.randn(1, 1, 256)
        x = torch.randn(1, 4, 128, 128)
        t = torch.tensor([100.0])
        cond = {
            'c_crossattn': [torch.randn(1, 77, 1024)],
            'y': y,
        }
        result = model.apply_model(x, t, cond)
        assert result.shape == x.shape


class TestSDXLRenderBlockers:
    """The four bugs that made SDXL impossible to render (fixed 2026-07-12).

    setup_sdxl_model existed but was never called by the pipeline, so none of
    these had ever executed.
    """

    def test_registers_alphas_cumprod(self):
        """sgm's DiffusionEngine has no alphas_cumprod, but k-diffusion's
        CompVisDenoiser reads it at construction:
        AttributeError: 'DiffusionEngine' object has no attribute 'alphas_cumprod'."""
        model = _MockSDXLModel()
        assert not hasattr(model, 'alphas_cumprod')
        setup_sdxl_model(model, 1024, 1024)
        assert model.alphas_cumprod.shape == (1000,)
        # float32 on the model's device, or CompVisDenoiser derives its sigmas on
        # the CPU while x is on the GPU -> "two devices, cuda:0 and cpu".
        assert model.alphas_cumprod.dtype == torch.float32
        assert model.alphas_cumprod.device == next(model.parameters()).device

    def test_does_not_hijack_an_already_registered_schedule(self):
        """AnimateDiff (non-Hotshot) registers its OWN betas before this runs, so
        setup_sdxl_model must leave them alone."""
        model = _MockSDXLModel()
        model.register_buffer('alphas_cumprod', torch.full((1000,), 0.5))
        setup_sdxl_model(model, 1024, 1024)
        assert torch.allclose(model.alphas_cumprod, torch.full((1000,), 0.5))

    def test_hotshot_keeps_the_standard_schedule(self):
        """HotshotXL deliberately does NOT replace alphas_cumprod (notebook:
        `if 'animatediff' in model_version and not mm.is_hotshot`). `ad_alphas_cumprod`
        is set for Hotshot too, so preferring it here would silently give Hotshot the
        wrong noise schedule."""
        model = _MockSDXLModel()
        model.ad_alphas_cumprod = torch.full((1000,), 0.5)   # set, but NOT registered
        setup_sdxl_model(model, 1024, 1024)
        assert not torch.allclose(model.alphas_cumprod, torch.full((1000,), 0.5))
        # LegacyDDPMDiscretization: alphas_cumprod decreases from ~1 toward 0.
        assert model.alphas_cumprod[0] > 0.99
        assert model.alphas_cumprod[-1] < 0.01

    def test_gives_the_unet_a_dtype(self):
        """sgm's UNetModel dropped self.dtype, but the shared ControlNet forward
        casts with x.type(self.dtype):
        AttributeError: 'UNetModel' object has no attribute 'dtype'."""
        model = _MockSDXLModel()
        assert not hasattr(model.model.diffusion_model, 'dtype')
        setup_sdxl_model(model, 1024, 1024)
        unet = model.model.diffusion_model
        assert unet.dtype == next(unet.parameters()).dtype

    def test_sets_conditioning_key(self):
        model = _MockSDXLModel()
        setup_sdxl_model(model, 1024, 1024)
        assert model.model.conditioning_key == 'c_crossattn'


class TestSDXLGetBatch:
    """The conditioner is keyed by embedder input_key ('txt' + size tensors).
    The old code imported a nonexistent `sgm.util.default_cond_input` and its
    ImportError fallback silently built a batch with no 'txt' -> KeyError: 'txt'."""

    def _value_dict(self):
        return {
            'prompt': ['a prompt'],
            'negative_prompt': [''],
            'orig_width': 1024, 'orig_height': 768,
            'target_width': 1024, 'target_height': 768,
            'crop_coords_top': 0, 'crop_coords_left': 0,
            'aesthetic_score': 6.0, 'negative_aesthetic_score': 2.5,
        }

    def test_builds_txt_and_size_tensors(self):
        from vibewarp.core.model_loader import sdxl_get_batch
        keys = ['txt', 'original_size_as_tuple', 'crop_coords_top_left',
                'target_size_as_tuple']
        batch, batch_uc = sdxl_get_batch(keys, self._value_dict(), [1], device='cpu')

        assert batch['txt'] == ['a prompt']
        assert batch_uc['txt'] == ['']
        # Height first — the order is load-bearing for SDXL micro-conditioning.
        assert torch.equal(batch['original_size_as_tuple'], torch.tensor([[768, 1024]]))
        assert torch.equal(batch['target_size_as_tuple'], torch.tensor([[768, 1024]]))
        assert torch.equal(batch['crop_coords_top_left'], torch.tensor([[0, 0]]))

    def test_uncond_clones_non_text_tensors(self):
        from vibewarp.core.model_loader import sdxl_get_batch
        batch, batch_uc = sdxl_get_batch(
            ['txt', 'original_size_as_tuple'], self._value_dict(), [1], device='cpu')
        assert torch.equal(batch_uc['original_size_as_tuple'],
                           batch['original_size_as_tuple'])

    def test_repeats_prompt_to_batch_size(self):
        from vibewarp.core.model_loader import sdxl_get_batch
        batch, _ = sdxl_get_batch(['txt'], self._value_dict(), [3], device='cpu')
        assert batch['txt'] == ['a prompt'] * 3

    def test_aesthetic_score_for_refiner(self):
        from vibewarp.core.model_loader import sdxl_get_batch
        batch, batch_uc = sdxl_get_batch(
            ['aesthetic_score'], self._value_dict(), [1], device='cpu')
        assert torch.equal(batch['aesthetic_score'], torch.tensor([[6.0]]))
        assert torch.equal(batch_uc['aesthetic_score'], torch.tensor([[2.5]]))
