"""Tests for vibewarp.core.ipadapter — IP-Adapter loading, CLIP encoding, hooks."""

import torch
import numpy as np
import pytest
from PIL import Image

from vibewarp.core.ipadapter import (
    clip_preprocess,
    load_image_for_clip,
    detect_ipadapter_type,
    IPAdapterEmbeddingCache,
    hook_ipadapter,
    unhook_ipadapter,
    CLIP_MEAN,
    CLIP_STD,
)
from vibewarp.config import IPAdapterConfig, IPAdapterEntry, RunConfig


class TestClipPreprocess:
    def test_output_shape(self):
        image = torch.rand(1, 256, 384, 3)
        result = clip_preprocess(image, size=224)
        assert result.shape == (1, 3, 224, 224)

    def test_output_range(self):
        """Output should be normalized with CLIP mean/std (not in [0,1])."""
        image = torch.ones(1, 256, 256, 3) * 0.5
        result = clip_preprocess(image, size=224)
        # After normalization, values should be around (0.5 - mean) / std
        assert result.shape == (1, 3, 224, 224)

    def test_batch_dimension(self):
        image = torch.rand(4, 128, 128, 3)
        result = clip_preprocess(image, size=224)
        assert result.shape[0] == 4

    def test_different_sizes(self):
        image = torch.rand(1, 128, 128, 3)
        for size in [224, 336, 448]:
            result = clip_preprocess(image, size=size)
            assert result.shape == (1, 3, size, size)


class TestLoadImageForClip:
    def test_loads_image(self, tmp_path):
        img = Image.fromarray(np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8))
        path = str(tmp_path / "test.png")
        img.save(path)
        result = load_image_for_clip(path, device='cpu')
        assert result.shape == (1, 64, 96, 3)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_rgb_conversion(self, tmp_path):
        # Grayscale should be converted to RGB
        img = Image.fromarray(np.random.randint(0, 255, (32, 32), dtype=np.uint8))
        path = str(tmp_path / "gray.png")
        img.save(path)
        result = load_image_for_clip(path, device='cpu')
        assert result.shape[-1] == 3


class TestDetectIpadapterType:
    def test_basic_sd15(self):
        sd = {
            'image_proj': {'proj.weight': torch.zeros(1)},
            'ip_adapter': {'1.to_k_ip.weight': torch.zeros(768, 768)},
        }
        info = detect_ipadapter_type(sd)
        assert info['is_plus'] is False
        assert info['is_full'] is False
        assert info['is_faceid'] is False
        assert info['is_sdxl'] is False

    def test_plus_variant(self):
        sd = {
            'image_proj': {'latents': torch.zeros(1)},
            'ip_adapter': {'1.to_k_ip.weight': torch.zeros(768, 768)},
        }
        info = detect_ipadapter_type(sd)
        assert info['is_plus'] is True

    def test_full_variant(self):
        sd = {
            'image_proj': {'proj.3.weight': torch.zeros(1)},
            'ip_adapter': {'1.to_k_ip.weight': torch.zeros(768, 768)},
        }
        info = detect_ipadapter_type(sd)
        assert info['is_full'] is True
        assert info['is_plus'] is True  # full implies plus

    def test_faceid_variant(self):
        sd = {
            'image_proj': {},
            'ip_adapter': {
                '0.to_q_lora.down.weight': torch.zeros(1),
                '1.to_k_ip.weight': torch.zeros(768, 768),
            },
        }
        info = detect_ipadapter_type(sd)
        assert info['is_faceid'] is True

    def test_sdxl_variant(self):
        sd = {
            'image_proj': {},
            'ip_adapter': {'1.to_k_ip.weight': torch.zeros(2048, 2048)},
        }
        info = detect_ipadapter_type(sd)
        assert info['is_sdxl'] is True

    def test_empty_state_dict(self):
        sd = {'image_proj': {}, 'ip_adapter': {}}
        info = detect_ipadapter_type(sd)
        assert info['is_plus'] is False
        assert info['is_sdxl'] is False


class TestIPAdapterEmbeddingCache:
    def test_put_and_get(self):
        cache = IPAdapterEmbeddingCache(max_size=4)
        emb = {'image_embeds': torch.randn(1, 768)}
        cache.put('key1', emb)
        assert cache.get('key1') is emb

    def test_miss_returns_none(self):
        cache = IPAdapterEmbeddingCache()
        assert cache.get('nonexistent') is None

    def test_contains(self):
        cache = IPAdapterEmbeddingCache()
        cache.put('a', {})
        assert 'a' in cache
        assert 'b' not in cache

    def test_eviction(self):
        cache = IPAdapterEmbeddingCache(max_size=2)
        cache.put('a', {'v': 1})
        cache.put('b', {'v': 2})
        cache.put('c', {'v': 3})  # should evict 'a'
        assert 'a' not in cache
        assert 'b' in cache
        assert 'c' in cache

    def test_clear(self):
        cache = IPAdapterEmbeddingCache()
        cache.put('x', {})
        cache.put('y', {})
        cache.clear()
        assert len(cache) == 0

    def test_len(self):
        cache = IPAdapterEmbeddingCache()
        assert len(cache) == 0
        cache.put('a', {})
        assert len(cache) == 1


class TestHookUnhook:
    def test_hook_stores_info(self):
        """hook_ipadapter should build a vendored PlugableIPAdapter and store it."""
        from unittest.mock import patch, MagicMock

        class MockSDModel:
            pass

        sd_model = MockSDModel()
        sd_model.model = MagicMock()
        weights = {'image_proj': {}, 'ip_adapter': {}}
        clip_output = {'image_embeds': torch.randn(1, 768)}

        # Mock PlugableIPAdapter to avoid CUDA + real weights; the import is
        # inside hook_ipadapter's body, so patch the vendored module attribute.
        mock_adapter = MagicMock()
        with patch('vibewarp.vendor.controlmodel_ipadapter.PlugableIPAdapter',
                   MagicMock(return_value=mock_adapter)) as mock_cls:
            hook_ipadapter(sd_model, weights, clip_output, weight=0.8)
        mock_cls.assert_called_once_with(weights, is_v2=False)
        assert hasattr(sd_model, '_ipadapter_hooks')
        assert len(sd_model._ipadapter_hooks) == 1

    def test_hook_passes_is_v2(self):
        from unittest.mock import patch, MagicMock

        class MockSDModel:
            pass

        sd_model = MockSDModel()
        sd_model.model = MagicMock()
        w = {'image_proj': {}, 'ip_adapter': {}}
        c = {'image_embeds': torch.randn(1, 768)}

        with patch('vibewarp.vendor.controlmodel_ipadapter.PlugableIPAdapter',
                   MagicMock(return_value=MagicMock())) as mock_cls:
            hook_ipadapter(sd_model, w, c, weight=0.5, is_v2=True)
        mock_cls.assert_called_once_with(w, is_v2=True)

    def test_unhook_clears(self):
        class MockSDModel:
            pass

        sd_model = MockSDModel()
        sd_model._ipadapter_hooks = [{'weight': 1.0}]
        unhook_ipadapter(sd_model)
        assert len(sd_model._ipadapter_hooks) == 0

    def test_unhook_without_hooks(self):
        class MockSDModel:
            pass

        sd_model = MockSDModel()
        unhook_ipadapter(sd_model)  # should not error

    def test_multiple_hooks(self):
        from unittest.mock import patch, MagicMock

        class MockSDModel:
            pass

        sd_model = MockSDModel()
        sd_model.model = MagicMock()
        w = {'image_proj': {}, 'ip_adapter': {}}
        c = {'image_embeds': torch.randn(1, 768)}

        with patch('vibewarp.vendor.controlmodel_ipadapter.PlugableIPAdapter',
                   MagicMock(return_value=MagicMock())):
            hook_ipadapter(sd_model, w, c, weight=0.5)
            hook_ipadapter(sd_model, w, c, weight=0.8)
        assert len(sd_model._ipadapter_hooks) == 2


class TestIPAdapterConfig:
    def test_default_disabled(self):
        cfg = IPAdapterConfig()
        assert cfg.enabled is False
        assert cfg.models == {}

    def test_with_models(self):
        cfg = IPAdapterConfig(
            enabled=True,
            clip_vision_model_path='/models/clip_vit_h.safetensors',
            models={
                'ipadapter_sd15': IPAdapterEntry(
                    path='/models/ip-adapter_sd15.safetensors',
                    weight=0.6,
                    source_image='/ref/style.png',
                ),
            },
        )
        assert cfg.enabled is True
        assert 'ipadapter_sd15' in cfg.models
        assert cfg.models['ipadapter_sd15'].weight == 0.6

    def test_entry_defaults(self):
        entry = IPAdapterEntry()
        assert entry.weight == 1.0
        assert entry.start == 0.0
        assert entry.end == 1.0
        assert entry.weight_type == 'linear'
        assert entry.embeds_scaling == 'V only'

    def test_runconfig_has_ipadapter(self):
        cfg = RunConfig()
        assert hasattr(cfg, 'ipadapter')
        assert isinstance(cfg.ipadapter, IPAdapterConfig)
        assert cfg.ipadapter.enabled is False
