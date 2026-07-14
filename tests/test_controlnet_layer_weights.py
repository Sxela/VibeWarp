"""Tests for layer-wise ControlNet weights."""

import pytest
import torch

from vibewarp.config import ControlNetEntry
from vibewarp.core.controlnet import (
    apply_layer_weights,
    normalize_controlnet_weights,
    DEFAULT_NUM_CN_LAYERS,
)


class TestApplyLayerWeights:
    def test_empty_outputs(self):
        result = apply_layer_weights([], layer_weights=[1.0, 0.5])
        assert result == []

    def test_global_weight_only(self):
        outputs = [torch.ones(1, 64, 8, 8), torch.ones(1, 128, 4, 4)]
        result = apply_layer_weights(outputs, global_weight=0.5)
        assert torch.allclose(result[0], torch.ones(1, 64, 8, 8) * 0.5)
        assert torch.allclose(result[1], torch.ones(1, 128, 4, 4) * 0.5)

    def test_per_layer_weights(self):
        outputs = [torch.ones(1, 64, 8, 8)] * 3
        result = apply_layer_weights(outputs, layer_weights=[0.5, 1.0, 0.25])
        assert torch.allclose(result[0], torch.ones(1, 64, 8, 8) * 0.5)
        assert torch.allclose(result[1], torch.ones(1, 64, 8, 8) * 1.0)
        assert torch.allclose(result[2], torch.ones(1, 64, 8, 8) * 0.25)

    def test_truncates_excess_weights(self):
        outputs = [torch.ones(1, 64, 8, 8)] * 2
        result = apply_layer_weights(outputs, layer_weights=[0.5, 1.0, 0.25, 0.75])
        assert len(result) == 2

    def test_pads_missing_weights_with_global(self):
        outputs = [torch.ones(1, 64, 8, 8)] * 3
        result = apply_layer_weights(outputs, layer_weights=[0.5], global_weight=0.8)
        assert torch.allclose(result[0], torch.ones(1, 64, 8, 8) * 0.5)
        assert torch.allclose(result[1], torch.ones(1, 64, 8, 8) * 0.8)
        assert torch.allclose(result[2], torch.ones(1, 64, 8, 8) * 0.8)

    def test_none_layer_weights_uses_global(self):
        outputs = [torch.ones(1, 64, 8, 8)] * 3
        result = apply_layer_weights(outputs, layer_weights=None, global_weight=2.0)
        for r in result:
            assert torch.allclose(r, torch.ones(1, 64, 8, 8) * 2.0)

    def test_preserves_tensor_shapes(self):
        outputs = [torch.randn(2, c, s, s) for c, s in [(64, 32), (128, 16), (256, 8)]]
        result = apply_layer_weights(outputs, layer_weights=[0.5, 1.0, 0.25])
        for orig, scaled in zip(outputs, result):
            assert scaled.shape == orig.shape


class TestNormalizeControlnetWeights:
    def test_single_model_normalizes_to_one(self):
        configs = {'depth': {'weight': 2.0}}
        result = normalize_controlnet_weights(configs)
        assert all(abs(w - 1.0) < 1e-6 for w in result['depth'])

    def test_two_equal_models_half_each(self):
        configs = {
            'depth': {'weight': 1.0},
            'canny': {'weight': 1.0},
        }
        result = normalize_controlnet_weights(configs)
        for layer_idx in range(DEFAULT_NUM_CN_LAYERS):
            assert abs(result['depth'][layer_idx] - 0.5) < 1e-6
            assert abs(result['canny'][layer_idx] - 0.5) < 1e-6

    def test_unequal_weights_normalized(self):
        configs = {
            'depth': {'weight': 3.0},
            'canny': {'weight': 1.0},
        }
        result = normalize_controlnet_weights(configs)
        assert abs(result['depth'][0] - 0.75) < 1e-6
        assert abs(result['canny'][0] - 0.25) < 1e-6

    def test_with_layer_weights(self):
        configs = {
            'depth': {'weight': 1.0, 'layer_weights': [1.0, 0.0, 1.0]},
            'canny': {'weight': 1.0, 'layer_weights': [0.0, 1.0, 1.0]},
        }
        result = normalize_controlnet_weights(configs, num_layers=3)
        # Layer 0: depth=1, canny=0 → depth=1, canny=0
        assert abs(result['depth'][0] - 1.0) < 1e-6
        assert abs(result['canny'][0] - 0.0) < 1e-6
        # Layer 1: depth=0, canny=1 → depth=0, canny=1
        assert abs(result['depth'][1] - 0.0) < 1e-6
        assert abs(result['canny'][1] - 1.0) < 1e-6
        # Layer 2: depth=1, canny=1 → each 0.5
        assert abs(result['depth'][2] - 0.5) < 1e-6
        assert abs(result['canny'][2] - 0.5) < 1e-6

    def test_empty_configs(self):
        assert normalize_controlnet_weights({}) == {}

    def test_pads_short_layer_weights(self):
        configs = {'depth': {'weight': 2.0, 'layer_weights': [0.5]}}
        result = normalize_controlnet_weights(configs, num_layers=3)
        # Layer 0: 0.5 → normalized to 1.0 (only model)
        assert abs(result['depth'][0] - 1.0) < 1e-6
        # Layer 1: padded with global weight 2.0 → normalized to 1.0
        assert abs(result['depth'][1] - 1.0) < 1e-6


class TestControlNetEntryLayerWeights:
    def test_default_none(self):
        entry = ControlNetEntry()
        assert entry.layer_weights is None

    def test_custom_layer_weights(self):
        entry = ControlNetEntry(layer_weights=[1.0, 0.5, 0.0] + [1.0] * 10)
        assert len(entry.layer_weights) == 13
        assert entry.layer_weights[1] == 0.5
