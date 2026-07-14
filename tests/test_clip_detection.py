"""Tests for CLIP vision model auto-detection."""

import torch
import pytest

from vibewarp.core.ipadapter import (
    detect_clip_variant,
    get_clip_model_url,
    VIT_H_VARIANTS,
    VIT_G_VARIANTS,
)


class TestDetectClipVariant:
    def test_known_vit_h_variants(self):
        for name in VIT_H_VARIANTS:
            assert detect_clip_variant(name) == 'vit_h', f"Failed for {name}"

    def test_known_vit_g_variants(self):
        for name in VIT_G_VARIANTS:
            assert detect_clip_variant(name) == 'vit_g', f"Failed for {name}"

    def test_name_with_vit_h_substring(self):
        assert detect_clip_variant('custom_model_vit_h_v3') == 'vit_h'
        assert detect_clip_variant('my_adapter_vit-h') == 'vit_h'

    def test_name_with_vit_g_substring(self):
        assert detect_clip_variant('custom_vit_g_model') == 'vit_g'
        assert detect_clip_variant('adapter_vit-G') == 'vit_g'
        assert detect_clip_variant('clip_vit_bigg') == 'vit_g'

    def test_state_dict_1024_dim(self):
        sd = {'image_proj': {'proj.weight': torch.randn(768, 1024)}}
        assert detect_clip_variant(state_dict=sd) == 'vit_h'

    def test_state_dict_1280_dim(self):
        sd = {'image_proj': {'proj.weight': torch.randn(768, 1280)}}
        assert detect_clip_variant(state_dict=sd) == 'vit_g'

    def test_unknown_name_no_state_dict_defaults_vit_h(self):
        assert detect_clip_variant('totally_unknown_model') == 'vit_h'

    def test_empty_name_defaults_vit_h(self):
        assert detect_clip_variant('') == 'vit_h'

    def test_name_takes_priority_over_state_dict(self):
        # Name says vit_g, state dict says 1024 — name wins
        sd = {'image_proj': {'proj.weight': torch.randn(768, 1024)}}
        assert detect_clip_variant('ipadapter_sd15_vit_G', sd) == 'vit_g'

    def test_case_insensitive_name_matching(self):
        assert detect_clip_variant('IPADAPTER_SD15') == 'vit_h'
        assert detect_clip_variant('IpAdapter_SD15_VIT_G') == 'vit_g'

    def test_state_dict_with_no_weight_keys(self):
        sd = {'image_proj': {'bias': torch.randn(768)}}
        assert detect_clip_variant('', sd) == 'vit_h'

    def test_state_dict_with_1d_weight(self):
        sd = {'image_proj': {'weight': torch.randn(768)}}
        assert detect_clip_variant('', sd) == 'vit_h'


class TestGetClipModelUrl:
    def test_vit_h_url(self):
        url = get_clip_model_url('vit_h')
        assert 'image_encoder' in url
        assert 'models' in url

    def test_vit_g_url(self):
        url = get_clip_model_url('vit_g')
        assert 'image_encoder' in url
        assert 'sdxl_models' in url

    def test_unknown_variant_defaults_to_vit_h(self):
        url = get_clip_model_url('unknown')
        assert url == get_clip_model_url('vit_h')
