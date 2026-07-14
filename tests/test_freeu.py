"""Tests for FreeU (notebook Fourier_filter / apply_freeu): filter math,
channel-stage modulation, set/clear helpers, config and settings mapping."""

import json

import numpy as np
import pytest
import torch

from vibewarp.config import FreeUConfig, RunConfig
from vibewarp.core.freeu import (
    Fourier_filter,
    apply_freeu,
    clear_freeu,
    set_freeu,
)


class TestFourierFilter:
    def test_scale_one_is_identity(self):
        x = torch.randn(1, 4, 16, 16)
        out = Fourier_filter(x, threshold=1, scale=1.0)
        assert torch.allclose(out, x, atol=1e-5)

    def test_scale_zero_removes_low_frequencies(self):
        # constant image = pure DC component → scale=0 kills it
        x = torch.ones(1, 2, 16, 16)
        out = Fourier_filter(x, threshold=1, scale=0.0)
        assert out.abs().max() < 1e-5

    def test_high_frequencies_untouched(self):
        # alternating checkerboard = highest frequency → untouched by the
        # center (low-frequency) mask
        x = torch.zeros(1, 1, 16, 16)
        x[..., ::2, ::2] = 1.0
        x[..., 1::2, 1::2] = 1.0
        x = x - x.mean()  # remove DC so only high-freq remains
        out = Fourier_filter(x, threshold=1, scale=0.0)
        assert torch.allclose(out, x, atol=1e-4)

    def test_scale_damps_energy(self):
        x = torch.randn(1, 4, 32, 32)
        out = Fourier_filter(x, threshold=1, scale=0.2)
        assert out.pow(2).sum() < x.pow(2).sum()


class TestApplyFreeu:
    def test_1280_stage(self):
        h = torch.ones(1, 1280, 8, 8)
        hs = torch.ones(1, 1280, 8, 8)
        h2, hs2 = apply_freeu(h.clone(), hs, b1=1.2, b2=1.4, s1=0.9, s2=0.2)
        assert torch.allclose(h2[:, :640], torch.full((1, 640, 8, 8), 1.2))
        assert torch.allclose(h2[:, 640:], torch.ones(1, 640, 8, 8))
        # skip: DC-dominated constant → scaled by s1
        assert hs2.mean().item() == pytest.approx(0.9, abs=1e-4)

    def test_640_stage(self):
        h = torch.ones(1, 640, 8, 8)
        hs = torch.ones(1, 640, 8, 8)
        h2, hs2 = apply_freeu(h.clone(), hs, b1=1.2, b2=1.4, s1=0.9, s2=0.2)
        assert torch.allclose(h2[:, :320], torch.full((1, 320, 8, 8), 1.4))
        assert torch.allclose(h2[:, 320:], torch.ones(1, 320, 8, 8))
        assert hs2.mean().item() == pytest.approx(0.2, abs=1e-4)

    def test_other_stages_untouched(self):
        h = torch.ones(1, 320, 8, 8)
        hs = torch.ones(1, 320, 8, 8)
        h2, hs2 = apply_freeu(h.clone(), hs)
        assert torch.allclose(h2, h)
        assert hs2 is hs  # not filtered


class TestSetClear:
    def test_set_and_clear(self):
        class FakeUNet:
            pass

        unet = FakeUNet()
        set_freeu(unet, FreeUConfig(do_freeunet=True, b1=1.1, s2=0.3,
                                    apply_freeu_after_control=True))
        assert unet._freeu == {'after_control': True, 'b1': 1.1, 'b2': 1.4,
                               's1': 0.9, 's2': 0.3}
        clear_freeu(unet)
        assert not hasattr(unet, '_freeu')
        clear_freeu(unet)  # idempotent


class TestConfigAndSettings:
    def test_defaults_match_notebook(self):
        c = FreeUConfig()
        assert c.do_freeunet is False
        assert c.apply_freeu_after_control is False
        assert (c.b1, c.b2, c.s1, c.s2) == (1.2, 1.4, 0.9, 0.2)
        assert isinstance(RunConfig().freeu, FreeUConfig)

    def test_settings_mapping(self, tmp_path):
        from vibewarp.settings import load_warpfusion_settings
        p = tmp_path / 's.txt'
        p.write_text(json.dumps({
            'do_freeunet': True,
            'apply_freeu_after_control': True,
            'b1': 1.1, 'b2': 1.3, 's1': 0.8, 's2': 0.25,
        }))
        cfg = load_warpfusion_settings(str(p))
        f = cfg['freeu']
        assert f['do_freeunet'] is True
        assert f['apply_freeu_after_control'] is True
        assert (f['b1'], f['b2'], f['s1'], f['s2']) == (1.1, 1.3, 0.8, 0.25)

    def test_settings_defaults(self, tmp_path):
        from vibewarp.settings import load_warpfusion_settings
        p = tmp_path / 's.txt'
        p.write_text(json.dumps({}))
        cfg = load_warpfusion_settings(str(p))
        assert cfg['freeu']['do_freeunet'] is False
        assert cfg['freeu']['b1'] == 1.2
