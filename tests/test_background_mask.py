"""Tests for render-time background masking (notebook cell 42 + apply_mask).

Covers BackgroundMaskConfig, WarpFusion settings mapping, and
_apply_render_bg_mask compositing (color/image/init_video backgrounds,
invert, clipping, missing-alpha fallback).
"""

import json
import os

import numpy as np
import pytest
from PIL import Image

from vibewarp.config import BackgroundMaskConfig, RunConfig
from vibewarp.core.diffusion import _apply_render_bg_mask


def _setup_frames(tmp_path, alpha_value=None, size=(32, 32)):
    """Create video_frames + _alpha dirs with frame 1 (mask_frame_num=0)."""
    frames = tmp_path / 'video_frames'
    alpha_dir = tmp_path / 'video_frames_alpha'
    frames.mkdir(parents=True)
    alpha_dir.mkdir(parents=True)
    # raw video frame — solid blue (used by 'init_video' background)
    Image.new('RGB', size, (0, 0, 255)).save(str(frames / '000001.jpg'))
    if alpha_value is not None:
        Image.new('L', size, alpha_value).save(str(alpha_dir / '000001.jpg'))
    return str(frames)


def _make_config(**mask_kwargs):
    cfg = RunConfig()
    cfg.mask = BackgroundMaskConfig(use_background_mask=True, **mask_kwargs)
    return cfg


FG = (0, 255, 0)  # solid green foreground


class TestApplyRenderBgMask:
    def test_white_alpha_keeps_foreground(self, tmp_path):
        frames = _setup_frames(tmp_path, alpha_value=255)
        cfg = _make_config(background='color', background_source='red')
        out = _apply_render_bg_mask(Image.new('RGB', (32, 32), FG), 0, cfg, frames)
        assert np.array(out)[16, 16].tolist() == list(FG)

    def test_black_alpha_shows_background_color(self, tmp_path):
        frames = _setup_frames(tmp_path, alpha_value=0)
        cfg = _make_config(background='color', background_source='red')
        out = _apply_render_bg_mask(Image.new('RGB', (32, 32), FG), 0, cfg, frames)
        assert np.array(out)[16, 16].tolist() == [255, 0, 0]

    def test_init_video_background(self, tmp_path):
        frames = _setup_frames(tmp_path, alpha_value=0)
        cfg = _make_config(background='init_video')
        out = _apply_render_bg_mask(Image.new('RGB', (32, 32), FG), 0, cfg, frames)
        # background = raw video frame (solid blue); jpg compression tolerance
        px = np.array(out)[16, 16]
        assert px[2] > 200 and px[1] < 60

    def test_image_background(self, tmp_path):
        frames = _setup_frames(tmp_path, alpha_value=0)
        bg_path = str(tmp_path / 'bg.png')
        Image.new('RGB', (8, 8), (255, 255, 0)).save(bg_path)
        cfg = _make_config(background='image', background_source=bg_path)
        out = _apply_render_bg_mask(Image.new('RGB', (32, 32), FG), 0, cfg, frames)
        assert np.array(out)[16, 16].tolist() == [255, 255, 0]

    def test_invert_mask(self, tmp_path):
        frames = _setup_frames(tmp_path, alpha_value=255)
        cfg = _make_config(background='color', background_source='red',
                           invert_mask=True)
        out = _apply_render_bg_mask(Image.new('RGB', (32, 32), FG), 0, cfg, frames)
        # inverted white alpha → 0 → background shows
        assert np.array(out)[16, 16].tolist() == [255, 0, 0]

    def test_mask_clipping(self, tmp_path):
        # alpha 100: clip_low=150 → below threshold → 0 → background
        frames = _setup_frames(tmp_path, alpha_value=100)
        cfg = _make_config(background='color', background_source='red',
                           mask_clip_low=150)
        out = _apply_render_bg_mask(Image.new('RGB', (32, 32), FG), 0, cfg, frames)
        assert np.array(out)[16, 16].tolist() == [255, 0, 0]

        # alpha 200: clip_high=150 → above threshold → 255 → foreground
        frames2 = _setup_frames(tmp_path / 'b', alpha_value=200)
        cfg2 = _make_config(background='color', background_source='red',
                            mask_clip_high=150)
        out2 = _apply_render_bg_mask(Image.new('RGB', (32, 32), FG), 0, cfg2, frames2)
        assert np.array(out2)[16, 16].tolist() == list(FG)

    def test_missing_alpha_returns_unchanged(self, tmp_path):
        frames = _setup_frames(tmp_path, alpha_value=None)  # no alpha file
        cfg = _make_config(background='color', background_source='red')
        img = Image.new('RGB', (32, 32), FG)
        out = _apply_render_bg_mask(img, 0, cfg, frames)
        assert out is img

    def test_soft_alpha_blends(self, tmp_path):
        frames = _setup_frames(tmp_path, alpha_value=128)
        cfg = _make_config(background='color', background_source='black')
        out = _apply_render_bg_mask(Image.new('RGB', (32, 32), (255, 255, 255)),
                                    0, cfg, frames)
        px = np.array(out)[16, 16]
        assert 90 < px[0] < 170  # ~50% blend (jpg tolerance)


class TestConfigAndSettings:
    def test_defaults_match_notebook(self):
        c = BackgroundMaskConfig()
        assert c.use_background_mask is False
        assert c.invert_mask is False
        assert c.apply_mask_after_warp is True
        assert c.background == 'init_video'
        assert c.background_source == 'red'
        assert c.mask_clip_low == 0
        assert c.mask_clip_high == 255

    def test_runconfig_has_mask(self):
        cfg = RunConfig()
        assert isinstance(cfg.mask, BackgroundMaskConfig)

    def test_settings_mapping(self, tmp_path):
        from vibewarp.settings import load_warpfusion_settings
        raw = {
            'use_background_mask': True,
            'invert_mask': True,
            'apply_mask_after_warp': False,
            'background': 'color',
            'background_source': '#00ff00',
            'mask_clip_low': 10,
            'mask_clip_high': 250,
        }
        p = tmp_path / 'settings.txt'
        p.write_text(json.dumps(raw))
        cfg = load_warpfusion_settings(str(p))
        m = cfg['mask']
        assert m['use_background_mask'] is True
        assert m['invert_mask'] is True
        assert m['apply_mask_after_warp'] is False
        assert m['background'] == 'color'
        assert m['background_source'] == '#00ff00'
        assert m['mask_clip_low'] == 10
        assert m['mask_clip_high'] == 250

    def test_settings_defaults(self, tmp_path):
        from vibewarp.settings import load_warpfusion_settings
        p = tmp_path / 'settings.txt'
        p.write_text(json.dumps({}))
        cfg = load_warpfusion_settings(str(p))
        assert cfg['mask']['use_background_mask'] is False
        assert cfg['mask']['apply_mask_after_warp'] is True
