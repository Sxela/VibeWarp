"""Tests for render-time background masking (notebook cell 42 + apply_mask).

Covers BackgroundMaskConfig, WarpFusion settings mapping, and
_apply_render_bg_mask compositing (color/image/init_video backgrounds,
invert, clipping, missing-alpha fallback).
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vibewarp.config import BackgroundMaskConfig, RunConfig
from vibewarp.core.diffusion import _apply_render_bg_mask


@pytest.mark.parametrize('stage',['before diffusion','after diffusion'])
@pytest.mark.parametrize('version',['control_multi_v15','disco_512','qwen_image_edit_2511'])
@pytest.mark.parametrize('frame',[0,3])
def test_shared_rvm_mask_applies_once_at_selected_render_boundary(
        tmp_path,monkeypatch,capsys,stage,version,frame):
    from vibewarp.core import diffusion
    cfg=_make_config(background='color',background_source='black')
    cfg.model_version=version;cfg.mask.render_mask_stage=stage
    cfg.disco.pad_to_multiple_64=False
    cfg.video.width=cfg.video.height=16
    frames=tmp_path/'video_frames';frames.mkdir()
    raw_path=frames/f'{frame+1:06d}.jpg'
    Image.new('RGB',(16,16),'white').save(raw_path)
    alpha=tmp_path/'alpha';alpha.mkdir()
    alpha_path=alpha/f'{frame:06d}.npy';np.save(alpha_path,np.full((16,16),.5,dtype=np.float16))
    original_raw=raw_path.read_bytes();original_alpha=alpha_path.read_bytes()
    ctx=diffusion.RenderContext(config=cfg,batch_folder=str(tmp_path),video_frames_folder=str(frames),
                               raw_video_cache=SimpleNamespace(alpha=alpha))
    # The pipeline materializes the run masks from the RVM cache before rendering.
    from vibewarp.video.masks import materialize
    materialize(cfg,frames,tmp_path,ctx.raw_video_cache)
    state=diffusion.FrameState();seen=[]
    def render(context,state):
        with Image.open(state.init_image) as image:
            seen.append(image.getpixel((8,8)))
            return image.convert('RGB')
    monkeypatch.setattr(diffusion,'render_frame',render)
    image,path=diffusion._render_single_frame(ctx,state,frame,frame,str(tmp_path),'png')
    assert seen==[(128,128,128) if stage=='before diffusion' else (255,255,255)]
    assert image.getpixel((8,8))==(128,128,128)  # two applications would yield 64
    assert state.prev_frame.getpixel((8,8))==(128,128,128)
    with Image.open(path) as saved:assert saved.getpixel((8,8))==(128,128,128)
    assert raw_path.read_bytes()==original_raw and alpha_path.read_bytes()==original_alpha
    assert 'no mask for frame' not in capsys.readouterr().out


@pytest.mark.parametrize('stage',['before diffusion','after diffusion'])
@pytest.mark.parametrize('stylized',[False,True])
def test_animatediff_raw_and_overlap_inits_obey_diffusion_stage(tmp_path,stage,stylized):
    from vibewarp.core.diffusion import RenderContext,_adiff_init_path
    cfg=_make_config(background='color',background_source='black')
    cfg.mask.render_mask_stage=stage;cfg.video.width=cfg.video.height=16
    cfg.animatediff.overlap_stylized=stylized
    frames=tmp_path/'frames';frames.mkdir();Image.new('RGB',(16,16),'white').save(frames/'000001.jpg')
    if stylized:Image.new('RGB',(16,16),'white').save(tmp_path/f'{cfg.batch_name}(0)_000000.png')
    alpha=tmp_path/'alpha';alpha.mkdir();np.save(alpha/'000000.npy',np.full((16,16),.5,np.float16))
    ctx=RenderContext(config=cfg,batch_folder=str(tmp_path),video_frames_folder=str(frames),raw_video_cache=SimpleNamespace(alpha=alpha))
    from vibewarp.video.masks import materialize
    materialize(cfg,frames,tmp_path,ctx.raw_video_cache)
    path,is_stylized=_adiff_init_path(ctx,0,0,4)
    assert is_stylized is stylized
    with Image.open(path) as image:
        assert image.getpixel((8,8))==((128,128,128) if stage=='before diffusion' else (255,255,255))


def test_tint_and_render_mask_schema_and_settings_stay_independent():
    from dataclasses import asdict
    from vibewarp.config_io import config_schema,config_from_dict,validate_config
    from vibewarp.video.raw_processing import enabled,needs_alpha,transforms_raw
    props=config_schema()['properties']['mask']['properties']
    assert props['rvm_enabled']['label']=='Apply masked tint'
    assert props['use_background_mask']['tier']=='render'
    assert props['render_mask_stage']['choices']==['before diffusion','after diffusion']
    assert props['apply_mask_after_warp']['tier']=='hidden'
    for tint,mask in [(False,False),(True,False),(False,True),(True,True)]:
        cfg=RunConfig();cfg.mask.rvm_enabled=tint;cfg.mask.use_background_mask=mask
        assert needs_alpha(cfg)==enabled(cfg)==(tint or mask)
        assert transforms_raw(cfg)==tint
        cfg.mask.render_mask_stage='after diffusion';cfg.mask.apply_mask_after_warp=False
        restored=config_from_dict(asdict(cfg))
        assert restored.mask.render_mask_stage=='after diffusion'
        assert restored.mask.apply_mask_after_warp is False
    cfg.mask.render_mask_stage='before warp'
    assert any(error['path']=='mask.render_mask_stage' for error in validate_config(cfg,require_inputs=False))


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
        assert c.render_mask_stage == 'before diffusion'
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
            'render_mask_stage': 'after diffusion',
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
        assert m['render_mask_stage'] == 'after diffusion'
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
