"""Tests for per-frame IP-Adapter source resolution, re-hooking with the
content-hash embeds cache, and the WarpFusion settings split (ipadapter_*
entries inside controlnet_multimodel)."""

import json
import os
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from vibewarp.config import RunConfig
from vibewarp.core.diffusion import (
    FrameState,
    RenderContext,
    _resolve_ipadapter_source,
    _update_ipadapter_for_frame,
)


class TestResolveIpadapterSource:
    def _ctx(self, tmp_path):
        frames = tmp_path / 'frames'
        frames.mkdir(exist_ok=True)
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
            str(frames / '000001.jpg'))
        return RenderContext(config=RunConfig(), video_frames_folder=str(frames))

    def test_init_resolves_current_frame(self, tmp_path):
        ctx = self._ctx(tmp_path)
        state = FrameState(frame_num=0)
        p = _resolve_ipadapter_source('init', 0, ctx, state)
        assert p and p.endswith('000001.jpg')
        assert _resolve_ipadapter_source('raw_frame', 0, ctx, state) == p

    def test_stylized_uses_init_image(self, tmp_path):
        ctx = self._ctx(tmp_path)
        styl = tmp_path / 'warped.png'
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(str(styl))
        state = FrameState(frame_num=3)
        state.init_image = str(styl)
        state.prev_frame = Image.new('RGB', (8, 8))
        assert _resolve_ipadapter_source('stylized', 3, ctx, state) == str(styl)

    def test_stylized_first_frame_is_not_sent(self, tmp_path):
        ctx = self._ctx(tmp_path)
        state = FrameState(frame_num=0)
        state.init_image = None
        assert _resolve_ipadapter_source('stylized', 0, ctx, state) is None

    def test_temporal_sources_are_off_at_first_configured_range_frame(self, tmp_path):
        ctx = self._ctx(tmp_path)
        ctx._sequence_start_frame = 12
        ctx.batch_folder = str(tmp_path)
        ctx.config.batch_name = 'run'
        state = FrameState(
            frame_num=12,
            init_image=str(tmp_path / 'warped.png'),
            prev_frame=Image.new('RGB', (8, 8)),
        )
        Image.new('RGB', (8, 8)).save(state.init_image)
        Image.new('RGB', (8, 8)).save(tmp_path / 'run(0)_000011.png')

        assert _resolve_ipadapter_source(
            {'source': 'previous', 'image_path': ''}, 12, ctx, state) is None
        assert _resolve_ipadapter_source(
            {'source': 'warped', 'image_path': ''}, 12, ctx, state) is None

    def test_previous_and_fixed_sources_after_range_start(self, tmp_path):
        ctx = self._ctx(tmp_path)
        ctx._sequence_start_frame = 4
        ctx.batch_folder = str(tmp_path)
        ctx.config.batch_name = 'run'
        previous = tmp_path / 'run(0)_000004.png'
        fixed = tmp_path / 'style.png'
        Image.new('RGB', (8, 8)).save(previous)
        Image.new('RGB', (8, 8)).save(fixed)
        state = FrameState(frame_num=5, prev_frame=Image.new('RGB', (8, 8)))

        assert _resolve_ipadapter_source(
            {'source': 'previous', 'image_path': ''}, 5, ctx, state
        ) == str(previous)
        assert _resolve_ipadapter_source(
            {'source': 'upload', 'image_path': str(fixed)}, 4, ctx, state
        ) == str(fixed)

    def test_schedule_dict(self, tmp_path):
        ctx = self._ctx(tmp_path)
        img = tmp_path / 'ref.png'
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(str(img))
        state = FrameState(frame_num=7)
        assert _resolve_ipadapter_source({0: str(img)}, 7, ctx, state) == str(img)
        assert _resolve_ipadapter_source({'0': str(img)}, 7, ctx, state) == str(img)

    def test_missing_returns_none(self, tmp_path):
        ctx = self._ctx(tmp_path)
        state = FrameState(frame_num=99)
        assert _resolve_ipadapter_source('init', 99, ctx, state) is None
        assert _resolve_ipadapter_source('/nope/missing.png', 99, ctx, state) is None


class TestPerFrameRehook:
    def _setup(self, tmp_path, sources, cache_embeddings=True):
        frames = tmp_path / 'frames'
        frames.mkdir(exist_ok=True)
        for i in range(1, 4):
            Image.fromarray(
                np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
            ).save(str(frames / f'{i:06d}.jpg'))

        cfg = RunConfig()
        cfg.ipadapter.cache_embeddings = cache_embeddings
        cfg.ipadapter.cache_size = 10

        loaded = {}
        for i, src in enumerate(sources):
            loaded[f'ipa{i}'] = {
                'source_image': src[0] if isinstance(src, list) else src,
                'source_images': src if isinstance(src, list) else [],
                'weights': {'image_proj': {}, 'ip_adapter': {}},
                'path': f'ip-adapter_v2_{i}.bin' if i else 'ip-adapter.bin',
                'weight': 0.5, 'start': 0.0, 'end': 1.0,
                'weight_type': 'linear', 'combine_embeds': 'concat',
                'embeds_scaling': 'V only',
            }

        class FakeSD:
            pass

        ctx = RenderContext(
            config=cfg,
            loaded_ipadapters=loaded,
            clip_vision_model=MagicMock(),
            video_frames_folder=str(frames),
        )
        ctx.sd_model = FakeSD()
        return ctx

    def _run(self, ctx, frame_num, monkeypatch):
        import vibewarp.core.ipadapter as ipa_mod
        import vibewarp.vendor.controlmodel_ipadapter as vendor_mod

        hook = MagicMock()
        clear = MagicMock()
        encode = MagicMock(
            side_effect=lambda m, t: {'image_embeds': torch.randn(1, 4)})
        load_img = MagicMock(return_value=torch.zeros(1, 8, 8, 3))
        monkeypatch.setattr(ipa_mod, 'hook_ipadapter', hook)
        monkeypatch.setattr(ipa_mod, 'encode_clip_vision', encode)
        monkeypatch.setattr(ipa_mod, 'load_image_for_clip', load_img)
        monkeypatch.setattr(vendor_mod, 'clear_all_ip_adapter', clear)

        state = FrameState(frame_num=frame_num)
        _update_ipadapter_for_frame(ctx, state)
        return hook, clear, encode

    def test_multi_adapter_clear_once_hook_all(self, tmp_path, monkeypatch):
        """The old implementation unhooked ALL adapters inside its per-adapter
        loop — hooking adapter B wiped adapter A. Now: clear once, hook all."""
        ctx = self._setup(tmp_path, ['init', 'init'])
        hook, clear, encode = self._run(ctx, 0, monkeypatch)
        assert clear.call_count == 1
        assert hook.call_count == 2
        # Same image + encoder shares cached CLIP output, while each adapter
        # instance remains a separate attention hook with its own settings.
        assert encode.call_count == 1

    def test_static_sources_rehook_after_preprocessing(self, tmp_path, monkeypatch):
        img = tmp_path / 'static.png'
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(str(img))
        ctx = self._setup(tmp_path, [str(img)])
        hook, clear, _ = self._run(ctx, 0, monkeypatch)
        assert clear.call_count == 1
        assert hook.call_count == 1

    def test_embeds_cache_hits_on_unchanged_source(self, tmp_path, monkeypatch):
        ctx = self._setup(tmp_path, ['init'])
        _, _, encode1 = self._run(ctx, 0, monkeypatch)
        assert encode1.call_count == 1
        # same frame again, file unchanged -> cache hit, no new encode
        _, _, encode2 = self._run(ctx, 0, monkeypatch)
        assert encode2.call_count == 0

    def test_cache_disabled_reencodes(self, tmp_path, monkeypatch):
        ctx = self._setup(tmp_path, ['init'], cache_embeddings=False)
        _, _, encode1 = self._run(ctx, 0, monkeypatch)
        _, _, encode2 = self._run(ctx, 0, monkeypatch)
        assert encode1.call_count == 1
        assert encode2.call_count == 1

    def test_is_v2_flip_uc_combine_embeds_passed(self, tmp_path, monkeypatch):
        ctx = self._setup(tmp_path, ['init', 'init'])
        ctx.config.ipadapter.flip_uc = True
        hook, _, _ = self._run(ctx, 0, monkeypatch)
        kwargs0 = hook.call_args_list[0].kwargs
        kwargs1 = hook.call_args_list[1].kwargs
        assert kwargs0['is_v2'] is False   # 'ip-adapter.bin'
        assert kwargs1['is_v2'] is True    # 'ip-adapter_v2_1.bin'
        assert kwargs0['flip_uc'] is True
        assert kwargs0['combine_embeds'] == 'concat'

    def test_one_instance_combines_multiple_images_once(
            self, tmp_path, monkeypatch, capsys):
        first = tmp_path / 'first.png'
        second = tmp_path / 'second.png'
        Image.new('RGB', (8, 8), 'red').save(first)
        Image.new('RGB', (8, 8), 'blue').save(second)
        ctx = self._setup(tmp_path, [[str(first), str(second)]])
        ctx.batch_folder = str(tmp_path / 'run')
        ctx.loaded_ipadapters['ipa0']['combine_embeds'] = 'average'

        hook, clear, encode = self._run(ctx, 0, monkeypatch)

        assert clear.call_count == 1
        assert encode.call_count == 2
        assert hook.call_count == 1
        kwargs = hook.call_args.kwargs
        assert kwargs['combine_embeds'] == 'average'
        assert kwargs['clip_vision_output']['image_embeds'].shape == (1, 4)
        message = capsys.readouterr().out
        assert "Applied IP-Adapter ipa0: frame=0" in message
        assert "sources=2, combine=average, weight=0.5" in message
        debug = tmp_path / 'run' / 'debug'
        assert (debug / 'ipa_ipa0_source_1_000000.png').is_file()
        assert (debug / 'ipa_ipa0_source_2_000000.png').is_file()


class TestSettingsIpadapterSplit:
    def _load(self, tmp_path, raw):
        from vibewarp.settings import load_warpfusion_settings
        p = tmp_path / 's.txt'
        p.write_text(json.dumps(raw))
        return load_warpfusion_settings(str(p))

    def test_ipadapter_split_from_controlnet_multimodel(self, tmp_path):
        cfg = self._load(tmp_path, {
            'controlnet_multimodel': {
                'control_sd15_depth': {'weight': 1.0, 'start': 0, 'end': 1,
                                       'preprocess': 'global', 'source': 'global'},
                'ipadapter_sd15_plus': {'weight': 0.6, 'start': 0.1, 'end': 0.9,
                                        'source': 'stylized',
                                        'ip_weight_type': 'style transfer',
                                        'ip_combine_embeds': 'global',
                                        'ip_embeds_scaling': 'K+V'},
            },
            'ip_combine_embeds': 'add',
            'ip_flip_uc': True,
            'cache_ipadapter': False,
            'ipadapter_embeds_cache_size': 5,
        })
        # not parsed as a ControlNet
        assert 'ipadapter_sd15_plus' not in cfg['controlnet']['models']
        assert 'control_sd15_depth' in cfg['controlnet']['models']
        ipa = cfg['ipadapter']
        assert ipa['enabled'] is True
        m = ipa['models']['ipadapter_sd15_plus']
        assert m['model_key'] == 'ipadapter_sd15_plus'
        assert m['weight'] == 0.6
        assert m['source_image'] == {'source': 'warped', 'image_path': ''}
        assert m['weight_type'] == 'style transfer'
        assert m['combine_embeds'] == 'add'       # 'global' -> global fallback
        assert m['embeds_scaling'] == 'K+V'
        assert m['path'].endswith('ip-adapter-plus_sd15.bin')
        assert ipa['flip_uc'] is True
        assert ipa['cache_embeddings'] is False
        assert ipa['cache_size'] == 5

    def test_no_ipadapters_disabled(self, tmp_path):
        cfg = self._load(tmp_path, {})
        assert cfg['ipadapter']['enabled'] is False
        assert cfg['ipadapter']['cache_size'] == 10
        assert cfg['ipadapter']['flip_uc'] is False

    def test_discovers_warpfusion_clip_vision_directory(self, tmp_path):
        from vibewarp.settings import load_warpfusion_settings

        models = tmp_path / 'models'
        clip_dir = models / 'controlnet' / 'clip_vision'
        clip_dir.mkdir(parents=True)
        (clip_dir / 'clip_vision_vit_h.safetensors').write_bytes(b'placeholder')
        settings = tmp_path / 's.txt'
        settings.write_text(json.dumps({
            'controlnet_multimodel': {
                'ipadapter_sd15_plus': {
                    'weight': 1.0, 'source': 'raw_frame',
                },
            },
        }))

        cfg = load_warpfusion_settings(
            str(settings), models_root=str(models))
        assert cfg['ipadapter']['clip_vision_model_path'] == str(clip_dir)
