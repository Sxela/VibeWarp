"""Tests for keyframe captioning (notebook cell 30): keyframe selection,
offset remapping, caption generation (stubbed BLIP), piecewise lookup, and
{caption} prompt substitution."""

import json
import os
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

import vibewarp.core.captioning as cap_mod
from vibewarp.core.captioning import (
    apply_caption,
    generate_captions,
    get_caption,
    remap_keyframes,
    select_keyframes,
)


class TestSelectKeyframes:
    def test_every_nth(self):
        # notebook: list(range(1, len(frames), nth)) — 1-based
        assert select_keyframes(25, 'Every n-th frame', nth_frame=10) == [1, 11, 21]

    def test_user_defined(self):
        assert select_keyframes(
            100, 'User-defined keyframe list',
            user_keyframes=[3, 4, 5]) == [3, 4, 5]

    def test_content_aware_without_diffs_returns_none(self):
        assert select_keyframes(
            100, 'Content-aware scheduling keyframes', diffs=None) is None

    def test_content_aware_with_diffs(self):
        diffs = [0.1, 0.5, 0.2, 0.4]
        # [1] + 1-based indices where diff >= thresh
        assert select_keyframes(
            100, 'Content-aware scheduling keyframes',
            diffs=diffs, diff_thresh=0.33) == [1, 2, 4]


class TestRemapKeyframes:
    def test_none_mode(self):
        assert remap_keyframes([1, 11, 21], 'None') == [1, 11, 21]

    def test_fixed_offset_clamped(self):
        # +5 within gaps; last keyframe never moves
        assert remap_keyframes([1, 11, 21], 'Fixed', 5) == [6, 16, 21]
        # offset larger than gap → clamped to next keyframe
        assert remap_keyframes([1, 11, 21], 'Fixed', 100) == [11, 21, 21]

    def test_between_keyframes(self):
        assert remap_keyframes([1, 11, 21], 'Between Keyframes') == [6, 16, 21]

    def test_empty(self):
        assert remap_keyframes([], 'Fixed', 5) == []


def _fake_blip(monkeypatch, captions):
    """Stub _load_blip: returns a processor/model pair producing canned text."""
    calls = {'count': 0}

    processor = MagicMock()
    processor.return_value = MagicMock(to=lambda d: {'pixel_values': torch.zeros(1)})

    def decode(ids, skip_special_tokens=True):
        text = captions[calls['count'] % len(captions)]
        calls['count'] += 1
        return text

    processor.decode = decode

    model = MagicMock()
    model.generate = MagicMock(return_value=[[0]])
    param = torch.nn.Parameter(torch.zeros(1))
    model.parameters = lambda: iter([param])

    monkeypatch.setattr(cap_mod, '_load_blip', lambda vqa, device='cuda': (processor, model))
    return calls


def _make_frames(tmp_path, n):
    frames = tmp_path / 'video_frames'
    frames.mkdir()
    for i in range(1, n + 1):
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
            str(frames / f'{i:06d}.jpg'))
    return str(frames)


class TestGenerateCaptions:
    def test_writes_caption_files(self, tmp_path, monkeypatch):
        frames = _make_frames(tmp_path, 5)
        _fake_blip(monkeypatch, ['a cat', 'a dog'])
        folder = generate_captions(frames, [1, 3])
        assert folder == frames + 'Captions'
        files = sorted(os.listdir(folder))
        assert files == ['000001.txt', '000003.txt']
        assert open(os.path.join(folder, '000001.txt')).read() == 'a cat'
        assert open(os.path.join(folder, '000003.txt')).read() == 'a dog'

    def test_clears_stale_captions(self, tmp_path, monkeypatch):
        frames = _make_frames(tmp_path, 3)
        _fake_blip(monkeypatch, ['x'])
        folder = frames + 'Captions'
        os.makedirs(folder)
        with open(os.path.join(folder, '000099.txt'), 'w') as f:
            f.write('stale')
        generate_captions(frames, [1])
        assert sorted(os.listdir(folder)) == ['000001.txt']

    def test_out_of_range_keyframe_skipped(self, tmp_path, monkeypatch):
        frames = _make_frames(tmp_path, 2)
        _fake_blip(monkeypatch, ['x'])
        folder = generate_captions(frames, [1, 50])
        assert sorted(os.listdir(folder)) == ['000001.txt']


class TestGetCaption:
    def _folder(self, tmp_path, captions):
        folder = tmp_path / 'caps'
        folder.mkdir()
        for frame1, text in captions.items():
            (folder / f'{frame1:06d}.txt').write_text(text)
        return str(folder)

    def test_piecewise_lookup(self, tmp_path):
        # captions at 1-based keyframes 1, 11, 21
        folder = self._folder(tmp_path, {1: 'one', 11: 'eleven', 21: 'twentyone'})
        # frame_num is 0-based → frame_num+1 compared against keyframes
        assert get_caption(0, folder) == 'one'      # frame 1
        assert get_caption(5, folder) == 'one'      # frame 6
        assert get_caption(10, folder) == 'eleven'  # frame 11
        assert get_caption(19, folder) == 'eleven'  # frame 20
        assert get_caption(20, folder) == 'twentyone'
        assert get_caption(99, folder) == 'twentyone'

    def test_before_first_uses_first(self, tmp_path):
        folder = self._folder(tmp_path, {5: 'five'})
        assert get_caption(0, folder) == 'five'

    def test_missing_folder_returns_none(self):
        assert get_caption(0, '/nope/missing') is None
        assert get_caption(0, '') is None

    def test_empty_folder_returns_none(self, tmp_path):
        folder = tmp_path / 'empty'
        folder.mkdir()
        assert get_caption(0, str(folder)) is None


class TestApplyCaption:
    def test_substitutes(self):
        assert apply_caption('portrait of {caption}, oil painting', 'a cat') == \
            'portrait of a cat, oil painting'

    def test_noop_without_placeholder(self):
        assert apply_caption('plain prompt', 'a cat') == 'plain prompt'

    def test_noop_without_caption(self):
        assert apply_caption('has {caption}', None) == 'has {caption}'


class TestConfigAndSettings:
    def test_defaults(self):
        from vibewarp.config import CaptionConfig, RunConfig
        c = CaptionConfig()
        assert c.make_captions is False
        assert c.keyframe_source == 'Every n-th frame'
        assert c.nth_frame == 10
        assert c.offset_mode == 'Fixed'
        assert c.fixed_offset == 0
        assert c.blip_question == ''
        assert isinstance(RunConfig().captions, CaptionConfig)

    def test_settings_mapping(self, tmp_path):
        from vibewarp.settings import load_warpfusion_settings
        p = tmp_path / 's.txt'
        p.write_text(json.dumps({
            'make_captions': True,
            'keyframe_source': 'User-defined keyframe list',
            'user_defined_keyframes': [3, 4, 5],
            'nth_frame': 7,
            'offset_mode': 'Between Keyframes',
            'fixed_offset': 2,
            'blip_question': 'what color?',
        }))
        cfg = load_warpfusion_settings(str(p))
        c = cfg['captions']
        assert c['make_captions'] is True
        assert c['keyframe_source'] == 'User-defined keyframe list'
        assert c['user_defined_keyframes'] == [3, 4, 5]
        assert c['nth_frame'] == 7
        assert c['offset_mode'] == 'Between Keyframes'
        assert c['blip_question'] == 'what color?'
