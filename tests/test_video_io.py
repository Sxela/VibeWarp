"""Tests for vibewarp.video.input and vibewarp.utils.compose."""

import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from vibewarp.video.input import FrameDataset, extract_frames, fit_dimensions
from vibewarp.utils.compose import hstack, vstack


class TestFitDimensions:
    def test_landscape_preserves_source_aspect_ratio(self):
        assert fit_dimensions(1920, 1080, 512) == (512, 288)

    def test_portrait_preserves_source_aspect_ratio(self):
        assert fit_dimensions(1080, 1920, 768) == (432, 768)

    def test_does_not_upscale_smaller_video(self):
        assert fit_dimensions(320, 240, 512) == (320, 240)


class TestExtractFrames:
    @staticmethod
    def _capture_ffmpeg(monkeypatch, video_input, extracted_name):
        import types

        captured = {}

        def fake_run(cmd, **kwargs):
            captured['cmd'] = cmd
            return types.SimpleNamespace(stderr=b'')

        monkeypatch.setattr(video_input, '_find_ffmpeg', lambda: 'ffmpeg')
        monkeypatch.setattr(video_input.subprocess, 'run', fake_run)
        monkeypatch.setattr(video_input, 'glob', lambda pattern: [extracted_name])
        return captured

    def test_nonzero_range_preserves_absolute_frame_numbering(
            self, monkeypatch, tmp_path):
        import vibewarp.video.input as video_input

        captured = self._capture_ffmpeg(
            monkeypatch, video_input, '000061.jpg')
        extract_frames(
            'input.mp4', str(tmp_path), nth_frame=1,
            start_frame=60, end_frame=70)

        cmd = captured['cmd']
        assert cmd[cmd.index('-start_number') + 1] == '61'
        assert 'between(n\\,60\\,70)' in cmd[cmd.index('-vf') + 1]

    def test_zero_range_keeps_normal_one_based_numbering(
            self, monkeypatch, tmp_path):
        import vibewarp.video.input as video_input

        captured = self._capture_ffmpeg(
            monkeypatch, video_input, '000001.jpg')
        extract_frames('input.mp4', str(tmp_path), start_frame=0, end_frame=10)

        cmd = captured['cmd']
        assert cmd[cmd.index('-start_number') + 1] == '1'


class TestFrameDataset:
    def test_from_directory(self, tmp_path):
        # Create dummy frames
        for i in range(5):
            img = Image.new('RGB', (32, 32), color=(i * 50, 0, 0))
            img.save(os.path.join(str(tmp_path), f'frame_{i:04d}.png'))

        ds = FrameDataset(str(tmp_path))
        assert len(ds) == 5
        assert os.path.exists(ds[0])
        assert os.path.exists(ds[4])

    def test_clamps_index(self, tmp_path):
        for i in range(3):
            Image.new('RGB', (8, 8)).save(
                os.path.join(str(tmp_path), f'{i:04d}.png'))

        ds = FrameDataset(str(tmp_path))
        # Index beyond length should clamp to last
        assert ds[100] == ds[2]

    def test_empty_dir_raises(self, tmp_path):
        empty = os.path.join(str(tmp_path), 'empty')
        os.makedirs(empty)
        with pytest.raises(FileNotFoundError):
            FrameDataset(empty)

    def test_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            FrameDataset('/nonexistent/path/that/doesnt/exist')

    def test_glob_pattern(self, tmp_path):
        for i in range(3):
            Image.new('RGB', (8, 8)).save(
                os.path.join(str(tmp_path), f'img_{i}.jpg'))
        # Also create a PNG that shouldn't match
        Image.new('RGB', (8, 8)).save(
            os.path.join(str(tmp_path), 'other.png'))

        ds = FrameDataset(os.path.join(str(tmp_path), '*.jpg'))
        assert len(ds) == 3


class TestHStack:
    def test_basic(self):
        imgs = [Image.new('RGB', (32, 32), c) for c in [(255, 0, 0), (0, 255, 0)]]
        result = hstack(imgs)
        assert result.size == (64, 32)

    def test_from_files(self, tmp_path):
        paths = []
        for i in range(2):
            p = os.path.join(str(tmp_path), f'{i}.png')
            Image.new('RGB', (16, 16)).save(p)
            paths.append(p)
        result = hstack(paths)
        assert result.size == (32, 16)


class TestVStack:
    def test_basic(self):
        imgs = [Image.new('RGB', (32, 16)), Image.new('RGB', (32, 24))]
        result = vstack(imgs)
        assert result.size == (32, 40)
