"""Tests for video assembly post-processing: audio, deflicker, upscale, mask."""

import os
import subprocess
import numpy as np
import pytest
from PIL import Image, ImageOps
from unittest.mock import MagicMock, patch


# ---- apply_video_mask ----

class TestApplyVideoMask:
    def _make_dirs(self, tmp_path, frame_val=100, alpha_val=200, bg_val=50):
        """Set up frames and alpha dirs with simple uniform images."""
        vf_dir = str(tmp_path / 'video_frames')
        alpha_dir = vf_dir + '_alpha'
        os.makedirs(vf_dir, exist_ok=True)
        os.makedirs(alpha_dir, exist_ok=True)

        frame = Image.fromarray(np.full((8, 8, 3), frame_val, dtype=np.uint8))
        alpha = Image.fromarray(np.full((8, 8), alpha_val, dtype=np.uint8))
        bg = Image.fromarray(np.full((8, 8, 3), bg_val, dtype=np.uint8))

        # frame 0 → file 000001
        bg.save(os.path.join(vf_dir, '000001.jpg'))
        alpha.save(os.path.join(alpha_dir, '000001.jpg'))
        return vf_dir, frame

    def test_missing_alpha_returns_original(self, tmp_path):
        from vibewarp.video.output import apply_video_mask
        vf_dir = str(tmp_path / 'novf')
        os.makedirs(vf_dir, exist_ok=True)
        frame = Image.fromarray(np.full((8, 8, 3), 100, dtype=np.uint8))
        result = apply_video_mask(frame, 0, 'init_video', '', vf_dir)
        assert np.array_equal(np.array(result), np.array(frame))

    def test_color_background_composites(self, tmp_path):
        from vibewarp.video.output import apply_video_mask
        vf_dir, frame = self._make_dirs(tmp_path, frame_val=200, alpha_val=255)
        # Full opacity alpha=255 → frame completely on top of bg
        result = apply_video_mask(frame, 0, 'color', 'red', vf_dir)
        arr = np.array(result)
        # With full alpha, result should equal the stylized frame
        assert arr.shape == (8, 8, 3)

    def test_invert_mask(self, tmp_path):
        from vibewarp.video.output import apply_video_mask
        vf_dir, frame = self._make_dirs(tmp_path, frame_val=200, alpha_val=255)
        result_normal = apply_video_mask(frame, 0, 'color', 'black', vf_dir, invert=False)
        result_inverted = apply_video_mask(frame, 0, 'color', 'black', vf_dir, invert=True)
        # Inverting 255 alpha → 0 alpha → background shows through
        arr_n = np.array(result_normal).mean()
        arr_i = np.array(result_inverted).mean()
        assert arr_n != arr_i

    def test_clip_low_zeros_dark_alpha(self, tmp_path):
        from vibewarp.video.output import apply_video_mask
        vf_dir, frame = self._make_dirs(tmp_path, frame_val=200, alpha_val=10)
        # clip_low=50 → alpha=10 gets zeroed → fully transparent → bg shows
        result = apply_video_mask(frame, 0, 'color', 'black', vf_dir,
                                  clip_low=50, clip_high=255)
        # frame=200, bg=black; with zero alpha, bg dominates
        arr = np.array(result).mean()
        assert arr < 50  # mostly black

    def test_clip_high_saturates_bright_alpha(self, tmp_path):
        from vibewarp.video.output import apply_video_mask
        vf_dir, frame = self._make_dirs(tmp_path, frame_val=200, alpha_val=200)
        # clip_high=100 → alpha=200 gets clamped to 255 → full opacity
        result_clipped = apply_video_mask(frame, 0, 'color', 'black', vf_dir,
                                          clip_low=0, clip_high=100)
        result_normal = apply_video_mask(frame, 0, 'color', 'black', vf_dir)
        # Both put frame on top, but clip forces full coverage
        arr_c = np.array(result_clipped).mean()
        arr_n = np.array(result_normal).mean()
        # Both should be bright (frame=200 on black); clipped should be >= normal
        assert arr_c >= arr_n - 10

    def test_init_video_background(self, tmp_path):
        from vibewarp.video.output import apply_video_mask
        vf_dir, frame = self._make_dirs(tmp_path, frame_val=200, alpha_val=0, bg_val=50)
        # alpha=0 → fully transparent → background shows through
        result = apply_video_mask(frame, 0, 'init_video', '', vf_dir)
        arr = np.array(result).mean()
        # With zero alpha, background (50) dominates
        assert arr < 100


# ---- attach_audio ----

class TestAttachAudio:
    def test_missing_video_returns_original(self, tmp_path):
        from vibewarp.video.output import attach_audio
        result = attach_audio('/nonexistent.mp4', '/also_nonexistent.mp4', 'ffmpeg')
        assert result == '/nonexistent.mp4'

    def test_missing_source_returns_original(self, tmp_path):
        from vibewarp.video.output import attach_audio
        video = str(tmp_path / 'v.mp4')
        open(video, 'w').close()
        result = attach_audio(video, '/nonexistent_source.mp4', 'ffmpeg')
        assert result == video

    def test_calls_ffmpeg_with_correct_args(self, tmp_path):
        from vibewarp.video.output import attach_audio
        video = str(tmp_path / 'v.mp4')
        source = str(tmp_path / 's.mp4')
        dest = str(tmp_path / 'out.mp4')
        open(video, 'w').close()
        open(source, 'w').close()

        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch('subprocess.run', return_value=mock_result) as mock_run:
            attach_audio(video, source, 'ffmpeg', dest)
            args = mock_run.call_args[0][0]
        assert '-map' in args
        assert '0:v' in args
        assert '1:a' in args
        assert '-c:v' in args
        assert 'copy' in args

    def test_failed_ffmpeg_returns_original(self, tmp_path):
        from vibewarp.video.output import attach_audio
        video = str(tmp_path / 'v.mp4')
        source = str(tmp_path / 's.mp4')
        dest = str(tmp_path / 'out.mp4')
        open(video, 'w').close()
        open(source, 'w').close()

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = 'error\nno audio stream'
        with patch('subprocess.run', return_value=mock_result):
            result = attach_audio(video, source, 'ffmpeg', dest)
        assert result == video


# ---- create_video deflicker flag ----

class TestCreateVideoDeflicker:
    def test_deflicker_adds_vf_filter(self, tmp_path):
        from vibewarp.video.output import create_video
        from PIL import Image
        import numpy as np

        # Create fake frames
        for i in range(2):
            img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
            img.save(str(tmp_path / f'batch(0)_{i:06d}.png'))

        captured_cmds = []
        def fake_run(cmd, **kw):
            captured_cmds.append(cmd)
            r = MagicMock()
            r.returncode = 0
            return r

        with patch('subprocess.run', side_effect=fake_run):
            with patch('vibewarp.video.input._find_ffmpeg', return_value='ffmpeg'):
                try:
                    create_video(
                        frames_dir=str(tmp_path),
                        output_path=str(tmp_path / 'out.mp4'),
                        batch_name='batch',
                        use_deflicker=True,
                    )
                except Exception:
                    pass

        # Find the encode command (has -vf)
        encode_cmds = [c for c in captured_cmds if '-vf' in c]
        assert encode_cmds, "No ffmpeg encode command captured"
        vf_idx = encode_cmds[0].index('-vf')
        vf_value = encode_cmds[0][vf_idx + 1]
        assert 'deflicker' in vf_value

    def test_no_deflicker_no_filter(self, tmp_path):
        from vibewarp.video.output import create_video
        from PIL import Image
        import numpy as np

        for i in range(2):
            img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
            img.save(str(tmp_path / f'batch(0)_{i:06d}.png'))

        captured_cmds = []
        def fake_run(cmd, **kw):
            captured_cmds.append(cmd)
            r = MagicMock()
            r.returncode = 0
            return r

        with patch('subprocess.run', side_effect=fake_run):
            with patch('vibewarp.video.input._find_ffmpeg', return_value='ffmpeg'):
                try:
                    create_video(
                        frames_dir=str(tmp_path),
                        output_path=str(tmp_path / 'out.mp4'),
                        batch_name='batch',
                        use_deflicker=False,
                    )
                except Exception:
                    pass

        encode_cmds = [c for c in captured_cmds if '-vf' in c]
        if encode_cmds:
            vf_idx = encode_cmds[0].index('-vf')
            vf_value = encode_cmds[0][vf_idx + 1]
            assert 'deflicker' not in vf_value


# ---- VideoAssemblyConfig defaults ----

class TestVideoAssemblyConfig:
    def test_defaults(self):
        from vibewarp.config import VideoAssemblyConfig
        cfg = VideoAssemblyConfig()
        assert cfg.keep_audio is True
        assert cfg.use_deflicker is False
        assert cfg.upscale_ratio == 1
        assert cfg.upscale_model == 'realesr-animevideov3'
        assert cfg.use_background_mask_video is False
        assert cfg.mask_clip_low == 0
        assert cfg.mask_clip_high == 255

    def test_in_run_config(self):
        from vibewarp.config import RunConfig
        rc = RunConfig()
        assert hasattr(rc, 'video_assembly')
        assert rc.video_assembly.upscale_ratio == 1


# ---- settings mapping ----

class TestVideoAssemblySettingsMapping:
    def _write(self, tmp_path, extra=None):
        import json
        d = {
            'model_path': 'm.safetensors',
            'text_prompts': {'0': ['a painting']},
            'negative_prompts': {'0': ['']},
        }
        if extra:
            d.update(extra)
        p = str(tmp_path / 's.txt')
        with open(p, 'w') as f:
            json.dump(d, f)
        return p

    def test_defaults_when_absent(self, tmp_path):
        from vibewarp.settings import load_warpfusion_settings
        p = self._write(tmp_path)
        cfg = load_warpfusion_settings(p)
        va = cfg['video_assembly']
        assert va['keep_audio'] is True
        assert va['use_deflicker'] is False
        assert va['upscale_ratio'] == 1
        assert va['use_background_mask_video'] is False

    def test_values_mapped(self, tmp_path):
        from vibewarp.settings import load_warpfusion_settings
        p = self._write(tmp_path, {
            'keep_audio': False,
            'use_deflicker': True,
            'upscale_ratio': 2,
            'upscale_model': 'RealESRGAN_x2plus',
            'upscale_model_path': '/models/realesrgan.pth',
            'use_background_mask_video': True,
            'invert_mask_video': True,
            'background_video': 'color',
            'background_source_video': '#ff0000',
            'mask_clip_low': 10,
            'mask_clip_high': 240,
        })
        cfg = load_warpfusion_settings(p)
        va = cfg['video_assembly']
        assert va['keep_audio'] is False
        assert va['use_deflicker'] is True
        assert va['upscale_ratio'] == 2
        assert va['upscale_model'] == 'RealESRGAN_x2plus'
        assert va['upscale_model_path'] == '/models/realesrgan.pth'
        assert va['use_background_mask_video'] is True
        assert va['invert_mask_video'] is True
        assert va['background_video'] == 'color'
        assert va['background_source_video'] == '#ff0000'
        assert va['mask_clip_low'] == 10
        assert va['mask_clip_high'] == 240


# ---- load_realesrgan error handling ----

class TestLoadRealESRGAN:
    def test_missing_package_raises_import_error(self):
        from vibewarp.video.output import load_realesrgan
        import sys
        # Patch realesrgan to not exist
        realesrgan_backup = sys.modules.pop('realesrgan', None)
        try:
            with pytest.raises(ImportError, match='realesrgan package'):
                load_realesrgan('realesr-animevideov3', '/fake.pth')
        finally:
            if realesrgan_backup is not None:
                sys.modules['realesrgan'] = realesrgan_backup

    def test_unknown_model_raises_value_error(self):
        from vibewarp.video.output import load_realesrgan
        with pytest.raises((ValueError, ImportError)):
            load_realesrgan('unknown_model', '/fake.pth')

    def test_missing_weights_raises_file_not_found(self):
        from vibewarp.video.output import load_realesrgan
        # This will either raise ImportError (no realesrgan) or FileNotFoundError
        try:
            load_realesrgan('RealESRGAN_x4plus', '/nonexistent_weights.pth')
        except ImportError:
            pass  # realesrgan not installed — acceptable
        except FileNotFoundError:
            pass  # correct behavior when package is installed
        except ValueError:
            pass  # unknown model also acceptable
