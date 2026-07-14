"""Tests for AnimateDiff batch-mode frame rendering."""

import os
import pytest
import numpy as np
import torch
from PIL import Image
from unittest.mock import MagicMock, patch

from vibewarp.config import AnimateDiffConfig, RunConfig, VideoConfig
from vibewarp.core.diffusion import (
    RenderContext,
    FrameState,
    _resolve_frame_range,
    _render_single_frame,
    run_frames,
    run_frames_animatediff,
)


class TestResolveFrameRange:
    """frame_range is INCLUSIVE at both ends -- 0-15 is sixteen frames, which is what the
    two numbers in the UI say and what the notebook means. It is converted to the internal
    half-open bound here and nowhere else, so `range(start, end)` still works downstream.
    """

    def test_explicit_range_is_inclusive(self):
        ctx = RenderContext(config=RunConfig())
        start, end = _resolve_frame_range(ctx, [5, 20], resume_from=0)
        assert (start, end) == (5, 21)          # frames 5..20 -> 16 frames
        assert end - start == 16

    def test_the_range_that_broke_animatediff(self):
        """0-15 must be 16 frames: with context_length=16 it used to raise
        "AnimateDiff needs at least context_length (16) frames, got 15"."""
        ctx = RenderContext(config=RunConfig())
        start, end = _resolve_frame_range(ctx, [0, 15], resume_from=0)
        assert end - start == 16

    def test_resume_from(self):
        ctx = RenderContext(config=RunConfig())
        start, end = _resolve_frame_range(ctx, [0, 20], resume_from=10)
        assert start == 10
        assert end == 21

    def test_from_config(self):
        cfg = RunConfig(frame_range=[0, 50])
        ctx = RenderContext(config=cfg)
        start, end = _resolve_frame_range(ctx, None, resume_from=0)
        assert (start, end) == (0, 51)

    def test_invalid_range_raises(self):
        ctx = RenderContext(config=RunConfig())
        with pytest.raises(ValueError, match="Invalid frame range"):
            _resolve_frame_range(ctx, [10, 5], resume_from=0)

    def test_equal_range_raises(self):
        ctx = RenderContext(config=RunConfig())
        with pytest.raises(ValueError, match="Invalid frame range"):
            _resolve_frame_range(ctx, [5, 5], resume_from=0)


class TestRunFramesDelegation:
    def test_delegates_to_animatediff_when_enabled(self):
        """run_frames should call run_frames_animatediff when animatediff is enabled."""
        cfg = RunConfig(
            frame_range=[0, 10],
            animatediff=AnimateDiffConfig(enabled=True, batch_length=8, batch_overlap=2),
        )
        ctx = RenderContext(config=cfg, batch_folder='/tmp/test')

        with patch('vibewarp.core.diffusion.run_frames_animatediff') as mock_ad:
            mock_ad.return_value = []
            run_frames(ctx, frame_range=[0, 10])
            mock_ad.assert_called_once()

    def test_does_not_delegate_when_disabled(self):
        """run_frames should NOT call run_frames_animatediff when disabled."""
        cfg = RunConfig(
            frame_range=[0, 10],
            animatediff=AnimateDiffConfig(enabled=False),
        )
        ctx = RenderContext(config=cfg, batch_folder='/tmp/test')

        with patch('vibewarp.core.diffusion.run_frames_animatediff') as mock_ad:
            with patch('vibewarp.core.diffusion.render_frame') as mock_rf:
                mock_rf.return_value = Image.new('RGB', (64, 64))
                try:
                    run_frames(ctx, frame_range=[0, 3])
                except Exception:
                    pass  # May fail without real models
                mock_ad.assert_not_called()


class TestRunFramesAnimatediff:
    """The renderer must denoise a BATCH of frames jointly, not loop per frame.

    These tests previously mocked `render_frame` and asserted it was called once per
    frame — i.e. they encoded the broken behaviour (a batch-1 latent, motion module
    with nothing to attend across) as the expected contract. That is why the bug
    survived. They now assert the opposite.
    """

    def _ctx(self, tmp_path, frames=20, batch_length=16, overlap=4, context_length=16):
        vf = tmp_path / 'video_frames'
        vf.mkdir(parents=True, exist_ok=True)
        for n in range(1, frames + 2):
            Image.new('RGB', (64, 64)).save(vf / f'{n:06d}.jpg')
        batch = tmp_path / 'batch'
        batch.mkdir(exist_ok=True)
        cfg = RunConfig(
            frame_range=[0, frames],
            video=VideoConfig(width=64, height=64),
            animatediff=AnimateDiffConfig(
                enabled=True, batch_length=batch_length, batch_overlap=overlap,
                context_length=context_length, context_overlap=2,
            ),
        )
        return RenderContext(config=cfg, batch_folder=str(batch),
                             video_frames_folder=str(vf))

    def test_never_falls_back_to_per_frame_rendering(self, tmp_path):
        ctx = self._ctx(tmp_path)
        with patch('vibewarp.core.diffusion.render_frame') as per_frame,              patch('vibewarp.core.diffusion._adiff_run_batch') as run_batch:
            run_batch.side_effect = lambda ctx, frames, *a, **k: [
                Image.new('RGB', (64, 64)) for _ in frames]
            run_frames_animatediff(ctx, frame_range=[0, 20])
        per_frame.assert_not_called()
        assert run_batch.call_count >= 1

    def test_denoises_whole_batches(self, tmp_path):
        """Each call must receive many frames — that batch axis IS the temporal
        axis the motion module attends across."""
        ctx = self._ctx(tmp_path, frames=20, batch_length=16, overlap=4)
        sizes = []
        with patch('vibewarp.core.diffusion._adiff_run_batch') as run_batch:
            def record(ctx, frames, *a, **k):
                sizes.append(len(frames))
                return [Image.new('RGB', (64, 64)) for _ in frames]
            run_batch.side_effect = record
            run_frames_animatediff(ctx, frame_range=[0, 20])
        assert sizes
        assert all(size >= ctx.config.animatediff.context_length for size in sizes)

    def test_refuses_fewer_frames_than_a_context_window(self, tmp_path):
        """A window shorter than context_length would rearrange to the wrong
        temporal size inside the motion module."""
        ctx = self._ctx(tmp_path, frames=8, batch_length=16, context_length=16)
        with pytest.raises(ValueError, match="context_length"):
            run_frames_animatediff(ctx, frame_range=[0, 8])


class TestPipelineIPAdapterWiring:
    def test_apply_ipadapter_hooks_called(self):
        """pipeline.run() should call _apply_ipadapter_hooks when ipadapters loaded."""
        from unittest.mock import patch
        from vibewarp.pipeline import _apply_ipadapter_hooks

        class MockSDModel:
            pass

        sd_model = MockSDModel()
        mock_weights = {'image_proj': {}, 'ip_adapter': {}}
        ipadapters = {
            'test_adapter': {
                'weights': mock_weights,
                'type_info': {'is_plus': False, 'is_sdxl': False},
                'weight': 0.7,
                'start': 0.0,
                'end': 1.0,
                'source_image': '',  # no source image
                'weight_type': 'linear',
                'embeds_scaling': 'V only',
            }
        }
        config = RunConfig()

        # Mock hook_ipadapter to avoid needing a real UNet
        def fake_hook(sd_model, adapter_weights, clip_vision_output, **kwargs):
            if not hasattr(sd_model, '_ipadapter_hooks'):
                sd_model._ipadapter_hooks = []
            sd_model._ipadapter_hooks.append(kwargs)

        with patch('vibewarp.core.ipadapter.hook_ipadapter', side_effect=fake_hook):
            _apply_ipadapter_hooks(sd_model, ipadapters, config)
        assert hasattr(sd_model, '_ipadapter_hooks')
        assert len(sd_model._ipadapter_hooks) == 1

    def test_apply_ipadapter_hooks_weight(self):
        """Hook should preserve the weight from config."""
        from unittest.mock import patch
        from vibewarp.pipeline import _apply_ipadapter_hooks

        class MockSDModel:
            pass

        sd_model = MockSDModel()
        mock_weights = {'image_proj': {}, 'ip_adapter': {}}
        ipadapters = {
            'ipa1': {
                'weights': mock_weights,
                'type_info': {'is_plus': False, 'is_sdxl': False},
                'weight': 0.42,
                'start': 0.1,
                'end': 0.9,
                'source_image': '',
                'weight_type': 'ease in',
                'embeds_scaling': 'K+V',
            }
        }
        config = RunConfig()

        def fake_hook(sd_model, adapter_weights, clip_vision_output, **kwargs):
            if not hasattr(sd_model, '_ipadapter_hooks'):
                sd_model._ipadapter_hooks = []
            sd_model._ipadapter_hooks.append(kwargs)

        with patch('vibewarp.core.ipadapter.hook_ipadapter', side_effect=fake_hook):
            _apply_ipadapter_hooks(sd_model, ipadapters, config)
        hook = sd_model._ipadapter_hooks[0]
        assert hook['weight'] == 0.42
        assert hook['start'] == 0.1
        assert hook['end'] == 0.9
        assert hook['weight_type'] == 'ease in'
        assert hook['embeds_scaling'] == 'K+V'


class TestAnimateDiffNoise:
    """The notebook's AnimateDiff noise, and why we do not copy its RNG.

    do_run_adiff draws `big_noise = torch.randn((end_frame+1, 4, H//8, W//8),
    device='cpu')` ONCE for the whole video and slices `big_noise[frame_idxs]` --
    absolute-frame indexing, i.e. frame-keyed noise. Nothing seeds the CPU generator
    before that draw, so its VALUES are process-random: two notebook runs at the same
    seed were measured to disagree at MAE 0.30 (2026-07-13). We stay deterministic
    instead, and inject the notebook's noise when we need a comparable run.
    """

    def test_frame_keyed_noise_is_the_default(self):
        assert AnimateDiffConfig().frame_keyed_noise is True

    def test_noise_is_keyed_to_the_frame_not_the_batch(self):
        from vibewarp.core.diffusion import _adiff_frame_noise
        latents = torch.zeros(3, 4, 8, 6)
        # Frame 4 appears at a different position in each overlapping batch...
        first = _adiff_frame_noise([2, 3, 4], latents, seed=99)
        second = _adiff_frame_noise([4, 5, 6], latents, seed=99)
        # ...and must still receive the identical noise vector.
        assert torch.equal(first[2], second[0])
        # Different frames get different noise.
        assert not torch.equal(first[0], first[1])

    def test_noise_is_deterministic_across_runs(self):
        from vibewarp.core.diffusion import _adiff_frame_noise
        latents = torch.zeros(2, 4, 8, 6)
        again = _adiff_frame_noise([7, 8], latents, seed=1234)
        assert torch.equal(_adiff_frame_noise([7, 8], latents, seed=1234), again)

    def test_injected_noise_is_sliced_by_absolute_frame(self, tmp_path):
        from vibewarp.core.diffusion import _adiff_injected_noise
        big_noise = torch.randn(16, 4, 8, 6)
        path = tmp_path / 'diag_adiff_noise.pt'
        torch.save(big_noise, path)
        latents = torch.zeros(3, 4, 8, 6)
        with patch.dict(os.environ, {'VIBEWARP_ADIFF_NOISE': str(path)}):
            noise = _adiff_injected_noise([12, 13, 14], latents)
        assert torch.equal(noise, big_noise[[12, 13, 14]])

    def test_no_injection_without_the_env_var(self):
        from vibewarp.core.diffusion import _adiff_injected_noise
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('VIBEWARP_ADIFF_NOISE', None)
            assert _adiff_injected_noise([0], torch.zeros(1, 4, 8, 6)) is None

    def test_injection_rejects_a_noise_file_that_is_too_short(self, tmp_path):
        from vibewarp.core.diffusion import _adiff_injected_noise
        path = tmp_path / 'short.pt'
        torch.save(torch.randn(4, 4, 8, 6), path)
        with patch.dict(os.environ, {'VIBEWARP_ADIFF_NOISE': str(path)}):
            with pytest.raises(ValueError, match='needs frame 9'):
                _adiff_injected_noise([8, 9], torch.zeros(2, 4, 8, 6))


class TestAnimateDiffNoisePadding:
    """A short final batch repeats the last FRAME but must not repeat its NOISE.

    Notebook (do_run_adiff):
        # pad noise for those frames from the beginning of the vector (can't repeat noise)
        noise = torch.cat([big_noise[frame_idxs],
                           big_noise[:batch_length - len(frame_idxs)]])
        frame_idxs += [frame_idxs[-1]] * (batch_length - len(frame_idxs))

    Indexing the noise by the PADDED frame list instead gave every padded slot the same
    noise as the real last frame; measured MAE 0.199 on the padded tail (frame 63: 0.397)
    while every other region of the same run sat at 0.005-0.010.
    """

    def test_padded_slots_take_noise_from_the_head(self):
        from vibewarp.core.diffusion import _adiff_noise_indices
        # batch_length 8, real frames 48..51, padded with three copies of 51
        frames = [48, 49, 50, 51, 51, 51, 51]
        assert _adiff_noise_indices(frames) == [48, 49, 50, 51, 0, 1, 2]

    def test_unpadded_batch_is_untouched(self):
        from vibewarp.core.diffusion import _adiff_noise_indices
        assert _adiff_noise_indices([12, 13, 14, 15]) == [12, 13, 14, 15]

    def test_injected_noise_honours_the_padding_rule(self, tmp_path):
        from vibewarp.core.diffusion import _adiff_injected_noise
        big_noise = torch.randn(64, 4, 8, 6)
        path = tmp_path / 'diag_adiff_noise.pt'
        torch.save(big_noise, path)
        frames = [62, 63, 63, 63]
        latents = torch.zeros(len(frames), 4, 8, 6)
        with patch.dict(os.environ, {'VIBEWARP_ADIFF_NOISE': str(path)}):
            noise = _adiff_injected_noise(frames, latents)
        assert torch.equal(noise, big_noise[[62, 63, 0, 1]])
        # the padded slots must NOT be a repeat of frame 63's noise
        assert not torch.equal(noise[2], noise[1])

    def test_frame_keyed_noise_honours_the_padding_rule(self):
        from vibewarp.core.diffusion import _adiff_frame_noise
        latents = torch.zeros(4, 4, 8, 6)
        noise = _adiff_frame_noise([62, 63, 63, 63], latents, seed=7)
        head = _adiff_frame_noise([0, 1], torch.zeros(2, 4, 8, 6), seed=7)
        assert torch.equal(noise[2], head[0])
        assert torch.equal(noise[3], head[1])
        assert not torch.equal(noise[2], noise[1])
