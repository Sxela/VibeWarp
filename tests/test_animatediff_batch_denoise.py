"""AnimateDiff joint-batch denoising — the thing that was never implemented.

The old code rendered each frame separately (batch-1 latent), so the motion module
had no temporal axis to attend across and AnimateDiff did nothing. These tests pin
the invariants of the real path: the UNet must see a real batch of frames, exactly
`context_length` of them per window, with cond and uncond as SEPARATE forwards.
"""

import pytest
import torch

from vibewarp.core.animatediff import (
    TWO_EVAL_SAMPLERS,
    make_context_schedule,
    repeat_schedule_for_sampler,
)
from vibewarp.core.model_loader import CFGDenoiser
from vibewarp.core.sampling import TiledModelFn


class _RecordingInner(torch.nn.Module):
    """Stands in for the k-diffusion-wrapped UNet; records what it was called with."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, sigma, cond=None, **kwargs):
        self.calls.append({
            'frames': x.shape[0],
            'cond': cond,
            'hints': {k: v.shape[0] for k, v in (kwargs.get('_hints') or {}).items()},
        })
        # Return something that depends on cond so cond/uncond are distinguishable.
        value = float(cond.mean().item()) if torch.is_tensor(cond) else 0.0
        return torch.full_like(x, value)


def _denoiser(schedule, sd_model=None):
    inner = _RecordingInner()
    denoiser = CFGDenoiser(inner)
    denoiser._adiff_ctx_schedule = schedule
    denoiser._get_sd_model = lambda: sd_model
    return denoiser, inner


class TestContextSchedule:
    def test_every_window_is_exactly_context_length(self):
        """The motion module is told `set_video_length(context_length)` once, so a
        short window would rearrange to the wrong temporal size."""
        schedule = make_context_schedule(total_length=40, context_length=16,
                                         overlap=4, steps=10)
        for windows in schedule:
            for window in windows:
                assert len(window) == 16

    def test_every_frame_is_covered_each_step(self):
        """An uncovered frame keeps its zero-initialised prediction and decodes to
        garbage — the denoiser raises rather than allow it."""
        schedule = make_context_schedule(total_length=37, context_length=16,
                                         overlap=4, steps=6)
        for windows in schedule:
            covered = {i for window in windows for i in window}
            assert covered == set(range(37))

    def test_head_and_tail_are_always_windows(self):
        schedule = make_context_schedule(total_length=40, context_length=16,
                                         overlap=4, steps=3)
        for windows in schedule:
            assert list(range(16)) in windows
            assert list(range(24, 40)) in windows

    def test_windows_shift_between_steps(self):
        """The per-step offset is what keeps window seams from landing in the same
        place every step."""
        schedule = make_context_schedule(total_length=48, context_length=16,
                                         overlap=4, steps=8)
        distinct = {tuple(tuple(w) for w in sorted(windows)) for windows in schedule}
        assert len(distinct) > 1

    def test_batch_shorter_than_context_length_is_one_window(self):
        schedule = make_context_schedule(total_length=16, context_length=16,
                                         overlap=4, steps=3)
        for windows in schedule:
            assert windows == [list(range(16))]

    def test_two_eval_samplers_get_a_doubled_schedule(self):
        """dpmpp_sde and friends call the model twice per step, so they consume two
        window lists per step."""
        schedule = make_context_schedule(total_length=32, context_length=16,
                                         overlap=4, steps=5)
        doubled = repeat_schedule_for_sampler(schedule, 'sample_dpmpp_sde')
        assert len(doubled) == 2 * len(schedule)
        assert 'sample_dpmpp_sde' in TWO_EVAL_SAMPLERS

    def test_single_eval_sampler_is_left_alone(self):
        schedule = make_context_schedule(total_length=32, context_length=16,
                                         overlap=4, steps=5)
        assert repeat_schedule_for_sampler(schedule, 'sample_euler') is schedule


class TestBatchDenoise:
    def test_unet_sees_a_batch_of_frames_not_one(self):
        """The whole point: a batch-1 latent gives the motion module nothing to
        attend across."""
        schedule = [[list(range(8))]]
        denoiser, inner = _denoiser(schedule)
        x = torch.zeros(8, 4, 8, 8)
        sigma = torch.ones(8)
        denoiser(x, sigma, uncond=torch.zeros(8, 77, 4), cond=torch.ones(8, 77, 4),
                 cond_scale=7.0)
        assert inner.calls, "the inner model was never called"
        assert all(call['frames'] == 8 for call in inner.calls)

    def test_cond_and_uncond_are_separate_forwards(self):
        """Concatenating them (the standard CFG path) would make the temporal axis
        2F and corrupt every motion-module attention."""
        denoiser, inner = _denoiser([[list(range(4))]])
        x = torch.zeros(4, 4, 8, 8)
        denoiser(x, torch.ones(4), uncond=torch.zeros(4, 77, 4),
                 cond=torch.ones(4, 77, 4), cond_scale=7.0)
        assert len(inner.calls) == 2                      # one cond, one uncond
        assert all(call['frames'] == 4 for call in inner.calls)   # never 8

    def test_overlapping_windows_are_averaged(self):
        """A frame covered by two windows must be the mean of both, not the sum."""
        # Frame 2 and 3 appear in both windows.
        denoiser, inner = _denoiser([[[0, 1, 2, 3], [2, 3, 4, 5]]])
        x = torch.zeros(6, 4, 8, 8)
        cond = torch.ones(6, 77, 4)
        out = denoiser(x, torch.ones(6), uncond=torch.zeros(6, 77, 4),
                       cond=cond, cond_scale=1.0)
        # inner returns cond.mean() everywhere: cond=1 -> 1, uncond=0 -> 0.
        # CFG at scale 1: uncond + (cond - uncond) * 1 = 1 for every frame,
        # including the doubly-covered ones (sum would give 2).
        assert torch.allclose(out, torch.ones_like(out))

    def test_cfg_is_applied_after_averaging(self):
        denoiser, _ = _denoiser([[list(range(4))]])
        x = torch.zeros(4, 4, 8, 8)
        out = denoiser(x, torch.ones(4), uncond=torch.zeros(4, 77, 4),
                       cond=torch.full((4, 77, 4), 2.0), cond_scale=3.0)
        # uncond=0, cond=2 -> 0 + (2 - 0) * 3 = 6
        assert torch.allclose(out, torch.full_like(out, 6.0))

    def test_uncovered_frame_raises_instead_of_decoding_garbage(self):
        denoiser, _ = _denoiser([[[0, 1]]])       # frames 2,3 never covered
        with pytest.raises(RuntimeError, match="uncovered"):
            denoiser(torch.zeros(4, 4, 8, 8), torch.ones(4),
                     uncond=torch.zeros(4, 77, 4), cond=torch.ones(4, 77, 4),
                     cond_scale=7.0)

    def test_constant_prompt_is_broadcast_across_the_batch(self):
        """A single prompt arrives as [1,77,D] and must cover every frame."""
        denoiser, inner = _denoiser([[list(range(6))]])
        denoiser(torch.zeros(6, 4, 8, 8), torch.ones(6),
                 uncond=torch.zeros(1, 77, 4), cond=torch.ones(1, 77, 4),
                 cond_scale=7.0)
        for call in inner.calls:
            assert call['cond'].shape[0] == 6

    def test_schedule_is_consumed_one_entry_per_call(self):
        schedule = [[[0, 1, 2, 3]], [[0, 1, 2, 3]]]
        denoiser, inner = _denoiser(schedule)
        x = torch.zeros(4, 4, 8, 8)
        args = dict(uncond=torch.zeros(4, 77, 4), cond=torch.ones(4, 77, 4),
                    cond_scale=1.0)
        denoiser(x, torch.ones(4), **args)
        assert len(schedule) == 1        # one popped
        denoiser(x, torch.ones(4), **args)
        assert len(schedule) == 0

    def test_spatial_tiles_do_not_consume_extra_context_steps(self):
        schedule = [[[0, 1, 2, 3]], [[0, 1, 2, 3]]]
        denoiser, inner = _denoiser(schedule)
        denoiser._tiled_inner_model = TiledModelFn(
            inner, tiles_h=2, tiles_w=2, overlap_h=25, overlap_w=25)
        denoiser(
            torch.zeros(4, 4, 8, 8), torch.ones(4),
            uncond=torch.zeros(4, 77, 4), cond=torch.ones(4, 77, 4),
            cond_scale=1.0,
        )

        assert len(schedule) == 1
        # One temporal window, cond + uncond, four spatial tiles each.
        assert len(inner.calls) == 8

    def test_standard_path_is_untouched_without_a_schedule(self):
        """Non-AnimateDiff rendering must be completely unaffected."""
        denoiser = CFGDenoiser(_RecordingInner())
        assert denoiser._adiff_ctx_schedule is None


class TestReinjectStylized:
    """The previous batch's latents are pinned into the new batch's overlap frames
    on EVERY denoiser call (notebook reinject_stylized, default on).

    This — not `overlap_stylized` — is what carries identity across a batch seam.
    With style_strength=1 the init latent is discarded (skip_steps=0), so choosing a
    stylized init IMAGE achieves nothing; only the latent-level reinjection bites.
    """

    def _denoiser_with_reinject(self, latents, noise, lerp=0):
        denoiser, inner = _denoiser([[list(range(4))]])
        denoiser._adiff_stylized_latents = latents
        denoiser._adiff_noise = noise
        denoiser._adiff_lerp_frames = lerp
        return denoiser, inner

    def test_overlap_frames_are_pinned_to_the_previous_batch(self):
        latents = torch.full((2, 4, 8, 8), 5.0)     # 2 frames from the last batch
        noise = torch.zeros(4, 4, 8, 8)             # no noise -> exact pin
        denoiser, _ = self._denoiser_with_reinject(latents, noise)

        x = torch.zeros(4, 4, 8, 8)
        denoiser(x, torch.ones(4), uncond=torch.zeros(4, 77, 4),
                 cond=torch.ones(4, 77, 4), cond_scale=1.0)

        # x is mutated in place: the sampler carries the same tensor forward, so the
        # anchor steers the whole trajectory of those frames.
        assert torch.allclose(x[:2], torch.full((2, 4, 8, 8), 5.0))
        assert torch.allclose(x[2:], torch.zeros(2, 4, 8, 8))   # untouched

    def test_injected_noise_scales_with_sigma(self):
        """x[:n] = latents + noise[:n] * sigma[0] — as sigma shrinks the anchor
        tightens onto the previous batch's content."""
        latents = torch.zeros(2, 4, 8, 8)
        noise = torch.ones(4, 4, 8, 8)
        denoiser, _ = self._denoiser_with_reinject(latents, noise)

        x = torch.zeros(4, 4, 8, 8)
        denoiser(x, torch.full((4,), 3.0), uncond=torch.zeros(4, 77, 4),
                 cond=torch.ones(4, 77, 4), cond_scale=1.0)
        assert torch.allclose(x[:2], torch.full((2, 4, 8, 8), 3.0))

    def test_lerp_ramps_the_anchor_out(self):
        """lerp_reinject_stylized_frames fades the last anchored frames back toward
        the batch's own latent instead of ending the anchor abruptly."""
        latents = torch.zeros(3, 4, 8, 8)     # anchor value 0
        noise = torch.zeros(4, 4, 8, 8)
        denoiser, _ = self._denoiser_with_reinject(latents, noise, lerp=2)

        x = torch.ones(4, 4, 8, 8)            # batch's own latent = 1
        denoiser(x, torch.ones(4), uncond=torch.zeros(4, 77, 4),
                 cond=torch.ones(4, 77, 4), cond_scale=1.0)

        # weights = [0, 0, 1] over the 3 anchored frames: frame 0 fully pinned,
        # frame 2 fully back to its own latent.
        assert torch.allclose(x[0], torch.zeros(4, 8, 8))
        assert torch.allclose(x[2], torch.ones(4, 8, 8))

    def test_first_batch_has_nothing_to_reinject(self):
        denoiser, _ = _denoiser([[list(range(4))]])
        denoiser._adiff_stylized_latents = None
        denoiser._adiff_noise = torch.ones(4, 4, 8, 8)

        x = torch.ones(4, 4, 8, 8)
        denoiser(x, torch.ones(4), uncond=torch.zeros(4, 77, 4),
                 cond=torch.ones(4, 77, 4), cond_scale=1.0)
        assert torch.allclose(x, torch.ones(4, 4, 8, 8))   # untouched


class TestStatefulSamplerGuard:
    def test_stateful_samplers_are_flagged(self):
        """reinject_stylized mutates x in place, which corrupts the state these
        samplers carry between steps — verified to produce streaked frames."""
        from vibewarp.core.animatediff import STATEFUL_SAMPLERS
        assert 'sample_dpmpp_sde' in STATEFUL_SAMPLERS
        assert 'sample_euler' not in STATEFUL_SAMPLERS   # stateless: safe

    def test_warns_when_reinject_meets_a_stateful_sampler(self, tmp_path):
        from unittest.mock import patch

        from vibewarp.config import AnimateDiffConfig, RunConfig, VideoConfig
        from vibewarp.core.diffusion import RenderContext, _adiff_run_batch

        config = RunConfig(video=VideoConfig(width=64, height=64))
        config.animatediff = AnimateDiffConfig(
            enabled=True, reinject_stylized=True, context_length=8)
        ctx = RenderContext(config=config, batch_folder=str(tmp_path),
                            video_frames_folder=str(tmp_path))

        # Fail fast inside the batch; we only care that the warning is raised first.
        with patch('vibewarp.core.diffusion.get_frame_schedule',
                   side_effect=RuntimeError('stop')):
            with pytest.warns(RuntimeWarning, match='reinject_stylized'):
                with pytest.raises(RuntimeError):
                    _adiff_run_batch(ctx, list(range(8)), 0, lambda *a, **k: None,
                                     'sample_dpmpp_sde')


class TestControlNetHintWindows:
    def test_hints_are_sliced_per_window(self):
        """Hints are per-frame ([F,3,H,W]); a window must get its own rows, or every
        frame would be conditioned on the same structure."""
        class _Model:
            def __init__(self):
                self._cn_hints = {'depth': torch.arange(6).float().view(6, 1, 1, 1)}
                self.model = type('m', (), {'diffusion_model': type('u', (), {})()})()

        sd_model = _Model()
        seen = []

        class _Inner(torch.nn.Module):
            def forward(self, x, sigma, cond=None, **kwargs):
                seen.append(sd_model._cn_hints['depth'].flatten().tolist())
                return torch.zeros_like(x)

        denoiser = CFGDenoiser(_Inner())
        denoiser._adiff_ctx_schedule = [[[0, 1, 2], [3, 4, 5]]]
        denoiser._get_sd_model = lambda: sd_model

        denoiser(torch.zeros(6, 4, 8, 8), torch.ones(6),
                 uncond=torch.zeros(6, 77, 4), cond=torch.ones(6, 77, 4),
                 cond_scale=1.0)

        # cond+uncond per window -> 4 calls; windows carry their own hint rows.
        assert seen[0] == [0.0, 1.0, 2.0]
        assert seen[2] == [3.0, 4.0, 5.0]
        # The full-batch hints are restored afterwards.
        assert sd_model._cn_hints['depth'].shape[0] == 6
