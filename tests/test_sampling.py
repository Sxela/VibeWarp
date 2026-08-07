"""Tests for vibewarp.core.sampling."""

import math

import torch

from vibewarp.core.sampling import (
    TiledModelFn,
    alpha_sigma_to_t,
    append_dims,
    build_tile_specs,
    compute_tile_starts,
    expand_to_planes,
    get_premature_sigma_min,
    high_frequency_loss,
    t_to_alpha_sigma,
    tile_count_for_size,
    torch_interp1d,
)


class TestAppendDims:
    def test_basic(self):
        x = torch.randn(4)
        result = append_dims(x, 3)
        assert result.shape == (4, 1, 1)

    def test_no_extra_dims(self):
        x = torch.randn(4, 3)
        result = append_dims(x, 2)
        assert result.shape == (4, 3)


class TestExpandToPlanes:
    def test_basic(self):
        x = torch.randn(2, 4)
        shape = (2, 4, 8, 8)
        result = expand_to_planes(x, shape)
        assert result.shape == (2, 4, 8, 8)


class TestAlphaSigmaConversion:
    def test_roundtrip(self):
        t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        alpha, sigma = t_to_alpha_sigma(t)
        t_back = alpha_sigma_to_t(alpha, sigma)
        torch.testing.assert_close(t, t_back, atol=1e-6, rtol=1e-6)

    def test_boundaries(self):
        alpha_0, sigma_0 = t_to_alpha_sigma(torch.tensor(0.0))
        assert abs(alpha_0.item() - 1.0) < 1e-6  # t=0 -> pure signal
        assert abs(sigma_0.item()) < 1e-6

        alpha_1, sigma_1 = t_to_alpha_sigma(torch.tensor(1.0))
        assert abs(alpha_1.item()) < 1e-6  # t=1 -> pure noise
        assert abs(sigma_1.item() - 1.0) < 1e-6


class TestGetPrematureSigmaMin:
    def test_full_schedule(self):
        result = get_premature_sigma_min(50, 0.01, 14.6, stop_pct=1.0)
        assert result == 0.01

    def test_partial_schedule(self):
        result = get_premature_sigma_min(50, 0.01, 14.6, stop_pct=0.5)
        assert result > 0.01
        assert result < 14.6


class TestHighFrequencyLoss:
    def test_identical_images(self):
        img = torch.randn(1, 3, 32, 32)
        loss = high_frequency_loss(img, img)
        assert loss.item() < 1e-10

    def test_different_images(self):
        img1 = torch.randn(1, 3, 32, 32)
        img2 = torch.randn(1, 3, 32, 32)
        loss = high_frequency_loss(img1, img2)
        assert loss.item() > 0


class TestTorchInterp1d:
    def test_exact_points(self):
        xp = torch.tensor([0.0, 1.0, 2.0])
        fp = torch.tensor([0.0, 10.0, 20.0])
        x = torch.tensor([0.0, 1.0, 2.0])
        result = torch_interp1d(x, xp, fp)
        torch.testing.assert_close(result, fp)

    def test_midpoints(self):
        xp = torch.tensor([0.0, 1.0, 2.0])
        fp = torch.tensor([0.0, 10.0, 20.0])
        x = torch.tensor([0.5])
        result = torch_interp1d(x, xp, fp)
        assert abs(result.item() - 5.0) < 1e-5


class TestTiledModelFn:
    def test_tile_count_is_derived_per_dimension(self):
        # With no overlap the rule reduces to plain ceil().
        assert tile_count_for_size(64, 64) == 1
        assert tile_count_for_size(128, 64) == 2
        assert tile_count_for_size(129, 64) == 3

    def test_a_dimension_barely_over_the_tile_is_not_split(self):
        """REGRESSION: ceil(total/tile) split 1080px into 2 tiles for a 1024px target.

        The tile size is DERIVED from the count, so those two came out ~640px each: you
        asked for 1024px tiles, silently got 640px ones, at twice the model evals. The
        split threshold absorbs a small leftover into the existing tile instead.
        """
        tile, threshold = 1024 // 8, 20.0             # latent units
        assert tile_count_for_size(1080 // 8, tile, threshold) == 1
        assert tile_count_for_size(1300 // 8, tile, threshold) == 2

    def test_threshold_does_not_inflate_the_count(self):
        """The point is FEWER model evals, never more.

        A first attempt tied the count to the overlap (asking "how many tiles of exactly
        this size does it take to COVER the dimension"). With 35% overlap that needed 5
        tiles to cover 1920px with 512px tiles — slower than the bug it was fixing.
        """
        tile = 512 // 8
        for px in (512, 768, 1024, 1080, 1920, 2560):
            with_slack = tile_count_for_size(px // 8, tile, 20.0)
            plain_ceil = tile_count_for_size(px // 8, tile, 0.0)
            assert with_slack <= plain_ceil, f'{px}px: threshold ADDED tiles'

    def test_zero_threshold_is_the_old_ceil_behaviour(self):
        tile = 128
        assert tile_count_for_size(128, tile, 0.0) == 1
        assert tile_count_for_size(129, tile, 0.0) == 2

    def test_the_real_render_grid(self):
        """1080x1920 with 512px tiles: 4x3 = 12 evals/step before, 4x2 = 8 after."""
        tile = 512 // 8
        assert tile_count_for_size(1920 // 8, tile, 20.0) == 4
        assert tile_count_for_size(1080 // 8, tile, 20.0) == 2

    def test_count_never_collapses_to_zero(self):
        assert tile_count_for_size(1, 64, 20.0) == 1
        assert tile_count_for_size(0, 64, 20.0) == 1
        # A threshold big enough to swallow the whole dimension must still leave one tile.
        assert tile_count_for_size(64, 64, 500.0) == 1

    def test_two_tile_geometry_covers_dimension(self):
        starts, size = compute_tile_starts(16, 2, 4)
        assert starts == [0, 6]
        assert size == 10
        assert starts[-1] + size == 16

    def test_blend_weights_cover_every_pixel(self):
        specs = build_tile_specs(15, 17, 3, 2, 25, 25)
        weights = torch.zeros(15, 17)
        for top, bottom, left, right, window in specs:
            weights[top:bottom, left:right] += window
        assert len(specs) == 6
        assert bool((weights > 0).all())

    def test_tiles_every_model_evaluation_and_blends_back(self):
        calls = []

        def model(x, sigma, **kwargs):
            calls.append((tuple(x.shape), kwargs))
            return torch.full_like(x, 3.0)

        tiled = TiledModelFn(model, tiles_h=2, tiles_w=2,
                             overlap_h=25, overlap_w=25)
        x = torch.randn(2, 4, 12, 16)
        image_cond = torch.randn(2, 3, 96, 128)
        result = tiled(x, torch.ones(2), image_cond=image_cond,
                       cond=torch.randn(2, 77, 8))

        assert len(calls) == 4
        assert result.shape == x.shape
        torch.testing.assert_close(result, torch.full_like(x, 3.0))
        assert all(call[1]['image_cond'].shape[-2:] < image_cond.shape[-2:]
                   for call in calls)
        assert all(call[1]['cond'].shape == (2, 77, 8) for call in calls)

    def test_tile_size_automatically_selects_portrait_grid(self):
        calls = []

        def model(x, sigma, **kwargs):
            calls.append(tuple(x.shape[-2:]))
            return x

        tiled = TiledModelFn(model, tile_size=64)
        x = torch.randn(1, 4, 128, 64)
        torch.testing.assert_close(tiled(x, torch.ones(1)), x)

        assert len(calls) == 2  # 2 rows x 1 column

    def test_controlnet_hints_are_cropped_and_restored(self):
        class Model:
            pass

        sd_model = Model()
        original = torch.randn(1, 3, 64, 80)
        sd_model._cn_hints = {'depth': original}
        seen = []

        def model(x, sigma, **kwargs):
            seen.append(sd_model._cn_hints['depth'].shape[-2:])
            return x

        tiled = TiledModelFn(model, tiles_h=2, tiles_w=2, sd_model=sd_model)
        x = torch.randn(1, 4, 8, 10)
        torch.testing.assert_close(tiled(x, torch.ones(1)), x)

        assert len(seen) == 4
        assert all(h < 64 and w < 80 for h, w in seen)
        assert sd_model._cn_hints['depth'] is original

    def test_multiscale_is_whole_frame_below_100_then_tiles(self):
        calls = []

        def model(x, sigma, **kwargs):
            calls.append(tuple(x.shape[-2:]))
            return torch.full_like(x, 2.0)

        tiled = TiledModelFn(
            model, tile_size=64, scale_schedule={0: 50, 2: 100})
        tiled.set_sigma_schedule(torch.tensor([10.0, 5.0, 1.0, 0.0]))
        x = torch.randn(1, 4, 128, 128)

        coarse = tiled(x, torch.tensor([10.0]))
        assert calls == [(64, 64)]  # one whole-frame evaluation, never four tiles
        assert coarse.shape == x.shape

        calls.clear()
        full = tiled(x, torch.tensor([1.0]))
        assert len(calls) == 4
        assert all(height < 128 and width < 128 for height, width in calls)
        assert full.shape == x.shape

    def test_multiscale_resizes_spatial_conditioning_and_restores_hints(self):
        class Model:
            pass

        sd_model = Model()
        original_hint = torch.randn(2, 3, 1024, 1024)
        sd_model._cn_hints = {'depth': original_hint}
        seen = {}

        def model(x, sigma, **kwargs):
            seen['x'] = x.shape[-2:]
            seen['image_cond'] = kwargs['image_cond'].shape[-2:]
            seen['hint'] = sd_model._cn_hints['depth'].shape[-2:]
            return x

        tiled = TiledModelFn(
            model, tile_size=64, sd_model=sd_model,
            scale_schedule={0: 50})
        tiled.set_sigma_schedule(torch.tensor([10.0, 0.0]))
        x = torch.randn(2, 4, 128, 128)
        image_cond = torch.randn(2, 3, 1024, 1024)

        result = tiled(x, torch.tensor([10.0]), image_cond=image_cond)

        assert seen == {
            'x': (64, 64),
            'image_cond': (512, 512),
            'hint': (512, 512),
        }
        assert result.shape == x.shape
        assert sd_model._cn_hints['depth'] is original_hint

    def test_multiscale_schedule_uses_sampling_steps_not_model_call_count(self):
        tiled = TiledModelFn(
            lambda x, sigma, **kwargs: x,
            scale_schedule={0: 50, 2: 75, 3: 100})
        tiled.set_sigma_schedule(torch.tensor([10.0, 6.0, 3.0, 1.0, 0.0]))

        assert tiled._scale_for_sigma(torch.tensor([10.0])) == 50
        assert tiled._scale_for_sigma(torch.tensor([4.0])) == 50
        assert tiled._scale_for_sigma(torch.tensor([3.0])) == 75
        assert tiled._scale_for_sigma(torch.tensor([1.0])) == 100

    def test_multiscale_works_with_tiling_disabled(self):
        calls = []

        def model(x, sigma, **kwargs):
            calls.append(tuple(x.shape[-2:]))
            return x

        wrapped = TiledModelFn(
            model, tile_size=32, scale_schedule={0: 50, 1: 100},
            enable_tiling=False)
        wrapped.set_sigma_schedule(torch.tensor([10.0, 1.0, 0.0]))
        x = torch.randn(1, 4, 128, 128)

        assert wrapped(x, torch.tensor([10.0])).shape == x.shape
        assert calls == [(64, 64)]
        calls.clear()
        assert wrapped(x, torch.tensor([1.0])).shape == x.shape
        assert calls == [(128, 128)]  # one normal full-frame call, never tiles

    def test_multiscale_pixel_floor_uses_longest_side(self):
        calls = []

        def model(x, sigma, **kwargs):
            calls.append(tuple(x.shape[-2:]))
            return x

        # 1920x1080 output -> 240x135 latent. A scheduled 10% pass would be
        # far smaller, but a 512px floor requires a 64-unit longest latent side.
        wrapped = TiledModelFn(
            model, scale_schedule={0: 10}, enable_tiling=False,
            min_scale_size=512)
        wrapped.set_sigma_schedule(torch.tensor([10.0, 0.0]))
        x = torch.randn(1, 4, 135, 240)

        assert wrapped(x, torch.tensor([10.0])).shape == x.shape
        assert calls == [(32, 64)]

    def test_multiscale_pixel_floor_never_upscales_past_final_size(self):
        wrapped = TiledModelFn(
            lambda x, sigma, **kwargs: x,
            scale_schedule={0: 25}, enable_tiling=False,
            min_scale_size=2048)
        assert wrapped._scaled_size(64, 64, 25) == (64, 64)


class TestTiledSamplerLogging:
    """The tiled sampler must say what it actually did.

    It runs per frame, so the log has to be emitted once per configuration rather than
    once per frame — and a 1x1 grid means the setting is doing nothing at all while still
    paying the crop/blend cost, which is worth shouting about.
    """

    def _cfg(self, enabled=True, tile=512, overlap=35.0,
             scale_schedule=None):
        from vibewarp.config import RunConfig
        config = RunConfig()
        config.diffusion.tiled_sampler = enabled
        config.diffusion.sampler_tile_size = tile
        config.diffusion.sampler_tile_overlap = overlap
        config.diffusion.sampler_scale_schedule = (
            {0: 50, 10: 100} if scale_schedule is None else scale_schedule)
        return config

    def _wrap(self):
        from types import SimpleNamespace
        return SimpleNamespace(inner_model=lambda *a, **k: None)

    def test_logs_the_resolved_tile_grid(self, capsys):
        from vibewarp.core.diffusion import _configure_tiled_sampler
        wrap = self._wrap()

        _configure_tiled_sampler(wrap, self._cfg(tile=512), None, H=1024, W=1024)

        out = capsys.readouterr().out
        assert 'Tiled sampler: ON' in out
        assert '2x2 tiles' in out          # 1024px frame / 512px tiles
        assert '1024x1024' in out
        assert '0:50%, 10:100%' in out
        assert 'below 100% = whole frame' in out
        assert wrap._tiled_inner_model is not None

    def test_logs_once_per_configuration_not_once_per_frame(self, capsys):
        from vibewarp.core.diffusion import _configure_tiled_sampler
        wrap = self._wrap()
        config = self._cfg()

        for _ in range(3):                                  # three frames
            _configure_tiled_sampler(wrap, config, None, H=1024, W=1024)
        assert capsys.readouterr().out.count('Tiled sampler: ON') == 1

        # ...but a changed setting IS worth saying again
        _configure_tiled_sampler(wrap, self._cfg(tile=256), None, H=1024, W=1024)
        assert 'Tiled sampler: ON' in capsys.readouterr().out

    def test_warns_when_a_single_tile_covers_the_frame(self, capsys):
        """One tile is not tiling: it changes nothing and still costs the blend."""
        from vibewarp.core.diffusion import _configure_tiled_sampler

        _configure_tiled_sampler(self._wrap(), self._cfg(tile=1024), None, H=512, W=512)

        out = capsys.readouterr().out
        assert '1x1 tiles' in out
        assert 'WARNING' in out and 'no effect' in out

    def test_disabling_it_is_reported_and_uninstalls(self, capsys):
        from vibewarp.core.diffusion import _configure_tiled_sampler
        wrap = self._wrap()
        _configure_tiled_sampler(wrap, self._cfg(), None, H=1024, W=1024)
        capsys.readouterr()

        _configure_tiled_sampler(
            wrap, self._cfg(enabled=False, scale_schedule={0: 100}),
            None, H=1024, W=1024)

        assert 'sampler: disabled' in capsys.readouterr().out
        assert wrap._tiled_inner_model is None

    def test_multiscale_installs_when_tiling_is_off(self, capsys):
        from vibewarp.core.diffusion import _configure_tiled_sampler
        wrap = self._wrap()

        _configure_tiled_sampler(
            wrap, self._cfg(enabled=False, scale_schedule={0: 50, 10: 100}),
            None, H=1024, W=1024)

        out = capsys.readouterr().out
        assert 'Multiscale sampler: ON' in out
        assert 'tiling off' in out
        assert wrap._tiled_inner_model is not None
        assert wrap._tiled_inner_model.enable_tiling is False


class TestStepProgress:
    def test_every_selectable_sampler_accepts_a_callback(self):
        """The step bar is driven by the sampler's callback; one without it would crash."""
        import inspect
        from vibewarp.config_io import FIELD_CHOICES
        from vibewarp.core.diffusion import get_sampler_fn

        for name in FIELD_CHOICES['DiffusionConfig.sampler']:
            params = inspect.signature(get_sampler_fn(name)).parameters
            assert 'callback' in params, f'{name} cannot report step progress'

    def test_callback_returns_the_latent_unchanged(self):
        """REGRESSION: in this fork the callback's return value REPLACES x.

        Every sampler does `x = callback({...}).float()` — that is how WarpFusion
        implements masked diffusion. A progress callback that returned None nulled the
        latent, and the next step died with "'NoneType' object has no attribute 'float'".
        Standard k-diffusion ignores the return value, which is why this is easy to miss.
        """
        import torch
        from vibewarp.core.diffusion import _step_progress

        x = torch.randn(1, 4, 8, 8)
        with _step_progress(4, 'steps') as on_step:
            out = on_step({'i': 0, 'x': x, 'sigma': 14.6})
        assert out is x, 'the callback must hand the latent back, or it nulls x'
        assert out.float() is not None      # what the sampler does next

    def test_callback_survives_odd_payloads(self):
        import torch
        from vibewarp.core.diffusion import _step_progress

        x = torch.zeros(1, 4, 8, 8)
        with _step_progress(4, 'steps') as on_step:
            assert on_step({'i': 1, 'x': x, 'sigma': float('nan')}) is x
            assert on_step({'i': 2, 'x': x}) is x          # no sigma
            assert on_step('not a dict') == 'not a dict'   # forks vary; must not raise


class TestStepProgressAgainstRealSamplers:
    """Run the progress callback through the ACTUAL samplers.

    The unit tests above mock the sampler, so they never exercised the fork's
    `x = callback(...).float()` contract — the render blew up on the first step with
    "'NoneType' object has no attribute 'float'" while the suite stayed green.
    """

    def test_every_selectable_sampler_runs_with_the_progress_callback(self):
        import torch
        from vibewarp.config_io import FIELD_CHOICES
        from vibewarp.core.diffusion import _step_progress, get_sampler_fn

        def model(x, sigma, **kwargs):        # a denoiser that just shrinks the latent
            return x * 0.9

        sigmas = torch.linspace(3.0, 0.0, 5)
        for name in FIELD_CHOICES['DiffusionConfig.sampler']:
            sampler = get_sampler_fn(name)
            x = torch.randn(1, 4, 8, 8)
            with _step_progress(len(sigmas) - 1, name) as on_step:
                out = sampler(model, x, sigmas, extra_args={}, callback=on_step)
            assert out is not None, f'{name}: sampler returned None'
            assert torch.isfinite(out).all(), f'{name}: produced non-finite latents'
