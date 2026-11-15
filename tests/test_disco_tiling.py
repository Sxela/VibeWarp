from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vibewarp.config import RunConfig
from vibewarp.config_io import config_from_dict, validate_config
from vibewarp.core.disco_tiling import prepare_disco_denoisers
from vibewarp.core.sampling import build_tile_specs
from vibewarp.vendor.guided_diffusion.script_util import create_gaussian_diffusion


@pytest.fixture
def setup():
    config = RunConfig(model_version='disco_512')
    config.diffusion.tiled_sampler = True
    config.diffusion.sampler_tile_size = 128
    diffusion = create_gaussian_diffusion(steps=1000, learn_sigma=True,
        timestep_respacing='ddim4', rescale_timesteps=True)
    return config, diffusion


def test_aligned_overlap_geometry_covers_rectangular_frame():
    specs = build_tile_specs(1024, 1536, 2, 3, 35, 35, alignment=64)
    weights = torch.zeros(1024, 1536)
    for top, bottom, left, right, window in specs:
        assert all(value % 64 == 0 for value in (top, bottom, left, right))
        weights[top:bottom, left:right] += window
    assert len(specs) == 6 and (weights > 0).all()
    assert (weights > 1).any()  # overlapping predictions are normalized


def test_tiled_epsilon_and_variance_outputs_and_input_gradients(setup):
    config, diffusion = setup
    seen = []
    def model(x, t):
        seen.append(x.shape[-2:])
        return torch.cat((x.sin(), x * .25 + .1), dim=1)
    wrapped, _ = prepare_disco_denoisers(model, None, diffusion, config, 0)
    x = torch.randn(2, 3, 256, 384, requires_grad=True)
    expected, actual = model(x, None), wrapped(x, torch.tensor([750., 750.]))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.square().mean(), x)[0],
                               torch.autograd.grad(expected.square().mean(), x)[0])
    assert len(seen) > 2
    assert all(h % 64 == 0 and w % 64 == 0 for h, w in seen)
    assert all(h < 256 and w < 384 for h, w in seen[1:7])


def test_coarse_epsilon_restores_clean_prediction_not_averaged_noise(setup):
    config, diffusion = setup
    config.diffusion.sampler_scale_schedule = {0: 50, 1: 100}
    alpha = float(diffusion.sqrt_alphas_cumprod[-1])
    sigma = float(diffusion.sqrt_one_minus_alphas_cumprod[-1])
    seen = []
    def model(x, t):
        seen.append(x.detach().clone())
        return torch.cat(((x - alpha * .25) / sigma, torch.full_like(x, .4)), dim=1)
    wrapped, _ = prepare_disco_denoisers(model, None, diffusion, config, 0)
    x = torch.randn(1, 3, 256, 384, requires_grad=True)
    prediction = wrapped(x, torch.tensor([750.]))
    assert len(seen) == 1
    torch.testing.assert_close(seen[0], F.interpolate(x, (128, 192), mode='nearest-exact'))
    epsilon, variance = prediction.chunk(2, dim=1)
    torch.testing.assert_close((x - sigma * epsilon) / alpha, torch.full_like(x, .25), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(variance, torch.full_like(x, .4))
    assert torch.isfinite(torch.autograd.grad(prediction.mean(), x)[0]).all()


def test_secondary_is_tiled_and_schedule_uses_timestep_after_skip(setup):
    config, diffusion = setup
    config.diffusion.sampler_scale_schedule = {0: 50, 1: 100}
    calls = []
    def secondary(x, t):
        calls.append((x.shape[-2:], t.clone()))
        return SimpleNamespace(pred=x.square() * .2)
    _, preview = prepare_disco_denoisers(None, secondary, diffusion, config, skip=1)
    x = torch.randn(1, 3, 256, 256, requires_grad=True)
    coarse = preview(x, torch.tensor([500.]))
    assert [shape for shape, _ in calls] == [(128, 128)]
    assert coarse.shape == x.shape
    calls.clear()
    fine = preview(x, torch.tensor([250.]))
    assert len(calls) == 4
    assert torch.isfinite(torch.autograd.grad(fine.sum(), x)[0]).all()
    # PLMS may repeat or evaluate the next timestep: calls cannot advance the schedule.
    assert preview._scale_for_sigma(torch.tensor([500.])) == 50
    assert preview._scale_for_sigma(torch.tensor([250.])) == 100
    assert preview._scale_for_sigma(torch.tensor([0.])) == 100


def test_disabled_mode_returns_original_model(setup):
    config, diffusion = setup
    config.diffusion.tiled_sampler = False
    model = object()
    actual, preview = prepare_disco_denoisers(model, None, diffusion, config, 0)
    assert actual is model and preview is None


@pytest.mark.parametrize('settings,path', [
    ({'sampler_tile_size': 96}, 'diffusion.sampler_tile_size'),
    ({'sampler_scale_schedule': {1: 50}}, 'diffusion.sampler_scale_schedule'),
    ({'sampler_scale_schedule': {0: float('nan')}}, 'diffusion.sampler_scale_schedule'),
    ({'sampler_tile_overlap': 75}, 'diffusion.sampler_tile_overlap'),
    ({'sampler_tile_split_threshold': -1}, 'diffusion.sampler_tile_split_threshold'),
])
def test_pixel_tiling_validation(settings, path):
    config = config_from_dict({'model_version': 'disco_512', 'diffusion': settings})
    assert any(error['path'] == path for error in validate_config(config, require_inputs=False))


def test_tiling_controls_are_visible_for_disco_and_round_trip():
    from vibewarp.config_io import config_to_dict
    from vibewarp.ui_layout import model_families
    original = config_from_dict({'model_version': 'disco_512', 'diffusion': {
        'tiled_sampler': True, 'sampler_tile_size': 768, 'sampler_scale_schedule': {0: 25, 5: 100}}})
    restored = config_from_dict(config_to_dict(original))
    assert restored.diffusion.tiled_sampler
    assert restored.diffusion.sampler_scale_schedule == {0: 25, 5: 100}
    for name in ('tiled_sampler', 'sampler_tile_size', 'sampler_scale_schedule', 'sampler_scale_min_size',
                 'sampler_tile_overlap', 'sampler_tile_split_threshold'):
        assert 'disco' in model_families('diffusion', name)
