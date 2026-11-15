import pytest
import torch
import torch.nn.functional as F

from vibewarp.config import RunConfig
from vibewarp.core.disco_progressive import progressive_sizes, progressive_disco_samples, refine_noise


@pytest.mark.parametrize('new_size', [(64, 96), (73, 101)])
def test_refinement_preserves_coarse_noise_and_unit_variance(new_size):
    torch.manual_seed(42)
    coarse = torch.randn(32, 3, 32, 48)
    fine = refine_noise(coarse, new_size)
    ids = torch.arange(32*48).float().reshape(1, 1, 32, 48)
    ids = F.interpolate(ids, new_size, mode='nearest-exact').long().flatten()
    counts = torch.bincount(ids).float()
    sums = torch.zeros_like(coarse.flatten(2)).scatter_add_(2, ids.expand(32, 3, -1), fine.flatten(2))
    torch.testing.assert_close(sums/counts.sqrt(), coarse.flatten(2), atol=2e-6, rtol=2e-5)
    assert abs(float(fine.mean())) < .01
    assert .99 < float(fine.var()) < 1.01


@pytest.mark.parametrize('sampler', ['ddim', 'plms'])
def test_state_changes_size_only_at_boundaries_and_plms_restarts(sampler):
    seen, history = [], []
    class Diffusion:
        num_timesteps = 4
        sqrt_alphas_cumprod = [.9]*4
        sqrt_one_minus_alphas_cumprod = [.4]*4
        def q_sample(self, init, t, noise):
            return .9*init + .4*noise
        def step(self, model, x, t, **kwargs):
            seen.append((int(t[0]), x.shape[-2:]))
            history.append(kwargs.get('old_out'))
            return {'sample': x, 'pred_xstart': torch.zeros_like(x), 'old_eps': [torch.zeros_like(x)]}
        ddim_sample = step
        plms_sample = step
    sizes = [(64,64), (64,64), (128,128), (128,128)]
    init = torch.zeros(1,3,128,128)
    torch.manual_seed(3)
    out = list(progressive_disco_samples(Diffusion(), None, init, torch.randn_like(init), sizes,
               skip=0, sampler=sampler, cond_fn=None))
    assert seen == [(3,(64,64)), (2,(64,64)), (1,(128,128)), (0,(128,128))]
    assert [item['sample'].shape[-2:] for item in out] == sizes
    if sampler == 'plms':
        assert history[0] is None and history[1] is not None
        assert history[2] is None and history[3] is not None


def test_progressive_plan_resolves_actual_steps_and_rejects_unfinished_schedule():
    c = RunConfig()
    c.diffusion.sampler_scale_schedule = {0:25, 2:50, 4:100}
    assert progressive_sizes(c.diffusion,(2048,2048),6) == [(512,512)]*2 + [(1024,1024)]*2 + [(2048,2048)]*2
    with pytest.raises(ValueError, match='reach full size'):
        progressive_sizes(c.diffusion,(2048,2048),4)
    c.diffusion.sampler_scale_schedule = {0:100, 2:50, 4:100}
    with pytest.raises(ValueError, match='not decrease'):
        progressive_sizes(c.diffusion,(2048,2048),6)


@pytest.mark.parametrize('schedule', [{0:100, 2:50, 4:100}, {0:25, 40:100}])
def test_validation_rejects_decreasing_or_unfinished_disco_scales(schedule):
    from vibewarp.config_io import validate_config
    c = RunConfig(model_version='disco_512')
    c.diffusion.sampler_scale_schedule = schedule
    assert any(error['path'] == 'diffusion.sampler_scale_schedule'
               for error in validate_config(c, require_inputs=False))


def test_cutout_windows_keep_relative_footprint_across_scales():
    from vibewarp.core.disco import make_cutouts
    # A normalized coordinate ramp reveals crop field of view independently
    # of its encoder resize. At half resolution, use half-size crop windows.
    full = torch.linspace(0,1,512)[None,None,None,:].expand(1,3,512,512)
    half = F.interpolate(full,(256,256),mode='bilinear',align_corners=False)
    options = dict(size=64, overview=1, inner=4, power=1., gray_probability=0., augment=False)
    a = make_cutouts(full, generator=torch.Generator().manual_seed(4), **options)
    b = make_cutouts(half, crop_scale=.5, generator=torch.Generator().manual_seed(4), **options)
    # randint positions depend on integer bounds; compare the window's span,
    # which the coordinate ramp exposes regardless of its random location.
    # Exclude ResizeRight's constant-padded filter boundary/ringing: extrema
    # there depend on crop position rather than just the sampled field of view.
    span = lambda image: image[...,32,-5] - image[...,32,4]
    torch.testing.assert_close(span(a),span(b),atol=.005,rtol=.01)
