import pytest
import torch

from vibewarp.config import RunConfig
from vibewarp.config_io import validate_config,config_schema
from vibewarp.core.disco_guidance import cap_guidance_update,guidance_noise_cap
from vibewarp.vendor.guided_diffusion.script_util import create_gaussian_diffusion


def test_noise_cap_per_image_rms_and_inactive_values_are_exact():
    baseline = torch.full((2,3,4,4),.25)
    guided = baseline+torch.tensor([.1,10.])[:,None,None,None]
    sigma = torch.tensor([.5,.2])[:,None,None,None]
    stats = {}
    actual = cap_guidance_update(guided,baseline,sigma,50,stats)
    assert torch.equal(actual[0],guided[0])
    rms = (actual-baseline).square().mean((1,2,3)).sqrt()
    torch.testing.assert_close(rms,torch.tensor([.1,.1]))
    assert stats['guidance_cap_scale'] < 1
    assert cap_guidance_update(guided,baseline,sigma,0) is guided
    assert torch.equal(cap_guidance_update(guided,baseline,0.,100),baseline)


class Denoiser(torch.nn.Module):
    def forward(self,x,t,**kwargs):
        return torch.cat((x.tanh()*.1,torch.zeros_like(x)),dim=1)


def diffusion():
    return create_gaussian_diffusion(steps=1000,learn_sigma=True,
        timestep_respacing='ddim4',rescale_timesteps=True)


@pytest.mark.parametrize('sampler,eta',[('ddim',0.),('ddim',.7),('plms',0.)])
def test_cap_controls_actual_update_including_history_and_final_step(sampler,eta):
    d = diffusion()
    seen = []
    def cap(guided,baseline,sigma):
        result = cap_guidance_update(guided,baseline,sigma,10)
        rms = (result-baseline).square().mean((1,2,3)).sqrt()
        budget = sigma[:,0,0,0]*.1
        assert torch.all(rms <= budget+2e-5)
        seen.append(float(budget[0]))
        return result
    d.guidance_update_cap = cap
    sample = getattr(d,f'{sampler}_sample_loop_progressive')
    kwargs = {'eta':eta} if sampler == 'ddim' else {}
    outputs = list(sample(Denoiser(),(1,3,8,8),noise=torch.ones(1,3,8,8),device='cpu',
        clip_denoised=False,cond_fn=lambda x,t,**kw:torch.ones_like(x)*1000,model_kwargs={},**kwargs))
    assert len(seen) == 4 and seen[-1] < seen[0]
    assert all(torch.isfinite(item['sample']).all() for item in outputs)


@pytest.mark.parametrize('sampler,eta',[('ddim',0.),('ddim',.7),('plms',0.)])
def test_counterfactual_baseline_matches_unguided_rollout(sampler,eta):
    d = diffusion()
    sample = getattr(d,f'{sampler}_sample_loop_progressive')
    kwargs = {'eta':eta} if sampler == 'ddim' else {}
    inputs = dict(noise=torch.ones(1,3,8,8),device='cpu',clip_denoised=False,model_kwargs={},**kwargs)
    torch.manual_seed(123)
    reference = list(sample(Denoiser(),(1,3,8,8),**inputs))
    # Forcing the cap to return its baseline removes all explicit guidance,
    # including PLMS bootstrap and the historical Adams-Bashforth contribution.
    d.guidance_update_cap = lambda guided,baseline,sigma:baseline
    torch.manual_seed(123)
    actual = list(sample(Denoiser(),(1,3,8,8),cond_fn=lambda x,t,**kw:torch.ones_like(x)*1000,**inputs))
    for a,b in zip(actual,reference):
        torch.testing.assert_close(a['sample'],b['sample'],rtol=0,atol=0)


def test_hook_restored_on_failure_and_disabled_scope_is_exact():
    d = diffusion()
    original = lambda *args:None
    d.guidance_update_cap = original
    with pytest.raises(RuntimeError):
        with guidance_noise_cap(d,100):
            assert d.guidance_update_cap is not original
            raise RuntimeError('cancel')
    assert d.guidance_update_cap is original
    with guidance_noise_cap(d,0):
        assert d.guidance_update_cap is None
    assert d.guidance_update_cap is original


@pytest.mark.parametrize('value',[-1,float('nan'),float('inf')])
def test_invalid_noise_cap_rejected(value):
    config = RunConfig(model_version='disco_512')
    config.disco.guidance_noise_cap_percent = value
    assert any(error['path']=='disco.guidance_noise_cap_percent' for error in validate_config(config,require_inputs=False))


def test_noise_cap_schema_default_and_normal_run_validation():
    config = RunConfig(model_version='disco_512')
    assert config.disco.guidance_noise_cap_percent == 0
    config.disco.guidance_noise_cap_percent = 100
    assert not validate_config(config,require_inputs=False)
    field = config_schema()['properties']['disco']['properties']['guidance_noise_cap_percent']
    assert field['default'] == 0 and 'noise RMS' in field['label']
