import math
import pytest
import torch
from PIL import Image

from vibewarp.config import RunConfig
from vibewarp.config_io import config_schema,validate_config
from vibewarp.core.noise_variation import vary_fixed_noise,wiggle_offsets


def test_wiggle_is_bounded_wrap_translation_and_preserves_every_noise_value():
    base=torch.randn(1,3,31,47,generator=torch.Generator().manual_seed(3));original=base.clone()
    state=torch.get_rng_state().clone()
    shifted=vary_fixed_noise(base,0,17,4,wiggle=3)
    dy,dx=wiggle_offsets(17,4,3)
    assert -3<=dy<=3 and -3<=dx<=3
    assert torch.equal(shifted[...,dy%31,dx%47],base[...,0,0])
    assert torch.equal(shifted[...,(-1+dy)%31,(-1+dx)%47],base[...,-1,-1])
    assert torch.equal(shifted.flatten().sort().values,base.flatten().sort().values)
    torch.testing.assert_close(shifted.mean(),base.mean(),atol=1e-7,rtol=0)
    torch.testing.assert_close(shifted.std(),base.std(),atol=1e-7,rtol=0)
    assert torch.equal(shifted,vary_fixed_noise(base,0,17,4,wiggle=3))
    assert len({wiggle_offsets(17,frame,3) for frame in range(1,20)})>1
    assert torch.equal(base,original) and torch.equal(state,torch.get_rng_state())
    assert vary_fixed_noise(base,0,17,0,wiggle=3) is base
    assert vary_fixed_noise(base,0,17,4,wiggle=0) is base


def test_wiggle_composes_with_randomness_without_changing_fresh_rng():
    base=torch.randn(1,4,32,32)
    shifted=vary_fixed_noise(base,0,9,3,wiggle=2)
    fresh=vary_fixed_noise(base,1,9,3)
    actual=vary_fixed_noise(base,.25,9,3,wiggle=2)
    torch.testing.assert_close(actual,math.sqrt(.75)*shifted+math.sqrt(.25)*fresh)
    assert torch.equal(vary_fixed_noise(base,1,9,3,wiggle=2),fresh)


@pytest.mark.parametrize('direction,expected',[
    ([1,0],(0,6)), ([-1,0],(0,-6)),([0,1],(6,0)),([0,-1],(-6,0)),
    ([2,0],(0,6)),([1,1],(4,4)),([.001,.001],(4,4))])
def test_direction_rolls_cumulatively_at_normalized_speed(direction,expected):
    assert wiggle_offsets(42,3,2,direction)==expected
    assert wiggle_offsets(99,3,2,direction)==expected
    base=torch.arange(3*7*11).reshape(1,3,7,11).float()
    dy,dx=expected
    result=vary_fixed_noise(base,0,42,3,wiggle=2,direction=direction)
    assert torch.equal(result,torch.roll(base,(dy,dx),(-2,-1)))
    assert torch.equal(result,vary_fixed_noise(base,0,42,3,wiggle=2,direction=direction))
    assert torch.equal(result.flatten().sort().values,base.flatten().sort().values)


def test_zero_shift_is_off_for_every_direction_and_zero_direction_keeps_random_mode():
    base=torch.randn(1,3,8,8)
    assert vary_fixed_noise(base,0,9,4,wiggle=0,direction=[1,1]) is base
    assert vary_fixed_noise(base,0,9,0,wiggle=3,direction=[1,1]) is base
    assert wiggle_offsets(9,4,3,[0,0])==wiggle_offsets(9,4,3)


@pytest.mark.parametrize('direction',[[1],[1,2,3],[float('nan'),0],[0,float('inf')],[True,0]])
def test_invalid_wiggle_direction_is_rejected(direction):
    c=RunConfig();c.diffusion.fixed_seed_wiggle_direction=direction
    assert any(e['path']=='diffusion.fixed_seed_wiggle_direction' for e in validate_config(c,require_inputs=False,check_paths=False))
    with pytest.raises(ValueError):wiggle_offsets(1,1,1,direction)


@pytest.mark.parametrize('value',[-1,0.5,True])
def test_wiggle_validation(value):
    c=RunConfig();c.diffusion.fixed_seed_wiggle=value
    assert any(e['path']=='diffusion.fixed_seed_wiggle' for e in validate_config(c,require_inputs=False,check_paths=False))
    with pytest.raises(ValueError):vary_fixed_noise(torch.zeros(1,1,8,8),0,1,1,wiggle=value)


@pytest.mark.parametrize('strength',[0.,0.1,0.5,0.9,1.])
def test_fixed_noise_keeps_gaussian_statistics(strength):
    base=torch.randn(500_000,generator=torch.Generator().manual_seed(91))
    result=vary_fixed_noise(base,strength,91,7)
    assert abs(result.mean().item())<0.008
    assert abs(result.std().item()-1)<0.008
    # Fourth moment distinguishes the Gaussian law from simple norm matching.
    assert abs(result.pow(4).mean().item()-3)<0.08
    if strength==0:assert result is base


def test_fixed_noise_endpoints_variance_weights_and_resume():
    base=torch.randn(10000,generator=torch.Generator().manual_seed(1))
    state=torch.get_rng_state().clone()
    fresh=vary_fixed_noise(base,1.,17,9)
    mixed=vary_fixed_noise(base,0.3,17,9)
    torch.testing.assert_close(mixed,math.sqrt(.7)*base+math.sqrt(.3)*fresh)
    assert torch.equal(mixed,vary_fixed_noise(base,.3,17,9))
    assert not torch.equal(mixed,vary_fixed_noise(base,.3,17,10))
    assert not torch.equal(mixed,vary_fixed_noise(base,.3,18,9))
    assert torch.equal(state,torch.get_rng_state())
    assert vary_fixed_noise(base,1.,17,0) is base
    assert torch.equal(base,torch.randn(10000,generator=torch.Generator().manual_seed(1)))


@pytest.mark.parametrize('value',[-.01,1.01,float('nan'),float('inf')])
def test_fixed_randomness_validation(value):
    c=RunConfig();c.diffusion.fixed_seed_randomness=value
    assert any(e['path']=='diffusion.fixed_seed_randomness' for e in validate_config(c,require_inputs=False,check_paths=False))
    with pytest.raises(ValueError):vary_fixed_noise(torch.zeros(1),value,1,1)


def test_fixed_randomness_schema_is_native_seed_control():
    field=config_schema()['properties']['diffusion']['properties']['fixed_seed_randomness']
    assert field['group']=='Seed' and field['workflows']==['prompts']
    assert field['model_families']==['sd','disco']
    assert field['default']==0
    assert field['slider']=={'min':0,'max':1,'step':0.01,'scale':1}
    wiggle=config_schema()['properties']['diffusion']['properties']['fixed_seed_wiggle']
    assert wiggle['group']=='Seed' and wiggle['workflows']==['prompts']
    assert wiggle['default']==0 and wiggle['model_families']==['sd','disco']
    direction=config_schema()['properties']['diffusion']['properties']['fixed_seed_wiggle_direction']
    assert direction['group']=='Seed' and direction['vector_labels']==['X','Y']


def test_noise_randomness_settings_round_trip(tmp_path):
    import json
    from vibewarp.config_io import config_from_settings
    from vibewarp.settings import settings_to_dict
    c=RunConfig(model_version='disco_512');c.disco.fixed_seed=True
    c.diffusion.seed=170;c.diffusion.fixed_seed_randomness=.37;c.diffusion.fixed_seed_wiggle=3
    c.diffusion.fixed_seed_wiggle_direction=[-1,0.5]
    path=tmp_path/'noise.txt';path.write_text(json.dumps(settings_to_dict(c)))
    restored=config_from_settings(path)
    assert restored.diffusion.fixed_seed_randomness==.37
    assert restored.diffusion.fixed_seed_wiggle==3
    assert restored.diffusion.fixed_seed_wiggle_direction==[-1,0.5]
    assert restored.disco.fixed_seed and restored.diffusion.seed==170


def test_disco_sampler_receives_reproducible_mixed_initial_noise():
    from vibewarp.core.disco import DiscoBackend
    from vibewarp.core.disco_lora import guided_diffusion
    c=RunConfig(model_version='disco_512');d=c.disco
    d.steps=2;d.timestep_grid=[999,499,-1];d.style_strength=1;d.fixed_seed=True
    d.clip_guidance_scale=d.init_scale=d.image_scale=d.range_scale=0
    backend=DiscoBackend.__new__(DiscoBackend)
    backend.config=c;backend.device='cpu';backend.model=torch.nn.Identity()
    backend.secondary=None;backend.encoders=[];backend.lpips=None
    backend.diffusion=guided_diffusion(d.timestep_grid)
    calls=[]
    def sample(*args,**kwargs):
        noise=kwargs['noise'];calls.append(noise.clone())
        for _ in range(2):yield {'sample':noise,'pred_xstart':noise}
    backend.diffusion.ddim_sample_loop_progressive=sample
    def draw(frame,strength,fixed=True,wiggle=0,direction=None):
        c.diffusion.fixed_seed_randomness=strength;d.fixed_seed=fixed
        c.diffusion.fixed_seed_wiggle=wiggle
        c.diffusion.fixed_seed_wiggle_direction=direction or [0,0]
        backend.render(c,Image.new('RGB',(64,64)),'','',123,frame_num=frame)
        return calls[-1]
    original=draw(1,0)
    mixed=draw(1,.4)
    torch.testing.assert_close(mixed,vary_fixed_noise(original,.4,123,1))
    assert torch.equal(mixed,draw(1,.4))
    assert not torch.equal(mixed,draw(2,.4))
    assert torch.equal(original,draw(1,.4,False))
    assert torch.equal(draw(0,0),draw(0,1))
    shifted=draw(1,0,wiggle=3)
    assert torch.equal(shifted,vary_fixed_noise(original,0,123,1,wiggle=3))
    assert torch.equal(shifted,draw(1,0,wiggle=3))
    assert torch.equal(original,draw(1,0,False,wiggle=3))
    assert torch.equal(draw(4,0,wiggle=2,direction=[1,0]),torch.roll(original,(0,8),(-2,-1)))


def test_unguided_distilled_render_passes_variation_to_noise_sampler(monkeypatch):
    from vibewarp.config import DISCO_DISTILLED_PRESET
    from vibewarp.config_io import config_from_dict
    from vibewarp.core.disco_lora import DistilledDiscoBackend
    c=config_from_dict(DISCO_DISTILLED_PRESET);c.disco.fixed_seed=True
    c.diffusion.fixed_seed_randomness=.6;c.diffusion.fixed_seed_wiggle=2
    c.diffusion.fixed_seed_wiggle_direction=[0,1]
    backend=DistilledDiscoBackend.__new__(DistilledDiscoBackend);seen=[]
    monkeypatch.setattr(backend,'sample_tensor',lambda seed,**kw:seen.append(kw) or torch.zeros(1,3,512,512))
    backend.render(c,Image.new('RGB',(512,512)),'','',7,frame_num=4)
    assert seen[-1]['randomness']==.6 and seen[-1]['frame_num']==4
    assert seen[-1]['wiggle']==2
    assert seen[-1]['direction']==[0,1]
