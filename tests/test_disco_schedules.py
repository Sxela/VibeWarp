from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from vibewarp.config import RunConfig
from vibewarp.config_io import config_from_dict, config_schema, validate_config
from vibewarp.core.diffusion import get_frame_schedule
from vibewarp.core.disco import DiscoBackend, frame_guidance_loss, render_disco_frame
from vibewarp.render_events import render_event_scope
from vibewarp.vendor.guided_diffusion.script_util import create_gaussian_diffusion


def test_disco_schedule_resolution_interpolation_and_relative_frames():
    config=RunConfig(model_version='disco_512')
    config.disco.steps_schedule={'0':20,'10':40}
    config.disco.style_strength_schedule=[.5,.7,1.]
    assert get_frame_schedule(5,config)['steps']==30
    assert get_frame_schedule(1,config)['style_strength']==.7
    assert get_frame_schedule(20,config)['style_strength']==1
    config.keyframes_relative_to_frame_range=True
    config.frame_range=[100,150]
    assert get_frame_schedule(105,config)['steps']==30
    assert get_frame_schedule(101,config)['style_strength']==.7
    restored=config_from_dict(asdict(config))
    assert restored.disco.steps_schedule==config.disco.steps_schedule


@pytest.mark.parametrize('name,value',[
    ('steps_schedule',[]),('steps_schedule',{}),('steps_schedule',[1]),
    ('steps_schedule',{'-1':20}),('steps_schedule',{'bad':20}),
    ('steps_schedule',[float('nan')]),('steps_schedule',[[20,30]]),
    ('style_strength_schedule',[0]),('style_strength_schedule',[1.01]),
    ('style_strength_schedule',[float('inf')]),
])
def test_invalid_disco_schedules_rejected(name,value):
    config=RunConfig(model_version='disco_512')
    setattr(config.disco,name,value)
    assert any(e['path']==f'disco.{name}' for e in validate_config(config,require_inputs=False))


def test_progressive_validation_uses_scheduled_sampling_budget():
    config=RunConfig(model_version='disco_512')
    config.diffusion.sampler_scale_schedule={0:50,10:100}
    config.diffusion.sampler_scale_min_size=64
    config.disco.steps_schedule={0:50,10:10}
    config.disco.style_strength_schedule=[1.]
    assert any(e['path']=='diffusion.sampler_scale_schedule' for e in validate_config(config,require_inputs=False))
    config.disco.steps=4  # ignored when a schedule is provided
    config.disco.steps_schedule=[30]
    assert not validate_config(config,require_inputs=False)


def test_schedule_fields_exposed_in_render_schema():
    fields=config_schema()['properties']['disco']['properties']
    assert fields['mask_frame_guidance']['default'] is False
    assert fields['mask_frame_guidance']['tier']=='render'
    assert fields['mask_frame_guidance']['label']=='Mask frame guidance to consistent areas'
    for name in ('steps_schedule','style_strength_schedule'):
        assert fields[name]['tier']=='render'
        assert fields[name]['model_families']==['disco']


def test_frame_guidance_schema_has_one_visible_editor_per_loss():
    props=config_schema()['properties']['disco']['properties']
    for name,label in [('init_scale','LPIPS'),('image_scale','Image-space')]:
        assert props[name]['tier']=='hidden'
        assert 'workflows' not in props[name]
        schedule=props[name+'_schedule']
        assert schedule['label'].startswith(label+' frame guidance scale')
        assert schedule['schedule_fallback']==name
        assert schedule['schedule_domain']=='frame'
        assert schedule['schedule_interpolation']=='hold'
        assert schedule['workflows']==['clip']
    assert props['mask_frame_guidance']['group']=='General'
    assert props['mask_frame_guidance']['workspace_order']==props['image_scale_schedule']['workspace_order']+1
    assert props['use_secondary_model']['tier']!='hidden'


@pytest.mark.parametrize('scalar,schedule,frame,expected',[
    (1000,None,5,1000), (0,None,5,0), (1000,{0:0},5,0),
    (0,{0:0,1:50000},0,0), (0,{0:0,1:50000},1,50000),
    (0,{0:0,10:50000},5,0),
])
def test_frame_guidance_keeps_schedule_precedence_and_hold_with_global_blending(
        monkeypatch,scalar,schedule,frame,expected):
    class Denoiser(torch.nn.Module):
        def forward(self,x,t,**kwargs):
            return torch.cat([x*.1,torch.zeros_like(x)],dim=1)
    backend=DiscoBackend.__new__(DiscoBackend)
    backend.model,backend.secondary=Denoiser(),None
    backend.diffusion=create_gaussian_diffusion(steps=1000,learn_sigma=True,
        timestep_respacing='ddim4',rescale_timesteps=True)
    backend.encoders,backend.device,backend.lpips=[],'cpu',SimpleNamespace(spatial=False)
    config=RunConfig(model_version='disco_512')
    config.blend_json_schedules=True
    config.disco.sampler='ddim'
    config.disco.steps=4
    config.disco.style_strength=1
    config.disco.init_scale=config.disco.image_scale=scalar
    config.disco.init_scale_schedule=config.disco.image_scale_schedule=schedule
    strengths=[]
    def guidance(image,target,**kwargs):
        strengths.append((kwargs['lpips_scale'],kwargs['image_scale']))
        return image.sum()*0
    monkeypatch.setattr('vibewarp.core.disco.frame_guidance_loss',guidance)
    backend.render(config,Image.new('RGB',(64,64)),'','',42,frame_num=frame)
    assert strengths and all(pair==(expected,expected) for pair in strengths)


def test_schedules_survive_saved_settings_roundtrip(tmp_path):
    from vibewarp.settings import settings_to_dict,save_settings
    from vibewarp.config_io import config_from_settings
    config=RunConfig(model_version='disco_512')
    config.disco.steps_schedule={0:30,20:50}
    config.disco.mask_frame_guidance=True
    config.disco.lora_strength=.4
    config.disco.style_strength_schedule=[.65,.8,1.]
    path=save_settings(settings_to_dict(config),str(tmp_path),'schedules',0)
    restored=config_from_settings(path)
    assert get_frame_schedule(10,restored)['steps']==40
    assert restored.disco.style_strength_schedule==[.65,.8,1.]
    assert restored.disco.mask_frame_guidance is True
    assert restored.disco.lora_strength==.4


def test_notebook_lpips_compares_full_image_to_init_without_consistency_mask():
    image=torch.full((1,3,4,4),.2,requires_grad=True)
    init=torch.linspace(-.5,.5,48).reshape_as(image)
    lpips=lambda a,b:(a-b).square().mean()
    actual=frame_guidance_loss(image,init,lpips_model=lpips,lpips_scale=1000)
    expected=lpips(image,init).sum()*1000
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual,image)[0],
        torch.autograd.grad(expected,image)[0],rtol=0,atol=0)


def test_default_guidance_tracks_prepared_init_including_background_mask(tmp_path):
    config=RunConfig(model_version='disco_512')
    config.video.width=config.video.height=64
    prepared=tmp_path/'masked_init.png'
    warped=tmp_path/'warped.png'
    Image.new('RGB',(64,64),'blue').save(prepared)
    Image.new('RGB',(64,64),'red').save(warped)
    captured={}
    class Backend:
        def render(self,config,init,*args,**kwargs):
            captured.update(kwargs)
            captured['init']=init
            return init
    ctx=SimpleNamespace(config=config,video_frames_folder=str(tmp_path),batch_folder=str(tmp_path),disco_backend=Backend())
    from vibewarp.core.diffusion import FrameState
    render_disco_frame(ctx,FrameState(frame_num=1,init_image=str(prepared),guidance_image=str(warped)))
    assert captured['guidance_image'] is captured['init']
    assert captured['mask_guidance'] is False


@pytest.mark.parametrize('sampler',['ddim','plms'])
def test_scheduled_sampler_respacing_changes_budget_without_reloading_model(sampler):
    class Denoiser(torch.nn.Module):
        def forward(self,x,t,**kwargs):
            return torch.cat([x*.1,torch.zeros_like(x)],dim=1)
    backend=DiscoBackend.__new__(DiscoBackend)
    backend.model,backend.secondary=Denoiser(),None
    backend.diffusion=create_gaussian_diffusion(steps=1000,learn_sigma=True,
        timestep_respacing='ddim4',rescale_timesteps=True)
    backend.encoders,backend.device,backend.lpips=[],'cpu',None
    config=RunConfig(model_version='disco_512')
    config.disco.sampler=sampler
    config.disco.init_scale=0
    config.disco.steps_schedule={0:4,2:8}
    config.disco.style_strength_schedule={0:1.,2:.5}
    model=backend.model
    init=Image.new('RGB',(64,64))
    for frame,expected_steps,expected_total in [(0,4,4),(1,6,5),(2,8,4)]:
        events=[]
        with render_event_scope(events.append):
            image=backend.render(config,init,'','',42,frame_num=frame)
        assert backend.model is model
        assert backend.diffusion.num_timesteps==expected_steps
        assert len(events)==events[-1]['total_steps']==expected_total
        assert np.isfinite(np.asarray(image)).all()
    first=np.asarray(image)
    # Rebuilding the sampler after another budget retains deterministic frame seeds.
    backend.render(config,init,'','',42,frame_num=0)
    np.testing.assert_array_equal(first,np.asarray(backend.render(config,init,'','',42,frame_num=2)))
