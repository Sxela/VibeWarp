import json
import math

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from vibewarp.config import RunConfig
from vibewarp.config_io import config_from_dict, config_from_settings, config_schema, validate_config
from vibewarp.core.disco import DiscoBackend
from vibewarp.core.disco_lora import guided_diffusion
from vibewarp.disco_cutouts import held_value, inference_progress, legacy_timestep_trace
from vibewarp.settings import settings_to_dict


@pytest.mark.parametrize(('strength','repeats','grid','sampler'), [
    (1,1,[999,832,665,499,332,165,-1],'ddim'),
    (4/6,1,[999,832,665,499,332,165,-1],'ddim'),
    (1,3,[999,832,665,499,332,165,-1],'ddim'),
    (1,1,[999,981,850,400,55,0,-1],'ddim'),
    (1/6,1,[999,832,665,499,332,165,-1],'ddim'),
    (1,1,[999,832,665,499,332,165,-1],'plms'),
])
def test_real_sampling_uses_executed_step_progress(monkeypatch,strength,repeats,grid,sampler):
    class Denoiser(torch.nn.Module):
        def forward(self,x,t,**kwargs):
            return torch.cat([x*.1,torch.zeros_like(x)],dim=1)
    class Encoder:
        size=8
        def text(self,texts):return torch.tensor([[1.,0.,0.] for _ in texts])
        def image(self,image):return F.normalize(image.mean((-1,-2)),dim=-1)
    backend=DiscoBackend.__new__(DiscoBackend)
    backend.model,backend.secondary,backend.lpips=Denoiser(),None,None
    backend.device='cpu';backend.encoders=[Encoder()];backend.diffusion=guided_diffusion(grid)
    config=RunConfig(model_version='disco_512')
    d=config.disco
    d.sampler=sampler;d.steps=6;d.style_strength=strength;d.clip_grad_steps=repeats
    d.init_scale=d.image_scale=d.range_scale=d.tv_scale=d.sat_scale=0
    d.clip_guidance_scale=1;d.cut_icgray_schedule={0:.1,50:.9};d.inpaint=False
    calls=[]
    def cutouts(image,size,overview,inner,power,gray,**kwargs):
        calls.append((overview,inner,gray))
        return F.interpolate(image,(size,size),mode='area').repeat(overview+inner,1,1,1)
    monkeypatch.setattr('vibewarp.core.disco.make_cutouts',cutouts)
    backend.render(config,Image.new('RGB',(64,64),(80,100,120)),'red painting','',42)
    from vibewarp.core.disco_repeated import skip_for_strength
    total=6-skip_for_strength(6,strength)
    positions=[100*index/max(total-1,1) for index in range(total)]
    expected=[(4 if p<40 else 2 if p<60 else 0,12 if p<40 else 14 if p<60 else 16,.1 if p<50 else .9) for p in positions]
    assert calls==([expected[0],*expected] if sampler=='plms' else [value for value in expected for _ in range(repeats)])


def test_progress_and_hold_boundaries():
    assert [inference_progress(i,6) for i in range(6)]==[0,20,40,60,80,100]
    assert [inference_progress(i,4) for i in range(4)]==pytest.approx([0,100/3,200/3,100])
    assert inference_progress(0,1)==0
    schedule={0:4,40:2,60:0}
    assert [held_value(schedule,p) for p in (0,39.9,40,59.9,60,100)]==[4,4,2,2,0,0]
    assert held_value({0:4,math.nextafter(100/3,math.inf):2},100/3)==2
    assert held_value({0:4,1e-13:2},0)==4


@pytest.mark.parametrize('original_time',[True,False])
@pytest.mark.parametrize('sampler',['ddim','distilled'])
def test_legacy_axes_migrate_step_selections_once(original_time,sampler,tmp_path):
    boundary=400 if original_time else 40
    raw={'model_version':'disco_512','disco':{'sampler':sampler,'steps':6,'style_strength':4/6,
        'cut_schedule_original_time':original_time,'cut_overview':{0:4,boundary:2,boundary*1.5:0},
        'cut_innercut':{0:12,boundary:14,boundary*1.5:16},'cut_icgray_schedule':{0:.1,boundary*1.5:.9}}}
    # Version 2 prevents conversion while obtaining the old schedule trace.
    before=config_from_dict({**raw,'disco':{**raw['disco'],'cut_schedule_version':2}})
    times=legacy_timestep_trace(before)
    positions=[999-t if original_time else 100*(1-t/1000) for t in times]
    def strict(schedule,p):return next((schedule[k] for k in sorted(schedule,reverse=True) if k<=p),0)
    migrated=config_from_dict(raw)
    assert migrated.disco.cut_schedule_version==2
    assert not hasattr(migrated.disco,'cut_schedule_original_time')
    for name in ('cut_overview','cut_innercut','cut_icgray_schedule'):
        expected=[strict(getattr(before.disco,name),p) for p in positions]
        assert [held_value(getattr(migrated.disco,name),inference_progress(i,len(times))) for i in range(len(times))]==expected
    saved=settings_to_dict(migrated)
    assert 'disco_cut_schedule_original_time' not in saved
    path=tmp_path/'native.txt';path.write_text(json.dumps(saved))
    assert settings_to_dict(config_from_settings(path))==saved


@pytest.mark.parametrize(('name','schedule'),[
    ('cut_overview',{-1:2}),('cut_overview',{101:2}),('cut_overview',{math.inf:2}),
    ('cut_overview',{0:1.5}),('cut_overview',{0:True}),('cut_innercut',{0:-1}),
    ('cut_icgray_schedule',{0:1.1}),('cut_icgray_schedule',{101:.2}),
])
def test_invalid_percentages_and_cut_values(name,schedule):
    config=RunConfig(model_version='disco_512');setattr(config.disco,name,schedule)
    assert any(error['path']==f'disco.{name}' for error in validate_config(config,require_inputs=False))


def test_schema_uses_one_progress_domain_for_all_cutout_schedules():
    fields=config_schema()['properties']['disco']['properties']
    assert 'cut_schedule_original_time' not in fields
    assert fields['cut_schedule_version']['tier']=='hidden'
    for name in ('cut_overview','cut_innercut','cut_icgray_schedule'):
        assert fields[name]['schedule_domain']=='inference_progress'


def test_older_native_defaults_also_migrate_and_new_version_does_not():
    old=config_from_dict({'model_version':'disco_512','disco':{'steps':6,'style_strength':4/6}})
    new=config_from_dict({'model_version':'disco_512','disco':{'cut_schedule_version':2,'steps':6,'style_strength':4/6}})
    assert old.disco.cut_overview != new.disco.cut_overview
    assert new.disco.cut_overview == {0:4,40:2,60:0}


def test_fractional_boundary_and_version_survive_turbo_resume_fingerprinting(tmp_path):
    from vibewarp.core.turbo import TurboState
    from dataclasses import asdict
    config=RunConfig(model_version='disco_512')
    config.disco.cut_overview={0:4,math.nextafter(100/3,-math.inf):2,100:0}
    assert TurboState(config,tmp_path,0).signature == TurboState(config_from_dict(asdict(config)),tmp_path,0).signature
