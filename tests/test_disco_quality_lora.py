import copy
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch
from torch import nn
from PIL import Image
from safetensors.torch import save_file

from vibewarp.config import RunConfig
from vibewarp.config_io import config_from_dict,config_from_settings,validate_config,config_schema
from vibewarp.core.disco_lora import (AdapterLayer,DistilledDiscoBackend,TEACHER_SHA256,
    attach_quality_adapter,inject_adapters,quality_adapter_config,use_adapter)


def export(path,model,rank=3,alpha=6,quality=True,changes=None):
    model=copy.deepcopy(model)
    inject_adapters(model,rank,alpha)
    state={f'{name}.adapters.student.{key}':tensor
        for name,layer in model.named_modules() if isinstance(layer,AdapterLayer)
        for key,tensor in layer.adapters['student'].state_dict().items()}
    # Give both factors a measurable nonzero contribution.
    state={key:torch.full_like(tensor,.01) for key,tensor in state.items()}
    metadata=dict(teacher_sha256=TEACHER_SHA256,size=512,rank=rank,alpha=alpha,
        method='aesthetic real-image epsilon MSE' if quality else 'supervised DDIM endpoint regression',
        objective_version='aesthetic-epsilon-mse-v1' if quality else 'supervised-clean-sqrt-snr-v1',
        fp32_output_head=True,precision='fp32')
    if not quality: metadata['grid']=[999,749,499,249,-1]
    metadata.update(changes or {})
    save_file(state,str(path),metadata={'config':json.dumps(metadata),'iteration':'8908','weights':'raw'})
    model.load_state_dict(state,strict=False)
    return model


@pytest.mark.parametrize('base,shape',[(nn.Linear(4,5),(2,4)),
    (nn.Conv1d(3,4,3,padding=1),(1,3,7)),(nn.Conv2d(3,4,3,padding=1),(1,3,7,7))])
def test_quality_and_turbo_have_independent_ranks_scales_and_input_gradients(tmp_path,base,shape):
    model=nn.Sequential(base)
    quality=export(tmp_path/'quality.safetensors',model)
    inject_adapters(model,2,4)
    for parameter in model[0].adapters['student'].parameters(): nn.init.normal_(parameter,std=.1)
    attach_quality_adapter(model,tmp_path/'quality.safetensors')
    layer=model[0]
    assert layer.adapters['student'][0].weight.shape[0]==2
    assert layer.adapters['quality'][0].weight.shape[0]==3
    for q,t in ((0.,0.),(.5,0.),(0.,.7),(.75,.7)):
        x=torch.randn(shape,requires_grad=True)
        expected=base(x)
        if q: expected=expected+(q*2)*quality[0].adapters['student'](x)
        if t: expected=expected+(t*2)*layer.adapters['student'](x)
        with use_adapter(model,strength=t,quality_strength=q):
            actual=model(x)
            if q==t==0: assert torch.equal(actual,base(x))
            torch.testing.assert_close(actual,expected)
            torch.testing.assert_close(torch.autograd.grad(actual.sum(),x,retain_graph=True)[0],
                                       torch.autograd.grad(expected.sum(),x)[0])
        assert layer.active is None and layer.quality_strength==0 and layer.strength==1


@pytest.mark.parametrize('changes',[{'teacher_sha256':'wrong'},{'method':'wrong'},
    {'objective_version':'wrong'},{'rank':True},{'alpha':float('nan')},{'fp32_output_head':False},
    {'diffusion_steps':6},{'variance_training':True},{'precision':'bf16'}])
def test_invalid_quality_metadata_rejected(tmp_path,changes):
    path=tmp_path/'quality.safetensors'
    export(path,nn.Sequential(nn.Linear(3,4)),changes=changes)
    with pytest.raises(ValueError): quality_adapter_config(path)


def test_quality_only_needs_no_turbo_branch_and_validates_all_tensors(tmp_path):
    model=nn.Sequential(nn.Conv2d(3,4,1))
    path=tmp_path/'quality.safetensors'
    export(path,model)
    config,metadata=attach_quality_adapter(model,path)
    assert set(model[0].adapters)=={'quality'}
    assert metadata['iteration']=='8908' and config['rank']==3
    x=torch.randn(1,3,4,4)
    with use_adapter(model,None,strength=0,quality_strength=1):
        assert not torch.equal(model(x),model[0].base(x))
    bad=tmp_path/'bad.safetensors'
    header=dict(method='aesthetic real-image epsilon MSE',objective_version='aesthetic-epsilon-mse-v1',
                teacher_sha256=TEACHER_SHA256,size=512,rank=3,alpha=6)
    save_file({'extra':torch.zeros(1)},str(bad),metadata={'config':json.dumps(header)})
    with pytest.raises(ValueError,match='keys mismatch'):
        attach_quality_adapter(nn.Sequential(nn.Conv2d(3,4,1)),bad)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Conv2d(3,6,1)
        self.out=nn.Sequential(nn.Conv2d(6,6,1))
    def forward(self,x,t,**kwargs): return self.out(self.body(x))


@pytest.mark.parametrize('turbo',[False,True])
def test_quality_backend_loads_base_once_and_preserves_guided_video_path(tmp_path,monkeypatch,turbo):
    from vibewarp.vendor.guided_diffusion.script_util import create_gaussian_diffusion
    from vibewarp import pipeline
    base=TinyModel()
    base_path=tmp_path/'base.pt';torch.save(base.state_dict(),base_path)
    quality_path=tmp_path/'quality.safetensors';export(quality_path,base)
    turbo_path=tmp_path/'turbo.safetensors'
    if turbo: export(turbo_path,base,rank=2,alpha=4,quality=False)
    config=RunConfig(model_version='disco_512')
    d=config.disco
    d.quality_lora_enabled=True;d.quality_lora_path=str(quality_path)
    d.few_step_lora_path=str(turbo_path)  # nonexistent and unused for quality-only
    d.checkpoint_path=str(base_path);d.sampler='distilled' if turbo else 'ddim'
    d.use_secondary_model=False;d.eta=0;d.encoders=[];d.clip_guidance_scale=0;d.init_scale=0;d.image_scale=1
    d.steps_schedule={0:6};d.style_strength_schedule={0:.5};d.guidance_noise_cap_percent=15
    config.flow.flow_warp=False
    assert not validate_config(config,require_inputs=False)
    from vibewarp.core import disco_lora
    original_hash=disco_lora.sha256
    monkeypatch.setattr(disco_lora,'sha256',lambda path:TEACHER_SHA256 if str(path)==str(base_path) else original_hash(path))
    loads=[]
    def create(**kwargs):
        loads.append(True)
        return TinyModel(),create_gaussian_diffusion(steps=1000,learn_sigma=True,rescale_timesteps=True)
    monkeypatch.setattr('vibewarp.vendor.guided_diffusion.script_util.create_model_and_diffusion',create)
    backend=pipeline.load_models(config,device='cpu')['disco_backend']
    assert isinstance(backend,DistilledDiscoBackend)
    assert len(loads)==1 and backend.has_turbo==turbo
    assert backend.quality_metadata['iteration']=='8908'
    assert backend.diffusion.timestep_map==[165,332,499,665,832,999]
    before={name:layer.scale for name,layer in backend.model.named_modules() if isinstance(layer,AdapterLayer)}
    from vibewarp.core.diffusion import RenderContext,FrameState,_render_single_frame
    frames=tmp_path/'frames';frames.mkdir()
    for frame in range(5): Image.new('RGB',(64,64),(60+frame,80,100)).save(frames/f'{frame+1:06d}.jpg')
    config.video.width=config.video.height=64
    config.flow.flow_warp=True
    config.color.before_enabled=config.color.after_enabled=False
    ctx=RenderContext(config=config,disco_backend=backend,batch_folder=str(tmp_path),video_frames_folder=str(frames))
    def identity_warp(ctx,previous,*args,**kwargs):
        ctx._disco_consistency_mask=np.ones((64,64,3),dtype=np.float32)
        return (previous.copy(),previous.copy()) if kwargs.get('return_warped') else previous.copy()
    monkeypatch.setattr('vibewarp.core.diffusion.warp_between_frames',identity_warp)
    state=FrameState()
    for frame in range(5):
        # Independent per-render strengths, including both disabled, never
        # multiply scales cumulatively or lose branches between frames.
        d.quality_lora_strength=(0.,.5,1.,.75,0.)[frame]
        d.lora_strength=(0.,1.,.5,1.,0.)[frame]
        image,_=_render_single_frame(ctx,state,frame,0,str(tmp_path),'png')
        assert image.size==(64,64)
        for name,layer in backend.model.named_modules():
            if isinstance(layer,AdapterLayer):
                assert layer.active is None and layer.quality_strength==0 and layer.scale==before[name]
    # On the quality path the enabled noise-relative cap replaces the absolute
    # gradient clamp, so changing the latter cannot silently weaken guidance.
    d.quality_lora_strength=1
    d.lora_strength=1
    init=Image.new('RGB',(64,64),(60,80,100))
    a=backend.render(config,init,'','',42,guidance_image=init)
    d.clamp_max=1e-12
    b=backend.render(config,init,'','',42,guidance_image=init)
    np.testing.assert_array_equal(np.asarray(a),np.asarray(b))


@pytest.mark.parametrize('field,value,path',[('quality_lora_path','','disco.quality_lora_path'),
    ('quality_lora_strength',-1,'disco.quality_lora_strength'),
    ('quality_lora_strength',float('inf'),'disco.quality_lora_strength'),
    ('sampler','plms','disco.sampler')])
def test_quality_configuration_requires_compatible_guided_settings(field,value,path):
    config=RunConfig(model_version='disco_512')
    config.disco.quality_lora_enabled=True
    config.disco.quality_lora_path='quality.safetensors'
    config.disco.use_secondary_model=False
    setattr(config.disco,field,value)
    assert any(error['path']==path for error in validate_config(config,require_inputs=False))


def test_quality_settings_roundtrip_and_path_preview(tmp_path):
    from vibewarp.settings import save_settings,settings_to_dict
    from vibewarp.model_assets import required_model_files
    config=RunConfig(model_version='disco_512')
    config.disco.quality_lora_enabled=True
    config.disco.quality_lora_path=str(tmp_path/'quality.safetensors')
    config.disco.quality_lora_strength=.75
    path=save_settings(settings_to_dict(config),str(tmp_path),'quality',0)
    restored=config_from_settings(path)
    assert restored.disco.quality_lora_strength==.75 and restored.disco.quality_lora_enabled
    assert any(row['name']=='Quality LoRA' and row['status']=='missing' and not row['downloadable']
               for row in required_model_files(restored))
    props=config_schema()['properties']['disco']['properties']
    assert props['quality_lora_path']['tier']=='project'
    assert props['quality_lora_strength']['tier']=='render'
    assert not RunConfig().disco.quality_lora_enabled
