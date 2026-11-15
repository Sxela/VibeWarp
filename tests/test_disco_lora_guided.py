from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from vibewarp.config import RunConfig,DISCO_DISTILLED_GUIDED_PRESET
from vibewarp.config_io import apply_path_defaults,config_from_dict,validate_config
from vibewarp.core.disco_lora import (AdapterLayer,DistilledDiscoBackend,GRIDS,
    GuidedAdapterModel,guided_diffusion,inject_adapters,rescaled_grid)
from vibewarp.render_events import render_event_scope


@pytest.mark.parametrize('steps',[2,4,6,12,30,999,1000])
def test_rescaled_grids_use_original_vp_times_and_clean_endpoint(steps):
    from vibewarp.vendor.guided_diffusion.gaussian_diffusion import get_named_beta_schedule
    grid=rescaled_grid(GRIDS[4],steps)
    assert len(grid)==steps+1 and grid[0]==999 and grid[-1]==-1
    assert all(a>b for a,b in zip(grid,grid[1:]))
    if steps in GRIDS:
        assert grid==GRIDS[steps]
    diffusion=guided_diffusion(grid)
    assert diffusion.original_num_steps==1000
    assert list(reversed(diffusion.timestep_map))==grid[:-1]
    original=np.cumprod(1-get_named_beta_schedule('linear',1000))
    np.testing.assert_allclose(diffusion.alphas_cumprod,original[diffusion.timestep_map],rtol=1e-12,atol=1e-15)


def guided_config():
    config=config_from_dict(DISCO_DISTILLED_GUIDED_PRESET)
    config.disco.few_step_lora_path='adapter.safetensors'
    return config


def test_guided_mode_preserves_guidance_video_masks_flow_and_schedules():
    config=guided_config()
    config.video.width,config.video.height=513,767
    config.disco.steps_schedule={0:6,10:12}
    config.disco.style_strength_schedule=[1.,.65]
    config.disco.image_scale=100
    config.disco.mask_frame_guidance=True
    config.turbo.enabled=True
    apply_path_defaults(config)
    assert (config.video.width,config.video.height)==(576,768)
    assert not validate_config(config,require_inputs=False)
    assert config.disco.encoders and config.disco.clip_guidance_scale
    assert config.flow.flow_warp and config.disco.inpaint


@pytest.mark.parametrize('name,value',[('eta',.5),('use_secondary_model',True)])
def test_guided_mode_accepts_secondary_and_stochastic_ddim(name,value):
    config=guided_config()
    setattr(config.disco,name,value)
    assert not validate_config(config,require_inputs=False)


def test_quality_and_turbo_adapters_allow_secondary_and_guidance_together():
    config = guided_config()
    config.disco.quality_lora_enabled = True
    config.disco.quality_lora_path = 'quality.safetensors'
    assert config.disco.use_secondary_model
    assert config.disco.distilled_guided
    assert config.disco.encoders and config.disco.clip_guidance_scale > 0
    assert not validate_config(config, require_inputs=False)


def test_guided_assets_include_lora_and_encoders_without_downloading(tmp_path):
    from vibewarp.model_assets import required_model_files
    from vibewarp.web_downloads import DownloadManager
    config=guided_config()
    config.root_dir=str(tmp_path)
    config.flow.flow_warp=False
    config.disco.encoders=['ViT-B-32','siglip2-base']
    files=required_model_files(config)
    assert any(file['name']=='Distilled LoRA' for file in files)
    assert any(file['name']=='ViT-B-32.pt' for file in files)
    assert any(file['name'].startswith('siglip2-base') for file in files)
    assert any(spec[3]=='bundle' for spec in DownloadManager()._specs(config))
    assert not (tmp_path/'models').exists()


@pytest.mark.parametrize('progressive',[False,True])
def test_guided_lora_keeps_adapters_active_during_checkpointed_guidance_and_respaces(progressive):
    from vibewarp.vendor.guided_diffusion.nn import checkpoint
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer=torch.nn.Conv2d(3,6,1)
            self.times=[]
            torch.nn.init.constant_(self.layer.weight,.02)
            torch.nn.init.zeros_(self.layer.bias)
        def block(self,x):
            # Includes recomputation during autograd.grad, not just initial calls.
            assert self.layer.active=='student'
            assert self.layer.strength==.4
            return self.layer(x).sin()
        def forward(self,x,t,**kwargs):
            self.times.extend(int(value) for value in t)
            return checkpoint(self.block,(x,),self.parameters(),True)
    class Encoder:
        size=8
        calls=0
        def text(self,texts):
            return torch.tensor([[1.,0.,0.] for text in texts])
        def image(self,image):
            self.calls+=1
            return torch.nn.functional.normalize(image.mean((-1,-2)),dim=-1)
    model=Model()
    inject_adapters(model,2,4)
    torch.nn.init.constant_(model.layer.adapters['student'][1].weight,.01)
    model.eval().requires_grad_(False)
    backend=DistilledDiscoBackend.__new__(DistilledDiscoBackend)
    backend.model=GuidedAdapterModel(model)
    backend.grid=GRIDS[4]
    backend.diffusion=guided_diffusion(GRIDS[4])
    backend.device=torch.device('cpu')
    backend.encoders,backend.secondary=[Encoder()],None
    lpips_calls=[]
    def lpips(image,target):
        lpips_calls.append(image.shape[-2:])
        return (image-target).square().mean(1,keepdim=True)
    lpips.spatial=True
    backend.lpips=lpips
    config=guided_config()
    d=config.disco
    d.lora_strength=.4
    d.steps_schedule={0:6,1:12}
    d.style_strength_schedule={0:1.,1:.5}
    d.init_scale,d.image_scale,d.clip_guidance_scale=1.,1.,1.
    d.mask_frame_guidance=True
    d.guidance_noise_cap_percent=10
    d.cut_overview,d.cut_innercut={0:1},{0:0}
    d.skip_augs=True
    d.preview_interval=0
    config.diffusion.tiled_sampler=progressive
    config.diffusion.sampler_tile_size=64
    if progressive:
        config.diffusion.sampler_scale_schedule={0:50,2:100}
    init=Image.new('RGB',(128,128),(80,100,120))
    mask=np.ones((128,128,3),dtype=np.float32)
    mask[:64]=0
    for frame,steps in ((0,6),(1,12)):
        model.times.clear()
        events=[]
        with render_event_scope(events.append):
            image=backend.render(config,init,'a red painting','blur',42,
                frame_num=frame,mask=mask,guidance_image=init)
        assert len(events)==6  # scheduled strength reduces the second frame's budget
        assert image.size==(128,128)
        assert backend.diffusion.num_timesteps==steps
        assert set(model.times)<=set(rescaled_grid(GRIDS[4],steps)[:-1])
        assert model.layer.active is None
        assert all(parameter.grad is None for parameter in model.parameters())
        assert all(event['guidance_update_rms']<=event['guidance_noise_budget_rms']+1e-4 for event in events)
    assert backend.encoders[0].calls and lpips_calls
    if progressive:
        assert (64,64) in lpips_calls and (128,128) in lpips_calls


def test_distilled_frame_route_forwards_prompts_mask_target_and_strength(tmp_path):
    from vibewarp.core.disco import render_disco_frame
    from vibewarp.core.diffusion import FrameState
    config=guided_config()
    config.video.width=config.video.height=64
    config.text_prompts={0:'a painting'}
    config.negative_prompts={0:'blur'}
    config.disco.mask_frame_guidance=True
    path=tmp_path/'init.png'
    Image.new('RGB',(64,64)).save(path)
    captured={}
    class Backend:
        def render(self,config,init,prompt,negative,seed,**kwargs):
            captured.update(kwargs,prompt=prompt,negative=negative)
            return init
    ctx=SimpleNamespace(config=config,video_frames_folder=str(tmp_path),batch_folder=str(tmp_path),
        disco_backend=Backend(),_turbo_strength=.25,_disco_consistency_mask=np.ones((64,64,3)))
    render_disco_frame(ctx,FrameState(frame_num=1,init_image=str(path)))
    assert captured['prompt']=='a painting' and captured['negative']=='blur'
    assert captured['mask'] is not None and captured['guidance_image'] is not None
    assert captured['mask_guidance'] is True and captured['strength']==.25
