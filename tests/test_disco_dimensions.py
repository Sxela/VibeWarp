import pytest
import torch
from PIL import Image

from vibewarp.config import RunConfig
from vibewarp.config_io import apply_path_defaults, validate_config
from vibewarp.video.input import fit_dimensions


@pytest.mark.parametrize('width,height,maximum,expected',[
    (1280,720,768,(768,448)),(1920,1080,1000,(1024,576)),
    (720,1280,768,(448,768)),(512,512,512,(512,512)),
])
def test_source_dimensions_round_up_to_64(width,height,maximum,expected):
    assert fit_dimensions(width,height,maximum,multiple=64,round_up=True)==expected


def test_manual_disco_dimensions_round_up_before_validation():
    config=RunConfig(model_version='disco_512')
    config.video.width,config.video.height=513,431
    apply_path_defaults(config)
    assert (config.video.width,config.video.height)==(576,448)
    assert config.video.max_size==513
    assert not validate_config(config,require_inputs=False)
    apply_path_defaults(config)
    assert (config.video.width,config.video.height)==(576,448)


@pytest.mark.parametrize('sampler',['ddim','plms','distilled'])
def test_video_pipeline_uses_source_aspect_for_legacy_square_configs(monkeypatch,sampler):
    from vibewarp import pipeline
    config=RunConfig(model_version='disco_512')
    config.disco.sampler=sampler
    config.disco.distilled_guided=True
    config.video.width=config.video.height=512
    config.video.max_size=0
    config.video.video_init_path='portrait.mp4'
    monkeypatch.setattr('vibewarp.config_io.validate_config',lambda *a,**kw:[])
    calls=[]
    def resolve(path,max_size,**kwargs):
        calls.append((path,max_size,kwargs))
        return fit_dimensions(1080,1920,max_size,**kwargs)
    monkeypatch.setattr(pipeline,'resolve_video_dimensions',resolve)
    class ReachedSetup(Exception): pass
    def setup(config,**kwargs):
        assert (config.video.width,config.video.height)==(320,512)
        raise ReachedSetup()
    monkeypatch.setattr(pipeline,'setup_directories',setup)
    with pytest.raises(ReachedSetup):
        pipeline.run(config)
    assert calls==[('portrait.mp4',512,{'multiple':64,'round_up':True})]


def test_exact_unconditional_profile_keeps_fixed_reference_dimensions():
    from vibewarp.config import DISCO_DISTILLED_PRESET
    from vibewarp.config_io import config_from_dict
    from vibewarp.web import _ui_config
    config=config_from_dict(DISCO_DISTILLED_PRESET)
    config.disco.few_step_lora_path='adapter.safetensors'
    config=_ui_config(apply_path_defaults(config))
    assert config.video.max_size==0
    assert (config.video.width,config.video.height)==(512,512)
    assert not validate_config(config,require_inputs=False)


def test_direct_disco_render_rounds_input_and_guidance_mask():
    import numpy as np
    from vibewarp.core.disco import DiscoBackend
    from vibewarp.vendor.guided_diffusion.script_util import create_gaussian_diffusion
    class Denoiser(torch.nn.Module):
        def forward(self,x,t,**kwargs):
            return torch.cat([x*.1,torch.zeros_like(x)],dim=1)
    backend=DiscoBackend.__new__(DiscoBackend)
    backend.model,backend.secondary=Denoiser(),None
    backend.diffusion=create_gaussian_diffusion(steps=1000,learn_sigma=True,
        timestep_respacing='ddim4',rescale_timesteps=True)
    backend.encoders,backend.device,backend.lpips=[],'cpu',None
    config=RunConfig(model_version='disco_512')
    config.disco.steps=4
    config.disco.init_scale=0
    config.disco.image_scale=1
    image=backend.render(config,Image.new('RGB',(65,63)),'','',42,
        mask=np.ones((63,65,3),dtype=np.float32),guidance_image=Image.new('RGB',(65,63)))
    assert image.size==(128,64)
