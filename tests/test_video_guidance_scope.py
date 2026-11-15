"""Behavioral checks for the notebook integration and trusted temporal losses."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from PIL import Image

from vibewarp.config import RunConfig
from vibewarp.config_io import config_from_settings, validate_config
from vibewarp.settings import settings_to_dict
from vibewarp.core.trusted_guidance import masked_lpips_loss, masked_spherical_loss, resize_trust


def test_notebook_import_and_native_roundtrip(tmp_path):
    original = dict(diffusion_model='512x512_diffusion_uncond_finetune_008100',steps=6,
        lora_config={'turbo_path':'turbo.safetensors','quality_path':'quality.safetensors',
            'turbo_strength':.5,'quality_strength':1.,'grid':[699,582,466,349,233,116,-1]},
        text_prompts={0:['a painting']},eta=.55,use_secondary_model=True,clip_grad_steps=4,
        cutn_batches=4,cut_overview='[12]*400+[4]*600',cut_innercut='[4]*400+[12]*600',
        frames_skip_steps='34%',frames_scale=30000,inpainting_stop=1,max_frames=20,
        ViTB32=True,ViTB16=True,RN50=True,temporal_guidance_cc_trusted_only=True)
    path=tmp_path/'notebook.txt'; path.write_text(json.dumps(original))
    c=config_from_settings(path)
    assert c.model_version=='disco_512'
    assert c.disco.steps==6 and c.disco.clip_grad_steps==4
    assert c.disco.quality_lora_strength==1 and c.disco.lora_strength==.5
    assert c.disco.eta==.55 and c.disco.use_secondary_model
    assert c.disco.timestep_grid==[699,582,466,349,233,116,-1]
    assert c.disco.init_scale_schedule=={0:0.,1:30000.}
    assert c.disco.style_strength_schedule[1]==1-2/6
    from vibewarp.core.disco_repeated import skip_for_strength
    assert skip_for_strength(6,c.disco.style_strength_schedule[1])==2
    assert c.disco.cut_schedule_version==2
    assert c.disco.cut_overview[0]==12
    assert set(c.disco.cut_overview.values())=={12,4}
    assert max(c.disco.cut_overview)<=100
    assert not validate_config(c,require_inputs=False)
    native=tmp_path/'native.txt'; native.write_text(json.dumps(settings_to_dict(c)))
    loaded=config_from_settings(native)
    assert json.loads(json.dumps(settings_to_dict(c)))==json.loads(json.dumps(settings_to_dict(loaded)))


@pytest.mark.parametrize('expression',["__import__('os').system('bad')",'[1]*1000*1000','[1]*-1'])
def test_notebook_schedule_is_data_only(expression):
    from vibewarp.disco_settings import cut_schedule
    with pytest.raises(ValueError): cut_schedule(expression,{0:1})


def test_masked_losses_preserve_all_trusted_and_have_zero_rejected_gradients():
    from vibewarp.core.losses import spherical_dist_loss
    torch.manual_seed(22)
    x=torch.randn(1,4,8,8,requires_grad=True); target=torch.randn_like(x)
    full=torch.ones(1,1,8,8)
    assert torch.equal(masked_spherical_loss(x,target,full),spherical_dist_loss(x,target).sum())
    mask=full.clone(); mask[...,4:]=0
    pixel=lambda a,b:(a-b).square().mean(1,keepdim=True)
    loss=masked_lpips_loss(x,target,mask,pixel)+masked_spherical_loss(x,target,mask)
    grad=torch.autograd.grad(loss,x)[0]
    assert torch.isfinite(grad).all() and grad[...,:4].abs().sum()>0
    assert torch.count_nonzero(grad[...,4:])==0
    zero=masked_spherical_loss(x,target,torch.zeros_like(mask))
    assert zero==0 and torch.isfinite(zero)


def test_masked_pixel_loss_keeps_actual_context_and_full_area_normalization():
    x=torch.ones(1,3,8,8,requires_grad=True); target=torch.zeros_like(x)
    mask=torch.zeros(1,1,8,8); mask[...,:2]=1
    def spatial(a,b):
        assert torch.equal(a,x)  # no substitution with target outside trust
        return (a-b).square().mean(1,keepdim=True)
    loss=masked_lpips_loss(x,target,mask,spatial)
    assert loss.item()==.25
    grad=torch.autograd.grad(loss,x)[0]
    assert torch.count_nonzero(grad[...,2:])==0


def test_sd_temporal_wrapper_gates_applied_latent_correction():
    from vibewarp.core.diffusion import make_temporal_guidance_model_fn
    torch.manual_seed(1)
    x=torch.randn(1,4,8,8); target=torch.randn_like(x)
    mask=torch.zeros(8,8); mask[:,:4]=1
    def build(trust):
        return make_temporal_guidance_model_fn(lambda image,sigma,**kw:image.clone(),
            sd_model=None,target_image=None,target_latent=target,init_scale=0,init_latent_scale=1,
            clamp_grad=False,clamp_max=1,add_noise=False,guidance_start_code=None,
            source_label='prev warped',trusted_mask=trust)
    normal=build(None)(x,torch.ones(1))
    full=build(torch.ones_like(mask))(x,torch.ones(1))
    assert torch.equal(normal,full)
    empty=build(torch.zeros_like(mask))(x,torch.ones(1))
    assert torch.equal(empty,x)
    output=build(mask)(x,torch.ones(1))
    assert torch.equal(output[...,4:],x[...,4:])
    assert not torch.equal(output[...,:4],x[...,:4])


def test_original_grid_and_repeated_guidance_advance_only_once():
    from vibewarp.core.disco_repeated import install_multi_guidance,validate_grid
    from vibewarp.core.disco_lora import guided_diffusion
    grid=[699,582,466,349,233,116,-1]
    diffusion=guided_diffusion(validate_grid(grid,6))
    assert list(reversed(diffusion.timestep_map))[2:]==[466,349,233,116]
    states=[]; advances=[]
    class Step:
        alphas_cumprod=np.array([.5]); alphas_cumprod_prev=np.array([.7])
        def ddim_sample(self,model,x,t,cond_fn=None,**kw):
            advances.append(x.clone())
            return x+cond_fn(x,t)
        def _scale_timesteps(self,t): return t
    step=Step(); install_multi_guidance(step,4,False,4)
    def gradient(x,t,**kw): states.append(x.clone()); return torch.ones_like(x)
    step.ddim_sample(None,torch.zeros(1,1),torch.zeros(1,dtype=torch.long),cond_fn=gradient,eta=.55)
    assert len(states)==4 and len(advances)==1
    assert all(bool((b>a).all()) for a,b in zip(states,states[1:]))
    from vibewarp.core.disco_repeated import cut_batches_per_gradient
    assert cut_batches_per_gradient(4,4,False)==[4,4,4,4]
    assert cut_batches_per_gradient(4,4,True)==[1,1,1,1]


@pytest.mark.parametrize('kind',['linear','conv1','conv2'])
def test_fixed_lora_merge_preserves_outputs_and_input_gradients(kind):
    from vibewarp.core.disco_lora import AdapterLayer,merge_adapter_weights,use_adapter
    base={'linear':lambda:torch.nn.Linear(4,6),'conv1':lambda:torch.nn.Conv1d(4,6,3,padding=1),'conv2':lambda:torch.nn.Conv2d(4,6,3,padding=1)}[kind]()
    layer=AdapterLayer(base,2,2)
    layer.add_branch('quality',3); layer.quality_scale=1
    for branch in layer.adapters.values(): torch.nn.init.normal_(branch[1].weight,std=.1)
    x=torch.randn({'linear':(2,4),'conv1':(1,4,8),'conv2':(1,4,8,8)}[kind],requires_grad=True)
    with use_adapter(layer,strength=.5,quality_strength=1):
        expected=layer(x); grad=torch.autograd.grad(expected.sum(),x)[0]
    merge_adapter_weights(layer,.5,1)
    with use_adapter(layer,strength=.5,quality_strength=1): actual=layer(x)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(),x)[0],grad,atol=2e-6,rtol=2e-5)


def test_turbo_preroll_blends_diffused_frame_but_keeps_raw_feedback(tmp_path):
    from vibewarp.core.turbo import TurboState
    c=RunConfig(); c.turbo.interval=2
    state=TurboState(c,tmp_path,0); state._frame=1; state.active=False
    state.current_init=Image.new('RGB',(8,8),(20,20,20))
    result=state.finish(Image.new('RGB',(8,8),(100,100,100)))
    assert result.getpixel((0,0))==(60,60,60)
    assert state.feedback.getpixel((0,0))==(100,100,100)


def test_first_frame_native_setting_bypasses_diffusion_and_checks_dimensions(tmp_path):
    from vibewarp.core.diffusion import RenderContext,FrameState,_render_single_frame
    c=RunConfig(); c.video.width=c.video.height=64
    path=tmp_path/'reference.png'; Image.new('RGB',(64,64),'red').save(path)
    c.video.first_frame_path=str(path)
    ctx=RenderContext(config=c,batch_folder=str(tmp_path)); ctx._sequence_start_frame=0
    state=FrameState(); image,saved=_render_single_frame(ctx,state,0,0,str(tmp_path),'png')
    assert Path(saved).is_file() and image.getpixel((0,0))==(255,0,0)
    assert state.prev_frame is not None


def test_repeated_guidance_caps_total_correction_and_preserves_rng():
    from vibewarp.core.disco_lora import guided_diffusion
    from vibewarp.core.disco_repeated import install_multi_guidance
    from vibewarp.core.disco_guidance import guidance_noise_cap
    d=guided_diffusion([699,582,466,349,233,116,-1])
    install_multi_guidance(d,4,False,4)
    model=lambda x,t,**kw:torch.cat([torch.zeros_like(x),torch.zeros_like(x)],1)
    x=torch.randn(1,3,8,8); t=torch.tensor([3]); states=[]
    def guidance(x,t,**kw):
        states.append(torch.rand(()))
        return torch.ones_like(x)*100
    torch.manual_seed(100)
    with guidance_noise_cap(d,10):
        actual=d.ddim_sample(model,x,t,eta=.55,cond_fn=guidance)['sample']
    next_value=torch.rand(())
    torch.manual_seed(100)
    for _ in range(4):torch.rand(())
    baseline=d.ddim_sample(model,x,t,eta=.55,cond_fn=None)['sample']
    assert torch.equal(torch.rand(()),next_value)
    rms=(actual-baseline).square().mean().sqrt()
    assert rms<=float(d.sqrt_one_minus_alphas_cumprod[3])*.10001
    assert len(states)==4


def test_source_fps_and_frame_extraction_cadence(monkeypatch):
    from vibewarp.pipeline import resolve_output_fps
    c=RunConfig();c.video.extract_nth_frame=2
    class Capture:
        def __init__(self,path):pass
        def get(self,key):return 30
        def release(self):pass
    monkeypatch.setattr('cv2.VideoCapture',Capture)
    assert resolve_output_fps(c)==15
    c.video_assembly.fps=24
    assert resolve_output_fps(c)==24


def test_pixel_temporal_correction_cannot_spread_outside_trusted_latents(monkeypatch):
    import sys
    import torch.nn.functional as F
    from vibewarp.core.diffusion import make_temporal_guidance_model_fn
    class Spatial(torch.nn.Module):
        def forward(self,x,y):return (x-y).square().mean(1,keepdim=True)
    monkeypatch.setitem(sys.modules,'lpips',SimpleNamespace(LPIPS=lambda **kw:Spatial()))
    monkeypatch.setattr('vibewarp.core.diffusion._differentiable_decode_half',
        lambda model,x:F.avg_pool2d(x[:,:3],3,stride=1,padding=1))
    x=torch.randn(1,4,8,8);mask=torch.zeros(8,8);mask[:,:4]=1
    wrapped=make_temporal_guidance_model_fn(lambda image,sigma,**kw:image.clone(),sd_model=None,
        target_image=torch.zeros(1,3,16,16),target_latent=None,init_scale=1,init_latent_scale=0,
        clamp_grad=False,clamp_max=1,add_noise=False,guidance_start_code=None,
        source_label='prev warped + cc',trusted_mask=mask)
    output=wrapped(x,torch.ones(1))
    assert torch.equal(output[...,4:],x[...,4:])
    assert not torch.equal(output[:,:3,:,:4],x[:,:3,:,:4])
