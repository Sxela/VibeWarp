"""Regression coverage for the saved six-step quality/turbo notebook export."""
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from vibewarp.config import RunConfig
from vibewarp.config_io import apply_path_defaults, config_from_settings, config_schema, validate_config
from vibewarp.settings import settings_to_dict
from vibewarp.core.disco_optimizations import SecondaryPrecision, grouped_clip_gradient
from vibewarp.core.disco_secondary import DiffusionOutput
from vibewarp.video.padding import crop_visible, prepare_padded_frames, resolve_padded_geometry


def test_uppercase_notebook_export_imports_execution_modes_and_roundtrips(tmp_path):
    export = dict(diffusion_model='512x512_diffusion_uncond_finetune_008100',
        steps=6, skip_steps=2, max_frames=1, width=1088, height=1920,
        USE_TURBO_LORA=True, USE_QUALITY_LORA=True, LORA_PATH='turbo.safetensors',
        QUALITY_LORA_PATH='quality.safetensors', LORA_STRENGTH=.5, QUALITY_LORA_STRENGTH=1.,
        CLIP_GRAD_STEPS=4, CLIP_DIVIDE_CUTS=False, cutn_batches=4,
        SPEED_MERGE_LORAS=True, SPEED_SECONDARY_PRECISION='auto', SPEED_CUT_BATCH_GROUP=2,
        PAD_TO_MULTIPLE_64=True, _visible_size=[1080,1920], _canvas_size=[1088,1920],
        _effective_grid=[999,832,665,499,332,165,-1], FRAME_LOSS_CC_TRUSTED_ONLY=True,
        GUIDANCE_NOISE_CAP=False, GUIDANCE_CAP_PERCENT=10.,
        FIRST_FRAME_PATH='', EXPORT_CROP_TO_VISIBLE=True, EXPORT_KEEP_AUDIO=False,
        lora_config={'grid':[999,749,499,249,-1]}, ViTB32=True, eta=.55,
        text_prompts={0:['a painting']}, init_scale=0, frames_scale=30000)
    path=tmp_path/'notebook.txt'; path.write_text(json.dumps(export))
    c=config_from_settings(path)
    assert c.disco.sampler=='distilled' and c.disco.lora_strength==.5
    assert c.disco.quality_lora_enabled and c.disco.quality_lora_strength==1
    assert c.disco.clip_grad_steps==4 and not c.disco.clip_divide_cuts
    assert c.disco.merge_loras and c.disco.secondary_precision=='auto'
    assert c.disco.clip_precision=='fp16'
    assert c.disco.cut_batch_group==2 and c.disco.mask_frame_guidance
    assert c.disco.guidance_noise_cap_percent==0
    assert c.disco.timestep_grid==export['_effective_grid']
    assert c.frame_range==[0] and c.disco.pad_to_multiple_64
    assert resolve_padded_geometry(c)==((1080,1920),(1088,1920))
    assert c.video_assembly.crop_to_visible and not c.video_assembly.keep_audio
    assert not validate_config(c,require_inputs=False)
    native=tmp_path/'native.txt';native.write_text(json.dumps(settings_to_dict(c)))
    assert json.loads(json.dumps(settings_to_dict(config_from_settings(native))))==json.loads(json.dumps(settings_to_dict(c)))


def test_explicit_disable_and_zero_override_stale_stack_metadata(tmp_path):
    path=tmp_path/'off.txt'
    path.write_text(json.dumps(dict(diffusion_model='512',steps=6,USE_TURBO_LORA=False,
        USE_QUALITY_LORA=False,LORA_STRENGTH=0,QUALITY_LORA_STRENGTH=0,
        CLIP_GRAD_STEPS=4,clip_grad_steps=2,SPEED_MERGE_LORAS=True,merge_loras=False,
        lora_config={'turbo_path':'old-turbo','quality_path':'old-quality','turbo_strength':1})))
    c=config_from_settings(path)
    assert c.disco.sampler=='ddim' and not c.disco.quality_lora_enabled
    assert c.disco.lora_strength==0 and c.disco.quality_lora_strength==0
    assert c.disco.clip_grad_steps==2 and not c.disco.merge_loras


@pytest.mark.parametrize('field,value', [('cut_batch_group',0),('cut_batch_group',True),
    ('secondary_precision','bad'),('clip_precision','bf16')])
def test_execution_mode_validation(field,value):
    c=RunConfig(model_version='disco_512');setattr(c.disco,field,value)
    assert any(e['path']==f'disco.{field}' for e in validate_config(c,require_inputs=False))


def test_controls_are_available_in_model_filtered_ui_schema():
    s=config_schema()['properties']
    assert s['disco']['properties']['secondary_precision']['choices']==['auto']
    assert s['disco']['properties']['secondary_precision']['tier']=='hidden'
    assert s['disco']['properties']['cut_batch_group']['group']=='Cutouts'
    assert s['disco']['properties']['pad_to_multiple_64']['tier']=='advanced'
    assert s['disco']['properties']['clip_precision']['tier']=='hidden'
    assert s['disco']['properties']['clip_precision']['choices']==['fp16']
    assert s['video']['properties']['visible_width']['tier']=='advanced'
    assert s['video_assembly']['properties']['crop_to_visible']['tier']=='advanced'


@pytest.mark.parametrize('name',['ViT-B-32','ViT-B-16','ViT-L-14','RN50','RN101'])
@pytest.mark.parametrize('requested',[None,'fp32','auto'])
def test_classic_clip_always_loads_fp16_despite_legacy_precision(monkeypatch,name,requested):
    import sys
    from vibewarp.core.disco import GuidanceEncoder
    calls=[]
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.visual=SimpleNamespace(image_size=224)
        def to(self,device):
            calls.append(('device',device));return self
    def load(path,**kwargs):
        calls.append(('load',kwargs));return Model()
    monkeypatch.setitem(sys.modules,'open_clip',SimpleNamespace(get_tokenizer=lambda n:None,
        get_pretrained_cfg=lambda *a:dict(mean=[0,0,0],std=[1,1,1])))
    monkeypatch.setitem(sys.modules,'open_clip.openai',SimpleNamespace(load_openai_model=load))
    encoder=GuidanceEncoder(name,Path('models/disco'),'cuda',requested)
    assert calls==[('load',{'device':'cpu','precision':'fp16'}),('device','cuda')]
    assert encoder.precision=='fp16'


@pytest.mark.parametrize('name',['siglip2-base','siglip2-large'])
def test_siglip_loads_fp16_weights_directly_and_ignores_legacy_precision(monkeypatch,name):
    import sys
    from vibewarp.core.disco import GuidanceEncoder
    calls=[]
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.ones(1))
            self.config=SimpleNamespace(vision_config=SimpleNamespace(image_size=224))
        def to(self,**kwargs):
            calls.append(('to',kwargs));return super().to(dtype=kwargs['dtype'])
    def load(source,**kwargs):
        calls.append(('load',kwargs));return Model()
    processor=SimpleNamespace(tokenizer=None,image_processor=SimpleNamespace(
        image_mean=[.5]*3,image_std=[.5]*3))
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(
        AutoModel=SimpleNamespace(from_pretrained=load),
        AutoProcessor=SimpleNamespace(from_pretrained=lambda source:processor)))
    encoder=GuidanceEncoder(name,Path('models/disco'),'cuda','fp32')
    assert calls==[('load',{'torch_dtype':torch.float16}),
        ('to',{'device':'cuda','dtype':torch.float16})]
    assert encoder.precision=='fp16' and encoder.model.weight.dtype==torch.float16


@pytest.mark.parametrize('saved',['fp32','auto','fp16','bf16'])
def test_legacy_settings_cannot_restore_another_encoder_precision(tmp_path,saved):
    from vibewarp.config_io import config_from_dict
    c=config_from_dict({'disco':{'clip_precision':saved}})
    assert c.disco.clip_precision=='fp16'
    c.disco.clip_precision=saved
    assert apply_path_defaults(c).disco.clip_precision=='fp16'
    path=tmp_path/'legacy.txt'
    path.write_text(json.dumps({'diffusion_model':'512','clip_precision':saved,'CLIP_PRECISION':saved}))
    assert config_from_settings(path).disco.clip_precision=='fp16'


def test_secondary_bf16_matches_autocast_and_keeps_input_gradient():
    class Secondary(torch.nn.Module):
        def __init__(self):
            super().__init__();self.conv=torch.nn.Conv2d(3,3,1)
        def forward(self,x,t):
            value=self.conv(x)
            return DiffusionOutput(value,value*.3,value*.5)
    torch.manual_seed(9)
    model=Secondary().eval().requires_grad_(False)
    x=torch.randn(1,3,8,8,requires_grad=True);t=torch.zeros(1)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        expected=model(x,t).pred.float()
    actual=SecondaryPrecision(model,'bf16','cpu')(x,t)
    assert actual.v.dtype==actual.pred.dtype==actual.eps.dtype==torch.float32
    assert torch.equal(actual.pred,expected)
    expected_grad=torch.autograd.grad(expected.sum(),x)[0]
    actual_grad=torch.autograd.grad(actual.pred.sum(),x)[0]
    assert torch.equal(actual_grad,expected_grad) and actual_grad.abs().sum()>0
    assert SecondaryPrecision(model,'auto','cpu').precision=='bf16'
    assert all(parameter.dtype==torch.bfloat16 for parameter in model.parameters())
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize('saved',['fp32','fp16','bf16','auto'])
def test_saved_secondary_precision_cannot_override_device_policy(tmp_path,saved):
    from vibewarp.config_io import config_from_dict
    config=config_from_dict({'disco':{'secondary_precision':saved}})
    assert config.disco.secondary_precision=='auto'
    config.disco.secondary_precision=saved
    assert apply_path_defaults(config).disco.secondary_precision=='auto'
    path=tmp_path/'notebook.txt'
    path.write_text(json.dumps({'diffusion_model':'512','secondary_precision':saved}))
    assert config_from_settings(path).disco.secondary_precision=='auto'


@pytest.mark.parametrize(('supported','expected'),[(True,'bf16'),(False,'fp16')])
@pytest.mark.parametrize('requested',['fp32','fp16','bf16','auto'])
def test_secondary_policy_uses_the_target_cuda_device_without_fp32(monkeypatch,supported,expected,requested):
    from contextlib import nullcontext
    devices=[];checks=[]
    monkeypatch.setattr(torch.cuda,'device',lambda device:devices.append(device) or nullcontext())
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda **kwargs:checks.append(kwargs) or supported)
    model=torch.nn.Conv2d(3,3,1).requires_grad_(False)
    wrapped=SecondaryPrecision(model,requested,'cuda:1')
    assert wrapped.precision==expected
    assert devices==[torch.device('cuda:1')]
    assert checks==[{'including_emulation':False}]
    assert all(parameter.dtype==(torch.bfloat16 if supported else torch.float16) for parameter in model.parameters())


@pytest.mark.parametrize('modern',[False,True])
def test_encoders_cast_normalized_input_and_keep_fp32_loss_gradients(modern):
    from vibewarp.core.disco import GuidanceEncoder
    encoder=GuidanceEncoder.__new__(GuidanceEncoder)
    encoder.modern=modern;encoder.mean=[0,0,0];encoder.std=[1,1,1]
    def encode(image):
        assert image.dtype==torch.float16
        return image.mean((2,3))
    encoder.model=SimpleNamespace(visual=SimpleNamespace(conv1=SimpleNamespace(
        weight=torch.zeros(1,dtype=torch.float16))),encode_image=encode,
        get_image_features=lambda pixel_values:encode(pixel_values))
    x=torch.rand(1,3,8,8,requires_grad=True)
    result=encoder.image(x)
    assert result.dtype==torch.float32
    grad=torch.autograd.grad(result.sum(),x)[0]
    assert grad.dtype==torch.float32 and torch.isfinite(grad).all() and grad.abs().sum()>0


@pytest.mark.parametrize('group',[1,2,3,8])
def test_grouped_clip_preserves_gradient_weight_and_cutout_rng(group):
    torch.manual_seed(21)
    image=torch.randn(1,3,8,8,requires_grad=True)
    target=torch.nn.functional.normalize(torch.randn(2,3),dim=-1)
    weights=torch.tensor([.75,.25])
    class Encoder:
        def __init__(self):self.batch_sizes=[]
        def image(self,cuts):
            self.batch_sizes.append(len(cuts))
            return torch.nn.functional.normalize(cuts.mean((2,3)),dim=-1)
    def cuts(index):
        return torch.cat([image*torch.rand(())+1,image.square()+torch.rand(())])
    def run(group):
        torch.manual_seed(42);encoder=Encoder()
        grad=grouped_clip_gradient(image,encoder,target,weights,cuts,batches=5,group=group,scale=7)
        return grad,torch.get_rng_state(),encoder.batch_sizes
    expected,state,_=run(1);actual,actual_state,sizes=run(group)
    torch.testing.assert_close(actual,expected,atol=2e-7,rtol=2e-5)
    assert torch.equal(state,actual_state)
    assert len(sizes)==(5+group-1)//group and sum(sizes)==10


def test_padding_matches_notebook_jpeg_bytes_and_resume_uses_raw_cache(tmp_path):
    source=tmp_path/'raw';source.mkdir();output=tmp_path/'padded'
    path=source/'000001.jpg'
    Image.fromarray(np.arange(20*30*3,dtype=np.uint8).reshape(30,20,3)).save(path)
    raw=path.read_bytes();visible=(27,51);canvas=(64,64)
    prepare_padded_frames(source,output,visible,canvas)
    with Image.open(path) as image:
        pixels=np.asarray(image.convert('RGB').resize(visible,Image.Resampling.LANCZOS))
    expected=np.pad(pixels,((6,7),(18,19),(0,0)),mode='edge')
    stream=io.BytesIO();Image.fromarray(expected).save(stream,format='JPEG',quality=95)
    target=output/path.name
    assert target.read_bytes()==stream.getvalue() and path.read_bytes()==raw
    stamp=target.stat().st_mtime_ns
    prepare_padded_frames(source,output,visible,canvas)
    assert target.stat().st_mtime_ns==stamp
    # Changing geometry must start from the untouched raw JPEG.
    prepare_padded_frames(source,output,(32,48),canvas)
    assert target.read_bytes()!=stream.getvalue() and path.read_bytes()==raw
    target.write_bytes(b'corrupt')
    prepare_padded_frames(source,output,visible,canvas)
    assert target.read_bytes()==stream.getvalue()


def test_padded_manual_and_source_aspect_geometry(monkeypatch):
    c=RunConfig(model_version='disco_512');c.disco.pad_to_multiple_64=True
    c.video.width,c.video.height=1080,1920
    apply_path_defaults(c)
    assert resolve_padded_geometry(c)==((1080,1920),(1088,1920))
    c.video.visible_width=c.video.visible_height=0;c.video.max_size=1920
    import importlib
    monkeypatch.setattr(importlib.import_module('vibewarp.video.input'), 'probe_video',
                        lambda p:dict(width=720,height=1280))
    assert resolve_padded_geometry(c)==((1080,1920),(1088,1920))


def test_visible_crop_and_one_frame_limit_do_not_modify_canvas(tmp_path):
    from vibewarp.core.diffusion import _resolve_frame_range,RenderContext
    image=Image.fromarray(np.arange(64*64*3,dtype=np.uint8).reshape(64,64,3))
    assert np.array_equal(np.asarray(crop_visible(image,(27,51))),np.asarray(image)[6:57,18:45])
    assert image.size==(64,64)
    c=RunConfig(frame_range=[0])
    for n in range(5):Image.new('RGB',(8,8)).save(tmp_path/f'{n+1:06d}.jpg')
    ctx=RenderContext(config=c,video_frames_folder=str(tmp_path))
    assert _resolve_frame_range(ctx,None,0)==(0,1)
    assert _resolve_frame_range(ctx,None,1)==(1,1)
    # Editing the range fully replaces the imported one-frame selection.
    c.frame_range=[0,10]
    assert _resolve_frame_range(ctx,None,0)==(0,11)


def test_video_export_crops_before_upscale_and_keeps_rendered_png(tmp_path,monkeypatch):
    from vibewarp.video.output import create_video
    frame=tmp_path/'test(0)_000000.png';Image.new('RGB',(64,64),'red').save(frame)
    captured=[]
    def upscale(model,image,ratio):
        captured.append(image.size);return image.resize((54,102))
    monkeypatch.setattr('vibewarp.video.output.load_realesrgan',lambda *a:object())
    monkeypatch.setattr('vibewarp.video.output.upscale_frame',upscale)
    monkeypatch.setattr('vibewarp.video.output.subprocess.run',lambda *a,**kw:SimpleNamespace(returncode=0,stderr=''))
    create_video(str(tmp_path),str(tmp_path/'out.mp4'),batch_name='test',ffmpeg_path='fake',
        visible_size=(27,51),upscale_ratio=2)
    assert captured==[(27,51)] and Image.open(frame).size==(64,64)


@pytest.mark.parametrize('blend_mode', ['linear', 'optical flow'])
@pytest.mark.parametrize('upscale_ratio', [1, 2])
def test_export_geometry_follows_original_frame_pair_blending(tmp_path, monkeypatch, blend_mode, upscale_ratio):
    from vibewarp.video.output import create_video
    flow_dir = tmp_path/'flow'
    flow_dir.mkdir()
    for index, value in enumerate((40, 120, 200)):
        stem = f'test(0)_{index:06d}'
        Image.new('RGB', (64, 64), (value, value, value)).save(tmp_path/f'{stem}.png')
        np.save(flow_dir/f'{stem}.npy', np.zeros((64, 64, 2), dtype=np.float32))
    monkeypatch.setattr('vibewarp.video.output.load_realesrgan', lambda *a: object())
    monkeypatch.setattr('vibewarp.video.output.upscale_frame',
                        lambda model, image, ratio: image.resize((image.width*ratio, image.height*ratio)))
    monkeypatch.setattr('vibewarp.video.output.subprocess.run',
                        lambda *a, **kw: SimpleNamespace(returncode=0, stderr=''))
    create_video(str(tmp_path), str(tmp_path/'out.mp4'), batch_name='test', ffmpeg_path='fake',
                 blend_mode=blend_mode, blend=.5, flow_dir=str(flow_dir), check_consistency=False,
                 visible_size=(32, 48), upscale_ratio=upscale_ratio)
    # Notebook uses the original previous render: (120 + 200) / 2 = 160.
    for index, expected in enumerate((40, 80, 160)):
        with Image.open(tmp_path/'blend_temp'/f'{index:06d}.png') as image:
            assert image.size == (32*upscale_ratio, 48*upscale_ratio)
            assert image.getpixel((image.width//2, image.height//2)) == (expected,)*3
        assert Image.open(tmp_path/f'test(0)_{index:06d}.png').size == (64, 64)


@pytest.mark.parametrize('value,expected', [
    (.2, {0: .2}), ([.2], {0: .2}), ([.2, .4], {0: .2, 1: .4}),
    ({'0': .2, '60': .4}, {0: .2, 60: .4}), (None, None),
])
def test_grayscale_schedule_normalizes_to_progress_dict_and_roundtrips(tmp_path, value, expected):
    from vibewarp.config_io import config_from_dict
    c = config_from_dict({'model_version': 'disco_512', 'disco': {'cut_schedule_version':2, 'cut_icgray_schedule': value}})
    assert c.disco.cut_icgray_schedule == expected
    assert not validate_config(c, require_inputs=False)
    path = tmp_path/'native.txt'
    path.write_text(json.dumps(settings_to_dict(c)))
    assert config_from_settings(path).disco.cut_icgray_schedule == expected
    schema = config_schema()['properties']['disco']['properties']['cut_icgray_schedule']
    assert schema['type'] == 'object' and schema['schedule_domain'] == 'inference_progress'


@pytest.mark.parametrize('schedule', [{}, {0: -0.1}, {0: 1.1}, {101: .2}, {0: float('nan')}])
def test_grayscale_schedule_rejects_invalid_progress_and_probability(schedule):
    c = RunConfig(model_version='disco_512')
    c.disco.cut_icgray_schedule = schedule
    assert any(e['path'] == 'disco.cut_icgray_schedule' for e in validate_config(c, require_inputs=False))
