from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import re
import time

import numpy as np
import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient

from vibewarp.config import RunConfig
from vibewarp.config_io import config_schema, validate_config
from vibewarp.video import raw_processing as rp


def config(tmp_path):
    c=RunConfig(root_dir=str(tmp_path),model_path=str(tmp_path/'models'))
    video=tmp_path/'video.mp4';video.write_bytes(b'video-content')
    c.video.video_init_path=str(video)
    return c


def test_tint_soft_alpha_and_two_independent_opacities_preserve_input():
    raw=torch.full((1,3,1,3),0.4);before=raw.clone()
    alpha=torch.tensor([[[[0.,0.5,1.]]]])
    result=rp.tint_tensor(raw,alpha,[255,0,0],[0,0,255],0.6,0.2)
    fg=torch.tensor([1.,0.,0.]).view(1,3,1,1)
    bg=torch.tensor([0.,0.,1.]).view(1,3,1,1)
    expected=alpha*torch.lerp(raw,fg,0.6)+(1-alpha)*torch.lerp(raw,bg,0.2)
    torch.testing.assert_close(result,expected)
    assert torch.equal(raw,before)
    assert torch.equal(rp.tint_tensor(raw,alpha,[255]*3,[0]*3,0,0),raw)


@pytest.mark.parametrize('method',['mean','LAB','PDF'])
def test_color_kernels_preserve_input_and_strength_zero(method):
    torch.manual_seed(77)
    raw=torch.rand(1,3,12,12)*0.6+0.1;ref=torch.rand_like(raw)*0.4+0.4
    original=raw.clone()
    assert torch.equal(rp.color_tensor(raw,ref,method,0),raw)
    result=rp.color_tensor(raw,ref,method,1)
    assert result.shape==raw.shape and torch.isfinite(result).all()
    assert result.min()>=0 and result.max()<=1
    assert torch.equal(raw,original)
    if method=='mean':
        torch.testing.assert_close(result.mean((-2,-1)),ref.mean((-2,-1)),atol=0.005,rtol=0)


def test_lab_roundtrip():
    torch.manual_seed(7);rgb=torch.rand(1,3,17,13)
    torch.testing.assert_close(rp.lab_to_rgb(rp.rgb_to_lab(rgb)),rgb,atol=5e-6,rtol=1e-5)


def test_cache_identity_excludes_render_settings_and_tracks_video_content(tmp_path):
    c=config(tmp_path);a=rp.VideoCache(c)
    c.mask.foreground_opacity=1;c.video.width=128;c.frame_range=[5,10]
    assert rp.VideoCache(c).root==a.root
    c.video.extract_nth_frame=2
    assert rp.VideoCache(c).root!=a.root
    c.video.extract_nth_frame=1
    Path(c.video.video_init_path).write_bytes(b'changed-video')
    assert rp.VideoCache(c).root!=a.root


def test_gpu_extraction_only_generates_missing_overlaps(tmp_path,monkeypatch):
    c=config(tmp_path);c.video.extract_nth_frame=2
    cache=rp.VideoCache(c);calls=[]
    def extract(cmd,**kwargs):
        assert cmd[cmd.index('-hwaccel')+1]=='cuda'
        # Frames download in their native layout; nv12 would reject 10-bit/4:4:4.
        assert '-hwaccel_output_format' not in cmd and 'hwdownload' not in cmd[cmd.index('-vf')+1]
        first,last=map(int,re.search(r'between\(n\\,(\d+)\\,(\d+)\)',cmd[cmd.index('-vf')+1]).groups())
        calls.append((first,last));number=int(cmd[cmd.index('-start_number')+1])
        for n in range((last-first)//2+1):
            Image.new('RGB',(8,8),'red').save(Path(cmd[-1]).parent/f'{number+n:06d}.jpg')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(rp.subprocess,'run',extract)
    cache.ensure_frames(0,5);preserved=cache.frame_path(3).read_bytes()
    cache.ensure_frames(2,4);cache.ensure_frames(3,8)
    assert calls==[(0,10),(12,16)]
    assert cache.frame_path(3).read_bytes()==preserved
    assert all(cache.frame_path(i).exists() for i in range(9))


@pytest.mark.parametrize('tint,mask',[(True,False),(False,True),(True,True)])
def test_rvm_cache_restores_recurrence_and_reuses_subset(tmp_path,monkeypatch,tint,mask):
    c=config(tmp_path);c.mask.rvm_enabled=tint;c.mask.use_background_mask=mask
    model=tmp_path/'rvm.torchscript';model.write_bytes(b'model');c.mask.rvm_model_path=str(model)
    cache=rp.VideoCache(c);calls=[]
    def ensure(start,end):
        for i in range(start,end+1):
            if not cache.frame_path(i).exists():Image.new('RGB',(8,8),'red').save(cache.frame_path(i))
    cache.ensure_frames=ensure
    monkeypatch.setattr(rp,'require_cuda',lambda:None)
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self:self)
    original=rp.image_tensor
    monkeypatch.setattr(rp,'image_tensor',lambda image:original(image,'cpu'))
    class Model:
        def eval(self):return self
        def __call__(self,src,*args):
            rec=args[0];n=0 if rec is None else rec.item()
            calls.append(n)
            return src,torch.full((1,1,8,8),(n+1)/100),*[torch.tensor(n+1)]*4
    monkeypatch.setattr(torch.jit,'load',lambda *a,**kw:Model())
    cache.prepare(0,3)
    saved=(cache.alpha/'000002.npy').read_bytes()
    cache.prepare(1,2)
    assert calls==[0,1,2,3]
    cache.prepare(2,5)
    assert calls==[0,1,2,3,4,5]
    assert np.load(cache.alpha/'000005.npy')[0,0]==pytest.approx(0.06,abs=1e-4)
    assert (cache.alpha/'000002.npy').read_bytes()==saved
    c.mask.foreground_color=[200,0,0]
    cache.prepare(0,5)
    assert len(calls)==6


@pytest.mark.parametrize('name,value',[
    ('foreground_opacity',1.1),('background_opacity',-0.1),('raw_color_strength',float('nan')),
    ('rvm_downsample_ratio',0),('foreground_color',[1,2]),('background_color',[256,0,0]),
    ('raw_color_method','bad'),('raw_color_reference','bad')])
def test_invalid_preparation_settings_are_rejected(name,value):
    c=RunConfig();setattr(c.mask,name,value)
    assert any(e['path']=='mask.'+name for e in validate_config(c,require_inputs=False,check_paths=False))


def test_schema_exposes_all_model_controls_with_separate_raw_color_method():
    p=config_schema()['properties']['mask']['properties']
    for name in ('rvm_enabled','foreground_color','background_color'):
        assert 'model_families' not in p[name]
        assert p[name]['group']=='Background masking'
        assert p[name]['workflows']==['background']
        assert p[name]['preview']=='raw_video'
    assert p['foreground_color']['rgb_color']
    for name in ('foreground_opacity','background_opacity'):
        assert p[name]['slider']=={'min':0,'max':1,'step':0.01,'scale':1}
    assert p['raw_color_method']['choices']==['PDF','LAB','mean']
    tab=next(tab for tab in config_schema()['workflows'] if tab['id']=='background')
    assert tab['label']=='Mask & tint' and tab['groups'][0]=='Background masking'
    for field in p.values():
        if field['tier'] not in ('system','hidden'):
            assert field['workflows']==(['color'] if field.get('group')=='Raw input color match' else ['background'])


def test_preview_queues_on_shared_gpu_worker_and_returns_image(tmp_path,monkeypatch):
    from vibewarp.web import create_app
    from vibewarp.web_jobs import JobManager
    c=config(tmp_path);c.mask.rvm_enabled=True
    image=tmp_path/'preview.png';Image.new('RGB',(8,8),'blue').save(image)
    seen=[]
    monkeypatch.setattr(rp,'preview_frame',lambda conf,frame,reference:seen.append((frame,reference)) or str(image))
    jobs=JobManager(runner=lambda *a,**kw:pytest.fail('Preview must not start diffusion'))
    client=TestClient(create_app(jobs))
    response=client.post('/api/video/raw-preview',json={'config':asdict(c),'frame':4})
    assert response.status_code==202
    body=response.json();job=jobs.get(body['id'])
    deadline=time.monotonic()+5
    while job.state not in ('completed','failed') and time.monotonic()<deadline:time.sleep(0.01)
    assert job.state=='completed' and job.operation=='raw_preview'
    assert seen==[(4,'')]
    assert client.get(f'/api/jobs/{job.id}/preview').status_code==200
    assert not list(tmp_path.glob('images_out/*'))


def test_production_processing_rejects_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='require CUDA'):rp.require_cuda()


def test_tint_mask_matches_padded_visible_geometry_without_changing_cached_alpha(tmp_path,monkeypatch):
    c=config(tmp_path);c.model_version='disco_512';c.disco.pad_to_multiple_64=True
    c.video.visible_width=2;c.video.visible_height=4
    c.mask.rvm_enabled=True;c.mask.foreground_opacity=1;c.mask.foreground_color=[255,0,0]
    folder=tmp_path/'alpha';folder.mkdir()
    np.save(folder/'000000.npy',np.array([[0,1],[0,1]],dtype=np.float16))
    before=(folder/'000000.npy').read_bytes()
    monkeypatch.setattr(rp,'require_cuda',lambda:None)
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self:self)
    original=rp.image_tensor
    monkeypatch.setattr(rp,'image_tensor',lambda image:original(image,'cpu'))
    result=np.array(rp.process_image(Image.new('RGB',(4,4),'black'),c,SimpleNamespace(alpha=folder),0))
    assert np.all(result[:,0]==0) and np.all(result[:,-1,0]==255)
    assert (folder/'000000.npy').read_bytes()==before


def test_raw_preparation_runs_before_consistency_blending_and_receives_warped_palette():
    from vibewarp.flow.warp import warp_frame
    raw=Image.new('RGB',(16,16),'blue');stylized=Image.new('RGB',(16,16),'red')
    seen=[]
    def process(image,reference):
        seen.append((image.getpixel((8,8)),reference.getpixel((8,8))))
        return Image.new('RGB',image.size,'green')
    result,warped=warp_frame(stylized,raw,np.zeros((16,16,2),np.float32),
        weights=np.zeros((16,16,3),np.float32),blend=1,pad_pct=0.1,
        raw_transform=process,return_warped=True)
    assert seen==[((0,0,255),(255,0,0))]
    assert result.getpixel((8,8))==(0,128,0)
    assert warped.getpixel((8,8))==(255,0,0)
    assert raw.getpixel((8,8))==(0,0,255)


def test_render_mask_uses_the_same_visible_geometry_and_immutable_rvm_alpha(tmp_path):
    c=config(tmp_path);c.model_version='disco_512';c.disco.pad_to_multiple_64=True
    c.video.visible_width=2;c.video.visible_height=4
    c.mask.use_background_mask=True;c.mask.rvm_enabled=False
    c.mask.background='color';c.mask.background_source='black'
    folder=tmp_path/'alpha';folder.mkdir()
    np.save(folder/'000003.npy',np.array([[0,1],[0,1]],np.float16))
    before=(folder/'000003.npy').read_bytes()
    result=np.array(rp.apply_cached_mask(Image.new('RGB',(4,4),'white'),c,
        SimpleNamespace(alpha=folder),3,str(tmp_path/'frames')))
    assert np.all(result[:,:2]==0) and np.all(result[:,2:]==255)
    assert (folder/'000003.npy').read_bytes()==before


@pytest.mark.parametrize('version',['control_multi_v15','control_multi_sdxl','disco_512','flux2_klein_edit',
                                  'hidream_o1_edit','qwen_image_edit_2511','mage_flow_edit'])
def test_first_raw_init_uses_prepared_copy_for_every_renderer(tmp_path,monkeypatch,version):
    from vibewarp.core import diffusion
    c=config(tmp_path);c.model_version=version;c.mask.rvm_enabled=True
    c.video.width=c.video.height=16
    frames=tmp_path/'frames';frames.mkdir();Image.new('RGB',(16,16),'blue').save(frames/'000001.jpg')
    ctx=diffusion.RenderContext(config=c,video_frames_folder=str(frames),batch_folder=str(tmp_path))
    state=diffusion.FrameState();seen=[]
    original=(frames/'000001.jpg').read_bytes()
    monkeypatch.setattr(rp,'render_raw',lambda image,ctx,frame,*a:Image.new('RGB',image.size,'red'))
    def render(ctx,state):
        with Image.open(state.init_image) as opened:
            seen.append(opened.getpixel((8,8)));return opened.convert('RGB')
    monkeypatch.setattr(diffusion,'render_frame',render)
    diffusion._render_single_frame(ctx,state,0,0,str(tmp_path),'png')
    assert seen==[(255,0,0)]
    assert (frames/'000001.jpg').read_bytes()==original


@pytest.mark.parametrize('mode,warped,expected',[
    ('init stylized',True,'red'),('previous warped',True,'green'),
    ('previous warped',False,'blue')])
def test_raw_color_reference_is_selected_independently(tmp_path,monkeypatch,mode,warped,expected):
    c=config(tmp_path);c.mask.raw_color_enabled=True;c.mask.raw_color_reference=mode
    c.color.after_enabled=True;c.color.after_method='PDF';c.mask.raw_color_method='mean'
    c.video.width=c.video.height=8
    Image.new('RGB',(8,8),'red').save(tmp_path/f'{c.batch_name}(0)_000000.png')
    Image.new('RGB',(8,8),'blue').save(tmp_path/f'{c.batch_name}(0)_000001.png')
    ctx=SimpleNamespace(config=c,batch_folder=str(tmp_path),raw_video_cache=None,_sequence_start_frame=0)
    seen=[]
    def process(image,config,cache,frame,reference,**kwargs):
        seen.append(reference.getpixel((0,0)))
        return image
    monkeypatch.setattr(rp,'process_image',process)
    rp.render_raw(Image.new('RGB',(8,8)),ctx,2,Image.new('RGB',(8,8),'green') if warped else None)
    assert seen==[Image.new('RGB',(1,1),expected).getpixel((0,0))]
    assert c.color.after_method=='PDF' and c.mask.raw_color_method=='mean'


def test_raw_guidance_does_not_overwrite_the_prepared_init(tmp_path,monkeypatch):
    from vibewarp.core import diffusion
    c=config(tmp_path);c.model_version='control_multi_v15';c.mask.raw_color_enabled=True
    c.diffusion.guidance_mode='raw frame';c.video.width=c.video.height=16
    frames=tmp_path/'frames';frames.mkdir();Image.new('RGB',(16,16),'blue').save(frames/'000001.jpg')
    ctx=diffusion.RenderContext(config=c,video_frames_folder=str(frames),batch_folder=str(tmp_path))
    state=diffusion.FrameState();state.prev_frame=Image.new('RGB',(16,16),'white')
    # Init is prepared without a reference, guidance against the previous output.
    monkeypatch.setattr(rp,'render_raw',lambda image,ctx,frame,reference=None:
                        Image.new('RGB',image.size,'red' if reference is None else 'green'))
    seen={}
    def render(ctx,state):
        for name in ('init_image','guidance_image'):
            with Image.open(getattr(state,name)) as opened:
                seen[name]=opened.getpixel((8,8))
        return Image.new('RGB',(16,16))
    monkeypatch.setattr(diffusion,'render_frame',render)
    diffusion._render_single_frame(ctx,state,0,0,str(tmp_path),'png')
    assert seen=={'init_image':(255,0,0),'guidance_image':(0,128,0)}
