import numpy as np
import pytest
import torch
from PIL import Image

from vibewarp.config import RunConfig
from vibewarp.config_io import config_from_dict, validate_config
from vibewarp.core.disco import DiscoBackend, make_cutouts, weighted_prompts
from vibewarp.core.disco_attention import PixelAttention, configure_attention
from vibewarp.render_events import render_event_scope
from vibewarp.vendor.guided_diffusion.unet import QKVAttention, QKVAttentionLegacy, UNetModel
from vibewarp.vendor.guided_diffusion.script_util import create_gaussian_diffusion


@pytest.mark.parametrize('legacy', [True, False])
def test_sdpa_preserves_head_order_outputs_and_image_gradients(legacy):
    torch.manual_seed(12)
    manual = QKVAttentionLegacy(4) if legacy else QKVAttention(4)
    x = torch.randn(2, 3*4*8, 17, requires_grad=True)
    expected = manual(x)
    actual = PixelAttention(4, legacy=legacy)(x)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    gradient = torch.randn_like(actual)
    expected_grad = torch.autograd.grad(expected, x, gradient)[0]
    actual_grad = torch.autograd.grad(actual, x, gradient)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-5)


def test_checkpointed_frozen_unet_still_supports_input_gradient():
    model = UNetModel(image_size=16, in_channels=3, model_channels=32, out_channels=6,
        num_res_blocks=1, attention_resolutions=(1,), channel_mult=(1, 2),
        num_heads=2, use_checkpoint=True).eval().requires_grad_(False)
    configure_attention(model)
    image = torch.randn(1, 3, 16, 16, requires_grad=True)
    output = model(image, torch.tensor([100]))
    grad = torch.autograd.grad(output.square().mean(), image)[0]
    assert grad.shape == image.shape and torch.isfinite(grad).all()
    assert all(p.grad is None for p in model.parameters())


def test_cutouts_are_differentiable_and_seedable():
    image = torch.rand(1, 3, 32, 64, requires_grad=True)
    options = dict(size=16, overview=4, inner=3, power=.5, gray_probability=.2, augment=False)
    first = make_cutouts(image, generator=torch.Generator().manual_seed(12), **options)
    second = make_cutouts(image, generator=torch.Generator().manual_seed(12), **options)
    torch.testing.assert_close(first, second)
    assert first.shape == (7, 3, 16, 16)
    grad = torch.autograd.grad(first.sum(), image)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_cutout_working_cap_bounds_resize_intermediates_and_keeps_encoder_shape(monkeypatch):
    from vibewarp.vendor import resize_right
    original=resize_right.resize;shapes=[]
    def resize(image,**kwargs):
        shapes.append(image.shape[-2:]);return original(image,**kwargs)
    monkeypatch.setattr(resize_right,'resize',resize)
    image=torch.rand(1,3,96,160,requires_grad=True)
    cuts=make_cutouts(image,24,2,3,1.,0.,augment=False,max_size=64,
        generator=torch.Generator().manual_seed(9))
    assert cuts.shape==(5,3,24,24)
    assert shapes and all(max(shape)<=64 for shape in shapes)
    grad=torch.autograd.grad(cuts.sum(),image)[0]
    assert grad.shape==image.shape and torch.isfinite(grad).all() and grad.abs().sum()>0


def test_cutout_native_cap_preserves_values_and_rng():
    image=torch.rand(1,3,48,64)
    def run(**kwargs):
        torch.manual_seed(9)
        result=make_cutouts(image,16,4,3,1.,0.,augment=False,**kwargs)
        return result,torch.get_rng_state()
    a,state=run();b,other=run(max_size=0);c,last=run(max_size=128)
    assert torch.equal(a,b) and torch.equal(a,c)
    assert torch.equal(state,other) and torch.equal(state,last)


def test_cancellation_interrupts_cutout_construction():
    from vibewarp.cancellation import cancellation_scope
    checks=[]
    def cancel():
        checks.append(1);return len(checks)>=3
    with cancellation_scope(cancel),pytest.raises(KeyboardInterrupt):
        make_cutouts(torch.ones(1,3,48,64),16,4,8,1.,0.,augment=False)
    assert len(checks)==3


@pytest.mark.parametrize('cap',[0,32,1024,-1,1,31,32.5,True])
def test_cutout_working_cap_validation(cap):
    c=RunConfig(model_version='disco_512');c.disco.cut_max_size=cap
    errors=validate_config(c,require_inputs=False,check_paths=False)
    assert any(e['path']=='disco.cut_max_size' for e in errors)==(type(cap) is not int or cap not in (0,32,1024))


def test_cancellation_after_encoder_forward_skips_backward_and_later_batches():
    from vibewarp.cancellation import cancellation_scope
    from vibewarp.core.disco_optimizations import grouped_clip_gradient
    from threading import Event
    cancelled=Event();calls=[]
    class Encoder:
        def image(self,cuts):
            calls.append(1);cancelled.set();return cuts.mean((-2,-1))
    image=torch.rand(1,3,8,8,requires_grad=True)
    with cancellation_scope(cancelled.is_set),pytest.raises(KeyboardInterrupt):
        grouped_clip_gradient(image,Encoder(),torch.ones(1,3),torch.ones(1),lambda i:image,batches=10)
    assert calls==[1]


def test_notebook_cutouts_use_zero_padding_and_inclusive_gray_prefix():
    image=torch.ones(1,3,32,64)
    global_view=make_cutouts(image,32,1,0,1.,0.,augment=False)
    assert global_view[0,:,0,0].abs().max()<1e-7
    assert global_view[0,:,16,16].min()>.99
    colored=image*torch.tensor([.2,.5,.9])[None,:,None,None]
    crops=make_cutouts(colored,16,0,4,1.,0.,augment=False,
        generator=torch.Generator().manual_seed(4))
    assert torch.equal(crops[0,0],crops[0,1])
    assert not torch.equal(crops[1,0],crops[1,1])


def test_locked_cutout_augmentations_are_seeded_without_consuming_sampler_rng():
    image=torch.rand(1,3,32,64)
    options=dict(size=16,overview=4,inner=4,power=.5,gray_probability=.2,augment=True)
    generator=torch.Generator().manual_seed(42)
    state=generator.get_state().clone()
    sampler_state=torch.get_rng_state().clone()
    first=make_cutouts(image,generator=generator,**options)
    assert torch.equal(sampler_state,torch.get_rng_state())
    assert not torch.equal(state,generator.get_state())
    torch.manual_seed(999)
    second=make_cutouts(image,generator=torch.Generator().manual_seed(42),**options)
    torch.testing.assert_close(first,second,rtol=0,atol=0)


def test_weighted_prompts_keep_urls_and_negative_weights():
    assert weighted_prompts('painting:2\nhttps://host/image.png:0.4\nblur:-1') == [
        ('painting', 2), ('https://host/image.png', .4), ('blur', -1)]


@pytest.mark.parametrize('sampler', ['ddim', 'plms'])
@pytest.mark.parametrize('tiled', [False, True])
@pytest.mark.parametrize('progressive', [False, True])
@pytest.mark.parametrize('secondary', [False, True])
@pytest.mark.parametrize('noise_cap', [False, True])
def test_small_pixel_backend_samples_with_guidance_and_preview(tmp_path, sampler, monkeypatch, tiled, progressive, secondary, noise_cap):
    # A real Gaussian sampler with cheap differentiable denoiser/encoder doubles.
    class Denoiser(torch.nn.Module):
        def forward(self, x, t, **kwargs):
            return torch.cat([x*.1, torch.zeros_like(x)], dim=1)

    class Encoder:
        size = 8
        def text(self, texts):
            return torch.tensor([[1., 0., 0.] for _ in texts])
        def image(self, image):
            return torch.nn.functional.normalize(image.mean((-1, -2)), dim=-1)

    backend = DiscoBackend.__new__(DiscoBackend)
    backend.model, backend.secondary = Denoiser(), None
    if secondary:
        from types import SimpleNamespace
        backend.secondary = lambda x, t: SimpleNamespace(pred=x*.25)
    backend.diffusion = create_gaussian_diffusion(steps=1000, learn_sigma=True,
        timestep_respacing='ddim4', rescale_timesteps=True)
    backend.encoders, backend.device, backend.lpips = [Encoder()], 'cpu', None
    config = RunConfig(model_version='disco_512')
    config.diffusion.tiled_sampler = tiled
    config.diffusion.sampler_tile_size = 64
    if progressive:
        config.diffusion.sampler_scale_schedule = {0:50, 2:100}
    d = config.disco
    d.mask_frame_guidance = secondary
    d.guidance_noise_cap_percent = .1 if noise_cap else 0
    d.sampler, d.steps, d.init_scale = sampler, 4, 0
    d.cut_overview, d.cut_innercut, d.preview_interval = {0: 1}, {0: 0}, 1
    d.skip_augs = False
    d.cut_icgray_schedule = config_from_dict({
        'disco': {'cut_icgray_schedule': [.2]}}).disco.cut_icgray_schedule
    d.image_scale = 1
    d.guidance_every = 2
    from vibewarp.core.disco import frame_guidance_loss
    frame_losses = []
    def track_loss(*args, **kwargs):
        assert (kwargs['trusted_mask'] is not None) == d.mask_frame_guidance
        frame_losses.append(args[0].shape[-2:])
        return frame_guidance_loss(*args, **kwargs)
    monkeypatch.setattr('vibewarp.core.disco.frame_guidance_loss', track_loss)
    size = 128 if tiled or progressive else 64
    init = Image.new('RGB', (size, size), (80, 100, 120))
    events = []
    with render_event_scope(events.append):
        image = backend.render(config, init, 'red painting', '', 42,
            mask=np.ones((size, size, 3), dtype=np.float32), preview_path=tmp_path/'preview.png')
    assert image.size == (size, size)
    assert events[-1]['step'] == events[-1]['total_steps']
    if noise_cap:
        assert 'guidance_cap_scale' in events[-1]
        assert all(event['guidance_update_rms'] <= event['guidance_noise_budget_rms']+1e-4 for event in events)
    assert not hasattr(backend.diffusion,'guidance_update_cap')
    assert (tmp_path/'preview.png').is_file()
    assert np.isfinite(np.asarray(image)).all()
    assert len(frame_losses) >= len(events)  # also guide steps where CLIP is skipped
    if progressive:
        assert frame_losses[0] == (64,64) and frame_losses[-1] == (128,128)
    else:
        assert all(shape == (size,size) for shape in frame_losses)
    repeated = backend.render(config, init, 'red painting', '', 42,
        mask=np.ones((size, size, 3), dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(image), np.asarray(repeated))


def test_disco_settings_round_trip_and_validation(tmp_path):
    from vibewarp.settings import settings_to_dict, save_settings
    from vibewarp.config_io import config_from_settings
    c = config_from_dict({'model_version':'disco_512', 'disco':{'steps':12},
                          'turbo':{'enabled':True,'interval':4}})
    path = save_settings(settings_to_dict(c), str(tmp_path), 'disco', 0)
    restored = config_from_settings(path)
    assert restored.disco.steps == 12
    assert restored.turbo.enabled and restored.turbo.interval == 4
    assert not validate_config(c, require_inputs=False)
    c.disco.cut_overview, c.disco.cut_innercut = {0: 0}, {0: 0}
    assert any(e['path'] == 'disco.cut_innercut' for e in validate_config(c, require_inputs=False))


def test_siglip_adapter_preserves_image_gradients_with_transformers_outputs():
    from transformers import SiglipConfig, SiglipModel, BatchEncoding
    from vibewarp.core.disco import GuidanceEncoder
    config = SiglipConfig(
        text_config={'vocab_size':16, 'hidden_size':32, 'intermediate_size':64,
                     'num_hidden_layers':1, 'num_attention_heads':4, 'max_position_embeddings':8},
        vision_config={'hidden_size':32, 'intermediate_size':64, 'num_hidden_layers':1,
                       'num_attention_heads':4, 'image_size':16, 'patch_size':8})
    encoder = GuidanceEncoder.__new__(GuidanceEncoder)
    encoder.model = SiglipModel(config).eval().requires_grad_(False).half()
    encoder.modern, encoder.device = True, 'cpu'
    encoder.mean, encoder.std = [.5]*3, [.5]*3
    seen = {}
    def tokenize(prompts, **kwargs):
        seen.update(prompts=prompts, **kwargs)
        return BatchEncoding({'input_ids':torch.tensor([[1, 2, 3, 0]])})
    encoder.tokenizer = tokenize
    image = torch.rand(1, 3, 16, 16, requires_grad=True)
    embedding = encoder.image(image)
    text = encoder.text(['FLOWERS'])
    assert seen['prompts'] == ['flowers']
    assert seen['padding'] == 'max_length' and seen['max_length'] == 8 and seen['truncation']
    gradient = torch.autograd.grad((embedding*text).sum(), image)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_frame_guidance_losses_have_gradients_only_in_trusted_regions():
    from vibewarp.core.disco import frame_guidance_loss
    image = torch.full((1, 3, 4, 4), .2, requires_grad=True)
    target = torch.linspace(-.5, .5, 48).reshape(1, 3, 4, 4)
    mask = torch.zeros(1, 1, 4, 4)
    mask[..., :2] = 1
    lpips = lambda x, y: (x-y).square().mean(1,keepdim=True)
    loss = frame_guidance_loss(image, target, lpips_model=lpips,
        lpips_scale=2, image_scale=3, trusted_mask=mask)
    gradient = torch.autograd.grad(loss, image)[0]
    assert torch.isfinite(gradient).all()
    assert gradient[..., :2].abs().sum() > 0
    assert gradient[..., 2:].abs().sum() == 0


@pytest.mark.parametrize('use_mask',[False,True])
@pytest.mark.parametrize('mode,color', [
    ('prev stylized', (200, 40, 40)),
    ('prev warped', (40, 40, 200)),
    ('prev warped + cc', (40, 200, 40)),
    ('raw frame', (120, 120, 120)),
])
def test_disco_frame_guidance_source_is_independent_of_init(tmp_path, monkeypatch, mode, color, use_mask):
    from vibewarp.core.diffusion import RenderContext, FrameState, _render_single_frame
    c = RunConfig(model_version='disco_512')
    c.video.width = c.video.height = 64
    c.diffusion.guidance_mode = mode
    c.disco.mask_frame_guidance = use_mask
    c.color.before_enabled = c.color.after_enabled = False
    frames = tmp_path/'frames'
    frames.mkdir()
    Image.new('RGB', (64, 64), (120, 120, 120)).save(frames/'000002.jpg')
    previous = Image.new('RGB', (64, 64), (200, 40, 40))
    previous.save(tmp_path/f'{c.batch_name}(0)_000000.png')
    captured = {}
    class Backend:
        def render(self, config, init, *args, **kwargs):
            captured.update(kwargs)
            captured['init'] = init
            return init.copy()
    monkeypatch.setattr('vibewarp.core.diffusion.warp_between_frames', lambda *a, **k: (
        Image.new('RGB', (64, 64), (40, 200, 40)), Image.new('RGB', (64, 64), (40, 40, 200))))
    ctx = RenderContext(config=c, disco_backend=Backend(), batch_folder=str(tmp_path), video_frames_folder=str(frames))
    _render_single_frame(ctx, FrameState(prev_frame=previous), 1, 0, str(tmp_path), 'png')
    assert captured['init'].getpixel((0, 0)) == (40, 200, 40)
    assert captured['guidance_image'].getpixel((0, 0)) == color
    assert captured['mask_guidance'] == use_mask
