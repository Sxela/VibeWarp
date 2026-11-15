"""Disco Diffusion: pixel-space DDIM/PLMS with differentiable CLIP guidance."""
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from vibewarp.cancellation import raise_if_cancelled
from vibewarp.core.losses import spherical_dist_loss, tv_loss, range_loss
from vibewarp.render_events import render_event
from vibewarp.disco_catalog import CLASSIC_ENCODERS, MODERN_ENCODERS
from vibewarp.disco_cutouts import held_value as scheduled, inference_progress


CHECKPOINTS = {
    'disco_512': (
        'https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/c48ad0f73a0742cacaec8a87ce4e900acfb3821f/512x512_diffusion_uncond_finetune_008100.pt',
        '512x512_diffusion_uncond_finetune_008100.pt',
        '9c111ab89e214862b76e1fa6a1b3f1d329b1a88281885943d2cdbe357ad57648'),
    'disco_256': (
        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt',
        '256x256_diffusion_uncond.pt', ''),
}
SECONDARY_URL = 'https://huggingface.co/spaces/huggi/secondary_model_imagenet_2.pth/resolve/main/secondary_model_imagenet_2.pth'
SECONDARY_SHA = '983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a'


def disco_assets(config):
    from vibewarp.model_assets import ModelAsset, models_root
    root = models_root(config) / 'disco'
    url, name, digest = CHECKPOINTS[config.model_version]
    assets = []
    if not config.disco.checkpoint_path:
        assets.append(ModelAsset(url, str(root / name), digest))
    if config.disco.sampler == 'distilled' and not config.disco.distilled_guided:
        return assets
    if config.disco.use_secondary_model and not config.disco.secondary_path:
        assets.append(ModelAsset(SECONDARY_URL, str(root / 'secondary_model_imagenet_2.pth'), SECONDARY_SHA))
    import open_clip
    for name in config.disco.encoders:
        if name in CLASSIC_ENCODERS:
            spec = open_clip.get_pretrained_cfg(name, 'openai')
            url = spec['url'].replace('openaipublic.azureedge.net', 'openaipublic.blob.core.windows.net')
            assets.append(ModelAsset(url, str(root / 'clip' / f'{name}.pt'), url.split('/')[-2]))
        elif name not in MODERN_ENCODERS:
            raise ValueError(f'Unknown Disco guidance encoder: {name}')
    return assets


def ensure_guidance_assets(config):
    if config.disco.sampler == 'distilled' and not config.disco.distilled_guided:
        return
    from huggingface_hub import snapshot_download
    from vibewarp.model_assets import models_root
    for name in config.disco.encoders:
        if name in MODERN_ENCODERS:
            snapshot_download(MODERN_ENCODERS[name],
                local_dir=models_root(config) / 'disco' / name,
                allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt'], max_workers=1)


def frame_guidance_loss(image, target, *, lpips_model=None, lpips_scale=0.0,
                        image_scale=0.0, trusted_mask=None):
    """Existing LPIPS and spherical-distance guidance applied in pixel space."""
    from vibewarp.core.trusted_guidance import masked_lpips_loss, masked_spherical_loss
    if trusted_mask is not None and not bool((trusted_mask == 1).all()):
        loss = image.sum()*0
        if lpips_scale:
            loss = loss + masked_lpips_loss(image,target,trusted_mask,lpips_model)*lpips_scale
        if image_scale:
            loss = loss + masked_spherical_loss(image,target,trusted_mask)*image_scale
        return loss
    guided = image
    loss = image.sum() * 0
    if lpips_scale:
        loss = loss + lpips_model(guided, target).sum() * lpips_scale
    if image_scale:
        loss = loss + spherical_dist_loss(guided.float(), target.float()).sum() * image_scale
    return loss


def weighted_prompts(text):
    """One weighted prompt per line; URL colons are preserved."""
    result = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        prompt, separator, weight = line.rpartition(':')
        try:
            value = float(weight) if separator else 1.0
        except ValueError:
            prompt, value = line, 1.0
        else:
            prompt = prompt if separator else line
        if value:
            result.append((prompt, value))
    return result


def make_cutouts(image, size, overview, inner, power, gray_probability, *, generator=None, augment=True, crop_scale=1.0, max_size=0):
    """Notebook Dango video cutouts, with optional isolated RNG and coarse grids."""
    raise_if_cancelled()
    from torchvision import transforms as T
    from torchvision.transforms import functional as TF
    from vibewarp.vendor.resize_right import resize as resize_right
    if generator is not None:
        # Torchvision transforms use global RNGs. Isolate them so locked cutouts
        # include augmentations and do not disturb the sampler's random stream.
        devices=[image.device.index if image.device.index is not None else torch.cuda.current_device()] if image.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(generator.get_state())
            if image.is_cuda:
                torch.cuda.default_generators[devices[0]].manual_seed(generator.initial_seed())
            try:
                return make_cutouts(image,size,overview,inner,power,gray_probability,
                    augment=augment,crop_scale=crop_scale,max_size=max_size)
            finally:
                generator.set_state(torch.get_rng_state())
    _, _, height, width = image.shape
    if max_size and max(height,width) > max_size:
        ratio = max_size/max(height,width)
        shape = (max(1,round(height*ratio)),max(1,round(width*ratio)))
        image = F.interpolate(image,size=shape,mode='bilinear',align_corners=False,antialias=True)
        crop_scale *= ratio
        height,width = shape
    short_side=min(height,width)
    resize = lambda x: resize_right(x,out_shape=[x.shape[0],x.shape[1],size,size])
    if overview:
        # Preserve notebook padding, but avoid an unused full-frame graph when
        # the overview schedule requests zero full-frame cuts.
        padded = F.pad(image, ((height-short_side)//2,(height-short_side)//2,
                               (width-short_side)//2,(width-short_side)//2))
        global_view = resize(padded)
    views = []
    for index in range(overview):
        raise_if_cancelled()
        view = global_view
        if overview <= 4:
            if index >= 2:
                view = view.flip(-1)
            if index % 2:
                view = TF.rgb_to_grayscale(view, 3)
        views.append(view)
    minimum, maximum = min(height, width, max(1, round(size * crop_scale))), min(height, width)
    for index in range(inner):
        raise_if_cancelled()
        crop_size = int(torch.rand([]) ** power * (maximum-minimum) + minimum)
        x = int(torch.randint(width-crop_size+1, ()))
        y = int(torch.randint(height-crop_size+1, ()))
        view = image[:, :, y:y+crop_size, x:x+crop_size]
        # The inclusive prefix (one gray crop even at probability 0) is part
        # of the notebook's behavior, not a Bernoulli draw per crop.
        if index <= int(gray_probability*inner):
            view = TF.rgb_to_grayscale(view, 3)
        views.append(resize(view))
    if not views:
        raise ValueError('At least one guidance cutout is required')
    cuts = torch.cat(views)
    if augment:
        noise=T.Lambda(lambda x:x+torch.randn_like(x)*.01)
        cuts=T.Compose([
            T.RandomHorizontalFlip(p=.4),noise,
            T.RandomAffine(degrees=10,translate=(.05,.05),interpolation=T.InterpolationMode.BILINEAR),noise,
            T.RandomGrayscale(p=.1),noise,
            T.ColorJitter(brightness=.1,contrast=.1,saturation=.1,hue=.3),
        ])(cuts)
    return cuts


class GuidanceEncoder:
    def __init__(self, name, root, device, precision=None):
        self.name = name
        # Keep the old argument for callers restoring previous settings, but
        # never allow it to select FP32 or a device-dependent fallback.
        self.precision = 'fp16'
        if name in MODERN_ENCODERS:
            from transformers import AutoModel, AutoProcessor
            source = str(root / name)
            processor = AutoProcessor.from_pretrained(source)
            self.model = AutoModel.from_pretrained(source, torch_dtype=torch.float16)
            self.model = self.model.eval().requires_grad_(False).to(device=device, dtype=torch.float16)
            self.tokenizer = processor.tokenizer
            self.size = self.model.config.vision_config.image_size
            self.mean, self.std = processor.image_processor.image_mean, processor.image_processor.image_std
            self.modern = True
        else:
            import open_clip
            from open_clip.openai import load_openai_model
            # Original CLIP weights are TorchScript archives, not pickle
            # checkpoints. This loader also preserves OpenAI's QuickGELU.
            self.model = load_openai_model(str(root / 'clip' / f'{name}.pt'), device='cpu', precision=self.precision)
            self.model.eval().requires_grad_(False).to(device)
            self.tokenizer = open_clip.get_tokenizer(name)
            self.size = self.model.visual.image_size
            if isinstance(self.size, tuple):
                self.size = self.size[0]
            spec = open_clip.get_pretrained_cfg(name, 'openai')
            self.mean, self.std = spec['mean'], spec['std']
            self.modern = False
        self.device = device

    def text(self, prompts):
        with torch.no_grad():
            if self.modern:
                # FixRes SigLIP 2 uses lowercased text and fixed training-length
                # padding; tokenizer defaults can otherwise silently differ.
                tokens = self.tokenizer([prompt.lower() for prompt in prompts],
                    padding='max_length', max_length=self.model.config.text_config.max_position_embeddings,
                    truncation=True, return_tensors='pt').to(self.device)
                result = self.model.get_text_features(**tokens)
            else:
                result = self.model.encode_text(self.tokenizer(prompts).to(self.device))
        if not isinstance(result, torch.Tensor):
            result = result.pooler_output
        return F.normalize(result.float(), dim=-1)

    def image(self, image):
        mean = image.new_tensor(self.mean)[None, :, None, None]
        std = image.new_tensor(self.std)[None, :, None, None]
        image = (image-mean) / std
        # Both OpenCLIP and transformers vision encoders receive FP16 pixels.
        # The cast retains the gradient path to the FP32 noisy sampler input.
        image = image.to(torch.float16)
        result = (self.model.get_image_features(pixel_values=image) if self.modern
                  else self.model.encode_image(image))
        if not isinstance(result, torch.Tensor):
            result = result.pooler_output
        return F.normalize(result.float(), dim=-1)


class DiscoBackend:
    def __init__(self, config, device='cuda'):
        from vibewarp.model_assets import models_root
        from vibewarp.vendor.guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
        from vibewarp.core.disco_attention import configure_attention
        root = models_root(config) / 'disco'
        d = config.disco
        self.config = config
        if d.sampler not in ('ddim','plms'):
            raise ValueError('Use pipeline.load_models for the dedicated distilled LoRA backend')
        options = model_and_diffusion_defaults()
        base_steps = (1000 // d.steps) * d.steps if d.steps < 1000 else d.steps
        options.update(image_size=512 if config.model_version == 'disco_512' else 256,
            attention_resolutions='32,16,8', class_cond=False, diffusion_steps=base_steps,
            rescale_timesteps=True, timestep_respacing=f'ddim{d.steps}', learn_sigma=True,
            noise_schedule='linear', num_channels=256, num_head_channels=64,
            num_res_blocks=2, resblock_updown=True, use_checkpoint=d.gradient_checkpointing,
            use_fp16=str(device).startswith('cuda'), use_scale_shift_norm=True)
        if d.timestep_grid:
            options.update(diffusion_steps=1000, timestep_respacing='')
        self.model, self.diffusion = create_model_and_diffusion(**options)
        if d.timestep_grid:
            self.diffusion = self.diffusion_for_steps(d.steps)
        checkpoint = d.checkpoint_path or str(root / CHECKPOINTS[config.model_version][1])
        self.model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        self.model.eval().requires_grad_(False).to(device)
        if str(device).startswith('cuda'):
            self.model.convert_to_fp16()
        configure_attention(self.model, d.attention)
        self.secondary = None
        if d.use_secondary_model:
            from vibewarp.core.disco_secondary import SecondaryDiffusionImageNet2
            self.secondary = SecondaryDiffusionImageNet2()
            self.secondary.load_state_dict(torch.load(d.secondary_path or root / 'secondary_model_imagenet_2.pth',
                                                       map_location='cpu', weights_only=True))
            self.secondary.eval().requires_grad_(False).to(device)
            from vibewarp.core.disco_optimizations import SecondaryPrecision
            self.secondary = SecondaryPrecision(self.secondary, d.secondary_precision, device)
        self.encoders = [GuidanceEncoder(name, root, device) for name in d.encoders]
        self.device = device
        self.lpips = None

    def diffusion_for_steps(self, steps):
        if self.config.disco.timestep_grid:
            from vibewarp.core.disco_lora import guided_diffusion
            from vibewarp.core.disco_repeated import validate_grid
            return guided_diffusion(validate_grid(self.config.disco.timestep_grid,steps))
        from vibewarp.vendor.guided_diffusion.script_util import create_gaussian_diffusion
        return create_gaussian_diffusion(steps=(1000 // steps)*steps,
            learn_sigma=True, rescale_timesteps=True, timestep_respacing=f'ddim{steps}')

    def render(self, config, init_image, prompt, negative_prompt, seed, *, frame_num=0,
               mask=None, strength=None, preview_path=None, guidance_image=None, mask_guidance=None, trust_mask=None):
        self.config = config
        d, diffusion = config.disco, self.diffusion
        if mask_guidance is None:
            mask_guidance = d.mask_frame_guidance
        from vibewarp.core.diffusion import get_frame_schedule
        schedule = get_frame_schedule(frame_num, config)
        steps = schedule['steps']
        if not 2 <= steps <= 1000:
            raise ValueError('Disco scheduled steps must be between 2 and 1000')
        if diffusion.num_timesteps != steps:
            # Respacing changes only the sampler, never reloads model weights.
            self.diffusion = diffusion = self.diffusion_for_steps(steps)
        width,height=init_image.size
        aligned=(((width+63)//64)*64,((height+63)//64)*64)
        if aligned != init_image.size:
            init_image=init_image.resize(aligned,Image.Resampling.LANCZOS)
        image = np.array(init_image.convert('RGB')).copy()
        init = torch.from_numpy(image).permute(2, 0, 1)[None].to(self.device).float() / 127.5 - 1
        target = init
        if guidance_image is not None:
            reference = guidance_image.convert('RGB').resize(init_image.size, Image.Resampling.LANCZOS)
            pixels = np.array(reference).copy()
            target = torch.from_numpy(pixels).permute(2, 0, 1)[None].to(self.device).float()/127.5-1
        trusted = None
        if (trust_mask is not None or mask is not None) and mask_guidance:
            from vibewarp.core.trusted_guidance import resize_trust
            trusted = resize_trust(trust_mask if trust_mask is not None else mask, init)
            trusted = F.interpolate(trusted, init.shape[-2:], mode='bilinear', align_corners=False).clamp(0, 1)
        # The secondary U-Net downsamples five times.
        if init.shape[-2] % 64 or init.shape[-1] % 64:
            raise ValueError('Disco render dimensions must be multiples of 64')
        generator = torch.Generator(device=self.device).manual_seed(seed)
        if d.notebook_rng:
            from vibewarp.utils.misc import seed_everything
            seed_everything(seed)
        noise = torch.randn(init.shape, generator=None if d.notebook_rng else generator, device=self.device)
        if d.fixed_seed and (config.diffusion.fixed_seed_randomness or config.diffusion.fixed_seed_wiggle):
            from vibewarp.core.noise_variation import vary_fixed_noise
            noise = vary_fixed_noise(noise,config.diffusion.fixed_seed_randomness,seed,frame_num,wiggle=config.diffusion.fixed_seed_wiggle,direction=config.diffusion.fixed_seed_wiggle_direction)
        targets = []
        prompts = weighted_prompts(prompt) + [(p, -abs(w)) for p, w in weighted_prompts(negative_prompt)]
        image_prompts = weighted_prompts(str(scheduled_text(d.image_prompts, frame_num)))
        for encoder in self.encoders:
            embeds, weights = [], []
            if prompts:
                embeds.append(encoder.text([p for p, _ in prompts]))
                weights.extend(w for _, w in prompts)
            for path, weight in image_prompts:
                with Image.open(path) as reference:
                    ref = torch.from_numpy(np.array(reference.convert('RGB').resize((encoder.size, encoder.size))).copy())
                ref = ref.permute(2, 0, 1)[None].to(self.device).float() / 255
                with torch.no_grad():
                    embeds.append(encoder.image(ref))
                weights.append(weight)
            if not embeds:
                raise ValueError('Disco needs a text or image prompt')
            weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
            # Signed weights can cancel; normalize safely without losing negatives.
            denominator = weights.sum().abs()
            if denominator < 1e-3:
                denominator = weights.abs().sum()
            weights = weights / denominator.clamp_min(1e-8)
            targets.append((encoder, torch.cat(embeds), weights))
        strength = schedule['style_strength'] if strength is None else strength
        if not 0 < strength <= 1:
            raise ValueError('Disco scheduled style strength must be above 0 and at most 1')
        from vibewarp.core.disco_repeated import skip_for_strength
        skip = skip_for_strength(diffusion.num_timesteps, strength)
        total = diffusion.num_timesteps-skip
        print(f'Disco sampling frame {frame_num}: steps={steps}, strength={strength:g}, '
              f'executed={total}, seed={seed}')
        mapped_times = [value * 1000 / diffusion.original_num_steps for value in diffusion.timestep_map]
        from vibewarp.core.disco_tiling import prepare_disco_denoisers
        from vibewarp.core.disco_progressive import progressive_sizes, progressive_disco_samples, stage_denoiser
        sizes = progressive_sizes(config.diffusion, init.shape[-2:], total)
        progressive = any(size != init.shape[-2:] for size in sizes)
        denoiser, secondary_prediction = prepare_disco_denoisers(
            self.model, self.secondary, diffusion, config, skip, progressive=progressive)
        latest_preview = [None]
        nonfinite_guidance_values=[0]
        guidance_evaluations=[0]
        executed_step=[0]
        from vibewarp.utils.scheduling import get_scheduled_arg
        loss_scale = float(get_scheduled_arg(frame_num, d.init_scale_schedule, blend_json_schedules=False)) if d.init_scale_schedule is not None else d.init_scale
        image_scale = float(get_scheduled_arg(frame_num, d.image_scale_schedule, blend_json_schedules=False)) if d.image_scale_schedule is not None else d.image_scale
        use_spatial = trusted is not None and not bool((trusted == 1).all())
        if loss_scale and (self.lpips is None or getattr(self.lpips, 'spatial', False) != use_spatial):
            import lpips
            devices = [torch.device(self.device).index or torch.cuda.current_device()] if str(self.device).startswith('cuda') else []
            with torch.random.fork_rng(devices=devices):
                self.lpips = lpips.LPIPS(net='vgg', spatial=use_spatial).eval().requires_grad_(False).to(self.device)
        # Cutouts, augmentations and stochastic sampler steps must restart from
        # this frame's seed on resume, independently of prior model setup.
        from vibewarp.utils.misc import seed_everything
        if not d.notebook_rng:
            seed_everything(seed)

        def cond_fn(x, t, **kwargs):
            guidance_evaluations[0] += 1
            raise_if_cancelled()
            latest_preview[0] = None
            # PLMS's bootstrap also evaluates the next timestep. Resolve the
            # actual mapped time instead of assuming one cond_fn call per step.
            time_value = float(t[0])
            index = min(range(len(mapped_times)), key=lambda i: abs(mapped_times[i]-time_value))
            semantic_step = (total-1-index) % d.guidance_every == 0
            with torch.enable_grad():
                native_size = x.shape[-2:]
                if native_size != target.shape[-2:]:
                    guidance_target = F.interpolate(target, native_size, mode='area')
                    guidance_mask = F.interpolate(trusted, native_size, mode='area') if trusted is not None else None
                else:
                    guidance_target, guidance_mask = target, trusted
                x = x.detach().requires_grad_()
                fac = float(diffusion.sqrt_one_minus_alphas_cumprod[index])
                if self.secondary is not None:
                    predictor = stage_denoiser(secondary_prediction, native_size, init.shape[-2:])
                    out = predictor(x, t)
                else:
                    times = torch.full((x.shape[0],), index, device=x.device, dtype=torch.long)
                    predictor = stage_denoiser(denoiser, native_size, init.shape[-2:])
                    out = diffusion.p_mean_variance(predictor, x, times, clip_denoised=False)['pred_xstart']
                latest_preview[0] = out.detach()
                x_in = out * fac + x * (1-fac)
                image_grad = torch.zeros_like(x_in)
                progress = inference_progress(executed_step[0], total)
                for encoder, embedding_target, weights in targets if semantic_step and d.clip_guidance_scale else ():
                    batches = kwargs.get('_clip_cutn_batches', d.cutn_batches)
                    def make_cuts(cut_batch):
                        crop_generator = None
                        if d.lock_cutouts_across_frames:
                            # Repeated crop geometry across frames, independent of frame seed.
                            crop_generator = torch.Generator().manual_seed(index * 1009 + cut_batch)
                        return make_cutouts((x_in+1)/2, encoder.size,
                            int(scheduled(d.cut_overview, progress)), int(scheduled(d.cut_innercut, progress)),
                            d.cut_ic_pow, float(scheduled(d.cut_icgray_schedule, progress)) if d.cut_icgray_schedule else d.cut_icgray_p, generator=crop_generator, augment=not d.skip_augs,
                            crop_scale=min(native_size[0]/init.shape[-2], native_size[1]/init.shape[-1]),
                            max_size=d.cut_max_size)
                    from vibewarp.core.disco_optimizations import grouped_clip_gradient
                    image_grad += grouped_clip_gradient(x_in, encoder, embedding_target, weights, make_cuts,
                        batches=batches, group=d.cut_batch_group, scale=d.clip_guidance_scale,
                        balance=d.balance_guidance)
                regularization = (tv_loss(x_in).sum()*d.tv_scale
                    + (x_in-x_in.clamp(-1, 1)).abs().mean()*d.sat_scale)
                regularization = regularization + frame_guidance_loss(x_in, guidance_target,
                    lpips_model=self.lpips, lpips_scale=loss_scale,
                    image_scale=image_scale, trusted_mask=guidance_mask)
                image_grad += torch.autograd.grad(regularization, x_in, retain_graph=True)[0]
                grad = -torch.autograd.grad(x_in, x, image_grad, retain_graph=True)[0]
                # Range loss is defined on the denoiser prediction, rather
                # than its blended CLIP input. Differentiate that path directly.
                grad -= torch.autograd.grad(range_loss(out).sum()*d.range_scale, x)[0]
                if not torch.isfinite(grad).all():
                    nonfinite_guidance_values[0]+=int((~torch.isfinite(grad)).sum())
                    print(f'Nonfinite Disco guidance at original t={float(t[0]):g}; '
                          f'values={nonfinite_guidance_values[0]}')
                    if d.quality_lora_enabled:
                        grad=torch.nan_to_num(grad,nan=0.,posinf=0.,neginf=0.)
                    else:
                        return torch.zeros_like(x)
                if d.quality_lora_enabled and d.guidance_noise_cap_percent>0:
                    # Quality/turbo handoff: the state-relative cap replaces
                    # the absolute gradient clamp on this guided DDIM path.
                    return grad
                rms = grad.square().mean().sqrt().clamp_min(1e-8)
                return grad * (d.clamp_max/rms).clamp(max=1)

        adapted=d.sampler == 'distilled' or d.quality_lora_enabled
        sampler_name = 'ddim' if adapted else d.sampler
        inpainting_stop = min(d.inpainting_stop,total-1) if adapted else d.inpainting_stop
        sampler = diffusion.ddim_sample_loop_progressive if sampler_name == 'ddim' else diffusion.plms_sample_loop_progressive
        options = dict(noise=noise, clip_denoised=False, cond_fn=cond_fn, model_kwargs={},
            device=self.device, progress=False, skip_timesteps=skip, init_image=init,
            mask=mask if d.inpaint else None, inpainting_stop=inpainting_stop)
        if sampler_name == 'ddim':
            options['eta'] = d.eta
        result = None
        from vibewarp.core.disco_guidance import guidance_noise_cap
        cap_stats = {}
        samples = (progressive_disco_samples(diffusion, denoiser, init, noise, sizes,
                    skip=skip, sampler=sampler_name, cond_fn=cond_fn, eta=d.eta,
                    mask=mask if d.inpaint else None, inpainting_stop=inpainting_stop)
                   if progressive else sampler(denoiser, tuple(init.shape), **options))
        from vibewarp.core.disco_repeated import install_multi_guidance
        install_multi_guidance(diffusion, d.clip_grad_steps, d.clip_divide_cuts, d.cutn_batches)
        with guidance_noise_cap(diffusion,d.guidance_noise_cap_percent,cap_stats):
            for step, sample in enumerate(samples):
                raise_if_cancelled()
                # The capped final DDIM state may differ from its clean prediction.
                result = sample['sample'] if adapted and step == total-1 else sample['pred_xstart']
                index=total-step-1
                cap_stats.update(original_timestep=mapped_times[index],spaced_index=index,
                    noise_sigma=float(diffusion.sqrt_one_minus_alphas_cumprod[index]),
                    nonfinite_guidance_values=nonfinite_guidance_values[0], guidance_evaluations=guidance_evaluations[0], clip_grad_steps=d.clip_grad_steps, clip_divide_cuts=d.clip_divide_cuts)
                if preview_path and d.preview_interval > 0 and step % d.preview_interval == 0:
                    preview = latest_preview[0] if latest_preview[0] is not None else result
                    path = Path(preview_path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix('.tmp.png')
                    tensor_image(preview).save(temporary)
                    temporary.replace(path)
                    render_event(step=step+1, total_steps=total, preview=str(path), frame=frame_num, **cap_stats)
                else:
                    render_event(step=step+1, total_steps=total, frame=frame_num, **cap_stats)
                executed_step[0] = step + 1
        if not torch.isfinite(result).all():
            raise FloatingPointError('Nonfinite Disco sampler output')
        return tensor_image(result)


def tensor_image(tensor):
    pixels = tensor.detach().float().clamp(-1, 1)[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.rint((pixels+1)*127.5).astype(np.uint8))


def scheduled_text(schedule, frame):
    return next((schedule[k] for k in sorted(schedule, reverse=True) if k <= frame), '')


def render_disco_frame(ctx, state):
    from vibewarp.core.diffusion import _resolve_edit_prompts, get_frame_schedule
    config = ctx.config
    frame = state.frame_num
    raw = Path(ctx.video_frames_folder) / f'{frame+1:06d}.jpg'
    source = state.init_image or str(raw)
    with Image.open(source) as image:
        init = image.convert('RGB').resize((config.video.width, config.video.height), Image.Resampling.LANCZOS)
    if config.disco.sampler == 'distilled' and not config.disco.distilled_guided:
        seed = int(get_frame_schedule(frame, config)['seed'])
        if not config.disco.fixed_seed:
            seed += frame
        result = ctx.disco_backend.render(config, init, '', '', seed, frame_num=frame,
            preview_path=str(Path(ctx.batch_folder)/'temp'/'denoise_preview.png'))
        state.prev_frame = result
        return result
    guidance = init
    # The notebook compares LPIPS to the actual sampler init, including warp,
    # consistency blending and any background masking. Explicit source choices
    # remain available; the default follows that prepared init exactly.
    if state.guidance_image and config.diffusion.guidance_mode != 'prev warped + cc':
        with Image.open(state.guidance_image) as image:
            guidance = image.convert('RGB').copy()
    # Full-image notebook guidance is the default; optionally exclude unreliable
    # regions from both frame losses, independently of sampler inpainting.
    mask_guidance = config.disco.mask_frame_guidance
    if config.disco.init_scale or config.disco.image_scale:
        target_source = source if config.diffusion.guidance_mode == 'prev warped + cc' else state.guidance_image or source
        print(f'Disco frame guidance {frame}: source={target_source}; '
              f'LPIPS={config.disco.init_scale:g}, image={config.disco.image_scale:g}; '
              f'consistency masking={mask_guidance}')
    prompt, negative = _resolve_edit_prompts(ctx, frame)
    seed = int(get_frame_schedule(frame, config)['seed'])
    if not config.disco.fixed_seed:
        seed += frame
    # The trusted mask pins coherent warped content during early denoising.
    trust = getattr(ctx, '_temporal_consistency_mask', None)
    mask = getattr(ctx, '_disco_consistency_mask', None)
    if mask is not None:
        import cv2
        kernel = max(1, config.disco.inpaint_mask_blur) | 1
        mask = cv2.GaussianBlur(mask.astype(np.float32), (kernel, kernel), config.disco.inpaint_mask_sigma)
        mask = (mask >= config.disco.inpaint_mask_threshold).astype(np.float32)
    result = ctx.disco_backend.render(config, init, prompt, negative, seed,
        frame_num=frame, mask=mask,
        guidance_image=guidance, mask_guidance=mask_guidance, trust_mask=trust,
        strength=getattr(ctx, '_turbo_strength', None),
        preview_path=str(Path(ctx.batch_folder) / 'temp' / 'denoise_preview.png'))
    state.prev_frame = result
    return result
