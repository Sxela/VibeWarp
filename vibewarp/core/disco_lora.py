"""Guided and exact-reference Crowson adapter inference, adapted from tdm.core.

Contracts: discodiffusion-warp/LORA_INTEGRATION_HANDOFF.md and
QUALITY_TURBO_INTEGRATION_HANDOFF.md. This is a custom
Linear/Conv1d/Conv2d adapter and its saved VP schedule, not an SD/PEFT LoRA.
The training package is not imported into the application's model namespace.
"""
from contextlib import contextmanager, nullcontext
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import load_file

from vibewarp.cancellation import raise_if_cancelled
from vibewarp.render_events import render_event
from vibewarp.core.disco import DiscoBackend

TEACHER_SHA256 = '9c111ab89e214862b76e1fa6a1b3f1d329b1a88281885943d2cdbe357ad57648'
GRIDS = {4: [999, 749, 499, 249, -1], 6: [999, 832, 665, 499, 332, 165, -1]}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        while chunk := stream.read(8*1024*1024):
            raise_if_cancelled()
            digest.update(chunk)
    return digest.hexdigest()


def resolve_adapter(path):
    """Resolve a standalone export or the retained best of a completed run."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        pointer = json.loads((path/'best.json').read_text(encoding='utf-8'))
        generation = pointer['checkpoint']
        if (not isinstance(generation, str) or generation in ('', '.', '..')
                or '/' in generation or '\\' in generation):
            raise ValueError('Invalid best checkpoint pointer')
        best_root = (path/'best').resolve()
        selected = (best_root/generation/'best-ema.safetensors').resolve()
        if not selected.is_relative_to(best_root):
            raise ValueError('Best adapter resolves outside the retained best directory')
        # A run directory only counts once training marked the generation complete.
        # No digest check: adapters are safetensors, which cannot carry code.
        if not (selected.parent/'COMPLETE.json').is_file():
            raise ValueError('Best adapter generation is incomplete (no COMPLETE.json)')
        path = selected
    if not path.is_file() or path.suffix.lower() != '.safetensors':
        raise ValueError('Choose an inference .safetensors adapter or a run directory with best.json')
    return path


def adapter_config(path):
    with safe_open(str(path), framework='pt') as stream:
        config = json.loads((stream.metadata() or {}).get('config', '{}'))
    if config.get('teacher_sha256') != TEACHER_SHA256:
        raise ValueError('Few-step adapter/base checkpoint mismatch')
    grid = config.get('grid')
    steps = len(grid)-1 if isinstance(grid, list) else 0
    if (steps not in GRIDS or grid != GRIDS[steps]
            or not all(type(value) is int for value in grid) or config.get('size') != 512):
        raise ValueError('Unsupported trained schedule/resolution: expected the saved 4/6-step 512px grid')
    rank, alpha = config.get('rank'), config.get('alpha')
    if (type(rank) is not int or rank < 1 or type(alpha) not in (int,float)
            or not math.isfinite(alpha) or alpha <= 0):
        raise ValueError('Invalid adapter rank/alpha metadata')
    if config.get('objective_version') != 'supervised-clean-sqrt-snr-v1':
        raise ValueError('Use the supervised few-step adapter; earlier TDM objectives are unsupported')
    if config.get('fp32_output_head', True) is not True:
        raise ValueError('The supported adapter requires an FP32 output head')
    return config


def quality_adapter_config(path):
    with safe_open(str(path),framework='pt') as stream:
        metadata=stream.metadata() or {}
    config=json.loads(metadata.get('config','{}'))
    if (config.get('teacher_sha256') != TEACHER_SHA256 or config.get('size') != 512
            or config.get('diffusion_steps',1000) != 1000):
        raise ValueError('Quality adapter/base architecture mismatch')
    if (config.get('method') != 'aesthetic real-image epsilon MSE'
            or config.get('objective_version') != 'aesthetic-epsilon-mse-v1'):
        raise ValueError('Expected an aesthetic real-image epsilon MSE quality adapter')
    rank,alpha=config.get('rank'),config.get('alpha')
    if (type(rank) is not int or rank<1 or type(alpha) not in (int,float)
            or not math.isfinite(alpha) or alpha<=0):
        raise ValueError('Invalid quality adapter rank/alpha')
    if config.get('fp32_output_head',True) is not True or config.get('variance_training',False) is not False:
        raise ValueError('Quality adapter requires FP32 output head and epsilon-only training')
    if config.get('precision','fp16') not in ('fp16','fp32'):
        raise ValueError('Unsupported quality adapter precision')
    return config,metadata


class AdapterLayer(nn.Module):
    def __init__(self, base, rank, alpha, branch='student'):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.active = None
        self.scale = alpha/rank if branch == 'student' else 1.
        self.strength = 1.0
        self.quality_scale = alpha/rank if branch == 'quality' else 0.
        self.quality_strength = 0.
        self.adapters = nn.ModuleDict()
        self.add_branch(branch,rank)

    def add_branch(self,name,rank):
        if name in self.adapters:
            raise ValueError(f'Adapter branch already loaded: {name}')
        base=self.base
        if isinstance(base, nn.Linear):
            down = nn.Linear(base.in_features, rank, bias=False)
            up = nn.Linear(rank, base.out_features, bias=False)
        else:
            if base.groups != 1:
                raise ValueError('Grouped convolution adapters are unsupported')
            down = type(base)(base.in_channels, rank, base.kernel_size, stride=base.stride,
                              padding=base.padding, dilation=base.dilation, bias=False)
            up = type(base)(rank, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(down.weight, a=5**.5)
        nn.init.zeros_(up.weight)
        self.adapters[name]=nn.Sequential(down,up).to(device=base.weight.device,dtype=base.weight.dtype)

    def forward(self, x):
        result = self.base(x)
        if self.quality_strength:
            result=result+(self.quality_strength*self.quality_scale)*self.adapters['quality'](x)
        return result if self.active is None else result + (self.scale*self.strength)*self.adapters[self.active](x)


def inject_adapters(model, rank, alpha, branch='student'):
    if any(isinstance(layer, AdapterLayer) for layer in model.modules()):
        raise ValueError('Adapters are already injected; reload a clean base model')
    def inject(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                setattr(parent, name, AdapterLayer(child, rank, alpha,branch))
            else:
                inject(child)
    model.requires_grad_(False)
    inject(model)


@contextmanager
def use_adapter(model, name='student', *, strength=1.0, quality_strength=None):
    if not math.isfinite(strength) or strength < 0:
        raise ValueError('LoRA strength must be finite and nonnegative')
    if quality_strength is not None and (not math.isfinite(quality_strength) or quality_strength<0):
        raise ValueError('Quality LoRA strength must be finite and nonnegative')
    layers = [layer for layer in model.modules() if isinstance(layer, AdapterLayer)]
    previous = [(layer.active,layer.strength,layer.quality_strength) for layer in layers]
    try:
        for layer in layers:
            if name is not None and name not in layer.adapters:
                raise ValueError(f'Unknown adapter branch: {name}')
            if getattr(layer, '_merged', False):
                layer.active = None
                continue
            layer.active = name if strength != 0 else None
            layer.strength = strength
            if quality_strength is not None:
                if quality_strength and 'quality' not in layer.adapters:
                    raise ValueError('Quality adapter branch is not loaded')
                layer.quality_strength=quality_strength
        yield
    finally:
        for layer, (active,previous_strength,previous_quality) in zip(layers, previous):
            layer.active,layer.strength = active,previous_strength
            layer.quality_strength=previous_quality


def load_adapter_state(model, state, branch='student'):
    expected = {f'{name}.adapters.student.{key}':value.shape
                for name, layer in model.named_modules() if isinstance(layer, AdapterLayer)
                for key, value in layer.adapters[branch].state_dict().items()}
    if set(state) != set(expected):
        raise ValueError(f'Adapter keys mismatch: missing={set(expected)-set(state)}, extra={set(state)-set(expected)}')
    for key, tensor in state.items():
        if tensor.shape != expected[key] or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f'Invalid adapter tensor: {key}')
    remapped={key.replace('.adapters.student.',f'.adapters.{branch}.'):value for key,value in state.items()}
    model.load_state_dict(remapped, strict=False)


def attach_quality_adapter(model,path):
    config,metadata=quality_adapter_config(path)
    layers=[layer for layer in model.modules() if isinstance(layer,AdapterLayer)]
    if layers:
        for layer in layers:
            layer.add_branch('quality',config['rank'])
            layer.quality_scale=config['alpha']/config['rank']
    else:
        inject_adapters(model,config['rank'],config['alpha'],branch='quality')
    load_adapter_state(model,load_file(str(path)),branch='quality')
    return config,metadata


class VPSchedule:
    def __init__(self, alphas_cumprod, device):
        values = torch.as_tensor(np.asarray(alphas_cumprod), device=device, dtype=torch.float32)
        self.alpha, self.sigma = values.sqrt(), (1-values).sqrt()

    def coefficients(self, t, x):
        t = torch.as_tensor(t, device=x.device, dtype=torch.long)
        if (t < -1).any() or (t >= len(self.alpha)).any():
            raise ValueError('Timestep outside schedule')
        a = torch.where(t < 0, 1., self.alpha[t.clamp_min(0)])
        s = torch.where(t < 0, 0., self.sigma[t.clamp_min(0)])
        shape = (-1,) + (1,)*(x.ndim-1)
        return a.reshape(shape), s.reshape(shape)


def rollout(model, schedule, noise, grid, *, adapter='student', strength=1.0):
    """Yield the exact saved deterministic VP updates, with FP32 arithmetic."""
    x = noise.float()
    with use_adapter(model, adapter, strength=strength):
        for start, end in zip(grid[:-1], grid[1:]):
            raise_if_cancelled()
            time = torch.full((x.shape[0],), start, device=x.device, dtype=torch.long)
            epsilon = model(x, time)[:, :3].float()
            alpha, sigma = schedule.coefficients(start, x)
            clean = (x.float() - sigma*epsilon)/alpha
            alpha, sigma = schedule.coefficients(end, clean)
            x = alpha*clean.float() + sigma*epsilon.float()
            if not torch.isfinite(x).all():
                raise FloatingPointError('Nonfinite few-step output')
            yield x


def rescaled_grid(trained_grid, steps):
    """Retain the trained grid exactly, or resample its original VP time span."""
    if type(steps) is not int or not 2 <= steps <= 1000:
        raise ValueError('Choose between 2 and 1000 distilled sampling steps')
    if steps == len(trained_grid)-1:
        return list(trained_grid)
    return [1000*(steps-index)//steps-1 for index in range(steps+1)]


def guided_diffusion(grid):
    from vibewarp.vendor.guided_diffusion import gaussian_diffusion as gd
    from vibewarp.vendor.guided_diffusion.respace import SpacedDiffusion
    return SpacedDiffusion(use_timesteps=set(grid[:-1]),
        betas=gd.get_named_beta_schedule('linear',1000),
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.LEARNED_RANGE,
        loss_type=gd.LossType.MSE,rescale_timesteps=True)


class GuidedAdapterModel(nn.Module):
    """Autocast only the adapted UNet; keep losses and VP arithmetic in FP32."""
    def __init__(self, unet, fp16=True):
        super().__init__()
        self.unet,self.fp16=unet,fp16

    def forward(self,x,t,**kwargs):
        with torch.autocast(x.device.type,dtype=torch.float16,
                            enabled=x.device.type=='cuda' and self.fp16):
            return self.unet(x,t,**kwargs).float()


class DistilledDiscoBackend(DiscoBackend):
    def __init__(self, config, device='cuda'):
        from vibewarp.config_io import validate_config
        errors = validate_config(config, require_inputs=False)
        d=config.disco
        if (d.sampler != 'distilled' and not d.quality_lora_enabled) or errors:
            raise ValueError(f'Invalid adapted Disco configuration: {errors}')
        from vibewarp.core.disco import CHECKPOINTS
        from vibewarp.model_assets import models_root
        from vibewarp.core.disco_attention import configure_attention
        from vibewarp.vendor.guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
        self.config = config
        self.has_turbo=d.sampler == 'distilled'
        self.adapter_path=resolve_adapter(d.few_step_lora_path) if self.has_turbo else None
        self.adapter_metadata=adapter_config(self.adapter_path) if self.has_turbo else None
        self.grid=self.adapter_metadata['grid'] if self.has_turbo else GRIDS[4]
        self.quality_path=resolve_adapter(d.quality_lora_path) if d.quality_lora_enabled else None
        self.quality_config=self.quality_metadata=None
        if self.quality_path:
            self.quality_config,self.quality_metadata=quality_adapter_config(self.quality_path)
        if self.has_turbo and not d.distilled_guided and d.steps != len(self.grid)-1:
            raise ValueError('UI step count must match the adapter trained grid')
        base = config.disco.checkpoint_path or models_root(config)/'disco'/CHECKPOINTS['disco_512'][1]
        if sha256(base) != TEACHER_SHA256:
            raise ValueError('Teacher SHA-256 does not match the required Crowson 512px base')
        options = model_and_diffusion_defaults()
        options.update(image_size=512, num_channels=256, num_res_blocks=2,
            attention_resolutions='32,16,8', num_head_channels=64, learn_sigma=True,
            class_cond=False, resblock_updown=True, use_scale_shift_norm=True,
            use_checkpoint=config.disco.gradient_checkpointing, use_fp16=False, diffusion_steps=1000,
            noise_schedule='linear', timestep_respacing='', rescale_timesteps=True)
        self.model, diffusion = create_model_and_diffusion(**options)
        state = torch.load(base, map_location='cpu', weights_only=True, mmap=True)
        self.model.load_state_dict(state, strict=True)
        del state
        original_head = self.model.out.forward
        def fp32_head(x):
            with torch.autocast(device_type=x.device.type, enabled=False):
                return original_head(x.float())
        self.model.out.forward = fp32_head
        configure_attention(self.model, config.disco.attention)
        if self.has_turbo:
            inject_adapters(self.model, self.adapter_metadata['rank'], self.adapter_metadata['alpha'])
            load_adapter_state(self.model, load_file(str(self.adapter_path)))
            print(f'Turbo LoRA: {self.adapter_path}; '
                  f'rank={self.adapter_metadata["rank"]}, alpha={self.adapter_metadata["alpha"]}')
        if self.quality_path:
            self.quality_config,self.quality_metadata=attach_quality_adapter(self.model,self.quality_path)
            print(f'Quality LoRA: {self.quality_path}; iteration={self.quality_metadata.get("iteration","unknown")}; '
                  f'weights={self.quality_metadata.get("weights","unknown")}; '
                  f'rank={self.quality_config["rank"]}, alpha={self.quality_config["alpha"]}')
        self.model.eval().requires_grad_(False).to(device)
        self.schedule = VPSchedule(diffusion.alphas_cumprod, device)
        self.device = torch.device(device)
        self.secondary,self.lpips,self.encoders=None,None,[]
        self._merged_strengths = None
        if d.use_secondary_model:
            from vibewarp.core.disco_secondary import SecondaryDiffusionImageNet2
            self.secondary = SecondaryDiffusionImageNet2()
            self.secondary.load_state_dict(torch.load(d.secondary_path or models_root(config)/'disco/secondary_model_imagenet_2.pth', map_location='cpu', weights_only=True))
            self.secondary.eval().requires_grad_(False).to(device)
            from vibewarp.core.disco_optimizations import SecondaryPrecision
            self.secondary = SecondaryPrecision(self.secondary, d.secondary_precision, device)
        if d.merge_loras:
            merge_adapter_weights(self.model, d.lora_strength if self.has_turbo else 0., d.quality_lora_strength if self.quality_path else 0.)
            self._merged_strengths = (d.lora_strength, d.quality_lora_strength)
        self.diffusion=diffusion
        if d.sampler != 'distilled' or d.distilled_guided:
            from vibewarp.core.disco import GuidanceEncoder
            from vibewarp.core.diffusion import get_frame_schedule
            self.diffusion=self.diffusion_for_steps(get_frame_schedule(0,config)['steps'])
            self.encoders=[GuidanceEncoder(name,models_root(config)/'disco',device)
                           for name in config.disco.encoders]
            metadata=[value for value in (self.adapter_metadata,self.quality_config) if value]
            self.model=GuidedAdapterModel(self.model,all(value.get('precision','fp16')=='fp16' for value in metadata)).eval()

    def diffusion_for_steps(self, steps):
        from vibewarp.core.disco_repeated import validate_grid
        grid = self.config.disco.timestep_grid
        return guided_diffusion(validate_grid(grid,steps) if grid else rescaled_grid(self.grid,steps))

    def sample_tensor(self, seed, *, adapter='student', callback=None, lora_strength=1.0, randomness=0.0, frame_num=0, wiggle=0, direction=None):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn((1,3,512,512), generator=generator, device=self.device, dtype=torch.float32)
        if randomness or wiggle:
            from vibewarp.core.noise_variation import vary_fixed_noise
            noise = vary_fixed_noise(noise,randomness,seed,frame_num,wiggle=wiggle,direction=direction)
        precision = torch.autocast('cuda', dtype=torch.float16) if self.device.type == 'cuda' else nullcontext()
        with torch.inference_mode(), precision:
            result = None
            for step, result in enumerate(rollout(self.model, self.schedule, noise, self.grid,
                                                 adapter=adapter,strength=lora_strength), 1):
                if callback:
                    callback(step, result)
            return result

    def render(self, config, init_image, prompt, negative_prompt, seed, *, frame_num=0, preview_path=None, **kwargs):
        d=config.disco
        if getattr(self, '_merged_strengths', None) is not None and self._merged_strengths != (d.lora_strength,d.quality_lora_strength):
            raise ValueError('Merged LoRA strengths changed; reload the backend')
        if d.sampler != 'distilled' or d.distilled_guided:
            # Enclose the entire render, including checkpoint recomputation in
            # guidance backward. A forward-only adapter context would be wrong.
            turbo=d.sampler == 'distilled'
            print(f'Adapter strengths: quality={d.quality_lora_strength if d.quality_lora_enabled else 0:g}, '
                  f'turbo={d.lora_strength if turbo else 0:g}')
            with use_adapter(self.model,'student' if turbo else None,strength=d.lora_strength if turbo else 0.,
                             quality_strength=d.quality_lora_strength if d.quality_lora_enabled else 0.):
                return super().render(config,init_image,prompt,negative_prompt,seed,
                    frame_num=frame_num,preview_path=preview_path,**kwargs)
        if init_image.size != (512,512) or prompt.strip() or negative_prompt.strip():
            raise ValueError('The distilled adapter currently supports unconditional 512px generation only')
        if kwargs.get('mask') is not None or kwargs.get('guidance_image') is not None or kwargs.get('strength') is not None:
            raise ValueError('Guided, masked and strength-overridden few-step sampling is unvalidated')
        from vibewarp.core.disco import tensor_image
        def progress(step, result):
            event = dict(step=step, total_steps=len(self.grid)-1, frame=frame_num)
            if preview_path and config.disco.preview_interval > 0 and step % config.disco.preview_interval == 0:
                path = Path(preview_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix('.tmp.png')
                tensor_image(result).save(temporary)
                temporary.replace(path)
                event['preview'] = str(path)
            render_event(**event)
        options = {}
        if d.fixed_seed and (config.diffusion.fixed_seed_randomness or config.diffusion.fixed_seed_wiggle):
            options.update(randomness=config.diffusion.fixed_seed_randomness,frame_num=frame_num,
                           wiggle=config.diffusion.fixed_seed_wiggle,
                           direction=config.diffusion.fixed_seed_wiggle_direction)
        return tensor_image(self.sample_tensor(seed, callback=progress,lora_strength=config.disco.lora_strength,**options))


@torch.no_grad()
def merge_adapter_weights(model, turbo_strength, quality_strength):
    """Merge fixed custom Linear/Conv adapters in FP32; retain FP32 output head."""
    for layer in model.modules():
        if not isinstance(layer, AdapterLayer):
            continue
        for branch, strength, scale in [('student',turbo_strength,layer.scale), ('quality',quality_strength,layer.quality_scale)]:
            if not strength or branch not in layer.adapters:
                continue
            down,up=layer.adapters[branch]
            delta=up.weight.float().reshape(up.weight.shape[0],-1) @ down.weight.float().reshape(down.weight.shape[0],-1)
            layer.base.weight.add_((delta.reshape_as(layer.base.weight)*(strength*scale)).to(layer.base.weight.dtype))
        layer.active=None
        layer.quality_strength=0.
        layer._merged=True
