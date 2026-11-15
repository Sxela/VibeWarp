"""Progressive pixel sampling: resize state only at explicit scale boundaries."""
import math

import torch
import torch.nn.functional as F

from vibewarp.cancellation import raise_if_cancelled
from vibewarp.core.sampling import TiledModelFn


def progressive_sizes(options, final_size, steps):
    planner = TiledModelFn(None, spatial_alignment=64, pixels_per_unit=1,
                           min_scale_size=options.sampler_scale_min_size)
    schedule = sorted(options.sampler_scale_schedule.items())
    sizes = []
    for step in range(steps):
        scale = schedule[0][1]
        for key, value in schedule:
            if key > step:
                break
            scale = value
        sizes.append(planner._scaled_size(*final_size, scale))
    if any(b[0] < a[0] or b[1] < a[1] for a, b in zip(sizes, sizes[1:])):
        raise ValueError('Disco progressive scale sizes must not decrease')
    if sizes[-1] != tuple(final_size):
        raise ValueError(f'Disco scale schedule must reach full size within the {steps} executed steps')
    return sizes


def stage_denoiser(model, size, final_size):
    # Coarse stages are whole-frame passes. Only the final grid uses tiles.
    if tuple(size) != tuple(final_size) and isinstance(model, TiledModelFn):
        return model.model_fn
    return model


def refine_noise(coarse, size):
    """Add Gaussian detail while retaining each coarse cell's noise component.

    Nearest cell assignment partitions the fine grid into disjoint groups.
    Project fresh noise out of each group's constant direction, then fill that
    direction with the old noise divided by sqrt(group size). For independent
    unit Gaussian coarse noise, the resulting fine noise is unit Gaussian too.
    This preserves the coarse component, unlike simply drawing a fresh field.
    """
    if tuple(coarse.shape[-2:]) == tuple(size):
        return coarse
    height, width = coarse.shape[-2:]
    if size[0] < height or size[1] < width:
        raise ValueError('Noise refinement only supports growing spatial grids')
    fresh = torch.randn((*coarse.shape[:-2], *size), device=coarse.device, dtype=coarse.dtype)
    if size[0] % height == 0 and size[1] % width == 0:
        ratio = (size[0] // height, size[1] // width)
        means = F.avg_pool2d(fresh, ratio, ratio)
        expand = lambda value: F.interpolate(value, size, mode='nearest-exact')
        return fresh - expand(means) + expand(coarse) / math.sqrt(ratio[0] * ratio[1])
    ids = torch.arange(height * width, device=coarse.device, dtype=torch.float32).reshape(1, 1, height, width)
    ids = F.interpolate(ids, size, mode='nearest-exact').long().flatten()
    counts = torch.bincount(ids, minlength=height * width).to(coarse.dtype)
    flat = fresh.flatten(2)
    indices = ids.expand(*flat.shape[:-1], -1)
    sums = torch.zeros_like(coarse.flatten(2)).scatter_add_(2, indices, flat)
    means = (sums / counts).gather(2, indices)
    inherited = (coarse.flatten(2) / counts.sqrt()).gather(2, indices)
    return (flat - means + inherited).reshape_as(fresh)


def resize_state(state, clean, size, alpha, sigma):
    """Refine one noisy state at its current diffusion time, once per boundary."""
    epsilon = (state - alpha * clean) / max(sigma, 1e-8)
    clean = F.interpolate(clean, size, mode='bilinear', align_corners=False)
    return alpha * clean + sigma * refine_noise(epsilon, size)


def progressive_disco_samples(diffusion, model, init, noise, sizes, *, skip,
                              sampler, cond_fn, eta=0., mask=None, inpainting_stop=0):
    """Run the existing DDIM/PLMS step functions on a persistent current grid."""
    full_size = init.shape[-2:]
    shape = sizes[0]
    current_init = F.interpolate(init, shape, mode='area')
    current_noise = F.interpolate(noise, shape, mode='nearest-exact')
    first = diffusion.num_timesteps - skip - 1
    timestep = lambda index: torch.full((init.shape[0],), index, device=init.device, dtype=torch.long)
    state = diffusion.q_sample(current_init, timestep(first), current_noise)
    full_mask = None
    if mask is not None:
        full_mask = torch.as_tensor(mask, device=init.device, dtype=init.dtype).permute(2, 0, 1)[None]
    old_out = None
    previous_clean = None
    with torch.no_grad():
        for step, index in enumerate(range(first, -1, -1)):
            raise_if_cancelled()
            if sizes[step] != shape:
                shape = sizes[step]
                state = resize_state(state, previous_clean, shape,
                    float(diffusion.sqrt_alphas_cumprod[index]),
                    float(diffusion.sqrt_one_minus_alphas_cumprod[index]))
                current_init = F.interpolate(init, shape, mode='area')
                current_noise = F.interpolate(noise, shape, mode='nearest-exact')
                old_out = None  # Adams-Bashforth history belongs to the old grid.
            t = timestep(index)
            if full_mask is not None and index >= inpainting_stop:
                current_mask = F.interpolate(full_mask, shape, mode='nearest-exact')
                renoised = diffusion.q_sample(current_init, t, current_noise)
                state = state * (1-current_mask) + renoised * current_mask
            denoiser = stage_denoiser(model, shape, full_size)
            options = dict(clip_denoised=False, cond_fn=cond_fn, model_kwargs={})
            if sampler == 'ddim':
                out = diffusion.ddim_sample(denoiser, state, t, eta=eta, **options)
            else:
                final_restart = index == 0 and old_out is None
                out = diffusion.plms_sample(denoiser, state, t,
                    old_out={'old_eps': []} if final_restart else old_out,
                    order=1 if final_restart else 2, **options)
                old_out = out
            state, previous_clean = out['sample'], out['pred_xstart']
            yield out
