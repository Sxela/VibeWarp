"""Pixel-space adaptation of the shared per-step MultiDiffusion wrapper.

Reference: MagoStudio/ComfyUI-MagoNodes/wan_tiled_sampler.py (GPL-3.0).
Disco needs 64-pixel alignment and epsilon/learned-variance conversion rather
than WAN latent/conditioning handling. This mode changes the denoising field;
it is not lossless operation chunking.
"""
import torch
import torch.nn.functional as F

from vibewarp.core.sampling import TiledModelFn, alpha_sigma_to_t
from vibewarp.cancellation import raise_if_cancelled


def prepare_disco_denoisers(model, secondary, diffusion, config, skip, *, progressive=False):
    """Return per-render model callables without patching shared model weights.

    Masks and guidance losses cover the current whole-frame sampling grid.
    Recomputed tile activations bound secondary/no-secondary guidance backward
    memory instead of retaining every tile's neural network graph at once.
    """
    times = [value * 1000 / diffusion.original_num_steps for value in diffusion.timestep_map]

    def index_for(t):
        time = float(t[0])
        return min(range(len(times)), key=lambda i: abs(times[i] - time))

    def alpha_sigma(t, x):
        index = index_for(t)
        return (x.new_tensor(float(diffusion.sqrt_alphas_cumprod[index])),
                x.new_tensor(float(diffusion.sqrt_one_minus_alphas_cumprod[index])))

    def secondary_prediction(x, t, **kwargs):
        raise_if_cancelled()
        alpha, sigma = alpha_sigma(t, x)
        return secondary(x, alpha_sigma_to_t(alpha, sigma).expand(x.shape[0])).pred

    options = config.diffusion
    active = options.tiled_sampler or any(scale < 100 for scale in options.sampler_scale_schedule.values())
    if not active:
        return model, secondary_prediction if secondary is not None else None

    def denoise(x, t, **kwargs):
        raise_if_cancelled()
        return model(x, t, **kwargs)

    def restore_epsilon(prediction, small_x, full_x, t):
        # Only subsample noisy inputs. Restore a *clean* prediction before
        # recovering full-resolution epsilon, so high-frequency noise is not
        # mistaken for signal when the sampler keeps its full-resolution state.
        epsilon, variance = prediction.chunk(2, dim=1)
        alpha, sigma = alpha_sigma(t, small_x)
        clean = (small_x - sigma * epsilon) / alpha
        clean = F.interpolate(clean, size=full_x.shape[-2:], mode='bilinear', align_corners=False)
        epsilon = (full_x - alpha * clean) / sigma.clamp_min(1e-8)
        variance = F.interpolate(variance, size=full_x.shape[-2:], mode='bilinear', align_corners=False)
        return torch.cat((epsilon, variance), dim=1)

    common = dict(tile_size=options.sampler_tile_size,
        overlap_h=options.sampler_tile_overlap, overlap_w=options.sampler_tile_overlap,
        split_threshold=options.sampler_tile_split_threshold,
        scale_schedule={0:100} if progressive else options.sampler_scale_schedule,
        enable_tiling=options.tiled_sampler,
        min_scale_size=options.sampler_scale_min_size, spatial_alignment=64,
        pixels_per_unit=1, noise_resize_mode='nearest-exact', checkpoint_tiles=True)
    denoiser = TiledModelFn(denoise, prediction_channels=6,
                          restore_scaled=None if progressive else restore_epsilon, **common)
    preview = TiledModelFn(secondary_prediction, **common) if secondary is not None else None
    # Resolve by actual mapped timestep, including PLMS bootstrap evaluations.
    # A sentinel after time zero gives the last denoising step its own interval.
    schedule = torch.tensor(list(reversed(times[:len(times)-skip])) + [-1.0])
    for wrapper in (denoiser, preview):
        if wrapper is not None:
            wrapper.set_sigma_schedule(schedule)
    return denoiser, preview
