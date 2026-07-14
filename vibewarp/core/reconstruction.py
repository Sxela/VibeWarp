"""Reconstruction noise via DDIM inversion for AnimateDiff batch boundaries.

Implements the notebook's `find_noise_for_image_sigma_adjustment` and
`get_predicted_noise_adiff` functions. The core idea: given a clean image,
run diffusion forward (DDIM inversion) to find the noise that would
reconstruct it, then blend with random noise for temporal consistency.

This improves quality at AnimateDiff batch overlap boundaries by giving
the sampler noise that is "aware" of the previous batch's output rather
than pure random noise.
"""

import copy
import hashlib
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import trange

from vibewarp.config import ReconstructionNoiseConfig


def find_noise_for_image(
    sd_model: Any,
    model_wrap: Any,
    init_latent: torch.Tensor,
    prompt: str,
    neg_prompt: str = '',
    cfg_scale: float = 1.0,
    steps: int = 25,
    image_conditioning: Optional[Any] = None,
    separate_uncond: bool = True,
    steps_cutoff: int = 0,
    sliding_context: bool = False,
    context_schedule: Optional[List[List[List[int]]]] = None,
    cond_cache: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """DDIM inversion: find the noise that would reconstruct init_latent.

    Runs the diffusion process forward (low sigma → high sigma) to
    reconstruct what noise would produce the given latent. This is the
    clean equivalent of the notebook's `find_noise_for_image_sigma_adjustment`.

    Args:
        sd_model: loaded SD model (with apply_model, get_learned_conditioning)
        model_wrap: k-diffusion model wrapper (has get_sigmas, sigma_to_t, get_scalings)
        init_latent: (B, 4, H/8, W/8) latent to invert
        prompt: text prompt for conditioning
        neg_prompt: negative prompt
        cfg_scale: classifier-free guidance scale (1.0 = no guidance)
        steps: number of inversion steps
        image_conditioning: optional ControlNet conditioning dict
        separate_uncond: process uncond/cond separately (faster for batched)
        steps_cutoff: stop inversion early after this many steps (0 = all)
        sliding_context: use sliding context windows during inversion (rec_sliding_ctx)
        context_schedule: pre-computed context schedule for sliding_context mode

    Returns:
        Reconstructed noise tensor, same shape as init_latent.
    """
    if cond_cache is None:
        cond_cache = {}
    need_clip = prompt not in cond_cache or (cfg_scale != 1.0 and neg_prompt not in cond_cache)
    if need_clip:
        has_cond_stage = hasattr(sd_model, 'cond_stage_model')
        if has_cond_stage:
            from vibewarp.core.diffusion import _move_to_cuda, _offload_to_cpu
            _move_to_cuda(sd_model.cond_stage_model)
        with torch.inference_mode(), torch.amp.autocast('cuda'):
            if prompt not in cond_cache:
                cond_cache[prompt] = sd_model.get_learned_conditioning([prompt])
            if cfg_scale != 1.0 and neg_prompt not in cond_cache:
                cond_cache[neg_prompt] = sd_model.get_learned_conditioning([neg_prompt])
        if has_cond_stage:
            _offload_to_cpu(sd_model.cond_stage_model)
    cond = cond_cache[prompt]
    uncond = cond_cache.get(neg_prompt) if cfg_scale != 1.0 else None

    x = init_latent
    s_in = x.new_ones([x.shape[0]])

    # Determine denoiser type based on model parameterization
    is_v_param = getattr(sd_model, 'parameterization', 'eps') == 'v'

    # Get sigmas in forward (ascending) order for inversion
    sigmas = model_wrap.get_sigmas(steps).flip(0)
    skip = 1 if is_v_param else 0

    # Expand conditioning to match batch size
    if cond.shape[0] < x.shape[0]:
        cond = cond.repeat(x.shape[0], 1, 1)
    if uncond is not None and uncond.shape[0] < x.shape[0]:
        uncond = uncond.repeat(x.shape[0], 1, 1)

    # Truncate conditioning to 77 tokens max
    if cond.shape[1] > 77:
        cond = cond[:, :77, :]
    if uncond is not None and uncond.shape[1] > 77:
        uncond = uncond[:, :77, :]

    # Pre-compute context schedule for sliding context mode
    ctx_schedule = None
    if sliding_context and context_schedule is not None:
        ctx_schedule = [windows[:] for windows in context_schedule]

    with torch.inference_mode(), torch.amp.autocast('cuda'):
        for i in range(1, len(sigmas)):
            sigma_in = sigmas[i - 1] * s_in

            # Get scaling factors
            scalings = model_wrap.get_scalings(sigma_in)
            c_out, c_in = [_append_dims(k, x.ndim) for k in scalings[skip:]]

            if i == 1:
                t = model_wrap.sigma_to_t(sigmas[i] * s_in)
            else:
                t = model_wrap.sigma_to_t(sigma_in)

            if separate_uncond or cfg_scale == 1:
                x_scaled = x * c_in

                if sliding_context and ctx_schedule and len(ctx_schedule) > 0:
                    # Sliding context mode: process in context windows
                    step_windows = ctx_schedule.pop(0) if ctx_schedule else [[list(range(x.shape[0]))]]
                    eps_c = _apply_model_sliding_ctx(
                        sd_model, x_scaled, t, cond, image_conditioning,
                        step_windows, is_uc=False,
                    )
                    if cfg_scale != 1 and uncond is not None:
                        eps_uc = _apply_model_sliding_ctx(
                            sd_model, x_scaled, t, uncond, image_conditioning,
                            step_windows, is_uc=True,
                        )
                        denoised_c = x + eps_c * c_out
                        denoised_uc = x + eps_uc * c_out
                        denoised = denoised_uc + (denoised_c - denoised_uc) * cfg_scale
                        del eps_uc, denoised_c, denoised_uc
                    else:
                        denoised = x + eps_c * c_out
                else:
                    # Standard mode: process all frames at once
                    _set_uc_mask(sd_model, torch.ones(x_scaled.shape[0], device=x_scaled.device))
                    cond_dict = _build_cond_dict(cond, image_conditioning, is_uc=False)
                    eps_c = sd_model.apply_model(x_scaled, t, cond=cond_dict)

                    if cfg_scale != 1 and uncond is not None:
                        _set_uc_mask(sd_model, torch.zeros(x_scaled.shape[0], device=x_scaled.device))
                        uc_dict = _build_cond_dict(uncond, image_conditioning, is_uc=True)
                        eps_uc = sd_model.apply_model(x_scaled, t, cond=uc_dict)
                        denoised_c = x + eps_c * c_out
                        denoised_uc = x + eps_uc * c_out
                        denoised = denoised_uc + (denoised_c - denoised_uc) * cfg_scale
                        del eps_uc, denoised_c, denoised_uc
                    else:
                        denoised = x + eps_c * c_out

                del eps_c
            else:
                # Process cond+uncond together (original notebook path)
                x_in = torch.cat([x] * 2)
                sigma_in_2 = torch.cat([sigma_in] * 2)
                cond_in = torch.cat([uncond, cond]) if uncond is not None else cond
                # Notebook: uncond rows marked 0 (first half of [uncond, cond])
                uc_mask = torch.ones(cond_in.shape[0], device=cond_in.device)
                if uncond is not None:
                    uc_mask[:uncond.shape[0]] = 0
                _set_uc_mask(sd_model, uc_mask)
                c_out_2, c_in_2 = [_append_dims(k, x_in.ndim) for k in model_wrap.get_scalings(sigma_in_2)[skip:]]

                if i == 1:
                    t_2 = model_wrap.sigma_to_t(torch.cat([sigmas[i] * s_in] * 2))
                else:
                    t_2 = model_wrap.sigma_to_t(sigma_in_2)

                cond_dict = {"c_crossattn": [cond_in]}
                if image_conditioning is not None:
                    cond_dict['c_concat'] = image_conditioning

                eps = sd_model.apply_model(x_in * c_in_2, t_2, cond=cond_dict)

                if cfg_scale != 1:
                    denoised_uncond, denoised_cond = (x_in + eps * c_out_2).chunk(2)
                    denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cfg_scale
                    del denoised_uncond, denoised_cond
                else:
                    denoised = (x_in + eps * c_out_2).chunk(2)[0]

                del x_in, eps

            # Euler step in forward direction
            if i == 1:
                d = (x - denoised) / (2 * sigmas[i])
            else:
                d = (x - denoised) / sigmas[i - 1]

            dt = sigmas[i] - sigmas[i - 1]
            x = x + d * dt

            del denoised, d

            # Early cutoff
            if steps_cutoff > 0 and i >= steps - steps_cutoff:
                break

    # Normalize by final sigma
    return x / sigmas[-1]


def _set_uc_mask(sd_model: Any, mask: torch.Tensor) -> None:
    """Set uc_mask_shape on the UNet (0 = uncond row, 1 = cond row).

    Matches the notebook's `sd_model.model.diffusion_model.uc_mask_shape = ...`;
    consumed by IP-Adapter hooks. No-op if the model has no UNet attribute
    (e.g. mocks in tests).
    """
    diffusion_model = getattr(getattr(sd_model, 'model', None), 'diffusion_model', None)
    if diffusion_model is not None:
        diffusion_model.uc_mask_shape = mask


def _apply_model_sliding_ctx(
    sd_model: Any,
    x_scaled: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor,
    image_conditioning: Optional[Any],
    windows: List[List[int]],
    is_uc: bool = False,
) -> torch.Tensor:
    """Apply model in sliding context windows, averaging overlapping outputs.

    This is the rec_sliding_ctx path from the notebook: instead of processing
    all frames at once during DDIM inversion, we process them in overlapping
    context windows and average the results. This maintains temporal coherence
    in the reconstructed noise.

    Args:
        sd_model: SD model
        x_scaled: scaled input (B, C, H, W)
        t: timesteps (B,)
        cond: conditioning tensor (B, 77, D)
        image_conditioning: optional ControlNet conditioning
        windows: list of frame index lists for this step
        is_uc: whether this is the unconditional pass

    Returns:
        Averaged model output, same shape as x_scaled.
    """
    batch_size = x_scaled.shape[0]
    eps_final = torch.zeros_like(x_scaled)
    out_count = torch.zeros((batch_size, 1, 1, 1), device=x_scaled.device)

    for ids in windows:
        valid_ids = [idx for idx in ids if idx < batch_size]
        if not valid_ids:
            continue

        # Subset inputs for this window
        x_win = x_scaled[valid_ids]
        t_win = t[valid_ids] if t.ndim > 0 and t.shape[0] > 1 else t

        # Subset conditioning
        cond_win = cond[valid_ids] if cond.shape[0] > 1 else cond
        cond_dict = _build_cond_dict(cond_win, image_conditioning, is_uc=is_uc)

        # Subset image conditioning per window
        if image_conditioning is not None and isinstance(image_conditioning, dict):
            img_cond_win = {}
            for k, v in image_conditioning.items():
                if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] > 1:
                    img_cond_win[k] = v[valid_ids]
                else:
                    img_cond_win[k] = v
            cond_dict = _build_cond_dict(cond_win, img_cond_win, is_uc=is_uc)

        # Notebook: per-window uc mask — zeros for the uncond pass, ones for cond
        mask_val = torch.zeros if is_uc else torch.ones
        _set_uc_mask(sd_model, mask_val(x_win.shape[0], device=x_win.device))

        eps_win = sd_model.apply_model(x_win, t_win, cond=cond_dict)

        # Accumulate
        for j, idx in enumerate(valid_ids):
            eps_final[idx] += eps_win[j]
            out_count[idx] += 1

    # Average overlapping outputs
    return eps_final / out_count.clamp(min=1)


def get_predicted_noise_adiff_batched(
    sd_model: Any,
    model_wrap: Any,
    frame_paths: List[str],
    frame_idxs: List[int],
    random_noise: torch.Tensor,
    config: ReconstructionNoiseConfig,
    prompt: str = '',
    neg_prompt: str = '',
    image_size: Tuple[int, int] = (512, 512),
    image_conditioning: Optional[Any] = None,
    context_length: int = 16,
    context_overlap: int = 4,
    diffusion_steps: int = 25,
    cond_cache: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Batched DDIM inversion: encode all frames at once and invert as a batch.

    This is the notebook's `batched_adiff_rec_noise` mode. Instead of
    processing frames one-by-one, all overlap frames are encoded to latents
    and passed through `find_noise_for_image` as a single batched call.
    This is faster but requires batch_length == context_length.

    Args:
        sd_model: loaded SD model
        model_wrap: k-diffusion model wrapper
        frame_paths: list of frame image paths to invert
        frame_idxs: frame indices
        random_noise: (B, 4, H/8, W/8) random noise to blend with
        config: reconstruction noise settings
        prompt: text prompt for reconstruction pass
        neg_prompt: negative prompt
        image_size: (width, height) for image loading
        image_conditioning: optional ControlNet conditioning
        context_length: AnimateDiff context length (for sliding ctx schedule)
        context_overlap: AnimateDiff context overlap
        diffusion_steps: actual diffusion steps from schedule (for rec_steps_pct)

    Returns:
        Blended noise tensor (B, 4, H/8, W/8).
    """
    from vibewarp.core.diffusion import load_img_for_sd, encode_image_to_latent

    W, H = image_size
    rec_steps = max(1, int(diffusion_steps * config.steps_pct))
    randomness = config.randomness

    # Encode all frames to latents as a batch
    latents = []
    for frame_path in frame_paths:
        if not os.path.exists(frame_path):
            latents.append(torch.randn(1, 4, H // 8, W // 8))
            continue
        img_tensor = load_img_for_sd(frame_path, size=(W, H)).cuda()
        with torch.inference_mode(), torch.amp.autocast('cuda'):
            lat = encode_image_to_latent(sd_model, img_tensor)
        latents.append(lat.cpu())

    batched_latent = torch.cat(latents).cuda()

    # Build context schedule for sliding context mode if enabled
    sliding_ctx = getattr(config, 'sliding_context', False)
    ctx_sched = None
    if sliding_ctx:
        from vibewarp.core.animatediff import make_context_schedule
        ctx_sched = make_context_schedule(
            total_length=len(frame_paths),
            context_length=context_length,
            overlap=context_overlap,
            steps=rec_steps + 1,
        )

    # Single batched DDIM inversion
    img_cond = None
    if image_conditioning is not None and not config.disable_controlnets:
        img_cond = image_conditioning

    rec_noise = find_noise_for_image(
        sd_model=sd_model,
        model_wrap=model_wrap,
        init_latent=batched_latent,
        prompt=prompt,
        neg_prompt=neg_prompt,
        cfg_scale=config.cfg_scale,
        steps=rec_steps,
        image_conditioning=img_cond,
        separate_uncond=config.separate_uncond,
        steps_cutoff=config.steps_cutoff,
        sliding_context=sliding_ctx,
        context_schedule=ctx_sched,
        cond_cache=cond_cache,
    ).cpu()

    # Blend with random noise
    combined = _blend_noise(rec_noise, random_noise, randomness)
    return combined


def get_predicted_noise_adiff(
    sd_model: Any,
    model_wrap: Any,
    frame_paths: List[str],
    frame_idxs: List[int],
    random_noise: torch.Tensor,
    config: ReconstructionNoiseConfig,
    prompt: str = '',
    neg_prompt: str = '',
    image_size: Tuple[int, int] = (512, 512),
    image_conditioning: Optional[Any] = None,
    cache_dir: str = '',
    diffusion_steps: int = 25,
    cond_cache: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Reconstruct noise for AnimateDiff frames via DDIM inversion.

    This is the clean equivalent of the notebook's `get_predicted_noise_adiff`.
    For each frame, encodes the init image to latent space, runs DDIM inversion,
    and blends reconstructed noise with random noise.

    Args:
        sd_model: loaded SD model
        model_wrap: k-diffusion model wrapper
        frame_paths: list of frame image paths to invert
        frame_idxs: frame indices (for caching)
        random_noise: (B, 4, H/8, W/8) random noise to blend with
        config: reconstruction noise settings
        prompt: text prompt for reconstruction pass
        neg_prompt: negative prompt
        image_size: (width, height) for image loading
        image_conditioning: optional ControlNet conditioning
        cache_dir: directory for caching noise tensors

    Returns:
        Blended noise tensor (B, 4, H/8, W/8).
    """
    from vibewarp.core.diffusion import load_img_for_sd, encode_image_to_latent

    W, H = image_size
    rec_steps = max(1, int(diffusion_steps * config.steps_pct))
    rec_cfg = config.cfg_scale
    randomness = config.randomness

    # Check cache
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    noise_batch = []
    for i, frame_path in enumerate(frame_paths):
        frame_num = frame_idxs[i] if i < len(frame_idxs) else i

        # Try loading from cache
        cached = _load_cached_noise(cache_dir, frame_num, config) if cache_dir else None
        if cached is not None:
            noise_batch.append(cached)
            continue

        if not os.path.exists(frame_path):
            # No frame to invert — use random noise
            noise_batch.append(random_noise[i:i+1] if i < random_noise.shape[0] else torch.randn_like(random_noise[0:1]))
            continue

        # Encode frame to latent
        img_tensor = load_img_for_sd(frame_path, size=(W, H)).cuda()
        with torch.inference_mode(), torch.amp.autocast('cuda'):
            frame_latent = encode_image_to_latent(sd_model, img_tensor)

        # DDIM inversion
        img_cond = None
        if image_conditioning is not None and not config.disable_controlnets:
            # Subset conditioning for this frame
            if isinstance(image_conditioning, dict):
                img_cond = {k: v[i:i+1] if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] > i else v
                           for k, v in image_conditioning.items()}
            else:
                img_cond = image_conditioning

        rec_noise = find_noise_for_image(
            sd_model=sd_model,
            model_wrap=model_wrap,
            init_latent=frame_latent,
            prompt=prompt,
            neg_prompt=neg_prompt,
            cfg_scale=rec_cfg,
            steps=rec_steps,
            image_conditioning=img_cond,
            separate_uncond=config.separate_uncond,
            steps_cutoff=config.steps_cutoff,
            cond_cache=cond_cache,
        ).cpu()

        # Blend with random noise
        rand = random_noise[i:i+1] if i < random_noise.shape[0] else torch.randn_like(rec_noise)
        combined = _blend_noise(rec_noise, rand, randomness)

        # Cache
        if cache_dir:
            _save_cached_noise(cache_dir, frame_num, config, combined)

        noise_batch.append(combined)

    return torch.cat(noise_batch, dim=0)


def _blend_noise(
    rec_noise: torch.Tensor,
    rand_noise: torch.Tensor,
    randomness: float,
) -> torch.Tensor:
    """Blend reconstructed noise with random noise.

    Uses the notebook's blending formula that preserves noise magnitude:
    combined = ((1 - r) * rec + r * rand) / sqrt(r^2 + (1-r)^2)
    """
    if randomness <= 0:
        return rec_noise
    if randomness >= 1:
        return rand_noise

    combined = (1 - randomness) * rec_noise + randomness * rand_noise
    normalizer = (randomness ** 2 + (1 - randomness) ** 2) ** 0.5
    return combined / normalizer


def _append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Append dimensions to tensor to match target number of dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target is {target_dims}")
    return x[(...,) + (None,) * dims_to_append]


def _build_cond_dict(
    cond: torch.Tensor,
    image_conditioning: Optional[Any] = None,
    is_uc: bool = False,
) -> Dict[str, Any]:
    """Build conditioning dict for sd_model.apply_model."""
    d = {"c_crossattn": [cond]}
    if image_conditioning is not None:
        if isinstance(image_conditioning, dict):
            if is_uc:
                d['c_concat'] = {k: torch.zeros_like(v) if isinstance(v, torch.Tensor) else v
                                for k, v in image_conditioning.items()}
            else:
                d['c_concat'] = image_conditioning
        else:
            if is_uc:
                d['c_concat'] = [torch.zeros_like(image_conditioning)]
            else:
                d['c_concat'] = [image_conditioning]
    return d


def _noise_cache_key(config: ReconstructionNoiseConfig) -> str:
    """Generate a short hash of reconstruction settings for cache filename."""
    settings = {
        'cfg': config.cfg_scale,
        'steps_pct': config.steps_pct,
        'randomness': config.randomness,
        'source': config.source,
        'cutoff': config.steps_cutoff,
        'scale': config.noise_scale,
    }
    h = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
    return h


def _load_cached_noise(
    cache_dir: str, frame_num: int, config: ReconstructionNoiseConfig,
) -> Optional[torch.Tensor]:
    """Load cached reconstructed noise if it exists."""
    if not cache_dir:
        return None
    key = _noise_cache_key(config)
    path = os.path.join(cache_dir, f"{key}_{frame_num:06d}.pt")
    if os.path.exists(path):
        return torch.load(path, map_location='cpu')
    return None


def _save_cached_noise(
    cache_dir: str, frame_num: int, config: ReconstructionNoiseConfig,
    noise: torch.Tensor,
) -> None:
    """Save reconstructed noise to cache."""
    if not cache_dir:
        return
    key = _noise_cache_key(config)
    path = os.path.join(cache_dir, f"{key}_{frame_num:06d}.pt")
    torch.save(noise, path)
