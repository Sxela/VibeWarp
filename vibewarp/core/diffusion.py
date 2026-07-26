"""Main diffusion loop — do_run() orchestration.

Extracts the core rendering pipeline from the notebook's cell 20 do_run()
and cell 55 orchestration into clean, testable functions.

Key simplification vs notebook:
- RenderContext replaces ~100 globals
- render_frame() is a pure function (context in, image out)
- run_frames() is the frame loop orchestrator
"""

import gc
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from vibewarp.cancellation import raise_if_cancelled
from vibewarp.config import RunConfig
from vibewarp.utils.scheduling import get_scheduled_arg, interpolate_array

# Lazy imports for inter-frame processing (avoid circular / heavy imports at module level)
_color_match_fn = None
_brightness_fn = None


# ---- Dataclasses ----

@dataclass
class FrameState:
    """Mutable state carried between frames during a run."""
    frame_num: int = 0
    seed: int = 0
    init_image: Optional[str] = None
    init_latent: Optional[torch.Tensor] = None
    prev_frame: Optional[Image.Image] = None
    prev_latent: Optional[torch.Tensor] = None
    start_code: Optional[torch.Tensor] = None
    first_latent: Optional[torch.Tensor] = None


@dataclass
class RenderContext:
    """All dependencies needed by the render loop, passed explicitly.

    This replaces the notebook's globals() pattern.
    """
    config: RunConfig = field(default_factory=RunConfig)

    # Model references (set during initialization)
    sd_model: Any = None
    model_wrap: Any = None
    model_wrap_cfg: Any = None
    loaded_controlnets: Dict[str, Any] = field(default_factory=dict)
    loaded_ipadapters: Dict[str, Any] = field(default_factory=dict)
    raft_model: Any = None
    clip_vision_model: Any = None  # CLIP vision model for IP-Adapter encoding
    # AnimateDiff MotionWrapper — the renderer must tell it the context length.
    motion_module: Any = None
    # Flux edit pipeline (diffusers) — set for model_version='flux2_klein_edit'
    # instead of sd_model/model_wrap. See core/flux.py.
    edit_backend: Any = None
    flux_pipe: Any = None

    # Sampler function (from k_diffusion.sampling)
    sampler_fn: Optional[Callable] = None

    # Paths
    batch_folder: str = ''
    video_frames_folder: str = ''
    flow_folder: str = ''

    # Callbacks
    save_frame_fn: Optional[Callable] = None
    progress_fn: Optional[Callable] = None

    # Internal: per-batch reconstruction noise override (set by run_frames_animatediff)
    _rec_noise_override: Optional[Dict[str, Any]] = None

    # Conditioning cache: {prompt_text: conditioning_tensor}
    _cond_cache: Dict[str, Any] = field(default_factory=dict)


# ---- Model offload helpers ----

def _offload_to_cpu(*modules: Any) -> None:
    """Move model modules to CPU. Call torch.cuda.empty_cache() at frame boundaries, not here."""
    for m in modules:
        if m is not None and hasattr(m, 'cpu'):
            m.cpu()


# Per-frame phase profiler. Toggle with env var VIBEWARP_PROFILE=1.
# Phases accumulate in ctx._phase_times; run_frames prints per-frame summary.
import contextlib as _contextlib

_PROFILE_ENABLED = os.environ.get('VIBEWARP_PROFILE', '0') == '1'


@_contextlib.contextmanager
def _phase(ctx: Any, name: str):
    """Time a phase and add elapsed seconds to ctx._phase_times[name]."""
    if not _PROFILE_ENABLED or ctx is None:
        yield
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        if not hasattr(ctx, '_phase_times') or ctx._phase_times is None:
            ctx._phase_times = {}
        ctx._phase_times[name] = ctx._phase_times.get(name, 0.0) + elapsed


def _move_to_cuda(*modules: Any) -> None:
    """Move model modules to CUDA."""
    for m in modules:
        if m is not None and hasattr(m, 'cuda'):
            m.cuda()


# ---- Image ↔ Latent helpers ----

def load_img_for_sd(path: str, size: Tuple[int, int]) -> torch.Tensor:
    """Load and preprocess an image for SD inference.

    Args:
        path: path to image file
        size: (width, height) to resize to

    Returns:
        Tensor of shape (1, 3, H, W) in range [-1, 1].
    """
    image = Image.open(path).convert("RGB")
    image = image.resize(size, resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return (2.0 * image - 1.0).half().cuda()


def encode_image_to_latent(sd_model: Any, image_tensor: torch.Tensor) -> torch.Tensor:
    """Encode a preprocessed image tensor to latent space via the VAE.

    Args:
        sd_model: loaded SD model with first_stage_model
        image_tensor: (1, 3, H, W) tensor in [-1, 1]

    Returns:
        Latent tensor (1, 4, H/8, W/8).
    """
    _move_to_cuda(sd_model.first_stage_model)
    with torch.inference_mode(), torch.amp.autocast('cuda'):
        result = sd_model.get_first_stage_encoding(
            sd_model.encode_first_stage(image_tensor.half().cuda())
        )
    _offload_to_cpu(sd_model.first_stage_model)
    return result


def decode_latent_to_image(sd_model: Any, latent: torch.Tensor) -> Image.Image:
    """Decode a latent tensor back to a PIL image.

    Args:
        sd_model: loaded SD model with first_stage_model
        latent: (1, 4, H/8, W/8) latent tensor

    Returns:
        PIL Image.
    """
    _move_to_cuda(sd_model.first_stage_model)
    with torch.inference_mode(), torch.amp.autocast('cuda'):
        x_samples = sd_model.decode_first_stage(latent.half().cuda())
    _offload_to_cpu(sd_model.first_stage_model)
    image = x_samples[0].float().add(1).div(2).clamp(0, 1)
    return TF.to_pil_image(image)


# ---- Sigma schedule helpers ----

def get_premature_sigma_min(steps, sigma_max, sigma_min_nominal, rho=7.0):
    """Get the 'sigma before sigma_min' from a slightly longer ramp.

    See: https://github.com/crowsonkb/k-diffusion/pull/23#issuecomment-1234872495
    """
    min_inv_rho = sigma_min_nominal ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    ramp = np.linspace(0, 1, steps)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return float(sigmas[-1])


def compute_sigmas(
    model_wrap: Any,
    steps: int,
    use_karras: bool = True,
    end_karras_ramp_early: bool = False,
) -> torch.Tensor:
    """Compute the noise sigma schedule for sampling.

    Args:
        model_wrap: k-diffusion model wrapper (has .sigmas attribute)
        steps: number of diffusion steps
        use_karras: whether to use Karras noise schedule
        end_karras_ramp_early: use premature sigma_min

    Returns:
        1D tensor of sigmas (length = steps + 1, last element is 0).
    """
    if use_karras:
        from vibewarp.vendor.k_diffusion.sampling import get_sigmas_karras

        rho = 7.0
        sigma_max = model_wrap.sigmas[-1].item()
        sigma_min_nominal = model_wrap.sigmas[0].item()

        if end_karras_ramp_early:
            sigma_min = get_premature_sigma_min(
                steps=steps + 1,
                sigma_max=sigma_max,
                sigma_min_nominal=sigma_min_nominal,
                rho=rho,
            )
        else:
            sigma_min = sigma_min_nominal

        sigmas = get_sigmas_karras(
            n=steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            device='cuda',
        ).float()
    else:
        sigmas = model_wrap.get_sigmas(steps).float()

    return sigmas


# ---- Core sampling ----

def get_sampler_fn(sampler_name: str) -> Callable:
    """Look up a k-diffusion sampler function by name.

    Args:
        sampler_name: e.g. 'sample_euler_ancestral', 'sample_dpm_2'

    Returns:
        The sampler callable.
    """
    from vibewarp.vendor.k_diffusion import sampling
    if not hasattr(sampling, sampler_name):
        raise ValueError(
            f"Unknown sampler '{sampler_name}'. "
            f"Available: sample_euler, sample_euler_ancestral, sample_heun, "
            f"sample_dpm_2, sample_dpmpp_2m, sample_dpmpp_sde, sample_lcm, etc."
        )
    return getattr(sampling, sampler_name)


@_contextlib.contextmanager
def _step_progress(total: int, desc: str):
    """A progress bar over the DIFFUSION STEPS of one sample.

    The outer bars count frames (or AnimateDiff batches), which tells you nothing while a
    single slow sample is running — with a tiled sampler a step can take seconds and the
    render just looks hung.

    CRITICAL: in this k-diffusion fork the callback's return value REPLACES the latent —
    every sampler does `x = callback({...}).float()` (that is how WarpFusion implements
    masked diffusion). A callback that returns None nulls x and the next step dies with
    "'NoneType' object has no attribute 'float'". So this MUST hand `x` back untouched.
    Standard k-diffusion ignores the return value, which is exactly why it is easy to get
    wrong here.
    """
    bar = tqdm(total=total, desc=desc, leave=False, unit='step')

    def callback(info):
        # The fork passes {'x', 'i', 'sigma', 'sigma_hat', 'denoised'}; 'i' is 0-based.
        if not isinstance(info, dict):
            return info
        step = info.get('i')
        bar.n = min(total, (step + 1) if step is not None else bar.n + 1)
        sigma = info.get('sigma')
        if sigma is not None:
            try:
                bar.set_postfix(sigma=f'{float(sigma):.2f}', refresh=False)
            except (TypeError, ValueError):
                pass
        bar.refresh()
        return info['x']          # pass the latent straight through — see above

    try:
        yield callback
    finally:
        bar.close()


def _configure_tiled_sampler(model_wrap_cfg: Any, config: RunConfig, sd_model: Any,
                             H: Optional[int] = None, W: Optional[int] = None) -> None:
    """Install/remove spatial tiling below CFG and AnimateDiff context handling.

    Logs the tile grid it actually resolved to. This runs per frame, so the log is emitted
    only when the configuration changes — otherwise it would print on every frame.
    """
    diffusion = config.diffusion
    if not diffusion.tiled_sampler:
        if getattr(model_wrap_cfg, '_tiled_inner_model', None) is not None:
            print('Tiled sampler: disabled')
        model_wrap_cfg._tiled_inner_model = None
        model_wrap_cfg._tiled_log_key = None
        return

    from vibewarp.core.sampling import TiledModelFn, tile_count_for_size
    tile_latent = max(1, diffusion.sampler_tile_size // 8)
    overlap = diffusion.sampler_tile_overlap
    threshold = diffusion.sampler_tile_split_threshold
    model_wrap_cfg._tiled_inner_model = TiledModelFn(
        model_wrap_cfg.inner_model,
        tile_size=tile_latent,
        overlap_h=overlap,
        overlap_w=overlap,
        split_threshold=threshold,
        sd_model=sd_model,
    )

    key = (diffusion.sampler_tile_size, overlap, threshold, H, W)
    if getattr(model_wrap_cfg, '_tiled_log_key', None) == key:
        return
    model_wrap_cfg._tiled_log_key = key
    if not (H and W):
        print(f'Tiled sampler: ON (tile {diffusion.sampler_tile_size}px, '
              f'overlap {overlap}%, split threshold {threshold}%)')
        return

    rows = tile_count_for_size(H // 8, tile_latent, threshold)
    cols = tile_count_for_size(W // 8, tile_latent, threshold)

    # Report the tile size we ACTUALLY end up with, not the one that was asked for. The
    # tile is derived from the count, so the two differ — and that gap is exactly what hid
    # the old "1080px target 1024 -> two 640px tiles" bug.
    from vibewarp.core.sampling import build_tile_specs
    specs = build_tile_specs(H // 8, W // 8, rows, cols, overlap, overlap)
    tile_h = (specs[0][1] - specs[0][0]) * 8
    tile_w = (specs[0][3] - specs[0][2]) * 8
    actual = (f'{tile_h}x{tile_w}px' if tile_h != tile_w else f'{tile_h}px')

    print(f'Tiled sampler: ON — {rows}x{cols} tiles ({rows * cols} per model eval) '
          f'of {actual}, overlap {overlap}%, over {W}x{H} '
          f'(requested {diffusion.sampler_tile_size}px, split threshold {threshold}%)')
    if rows * cols == 1:
        # One tile is not tiling: it pays the crop/blend cost and changes nothing. Almost
        # always means sampler_tile_size >= the frame, i.e. the setting is doing nothing.
        print(f'  WARNING: a single tile covers the whole {W}x{H} frame, so tiling has no '
              f'effect (and still costs the blend). Lower sampler_tile_size below '
              f'{max(W, H)}px, or turn tiled_sampler off.')


def _encode_prompt_weighted(
    sd_model: Any,
    prompts: List[str],
    weights: List[float],
    cache: Dict[str, Any],
) -> Any:
    """Encode a (possibly multi-prompt) list and return a weighted-sum conditioning tensor.

    For a single prompt, returns the cached conditioning directly.
    For multiple prompts, encodes each, normalizes weights, and returns
    the weighted sum — matching the notebook's CFGDenoiser prompt_weights logic.
    """
    # Encode any uncached prompts
    need = [p for p in prompts if p not in cache]
    if need:
        _move_to_cuda(sd_model.cond_stage_model)
        with torch.inference_mode(), torch.amp.autocast('cuda'):
            for p in need:
                cache[p] = sd_model.get_learned_conditioning([p])
        _offload_to_cpu(sd_model.cond_stage_model)

    if len(prompts) == 1:
        return cache[prompts[0]]

    # Weighted sum of conditioning tensors
    tensors = [cache[p] for p in prompts]
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum()  # normalize

    result = tensors[0] * w[0].item()
    for t, wi in zip(tensors[1:], w[1:].tolist()):
        result = result + t * wi
    return result


def run_sd(
    ctx: RenderContext,
    init_image_path: Optional[str],
    skip_timesteps: int,
    H: int,
    W: int,
    text_prompt: str,
    neg_prompt: str,
    steps: int,
    seed: int,
    cfg_scale: float,
    state: FrameState,
    image_cond: Optional[torch.Tensor] = None,
    rec_noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Core Stable Diffusion sampling — simplified from notebook's run_sd().

    This performs:
    1. Encode text prompts → conditioning (supports multi-prompt '|' with weights)
    2. Encode init image → latent (if provided)
    3. Compute sigma schedule
    4. Add noise to init latent (img2img) or generate random start (txt2img)
    5. Run k-diffusion sampler
    6. Return raw samples

    Args:
        ctx: render context with models and config
        init_image_path: path to init image (None for txt2img)
        skip_timesteps: number of timesteps to skip (for img2img strength)
        H: image height
        W: image width
        text_prompt: positive prompt string (may contain '|' multi-prompt with :weight suffixes)
        neg_prompt: negative prompt string
        steps: number of diffusion steps
        seed: random seed
        cfg_scale: classifier-free guidance scale
        state: frame state (for start_code persistence)

    Returns:
        (x_samples, latent) — decoded sample tensor and raw latent.
    """
    from vibewarp.utils.misc import seed_everything
    from vibewarp.core.prompt import split_weighted_prompts
    seed_everything(seed)

    sd_model = ctx.sd_model
    model_wrap = ctx.model_wrap
    model_wrap_cfg = ctx.model_wrap_cfg
    model_wrap_cfg._sd_batch_size = ctx.config.diffusion.sd_batch_size
    _configure_tiled_sampler(model_wrap_cfg, ctx.config, sd_model, H, W)

    C, f = 4, 8  # latent channels, downscale factor
    shape = [C, H // f, W // f]

    # Parse multi-prompt format (splits on '|', extracts ':weight' suffixes)
    pos_prompts, pos_weights = split_weighted_prompts(text_prompt)
    neg_prompts, neg_weights = split_weighted_prompts(neg_prompt)

    # Encode conditioning with caching and weighted blending
    from vibewarp.core.model_loader import is_sdxl_model
    is_sdxl = is_sdxl_model(ctx.config.model_version)

    cache = ctx._cond_cache
    c = _encode_prompt_weighted(sd_model, pos_prompts, pos_weights, cache)
    uc = _encode_prompt_weighted(sd_model, neg_prompts, neg_weights, cache) if cfg_scale != 1.0 else None

    # SDXL: extract crossattn/vector and set up vector_in on conditioner
    if is_sdxl and isinstance(c, dict) and 'crossattn' in c:
        c_crossattn = c['crossattn'][:, :77, :]
        c_vector = c.get('vector')
        if uc is not None and isinstance(uc, dict):
            uc_crossattn = uc['crossattn'][:, :77, :]
            uc_vector = uc.get('vector')
            if c_vector is not None and uc_vector is not None:
                vector_in = torch.cat([uc_vector, c_vector])
                sd_model.conditioner.vector_in = vector_in
            uc = uc_crossattn
        c = c_crossattn

    # Compute sigma schedule
    sigmas = compute_sigmas(
        model_wrap, steps,
        use_karras=ctx.config.diffusion.use_karras_noise,
    )

    t_enc = steps - skip_timesteps

    # Encode init image if provided. Offload VAE before sampler runs to
    # keep VRAM headroom for UNet + ControlNet + IP-Adapter.
    x0 = None
    if init_image_path is not None:
        with _phase(ctx, 'vae_encode'):
            init_image_sd = load_img_for_sd(init_image_path, size=(W, H))
            _move_to_cuda(sd_model.first_stage_model)
            with torch.inference_mode(), torch.amp.autocast('cuda'):
                x0 = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(init_image_sd))
            _offload_to_cpu(sd_model.first_stage_model)
        # Notebook diffuse() draws guidance_start_code = randn_like(init_latent)
        # here whenever guidance_use_start_code is on (default), even though the
        # guidance itself is deprecated. The draw advances the CUDA RNG stream,
        # so skipping it desynchronizes every subsequent same-seed randn.
        if ctx.config.diffusion.guidance_use_start_code:
            torch.randn_like(x0)

    # Build extra_args for CFGDenoiser
    # Per-step CFG ramp: store on model_wrap_cfg for popping each step
    if isinstance(cfg_scale, list):
        model_wrap_cfg._cfg_schedule = list(cfg_scale)  # reversed list, pop from end
        effective_cfg = cfg_scale[-1] if cfg_scale else 7.5  # first step value
    else:
        model_wrap_cfg._cfg_schedule = None
        effective_cfg = cfg_scale
    extra_args = {
        'cond': c,
        'uncond': uc,
        'cond_scale': effective_cfg,
    }
    if image_cond is not None:
        extra_args['image_cond'] = image_cond.to(x0.device if x0 is not None else 'cuda')

    # run_sd renders ONE frame. AnimateDiff never comes through here — it denoises
    # a whole batch of frames in run_frames_animatediff, with the context windows
    # applied inside CFGDenoiser. (The old code wrapped this single-frame sampler
    # in a "context sampler" whose windowing was dead code, because a one-frame
    # latent always hit its `batch_size <= 1` short-circuit.)
    sampler = ctx.sampler_fn or get_sampler_fn(ctx.config.diffusion.sampler)

    # Move UNet to GPU for sampling
    _move_to_cuda(sd_model.model)

    # Notebook does NOT use autocast around the sampler loop — only inside CFGDenoiser
    # (around the UNet call). Sampler arithmetic (euler steps, sigma scaling) must stay
    # float32 to avoid precision loss during iterative denoising.
    # Wrap model in static thresholding (notebook: dynamic_thresh)
    from vibewarp.core.model_loader import make_static_thresh_model_fn
    _dyn_thresh = ctx.config.diffusion.dynamic_thresh if ctx is not None else 30.0
    model_fn = make_static_thresh_model_fn(model_wrap_cfg, value=_dyn_thresh)

    noise_mode = ctx.config.diffusion.noise_mode
    code_randomness = ctx.config.diffusion.code_randomness

    with torch.inference_mode():
        if skip_timesteps > 0 and x0 is not None:
            # img2img: add noise to init latent and denoise partially
            # Cast x0 to float32 for sampler arithmetic precision
            x0_f = x0.float()

            # Noise source priority: rec_noise > rec_noise_override > fixed_code > default
            if rec_noise is not None:
                # Reconstruction noise (DDIM inversion)
                rec_config = ctx.config.reconstruction_noise
                rand_noise = torch.randn_like(x0_f)
                from vibewarp.core.reconstruction import _blend_noise
                combined = _blend_noise(rec_noise.float(), rand_noise, rec_config.randomness)
                noise = combined - (x0_f / sigmas[0])
                noise = noise * sigmas[steps - t_enc - 1]
                noise = noise * rec_config.noise_scale
            elif (ctx._rec_noise_override is not None
                    and state.frame_num in ctx._rec_noise_override.get('frame_indices', [])):
                # AnimateDiff batched reconstruction noise override
                rec_data = ctx._rec_noise_override
                idx = rec_data['frame_indices'].index(state.frame_num)
                if idx < rec_data['noise'].shape[0]:
                    rec_noise_ad = rec_data['noise'][idx:idx+1].to(x0_f.device).float()
                    rec_scale = ctx.config.reconstruction_noise.noise_scale
                    noise = (rec_noise_ad - x0_f / sigmas[0]) * rec_scale * sigmas[steps - t_enc - 1]
                else:
                    noise = torch.randn_like(x0_f) * sigmas[steps - t_enc - 1]
            elif noise_mode == 'fixed':
                # Persistent start_code across frames (notebook: noise_mode='fixed')
                # Initialize on first frame; persist for subsequent frames.
                if state.start_code is None:
                    state.start_code = torch.randn_like(x0_f)
                rand_code = torch.randn_like(x0_f)
                denom = (code_randomness ** 2 + (1 - code_randomness) ** 2) ** 0.5
                combined = ((1 - code_randomness) * state.start_code.float()
                            + code_randomness * rand_code) / denom
                noise = combined - (x0_f / sigmas[0])
                noise = noise * sigmas[steps - t_enc - 1]
            else:
                noise = torch.randn_like(x0_f) * sigmas[steps - t_enc - 1]

            if os.environ.get('VIBEWARP_DEBUG_DIFFUSION'):
                print(f"[diag] img2img t_enc={t_enc} noise8="
                      f"{[round(v, 5) for v in noise.flatten()[:8].tolist()]}")
            if t_enc != 0:
                xi = x0_f + noise
                sigma_sched = sigmas[steps - t_enc - 1:]
                with _phase(ctx, 'sampler'), \
                        _step_progress(len(sigma_sched) - 1, 'img2img steps') as on_step:
                    samples_ddim = sampler(
                        model_fn, xi, sigma_sched,
                        extra_args=extra_args, callback=on_step,
                    )
            else:
                samples_ddim = x0
        else:
            # txt2img: start from noise (pure random, reconstructed, or fixed_code)
            if rec_noise is not None:
                rec_config = ctx.config.reconstruction_noise
                rand_noise = torch.randn([1, *shape], device='cuda')
                from vibewarp.core.reconstruction import _blend_noise
                x = _blend_noise(rec_noise.float(), rand_noise, rec_config.randomness)
                x = x * rec_config.noise_scale * sigmas[0]
            elif noise_mode == 'fixed':
                if state.start_code is None:
                    state.start_code = torch.randn([1, *shape], device='cuda')
                rand_code = torch.randn([1, *shape], device='cuda')
                denom = (code_randomness ** 2 + (1 - code_randomness) ** 2) ** 0.5
                combined = ((1 - code_randomness) * state.start_code.float()
                            + code_randomness * rand_code) / denom
                x = combined * sigmas[0]
            else:
                x = torch.randn([1, *shape], device='cuda') * sigmas[0]
            if os.environ.get('VIBEWARP_DEBUG_DIFFUSION'):
                print(f"[diag] sigmas n={len(sigmas)} first={sigmas[0].item():.4f} "
                      f"last={sigmas[-1].item():.4f} sched={[round(s.item(),3) for s in sigmas]}")
                print(f"[diag] x_in std={x.std().item():.4f} cfg={effective_cfg} "
                      f"sampler={ctx.config.diffusion.sampler} "
                      f"x8={[round(v, 5) for v in x.flatten()[:8].tolist()]}")
            with _phase(ctx, 'sampler'), \
                    _step_progress(len(sigmas) - 1, 'txt2img steps') as on_step:
                samples_ddim = sampler(
                    model_fn, x, sigmas,
                    extra_args=extra_args, callback=on_step,
                )
            if os.environ.get('VIBEWARP_DEBUG_DIFFUSION'):
                print(f"[diag] samples std={samples_ddim.std().item():.4f} "
                      f"mean={samples_ddim.mean().item():.4f} "
                      f"min={samples_ddim.min().item():.4f} max={samples_ddim.max().item():.4f}")

    model_wrap_cfg._tiled_inner_model = None

    # Offload UNet, bring VAE for decode
    _offload_to_cpu(sd_model.model)

    # Store first latent for normalization
    if state.first_latent is None:
        state.first_latent = samples_ddim.detach()

    # Decode (VAE → GPU, then offload)
    with _phase(ctx, 'vae_decode'):
        _move_to_cuda(sd_model.first_stage_model)
        with torch.inference_mode(), torch.amp.autocast('cuda'):
            x_samples = sd_model.decode_first_stage(samples_ddim)
        _offload_to_cpu(sd_model.first_stage_model)

    return x_samples, samples_ddim


# ---- Inter-frame warping ----

def _apply_render_bg_mask(
    image: Image.Image,
    mask_frame_num: int,
    config,
    video_frames_folder: str,
) -> Image.Image:
    """Render-time background masking (notebook `apply_mask`, image mode).

    Composites `image` onto the configured background using the alpha mask at
    `{video_frames_folder}_alpha/{mask_frame_num+1:06d}.jpg`. Warns and
    returns the image unchanged when the alpha file is missing.
    """
    mc = config.mask
    alpha_path = os.path.join(
        video_frames_folder + '_alpha', f'{mask_frame_num + 1:06d}.jpg')
    if not os.path.exists(alpha_path):
        print(f"  WARNING: use_background_mask is on but alpha mask not found: "
              f"{alpha_path} — frame left unmasked")
        return image
    from vibewarp.video.output import apply_video_mask
    return apply_video_mask(
        image, mask_frame_num, mc.background, mc.background_source,
        video_frames_folder, invert=mc.invert_mask,
        clip_low=mc.mask_clip_low, clip_high=mc.mask_clip_high,
    )


def warp_between_frames(
    ctx: RenderContext,
    prev_frame: Image.Image,
    frame_num: int,
    flow_blend: float = 0.5,
    consistency_params: Optional[dict] = None,
) -> Image.Image:
    """Warp previous rendered frame toward current video frame using optical flow.

    This is the clean equivalent of the notebook's do_3d_step().

    Args:
        ctx: render context (needs raft_model, video_frames_folder, flow_folder)
        prev_frame: previous rendered frame (PIL Image)
        frame_num: current frame number (0-indexed)
        flow_blend: blend ratio between warped previous and current video frame

    Returns:
        Warped+blended PIL Image to use as init for the next SD render.
    """
    config = ctx.config

    # Locate the two video frames for flow computation
    # Video frames are 1-based (ffmpeg %06d.jpg starts at 000001.jpg),
    # so render frame N corresponds to video frame {N+1:06d}.jpg.
    # frame_num here is the previous render frame index (0-based), so:
    #   frame1 = video frame for prev render frame = {frame_num + 1}
    #   frame2 = video frame for current render frame = {frame_num + 2}
    frame1_path = os.path.join(ctx.video_frames_folder, f"{frame_num + 1:06d}.jpg")
    frame2_path = os.path.join(ctx.video_frames_folder, f"{frame_num + 2:06d}.jpg")

    if not os.path.exists(frame1_path) or not os.path.exists(frame2_path):
        # Can't compute flow without video frames — return prev_frame as-is
        return prev_frame

    # Flow and consistency map paths
    flow_folder = ctx.flow_folder or os.path.join(ctx.batch_folder, 'flow')
    os.makedirs(flow_folder, exist_ok=True)
    flow_path = os.path.join(flow_folder, f"{frame_num:06d}.jpg.npy")

    cc_path = ''
    weights = None
    if config.flow.check_consistency:
        cc_path = os.path.join(flow_folder, f"{frame_num:06d}.jpg_12-21_cc.jpg")

    # Compute or load flow
    # Notebook: flow_maxsize=0 means compute at rendering resolution (width_height),
    # NOT at original video frame resolution. Any positive value limits max dimension.
    import time as _time
    from vibewarp.core.timing import log_time as _log_time
    from vibewarp.flow.flow_utils import get_flow_and_cc
    flow_maxsize_cfg = config.flow.flow_maxsize
    effective_flow_size = None
    effective_flow_maxsize = None
    if flow_maxsize_cfg in (0, None, -1):
        # Notebook: flow_width_height = width_height (exact rendering resolution)
        effective_flow_size = (config.video.width, config.video.height)
    else:
        effective_flow_maxsize = flow_maxsize_cfg

    _t_flow = _time.time()
    flow, cc = get_flow_and_cc(
        frame1_path=frame1_path,
        frame2_path=frame2_path,
        flow_path=flow_path,
        cc_path=cc_path,
        model=ctx.raft_model,
        flow_maxsize=effective_flow_maxsize,
        flow_size=effective_flow_size,
        half=config.flow.flow_lq,
        num_flow_updates=config.flow.num_flow_updates,
        dilation=config.flow.missed_consistency_dilation,
        edge_width=config.flow.edge_consistency_width,
        force=config.flow.force_flow_generation,
    )
    _log_time(f"RAFT inference (frame {frame_num})", _time.time() - _t_flow)

    # Resolve per-frame consistency params (scheduled or static)
    cp = consistency_params or {}
    missed_w = cp.get('missed_consistency_weight', config.flow.missed_consistency_weight)
    overshoot_w = cp.get('overshoot_consistency_weight', config.flow.overshoot_consistency_weight)
    edges_w = cp.get('edges_consistency_weight', config.flow.edges_consistency_weight)
    cc_blur = int(cp.get('consistency_blur', config.flow.consistency_blur))
    cc_dilate = int(cp.get('consistency_dilate', config.flow.consistency_dilate))
    forward_clip = float(cp.get('soften_consistency_mask', config.flow.soften_consistency_mask))

    if cc is not None:
        # Process 3-channel CC map into B&W weights (matches notebook's load_cc)
        from vibewarp.flow.consistency import load_cc
        weights = load_cc(
            cc,
            missed_consistency_weight=missed_w,
            overshoot_consistency_weight=overshoot_w,
            edges_consistency_weight=edges_w,
            blur=cc_blur,
            dilate=cc_dilate,
        )

    # Before-diffusion matching colorizes raw/current pixels toward the warped
    # previous render before the consistency mask blends them together.
    match_color_fn = None
    if _uses_colormatch(config, 'before'):
        match_color_fn = lambda stylized, raw: _match_color_array(
            stylized, raw, config, 'before').astype(np.float64)

    # Set up brightness callback
    adjust_brightness_fn = None
    if config.brightness.enable:
        from vibewarp.color.correction import adjust_brightness
        bc = config.brightness
        adjust_brightness_fn = lambda img: adjust_brightness(
            img,
            high_brightness_threshold=bc.high_brightness_threshold,
            high_brightness_adjust_ratio=bc.high_brightness_adjust_ratio,
            high_brightness_adjust_fix_amount=bc.high_brightness_adjust_fix_amount,
            max_brightness_threshold=bc.max_brightness_threshold,
            low_brightness_threshold=bc.low_brightness_threshold,
            low_brightness_adjust_ratio=bc.low_brightness_adjust_ratio,
            low_brightness_adjust_fix_amount=bc.low_brightness_adjust_fix_amount,
            min_brightness_threshold=bc.min_brightness_threshold,
        )

    # Warp — resize flow, cc, and video frame to match target rendering resolution
    import cv2 as _cv2
    from vibewarp.flow.warp import warp_frame
    target_w, target_h = config.video.width, config.video.height
    current_video_frame = Image.open(frame2_path).convert('RGB').resize((target_w, target_h), Image.LANCZOS)

    # Background mask BEFORE warp (notebook: use_background_mask and not
    # apply_mask_after_warp → mask the raw current video frame).
    # frame_num here is the previous render frame → current = frame_num + 1.
    if config.mask.use_background_mask and not config.mask.apply_mask_after_warp:
        current_video_frame = _apply_render_bg_mask(
            current_video_frame, frame_num + 1, config, ctx.video_frames_folder)

    # Resize flow field to target resolution, scaling displacement values
    # to match. Matches notebook: flow_ratio = flow_wh / target_wh; flow *= ratio
    flow_h, flow_w = flow.shape[:2]
    if flow_h != target_h or flow_w != target_w:
        flow_wh = np.array([flow_w, flow_h], dtype=np.float32)
        target_wh = np.array([target_w, target_h], dtype=np.float32)
        flow = _cv2.resize(flow, (target_w, target_h), interpolation=_cv2.INTER_LINEAR).astype('float32')
        flow_ratio = (flow_wh / target_wh)[None, None, ...]
        flow = flow * flow_ratio
        if weights is not None:
            weights = _cv2.resize(weights, (target_w, target_h), interpolation=_cv2.INTER_LINEAR)
            if weights.ndim == 2:
                weights = weights[..., None]
            if weights.shape[2] == 1:
                weights = np.repeat(weights, 3, axis=2)

    # Save the exact effective mask used by the blend. Frame N is produced by
    # flow N-1 -> N, hence the +1 relative to this function's frame index.
    if weights is not None and ctx.batch_folder:
        debug_dir = os.path.join(ctx.batch_folder, 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        effective = np.clip(weights, forward_clip, 1.0)
        mask = effective[..., 0] if effective.ndim == 3 else effective
        Image.fromarray(
            np.clip(mask * 255.0, 0, 255).round().astype(np.uint8), mode='L',
        ).save(os.path.join(
            debug_dir, f"consistency_mask_{frame_num + 1:06d}.png"))

    warped = warp_frame(
        frame1=prev_frame,
        frame2=current_video_frame,
        flow=flow,
        blend=flow_blend,
        weights=weights,
        forward_clip=forward_clip,
        pad_pct=config.warp.padding_ratio,
        padding_mode=config.warp.padding_mode,
        warp_mul=config.warp.warp_strength,
        match_color_fn=match_color_fn,
        adjust_brightness_fn=adjust_brightness_fn,
    )

    # Background mask AFTER warp (notebook default: apply_mask_after_warp).
    if config.mask.use_background_mask and config.mask.apply_mask_after_warp:
        warped = _apply_render_bg_mask(
            warped, frame_num + 1, config, ctx.video_frames_folder)

    return warped


def _uses_colormatch(config: RunConfig, phase: str) -> bool:
    """Whether color matching is enabled for the before/after render phase."""
    return (
        bool(getattr(config.color, f'{phase}_enabled'))
        and getattr(config.color, f'{phase}_strength') > 0
    )


def _match_color_array(
    reference, target, config: RunConfig, phase: str
) -> np.ndarray:
    """Match target colors with the independent settings for one phase."""
    from vibewarp.color.matching import match_color_var

    transfer_fn = None
    regrain_fn = None
    method = getattr(config.color, f'{phase}_method').lower()
    regrain = getattr(config.color, f'{phase}_regrain')
    if method in ('lab', 'mean') or regrain:
        from vibewarp.vendor.python_color_transfer.color_transfer import ColorTransfer
        transfer = ColorTransfer()
        if method == 'lab':
            transfer_fn = transfer.lab_transfer
        elif method == 'mean':
            transfer_fn = transfer.mean_std_transfer
        if regrain:
            regrain_fn = transfer.RG.regrain

    return match_color_var(
        reference,
        target,
        opacity=getattr(config.color, f'{phase}_strength'),
        transfer_fn=transfer_fn,
        regrain_fn=regrain_fn,
        regrain=regrain,
    )


def _apply_post_colormatch(
    image: Image.Image,
    reference: Image.Image | str | None,
    config: RunConfig,
) -> Image.Image:
    """Match a rendered frame to the previous rendered frame's palette."""
    if not _uses_colormatch(config, 'after'):
        return image
    if isinstance(reference, Image.Image):
        previous = reference.convert('RGB')
    elif reference and os.path.exists(reference):
        previous = Image.open(reference).convert('RGB')
    else:
        return image
    previous = previous.resize(
        image.size, Image.LANCZOS)
    matched = _match_color_array(
        previous, image.convert('RGB'), config, 'after')
    return Image.fromarray(matched)


# ---- Per-frame IP-Adapter ----

def _resolve_ipadapter_source(
    source, frame_num: int, ctx: RenderContext, state: FrameState,
) -> Optional[str]:
    """Resolve an IP-Adapter source_image for this frame.

    Notebook semantics (ipadapters share the CN source machinery):
    - 'init' / 'raw_frame' → current raw video frame
    - 'stylized' → warped previous render (frame 0: falls back to raw frame)
    - directory → {frame_num:06d}.png|jpg inside it
    - dict → frame schedule {frame: path}
    - anything else → static path (returned as-is)
    Returns None when nothing usable exists for this frame.
    """
    if isinstance(source, dict):
        resolved = _get_prompt_for_frame(frame_num, source)
        return resolved if resolved and os.path.exists(resolved) else None

    if source in ('init', 'raw_frame'):
        p = os.path.join(ctx.video_frames_folder, f"{frame_num + 1:06d}.jpg")
        return p if os.path.exists(p) else None

    if source == 'stylized':
        if state.init_image and os.path.exists(state.init_image):
            return state.init_image
        # frame 0: no previous render yet — notebook falls back to raw frame
        p = os.path.join(ctx.video_frames_folder, f"{frame_num + 1:06d}.jpg")
        return p if os.path.exists(p) else None

    if os.path.isdir(source):
        for ext in ('png', 'jpg', 'jpeg'):
            candidate = os.path.join(source, f"{frame_num:06d}.{ext}")
            if os.path.exists(candidate):
                return candidate
        return None

    return source if source and os.path.exists(source) else None


_PER_FRAME_IPA_SOURCES = ('init', 'raw_frame', 'stylized')


def _update_ipadapter_for_frame(ctx: RenderContext, state: FrameState) -> None:
    """Re-hook IP-Adapters whose source image changes per frame.

    Notebook equivalent: get_controlnet_annotations() clears ALL adapter
    hooks each frame, then hooks every adapter with (possibly cached) CLIP
    embeddings for that frame's source. Embeds are cached by file CONTENT
    hash + adapter key (paths repeat for 'stylized' sources while contents
    change — hashing handles that), size-limited FIFO.

    Static-path sources produce identical embeds every frame, so when no
    adapter has a per-frame source this is a no-op (the pipeline's initial
    hooks stay — equivalent to the notebook's re-hook with cached embeds).
    """
    if not ctx.loaded_ipadapters or ctx.sd_model is None:
        return
    if ctx.clip_vision_model is None:
        return

    frame_num = state.frame_num
    config = ctx.config

    # Resolve every adapter's source for this frame; determine whether any
    # of them is per-frame (needs a re-hook at all).
    resolved = {}
    any_per_frame = False
    for ipa_key, ipa_data in ctx.loaded_ipadapters.items():
        source = ipa_data.get('source_image', '')
        resolved[ipa_key] = _resolve_ipadapter_source(source, frame_num, ctx, state)
        if isinstance(source, dict) or (isinstance(source, str) and (
                source in _PER_FRAME_IPA_SOURCES or os.path.isdir(source))):
            any_per_frame = True

    if not any_per_frame:
        return  # all static — initial hooks from pipeline.run() are current

    from vibewarp.core.ipadapter import (
        load_image_for_clip,
        encode_clip_vision,
        hook_ipadapter,
        IPAdapterEmbeddingCache,
    )
    from vibewarp.vendor.controlmodel_ipadapter import clear_all_ip_adapter
    from vibewarp.settings import generate_file_hash

    # Content-hash embeds cache (notebook: ipadapter_embeds_cache), FIFO,
    # lazily created on the context.
    cache = None
    if config.ipadapter.cache_embeddings:
        cache = getattr(ctx, '_ipa_embed_cache', None)
        if cache is None:
            cache = IPAdapterEmbeddingCache(max_size=config.ipadapter.cache_size)
            ctx._ipa_embed_cache = cache

    # Notebook order: clear ALL hooks once, then hook every adapter.
    # (Per-adapter unhook inside the loop would wipe previously hooked
    # adapters — clear_all is global.)
    clear_all_ip_adapter()
    if hasattr(ctx.sd_model, '_ipadapter_hooks'):
        ctx.sd_model._ipadapter_hooks.clear()

    for ipa_key, ipa_data in ctx.loaded_ipadapters.items():
        source_path = resolved[ipa_key]
        if source_path is None:
            print(f"  IP-Adapter {ipa_key}: no source for frame {frame_num}, skipping")
            continue

        cache_key = f"{generate_file_hash(source_path)}_{ipa_key}" if cache is not None else None
        clip_output = cache.get(cache_key) if cache is not None else None
        if clip_output is None:
            image_tensor = load_image_for_clip(source_path, device='cuda')
            clip_output = encode_clip_vision(ctx.clip_vision_model, image_tensor)
            if cache is not None:
                cache.put(cache_key, clip_output)

        adapter_path = ipa_data.get('path', '') or ''
        hook_ipadapter(
            sd_model=ctx.sd_model,
            adapter_weights=ipa_data['weights'],
            clip_vision_output=clip_output,
            weight=ipa_data['weight'],
            start=ipa_data['start'],
            end=ipa_data['end'],
            weight_type=ipa_data.get('weight_type', 'linear'),
            embeds_scaling=ipa_data.get('embeds_scaling', 'V only'),
            is_v2='v2' in os.path.basename(adapter_path).lower(),
            combine_embeds=ipa_data.get('combine_embeds', 'concat'),
            flip_uc=config.ipadapter.flip_uc,
        )


# ---- Per-frame reconstruction noise ----

def _compute_per_frame_rec_noise(
    ctx: RenderContext,
    state: FrameState,
    rec_frame_path: str,
    W: int, H: int,
    text_prompt: str,
    neg_prompt: str,
    steps: int,
    skip_steps: int,
) -> Optional[torch.Tensor]:
    """Compute reconstruction noise for a single frame via DDIM inversion.

    This is the non-AnimateDiff path: encode the reconstruction source frame
    to latent space, then run DDIM inversion to find the noise.

    Matches notebook: rec_steps = int(steps * rec_steps_pct)
    """
    from vibewarp.core.reconstruction import find_noise_for_image
    rec_config = ctx.config.reconstruction_noise

    if rec_frame_path is None or not os.path.exists(rec_frame_path):
        return None

    # Encode to latent (VAE offloading handled by encode_image_to_latent)
    rec_img = load_img_for_sd(rec_frame_path, size=(W, H))
    with torch.inference_mode(), torch.amp.autocast('cuda'):
        rec_latent = encode_image_to_latent(ctx.sd_model, rec_img)

    # Use rec-specific prompt if available, else main prompt
    # Notebook: rec_prompts_series overrides text_prompt for DDIM inversion,
    # and gets its own {caption} substitution (text_prompt arrives substituted)
    if rec_config.prompts:
        rec_prompt = _get_prompt_for_frame(state.frame_num, rec_config.prompts)
        from vibewarp.core.captioning import get_caption, apply_caption
        _cap_folder = (ctx.video_frames_folder + 'Captions'
                       if ctx.video_frames_folder else '')
        rec_prompt = apply_caption(rec_prompt, get_caption(state.frame_num, _cap_folder))
    else:
        rec_prompt = text_prompt
    if rec_config.neg_prompts:
        rec_neg = _get_prompt_for_frame(state.frame_num, rec_config.neg_prompts)
    else:
        rec_neg = neg_prompt

    # Compute inversion steps — notebook uses: int(steps * rec_steps_pct)
    rec_steps = max(1, int(steps * rec_config.steps_pct))

    # Optionally disable ControlNets during DDIM inversion. The patched
    # apply_model short-circuits when `_cn_hints` is empty, so we swap it
    # out around the inversion call. Notebook: `disable_rec_noise_controlnets`.
    saved_cn_hints = None
    if rec_config.disable_controlnets and hasattr(ctx.sd_model, '_cn_hints'):
        saved_cn_hints = ctx.sd_model._cn_hints
        ctx.sd_model._cn_hints = {}

    # Run DDIM inversion (UNet needed on GPU)
    _move_to_cuda(ctx.sd_model.model)
    try:
        rec_noise = find_noise_for_image(
            sd_model=ctx.sd_model,
            model_wrap=ctx.model_wrap,
            init_latent=rec_latent,
            prompt=rec_prompt,
            neg_prompt=rec_neg,
            cfg_scale=rec_config.cfg_scale,
            steps=rec_steps,
            separate_uncond=rec_config.separate_uncond,
            steps_cutoff=rec_config.steps_cutoff,
            cond_cache=ctx._cond_cache,
        )
    finally:
        _offload_to_cpu(ctx.sd_model.model)
        if saved_cn_hints is not None:
            ctx.sd_model._cn_hints = saved_cn_hints

    return rec_noise


# ---- Frame rendering ----

def render_frame(ctx: RenderContext, state: FrameState) -> Image.Image:
    """Render a single frame using Stable Diffusion with warping.

    This is the core of do_run() extracted into a single-frame function.

    Args:
        ctx: render context with models and config
        state: mutable frame state

    Returns:
        Rendered PIL Image for this frame.
    """
    config = ctx.config

    # Flux edit models take an entirely separate render path (diffusers DiT),
    # not the CompVis / k-diffusion sampler below.
    from vibewarp.core.model_loader import is_edit_model
    if is_edit_model(config.model_version):
        return render_frame_edit(ctx, state)

    frame_num = state.frame_num
    W = config.video.width
    H = config.video.height

    # Get scheduled parameters for this frame
    sched = get_frame_schedule(frame_num, config)
    steps = sched['steps']
    cfg_scale = sched['cfg_scale']
    seed = sched['seed']

    # Resolve text prompt for this frame — strip LORA tags, apply per-frame LORA schedule
    from vibewarp.core.prompt import split_lora_from_prompts, get_prompt_and_loras_for_frame
    if not hasattr(ctx, '_lora_schedule'):
        # Build LORA schedule once from prompts (strips <lora:name:weight> tags)
        ctx._lora_schedule = {}
        ctx._clean_prompts = config.text_prompts
        if config.lora_dir:
            ctx._clean_prompts, ctx._lora_schedule = split_lora_from_prompts(config.text_prompts)

    clean_prompts = getattr(ctx, '_clean_prompts', config.text_prompts)
    lora_schedule = getattr(ctx, '_lora_schedule', {})

    text_prompt = _get_prompt_for_frame(frame_num, clean_prompts)
    neg_prompt = _get_prompt_for_frame(frame_num, config.negative_prompts)

    # {caption} placeholder substitution (notebook cell 30 keyframe captions;
    # the notebook substitutes in text and rec prompts, not neg prompts)
    from vibewarp.core.captioning import get_caption, apply_caption
    _captions_folder = (ctx.video_frames_folder + 'Captions'
                        if ctx.video_frames_folder else '')
    _caption = get_caption(frame_num, _captions_folder)
    text_prompt = apply_caption(text_prompt, _caption)

    # Apply per-frame LORA weights
    if config.lora_dir and lora_schedule:
        from vibewarp.core.lora import load_loras_for_frame
        from vibewarp.utils.scheduling import get_scheduled_arg
        lora_names = list(lora_schedule.keys())
        lora_weights = [
            float(get_scheduled_arg(frame_num, lora_schedule[n], blend_json_schedules=False) or 0.0)
            for n in lora_names
        ]
        load_loras_for_frame(ctx.sd_model, lora_names, lora_weights, config.lora_dir,
                             precision=getattr(config, 'lora_merge_precision', 'fp16'))

    # Determine init image and skip steps
    if state.init_image is not None:
        init_image_path = state.init_image
        style_strength = sched['style_strength']
        skip_steps = int(steps - steps * style_strength)
    else:
        init_image_path = None
        skip_steps = 0

    # Increment seed for non-fixed-seed mode
    if not config.diffusion.fixed_seed and frame_num > 0:
        seed += frame_num

    # Compute per-frame reconstruction noise (non-AnimateDiff mode)
    # Note: rec noise works even with style_strength=1 (skip_steps=0, txt2img path).
    # The rec_source setting determines what image to invert:
    #   'init' → raw video frame, 'stylized' → previous rendered frame
    rec_noise_tensor = None
    rec_config = config.reconstruction_noise
    if rec_config.enabled and not config.animatediff.enabled:
        # Determine rec source frame (independent of init_image_path)
        rec_frame_path = None
        if rec_config.source == 'stylized' and state.prev_frame is not None:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            state.prev_frame.save(tmp.name)
            rec_frame_path = tmp.name
        elif rec_config.source == 'init' and ctx.video_frames_folder:
            rec_frame_path = os.path.join(
                ctx.video_frames_folder, f"{frame_num + 1:06d}.jpg"
            )
        elif init_image_path is not None:
            rec_frame_path = init_image_path

        if rec_frame_path is not None and os.path.exists(rec_frame_path):
            with _phase(ctx, 'rec_noise'):
                rec_noise_tensor = _compute_per_frame_rec_noise(
                    ctx, state, rec_frame_path, W, H,
                    text_prompt, neg_prompt, steps, skip_steps,
                )
            # Clean up temp file if we created one
            if rec_config.source == 'stylized' and state.prev_frame is not None:
                try:
                    os.unlink(rec_frame_path)
                except OSError:
                    pass

    # Prepare per-ControlNet hint images if ControlNets are loaded and patched
    if ctx.loaded_controlnets and hasattr(ctx.sd_model, '_cn_hints'):
        from vibewarp.core.controlnet import prepare_cn_hints_for_frame
        # Build source paths for each ControlNet
        # source="global" → current video frame (structural guidance)
        # source="stylized" → init_image (warped previous render)
        # source=<path> → explicit file path
        cn_sources = {}
        current_video_frame = None
        if ctx.video_frames_folder:
            vf = os.path.join(ctx.video_frames_folder, f"{frame_num + 1:06d}.jpg")
            if os.path.exists(vf):
                current_video_frame = vf
            else:
                print(f"  WARNING: video frame not found: {vf}")

        # Log source images for debugging
        if frame_num == 0 or frame_num % 10 == 0:
            print(f"  Frame {frame_num}: init_image={state.init_image}, video_frame={current_video_frame}")

        # Resolve global cond_image_src setting (notebook: 'init'=raw video, 'stylized'=rendered)
        cond_image_src = config.controlnet.cond_image_src

        from vibewarp.core.controlnet import ANNOTATOR_MAP

        for cn_key, cn_data in ctx.loaded_controlnets.items():
            source_setting = cn_data.get('source', 'global') if isinstance(cn_data, dict) else 'global'
            annotator_type = (cn_data.get('annotator', '') if isinstance(cn_data, dict) else '') \
                or ANNOTATOR_MAP.get(cn_key, 'canny')

            # temporalnet: conditioning is the PREVIOUS frame — resolved via
            # temporalnet_source (notebook), not the generic cond source.
            if annotator_type == 'temporalnet':
                prev_path = None
                if config.controlnet.temporalnet_source == 'stylized':
                    # Previous rendered output
                    from glob import glob as _glob
                    matches = _glob(os.path.join(
                        ctx.batch_folder,
                        f"{config.batch_name}(0)_{frame_num - 1:06d}.*")) if ctx.batch_folder else []
                    prev_path = matches[0] if matches else None
                else:  # 'init' — previous raw video frame (frames are 1-indexed)
                    p = os.path.join(ctx.video_frames_folder, f"{frame_num:06d}.jpg")
                    prev_path = p if os.path.exists(p) else None

                if prev_path:
                    cn_sources[cn_key] = prev_path
                elif config.controlnet.temporalnet_skip_1st_frame:
                    # Notebook: drop temporalnet for this frame
                    print(f"  CN {cn_key}: no previous frame, skipping (temporalnet_skip_1st_frame)")
                elif current_video_frame:
                    # Notebook fallback: use the current frame as prev
                    cn_sources[cn_key] = current_video_frame
                continue

            # "global" means use the cond_image_src setting
            if source_setting in ('global', '', None):
                source_setting = cond_image_src

            source_path = None
            if source_setting in ('init', 'raw_frame'):
                # Raw video frame
                if current_video_frame:
                    source_path = current_video_frame
                else:
                    print(f"  CN {cn_key}: source=init but no video frame found")
            elif source_setting == 'stylized':
                # Warped previous render
                if state.init_image and os.path.exists(state.init_image):
                    source_path = state.init_image
                else:
                    # First frame has no stylized yet — fall back to video frame
                    source_path = current_video_frame
            elif os.path.exists(source_setting):
                # Explicit file path
                source_path = source_setting
            else:
                print(f"  CN {cn_key}: unknown source '{source_setting}'")

            if source_path:
                cn_sources[cn_key] = source_path

        # Inpaint CN mask (notebook: control_sd15_inpaint_mask_source).
        # 'consistency_mask' → this frame's CC map processed like load_cc;
        # 'none' → all-keep mask after the first frame; missing mask → the
        # inpaint CN is skipped for the frame (handled in prepare_cn_hints).
        inpaint_mask = None
        from vibewarp.core.controlnet import INPAINT_CN_KEYS
        if INPAINT_CN_KEYS.intersection(ctx.loaded_controlnets.keys()):
            mask_src = config.controlnet.inpaint_mask_source
            if mask_src == 'consistency_mask':
                flow_folder = ctx.flow_folder or (
                    os.path.join(ctx.batch_folder, 'flow') if ctx.batch_folder else '')
                # The cc map for rendering frame N is written while warping
                # N-1 → N and is named after the PREVIOUS render index (the
                # notebook loads {video frame N}.jpg cc, which is the same
                # map under its 1-based naming). {N:06d} does not exist yet
                # at render time — that lookup silently skipped the inpaint
                # CN on every frame.
                cc_file = ''
                if flow_folder and frame_num > getattr(ctx, 'start_frame', 0):
                    cc_file = os.path.join(
                        flow_folder, f"{frame_num - 1:06d}.jpg_12-21_cc.jpg")
                if cc_file and os.path.exists(cc_file):
                    from vibewarp.flow.consistency import load_cc
                    # Static config weights; per-frame CC schedules aren't
                    # re-resolved here (warp-side scheduling is unaffected).
                    inpaint_mask = load_cc(
                        np.array(Image.open(cc_file)),
                        missed_consistency_weight=config.flow.missed_consistency_weight,
                        overshoot_consistency_weight=config.flow.overshoot_consistency_weight,
                        edges_consistency_weight=config.flow.edges_consistency_weight,
                        blur=int(config.flow.consistency_blur),
                        dilate=int(config.flow.consistency_dilate),
                    )
            elif mask_src in ('none', None, '', 'None', 'off'):
                if frame_num > getattr(ctx, 'start_frame', 0):
                    inpaint_mask = np.ones((H, W, 3), dtype=np.float32)

        # Debug directory for saving preprocessor outputs and diffusion inputs
        debug_dir = os.path.join(ctx.batch_folder, 'debug') if ctx.batch_folder else None

        with _phase(ctx, 'cn_annotate'):
            prepare_cn_hints_for_frame(
                ctx.sd_model, ctx.loaded_controlnets,
                cn_sources, H, W,
                debug_dir=debug_dir,
                frame_num=frame_num,
                depth_detector=config.controlnet.depth_detector,
                inpaint_mask=inpaint_mask,
                seg_detector=config.controlnet.seg_detector,
                annotator_opts={
                    'scribble_detector': config.controlnet.scribble_detector,
                    'softedge_detector': config.controlnet.softedge_detector,
                    'canny_low_threshold': config.controlnet.canny_low_threshold,
                    'canny_high_threshold': config.controlnet.canny_high_threshold,
                    'mlsd_value_threshold': config.controlnet.mlsd_value_threshold,
                    'mlsd_distance_threshold': config.controlnet.mlsd_distance_threshold,
                    'qr_cn_mask_grayscale': config.controlnet.qr_cn_mask_grayscale,
                    'qr_cn_mask_invert': config.controlnet.qr_cn_mask_invert,
                    'qr_cn_mask_thresh': config.controlnet.qr_cn_mask_thresh,
                    'qr_cn_mask_clip_low': config.controlnet.qr_cn_mask_clip_low,
                    'qr_cn_mask_clip_high': config.controlnet.qr_cn_mask_clip_high,
                    'downscale_tile': config.controlnet.downscale_tile,
                },
            )

        # Save the init image at target resolution (what actually enters diffusion)
        if debug_dir and init_image_path and os.path.exists(init_image_path):
            os.makedirs(debug_dir, exist_ok=True)
            Image.open(init_image_path).convert('RGB').resize((W, H), Image.LANCZOS).save(
                os.path.join(debug_dir, f"diffusion_input_{frame_num:06d}.jpg"), quality=95)

    image_cond = None  # ControlNet hints are now on sd_model, not passed as extra_args

    # Per-frame IP-Adapter re-hooking when source images are per-frame
    _update_ipadapter_for_frame(ctx, state)

    # Run SD sampling
    with _phase(ctx, 'run_sd_total'):
        x_samples, latent = run_sd(
            ctx,
            init_image_path=init_image_path,
            skip_timesteps=skip_steps,
            H=H, W=W,
            text_prompt=text_prompt,
            neg_prompt=neg_prompt,
            steps=steps,
            seed=seed,
            cfg_scale=cfg_scale,
            state=state,
            image_cond=image_cond,
            rec_noise=rec_noise_tensor,
        )

    # Convert to PIL (move to float32 for PIL conversion).
    # Softcap first (notebook: applied to the decoded [-1,1] tensor before
    # the clamp — soft-compresses oversaturated values to damp the
    # cross-frame feedback loop).
    image_tensor = x_samples[0].float()
    if config.diffusion.do_softcap:
        from vibewarp.core.losses import softcap
        image_tensor = softcap(
            image_tensor,
            thresh=config.diffusion.softcap_thresh,
            q=config.diffusion.softcap_q,
        )
    image_tensor = image_tensor.add(1).div(2).clamp(0, 1)
    image = TF.to_pil_image(image_tensor)

    # Update state
    state.prev_frame = image
    state.prev_latent = latent.detach()

    # Memory cleanup
    del x_samples, image_tensor, latent, rec_noise_tensor
    gc.collect()
    torch.cuda.empty_cache()

    return image


def render_frame_edit(ctx: RenderContext, state: FrameState) -> Image.Image:
    """Render a single frame with the configured image-edit backend.

    The model-agnostic frame loop (`_render_single_frame`) has already set
    `state.init_image` to the image to restyle: the raw video frame on the first
    frame, the flow-warped consistency-weighted previous render thereafter. This
    function edits that image with the frame's prompt.

    Both edit backends receive the same ordered reference list resolved from
    raw, previous-stylized, warped/consistency-composited, or uploaded images.
    """
    config = ctx.config
    from vibewarp.core.model_loader import (
        is_flux_model, is_hidream_model, is_mage_flow_edit_model,
        is_qwen_edit_model)
    flux_backend = is_flux_model(config.model_version)
    if flux_backend:
        family, edit_config = 'flux', config.flux
    elif is_hidream_model(config.model_version):
        family, edit_config = 'hidream', config.hidream
    elif is_qwen_edit_model(config.model_version):
        family, edit_config = 'qwen', config.qwen
    elif is_mage_flow_edit_model(config.model_version):
        family, edit_config = 'mage', config.mage
    else:
        raise ValueError(f"Unsupported edit model: {config.model_version!r}")
    frame_num = state.frame_num
    W = config.video.width
    H = config.video.height

    sched = get_frame_schedule(frame_num, config)
    seed = sched['seed']
    if not edit_config.fixed_seed and frame_num > 0:
        seed += frame_num

    # Prompts (with {caption} substitution, matching the SD path). LORA tags are
    # meaningless for Flux, so we do not strip/apply them here.
    text_prompt = _get_prompt_for_frame(frame_num, config.text_prompts)
    neg_prompt = _get_prompt_for_frame(frame_num, config.negative_prompts)
    from vibewarp.core.captioning import get_caption, apply_caption
    _captions_folder = (ctx.video_frames_folder + 'Captions'
                        if ctx.video_frames_folder else '')
    text_prompt = apply_caption(text_prompt, get_caption(frame_num, _captions_folder))

    # Edit target: the warped previous render (or raw frame 0). Falls back to the
    # raw current video frame if init_image was never set.
    edit_path = state.init_image
    if not edit_path or not os.path.exists(edit_path):
        edit_path = os.path.join(
            ctx.video_frames_folder, f"{frame_num + 1:06d}.jpg") if ctx.video_frames_folder else None
    if not edit_path or not os.path.exists(edit_path):
        raise FileNotFoundError(
            f"Edit frame {frame_num}: no edit image (init_image={state.init_image!r})")
    warped_image = Image.open(edit_path).convert('RGB').resize(
        (W, H), Image.Resampling.LANCZOS)
    raw_path = (os.path.join(
        ctx.video_frames_folder, f"{frame_num + 1:06d}.jpg")
        if ctx.video_frames_folder else None)
    if raw_path and os.path.exists(raw_path):
        raw_image = Image.open(raw_path).convert('RGB').resize(
            (W, H), Image.Resampling.LANCZOS)
    else:
        # Frame zero and direct unit-level renders may only have init_image.
        raw_image = warped_image
    previous_image = state.prev_frame.convert('RGB') if state.prev_frame else None
    # On the first rendered frame init_image is the raw frame, not a temporal
    # warp. Mark it unavailable for optional warped-reference slots so the raw
    # image is not sent to the model twice. Required Image 1 still falls back
    # to raw in resolve_reference_images.
    warped_reference = warped_image if previous_image is not None else None

    from vibewarp.core.edit_references import (
        MAX_EDIT_REFERENCES, resolve_reference_images)
    references = resolve_reference_images(
        edit_config.references,
        raw_image=raw_image,
        previous_image=previous_image,
        warped_image=warped_reference,
        size=(W, H),
        max_references=MAX_EDIT_REFERENCES[family],
        label_opacity=config.reference_label_opacity,
    )
    from vibewarp.core.edit import render_edit_frame as run_edit_backend
    with _phase(ctx, 'run_sd_total'):
        image = run_edit_backend(
            ctx.edit_backend if ctx.edit_backend is not None else ctx.flux_pipe,
            config,
            edit_image=references[0],
            reference_images=references,
            prompt=text_prompt,
            negative_prompt=neg_prompt,
            seed=seed,
            width=W, height=H,
            debug_dir=(os.path.join(ctx.batch_folder, 'debug')
                       if ctx.batch_folder else None),
            frame_num=frame_num,
        )

    if image.size != (W, H):
        image = image.resize((W, H), Image.LANCZOS)

    state.prev_frame = image
    gc.collect()
    torch.cuda.empty_cache()
    return image


def render_frame_flux(ctx: RenderContext, state: FrameState) -> Image.Image:
    """Backward-compatible alias for the generalized edit renderer."""
    return render_frame_edit(ctx, state)


def _get_prompt_for_frame(frame_num: int, prompts: Dict[int, str]) -> str:
    """Get the active prompt for a given frame number.

    Prompts are keyed by frame number. The active prompt is the one
    with the highest key <= frame_num.

    Args:
        frame_num: current frame index
        prompts: dict mapping frame numbers to prompt strings

    Returns:
        The active prompt string.
    """
    if not prompts:
        return ""
    active_key = 0
    for k in sorted(prompts.keys()):
        if k <= frame_num:
            active_key = k
        else:
            break
    return prompts.get(active_key, "")


# ---- Frame loop ----

def _resolve_frame_range(
    ctx: RenderContext,
    frame_range: Optional[List[int]],
    resume_from: int,
) -> Tuple[int, int]:
    """Resolve start/end frame from config and arguments.

    `frame_range` is INCLUSIVE at both ends — 0-15 is sixteen frames, which is what the
    two numbers in the UI say and what the notebook means. Everything downstream works on
    a half-open range (`range(start, end)`, `end - start`), so the inclusive end is
    converted to the exclusive bound HERE and nowhere else.

    `[0, 0]` (the default) still means "every extracted frame" rather than "just frame 0":
    it is the notebook's not-set sentinel and settings files rely on it.
    """
    config = ctx.config
    if frame_range and len(frame_range) == 2:
        start_frame, end_frame = frame_range
    else:
        start_frame = 0
        end_frame = config.frame_range[1] if len(config.frame_range) > 1 else 0

    if end_frame > start_frame:
        end_frame += 1                      # inclusive -> exclusive
    else:
        # Not set (or degenerate): render everything that was extracted.
        if ctx.video_frames_folder and os.path.isdir(ctx.video_frames_folder):
            from glob import glob as _glob
            frames = _glob(os.path.join(ctx.video_frames_folder, '*.*'))
            end_frame = len(frames)         # already an exclusive count
        if end_frame <= start_frame:
            raise ValueError(f"Invalid frame range: [{start_frame}, {end_frame}]. "
                             "Set frame_range or ensure video frames were extracted.")

    start_frame = max(start_frame, resume_from)
    return start_frame, end_frame


def _render_single_frame(
    ctx: RenderContext,
    state: FrameState,
    frame_num: int,
    start_frame: int,
    batch_folder: str,
    fmt: str,
) -> Tuple[Image.Image, str]:
    """Set up init image, render a single frame, and save it.

    Returns (image, filepath) tuple.
    """
    config = ctx.config
    state.frame_num = frame_num

    # Set init image from video frames (for img2img after first frame)
    if frame_num > start_frame and ctx.video_frames_folder:
        prev_frame_path = os.path.join(
            batch_folder,
            f"{config.batch_name}(0)_{frame_num - 1:06d}.{fmt}"
        )
        if os.path.exists(prev_frame_path):
            # Inter-frame warping: warp prev rendered frame using optical flow
            if config.flow.flow_warp and state.prev_frame is not None:
                sched_for_warp = get_frame_schedule(frame_num, config)
                import time as _wtime
                from vibewarp.core.timing import log_time as _wlog
                _t_warp = _wtime.time()
                with _phase(ctx, 'warp'):
                    warped = warp_between_frames(
                        ctx, state.prev_frame, frame_num - 1,
                        flow_blend=sched_for_warp['flow_blend'],
                        consistency_params=sched_for_warp,
                    )
                _wlog(f"warp (frame {frame_num})", _wtime.time() - _t_warp)
                warped_path = os.path.join(
                    batch_folder, f"_warped_{frame_num:06d}.{fmt}"
                )
                warped.save(warped_path)
                print(f"  Warped frame {frame_num}: {warped.size} (target: {config.video.width}x{config.video.height})")
                state.init_image = warped_path
            else:
                state.init_image = prev_frame_path
    elif ctx.video_frames_folder:
        video_frame_path = os.path.join(
            ctx.video_frames_folder,
            f"{frame_num + 1:06d}.jpg"
        )
        if os.path.exists(video_frame_path):
            state.init_image = video_frame_path

    # Background-mask the init image right before diffusion (notebook masks
    # init_image/prevFrameScaled here for ALL frames — on warped frames this
    # is a second application on top of the warp-time mask, matching the
    # notebook's behavior; also covers frame 0 and non-warped inits).
    if config.mask.use_background_mask and state.init_image and ctx.video_frames_folder:
        masked = _apply_render_bg_mask(
            Image.open(state.init_image).convert('RGB'),
            frame_num, config, ctx.video_frames_folder)
        masked_path = os.path.join(batch_folder, f"_masked_init_{frame_num:06d}.{fmt}")
        masked.save(masked_path)
        state.init_image = masked_path

    # Preserve the actual previous output as the after-diffusion palette
    # reference. render_frame updates state.prev_frame to the new output.
    previous_palette = (
        state.prev_frame.convert('RGB').copy()
        if state.prev_frame is not None else None)

    # Render
    import time as _rtime
    from vibewarp.core.timing import log_time as _rlog
    _t_render = _rtime.time()
    image = render_frame(ctx, state)
    _rlog(f"diffusion render (frame {frame_num})", _rtime.time() - _t_render)

    # Let the first rendered frame establish the stylized palette. From the
    # second frame onward, "after" matches each new output back to the actual
    # previous stylized frame's colors. This applies to both the SD and edit
    # render paths and updates the feedback image too.
    if frame_num > start_frame:
        with _phase(ctx, 'colormatch_after'):
            image = _apply_post_colormatch(image, previous_palette, config)
        state.prev_frame = image

    # Save
    filename = f"{config.batch_name}(0)_{frame_num:06d}.{fmt}"
    filepath = os.path.join(batch_folder, filename)
    os.makedirs(batch_folder, exist_ok=True)
    image.save(filepath)

    return image, filepath


def _precache_conditioning(ctx: RenderContext, start_frame: int, end_frame: int) -> None:
    """Pre-encode all unique prompts so CLIP only loads once."""
    sd_model = ctx.sd_model
    if sd_model is None:
        return

    config = ctx.config
    cache = ctx._cond_cache

    from vibewarp.core.prompt import split_lora_from_prompts, split_weighted_prompts

    # Build clean prompts (LORA tags stripped) for caching
    clean_text = config.text_prompts
    if config.lora_dir:
        clean_text, _ = split_lora_from_prompts(config.text_prompts)

    # Collect all unique sub-prompts across the frame range
    prompts = set()
    for frame_num in range(start_frame, end_frame):
        raw_pos = _get_prompt_for_frame(frame_num, clean_text)
        raw_neg = _get_prompt_for_frame(frame_num, config.negative_prompts)
        for raw in (raw_pos, raw_neg):
            sub_prompts, _ = split_weighted_prompts(raw)
            prompts.update(sub_prompts)

    # Remove already-cached prompts
    prompts -= set(cache.keys())
    if not prompts:
        return

    _move_to_cuda(sd_model.cond_stage_model)
    with torch.inference_mode(), torch.amp.autocast('cuda'):
        for p in prompts:
            cache[p] = sd_model.get_learned_conditioning([p])
    _offload_to_cpu(sd_model.cond_stage_model)


def run_frames(
    ctx: RenderContext,
    frame_range: Optional[List[int]] = None,
    resume_from: int = 0,
) -> List[str]:
    """Run the full frame rendering loop.

    This is the top-level orchestrator extracted from cell 55's "Do the Run".
    Delegates to run_frames_animatediff() when AnimateDiff is enabled.

    Args:
        ctx: render context with models, config, and paths
        frame_range: [start, end] frame range, or None for all frames
        resume_from: frame number to resume from

    Returns:
        List of paths to rendered frame images.
    """
    config = ctx.config

    # AnimateDiff batch mode: process frames in overlapping temporal batches
    if config.animatediff.enabled:
        return run_frames_animatediff(ctx, frame_range=frame_range, resume_from=resume_from)

    sequence_start, end_frame = _resolve_frame_range(ctx, frame_range, 0)
    start_frame = max(sequence_start, resume_from)
    fmt = config.video.save_img_format

    state = FrameState(
        frame_num=start_frame,
        seed=config.diffusion.seed if config.diffusion.seed >= 0 else 0,
    )

    output_paths = []
    batch_folder = ctx.batch_folder

    # A resumed frame is still temporally downstream of the configured first
    # frame. Restore the last completed output so optical-flow/edit-reference
    # feedback continues exactly where the cancelled job stopped.
    if start_frame > sequence_start:
        previous_path = os.path.join(
            batch_folder,
            f"{config.batch_name}(0)_{start_frame - 1:06d}.{fmt}",
        )
        if not os.path.isfile(previous_path):
            raise FileNotFoundError(
                f"Cannot resume at frame {start_frame}: previous output is missing: "
                f"{previous_path}")
        with Image.open(previous_path) as previous:
            state.prev_frame = previous.convert('RGB').copy()

    # Pre-cache all unique prompt embeddings (CLIP on GPU once, then offload)
    _precache_conditioning(ctx, start_frame, end_frame)

    pbar = tqdm(range(start_frame, end_frame), desc="Rendering frames")
    for frame_num in pbar:
        raise_if_cancelled()
        frame_ts = time.time()
        if _PROFILE_ENABLED:
            ctx._phase_times = {}

        image, filepath = _render_single_frame(
            ctx, state, frame_num, sequence_start, batch_folder, fmt,
        )
        output_paths.append(filepath)

        if ctx.save_frame_fn:
            ctx.save_frame_fn(image, frame_num)
        if ctx.progress_fn:
            ctx.progress_fn(frame_num - start_frame + 1, end_frame - start_frame)

        # Inter-frame memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

        elapsed = time.time() - frame_ts
        pbar.set_postfix({"time": f"{elapsed:.1f}s"})

        if _PROFILE_ENABLED and getattr(ctx, '_phase_times', None):
            pt = ctx._phase_times
            parts = [f"{k}={v:.2f}s" for k, v in sorted(pt.items(), key=lambda kv: -kv[1])]
            accounted = sum(pt.values())
            other = max(0.0, elapsed - accounted)
            print(f"  [profile f{frame_num}] total={elapsed:.2f}s | "
                  + " | ".join(parts)
                  + f" | other={other:.2f}s")

    # Unload any LORAs applied during last frame
    if config.lora_dir and ctx.sd_model is not None:
        try:
            from vibewarp.core.lora import unload_loras
            unload_loras(ctx.sd_model)
        except Exception:
            pass

    return output_paths


def _adiff_init_path(ctx, frame_num: int, index_in_batch: int,
                     batch_overlap: int) -> Tuple[str, bool]:
    """Init image for a frame in an AnimateDiff batch (notebook dump 6609).

    Returns (path, is_stylized). The raw video frame — AnimateDiff does NOT
    optical-flow warp; temporal attention is what carries coherence. The exception
    is the first `batch_overlap` frames, which reuse the PREVIOUS batch's render
    when `overlap_stylized` is on. Those are also the frames that get re-injected as
    latents during sampling (reinject_stylized), which is what actually stitches
    consecutive batches together.
    """
    fmt = ctx.config.video.save_img_format
    if ctx.config.animatediff.overlap_stylized and index_in_batch <= batch_overlap:
        stylized = os.path.join(
            ctx.batch_folder,
            f"{ctx.config.batch_name}(0)_{frame_num:06d}.{fmt}")
        if os.path.exists(stylized):
            return stylized, True
    return os.path.join(ctx.video_frames_folder, f"{frame_num + 1:06d}.jpg"), False


def _adiff_prepare_conditioning(ctx, frames: List[int]):
    """Per-frame conditioning stacked on the batch axis -> [F, 77, D]."""
    from vibewarp.core.prompt import split_lora_from_prompts, split_weighted_prompts

    config = ctx.config
    text = config.text_prompts
    if config.lora_dir:
        text, _ = split_lora_from_prompts(config.text_prompts)

    conds, unconds = [], []
    for frame_num in frames:
        raw_pos = _get_prompt_for_frame(frame_num, text)
        raw_neg = _get_prompt_for_frame(frame_num, config.negative_prompts)
        pos, pos_w = split_weighted_prompts(raw_pos)
        neg, neg_w = split_weighted_prompts(raw_neg)
        conds.append(_encode_prompt_weighted(ctx.sd_model, pos, pos_w, ctx._cond_cache))
        unconds.append(_encode_prompt_weighted(ctx.sd_model, neg, neg_w, ctx._cond_cache))
    return _adiff_stack_cond(conds), _adiff_stack_cond(unconds)


def _adiff_stack_cond(per_frame: List[Any]):
    """Stack per-frame conditioning on the batch axis.

    SDXL conditioning is a dict ({'crossattn', 'vector'}), so each key is stacked
    separately; SD1.5 is a plain tensor.
    """
    if per_frame and isinstance(per_frame[0], dict):
        return {key: torch.cat([cond[key] for cond in per_frame])
                for key in per_frame[0]}
    return torch.cat(per_frame)


def _adiff_prepare_hints(ctx, frames: List[int], init_paths: List[str], H: int, W: int):
    """Per-frame ControlNet hints stacked to {cn_key: [F, 3, H, W]}.

    The notebook concatenates each frame's annotations (dump 6657). Hints must be
    per-frame — a single hint repeated across the batch would feed every frame the
    same structure and defeat the point.
    """
    from vibewarp.core.controlnet import prepare_cn_hints_for_frame

    if not ctx.loaded_controlnets:
        return {}

    config = ctx.config
    debug_dir = os.path.join(ctx.batch_folder, 'debug') if ctx.batch_folder else None
    stacked: Dict[str, List[torch.Tensor]] = {}

    for frame_num, init_path in zip(frames, init_paths):
        sources = {key: init_path for key in ctx.loaded_controlnets}
        prepare_cn_hints_for_frame(
            ctx.sd_model, ctx.loaded_controlnets, sources, H, W,
            t_ratio=0.0, debug_dir=debug_dir, frame_num=frame_num,
            depth_detector=config.controlnet.depth_detector,
            seg_detector=config.controlnet.seg_detector,
        )
        for key, hint in (ctx.sd_model._cn_hints or {}).items():
            stacked.setdefault(key, []).append(hint)

    return {key: torch.cat(hints) for key, hints in stacked.items()}


def _adiff_noise_indices(frames: List[int]) -> List[int]:
    """Noise index per batch slot, notebook-exact.

    A short final batch is padded by REPEATING the last frame, but the notebook
    pointedly does NOT repeat its noise:

        # pad noise for those frames from the beginning of the vector (can't repeat noise)
        noise = torch.cat([big_noise[frame_idxs], big_noise[:batch_length-len(frame_idxs)]])
        frame_idxs += [frame_idxs[-1]] * (batch_length - len(frame_idxs))

    So padded slots draw noise from the HEAD of the noise vector (frames 0, 1, 2...).
    We used to index the noise by the padded frame list, giving every padded slot the
    SAME noise as the real last frame -- measured: the padded tail of a 64-frame /
    batch-32 run diverged at MAE 0.199 (frame 63: 0.397) while every other region sat
    at 0.005-0.010.

    Real frames are strictly increasing, so any trailing run of equal indices is padding.
    """
    real = len(frames)
    while real > 1 and frames[real - 1] == frames[real - 2]:
        real -= 1
    pad = len(frames) - real
    return [int(f) for f in frames[:real]] + list(range(pad))


def _adiff_injected_noise(frames: List[int],
                          latents: torch.Tensor) -> Optional[torch.Tensor]:
    """Parity hook: use a noise tensor dumped from the notebook, if one is given.

    The notebook's AnimateDiff noise is process-random (see _adiff_frame_noise), so
    its output cannot be reproduced from a seed — not by us, and not by the notebook
    itself. An image-level MAE gate for AnimateDiff is therefore meaningless unless
    both sides start from the SAME noise. Point VIBEWARP_ADIFF_NOISE at the
    `diag_adiff_noise.pt` the reference probe saves, and the AnimateDiff math (motion
    module, context windows, reinject, seams) becomes measurable on its own.

    The tensor is the notebook's `big_noise`: [total_frames, 4, H//8, W//8], indexed
    by ABSOLUTE frame number.
    """
    path = os.environ.get('VIBEWARP_ADIFF_NOISE')
    if not path:
        return None
    big_noise = torch.load(path, map_location='cpu')
    want = (len(frames),) + tuple(latents.shape[1:])
    index = torch.tensor(_adiff_noise_indices(frames))
    if int(index.max()) >= big_noise.shape[0]:
        raise ValueError(
            f'{path}: holds {big_noise.shape[0]} frames, run needs frame '
            f'{int(index.max())}')
    noise = big_noise[index]
    if tuple(noise.shape) != want:
        raise ValueError(f'{path}: noise {tuple(noise.shape)} != latents {want}')
    print(f'[parity] AnimateDiff noise injected from {path} '
          f'for frames {frames[0]}-{frames[-1]}')
    return noise.to(device=latents.device, dtype=latents.dtype)


def _adiff_frame_noise(frames: List[int], latents: torch.Tensor, seed: int) -> torch.Tensor:
    """Noise keyed to the ABSOLUTE frame number, not to the batch.

    Batches overlap (0-7, 4-11, 8-15...), so a frame appears in more than one batch.
    Keying the noise to the frame means it gets the IDENTICAL noise vector every
    time, which matters twice over:

    * reinject_stylized re-noises the overlap frames as `latents + noise*sigma`. If
      that noise differs from the one that produced those latents, the frame is
      re-noised against a conflicting sample — visible as grain.
    * the frames a batch does NOT pin start from the same noise as their
      counterparts in the previous batch, so they land on the same identity instead
      of inventing a new one.

    This MATCHES the notebook structurally (verified 2026-07-13). do_run_adiff draws
    `big_noise = torch.randn((end_frame+1, 4, H//8, W//8), device='cpu')` ONCE for the
    whole video, then slices `noise = big_noise[frame_idxs]` — i.e. absolute-frame
    indexing. An earlier version of this docstring claimed the notebook keys noise by
    position-in-batch, citing `args.seed += 1`; that was WRONG. The seed increment
    feeds `seed_everything(args.seed)` in prepare_latents, which drives the SAMPLER's
    RNG. It does not produce the initial noise.

    We deliberately do NOT reproduce the notebook's RNG stream, because it does not
    have one: nothing seeds the CPU generator before `big_noise` is drawn, so the
    notebook's AnimateDiff noise is process-random. Measured: two notebook runs at
    the SAME seed disagree at MAE 0.30 — more than the notebook disagrees with us.
    Its AnimateDiff renders are not seed-reproducible at all. Ours are, by keying a
    per-frame generator on (seed + frame_num). Bug not replicated, on purpose.
    """
    shape = latents.shape[1:]
    rows = []
    # Padded slots in a short final batch take noise from the head of the vector, not
    # a repeat of the last frame's noise -- see _adiff_noise_indices.
    for frame_num in _adiff_noise_indices(frames):
        generator = torch.Generator(device='cpu').manual_seed(seed + int(frame_num))
        rows.append(torch.randn(shape, generator=generator, dtype=torch.float32))
    return torch.stack(rows).to(device=latents.device, dtype=latents.dtype)


def _adiff_run_batch(
    ctx: RenderContext,
    frames: List[int],
    batch_index: int,
    sampler: Callable,
    sampler_name: str,
) -> List[Image.Image]:
    """Denoise one batch of frames JOINTLY and return their decoded images.

    The batch axis is the temporal axis: `x` is [F, 4, h, w] and there is exactly
    ONE sampler call for the whole batch. Sliding context windows are applied inside
    CFGDenoiser (see `_forward_animatediff`).
    """
    from vibewarp.core.animatediff import (
        STATEFUL_SAMPLERS,
        make_context_schedule,
        repeat_schedule_for_sampler,
    )
    from vibewarp.core.model_loader import make_static_thresh_model_fn
    from vibewarp.utils.misc import seed_everything

    config = ctx.config
    ad = config.animatediff

    if (batch_index == 0 and ad.reinject_stylized
            and sampler_name in STATEFUL_SAMPLERS):
        warnings.warn(
            f"AnimateDiff: reinject_stylized mutates the latent in place every step, "
            f"which corrupts the state {sampler_name} carries between steps "
            f"(old_denoised / its own noise sampler) — expect streaked, degraded "
            f"frames after the first batch. Use a stateless sampler such as "
            f"sample_euler (roughly double the steps for a Turbo model), or set "
            f"animatediff.reinject_stylized=false and accept that consecutive "
            f"batches will drift apart.",
            RuntimeWarning, stacklevel=2)
    H, W = config.video.height, config.video.width
    count = len(frames)

    # Schedules come from the batch's FIRST frame — in AnimateDiff mode they are
    # per-batch, not per-frame (the notebook reads scheduled_args[...][0]).
    sched = get_frame_schedule(frames[0], config)
    steps = int(sched['steps'])
    style_strength = float(sched['style_strength'])
    cfg_scale = sched['cfg_scale']
    skip_steps = int(steps - steps * style_strength)

    # Notebook: seed_everything(args.seed) per batch, then `args.seed += 1` at the end
    # of each batch unless fixed_seed (dump 7173; fixed_seed defaults False).
    seed = config.diffusion.seed if config.diffusion.seed >= 0 else 0
    batch_seed = seed if config.diffusion.fixed_seed else seed + batch_index
    seed_everything(batch_seed)

    resolved = [
        _adiff_init_path(ctx, frame_num, i, ad.batch_overlap)
        for i, frame_num in enumerate(frames)
    ]
    init_paths = [path for path, _ in resolved]
    # Leading frames that came from the previous batch's render — the ones the
    # reinjection anchors to. Only the contiguous run at the head counts.
    stylized_count = 0
    for _, is_stylized in resolved:
        if not is_stylized:
            break
        stylized_count += 1

    cond, uncond = _adiff_prepare_conditioning(ctx, frames)
    hints = _adiff_prepare_hints(ctx, frames, init_paths, H, W)
    if ctx.sd_model is not None:
        ctx.sd_model._cn_hints = hints

    sigmas = compute_sigmas(ctx.model_wrap, steps,
                            use_karras=config.diffusion.use_karras_noise)

    # Encode every init frame and stack them on the batch (= temporal) axis.
    _move_to_cuda(ctx.sd_model.first_stage_model)
    with torch.inference_mode(), torch.amp.autocast('cuda'):
        latents = torch.cat([
            encode_image_to_latent(ctx.sd_model, load_img_for_sd(path, (W, H)))
            for path in init_paths
        ])
    _offload_to_cpu(ctx.sd_model.first_stage_model)

    injected = _adiff_injected_noise(frames, latents)
    if injected is not None:
        noise = injected
    elif ad.frame_keyed_noise:
        noise = _adiff_frame_noise(frames, latents, seed)
    else:
        # Legacy: one draw per batch, so noise is indexed by position in the batch.
        # NOT what the notebook does (see _adiff_frame_noise) — kept only so existing
        # runs can be reproduced.
        noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype)
    if ad.pingpong_noise:
        # Mirror the first half onto the second so the clip loops back on itself.
        half = noise[: count // 2]
        noise = torch.cat([half, torch.flip(half, dims=[0])])[:count]

    if skip_steps > 0:
        index = max(0, skip_steps - 1)
        x = latents + noise * sigmas[index]
        sigma_sched = sigmas[index:]
    else:
        x = noise * sigmas[0]
        sigma_sched = sigmas

    # Every context window is exactly context_length frames — the temporal length
    # the motion module is told to expect.
    if ctx.motion_module is not None and hasattr(ctx.motion_module, 'set_video_length'):
        ctx.motion_module.set_video_length(ad.context_length)

    schedule = make_context_schedule(
        total_length=count,
        context_length=ad.context_length,
        overlap=ad.context_overlap,
        steps=len(sigma_sched) + 1,
    )

    model_wrap_cfg = ctx.model_wrap_cfg
    _configure_tiled_sampler(model_wrap_cfg, config, ctx.sd_model, H, W)
    model_wrap_cfg._adiff_ctx_schedule = repeat_schedule_for_sampler(schedule, sampler_name)
    model_wrap_cfg._adiff_zero_uncond_hints = False

    # reinject_stylized: pin the overlap frames to the previous batch's render on
    # every denoiser call. `latents` already holds their encoded images (they were
    # the init paths), so no second encode is needed.
    model_wrap_cfg._adiff_stylized_latents = (
        latents[:stylized_count] if (ad.reinject_stylized and stylized_count) else None)
    model_wrap_cfg._adiff_noise = noise
    model_wrap_cfg._adiff_lerp_frames = ad.lerp_reinject_stylized_frames
    model_fn = make_static_thresh_model_fn(model_wrap_cfg, config.diffusion.dynamic_thresh)

    _move_to_cuda(ctx.sd_model.model)
    if os.environ.get('VIBEWARP_DEBUG_DIFFUSION'):
        # SDXL conditioning is a dict ({'crossattn','vector'}), SD1.5 a tensor.
        shape = (cond['crossattn'].shape if isinstance(cond, dict) else cond.shape)
        print(f"[diag] adiff batch={batch_index} frames={frames[0]}-{frames[-1]} "
              f"x={tuple(x.shape)} cond={tuple(shape)} "
              f"context={ad.context_length} windows/step={len(schedule[0])} "
              f"skip_steps={skip_steps} "
              f"reinject={stylized_count if ad.reinject_stylized else 0} frames")
    try:
        with torch.inference_mode(), \
                _step_progress(len(sigma_sched) - 1,
                               f'batch {batch_index} steps') as on_step:
            samples = sampler(
                model_fn, x, sigma_sched,
                extra_args={'cond': cond, 'uncond': uncond, 'cond_scale': cfg_scale},
                callback=on_step,
            )
    finally:
        # Never leave stale state behind — a later single-frame render would
        # otherwise take the AnimateDiff path with the wrong window list, and the
        # next batch would reinject this batch's latents.
        model_wrap_cfg._adiff_ctx_schedule = None
        model_wrap_cfg._adiff_stylized_latents = None
        model_wrap_cfg._adiff_noise = None
        model_wrap_cfg._tiled_inner_model = None

    _move_to_cuda(ctx.sd_model.first_stage_model)
    with torch.inference_mode(), torch.amp.autocast('cuda'):
        images = [decode_latent_to_image(ctx.sd_model, sample[None]) for sample in samples]
    _offload_to_cpu(ctx.sd_model.first_stage_model)
    return images


def run_frames_animatediff(
    ctx: RenderContext,
    frame_range: Optional[List[int]] = None,
    resume_from: int = 0,
) -> List[str]:
    """Render with AnimateDiff: each batch of frames is denoised JOINTLY.

    The batch axis IS the temporal axis the motion module attends across, so a
    batch of F frames becomes one [F, 4, h, w] latent and ONE sampler call. Inside
    the denoiser, sliding context windows of exactly `context_length` frames cover
    the batch (see CFGDenoiser._forward_animatediff).

    This replaces the earlier implementation, which looped `_render_single_frame`
    per frame: with a batch-1 latent the motion module has nothing to attend across,
    so AnimateDiff did nothing at all.
    """
    from vibewarp.core.animatediff import (
        make_context_schedule,
        repeat_schedule_for_sampler,
        split_into_batches,
    )
    from vibewarp.core.model_loader import make_static_thresh_model_fn
    from vibewarp.utils.misc import seed_everything

    config = ctx.config
    ad = config.animatediff
    start_frame, end_frame = _resolve_frame_range(ctx, frame_range, resume_from)
    fmt = config.video.save_img_format
    H, W = config.video.height, config.video.width
    total_frames = end_frame - start_frame

    if total_frames < ad.context_length:
        raise ValueError(
            f"AnimateDiff needs at least context_length ({ad.context_length}) frames, "
            f"got {total_frames}. Extract more frames or lower context_length.")

    batches = split_into_batches(
        total_frames=total_frames,
        batch_length=ad.batch_length,
        batch_overlap=ad.batch_overlap,
        start_frame=start_frame,
    )

    _precache_conditioning(ctx, start_frame, end_frame)

    sampler_name = config.diffusion.sampler
    sampler = ctx.sampler_fn or get_sampler_fn(sampler_name)
    model_wrap_cfg = ctx.model_wrap_cfg
    output_paths: List[str] = []
    previous_tail: List[Image.Image] = []

    pbar = tqdm(total=len(batches), desc="AnimateDiff batches")
    for batch_index, batch_frames in enumerate(batches):
        frames = [f for f in batch_frames if f < end_frame]
        if len(frames) < ad.context_length:
            # A short trailing batch cannot fill a context window; pull it back so
            # it still ends on the final frame.
            frames = list(range(max(start_frame, end_frame - ad.context_length), end_frame))

        images = _adiff_run_batch(ctx, frames, batch_index, sampler, sampler_name)

        images = _adiff_blend_seam(
            images, previous_tail, ad.batch_overlap if batch_index > 0 else 0,
            ad.blend_batch_outputs)

        for i, (frame_num, image) in enumerate(zip(frames, images)):
            filename = f"{config.batch_name}(0)_{frame_num:06d}.{fmt}"
            filepath = os.path.join(ctx.batch_folder, filename)
            image.save(filepath)
            is_new_frame = filepath not in output_paths
            if is_new_frame:
                output_paths.append(filepath)
            if ctx.save_frame_fn:
                ctx.save_frame_fn(image, frame_num)
            if ctx.progress_fn and is_new_frame:
                # Overlapping AnimateDiff batches revisit seam frames. Progress
                # counts unique selected frames, not batch work or source indices.
                ctx.progress_fn(len(output_paths), end_frame - start_frame)

        previous_tail = images[-ad.batch_overlap:] if ad.batch_overlap > 0 else []
        gc.collect()
        torch.cuda.empty_cache()
        pbar.update(1)

    pbar.close()
    return output_paths


def _adiff_blend_seam(images, previous_tail, overlap: int, blend: bool):
    """Cross-fade a batch's opening frames into the previous batch's closing ones.

    notebook save_adiff_frames (dump 6905): lerp = min(j/batch_overlap, 1), so the
    new batch ramps in from 0 to 1 across the overlap instead of cutting hard.
    """
    if not blend or overlap <= 0 or not previous_tail:
        return images
    out = list(images)
    for j in range(min(overlap, len(previous_tail), len(out))):
        alpha = min(j / overlap, 1.0) if overlap else 1.0
        out[j] = Image.blend(previous_tail[j], out[j], alpha)
    return out
    return output_paths


def _apply_reconstruction_noise_for_batch(
    ctx: RenderContext,
    state: FrameState,
    batch_frames: List[int],
    ad_config: Any,
    rec_config: Any,
    batch_folder: str,
    fmt: str,
) -> None:
    """Compute reconstruction noise for overlap frames of a new batch.

    Finds previously-rendered frames from the overlap region and runs DDIM
    inversion to get noise that "knows" about the previous batch's output.
    Stores the result in ctx._rec_noise_override for run_sd() to pick up.
    """
    from vibewarp.core.reconstruction import get_predicted_noise_adiff, get_predicted_noise_adiff_batched

    config = ctx.config
    overlap = ad_config.batch_overlap
    W, H = config.video.width, config.video.height

    # Collect paths for overlap frames (from previous batch output)
    overlap_frame_paths = []
    overlap_frame_idxs = []
    for frame_num in batch_frames[:overlap]:
        if rec_config.source == 'stylized':
            path = os.path.join(batch_folder, f"{config.batch_name}(0)_{frame_num:06d}.{fmt}")
        else:
            # 'init' source — use video frame
            path = os.path.join(ctx.video_frames_folder, f"{frame_num + 1:06d}.jpg") if ctx.video_frames_folder else ''
        overlap_frame_paths.append(path)
        overlap_frame_idxs.append(frame_num)

    if not overlap_frame_paths:
        return

    # Generate random noise for blending
    C, f = 4, 8
    rand_noise = torch.randn(len(overlap_frame_paths), C, H // f, W // f)

    # Get text prompt for reconstruction — use rec-specific prompt if available
    if rec_config.prompts:
        prompt = _get_prompt_for_frame(batch_frames[0], rec_config.prompts)
    else:
        prompt = _get_prompt_for_frame(batch_frames[0], config.text_prompts)
    neg_prompt = ''
    if rec_config.neg_prompts:
        neg_prompt = _get_prompt_for_frame(batch_frames[0], rec_config.neg_prompts)
    sched = get_frame_schedule(batch_frames[0], config)
    diffusion_steps = sched.get('steps', 25)

    try:
        # UNet on GPU for DDIM inversion
        _move_to_cuda(ctx.sd_model.model)
        if rec_config.batched:
            # Batched mode: encode all frames at once and invert as a single batch
            rec_noise = get_predicted_noise_adiff_batched(
                sd_model=ctx.sd_model,
                model_wrap=ctx.model_wrap,
                frame_paths=overlap_frame_paths,
                frame_idxs=overlap_frame_idxs,
                random_noise=rand_noise,
                config=rec_config,
                prompt=prompt,
                neg_prompt=neg_prompt,
                image_size=(W, H),
                context_length=ad_config.context_length,
                context_overlap=ad_config.context_overlap,
                diffusion_steps=diffusion_steps,
                cond_cache=ctx._cond_cache,
            )
        else:
            # Per-frame mode: invert each frame individually
            rec_noise = get_predicted_noise_adiff(
                sd_model=ctx.sd_model,
                model_wrap=ctx.model_wrap,
                frame_paths=overlap_frame_paths,
                frame_idxs=overlap_frame_idxs,
                random_noise=rand_noise,
                config=rec_config,
                prompt=prompt,
                neg_prompt=neg_prompt,
                image_size=(W, H),
                cache_dir=rec_config.cache_dir,
                diffusion_steps=diffusion_steps,
                cond_cache=ctx._cond_cache,
            )
        _offload_to_cpu(ctx.sd_model.model)
        # Store for run_sd() to use as initial noise
        ctx._rec_noise_override = {
            'noise': rec_noise,
            'frame_indices': overlap_frame_idxs,
        }
    except Exception:
        ctx._rec_noise_override = None


# ---- Schedule helper ----

def get_frame_schedule(frame_num: int, config: RunConfig) -> dict:
    """Compute all scheduled parameters for a given frame number.

    Uses get_scheduled_arg to resolve per-frame schedules (dict/list/scalar).

    Args:
        frame_num: current frame index
        config: run configuration

    Returns:
        Dict of parameter names to their scheduled values for this frame.
    """
    d = config.diffusion
    w = config.warp

    def _resolve(schedule, default):
        """Resolve a schedule to its value for this frame."""
        if schedule is None:
            return default
        val = get_scheduled_arg(frame_num, schedule, blend_json_schedules=True)
        # Handle nested lists (e.g., [[3,7,3]] from WarpFusion cfg_scale_schedule)
        while isinstance(val, list):
            val = val[0] if val else default
        return val

    def _resolve_cfg(schedule, default, steps):
        """Resolve cfg_scale schedule — may return a per-step list."""
        if schedule is None:
            return float(default)
        val = get_scheduled_arg(frame_num, schedule, blend_json_schedules=True)
        if isinstance(val, list):
            # Per-step CFG ramp (e.g., [3, 7, 3]) — interpolate across steps
            return interpolate_array(val, steps)
        return float(val)

    steps = int(_resolve(d.steps_schedule, d.steps))
    style_strength = float(_resolve(d.style_strength_schedule, d.style_strength))
    skip_steps = int(steps - steps * style_strength)
    effective_steps = steps - skip_steps + 1

    fl = config.flow
    return {
        'steps': steps,
        'cfg_scale': _resolve_cfg(d.cfg_scale_schedule, d.cfg_scale, effective_steps),
        'seed': d.seed if d.seed >= 0 else 0,
        'style_strength': style_strength,
        'flow_blend': float(_resolve(w.flow_blend_schedule, w.flow_blend)),
        # Consistency mask params — schedulable per frame (notebook: *_schedule)
        'missed_consistency_weight': float(_resolve(fl.missed_consistency_schedule, fl.missed_consistency_weight)),
        'overshoot_consistency_weight': float(_resolve(fl.overshoot_consistency_schedule, fl.overshoot_consistency_weight)),
        'edges_consistency_weight': float(_resolve(fl.edges_consistency_schedule, fl.edges_consistency_weight)),
        'consistency_blur': int(_resolve(fl.consistency_blur_schedule, fl.consistency_blur)),
        'consistency_dilate': int(_resolve(fl.consistency_dilate_schedule, fl.consistency_dilate)),
        'soften_consistency_mask': float(_resolve(fl.soften_consistency_schedule, fl.soften_consistency_mask)),
    }
