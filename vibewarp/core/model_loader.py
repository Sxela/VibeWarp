"""SD model loading — extracted from notebook cell 23.

Handles loading Stable Diffusion checkpoints (safetensors / .ckpt),
optional VAE replacement, and wrapping with k-diffusion denoisers.

Model pickle cache: after first load the model (CPU fp16, pre-device-move)
is serialised to a .pkl next to the checkpoint. Subsequent loads with the
same checkpoint+VAE skip both instantiate_from_config and load_state_dict,
which together take ~30-60 s on first run.
"""

import contextlib
import gc
import hashlib
import json
import math
import os
import pickle
import time
from typing import Any, Dict, Optional, Tuple

import torch
from omegaconf import OmegaConf

from vibewarp.vendor.ldm.util import instantiate_from_config
from vibewarp.vendor.cldm.model import load_state_dict


# ---- Model pickle cache ----

def _code_fingerprint() -> str:
    """Short digest of the vendored model source that gets baked into the pkl.

    The cache pickles model objects whose classes live in the vendored ldm /
    sgm / cldm packages, so a cache is only valid for the exact code that
    produced it. Including this fingerprint in the cache key means a changed
    or updated install can never silently reuse an incompatible pickle.
    Uses file names + sizes (stable across git checkouts, changes when code
    changes) rather than full content reads, so it stays cheap.
    """
    vendor = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'vendor'))
    parts = []
    # The pickle also bakes in classes from these libraries (CLIPTextModel,
    # open_clip's transformer, torch modules), so a version change can silently
    # invalidate a cache the vendored-source hash alone would still call valid —
    # e.g. open_clip 2.24 vs 3.3 swap the text encoder's tensor layout.
    for name in ('torch', 'transformers', 'open_clip'):
        try:
            module = __import__(name)
            parts.append(f"{name}:{getattr(module, '__version__', '?')}")
        except Exception:
            parts.append(f"{name}:absent")
    for pkg in ('ldm', 'sgm', 'cldm'):
        root = os.path.join(vendor, pkg)
        for dirpath, _dirs, files in os.walk(root):
            if '__pycache__' in dirpath:
                continue
            for fn in files:
                if fn.endswith('.py'):
                    p = os.path.join(dirpath, fn)
                    try:
                        parts.append(f"{os.path.relpath(p, vendor)}:{os.path.getsize(p)}")
                    except OSError:
                        pass
    return hashlib.md5('|'.join(sorted(parts)).encode()).hexdigest()[:8]


def _cache_dir() -> str:
    """Cache dir lives with the install (next to the package), NOT in the
    shared models folder — the pickle is code-version specific, so each
    install/version keeps its own and never cross-contaminates another."""
    install_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(install_root, '.vibewarp_cache')


def _cache_path(ckpt_path: str, vae_ckpt_path: Optional[str], half: bool,
                variant: str = '') -> str:
    """Return path for the pkl cache file for this checkpoint combination."""
    key = f"{os.path.abspath(ckpt_path)}|{os.path.abspath(vae_ckpt_path) if vae_ckpt_path else ''}|{half}"
    key += f"|{_code_fingerprint()}"
    if variant:
        key += f"|{variant}"
    suffix = hashlib.md5(key.encode()).hexdigest()[:8]
    base = os.path.splitext(os.path.basename(ckpt_path))[0]
    if vae_ckpt_path:
        base += '__' + os.path.splitext(os.path.basename(vae_ckpt_path))[0]
    return os.path.join(_cache_dir(), f"{base}_{suffix}.pkl")


def _cache_is_valid(cache_path: str, ckpt_path: str, vae_ckpt_path: Optional[str]) -> bool:
    if not os.path.exists(cache_path):
        return False
    cache_mtime = os.path.getmtime(cache_path)
    if os.path.getmtime(ckpt_path) > cache_mtime:
        return False
    if vae_ckpt_path and os.path.getmtime(vae_ckpt_path) > cache_mtime:
        return False
    return True


@contextlib.contextmanager
def _skip_weight_init():
    """Replace torch.nn.init functions with no-ops during model instantiation.

    Weights loaded from a checkpoint immediately overwrite random init anyway,
    so skipping it saves ~5-15 s on SD 1.5-sized models.
    """
    import torch.nn.init as _init
    originals = {}
    noop = lambda t, *a, **kw: t
    for name in dir(_init):
        if name.startswith('_') or name == 'calculate_gain':
            continue
        fn = getattr(_init, name)
        if callable(fn):
            originals[name] = fn
            setattr(_init, name, noop)
    try:
        yield
    finally:
        for name, fn in originals.items():
            setattr(_init, name, fn)


# Text-encoder weights live under these prefixes in an SD/SDXL state dict.
TEXT_ENCODER_PREFIXES = ('cond_stage_model.', 'conditioner.embedders.')


@contextlib.contextmanager
def _skip_pretrained_text_encoders():
    """Build the text encoders empty instead of downloading pretrained weights.

    The ldm/sgm configs construct their text encoders with
    `CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")` and
    `open_clip.create_model_and_transforms(..., pretrained="laion2b_s39b_b160k")`,
    which fetch ~1.7 GB (CLIP-L) and ~10 GB (OpenCLIP bigG) from the network —
    and then `load_state_dict` immediately overwrites every one of those tensors
    with the checkpoint's own encoders. A full SDXL checkpoint carries all of
    them (198 CLIP-L + 390 bigG keys), so the download is pure waste, and it
    makes a render silently depend on HuggingFace being reachable.

    Architecture comes from the config instead; the weights come from the
    checkpoint, as they always did. Numerically this is a no-op. Tokenizers are
    NOT skipped — they are small and are not stored in the checkpoint.

    The caller must verify the checkpoint actually supplied the encoder weights
    (see `_missing_text_encoder_keys`); a checkpoint without them would
    otherwise render with a randomly initialised text encoder.
    """
    # (target, name, original_or_None, had_own_attr). `from_pretrained` is
    # INHERITED from PreTrainedModel, so patching CLIPTextModel only shadows it;
    # restoring must delete the shadow, not setattr a bound method back onto the
    # class (that would replace the classmethod with a permanently-bound one).
    patched = []

    try:
        import transformers

        def from_config(version, *args, **kwargs):
            # Config only — a few KB of JSON, not the weights.
            cfg = transformers.CLIPTextConfig.from_pretrained(version)
            return transformers.CLIPTextModel(cfg)

        cls = transformers.CLIPTextModel
        had_own = 'from_pretrained' in cls.__dict__
        patched.append((cls, 'from_pretrained', cls.__dict__.get('from_pretrained'), had_own))
        cls.from_pretrained = from_config
    except Exception:
        pass

    try:
        import open_clip
        original = open_clip.create_model_and_transforms

        def without_pretrained(*args, **kwargs):
            kwargs['pretrained'] = None
            return original(*args, **kwargs)

        patched.append((open_clip, 'create_model_and_transforms', original, True))
        open_clip.create_model_and_transforms = without_pretrained
    except Exception:
        pass

    try:
        yield
    finally:
        for target, name, original, had_own in patched:
            if had_own:
                setattr(target, name, original)
            else:
                delattr(target, name)  # unshadow the inherited implementation


def _missing_text_encoder_keys(missing: list) -> list:
    """Text-encoder keys the checkpoint failed to provide."""
    return [k for k in missing if k.startswith(TEXT_ENCODER_PREFIXES)]


def _configure_hf_logging():
    """Quiet transformers' model-load reports unless debugging.

    Loading CLIPTextModel from a full CLIP checkpoint logs a large table of
    'UNEXPECTED' vision-tower keys — harmless, the text encoder just ignores
    the vision weights. Hidden by default; set VIBEWARP_DEBUG=1 to restore
    transformers' normal verbosity (and this whole report).
    """
    try:
        from transformers.utils import logging as hf_logging
        if os.environ.get('VIBEWARP_DEBUG'):
            hf_logging.set_verbosity_info()
        else:
            hf_logging.set_verbosity_error()
    except Exception:
        pass


# ---- Config paths for model architectures ----

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'configs')

# Default OmegaConf configs for different model versions
MODEL_CONFIGS = {
    'control_multi_v15': 'v1-inference.yaml',
    'control_multi_v21': 'v2-inference-v.yaml',
    'v1_5': 'v1-inference.yaml',
    'v2_1': 'v2-inference-v.yaml',
    'sdxl_base': 'sd_xl_base.yaml',
    'sdxl_refiner': 'sd_xl_refiner.yaml',
    'control_multi_sdxl': 'sd_xl_base.yaml',
    'control_multi_animatediff_sdxl': 'sd_xl_base.yaml',
}


def is_sdxl_model(model_version: str) -> bool:
    """Check if a model version string refers to an SDXL model."""
    return 'sdxl' in model_version.lower()


def is_flux_model(model_version: str) -> bool:
    """Check if a model version string refers to a Flux (diffusers) model.

    Flux models take an entirely separate render path (core/flux.py) — a
    flow-matching DiT driven through `diffusers` — rather than the CompVis /
    k-diffusion stack the SD/SDXL versions use.
    """
    return 'flux' in model_version.lower()


def is_hidream_model(model_version: str) -> bool:
    """Check if a model version selects the HiDream-O1 edit backend."""
    return 'hidream' in model_version.lower()


def is_edit_model(model_version: str) -> bool:
    """Whether rendering is delegated to an external image-edit backend."""
    return is_flux_model(model_version) or is_hidream_model(model_version)


def configure_sdxl_clip_layer_norm(sd_model: Any, apply_final_layer_norm: bool) -> None:
    """Toggle CLIP's final layer norm on SDXL's clip-skipped hidden state.

    True = notebook/A1111 behaviour (sd_hijack applies it). False leaves the raw
    hidden state, ~10x larger in magnitude, which follows the prompt much harder but
    does not match the notebook. See DiffusionConfig.clip_final_layer_norm.
    """
    conditioner = getattr(sd_model, 'conditioner', None)
    if conditioner is None:
        return
    for embedder in getattr(conditioner, 'embedders', []):
        if hasattr(embedder, 'layer') and hasattr(embedder, 'transformer'):
            embedder.apply_final_layer_norm = apply_final_layer_norm
    if not apply_final_layer_norm:
        print("SDXL CLIP: final layer norm DISABLED (stronger prompt, not notebook-exact)")


def configure_clip_skip(sd_model: Any, clip_skip: int) -> None:
    """Configure SD1.x CLIP conditioning like WarpFusion's A1111 hijack."""
    encoder = getattr(sd_model, 'cond_stage_model', None)
    if encoder is None or not hasattr(encoder, 'layer'):
        return
    clip_skip = min(max(int(clip_skip), 1), 12)
    if clip_skip == 1:
        encoder.layer = 'last'
        encoder.layer_idx = None
    else:
        encoder.layer = 'hidden'
        encoder.layer_idx = -clip_skip
    print(f"Using clip_skip value of {clip_skip}")


def _ensure_vendor_on_path():
    """Add vibewarp/vendor to sys.path so bare 'ldm', 'sgm', 'k_diffusion'
    imports resolve to the vendored (patched) copies.

    The vendored packages use bare top-level imports internally (e.g.
    `from ldm.modules...`), so the vendor dir must be importable as a
    top-level location. Everything needed is vendored — no external clone
    directories are consulted.
    """
    import sys
    vendor_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'vendor'))
    if vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)


def load_sd_model(
    ckpt_path: str,
    config_path: str,
    vae_ckpt_path: Optional[str] = None,
    half_precision: bool = True,
    device: str = 'cuda',
) -> Any:
    """Load a Stable Diffusion model from checkpoint.

    On first call: instantiate architecture (with weight-init skipped),
    load state dict, optionally replace VAE, convert to fp16, then save
    a pickle cache in the install's `.vibewarp_cache/` (keyed by checkpoint,
    VAE, precision, and a fingerprint of the vendored model code).

    On subsequent calls with the same checkpoint+VAE and unchanged code:
    load directly from the pickle cache — skips both architecture init and
    safetensors parsing. A cache that fails to load (moved install, changed
    code, corruption) is discarded and rebuilt, never fatal.

    Args:
        ckpt_path: path to .safetensors or .ckpt checkpoint
        config_path: path to OmegaConf .yaml config for the model architecture
        vae_ckpt_path: optional path to a replacement VAE checkpoint
        half_precision: whether to use float16
        device: target device ('cuda' or 'cpu')

    Returns:
        Loaded SD model instance.
    """
    _ensure_vendor_on_path()
    _configure_hf_logging()

    # Attention classes are instantiated with the model and preserved by the
    # pickle. Keep parity/manual-attention models separate from normal
    # SDPA/xformers caches or the requested backend is silently ignored.
    parity_mode = os.environ.get('VIBEWARP_PARITY_MODE', '').lower() in ('1', 'true', 'yes')
    cache_variant = 'parity-manual-attention' if parity_mode else ''
    pkl = _cache_path(ckpt_path, vae_ckpt_path, half_precision, cache_variant)

    if _cache_is_valid(pkl, ckpt_path, vae_ckpt_path):
        from vibewarp.core.timing import log_time
        t0 = time.time()
        print(f"Loading model from cache: {os.path.basename(pkl)}")
        try:
            with open(pkl, 'rb') as f:
                model = pickle.load(f)
            elapsed = time.time() - t0
            log_time(f"SD model pkl load ({os.path.basename(ckpt_path)})", elapsed)
            model.to(device)
            model.eval()
            return model
        except Exception as e:
            # The cache is a pure speedup; a cache written by a different
            # environment (moved install, changed vendored code, Python/torch
            # skew, corruption) must not be fatal. Warn, discard it, and rebuild
            # from the checkpoint below — the fresh load rewrites the cache.
            print(f"  Cache load failed ({type(e).__name__}: {e}); "
                  f"rebuilding from checkpoint")
            try:
                os.remove(pkl)
            except OSError:
                pass

    # ---- First-time load ----
    t0 = time.time()
    config = OmegaConf.load(config_path)
    # Skip the pretrained CLIP / OpenCLIP downloads: the checkpoint overwrites
    # those weights below anyway. Verified after load_state_dict.
    with _skip_weight_init(), _skip_pretrained_text_encoders():
        model = instantiate_from_config(config.model)
    print(f"  Model instantiated: {time.time() - t0:.1f}s")

    sd = load_state_dict(ckpt_path, location='cpu')

    if vae_ckpt_path is not None:
        print(f"Loading VAE from {vae_ckpt_path}")
        vae_sd = load_state_dict(vae_ckpt_path, location='cpu')
        sd = {
            k: vae_sd[k[len("first_stage_model."):]]
            if k.startswith("first_stage_model.") and k[len("first_stage_model."):] in vae_sd
            else v
            for k, v in sd.items()
        }

    if 'edm_vpred.sigma_max' in sd:
        model.parameterization = 'v'
    else:
        model.parameterization = getattr(model, 'parameterization', 'eps')

    m, u = model.load_state_dict(sd, strict=False)
    if m:
        print(f"Missing keys: {len(m)}")
    if u:
        print(f"Unexpected keys: {len(u)}")

    # The text encoders were built empty (no pretrained download), so the
    # checkpoint MUST have supplied them. If it didn't, we'd be one silent step
    # from rendering with a randomly initialised text encoder — rebuild that
    # part from the pretrained weights instead, which is the old behaviour.
    missing_encoder = _missing_text_encoder_keys(m)
    if missing_encoder:
        print(f"  Checkpoint is missing {len(missing_encoder)} text-encoder key(s) "
              f"(e.g. {missing_encoder[0]}) — falling back to pretrained CLIP weights")
        del model
        gc.collect()
        with _skip_weight_init():
            model = instantiate_from_config(config.model)
        model.parameterization = 'v' if 'edm_vpred.sigma_max' in sd else \
            getattr(model, 'parameterization', 'eps')
        m, u = model.load_state_dict(sd, strict=False)
        still_missing = _missing_text_encoder_keys(m)
        if still_missing:
            print(f"  WARNING: {len(still_missing)} text-encoder key(s) still missing "
                  f"after the pretrained fallback")

    del sd
    gc.collect()

    if half_precision:
        model.half()
    model.eval()

    # Save pickle cache (CPU, device-agnostic)
    try:
        os.makedirs(os.path.dirname(pkl), exist_ok=True)
        t1 = time.time()
        print(f"Saving model cache: {os.path.basename(pkl)} ...")
        with open(pkl, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Cache saved: {time.time() - t1:.1f}s")
    except Exception as e:
        print(f"Warning: could not save model cache: {e}")

    model.to(device)
    torch.cuda.empty_cache()
    gc.collect()

    return model


def wrap_model_for_kdiffusion(
    sd_model: Any,
    model_version: str = 'control_multi_v15',
    quantize: bool = True,
) -> Tuple[Any, Any]:
    """Wrap an SD model with k-diffusion denoisers.

    Args:
        sd_model: loaded SD model
        model_version: model version string
        quantize: whether to quantize sigma schedule

    Returns:
        (model_wrap, model_wrap_cfg) tuple
    """
    from vibewarp.vendor.k_diffusion import external

    # Choose denoiser based on parameterization
    param = getattr(sd_model, 'parameterization', 'eps')
    if param == 'v':
        model_wrap = external.CompVisVDenoiser(sd_model, quantize=quantize)
    else:
        model_wrap = external.CompVisDenoiser(sd_model, quantize=quantize)

    # Create CFG wrapper with static thresholding (matches notebook's dynamic_thresh=30)
    model_wrap_cfg = CFGDenoiser(model_wrap)

    return model_wrap, model_wrap_cfg


def make_static_thresh_model_fn(model, value=30.):
    """Wrap a model to clamp its output to [-value, value].

    Matches notebook's make_static_thresh_model_fn(model, dynamic_thresh).
    """
    def model_fn(x, sigma, **kwargs):
        return model(x, sigma, **kwargs).clamp(-value, value)
    return model_fn


def _expand_to_frames(cond, frames: int):
    """Broadcast a single conditioning row across the frame batch.

    Per-frame prompts arrive already stacked as [F, 77, D]; a constant prompt
    arrives as [1, 77, D] and has to cover every frame. SDXL conditioning is a dict
    ({'crossattn', 'vector'}), so every tensor in it gets the same treatment.
    """
    if isinstance(cond, dict):  # SDXL
        return {key: _expand_to_frames(value, frames) if torch.is_tensor(value) else value
                for key, value in cond.items()}
    if torch.is_tensor(cond) and cond.shape[0] == 1 and frames > 1:
        return cond.repeat(frames, *([1] * (cond.dim() - 1)))
    return cond


def _index_cond(cond, index):
    """Take a context window's rows out of the conditioning (tensor or SDXL dict)."""
    if isinstance(cond, dict):
        return {key: (value[index] if torch.is_tensor(value) else value)
                for key, value in cond.items()}
    return cond[index]


def _slice_window_kwargs(kwargs: dict, index, frames: int) -> dict:
    """Take a context window's rows out of any per-frame kwarg (e.g. image_cond)."""
    out = dict(kwargs)
    value = kwargs.get('image_cond')
    if torch.is_tensor(value) and value.shape[0] == frames:
        out['image_cond'] = value[index]
    return out


class CFGDenoiser(torch.nn.Module):
    """Classifier-free guidance wrapper for k-diffusion sampling.

    Runs the model twice (conditional + unconditional) and blends
    according to cfg_scale. Supports both SD1.5 (tensor cond) and
    SDXL (dict cond with crossattn + vector keys).

    With `_adiff_ctx_schedule` set, switches to the AnimateDiff path: the latent is
    a batch of frames, and each denoiser call slides context windows across it.
    """

    def __init__(self, model):
        super().__init__()
        self.inner_model = model
        # Set by the AnimateDiff renderer: one list of frame-index windows per
        # denoiser call, popped as sampling proceeds.
        self._adiff_ctx_schedule = None
        self._adiff_zero_uncond_hints = False
        # Optional spatial wrapper around the raw k-diffusion model. Keeping it
        # below CFG/context scheduling ensures those schedules advance once per
        # denoiser evaluation, not once per spatial tile.
        self._tiled_inner_model = None

    def _call_inner_model(self, x, sigma, **kwargs):
        model = self._tiled_inner_model or self.inner_model
        return model(x, sigma, **kwargs)

    def _reinject_stylized(self, x, sigma) -> None:
        """Overwrite the batch's leading frames with the previous batch's latents.

        `x[:n] = stylized_latents + noise[:n] * sigma[0]` — the same latents re-noised
        to the current step. As sampling proceeds sigma shrinks, so the anchor tightens
        onto the previous batch's content. `lerp_reinject_stylized_frames` ramps the
        last few anchored frames back toward the batch's own latent so the anchor
        fades out instead of ending abruptly.
        """
        latents = getattr(self, '_adiff_stylized_latents', None)
        noise = getattr(self, '_adiff_noise', None)
        if latents is None or noise is None or latents.shape[0] == 0:
            return

        count = min(latents.shape[0], x.shape[0])
        injected = latents[:count].to(x.dtype) + noise[:count].to(x.dtype) * sigma[0]

        lerp_frames = int(getattr(self, '_adiff_lerp_frames', 0) or 0)
        if lerp_frames > 0 and count > 1:
            ramp = min(lerp_frames, count - 1)
            weights = torch.cat([
                torch.zeros(count - ramp, device=x.device),
                torch.linspace(0, 1, ramp, device=x.device),
            ]).view(-1, 1, 1, 1).to(x.dtype)
            x[:count] = x[:count] * weights + injected * (1 - weights)
        else:
            x[:count] = injected

    def _forward_animatediff(self, x, sigma, uncond, cond, cfg_scale, **kwargs):
        """Denoise a batch of frames with sliding temporal context windows.

        Port of the notebook's CFGDenoiser_adiff (dump line 10269). Every window is
        exactly `context_length` frames — which is what `set_video_length` was told —
        so the motion module always sees the temporal length it expects. Windows
        overlap, so each frame's prediction is the mean over the windows covering it.
        """
        sd_model = self._get_sd_model()
        frames = x.shape[0]

        # Pin the overlap frames to the PREVIOUS batch's render (notebook
        # reinject_stylized, default on). Done here — inside the denoiser, on every
        # call — rather than at init, because with style_strength=1 the init latent
        # is discarded (skip_steps=0) and picking a stylized init image achieves
        # nothing. Mutating x in place is deliberate: the sampler carries the same
        # tensor forward, so this steers the whole trajectory of those frames, which
        # is what actually holds identity across a batch seam.
        self._reinject_stylized(x, sigma)

        # One window list per denoiser call. Samplers that evaluate the model twice
        # per step consume two entries per step; the renderer pre-doubles the
        # schedule for those, mirroring the notebook's repeat_list.
        windows = self._adiff_ctx_schedule.pop(0) if self._adiff_ctx_schedule else [list(range(frames))]

        cond_b = _expand_to_frames(cond, frames)
        uncond_b = _expand_to_frames(uncond, frames)

        cond_final = torch.zeros_like(x)
        uncond_final = torch.zeros_like(x)
        counts = torch.zeros((frames, 1, 1, 1), device=x.device, dtype=x.dtype)

        unet = sd_model.model.diffusion_model if sd_model is not None else None
        ones = torch.ones(frames, device=x.device)
        zeros = torch.zeros(frames, device=x.device)

        # ControlNet hints are per-frame ([F,3,H,W] per net) and live on the model.
        # Swapping in each window's slice lets the existing ControlNet forward run
        # unchanged — it just sees a batch that matches the window.
        all_hints = dict(getattr(sd_model, '_cn_hints', {}) or {}) if sd_model else {}

        with torch.amp.autocast('cuda'):
            for ids in windows:
                index = torch.as_tensor(ids, device=x.device, dtype=torch.long)
                window_kwargs = _slice_window_kwargs(kwargs, index, frames)

                if sd_model is not None and all_hints:
                    sd_model._cn_hints = {
                        key: (hint[index] if hint.shape[0] == frames else hint)
                        for key, hint in all_hints.items()
                    }
                if unet is not None:
                    unet.uc_mask_shape = ones[index]
                out_cond = self._call_inner_model(
                    x[index], sigma[index], cond=_index_cond(cond_b, index), **window_kwargs)

                if sd_model is not None and all_hints and self._adiff_zero_uncond_hints:
                    sd_model._cn_hints = {
                        key: torch.zeros_like(hint)
                        for key, hint in sd_model._cn_hints.items()
                    }
                if unet is not None:
                    unet.uc_mask_shape = zeros[index]
                out_uncond = self._call_inner_model(
                    x[index], sigma[index], cond=_index_cond(uncond_b, index), **window_kwargs)

                cond_final[index] += out_cond.to(cond_final.dtype)
                uncond_final[index] += out_uncond.to(uncond_final.dtype)
                counts[index] += 1

        if sd_model is not None and all_hints:
            sd_model._cn_hints = all_hints   # restore the full-batch hints

        # Every frame must be covered by at least one window, or it would keep its
        # zero-initialised prediction and decode to garbage.
        if not bool((counts > 0).all()):
            raise RuntimeError(
                "AnimateDiff context schedule left some frames uncovered")

        cond_final = cond_final / counts
        uncond_final = uncond_final / counts
        return uncond_final + (cond_final - uncond_final) * cfg_scale

    def forward(self, x, sigma, uncond, cond, cond_scale=None, cfg_scale=None, **kwargs):
        cfg_scale = cond_scale if cond_scale is not None else (cfg_scale or 1.0)
        # Per-step CFG schedule: pop next value
        if hasattr(self, '_cfg_schedule') and self._cfg_schedule:
            cfg_scale = self._cfg_schedule.pop()

        # AnimateDiff denoises a whole batch of frames at once, and the motion
        # module attends across that batch axis — so it needs its own path
        # (notebook: CFGDenoiser_adiff). Crucially it runs cond and uncond as
        # SEPARATE forwards; the standard path below concatenates them, which would
        # make the temporal axis 2F and corrupt every temporal attention.
        if getattr(self, '_adiff_ctx_schedule', None):
            return self._forward_animatediff(x, sigma, uncond, cond, cfg_scale, **kwargs)

        # Update ControlNet timestep ratio for time gating
        sd_model = self._get_sd_model()
        if sd_model is not None and hasattr(sd_model, '_cn_t_ratio'):
            # Convert sigma to timestep ratio: higher sigma = earlier in diffusion = lower t_ratio
            sigma_max = self.inner_model.sigmas[-1].item() if hasattr(self.inner_model, 'sigmas') else 1.0
            sigma_val = sigma[0].item() if sigma.numel() > 0 else 0.0
            sd_model._cn_t_ratio = 1.0 - (sigma_val / max(sigma_max, 1e-6))

        # Handle SDXL dict conditioning (crossattn + vector)
        if isinstance(cond, dict) and 'crossattn' in cond:
            cond_in = torch.cat([uncond['crossattn'], cond['crossattn']])
            # Combine vector conditioning for SDXL
            uc_vector = uncond.get('vector')
            c_vector = cond.get('vector')
            if uc_vector is not None and c_vector is not None:
                # Expand vectors to match batch size
                if uc_vector.shape[0] < x.shape[0]:
                    uc_vector = uc_vector.repeat(x.shape[0], 1, 1)
                if c_vector.shape[0] < x.shape[0]:
                    c_vector = c_vector.repeat(x.shape[0], 1, 1)
                vector_in = torch.cat([uc_vector, c_vector])
                # Store on model for apply_model_vector to pick up
                sd_model = self._get_sd_model()
                if sd_model is not None and hasattr(sd_model, 'conditioner'):
                    sd_model.conditioner.vector_in = vector_in
        else:
            cond_in = torch.cat([uncond, cond])

        # Match notebook sd_batch_size: with 2, uc+c share one UNet call;
        # with 1, they are evaluated in separate calls. x itself is normally
        # batch 1 for this CFG path, repeated once per conditioning row.
        x_in = torch.cat([x] * cond_in.shape[0])
        sigma_in = torch.cat([sigma] * cond_in.shape[0])

        # Notebook CFGDenoiser marks which batch rows are uncond (0) vs cond (1)
        # on the UNet; consumed by IP-Adapter hooks (uncond rows get the uncond
        # image embedding) and CN zero-uncond logic. Batch layout here is
        # [uncond..., cond...] (x doubled), so the first half is uncond.
        if sd_model is not None and hasattr(sd_model, 'model'):
            uc_rows = cond_in.shape[0] - x.shape[0] if cond_in.shape[0] > x.shape[0] else 1
            uc_mask = torch.ones(cond_in.shape[0], device=cond_in.device)
            uc_mask[:uc_rows] = 0
            sd_model.model.diffusion_model.uc_mask_shape = uc_mask

        # Forward image_cond for ControlNet (duplicate for uncond+cond batch)
        inner_kwargs = dict(kwargs)
        if 'image_cond' in inner_kwargs:
            ic = inner_kwargs.pop('image_cond')
            inner_kwargs['image_cond'] = torch.cat([ic, ic])

        if os.environ.get('VIBEWARP_DEBUG_DIFFUSION') and not getattr(self, '_input_probed', False):
            def diag_tensor(value):
                cpu = value.detach().float().cpu().contiguous()
                return {
                    'shape': list(value.shape), 'dtype': str(value.dtype),
                    'sha256_f32': hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
                    'first8': [round(v, 6) for v in cpu.flatten()[:8].tolist()],
                }
            probe = {
                'x': diag_tensor(x_in), 'sigma': diag_tensor(sigma_in),
                'uc': diag_tensor(cond_in[:1]), 'c': diag_tensor(cond_in[1:]),
                'sd_batch_size': getattr(self, '_sd_batch_size', 2),
            }
            # SDXL: the pooled/size vector drives style as hard as the crossattn,
            # so a mismatch shows up as "same composition, different look".
            _y = getattr(getattr(sd_model, 'conditioner', None), 'vector_in', None)
            if _y is not None:
                probe['y'] = diag_tensor(_y)
            print('[diag] cfg_model_inputs=' + json.dumps(probe, sort_keys=True))
            self._input_probed = True

        # Autocast only around UNet calls, not sampler arithmetic (matches notebook).
        batch_size = max(1, int(getattr(self, '_sd_batch_size', 2)))
        with torch.amp.autocast('cuda'):
            outputs = []
            for start in range(0, cond_in.shape[0], batch_size):
                stop = start + batch_size
                batch_kwargs = dict(inner_kwargs)
                if 'image_cond' in batch_kwargs:
                    batch_kwargs['image_cond'] = batch_kwargs['image_cond'][start:stop]
                outputs.append(self._call_inner_model(
                    x_in[start:stop], sigma_in[start:stop],
                    cond=cond_in[start:stop], **batch_kwargs))
            uncond_out, cond_out = torch.cat(outputs).chunk(2)
        return uncond_out + cfg_scale * (cond_out - uncond_out)

    def _get_sd_model(self) -> Optional[Any]:
        """Get the underlying SD model from the k-diffusion wrapper."""
        inner = self.inner_model
        if hasattr(inner, 'inner_model'):
            return inner.inner_model
        return None


def load_controlnet(
    controlnet_path: str,
    model_version: str = 'control_multi_v15',
    device: str = 'cuda',
) -> Any:
    """Load a ControlNet model as a proper nn.Module.

    Uses the same pickle cache strategy as load_sd_model — fast on second run.

    Args:
        controlnet_path: path to ControlNet .pth or .safetensors file
        model_version: model version string for architecture selection
        device: target device

    Returns:
        Instantiated ControlNet nn.Module with weights loaded.
    """
    from vibewarp.vendor.cldm.cldm import instantiate_controlnet

    parity_mode = os.environ.get('VIBEWARP_PARITY_MODE', '').lower() in ('1', 'true', 'yes')
    attention_variant = 'parity-manual-attention' if parity_mode else 'default-attention'
    # v2 matches the shipped cldm_v15.yaml head layout (8 heads, legacy=False).
    cache_variant = f'controlnet-arch-v2-{attention_variant}'
    pkl = _cache_path(controlnet_path, None, True, cache_variant)

    if _cache_is_valid(pkl, controlnet_path, None):
        from vibewarp.core.timing import log_time
        t0 = time.time()
        print(f"  Loading ControlNet from cache: {os.path.basename(pkl)}")
        with open(pkl, 'rb') as f:
            cn_model = pickle.load(f)
        log_time(f"CN pkl load ({os.path.basename(controlnet_path)})", time.time() - t0)
        cn_model.to(device)
        return cn_model

    sd = load_state_dict(controlnet_path, location='cpu')
    with _skip_weight_init():
        cn_model = instantiate_controlnet(sd, model_version=model_version, device='cpu')
    del sd
    gc.collect()

    try:
        os.makedirs(os.path.dirname(pkl), exist_ok=True)
        with open(pkl, 'wb') as f:
            pickle.dump(cn_model, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Warning: could not cache ControlNet: {e}")

    cn_model.to(device)
    return cn_model


def _model_device(sd_model: Any) -> torch.device:
    try:
        return next(sd_model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device('cpu')


def sdxl_get_batch(keys, value_dict: dict, N: list, device='cuda'):
    """Build the SDXL conditioner batch (notebook `get_batch`, cell ~5808).

    SDXL's conditioner is keyed by embedder `input_key`, not by prompt: it needs
    'txt' plus the size/crop micro-conditioning tensors. Returns (batch, batch_uc).
    """
    import math

    import numpy as np

    batch: Dict[str, Any] = {}
    batch_uc: Dict[str, Any] = {}
    for key in keys:
        if key == 'txt':
            if len(value_dict['prompt']) != N[0]:
                batch['txt'] = (
                    np.repeat([value_dict['prompt']], repeats=math.prod(N))
                    .reshape(N).tolist()
                )
            else:
                batch['txt'] = value_dict['prompt']
            batch_uc['txt'] = (
                np.repeat([value_dict['negative_prompt']], repeats=math.prod(N))
                .reshape(N).tolist()
            )
        elif key == 'original_size_as_tuple':
            # Height first — the order is load-bearing.
            batch['original_size_as_tuple'] = torch.tensor(
                [value_dict['orig_height'], value_dict['orig_width']]
            ).to(device).repeat(*N, 1)
        elif key == 'crop_coords_top_left':
            batch['crop_coords_top_left'] = torch.tensor(
                [value_dict['crop_coords_top'], value_dict['crop_coords_left']]
            ).to(device).repeat(*N, 1)
        elif key == 'aesthetic_score':
            batch['aesthetic_score'] = torch.tensor(
                [value_dict['aesthetic_score']]).to(device).repeat(*N, 1)
            batch_uc['aesthetic_score'] = torch.tensor(
                [value_dict['negative_aesthetic_score']]).to(device).repeat(*N, 1)
        elif key == 'target_size_as_tuple':
            batch['target_size_as_tuple'] = torch.tensor(
                [value_dict['target_height'], value_dict['target_width']]
            ).to(device).repeat(*N, 1)
        else:
            batch[key] = value_dict[key]

    for key in batch:
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def setup_sdxl_model(
    sd_model: Any,
    width: int = 1024,
    height: int = 1024,
) -> None:
    """Patch an SDXL model to work with the standard SD1.5-style API.

    After calling this, sd_model.get_learned_conditioning() will return
    a dict with 'crossattn' and 'vector' keys, and sd_model.apply_model()
    will automatically pass vector_in to the conditioner.

    This replicates the notebook's SDXL monkey-patching (lines 1586-1649).

    Args:
        sd_model: loaded SDXL model (with conditioner attribute)
        width: target image width (for conditioning metadata)
        height: target image height (for conditioning metadata)
    """
    if not hasattr(sd_model, 'conditioner'):
        return

    # sgm's DiffusionEngine has no alphas_cumprod — it discretizes elsewhere —
    # but k-diffusion's CompVisDenoiser reads it at construction. The notebook
    # registers it from LegacyDDPMDiscretization; without this the wrap dies with
    # "'DiffusionEngine' object has no attribute 'alphas_cumprod'".
    if not hasattr(sd_model, 'alphas_cumprod'):
        from vibewarp.vendor.sgm.modules.diffusionmodules.discretizer import (
            LegacyDDPMDiscretization,
        )
        # Only reached when nothing else registered alphas_cumprod. AnimateDiff
        # (non-Hotshot) registers its OWN betas in register_animatediff_schedule and
        # so never gets here; HotshotXL deliberately does not (it keeps the standard
        # schedule), which is exactly the notebook's
        # `if 'animatediff' in model_version and not mm.is_hotshot` split. Do NOT
        # prefer `ad_alphas_cumprod` here — that attribute is set for Hotshot too,
        # and using it would silently give Hotshot the wrong noise schedule.
        alphas = torch.from_numpy(LegacyDDPMDiscretization().alphas_cumprod)
        # from_numpy gives a CPU float64 tensor, and we are registering onto a model
        # that has already been moved to the GPU — so pin it to the model's device
        # and to float32 (as ldm stores it). Otherwise CompVisDenoiser derives its
        # sigmas on the CPU and sampling dies with a cuda/cpu device mismatch.
        alphas = alphas.to(device=_model_device(sd_model), dtype=torch.float32)
        sd_model.register_buffer('alphas_cumprod', alphas)

    # sgm's UNetModel deliberately dropped `self.dtype` ("use_fp16 was dropped and
    # has no effect anymore") because it assumes autocast. Our ControlNet forward
    # is shared with SD1.5 and casts the input with `x.type(self.dtype)`, so give
    # the UNet the attribute back, taken from its real parameter dtype — the same
    # idiom the notebook uses for its own get_dtype patch.
    unet = sd_model.model.diffusion_model
    if not hasattr(unet, 'dtype'):
        unet.dtype = next(unet.parameters()).dtype

    # Alias cond_stage_model to conditioner for compatibility
    sd_model.cond_stage_model = sd_model.conditioner
    sd_model.model.conditioning_key = 'c_crossattn'

    width_height = (width, height)

    def apply_model_vector(x_noisy, t, cond, **kwargs):
        if isinstance(cond, dict) and 'c_crossattn' in cond:
            # Standard cond dict — wrap crossattn with vector
            crossattn = cond['c_crossattn']
            if isinstance(crossattn, list):
                crossattn = crossattn[0]
            wrapped = {
                'crossattn': crossattn,
                'vector': sd_model.conditioner.vector_in,
            }
            # Pass through y if present
            if 'y' in cond:
                wrapped['vector'] = cond['y']
            return sd_model.model.forward(x_noisy, t, wrapped, **kwargs)
        elif isinstance(cond, torch.Tensor):
            # Plain tensor cond — wrap with vector
            wrapped = {
                'crossattn': cond,
                'vector': sd_model.conditioner.vector_in,
            }
            return sd_model.model.forward(x_noisy, t, wrapped, **kwargs)
        else:
            return sd_model.model.forward(x_noisy, t, cond, **kwargs)

    def get_first_stage_encoding_sdxl(z):
        """SDXL doesn't scale latents (identity encoding)."""
        return z

    def get_learned_conditioning_sdxl(prompts):
        """SDXL conditioning via dual text encoders.

        Returns dict with 'crossattn' (77, D) and 'vector' (D_vec) keys.
        """
        W, H = width_height

        def get_unique_embedder_keys_from_conditioner(conditioner):
            return list(set([x.input_key for x in conditioner.embedders]))

        init_dict = {
            "orig_width": W,
            "orig_height": H,
            "target_width": W,
            "target_height": H,
        }

        if isinstance(prompts, str):
            prompts = [prompts]

        value_dict = dict(init_dict)
        value_dict["prompt"] = prompts
        value_dict["negative_prompt"] = ['']
        value_dict["crop_coords_top"] = 0
        value_dict["crop_coords_left"] = 0
        value_dict["aesthetic_score"] = 6.0
        value_dict["negative_aesthetic_score"] = 2.5

        # NB: `sgm.util.default_cond_input` does not exist — the old code's
        # ImportError fallback silently built a batch with no 'txt' key, so the
        # conditioner died with KeyError: 'txt'. Use the notebook's get_batch.
        batch_c, _ = sdxl_get_batch(
            get_unique_embedder_keys_from_conditioner(sd_model.conditioner),
            value_dict,
            [len(prompts)],
            device=_model_device(sd_model),
        )

        ucg_rates = []
        for embedder in sd_model.conditioner.embedders:
            ucg_rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0

        c = sd_model.conditioner(batch_c)

        for embedder, rate in zip(sd_model.conditioner.embedders, ucg_rates):
            embedder.ucg_rate = rate

        return c

    sd_model.apply_model = apply_model_vector
    sd_model.get_first_stage_encoding = get_first_stage_encoding_sdxl
    sd_model.get_learned_conditioning = get_learned_conditioning_sdxl


def split_sdxl_cond(
    sd_model: Any,
    prompts: list,
) -> Dict[str, torch.Tensor]:
    """Encode multiple prompts for SDXL and stack into crossattn + vector.

    Equivalent to the notebook's split_stack_sdxl_cond.

    Args:
        sd_model: loaded SDXL model
        prompts: list of prompt strings

    Returns:
        Dict with 'crossattn' (B, 77, D) and 'vector' (B, D_vec) tensors.
    """
    crossattn_list = []
    vector_list = []
    for prompt in prompts:
        cond = sd_model.get_learned_conditioning(prompt)
        crossattn_list.append(cond['crossattn'][:, :77, :])
        vector_list.append(cond['vector'])
    return {
        'crossattn': torch.stack(crossattn_list),
        'vector': torch.stack(vector_list),
    }
