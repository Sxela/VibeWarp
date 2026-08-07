"""Local SD/SDXL U-Net acceleration state.

The actual block execution lives in ``controlnet.controlled_unet_forward`` so
ControlNet and FreeU keep sharing one forward implementation. This module owns
only configuration, per-frame state, and optional torch.compile setup.
"""

from __future__ import annotations

import os
from typing import Any


VALID_CACHE_MODES = ('off', 'deepcache', 'first_block')


def _unet_from(sd_model: Any) -> Any:
    model = getattr(sd_model, 'model', None)
    return getattr(model, 'diffusion_model', None)


def configure_unet_acceleration(
    sd_model: Any,
    *,
    cache_mode: str = 'off',
    cache_interval: int = 2,
    cache_threshold: float = 0.05,
    compile_unet: bool = False,
    compile_cache_dir: str = '',
) -> None:
    """Configure the already-patched unified U-Net forward."""
    if cache_mode not in VALID_CACHE_MODES:
        raise ValueError(f"Unknown U-Net cache mode: {cache_mode}")
    if compile_unet and cache_mode != 'off':
        raise ValueError(
            "Compiled U-Net and U-Net caching cannot currently be combined")

    unet = _unet_from(sd_model)
    if unet is None:
        raise ValueError("Cannot configure U-Net acceleration: U-Net is missing")
    unet._vw_cache_mode = cache_mode
    unet._vw_cache_interval = max(2, int(cache_interval))
    unet._vw_cache_threshold = max(0.0, float(cache_threshold))
    reset_unet_cache(sd_model)

    if cache_mode == 'deepcache':
        print(
            f"  U-Net acceleration: DeepCache "
            f"(interval={unet._vw_cache_interval}, reset per frame)")
    elif cache_mode == 'first_block':
        print(
            f"  U-Net acceleration: First Block Cache "
            f"(threshold={unet._vw_cache_threshold:g}, reset per frame)")
    else:
        print("  U-Net cache: off")

    if compile_unet:
        import torch

        if not hasattr(torch, 'compile'):
            raise RuntimeError("Compile U-Net requires PyTorch 2 or newer")
        cache_dir = configure_persistent_compile_cache(compile_cache_dir)
        # Compile the unified forward rather than replacing the module. The
        # ControlNet apply_model closure retains this exact module instance.
        unet.forward = torch.compile(
            unet.forward, mode='reduce-overhead', fullgraph=False)
        unet._vw_compiled = True
        print(
            "  U-Net compilation enabled "
            f"(persistent cache={cache_dir}; first call per shape may compile)")
    else:
        unet._vw_compiled = False


def configure_persistent_compile_cache(cache_dir: str) -> str:
    """Use a stable Inductor/FX/Triton cache across VibeWarp processes."""
    cache_dir = os.path.abspath(
        cache_dir or os.path.join('.vibewarp_cache', 'torchinductor'))
    # Respect an explicit machine-level override. Otherwise keep artifacts out
    # of the ephemeral user temp folder and inside VibeWarp's ignored cache.
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', cache_dir)
    effective_dir = os.path.abspath(os.environ['TORCHINDUCTOR_CACHE_DIR'])
    os.makedirs(effective_dir, exist_ok=True)
    os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE'] = '1'
    return effective_dir


def compile_auxiliary_models(sd_model: Any, loaded_controlnets: dict) -> None:
    """Compile active ControlNets plus raw VAE encode/decode entry points."""
    import torch

    compiled_controlnets = 0
    seen = set()
    for data in loaded_controlnets.values():
        model = data.get('model')
        if model is None or id(model) in seen:
            continue
        seen.add(id(model))
        model.forward = torch.compile(
            model.forward, mode='reduce-overhead', fullgraph=False)
        model._vw_compiled = True
        compiled_controlnets += 1

    vae = getattr(sd_model, 'first_stage_model', None)
    if vae is None:
        raise ValueError("Cannot compile VAE: first_stage_model is missing")
    for method_name in ('encode', 'decode'):
        method = getattr(vae, method_name, None)
        if method is None:
            raise ValueError(f"Cannot compile VAE: {method_name} is missing")
        setattr(
            vae, method_name,
            torch.compile(method, mode='reduce-overhead', fullgraph=False))
    vae._vw_compiled = True
    print(
        f"  Auxiliary compilation enabled: {compiled_controlnets} ControlNet(s), "
        "VAE encode + decode")


def reset_unet_cache(sd_model: Any) -> None:
    """Drop all cached activations and start a new frame."""
    if sd_model is None:
        return
    unet = _unet_from(sd_model)
    if unet is None:
        return
    unet._vw_cache_runtime = {
        'evaluation': -1,
        'slot_cursor': 0,
        'slots': {},
        'hits': 0,
        'misses': 0,
    }


def begin_unet_evaluation(sd_model: Any) -> None:
    """Mark one logical denoiser evaluation before CFG batching/tiling."""
    if sd_model is None:
        return
    unet = _unet_from(sd_model)
    if unet is None:
        return
    if getattr(unet, '_vw_cache_mode', 'off') == 'off':
        return
    runtime = unet._vw_cache_runtime
    runtime['evaluation'] += 1
    runtime['slot_cursor'] = 0


def next_cache_slot(unet: Any) -> tuple[dict, bool]:
    """Return this CFG/tile stream's state and whether it is a refresh pass."""
    runtime = unet._vw_cache_runtime
    index = runtime['slot_cursor']
    runtime['slot_cursor'] += 1
    slot = runtime['slots'].setdefault(index, {})
    interval = max(2, int(getattr(unet, '_vw_cache_interval', 2)))
    refresh = runtime['evaluation'] % interval == 0
    return slot, refresh


def log_unet_cache_summary(sd_model: Any) -> None:
    """Print evidence that the selected cache actually ran for this frame."""
    if sd_model is None:
        return
    unet = _unet_from(sd_model)
    if unet is None:
        return
    mode = getattr(unet, '_vw_cache_mode', 'off')
    if mode == 'off':
        return
    runtime = unet._vw_cache_runtime
    print(
        f"  U-Net cache frame stats: mode={mode} "
        f"hits={runtime['hits']} full={runtime['misses']}")
