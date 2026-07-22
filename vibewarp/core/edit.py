"""Shared dispatch seam for non-SD frame-level image-edit backends.

The surrounding VibeWarp loop owns optical-flow warping, consistency blending,
colour matching and frame feedback.  Backends only receive one prepared edit
target, optional reference images, and text/noise controls, then return RGB.
"""

from typing import Any

from PIL import Image

from vibewarp.config import RunConfig


def load_edit_backend(config: RunConfig, device: str = 'cuda') -> Any:
    """Load the backend selected by ``config.model_version``."""
    from vibewarp.core.model_loader import is_flux_model, is_hidream_model

    if is_flux_model(config.model_version):
        from vibewarp.core.flux import load_flux_pipeline
        return load_flux_pipeline(config, device=device)
    if is_hidream_model(config.model_version):
        from vibewarp.core.hidream_comfy import ComfyHiDreamClient
        return ComfyHiDreamClient(config)
    raise ValueError(f"Unsupported edit model: {config.model_version!r}")


def render_edit_frame(
    backend: Any,
    config: RunConfig,
    edit_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    context_image: Image.Image | None = None,
    prev_context_image: Image.Image | None = None,
    mask_context_image: Image.Image | None = None,
    debug_dir: str | None = None,
    frame_num: int | None = None,
) -> Image.Image:
    """Render one prepared frame using the configured edit backend."""
    from vibewarp.core.model_loader import is_flux_model, is_hidream_model

    if is_hidream_model(config.model_version):
        return backend.render(
            config=config,
            edit_image=edit_image,
            context_image=context_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=height,
            debug_dir=debug_dir,
            frame_num=frame_num,
        )
    if is_flux_model(config.model_version):
        from vibewarp.core.flux import render_flux_frame
        return render_flux_frame(
            backend, config,
            edit_image=edit_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=height,
            context_image=context_image,
            prev_context_image=prev_context_image,
            mask_context_image=mask_context_image,
            debug_dir=debug_dir,
            frame_num=frame_num,
        )
    raise ValueError(f"Unsupported edit model: {config.model_version!r}")
