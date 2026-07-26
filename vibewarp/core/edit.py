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
    from vibewarp.core.model_loader import (
        is_flux_model, is_hidream_model, is_mage_flow_edit_model,
        is_qwen_edit_model)

    if is_flux_model(config.model_version):
        from vibewarp.core.flux import load_flux_pipeline
        return load_flux_pipeline(config, device=device)
    if is_hidream_model(config.model_version):
        from vibewarp.core.hidream_comfy import ComfyHiDreamClient
        return ComfyHiDreamClient(config)
    if is_qwen_edit_model(config.model_version):
        from vibewarp.core.qwen_comfy import ComfyQwenEditClient
        return ComfyQwenEditClient(config)
    if is_mage_flow_edit_model(config.model_version):
        from vibewarp.core.mage_comfy import ComfyMageFlowEditClient
        return ComfyMageFlowEditClient(config)
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
    reference_images: list[Image.Image] | None = None,
    debug_dir: str | None = None,
    frame_num: int | None = None,
) -> Image.Image:
    """Render one prepared frame using the configured edit backend."""
    from vibewarp.core.model_loader import (
        is_flux_model, is_hidream_model, is_mage_flow_edit_model,
        is_qwen_edit_model)

    if (is_hidream_model(config.model_version)
            or is_qwen_edit_model(config.model_version)
            or is_mage_flow_edit_model(config.model_version)):
        return backend.render(
            config=config,
            edit_image=edit_image,
            reference_images=reference_images,
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
            reference_images=reference_images,
            debug_dir=debug_dir,
            frame_num=frame_num,
        )
    raise ValueError(f"Unsupported edit model: {config.model_version!r}")
