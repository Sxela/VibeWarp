"""Shared RunConfig serialization, construction, schema, and validation."""

from __future__ import annotations

import json
import os
import types
from dataclasses import MISSING, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Union, get_args, get_origin, get_type_hints

from vibewarp.config import (
    FLUX_MODEL_DEFAULTS, MAGE_MODEL_DEFAULTS, QWEN_MODEL_DEFAULTS, ColorConfig,
    EditReferenceConfig, FluxConfig, HiDreamConfig, MageFlowEditConfig,
    QwenEditConfig, RunConfig,
    flux_model_files,
    mage_model_defaults,
    qwen_model_files)
from vibewarp.controlnet_catalog import (
    CONTROLNET_CATALOG,
    CONTROLNET_MODES,
    normalize_model_version,
)
from vibewarp.ipadapter_catalog import (
    IPADAPTER_CATALOG,
    resolve_ipadapter_path,
)

FIELD_CHOICES = {
    # The two control_multi_* versions cover everything the UI needs: with no
    # ControlNets selected they behave as plain SD1.5 / SDXL. Deliberately omitted:
    # 'v1_5'/'v2_1' (redundant / SD2.1 unwanted) and 'control_multi_animatediff_sdxl'
    # (AnimateDiff's render path is broken — see the gap analysis). The engine still
    # loads all of them (core/model_loader.MODEL_CONFIGS) for settings files that
    # name them; this list only controls what the UI offers.
    # Selecting an SDXL version also switches the ControlNet panel to the SDXL nets.
    # External edit models ignore the SD checkpoint / ControlNet / IP-Adapter
    # stack while retaining VibeWarp's flow, consistency and colour pipeline.
    "RunConfig.model_version": [
        "control_multi_v15", "control_multi_sdxl",
        "flux2_klein_edit", "flux2_klein_9b_edit", "hidream_o1_edit",
        "qwen_image_edit_2511", "qwen_image_edit_2511_gguf",
        "mage_flow_edit", "mage_flow_edit_turbo",
    ],
    "FluxConfig.backend": ["comfy", "diffusers"],
    "HiDreamConfig.sampler": [
        "dpmpp_2m_sde_gpu", "dpmpp_2m_sde", "dpmpp_2m", "euler", "heun",
    ],
    "HiDreamConfig.scheduler": ["simple", "normal", "sgm_uniform"],
    "HiDreamConfig.patch_seam_passes": ["2", "4", "ramp_2_4", "ramp_2_4_8"],
    "HiDreamConfig.patch_seam_blend": ["median", "average", "window"],
    "QwenEditConfig.sampler": ["euler", "dpmpp_2m", "dpmpp_2m_sde"],
    "QwenEditConfig.scheduler": ["simple", "normal", "sgm_uniform"],
    "MageFlowEditConfig.sampler": ["euler", "dpmpp_2m", "dpmpp_2m_sde"],
    "MageFlowEditConfig.scheduler": ["simple", "normal", "sgm_uniform"],
    "RunConfig.lora_merge_precision": ["fp16", "fp32"],
    "RunConfig.animation_mode": ["Video Input"],
    "WarpConfig.warp_mode": ["use_image"],
    "WarpConfig.warp_towards_init": ["off", "init"],
    "WarpConfig.padding_mode": ["reflect", "edge", "symmetric", "wrap", "constant"],
    "DiffusionConfig.sampler": ["sample_euler_ancestral", "sample_euler", "sample_dpm_2", "sample_dpm_2_ancestral", "sample_heun", "sample_lms", "sample_dpmpp_2m", "sample_dpmpp_2s_ancestral", "sample_dpmpp_sde", "sample_lcm"],
    "DiffusionConfig.sampler_tile_size": [256, 512, 768, 1024],
    "DiffusionConfig.noise_mode": ["default", "fixed", "reconstructed"],
    "DiffusionConfig.guidance_mode": [
        "prev stylized", "prev warped", "prev warped + cc"],
    "VideoConfig.save_img_format": ["png", "jpg"],
    "ColorConfig.before_method": ["PDF", "LAB", "mean"],
    "ColorConfig.after_method": ["PDF", "LAB", "mean"],
    "ColorConfig.colormatch_method": ["PDF", "LAB", "mean"],
    "ColorConfig.colormatch_mode": ["before", "after", "both"],
    # 'custom' keeps hand-authored layer_weights/zero_uncond instead of deriving
    # them from a preset — see vibewarp.controlnet_catalog.resolve_mode.
    "ControlNetEntry.mode": list(CONTROLNET_MODES),
    "ControlNetConfig.mode": ["internal", "external"],
    "ControlNetConfig.cond_image_src": ["init", "stylized", "cond_video"],
    "ControlNetConfig.depth_detector": ["Midas", "Zoe", "depth_anything"],
    "ControlNetConfig.seg_detector": ["Seg_OFADE20K", "Seg_OFCOCO", "Seg_UFADE20K", "Seg_SAM"],
    "ControlNetConfig.scribble_detector": ["PIDI", "HED"],
    "ControlNetConfig.softedge_detector": ["PIDI", "HED"],
    "ControlNetConfig.temporalnet_source": ["init", "stylized"],
    "ControlNetConfig.inpaint_mask_source": ["consistency_mask", "none"],
    "VideoAssemblyConfig.blend_mode": ["None", "linear", "optical flow"],
    "VideoAssemblyConfig.upscale_ratio": [1, 2, 4],
    "VideoAssemblyConfig.upscale_model": ["realesr-animevideov3", "realesr-general-x4v3", "RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B", "RealESRGAN_x2plus"],
    "VideoAssemblyConfig.background_video": ["color", "image", "init_video"],
    "SceneConfig.diff_function": ["rmse", "lpips", "rmse+lpips"],
    "CaptionConfig.keyframe_source": ["Every n-th frame", "User-defined keyframe list", "Content-aware scheduling keyframes"],
    "CaptionConfig.offset_mode": ["Fixed", "Between Keyframes", "None"],
    "BackgroundMaskConfig.background": ["color", "image", "init_video"],
    "ReconstructionNoiseConfig.source": ["init", "stylized"],
    "IPAdapterEntry.weight_type": ["linear", "ease in", "ease out", "ease in-out", "reverse in-out", "weak input", "weak middle", "weak output", "strong middle", "style transfer", "composition", "strong style transfer", "style and composition", "strong style and composition", "style transfer precise"],
    "IPAdapterEntry.combine_embeds": [
        "concat", "add", "subtract", "average", "norm average"],
    "IPAdapterEntry.embeds_scaling": ["V only", "K+V", "K+V w/ C penalty", "K+mean(V) w/ C penalty"],
}


class ConfigError(ValueError):
    """A configuration payload could not be converted to RunConfig."""


def _convert(value: Any, annotation: Any, path: str) -> Any:
    if annotation is Any:
        return value
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        candidates = [arg for arg in args if arg is not type(None)]
        return _convert(value, candidates[0], path) if candidates else value
    if origin in (list, List):
        if not isinstance(value, list):
            raise ConfigError(f"{path} must be a list")
        item_type = args[0] if args else Any
        return [_convert(item, item_type, f"{path}[{index}]") for index, item in enumerate(value)]
    if origin in (dict, Dict):
        if not isinstance(value, dict):
            raise ConfigError(f"{path} must be an object")
        key_type, item_type = args or (Any, Any)
        converted = {}
        for key, item in value.items():
            new_key = int(key) if key_type is int else key
            converted[new_key] = _convert(item, item_type, f"{path}.{key}")
        return converted
    if isinstance(annotation, type) and is_dataclass(annotation):
        return config_from_dict(value, annotation, path)
    if annotation in (str, int, float, bool):
        if not isinstance(value, annotation):
            # JSON numbers commonly arrive as int where a float is expected.
            if annotation is float and isinstance(value, int) and not isinstance(value, bool):
                return float(value)
            raise ConfigError(f"{path} must be {annotation.__name__}")
    return value


def config_from_dict(data: Any, cls: type = RunConfig, path: str = "config") -> Any:
    """Construct a dataclass recursively, rejecting unknown fields."""
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must be an object")
    if cls is ColorConfig:
        data = dict(data)
        has_new_surface = any(key in data for key in (
            'before_enabled', 'before_strength', 'before_method',
            'before_regrain', 'after_enabled', 'after_strength',
            'after_method', 'after_regrain',
        ))
        if not has_new_surface:
            strength = data.get('match_color_strength', 0.0)
            method = data.get('colormatch_method', 'PDF')
            regrain = bool(data.get('colormatch_regrain', False))
            mode = data.get('colormatch_mode')
            if mode not in ('before', 'after', 'both'):
                mode = 'after' if data.get('colormatch_after', True) else 'before'
            enabled = isinstance(strength, (int, float)) and strength > 0
            migrated_strength = strength if enabled else 0.5
            data.update({
                'before_enabled': enabled and mode in ('before', 'both'),
                'before_strength': migrated_strength,
                'before_method': method,
                'before_regrain': regrain,
                'after_enabled': enabled and mode in ('after', 'both'),
                'after_strength': migrated_strength,
                'after_method': method,
                'after_regrain': regrain,
            })
    if cls is FluxConfig and 'feed_init_as_context' in data:
        data = dict(data)
        feed_raw = data.pop('feed_init_as_context')
        if not isinstance(feed_raw, bool):
            raise ConfigError(f"{path}.feed_init_as_context must be bool")
        data.setdefault('reference_mode', 'raw + warped' if feed_raw else 'warped only')
    if cls is HiDreamConfig and 'feed_raw_frame_as_reference' in data:
        # Migrate the short-lived boolean surface from the first multi-reference
        # implementation. Explicit new selectors take precedence.
        data = dict(data)
        feed_raw = data.pop('feed_raw_frame_as_reference')
        if not isinstance(feed_raw, bool):
            raise ConfigError(
                f"{path}.feed_raw_frame_as_reference must be bool")
        data.setdefault('reference_mode', 'raw + warped' if feed_raw else 'warped only')
    if cls in (FluxConfig, HiDreamConfig, QwenEditConfig, MageFlowEditConfig):
        # Migrate the two previous experimental reference surfaces into the
        # unified ordered list. New ``references`` always wins when both exist.
        data = dict(data)
        mode = data.pop('reference_mode', 'raw + warped')
        unwarped = data.pop('use_unwarped_style_reference', False)
        label_raw = data.pop('label_raw_reference', False)
        label_style = data.pop('label_style_reference', False)
        feed_prev = data.pop('feed_prev_frame_as_context', False)
        data.pop('feed_consistency_mask_as_context', None)
        if 'references' not in data:
            if mode == 'raw only':
                refs = [{'source': 'raw', 'label': label_raw}]
            elif mode == 'warped only':
                refs = [{'source': 'warped', 'label': label_style}]
            else:
                refs = [
                    {'source': 'raw', 'label': label_raw},
                    {'source': 'previous' if unwarped else 'warped',
                     'label': label_style},
                ]
            if cls is FluxConfig and feed_prev and not unwarped:
                refs.append({'source': 'previous', 'label': False})
            refs.append({'source': 'none', 'label': False})
            data['references'] = refs
    known = {field.name: field for field in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(f"{path} has unknown field(s): {', '.join(unknown)}")
    hints = get_type_hints(cls)
    kwargs = {
        key: _convert(value, hints.get(key, Any), f"{path}.{key}")
        for key, value in data.items()
    }
    try:
        result = cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"Invalid {path}: {exc}") from exc
    if cls is RunConfig and result.model_version in FLUX_MODEL_DEFAULTS:
        for key, value in flux_model_files(result).items():
            setattr(result.flux, key, value)
    if cls is RunConfig and result.model_version in QWEN_MODEL_DEFAULTS:
        for key, value in qwen_model_files(result).items():
            setattr(result.qwen, key, value)
    if cls is RunConfig and result.model_version in MAGE_MODEL_DEFAULTS:
        for key, value in mage_model_defaults(result).items():
            setattr(result.mage, key, value)
    return result


def config_from_json(path: str | Path) -> RunConfig:
    with open(path, encoding="utf-8") as handle:
        return config_from_dict(json.load(handle))


def config_from_settings(path: str | Path, models_root: str | None = None) -> RunConfig:
    """Load nested JSON, VibeWarp saved settings, or WarpFusion settings."""
    from vibewarp.settings import is_vibewarp_settings, load_settings, load_vibewarp_settings, load_warpfusion_settings
    raw = load_settings(str(path))
    if any(isinstance(raw.get(key), dict) for key in ("diffusion", "video", "flow")):
        data = raw
    elif is_vibewarp_settings(raw):
        data = load_vibewarp_settings(str(path))
    else:
        data = load_warpfusion_settings(str(path), models_root=models_root)
    return config_from_dict(data)


def config_to_dict(config: RunConfig) -> dict:
    return asdict(config)


# Fields that may hold a filesystem path but aren't named like one, so the
# name heuristic below can't find them. Each also accepts non-path values
# ('init', 'stylized', a colour name) — stripping quotes is harmless for those.
_EXTRA_PATH_FIELDS = frozenset({
    "ControlNetEntry.source",
    "IPAdapterEntry.source_image",
    "BackgroundMaskConfig.background_source",
    "VideoAssemblyConfig.background_source_video",
})
_PATH_NAME_HINTS = ("path", "dir", "folder")
_QUOTES = ("\"", "'")


def _is_path_field(cls_name: str, field_name: str) -> bool:
    if f"{cls_name}.{field_name}" in _EXTRA_PATH_FIELDS:
        return True
    return any(hint in field_name for hint in _PATH_NAME_HINTS)


def strip_path_quotes(value: str) -> str:
    """Remove surrounding quotes and whitespace from a pasted path.

    Windows Explorer's "Copy as path" yields `"C:\\models\\ControlNet"`, quotes
    included. Pasting that verbatim makes every downstream os.path call miss.
    """
    cleaned = value.strip()
    while len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in _QUOTES:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def sanitize_paths(config: Any) -> Any:
    """Strip pasted quotes from every path-bearing string in the config tree."""
    for item in fields(config):
        value = getattr(config, item.name)
        if isinstance(value, str):
            if _is_path_field(type(config).__name__, item.name):
                setattr(config, item.name, strip_path_quotes(value))
        elif is_dataclass(value):
            sanitize_paths(value)
        elif isinstance(value, dict):
            for entry in value.values():
                if is_dataclass(entry):
                    sanitize_paths(entry)
        elif isinstance(value, list):
            for entry in value:
                if is_dataclass(entry):
                    sanitize_paths(entry)
    return config


def apply_path_defaults(config: RunConfig) -> RunConfig:
    """Apply defaults for blank path fields that have writable fallbacks."""
    # Before the defaults: a value that is nothing but quotes must count as blank.
    sanitize_paths(config)
    if not config.root_dir.strip():
        config.root_dir = os.getcwd()
    if not config.output_dir.strip():
        config.output_dir = "images_out"
    return config


def validate_config(
    config: RunConfig, *, require_inputs: bool = True, check_paths: bool = False
) -> List[dict]:
    """Return frontend-friendly validation errors."""
    errors: List[dict] = []

    def error(path: str, message: str) -> None:
        errors.append({"path": path, "message": message})

    from vibewarp.ui_layout import model_family_for_version
    uses_sd_stack = model_family_for_version(config.model_version) == 'sd'

    if require_inputs and uses_sd_stack and not config.sd_checkpoint_path:
        error("sd_checkpoint_path", "A Stable Diffusion checkpoint is required")
    if require_inputs and not config.video.video_init_path:
        error("video.video_init_path", "An input video is required")
    if (check_paths and uses_sd_stack and config.sd_checkpoint_path
            and not os.path.isfile(config.sd_checkpoint_path)):
        error("sd_checkpoint_path", f"Checkpoint file does not exist: {config.sd_checkpoint_path}")
    if check_paths and config.video.video_init_path and not os.path.isfile(config.video.video_init_path):
        error("video.video_init_path", f"Input video does not exist: {config.video.video_init_path}")
    input_directories = {"root_dir": config.root_dir}
    if uses_sd_stack:
        input_directories.update({
            "lora_dir": config.lora_dir,
            "controlnet.model_dir": config.controlnet.model_dir,
        })
    if check_paths:
        for path, directory in input_directories.items():
            if directory and not os.path.isdir(directory):
                error(path, f"Directory does not exist: {directory}")
        if (uses_sd_stack and config.model_path
                and not os.path.exists(config.model_path)):
            error("model_path", f"Model path does not exist: {config.model_path}")
    # A ControlNet only fits the base model it was trained for — an SD1.5 net on an
    # SDXL model dies at load with a shape mismatch. Easy to hit by switching
    # model_version in the UI with nets already configured, so catch it up front.
    if uses_sd_stack and config.controlnet.enabled and config.controlnet.models:
        family = normalize_model_version(config.model_version)
        if family:
            for key in config.controlnet.models:
                spec = CONTROLNET_CATALOG.get(key)
                if spec and spec.model_version != family:
                    error(f"controlnet.models.{key}",
                          f"{key} is a {spec.model_version} ControlNet, but the model "
                          f"is {family} — remove it or switch the model version")

    if uses_sd_stack and config.ipadapter.enabled:
        if not config.ipadapter.models:
            error(
                "ipadapter.models",
                "IP-Adapter is enabled but no adapter models are configured")
        allowed_ip_sources = {'none', 'off', 'previous', 'warped', 'upload', 'fixed'}
        for key, entry in config.ipadapter.models.items():
            model_key = getattr(entry, 'model_key', '') or key
            spec = IPADAPTER_CATALOG.get(model_key)
            family = normalize_model_version(config.model_version)
            if spec and family and spec.model_version != family:
                error(
                    f"ipadapter.models.{key}",
                    f"{model_key} is a {spec.model_version} IP-Adapter, but the model "
                    f"is {family} — remove it or switch the model version")
            if check_paths and entry.weight != 0:
                checkpoint = resolve_ipadapter_path(
                    model_key, entry.path, config.controlnet.model_dir)
                if not checkpoint:
                    error(
                        f"ipadapter.models.{key}.path",
                        f"{key}: no IP-Adapter checkpoint is configured")
                elif not os.path.isfile(checkpoint):
                    error(
                        f"ipadapter.models.{key}.path",
                        f"{key}: IP-Adapter checkpoint does not exist: "
                        f"{checkpoint}")
            references = list(getattr(entry, 'source_images', None) or
                              [entry.source_image])
            for ref_index, reference in enumerate(references):
                item_path = f"ipadapter.models.{key}.source_images.{ref_index}"
                if isinstance(reference, dict) and 'source' not in reference:
                    # Legacy frame-keyed path schedule.
                    if check_paths:
                        for image_path in reference.values():
                            if image_path and not os.path.isfile(image_path):
                                error(
                                    item_path,
                                    f"scheduled reference image does not "
                                    f"exist: {image_path}")
                    continue
                if isinstance(reference, dict) and 'source' in reference:
                    source = reference.get('source', 'none')
                    image_path = reference.get('image_path', '')
                else:
                    source = reference
                    image_path = (reference if isinstance(reference, str)
                                  and reference not in (
                                      '', 'none', 'off', 'previous', 'prev_frame',
                                      'stylized', 'warped', 'raw_frame', 'init')
                                  else '')
                    source = {
                        '': 'none', 'off': 'none', 'prev_frame': 'previous',
                        'stylized': 'warped',
                    }.get(source, source)
                if source in ('raw_frame', 'init'):
                    error(
                        item_path,
                        f"{key}: raw-frame input is not supported; choose Off, a "
                        "previous stylized source, or a fixed image")
                elif source not in allowed_ip_sources and not image_path:
                    error(item_path, f"{key}: unknown image source {source!r}")
                if source in ('upload', 'fixed'):
                    if not image_path:
                        error(item_path, f"{key}: choose a fixed reference image")
                    elif check_paths and not os.path.isfile(image_path):
                        error(
                            item_path,
                            f"{key}: fixed reference image does not exist: {image_path}")

    # Without a motion module, `enabled` still switches the render into AnimateDiff
    # batch mode but no temporal attention is injected — you pay the batching cost
    # and the output looks unchanged. Fail instead of quietly doing nothing.
    if (uses_sd_stack and config.animatediff.enabled
            and not config.animatediff.motion_module_path):
        error("animatediff.motion_module_path",
              "AnimateDiff is enabled but no motion module is set")
    if (check_paths and uses_sd_stack and config.animatediff.enabled
            and config.animatediff.motion_module_path
            and not os.path.isfile(config.animatediff.motion_module_path)):
        error("animatediff.motion_module_path",
              f"Motion module does not exist: {config.animatediff.motion_module_path}")
    if config.video.width <= 0 or config.video.width % 8:
        error("video.width", "Width must be positive and divisible by 8")
    if config.video.height <= 0 or config.video.height % 8:
        error("video.height", "Height must be positive and divisible by 8")
    if len(config.frame_range) != 2 or config.frame_range[0] < 0 or config.frame_range[1] < 0:
        error("frame_range", "Frame range must contain two non-negative frame numbers")
    elif config.frame_range[1] and config.frame_range[1] < config.frame_range[0]:
        error("frame_range", "End frame must be zero (auto) or at least the start frame")
    if uses_sd_stack:
        if config.diffusion.steps < 1:
            error("diffusion.steps", "Diffusion steps must be at least 1")
        if not 0 <= config.diffusion.style_strength <= 1:
            error("diffusion.style_strength", "Style strength must be between 0 and 1")
        if config.diffusion.guidance_mode not in FIELD_CHOICES["DiffusionConfig.guidance_mode"]:
            error("diffusion.guidance_mode", "Unknown SD/SDXL guidance mode")
        if (config.diffusion.sampler_tile_size < 8
                or config.diffusion.sampler_tile_size % 8):
            error("diffusion.sampler_tile_size",
                  "Sampler tile size must be at least 8 pixels and divisible by 8")
        if not 0 <= config.diffusion.sampler_tile_overlap <= 50:
            error("diffusion.sampler_tile_overlap",
                  "Sampler tile overlap must be between 0 and 50 percent")
    if config.video.extract_nth_frame < 1:
        error("video.extract_nth_frame", "Frame extraction interval must be at least 1")
    if config.video.max_size < 0:
        error("video.max_size", "Max size cannot be negative")
    for phase in ('before', 'after'):
        strength = getattr(config.color, f'{phase}_strength')
        if not 0 <= strength <= 1:
            error(
                f"color.{phase}_strength",
                f"{phase.title()} color-match strength must be between 0 and 1",
            )
    family = model_family_for_version(config.model_version)
    if family in ('flux', 'hidream', 'qwen'):
        if not 0 <= config.reference_label_opacity <= 1:
            error("reference_label_opacity",
                  "Reference label opacity must be between 0 and 1")
        refs = getattr(config, family).references
        path = f'{family}.references'
        allowed = {'raw', 'previous', 'warped', 'upload', 'none'}
        if not refs:
            error(path, "At least one reference image is required")
        elif refs[0].source in ('none', 'upload'):
            error(path, "Reference image 1 must use a video-frame source")
        from vibewarp.core.edit_references import MAX_EDIT_REFERENCES
        max_references = MAX_EDIT_REFERENCES[family]
        if len(refs) > max_references:
            error(path, f"This model supports at most {max_references} reference images")
        saw_none = False
        for index, ref in enumerate(refs):
            item_path = f'{path}[{index}]'
            if ref.source not in allowed:
                error(item_path, f"Unknown reference source: {ref.source}")
            if index > 0 and ref.source == 'none':
                saw_none = True
            elif saw_none and ref.source != 'none':
                error(item_path, "References after the first 'none' slot are not sent")
            if ref.source == 'upload':
                if not ref.image_path:
                    error(item_path, "Choose an uploaded reference image")
                elif check_paths and not os.path.isfile(ref.image_path):
                    error(item_path, f"Uploaded image does not exist: {ref.image_path}")
        if family == 'qwen':
            if config.qwen.steps < 1:
                error("qwen.steps", "Qwen steps must be at least 1")
            if config.qwen.guidance_scale < 0:
                error("qwen.guidance_scale", "Qwen guidance cannot be negative")
            if config.qwen.lora_strength < 0:
                error("qwen.lora_strength", "Qwen LoRA strength cannot be negative")
    return errors


def _type_schema(annotation: Any) -> dict:
    if annotation is Any:
        return {"type": "json"}
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, types.UnionType):
        nullable = type(None) in args
        candidates = [arg for arg in args if arg is not type(None)]
        schema = _type_schema(candidates[0]) if candidates else {"type": "json"}
        schema["nullable"] = nullable
        return schema
    if origin in (list, List):
        return {"type": "array", "items": _type_schema(args[0] if args else Any)}
    if origin in (dict, Dict):
        return {"type": "object", "additional": _type_schema(args[1] if args else Any)}
    if isinstance(annotation, type) and is_dataclass(annotation):
        return _dataclass_schema(annotation)
    return {"type": {str: "string", int: "integer", float: "number", bool: "boolean"}.get(annotation, "json")}


def _dataclass_schema(cls: type, section: str | None = None) -> dict:
    from vibewarp.ui_layout import (ALL_MODEL_FAMILIES, classify, label,
                                    model_families)

    hints = get_type_hints(cls)
    properties = {}
    for item in fields(cls):
        annotation = hints.get(item.name, Any)
        if isinstance(annotation, type) and is_dataclass(annotation):
            # A nested section: its fields are classified under the section's own name.
            schema = _dataclass_schema(annotation, section=item.name)
        else:
            schema = _type_schema(annotation)
        choices = FIELD_CHOICES.get(f"{cls.__name__}.{item.name}")
        if choices:
            schema["choices"] = choices
        if item.default is not MISSING:
            schema["default"] = item.default
        elif item.default_factory is not MISSING:
            schema["default"] = item.default_factory()
        # Which UI tab this field belongs to. The nav is built from this, so an
        # unclassified field renders nowhere (see ui_layout.unclassified()).
        if section is not None:
            placed = classify(section, item.name)
            if placed:
                schema["tier"], schema["group"] = placed
            compatible = model_families(section, item.name)
            if compatible != ALL_MODEL_FAMILIES:
                schema["model_families"] = list(compatible)
            # A hoisted field needs a name that means something away from its section:
            # animatediff.enabled on the Model card would otherwise just say "Enabled".
            display = label(section, item.name)
            if display:
                schema["label"] = display
        properties[item.name] = schema
    return {"type": "dataclass", "title": cls.__name__, "description": cls.__doc__ or "", "properties": properties}


def config_schema() -> dict:
    """Return a dependency-free schema used to render the web form.

    Top-level RunConfig fields are classified under the section name "main".
    """
    schema = _dataclass_schema(RunConfig, section="main")
    from vibewarp.ui_layout import (GROUP_ORDER, MODEL_FAMILY_BY_VERSION,
                                    TIER_LABELS, TIERS)
    # `groups` is the DECLARED order (ui_layout), not the order fields happen to appear on
    # RunConfig -- otherwise "Input Video" would sort behind "Output" and "Model".
    schema["tiers"] = [{"id": tier, "label": TIER_LABELS[tier],
                        "groups": list(GROUP_ORDER.get(tier, []))} for tier in TIERS]
    schema["model_family_by_version"] = dict(MODEL_FAMILY_BY_VERSION)
    schema["default_model_family"] = "sd"
    from vibewarp.core.edit_references import (
        MAX_EDIT_REFERENCES, REFERENCE_SOURCE_LABELS)
    schema["max_edit_references"] = dict(MAX_EDIT_REFERENCES)
    schema["edit_reference_sources"] = dict(REFERENCE_SOURCE_LABELS)
    schema["flux_model_defaults"] = {
        version: dict(defaults)
        for version, defaults in FLUX_MODEL_DEFAULTS.items()
    }
    schema["qwen_model_defaults"] = {
        version: dict(defaults)
        for version, defaults in QWEN_MODEL_DEFAULTS.items()
    }
    schema["mage_model_defaults"] = {
        version: dict(defaults)
        for version, defaults in MAGE_MODEL_DEFAULTS.items()
    }
    return schema
