"""Shared RunConfig serialization, construction, schema, and validation."""

from __future__ import annotations

import json
import math
import os
import types
from dataclasses import MISSING, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Union, get_args, get_origin, get_type_hints

from vibewarp.config import (
    FLUX_MODEL_DEFAULTS, MAGE_MODEL_DEFAULTS, QWEN_MODEL_DEFAULTS, ColorConfig,
    ContactSheetConfig, EditReferenceConfig, FluxConfig, HiDreamConfig, MageFlowEditConfig,
    QwenEditConfig, RunConfig, DiscoConfig, VideoConfig,
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
    "BackgroundMaskConfig.render_mask_stage": ['before diffusion', 'after diffusion'],
    "BackgroundMaskConfig.mask_source": ["RVM matting", "mask video", "image folder"],
    "BackgroundMaskConfig.raw_color_method": ["PDF", "LAB", "mean"],
    "BackgroundMaskConfig.raw_color_reference": ["init stylized", "previous warped"],
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
        "disco_512", "disco_256",
    ],
    "FluxConfig.backend": ["comfy", "diffusers"],
    "DiscoConfig.sampler": ["ddim", "plms", "distilled"],
    "DiscoConfig.attention": ["sdpa", "manual", "flash", "sage"],
    "DiscoConfig.secondary_precision": ["auto"],
    "DiscoConfig.clip_precision": ["fp16"],
    "HiDreamConfig.sampler": [
        "dpmpp_2m_sde_gpu", "dpmpp_2m_sde", "dpmpp_2m", "euler", "heun",
    ],
    "HiDreamConfig.scheduler": ["simple", "normal", "sgm_uniform"],
    "HiDreamConfig.patch_seam_passes": ["2", "4", "ramp_2_4", "ramp_2_4_8"],
    "HiDreamConfig.patch_seam_blend": ["median", "average", "window"],
    "QwenEditConfig.sampler": ["euler", "dpmpp_2m", "dpmpp_2m_sde"],
    "QwenEditConfig.scheduler": ["simple", "normal", "sgm_uniform"],
    "ContactSheetConfig.mode": ["off", "raw_triplets", "reinject"],
    "ContactSheetConfig.layout": ["adaptive", "row", "column", "grid"],
    "MageFlowEditConfig.sampler": ["euler", "dpmpp_2m", "dpmpp_2m_sde"],
    "MageFlowEditConfig.scheduler": ["simple", "normal", "sgm_uniform"],
    "RunConfig.lora_merge_precision": ["fp16", "fp32"],
    "RunConfig.animation_mode": ["Video Input"],
    "WarpConfig.warp_mode": ["use_image"],
    "WarpConfig.warp_towards_init": ["off", "init"],
    "WarpConfig.padding_mode": ["reflect", "edge", "symmetric", "wrap", "constant"],
    "DiffusionConfig.sampler": ["sample_euler_ancestral", "sample_euler", "sample_dpm_2", "sample_dpm_2_ancestral", "sample_heun", "sample_lms", "sample_dpmpp_2m", "sample_dpmpp_2s_ancestral", "sample_dpmpp_sde", "sample_lcm"],
    "DiffusionConfig.unet_cache": ["off", "deepcache", "first_block"],
    "DiffusionConfig.sampler_tile_size": [256, 512, 768, 1024],
    "DiffusionConfig.noise_mode": ["default", "fixed", "reconstructed"],
    "DiffusionConfig.guidance_mode": [
        "prev stylized", "prev warped", "prev warped + cc", "raw frame"],
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
    "IPAdapterEntry.weight_type": ["linear", "ease in", "ease out", "ease in-out", "reverse in-out", "weak input", "weak middle", "weak output", "strong middle", "style transfer", "composition", "strong style transfer", "style and composition", "strong style and composition", "style transfer precise", "composition precise"],
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
            try:
                new_key = int(key) if key_type is int else float(key) if key_type is float else key
            except (ValueError, TypeError) as exc:
                raise ConfigError(f'{path} requires numeric percentage keys') from exc
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
    if cls is VideoConfig:
        data=dict(data)
        data.pop('first_frame_sha256',None)
    disco_input = data.get('disco', {}) if cls is RunConfig else {}
    legacy_cutouts = (cls is RunConfig and 'disco' in data and isinstance(disco_input, dict)
        and type(disco_input.get('cut_schedule_version', 1)) is int
        and disco_input.get('cut_schedule_version', 1) == 1)
    legacy_original_time = bool(disco_input.get('cut_schedule_original_time', False)) if isinstance(disco_input, dict) else False
    if cls is RunConfig and isinstance(data.get('video'), dict) and 'max_frames' in data['video']:
        from vibewarp.frame_range import imported_frame_range
        data = dict(data)
        data['video'] = dict(data['video'])
        count = data['video'].pop('max_frames')
        data['frame_range'] = imported_frame_range(data.get('frame_range'), count)
    if cls is DiscoConfig:
        data = dict(data)
        data.pop('cut_schedule_original_time', None)
        # LoRAs are safetensors (no pickle), so their checksums are no longer
        # pinned; older saved settings still carry the digests.
        data.pop('few_step_lora_sha256', None)
        data.pop('quality_lora_sha256', None)
        # Older native settings exposed FP32/auto. Encoder loading is now FP16
        # for every CLIP/SigLIP model, including restored settings.
        data['clip_precision'] = 'fp16'
        if data.get('secondary_precision','auto') in ('fp32','fp16','bf16','auto'):
            data['secondary_precision'] = 'auto'
        # Migrate constant/list values emitted by older schedule editors to
        # the progress-keyed dictionary used by the cutout schedules.
        gray = data.get('cut_icgray_schedule')
        if isinstance(gray, list):
            data['cut_icgray_schedule'] = dict(enumerate(gray))
        elif isinstance(gray, (int, float)) and not isinstance(gray, bool):
            data['cut_icgray_schedule'] = {0: gray}
        for name in ('init_scale_schedule','image_scale_schedule'):
            if isinstance(data.get(name),dict):
                try:
                    data[name] = {int(k):v for k,v in data[name].items()}
                except (ValueError,TypeError) as exc:
                    raise ConfigError(f'{path}.{name} requires integer schedule keys') from exc
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
    if legacy_cutouts:
        from vibewarp.disco_cutouts import migrate_cutout_schedules
        try:
            migrate_cutout_schedules(result, legacy_original_time)
        except (ValueError, TypeError, IndexError, KeyError) as exc:
            raise ConfigError(f'Cannot migrate Disco cutout schedules: {exc}') from exc
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


def settings_as_dict(path: str | Path, models_root: str | None = None) -> dict:
    """The nested settings a file actually specifies, BEFORE defaults fill in.

    Comparing two runs needs this: once defaults are applied, a setting a run
    never saved is indistinguishable from one saved at its default value, so a
    run that predates a feature looks like it opted out of it.
    """
    from vibewarp.settings import is_vibewarp_settings, load_settings, load_vibewarp_settings, load_warpfusion_settings
    raw = load_settings(str(path))
    if any(isinstance(raw.get(key), dict) for key in ("diffusion", "video", "flow")):
        return raw
    if is_vibewarp_settings(raw):
        return load_vibewarp_settings(str(path))
    return load_warpfusion_settings(str(path), models_root=models_root)


def config_from_settings(path: str | Path, models_root: str | None = None) -> RunConfig:
    """Load nested JSON, VibeWarp saved settings, or WarpFusion settings."""
    return config_from_dict(settings_as_dict(path, models_root=models_root))


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
    config.disco.clip_precision = 'fp16'
    if config.disco.secondary_precision in ('fp32','fp16','bf16','auto'):
        config.disco.secondary_precision = 'auto'
    if not config.root_dir.strip():
        config.root_dir = os.getcwd()
    if not config.output_dir.strip():
        config.output_dir = "images_out"
    if config.model_version.startswith('disco_') and (config.disco.sampler != 'distilled' or config.disco.distilled_guided):
        # Old distilled profiles saved a fixed square with max_size=0. Guided
        # video renders use source aspect, retaining that square's longest edge
        # only as the size budget (the same default the UI uses for SD).
        if (config.disco.pad_to_multiple_64 and config.video.max_size == 0
                and not config.video.visible_width and not config.video.visible_height):
            config.video.visible_width, config.video.visible_height = config.video.width, config.video.height
        if config.video.max_size == 0 and not config.disco.pad_to_multiple_64:
            config.video.max_size=max(config.video.width,config.video.height)
        for axis in ('width','height'):
            value=getattr(config.video,axis)
            if value > 0:
                setattr(config.video,axis,((value+63)//64)*64)
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
    if not math.isfinite(config.diffusion.fixed_seed_randomness) or not 0 <= config.diffusion.fixed_seed_randomness <= 1:
        error('diffusion.fixed_seed_randomness','Choose a finite randomness strength between 0 and 1')
    if type(config.diffusion.fixed_seed_wiggle) is not int or config.diffusion.fixed_seed_wiggle < 0:
        error('diffusion.fixed_seed_wiggle','Choose a nonnegative integer pixel distance')
    direction=config.diffusion.fixed_seed_wiggle_direction
    if (not isinstance(direction,list) or len(direction)!=2
            or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in direction)):
        error('diffusion.fixed_seed_wiggle_direction','Choose two finite direction values [x, y]')
    m = config.mask
    for name in ('foreground_opacity','background_opacity','raw_color_strength'):
        value = getattr(m,name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            error('mask.'+name,'Choose a finite opacity/strength between 0 and 1')
    if not math.isfinite(m.rvm_downsample_ratio) or not 0 < m.rvm_downsample_ratio <= 1:
        error('mask.rvm_downsample_ratio','RVM downsample ratio must be greater than 0 and at most 1')
    for name in ('foreground_color','background_color'):
        value = getattr(m,name)
        if not isinstance(value,list) or len(value)!=3 or any(type(c) is not int or not 0<=c<=255 for c in value):
            error('mask.'+name,'Choose three RGB integers between 0 and 255')
    if m.raw_color_method not in ('PDF','LAB','mean'):
        error('mask.raw_color_method','Choose PDF, LAB or mean')
    if m.raw_color_reference not in ('init stylized','previous warped'):
        error('mask.raw_color_reference','Choose init stylized or previous warped')
    if m.render_mask_stage not in ('before diffusion', 'after diffusion'):
        error('mask.render_mask_stage', 'Choose before diffusion or after diffusion')
    from vibewarp.video import masks as _masks
    if m.mask_source not in _masks.MASK_SOURCES:
        error('mask.mask_source', 'Choose RVM matting, mask video or image folder')
    elif m.mask_source != _masks.RVM and _masks.wanted(config):
        if not m.mask_path:
            error('mask.mask_path', f'Choose the {m.mask_source} that holds the background masks')
        elif check_paths and m.mask_source == _masks.VIDEO and not os.path.isfile(m.mask_path):
            error('mask.mask_path', f'Mask video does not exist: {m.mask_path}')
        elif check_paths and m.mask_source == _masks.FOLDER and not os.path.isdir(m.mask_path):
            error('mask.mask_path', f'Mask folder does not exist: {m.mask_path}')
    if (m.rvm_enabled or m.use_background_mask or m.raw_color_enabled) and require_inputs and not config.video.video_init_path:
        error('video.video_init_path','Raw preparation requires an input video')
    if config.diffusion.temporal_guidance_cc_trusted_only and config.diffusion.guidance_mode == 'prev stylized':
        error('diffusion.guidance_mode', 'CC trust requires a warped temporal guidance target')
    if config.video_assembly.fps < 0 or not math.isfinite(config.video_assembly.fps):
        error('video_assembly.fps', 'FPS must be finite and nonnegative')
    if check_paths and config.video.first_frame_path and not Path(config.video.first_frame_path).is_file():
        error('video.first_frame_path', 'First-frame reference does not exist')

    if config.model_version.startswith('disco_'):
        d = config.disco
        if type(d.cut_max_size) is not int or (d.cut_max_size != 0 and d.cut_max_size < 32):
            error('disco.cut_max_size','Use 0 for native size, or an integer of at least 32 pixels')
        distilled = d.sampler == 'distilled'
        if d.secondary_precision != 'auto':
            error('disco.secondary_precision', 'Secondary model precision is automatic: BF16 or FP16')
        if d.clip_precision not in FIELD_CHOICES['DiscoConfig.clip_precision']:
            error('disco.clip_precision', 'CLIP and SigLIP guidance encoders require FP16')
        if type(d.cut_batch_group) is not int or d.cut_batch_group < 1:
            error('disco.cut_batch_group', 'CLIP cut batch group must be a positive integer')
        if not 0 <= d.eta <= 1:
            error('disco.eta', 'DDIM eta must be between zero and one')
        if d.clip_grad_steps < 1 or (d.clip_divide_cuts and d.clip_grad_steps > d.cutn_batches):
            error('disco.clip_grad_steps', 'Use positive passes and at least one cut batch per split pass')
        if d.clip_grad_steps > 1 and d.sampler not in ('ddim', 'distilled'):
            error('disco.clip_grad_steps', 'Repeated guidance currently requires DDIM')
        if d.timestep_grid:
            from vibewarp.core.disco_repeated import validate_grid
            try:
                validate_grid(d.timestep_grid, d.steps)
            except ValueError as exc:
                error('disco.timestep_grid', str(exc))
            if distilled and not d.distilled_guided:
                error('disco.timestep_grid', 'Unconditional distilled mode uses the trained grid')
        if d.inpaint_mask_sigma < 0:
            error('disco.inpaint_mask_sigma', 'Blur sigma cannot be negative')
        if not math.isfinite(d.lora_strength) or d.lora_strength < 0:
            error('disco.lora_strength', 'LoRA strength must be finite and nonnegative')
        if not math.isfinite(d.quality_lora_strength) or d.quality_lora_strength < 0:
            error('disco.quality_lora_strength', 'Quality LoRA strength must be finite and nonnegative')
        if d.quality_lora_enabled:
            if config.model_version != 'disco_512':
                error('model_version', 'Quality LoRA requires the Crowson Disco 512 base')
            if not d.quality_lora_path.strip():
                error('disco.quality_lora_path', 'Choose the quality adapter .safetensors file')
            elif check_paths and not Path(d.quality_lora_path).is_file():
                error('disco.quality_lora_path', 'Quality adapter file does not exist')
            if d.sampler not in ('ddim','distilled'):
                error('disco.sampler', 'Quality LoRA requires DDIM or guided distilled sampling')
            if distilled and not d.distilled_guided:
                error('disco.distilled_guided', 'Quality LoRA requires the guided sampling path')
            if config.animatediff.enabled or config.contact_sheet.mode != 'off':
                error('disco.quality_lora_enabled', 'Quality LoRA uses sequential Disco frames without AnimateDiff or contact sheets')
        if distilled:
            if config.model_version != 'disco_512':
                error('model_version', 'The distilled adapter requires Disco 512')
            if not d.few_step_lora_path.strip():
                error('disco.few_step_lora_path', 'Choose the few-step adapter .safetensors file or its run directory')
            if check_paths and d.few_step_lora_path and not Path(d.few_step_lora_path).exists():
                error('disco.few_step_lora_path', 'Adapter path does not exist')
        if distilled and not d.distilled_guided:
            if (config.video.width, config.video.height) != (512,512) or config.video.max_size:
                error('video.width', 'The distilled adapter currently requires fixed 512x512 output')
            if d.steps not in (4,6):
                error('disco.steps', 'Use the adapter trained step count: four or six')
            for field in ('clip_guidance_scale','init_scale','image_scale','tv_scale','range_scale','sat_scale','eta','guidance_noise_cap_percent'):
                if getattr(d,field) != 0:
                    error(f'disco.{field}', 'The validated distilled mode is unconditional, without guidance or stochastic sampling')
            for field in ('use_secondary_model','inpaint'):
                if getattr(d,field):
                    error(f'disco.{field}', 'Disable this feature for the distilled reference mode')
            if d.style_strength != 1:
                error('disco.style_strength', 'Distilled sampling requires a full-noise start with no skipped steps')
            if d.encoders:
                error('disco.encoders', 'Distilled reference mode does not use guidance encoders')
            if any(str(value).strip() for value in d.image_prompts.values()):
                error('disco.image_prompts', 'Image prompting is not validated with the distilled adapter')
            if d.attention != 'sdpa':
                error('disco.attention', 'The distilled parity path uses SDPA')
            for field in ('text_prompts','negative_prompts'):
                if any(str(value).strip() for value in getattr(config,field).values()):
                    error(field, 'The distilled reference mode does not support text guidance')
            for section, field in (('flow','flow_warp'),('turbo','enabled'),('animatediff','enabled'),
                                   ('color','before_enabled'),('color','after_enabled'),('diffusion','tiled_sampler')):
                if getattr(getattr(config,section),field):
                    error(f'{section}.{field}', 'This feature is not validated with the distilled adapter')
            if config.contact_sheet.mode != 'off':
                error('contact_sheet.mode', 'Contact sheets are not supported in distilled mode')
            if any(value != 100 for value in config.diffusion.sampler_scale_schedule.values()):
                error('diffusion.sampler_scale_schedule', 'Distilled sampling uses the fixed trained resolution')
        if distilled and d.distilled_guided:
            if config.animatediff.enabled:
                error('animatediff.enabled', 'Disco does not support AnimateDiff')
            if config.contact_sheet.mode != 'off':
                error('contact_sheet.mode', 'Disco does not support contact-sheet batching')
        if config.diffusion.guidance_mode not in FIELD_CHOICES['DiffusionConfig.guidance_mode']:
            error('diffusion.guidance_mode', 'Unknown frame guidance source')
        for name in ('init_scale', 'image_scale'):
            if getattr(d, name) < 0:
                error(f'disco.{name}', 'Guidance scale cannot be negative')
        from vibewarp.disco_catalog import CLASSIC_ENCODERS, MODERN_ENCODERS
        if any(name not in (*CLASSIC_ENCODERS, *MODERN_ENCODERS) for name in d.encoders):
            error('disco.encoders', 'Unknown guidance encoder')
        if d.sampler not in FIELD_CHOICES['DiscoConfig.sampler']:
            error('disco.sampler', 'Choose DDIM, PLMS or the distilled adapter mode')
        if d.attention not in FIELD_CHOICES['DiscoConfig.attention']:
            error('disco.attention', 'Unknown attention backend')
        if d.preview_interval < 0 or d.inpainting_stop < 0:
            error('disco.preview_interval', 'Preview interval and inpainting stop cannot be negative')
        if not 0 <= d.inpaint_mask_threshold <= 1 or d.inpaint_mask_blur < 0:
            error('disco.inpaint_mask_threshold', 'Mask threshold must be between 0 and 1; blur cannot be negative')
        for name in ('steps', 'cutn_batches', 'guidance_every'):
            if getattr(d, name) < 1:
                error(f'disco.{name}', 'Must be at least 1')
        if d.steps < 2 or d.steps > 1000:
            error('disco.steps', 'Choose between 2 and 1000 steps')
        if not 0 < d.style_strength <= 1:
            error('disco.style_strength', 'Style strength must be greater than 0 and at most 1')
        for name, lower, upper in (('steps_schedule',2,1000),('style_strength_schedule',0,1)):
            schedule = getattr(d,name)
            if schedule is None:
                continue
            if distilled and not d.distilled_guided:
                error(f'disco.{name}', 'Frame schedules are not supported by the distilled reference mode')
            values = list(schedule.values()) if isinstance(schedule,dict) else schedule if isinstance(schedule,list) else [schedule]
            if isinstance(schedule,dict):
                try:
                    if any(int(key) < 0 or str(int(key)) != str(key) for key in schedule):
                        raise ValueError()
                except (ValueError,TypeError,OverflowError):
                    error(f'disco.{name}', 'Schedule frame keys must be nonnegative integers')
            valid = bool(values) and all(isinstance(value,(int,float)) and not isinstance(value,bool)
                and math.isfinite(value) and lower <= value <= upper
                and (name != 'style_strength_schedule' or value > 0) for value in values)
            if not valid:
                error(f'disco.{name}', 'Use numeric schedule values from 2 to 1000' if name == 'steps_schedule'
                      else 'Use numeric schedule values greater than 0 and at most 1')
        if not d.encoders and d.clip_guidance_scale > 0 and (not distilled or d.distilled_guided):
            error('disco.encoders', 'Select at least one guidance encoder')
        if d.clamp_max <= 0:
            error('disco.clamp_max', 'Gradient clamp must be positive')
        if not math.isfinite(d.guidance_noise_cap_percent) or d.guidance_noise_cap_percent < 0:
            error('disco.guidance_noise_cap_percent', 'Noise-relative cap must be finite and nonnegative; 0 disables it')
        if d.cut_ic_pow <= 0 or not 0 <= d.cut_icgray_p <= 1:
            error('disco.cut_ic_pow', 'Crop exponent must be positive and grayscale probability between 0 and 1')
        if type(d.cut_schedule_version) is not int or d.cut_schedule_version != 2:
            error('disco.cut_schedule_version', 'Use cutout schedule version 2 (inference progress percentages)')
        def valid_percentage(key):
            return type(key) in (int, float) and math.isfinite(key) and 0 <= key <= 100
        if d.cut_icgray_schedule is not None:
            gray = d.cut_icgray_schedule
            if (not isinstance(gray, dict) or not gray
                    or any(not valid_percentage(k)
                           or isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) or not 0 <= v <= 1
                           for k, v in gray.items())):
                error('disco.cut_icgray_schedule',
                      'Use inference progress keys 0–100% and grayscale probabilities between 0 and 1')
        for name in ('cut_overview', 'cut_innercut'):
            schedule = getattr(d, name)
            if not schedule or any(not valid_percentage(k) or type(v) is not int or v < 0 for k, v in schedule.items()):
                error(f'disco.{name}', 'Use inference progress keys 0–100% and nonnegative integer cut counts per batch')
        for key in set(d.cut_overview) | set(d.cut_innercut) | {0}:
            def value_at(schedule):
                return next((schedule[k] for k in sorted(schedule, reverse=True) if k <= key), 0)
            if value_at(d.cut_overview) + value_at(d.cut_innercut) < 1:
                error('disco.cut_innercut', 'At least one cutout is required throughout sampling')
                break
    if config.turbo.enabled:
        if not config.flow.flow_warp:
            error('turbo.enabled', 'Turbo requires optical flow warping')
        if config.animatediff.enabled:
            error('turbo.enabled', 'Turbo uses sequential frames and cannot run with AnimateDiff')
        if config.contact_sheet.mode != 'off':
            error('turbo.enabled', 'Turbo cannot run with contact-sheet batching')
        if (config.turbo.intermediate_strength > 0
                and model_family_for_version(config.model_version) not in ('sd', 'disco')):
            error('turbo.intermediate_strength', 'Light diffusion on turbo frames requires SD or Disco')
        if config.turbo.interval < 2 or config.turbo.preroll < 0:
            error('turbo.interval', 'Interval must be at least 2; preroll cannot be negative')
        for name in ('intermediate_strength', 'consistency_soften', 'scene_cut_threshold'):
            if not 0 <= getattr(config.turbo, name) <= 1:
                error(f'turbo.{name}', 'Must be between 0 and 1')

    auto_sd15 = config.model_version in ('control_multi_v15', 'v1_5') and not config.sd_checkpoint_path
    if require_inputs and uses_sd_stack and not config.sd_checkpoint_path and not auto_sd15:
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
            if check_paths and entry.weight != 0 and (entry.path or spec is None):
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
    if config.video.width <= 0 or (not config.model_version.startswith('disco_') and config.video.width % 8):
        error("video.width", "Width must be positive and divisible by 8")
    if config.video.height <= 0 or (not config.model_version.startswith('disco_') and config.video.height % 8):
        error("video.height", "Height must be positive and divisible by 8")
    visible = (config.video.visible_width, config.video.visible_height)
    if any(type(v) is not int or v < 0 for v in visible) or bool(visible[0]) != bool(visible[1]):
        error('video.visible_width', 'Set both visible dimensions to positive integers, or both to zero')
    padded = config.model_version.startswith('disco_') and config.disco.pad_to_multiple_64
    if any(visible) and not padded:
        error('video.visible_width', 'Visible dimensions require Disco canvas padding')
    if config.video_assembly.crop_to_visible and not padded:
        error('video_assembly.crop_to_visible', 'Visible-content cropping requires Disco canvas padding')
    if len(config.frame_range) not in (1, 2) or any(value < 0 for value in config.frame_range):
        error("frame_range", "Frame range must contain one frame number or two non-negative frame numbers")
    elif len(config.frame_range) == 2 and config.frame_range[1] and config.frame_range[1] < config.frame_range[0]:
        error("frame_range", "End frame must be zero (auto) or at least the start frame")
    if uses_sd_stack:
        if config.diffusion.steps < 1:
            error("diffusion.steps", "Diffusion steps must be at least 1")
        if not 0 <= config.diffusion.style_strength <= 1:
            error("diffusion.style_strength", "Style strength must be between 0 and 1")
        if config.diffusion.guidance_mode not in FIELD_CHOICES["DiffusionConfig.guidance_mode"]:
            error("diffusion.guidance_mode", "Unknown SD/SDXL guidance mode")
        if config.diffusion.init_scale < 0:
            error("diffusion.init_scale", "Pixel guidance scale cannot be negative")
        if config.diffusion.init_latent_scale < 0:
            error("diffusion.init_latent_scale", "Latent guidance scale cannot be negative")
        if config.diffusion.clamp_max <= 0:
            error("diffusion.clamp_max", "Maximum gradient RMS must be positive")
        if config.diffusion.unet_cache not in FIELD_CHOICES["DiffusionConfig.unet_cache"]:
            error("diffusion.unet_cache", "Unknown U-Net cache mode")
        if config.diffusion.unet_cache_interval < 2:
            error("diffusion.unet_cache_interval",
                  "DeepCache interval must be at least 2")
        if config.diffusion.unet_cache_threshold < 0:
            error("diffusion.unet_cache_threshold",
                  "First Block Cache threshold cannot be negative")
        if config.diffusion.compile_unet and config.diffusion.unet_cache != 'off':
            error("diffusion.compile_unet",
                  "Compiled U-Net and U-Net caching cannot currently be combined")
        if (config.diffusion.unet_cache != 'off'
                and (config.diffusion.init_scale > 0
                     or config.diffusion.init_latent_scale > 0)):
            error("diffusion.unet_cache",
                  "U-Net caching cannot be combined with gradient guidance")
    if uses_sd_stack or config.model_version.startswith('disco_'):
        alignment = 64 if config.model_version.startswith('disco_') else 8
        if (config.diffusion.sampler_tile_size < alignment
                or config.diffusion.sampler_tile_size % alignment):
            error("diffusion.sampler_tile_size",
                  f"Sampler tile size must be at least {alignment} pixels and divisible by {alignment}")
        if not 0 <= config.diffusion.sampler_tile_overlap <= 50:
            error("diffusion.sampler_tile_overlap",
                  "Sampler tile overlap must be between 0 and 50 percent")
        if not 0 <= config.diffusion.sampler_tile_split_threshold <= 100:
            error('diffusion.sampler_tile_split_threshold', 'Tile split threshold must be between 0 and 100 percent')
        scale_schedule = config.diffusion.sampler_scale_schedule
        if not scale_schedule:
            error("diffusion.sampler_scale_schedule",
                  "Scale schedule must contain a step 0 value")
        else:
            if 0 not in scale_schedule:
                error("diffusion.sampler_scale_schedule",
                      "Scale schedule must start at sampling step 0")
            for step, scale in scale_schedule.items():
                if step < 0:
                    error("diffusion.sampler_scale_schedule",
                          "Scale schedule steps cannot be negative")
                    break
                if not 0 < scale <= 100:
                    error("diffusion.sampler_scale_schedule",
                          "Scale schedule values must be above 0 and at most 100 percent")
                    break
        if config.diffusion.sampler_scale_min_size < 0:
            error("diffusion.sampler_scale_min_size",
                  "Minimum multiscale size cannot be negative")
        if (config.model_version.startswith('disco_') and scale_schedule
                and 0 in scale_schedule
                and all(step >= 0 and 0 < value <= 100 for step, value in scale_schedule.items())):
            ordered = sorted(scale_schedule.items())
            if any(b[1] < a[1] for a, b in zip(ordered, ordered[1:])):
                error('diffusion.sampler_scale_schedule', 'Disco progressive scales must not decrease')
            if not any(e['path'] in ('disco.steps_schedule','disco.style_strength_schedule') for e in errors):
                from vibewarp.utils.scheduling import get_scheduled_arg
                d=config.disco
                frames={0}
                for schedule in (d.steps_schedule,d.style_strength_schedule):
                    if isinstance(schedule,dict): frames.update(int(key) for key in schedule)
                    elif isinstance(schedule,list): frames.update(range(len(schedule)))
                for frame in sorted(frames):
                    steps=int(get_scheduled_arg(frame,d.steps_schedule,True) if d.steps_schedule is not None else d.steps)
                    strength=float(get_scheduled_arg(frame,d.style_strength_schedule,True) if d.style_strength_schedule is not None else d.style_strength)
                    if 2 <= steps <= 1000 and 0 < strength <= 1:
                        total=steps-min(steps-1,int(steps*(1-strength)))
                        final_scale=next(value for step,value in reversed(ordered) if step < total)
                        if (final_scale < 100 and config.diffusion.sampler_scale_min_size
                                < max(config.video.width,config.video.height)):
                            error('diffusion.sampler_scale_schedule',
                                  f'Disco schedule must reach full size within {total} executed steps (schedule frame {frame})')
                            break
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
    if config.contact_sheet.mode not in FIELD_CHOICES["ContactSheetConfig.mode"]:
        error("contact_sheet.mode", "Unknown contact-sheet mode")
    if config.contact_sheet.layout not in FIELD_CHOICES["ContactSheetConfig.layout"]:
        error("contact_sheet.layout", "Unknown contact-sheet layout")
    if config.contact_sheet.gutter < 0:
        error("contact_sheet.gutter", "Contact-sheet gutter cannot be negative")
    if config.contact_sheet.mode != 'off':
        if family not in ('flux', 'hidream', 'qwen', 'mage'):
            error("contact_sheet.mode",
                  "Contact sheets require a ComfyUI image-edit model")
        elif family == 'flux' and config.flux.backend != 'comfy':
            error("contact_sheet.mode",
                  "Contact sheets currently require the ComfyUI Flux backend")
        else:
            refs = getattr(config, family).references
            if any(ref.source != 'none' for ref in refs[1:]):
                error(
                    f"{family}.references",
                    "Contact-sheet mode does not yet support optional references")
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
                                    model_families, workspace_metadata)

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
        if 'schedule_domain' in item.metadata:
            schema['schedule_domain'] = item.metadata['schedule_domain']
        if 'schedule_value' in item.metadata:
            schema['schedule_value'] = item.metadata['schedule_value']
        if 'schedule_interpolation' in item.metadata:
            schema['schedule_interpolation'] = item.metadata['schedule_interpolation']
        if item.name.endswith('_schedule'):
            schema.setdefault('schedule_domain', 'frame')
            fallback = item.name.removesuffix('_schedule')
            if fallback in hints:
                schema['schedule_fallback'] = fallback
        if item.name in ('text_prompts', 'negative_prompts', 'prompts', 'neg_prompts', 'image_prompts'):
            schema.update(schedule_domain='frame', schedule_kind='text')
        if item.metadata.get('rgb_color'):
            schema['rgb_color'] = True
        if 'slider' in item.metadata:
            schema['slider'] = dict(item.metadata['slider'])
        if 'vector_labels' in item.metadata:
            schema['vector_labels'] = list(item.metadata['vector_labels'])
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
                schema.update(workspace_metadata(section, item.name))
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
                                    TIER_LABELS, TIERS, WORKFLOW_GROUPS,
                                    WORKFLOW_LABELS, MODEL_LABELS, MODEL_GROUP_LABELS,
                                    model_runtime_group)
    # `groups` is the DECLARED order (ui_layout), not the order fields happen to appear on
    # RunConfig -- otherwise "Input Video" would sort behind "Output" and "Model".
    schema["tiers"] = [{"id": tier, "label": TIER_LABELS[tier],
                        "groups": list(GROUP_ORDER.get(tier, []))} for tier in TIERS]
    schema["model_family_by_version"] = dict(MODEL_FAMILY_BY_VERSION)
    schema["default_model_family"] = "sd"
    schema['workflows'] = [
        {'id': key, 'label': WORKFLOW_LABELS[key], 'groups': list(groups)}
        for key, groups in WORKFLOW_GROUPS.items()
    ]
    # Disco 256 remains readable in old settings and available to the engine,
    # but is no longer offered for new UI selections.
    schema['model_groups'] = [{'id': key, 'label': label} for key, label in MODEL_GROUP_LABELS.items()]
    schema['model_options'] = [
        {'value': key, 'label': value, 'group': model_runtime_group(key)}
        for key, value in MODEL_LABELS.items()]
    schema['properties']['model_version']['choices'] = list(MODEL_LABELS)
    schema['properties']['model_version']['choice_labels'] = dict(MODEL_LABELS)
    from vibewarp.disco_catalog import CLASSIC_ENCODERS, MODERN_ENCODERS
    schema['disco_encoders'] = list(CLASSIC_ENCODERS) + list(MODERN_ENCODERS)
    schema['properties']['disco']['properties']['encoders']['encoder_options'] = schema['disco_encoders']
    from vibewarp.config import DISCO_DISTILLED_GUIDED_PRESET
    schema['disco_distilled_preset'] = DISCO_DISTILLED_GUIDED_PRESET
    defaults = asdict(RunConfig())
    schema['disco_guided_restore'] = {
        key: ({field:defaults[key][field] for field in value}
              if key not in ('text_prompts','negative_prompts','model_version') else defaults[key])
        for key,value in DISCO_DISTILLED_GUIDED_PRESET.items()}
    schema['disco_guided_restore']['model_version'] = 'disco_512'
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
