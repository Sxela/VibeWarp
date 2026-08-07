"""Settings persistence - save/load run settings to JSON files."""

import hashlib
import json
import ntpath
import os
import posixpath
import re
import time
from typing import Any, Dict, Optional

from vibewarp.controlnet_catalog import controlnet_filenames, resolve_mode


def generate_file_hash(input_file):
    """Generate a deterministic hash from file name, size, and creation time.

    (c) Alex Spirin 2023
    """
    file_name = os.path.basename(input_file)
    file_size = os.path.getsize(input_file)
    creation_time = os.path.getctime(input_file)

    hasher = hashlib.sha256()
    hasher.update(file_name.encode('utf-8'))
    hasher.update(str(file_size).encode('utf-8'))
    hasher.update(str(creation_time).encode('utf-8'))
    return hasher.hexdigest()


def save_settings(settings: Dict[str, Any], output_dir: str,
                  batch_name: str, batch_num: int,
                  path: Optional[str] = None) -> str:
    """Save run settings to a JSON file.

    Args:
        settings: dict of all run settings
        output_dir: base output directory
        batch_name: name of the current batch
        batch_num: batch number
        path: optional sub-path for organizing settings

    Returns:
        Path to the saved settings file.
    """
    settings_dir = os.path.join(output_dir, "settings")
    os.makedirs(settings_dir, exist_ok=True)

    if path is not None:
        t = time.time()
        sub_dir = os.path.join(settings_dir, str(path))
        os.makedirs(sub_dir, exist_ok=True)
        fname = os.path.join(sub_dir, f"{batch_name}({path})_({t})_settings.txt")
    else:
        fname = os.path.join(settings_dir, f"{batch_name}({batch_num})_settings.txt")

    # Avoid overwriting - append modification time if exists
    if os.path.exists(fname):
        s_meta = os.path.getmtime(fname)
        base, ext = os.path.splitext(fname)
        fname = f"{base}_{s_meta}{ext}"

    settings['settings_filename'] = fname

    with open(fname, "w", encoding='utf8') as f:
        json.dump(settings, f, ensure_ascii=False, indent=4)

    return fname


def load_settings(settings_path: str) -> Dict[str, Any]:
    """Load settings from a JSON file or from EXIF data in an image.

    Args:
        settings_path: path to a settings .txt file or an image with EXIF settings

    Returns:
        Dict of settings.
    """
    if settings_path.endswith('.txt') or settings_path.endswith('.json'):
        with open(settings_path, 'r', encoding='utf8') as f:
            return json.load(f)

    # Try loading from image EXIF
    try:
        import piexif
        from PIL import Image
        img = Image.open(settings_path)
        exif_dict = piexif.load(img.info.get('exif', b''))
        user_comment = exif_dict.get('Exif', {}).get(piexif.ExifIFD.UserComment, b'')
        if isinstance(user_comment, bytes):
            user_comment = user_comment.decode('utf-8', errors='ignore')
        # Strip any leading charset markers (e.g., "ASCII\x00\x00\x00")
        if '\x00' in user_comment:
            user_comment = user_comment.split('\x00')[-1]
        return json.loads(user_comment)
    except Exception:
        raise ValueError(f"Could not load settings from {settings_path}")


_VIBEWARP_SECTIONS = [
    # Longer prefixes must come before shorter ones that share a stem
    'reconstruction_noise', 'video_assembly', 'animatediff', 'ipadapter',
    'diffusion', 'brightness', 'captions', 'scene', 'freeu',
    'color', 'warp', 'flow', 'video', 'mask', 'vae', 'hidream', 'flux', 'qwen',
    'mage',
]
_VIBEWARP_TOP_LEVEL = {
    'sd_checkpoint_path', 'model_path', 'model_version', 'batch_name',
    'output_dir', 'text_prompts', 'negative_prompts',
    'reference_label_opacity',
    'lora_dir', 'lora_merge_precision', 'frame_range',
    'keyframes_relative_to_frame_range',
}
# Keys present in saved settings that have no RunConfig equivalent
_VIBEWARP_SKIP = {'root_dir', 'animation_mode', 'settings_filename'}
_CN_MODEL_NAMES = [
    'control_sd15_inpaint', 'control_sd15_depth', 'control_sd15_softedge',
    'control_sd15_ip2p', 'control_sd15_tile', 'control_sd15_openpose',
    'control_sd15_seg', 'control_sd15_scribble', 'control_sd15_lineart',
    'control_sd15_shuffle', 'control_sd15_normal',
]


def is_vibewarp_settings(raw: Dict[str, Any]) -> bool:
    """Return True if raw dict looks like VibeWarp's own flat saved format."""
    # `diffusion_steps` is also a native WarpFusion notebook key, so it cannot
    # distinguish the formats. These section-prefixed keys are emitted only by
    # settings_to_dict(RunConfig).
    vw_keys = {'warp_flow_blend', 'reconstruction_noise_steps_pct',
               'vae_use_tiled_vae', 'flow_flow_warp'}
    return bool(vw_keys & raw.keys())


def load_vibewarp_settings(settings_path: str) -> Dict[str, Any]:
    """Load VibeWarp's own flat settings format and convert to nested config dict.

    The flat format uses prefixed keys like 'diffusion_steps', 'warp_flow_blend', etc.
    Returns a nested dict compatible with config_from_json / RunConfig.
    """
    raw = load_settings(settings_path)

    sections: Dict[str, Dict] = {s: {} for s in _VIBEWARP_SECTIONS}
    controlnet: Dict[str, Any] = {'enabled': True, 'models': {}}
    result: Dict[str, Any] = {}
    legacy_ipadapter_models: Dict[str, Dict[str, Any]] = {}
    ipadapter_entry_fields = (
        'combine_embeds', 'embeds_scaling', 'source_images', 'source_image',
        'weight_type', 'model_key', 'path', 'weight', 'start', 'end',
    )
    ipadapter_config_fields = {
        'enabled', 'clip_vision_model_path', 'models',
        'cache_embeddings', 'cache_size', 'flip_uc',
    }

    for key, value in raw.items():
        if key in _VIBEWARP_SKIP:
            continue
        if key in _VIBEWARP_TOP_LEVEL:
            result[key] = value
            continue

        matched = False

        # Sub-config sections
        for section in _VIBEWARP_SECTIONS:
            if key.startswith(section + '_'):
                subkey = key[len(section) + 1:]
                if section == 'ipadapter' and subkey not in ipadapter_config_fields:
                    # A short-lived writer bug flattened ipadapter.models one
                    # level too far (for example ipadapter_sd15_weight).
                    migrated_entry_field = False
                    for field_name in ipadapter_entry_fields:
                        suffix = '_' + field_name
                        if subkey.endswith(suffix):
                            instance = subkey[:-len(suffix)]
                            legacy_ipadapter_models.setdefault(
                                instance, {})[field_name] = value
                            migrated_entry_field = True
                            break
                    if not migrated_entry_field:
                        sections[section][subkey] = value
                else:
                    sections[section][subkey] = value
                matched = True
                break

        if matched:
            continue

        # ControlNet model entries: control_sd15_inpaint_weight, etc.
        if key.startswith('control_'):
            for model_name in _CN_MODEL_NAMES:
                if key.startswith(model_name + '_'):
                    field = key[len(model_name) + 1:]
                    controlnet['models'].setdefault(model_name, {})[field] = value
                    matched = True
                    break

        if matched:
            continue

        if key.startswith('controlnet_'):
            controlnet[key[len('controlnet_'):]] = value
        elif key.startswith('prompts_'):
            frame = key[len('prompts_'):]
            sections['reconstruction_noise'].setdefault('prompts', {})[frame] = value
        elif key.startswith('neg_prompts_'):
            frame = key[len('neg_prompts_'):]
            sections['reconstruction_noise'].setdefault('neg_prompts', {})[frame] = value
        elif key.startswith('steps_schedule_'):
            frame = key[len('steps_schedule_'):]
            sections['diffusion'].setdefault('steps_schedule', {})[frame] = value
        # else: unknown key, skip

    result['controlnet'] = controlnet
    if legacy_ipadapter_models:
        migrated: Dict[str, Dict[str, Any]] = {}
        for instance, entry in legacy_ipadapter_models.items():
            model_key = entry.get('model_key') or f'ipadapter_{instance}'
            instance_key = model_key
            suffix = 2
            while instance_key in migrated:
                instance_key = f'{model_key}__{suffix}'
                suffix += 1
            migrated[instance_key] = entry
        # A correctly nested map, if present, is authoritative.
        migrated.update(sections['ipadapter'].get('models', {}))
        sections['ipadapter']['models'] = migrated
    for section, data in sections.items():
        if data:
            result[section] = data

    return result


def _extract_scalar(value, default=0.0):
    """Extract a scalar from a value that may be a nested list or schedule."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, list):
        if not value:
            return default
        first = value[0]
        if isinstance(first, (int, float)):
            return first
        if isinstance(first, list) and first:
            return first[0] if isinstance(first[0], (int, float)) else default
    return default


_WINDOWS_PATH = re.compile(r'^[A-Za-z]:[\\/]')
_BACKSLASH = '\\'


def _pathmod(path: str):
    r"""The path module that actually understands this string.

    WarpFusion settings files are written on Windows and carry `C:\models\...` even when
    VibeWarp runs on Linux -- and posixpath sees no separators in that at all. `basename()`
    returns the WHOLE string, `isabs()` says False, and the models-root fallback then hunts
    for a file literally named 'C:\models\checkpoints\model.safetensors'. It never resolves.

    This only misbehaves on a non-Windows host, which is why it survived until CI ran the
    suite on Linux.
    """
    if _WINDOWS_PATH.match(path) or _BACKSLASH in path:
        return ntpath
    return posixpath


def path_basename(path: str) -> str:
    """Basename of a path written with EITHER separator, on any host OS."""
    return _pathmod(path).basename(path) if path else path


def path_is_absolute(path: str) -> bool:
    return bool(path) and _pathmod(path).isabs(path)


def path_parent(path: str, levels: int = 1) -> str:
    """Strip `levels` trailing components, honouring the path's own separator style."""
    module = _pathmod(path)
    for _ in range(levels):
        path = module.dirname(path)
    return path


def _find_in_roots(filename, roots, subdirs) -> Optional[str]:
    """Return the first existing <root>/<subdir>/<filename>, else None.

    Searches each root in order, and within a root each subdir in order
    (use '' for the root itself). Lets model files resolve across the
    installer's ./models layout, an explicit --models-root, and the root
    inferred from an absolute path in a settings file.
    """
    for root in roots:
        for subdir in subdirs:
            candidate = os.path.join(root, subdir, filename) if subdir \
                else os.path.join(root, filename)
            if os.path.isfile(candidate):
                return candidate
    return None


def load_warpfusion_settings(settings_path: str, models_root: Optional[str] = None) -> Dict[str, Any]:
    """Load WarpFusion notebook settings and convert to VibeWarp config format.

    Maps the flat WarpFusion settings dict (as exported by the notebook) into
    the nested VibeWarp config structure expected by config_from_json / RunConfig.

    Args:
        settings_path: path to a WarpFusion settings .txt file
        models_root: optional root directory for resolving model paths.
            If provided, ControlNet model names are resolved relative to
            a 'controlnet/' subdirectory under this root.

    Returns:
        Dict in VibeWarp config format (nested sub-config dicts).
    """
    raw = load_settings(settings_path)

    # Resolve model paths against a list of candidate model roots, searched
    # in order:
    #   1. explicit models_root (e.g. CLI --models-root)
    #   2. ./models under the current working directory (installer default)
    #   3. the root inferred from an absolute checkpoint path in the settings
    #      file, e.g. C:\models\checkpoints\rev.safetensors → C:\models
    # An absolute checkpoint path that already exists on disk always wins as
    # is, so a machine with a populated C:\models keeps working; a fresh
    # machine that downloaded into ./models resolves there instead. No path
    # is hardcoded — the default follows the working directory.
    model_path = raw.get('model_path', '')
    model_roots: list = []
    if models_root:
        model_roots.append(models_root)
    model_roots.append(os.path.join(os.getcwd(), 'models'))
    if model_path:
        # The settings file's own separator style decides how this is split -- a Windows
        # path must not be run through posixpath just because the host is Linux.
        inferred = (path_parent(model_path, 2) if path_is_absolute(model_path)
                    else os.path.dirname(os.path.dirname(os.path.abspath(model_path))))
        if inferred:
            model_roots.append(inferred)

    if model_path and not os.path.isfile(model_path):
        resolved = _find_in_roots(path_basename(model_path), model_roots,
                                  ('checkpoints', ''))
        if resolved:
            model_path = resolved

    # Put the root that actually holds the resolved checkpoint first, so CN /
    # IP-Adapter lookups prefer the same layout the checkpoint came from.
    if os.path.isfile(model_path):
        resolved_root = os.path.dirname(os.path.dirname(os.path.abspath(model_path)))
        model_roots = [resolved_root] + [r for r in model_roots if r != resolved_root]
    models_root = model_roots[0] if model_roots else models_root

    # Map model_version from notebook format to VibeWarp format
    model_version = raw.get('model_version', 'control_multi')
    version_map = {
        'control_multi': 'control_multi_v15',
        'control_multi_animatediff': 'control_multi_v15',
        'control_multi_v2': 'control_multi_v21',
        'control_multi_sdxl': 'control_multi_sdxl',
        'control_multi_animatediff_sdxl': 'control_multi_animatediff_sdxl',
    }
    model_version = version_map.get(model_version, model_version)

    # Parse text prompts — notebook stores as {frame_str: [prompt_list]}
    text_prompts = {}
    for k, v in raw.get('text_prompts', {}).items():
        if isinstance(v, list):
            text_prompts[int(k)] = ', '.join(v)
        else:
            text_prompts[int(k)] = str(v)

    negative_prompts = {}
    for k, v in raw.get('negative_prompts', {}).items():
        if isinstance(v, list):
            negative_prompts[int(k)] = ', '.join(v)
        else:
            negative_prompts[int(k)] = str(v)

    # Parse reconstruction prompts — notebook: rec_prompts, rec_neg_prompts
    # Same format as text_prompts: {frame_str: [prompt_list]}
    # None means "use main text_prompts / negative_prompts"
    rec_prompts = None
    raw_rec_prompts = raw.get('rec_prompts')
    if isinstance(raw_rec_prompts, dict) and raw_rec_prompts:
        rec_prompts = {}
        for k, v in raw_rec_prompts.items():
            if isinstance(v, list):
                rec_prompts[int(k)] = ', '.join(v)
            else:
                rec_prompts[int(k)] = str(v)

    rec_neg_prompts = None
    raw_rec_neg = raw.get('rec_neg_prompts')
    if isinstance(raw_rec_neg, dict) and raw_rec_neg:
        rec_neg_prompts = {}
        for k, v in raw_rec_neg.items():
            if isinstance(v, list):
                rec_neg_prompts[int(k)] = ', '.join(v)
            else:
                rec_neg_prompts[int(k)] = str(v)

    # Parse steps_schedule — notebook stores as {frame_str: steps}
    steps_schedule = None
    raw_steps = raw.get('steps_schedule')
    if isinstance(raw_steps, dict) and raw_steps:
        steps_schedule = {int(k): v for k, v in raw_steps.items()}

    # Parse cfg_scale_schedule
    cfg_scale_schedule = raw.get('cfg_scale_schedule')

    # Parse ControlNet multimodel
    controlnet_models = {}
    ipadapter_models = {}
    cn_multimodel = raw.get('controlnet_multimodel', '')
    if isinstance(cn_multimodel, str) and cn_multimodel:
        cn_multimodel = json.loads(cn_multimodel)
    if isinstance(cn_multimodel, dict):
        # Notebook controlnet name -> checkpoint filename (all known nets,
        # SD1.5 + SDXL). Single source of truth: vibewarp.controlnet_catalog.
        cn_file_map = controlnet_filenames()
        # Map preprocess field values from notebook to annotator type names
        _preprocess_to_annotator = {
            'canny': 'canny',
            'depth': 'depth',
            'depth_anything': 'depth',
            'midas': 'depth',
            'softedge': 'softedge',
            'PIDI': 'softedge',
            'pidi': 'softedge',
            'openpose': 'openpose',
            'dw_pose': 'dwpose',
            'DW': 'dwpose',
            'scribble': 'scribble',
            'lineart': 'lineart',
            'lineart_anime': 'lineart_anime',
            'mlsd': 'mlsd',
            'normalbae': 'normalbae',
            'seg': 'seg',
            'shuffle': 'shuffle',
            'inpaint': 'inpaint',
            'ip2p': 'ip2p',
            'tile': 'tile',
            'global': '',  # use auto-detection from model key
            '': '',
        }
        # IP-Adapters live inside controlnet_multimodel in WarpFusion settings
        # ("ipadapter_*" keys) — split them out; they are NOT ControlNets.
        _GLOBAL_KEYS = ('global', '', -1, '-1', 'global_settings')
        _ipa_filenames = {
            'ipadapter_sd15': 'ip-adapter_sd15.bin',
            'ipadapter_sd15_light': 'ip-adapter_sd15_light.safetensors',
            'ipadapter_sd15_plus': 'ip-adapter-plus_sd15.bin',
            'ipadapter_sd15_plus_face': 'ip-adapter-plus-face_sd15.bin',
            'ipadapter_sd15_full_face': 'ip-adapter-full-face_sd15.bin',
            'ipadapter_sd15_vit_G': 'ip-adapter_sd15_vit-G.bin',
            'ipadapter_sdxl': 'ip-adapter_sdxl.bin',
            'ipadapter_sdxl_vit_h': 'ip-adapter_sdxl_vit-h.bin',
            'ipadapter_sdxl_plus_vit_h': 'ip-adapter-plus_sdxl_vit-h.bin',
            'ipadapter_sdxl_plus_face_vit_h': 'ip-adapter-plus-face_sdxl_vit-h.bin',
            'ipadapter_sd15_faceid': 'ip-adapter-faceid_sd15.bin',
            'ipadapter_sd15_faceid_plus': 'ip-adapter-faceid-plus_sd15.bin',
            'ipadapter_sd15_faceid_plus_v2': 'ip-adapter-faceid-plusv2_sd15.bin',
            'ipadapter_sdxl_faceid': 'ip-adapter-faceid_sdxl.bin',
        }

        def _ipa_global(cn_cfg, key, raw_key, default):
            v = cn_cfg.get(key, 'global')
            if v in _GLOBAL_KEYS:
                return raw.get(raw_key, default)
            return v

        def _ipadapter_reference(source):
            """Translate WarpFusion's string source into the unified UI shape."""
            if source in ('stylized', 'warped'):
                return {'source': 'warped', 'image_path': ''}
            if source in ('previous', 'prev_frame'):
                return {'source': 'previous', 'image_path': ''}
            if source in ('', 'none', 'off', 'raw_frame', 'init'):
                return {'source': 'none', 'image_path': ''}
            return {'source': 'upload', 'image_path': str(source)}

        for cn_name, cn_cfg in cn_multimodel.items():
            if 'ipadapter' in cn_name:
                ipa_filename = _ipa_filenames.get(cn_name, f'{cn_name}.bin')
                ipa_path = _find_in_roots(
                    ipa_filename, model_roots,
                    ('ControlNet', 'controlnet', 'ipadapter', '')) or ipa_filename
                ipadapter_models[cn_name] = {
                    'model_key': cn_name,
                    'path': ipa_path,
                    'weight': cn_cfg.get('weight', 1.0),
                    'start': cn_cfg.get('start', 0.0),
                    'end': cn_cfg.get('end', 1.0),
                    'source_image': _ipadapter_reference(
                        cn_cfg.get('source', '')
                        if cn_cfg.get('source', '') not in _GLOBAL_KEYS else ''),
                    'weight_type': _ipa_global(cn_cfg, 'ip_weight_type', 'ip_weight_type', 'linear'),
                    'combine_embeds': _ipa_global(cn_cfg, 'ip_combine_embeds', 'ip_combine_embeds', 'concat'),
                    'embeds_scaling': _ipa_global(cn_cfg, 'ip_embeds_scaling', 'ip_embeds_scaling', 'V only'),
                }
                continue
            cn_filename = cn_file_map.get(cn_name, f'{cn_name}.pth')
            # Check standard CN subdirs under each candidate model root
            cn_path = _find_in_roots(
                cn_filename, model_roots,
                ('ControlNet', 'controlnet', '')) or cn_filename
            raw_preprocess = cn_cfg.get('preprocess', 'global')
            annotator = _preprocess_to_annotator.get(raw_preprocess, raw_preprocess)
            # Notebook: per-model detect_resolution of -1/'global' falls back
            # to the GLOBAL detect_resolution setting, not the render size.
            detect_resolution = cn_cfg.get('detect_resolution', -1)
            if detect_resolution in _GLOBAL_KEYS:
                detect_resolution = raw.get('detect_resolution', -1)
            cn_mode = cn_cfg.get('mode', 'global')
            if cn_mode in _GLOBAL_KEYS:
                cn_mode = raw.get('controlnet_mode', 'balanced')
            preset = resolve_mode(cn_mode)
            if preset is not None:
                layer_weights, zero_uncond = preset
            else:
                layer_weights = cn_cfg.get('layer_weights')
                zero_uncond = bool(cn_cfg.get('zero_uncond', False))
            controlnet_models[cn_name] = {
                'path': cn_path,
                'weight': cn_cfg.get('weight', 1.0),
                'start': cn_cfg.get('start', 0.0),
                'end': cn_cfg.get('end', 1.0),
                'source': cn_cfg.get('source', ''),
                'annotator': annotator,
                'detect_resolution': detect_resolution,
                'mode': cn_mode,
                'layer_weights': layer_weights,
                'zero_uncond': zero_uncond,
            }

    clip_vision_path = raw.get('ipadapter_clip_vision_model_path', '')
    if not clip_vision_path and ipadapter_models:
        # WarpFusion keeps both encoders in controlnet/clip_vision using these
        # stable filenames. Point at the directory so runtime can select H/G.
        found_clip = _find_in_roots(
            'clip_vision_vit_h.safetensors', model_roots,
            (os.path.join('controlnet', 'clip_vision'),
             os.path.join('ControlNet', 'clip_vision'), 'clip_vision'))
        if not found_clip:
            found_clip = _find_in_roots(
                'clip_vision_vit_bigg.safetensors', model_roots,
                (os.path.join('controlnet', 'clip_vision'),
                 os.path.join('ControlNet', 'clip_vision'), 'clip_vision'))
        if found_clip:
            clip_vision_path = os.path.dirname(found_clip)

    # Derive scalar fallback for steps from schedule (first frame or dict first value)
    _steps_default = 25
    if steps_schedule:
        first_key = min(steps_schedule.keys())
        _steps_default = steps_schedule[first_key]

    legacy_color_strength = raw.get('match_color_strength', 0.0)
    legacy_color_method = raw.get('colormatch_method', 'PDF')
    legacy_color_regrain = raw.get('colormatch_regrain') or False
    legacy_color_mode = raw.get(
        'colormatch_mode',
        'after' if raw.get('colormatch_after', True) else 'before')
    legacy_color_enabled = legacy_color_strength > 0
    migrated_color_strength = (
        legacy_color_strength if legacy_color_enabled else 0.5)

    # Build VibeWarp config dict
    config = {
        'sd_checkpoint_path': model_path,
        'model_version': model_version,
        'batch_name': raw.get('batch_name', 'warpfusion'),
        'text_prompts': text_prompts,
        'negative_prompts': negative_prompts,
        'frame_range': raw.get('frame_range', [0, 0]),
        'keyframes_relative_to_frame_range': raw.get(
            'keyframes_relative_to_frame_range', False),
        'lora_dir': raw.get('lora_dir', ''),
        # vibewarp extension (not a notebook key): 'fp16' = notebook-exact
        # merge arithmetic, 'fp32' = higher-precision deltas
        'lora_merge_precision': raw.get('lora_merge_precision', 'fp16'),
        'diffusion': {
            'steps': _steps_default,
            'cfg_scale': _extract_scalar(cfg_scale_schedule, 7.5),
            'sampler': raw.get('sampler', 'sample_euler_ancestral'),
            'do_softcap': raw.get('do_softcap', False),
            'softcap_thresh': float(raw.get('softcap_thresh', 0.9)),
            'softcap_q': float(raw.get('softcap_q', 1.0)),
            'seed': raw.get('seed', -1),
            'fixed_seed': raw.get('fixed_seed', False),
            'use_karras_noise': raw.get('use_karras_noise', True),
            'clip_skip': raw.get('clip_skip', 1),
            'sd_batch_size': raw.get('sd_batch_size', 1),
            'tiled_sampler': raw.get('tiled_sampler', False),
            'sampler_scale_schedule': raw.get(
                'sampler_scale_schedule', {0: 100.0}),
            'sampler_scale_min_size': raw.get(
                'sampler_scale_min_size', 0),
            'sampler_tile_size': raw.get('sampler_tile_size', 512),
            'sampler_tile_overlap': raw.get('sampler_tile_overlap', 35.0),
            'sampler_tile_split_threshold': raw.get('sampler_tile_split_threshold', 20.0),
            'style_strength': _extract_scalar(raw.get('style_strength_schedule'), 0.65),
            'guidance_mode': raw.get('guidance_mode', 'prev warped + cc'),
            'init_scale': _extract_scalar(raw.get('init_scale_schedule'), 0.0),
            'init_latent_scale': _extract_scalar(raw.get('latent_scale_schedule'), 0.0),
            'clamp_grad': raw.get('clamp_grad', True),
            'clamp_max': float(raw.get('clamp_max', 2.0)),
            'guidance_add_noise': raw.get('add_noise_to_latent', True),
            'dynamic_thresh': raw.get('dynamic_thresh', 30.0),
            'code_randomness': raw.get('code_randomness', 0.0),
            'noise_mode': raw.get('noise_mode', 'default'),
            'guidance_use_start_code': raw.get('guidance_use_start_code', True),
            'steps_schedule': steps_schedule,
            'cfg_scale_schedule': cfg_scale_schedule,
            'style_strength_schedule': raw.get('style_strength_schedule'),
        },
        'video': {
            'video_init_path': raw.get('video_init_path', ''),
            'extract_nth_frame': raw.get('extract_nth_frame', 1),
            'width': raw.get('width', 512),
            'height': raw.get('height', 512),
            'max_size': raw.get('max_size', 0),
        },
        'flow': {
            'flow_warp': raw.get('flow_warp', True),
            'check_consistency': raw.get('check_consistency', True),
            'use_legacy_cc': raw.get('use_legacy_cc', False),
            'flow_lq': raw.get('flow_lq', True),
            'num_flow_updates': raw.get('num_flow_updates', 20),
            'num_flow_workers': raw.get('num_flow_workers', 0),
            'flow_threads': raw.get('flow_threads', 4),
            'flow_maxsize': raw.get('flow_maxsize', 0),
            'missed_consistency_dilation': raw.get('missed_consistency_dilation', 2),
            'edge_consistency_width': raw.get('edge_consistency_width', 11),
            'missed_consistency_weight': raw.get('missed_consistency_weight', 1.0),
            'overshoot_consistency_weight': raw.get('overshoot_consistency_weight', 1.0),
            'edges_consistency_weight': raw.get('edges_consistency_weight', 1.0),
            'consistency_blur': raw.get('consistency_blur', 1),
            'consistency_dilate': raw.get('consistency_dilate', 3),
            'soften_consistency_mask': raw.get('soften_consistency_mask', 0.0),
            # Per-frame schedules (notebook: missed/overshoot/edges/blur/dilate/soften _schedule)
            'missed_consistency_schedule': raw.get('missed_consistency_schedule'),
            'overshoot_consistency_schedule': raw.get('overshoot_consistency_schedule'),
            'edges_consistency_schedule': raw.get('edges_consistency_schedule'),
            'consistency_blur_schedule': raw.get('consistency_blur_schedule'),
            'consistency_dilate_schedule': raw.get('consistency_dilate_schedule'),
            'soften_consistency_schedule': raw.get('soften_consistency_schedule'),
            'lazy_warp': raw.get('lazy_warp', True),
            'force_flow_generation': raw.get('force_flow_generation', False),
        },
        'warp': {
            'warp_mode': raw.get('warp_mode', 'use_image'),
            'warp_strength': raw.get('warp_strength', 1.0),
            'warp_num_k': raw.get('warp_num_k', 128),
            'warp_forward': raw.get('warp_forward', True),
            'warp_towards_init': raw.get('warp_towards_init', 'off'),
            'padding_ratio': raw.get('padding_ratio', 0.2),
            'padding_mode': raw.get('padding_mode', 'reflect'),
            'flow_blend_schedule': raw.get('flow_blend_schedule'),
        },
        'brightness': {
            'enable': raw.get('enable_adjust_brightness', False),
            'high_brightness_threshold': raw.get('high_brightness_threshold', 180),
            'high_brightness_adjust_ratio': raw.get('high_brightness_adjust_ratio', 0.97),
            'high_brightness_adjust_fix_amount': raw.get('high_brightness_adjust_fix_amount', 2),
            'max_brightness_threshold': raw.get('max_brightness_threshold', 254),
            'low_brightness_threshold': raw.get('low_brightness_threshold', 40),
            'low_brightness_adjust_ratio': raw.get('low_brightness_adjust_ratio', 1.03),
            'low_brightness_adjust_fix_amount': raw.get('low_brightness_adjust_fix_amount', 2),
            'min_brightness_threshold': raw.get('min_brightness_threshold', 1),
        },
        'color': {
            'before_enabled': (
                legacy_color_enabled
                and legacy_color_mode in ('before', 'both')),
            'before_strength': migrated_color_strength,
            'before_method': legacy_color_method,
            'before_regrain': legacy_color_regrain,
            'after_enabled': (
                legacy_color_enabled
                and legacy_color_mode in ('after', 'both')),
            'after_strength': migrated_color_strength,
            'after_method': legacy_color_method,
            'after_regrain': legacy_color_regrain,
            'match_color_strength': legacy_color_strength,
            'colormatch_method': legacy_color_method,
            'colormatch_regrain': legacy_color_regrain,
            'colormatch_mode': legacy_color_mode,
            'colormatch_after': raw.get('colormatch_after', True),
            'colormatch_turbo': raw.get('colormatch_turbo', False),
        },
        'controlnet': {
            'enabled': bool(controlnet_models),
            'model_dir': raw.get('controlnet_model_dir', ''),
            'models': controlnet_models,
            'mode': raw.get('controlnet_multimodel_mode', 'internal'),
            'cond_image_src': raw.get('cond_image_src', 'init'),
            'normalize_weights': raw.get('normalize_cn_weights', False),
            'depth_detector': raw.get('control_sd15_depth_detector', 'Midas'),
            'seg_detector': raw.get('control_sd15_seg_detector', 'Seg_UFADE20K'),
            'scribble_detector': raw.get('control_sd15_scribble_detector', 'PIDI'),
            'softedge_detector': raw.get('control_sd15_softedge_detector', 'PIDI'),
            'canny_low_threshold': int(raw.get('low_threshold', 100)),
            'canny_high_threshold': int(raw.get('high_threshold', 200)),
            'mlsd_value_threshold': float(raw.get('value_threshold', 0.1)),
            'mlsd_distance_threshold': float(raw.get('distance_threshold', 0.1)),
            'qr_cn_mask_grayscale': raw.get('qr_cn_mask_grayscale', False),
            'qr_cn_mask_invert': raw.get('qr_cn_mask_invert', False),
            'qr_cn_mask_thresh': int(raw.get('qr_cn_mask_thresh', 0)),
            'qr_cn_mask_clip_low': int(raw.get('qr_cn_mask_clip_low', 0)),
            'qr_cn_mask_clip_high': int(raw.get('qr_cn_mask_clip_high', 255)),
            'downscale_tile': int(raw.get('downscale_tile', 4)),
            'temporalnet_source': raw.get('temporalnet_source', 'init'),
            'temporalnet_skip_1st_frame': raw.get('temporalnet_skip_1st_frame', True),
            'inpaint_mask_source': raw.get('control_sd15_inpaint_mask_source', 'consistency_mask'),
        },
        'ipadapter': {
            'enabled': bool(ipadapter_models),
            'clip_vision_model_path': clip_vision_path,
            'models': ipadapter_models,
            'cache_embeddings': raw.get('cache_ipadapter', True),
            'cache_size': int(raw.get('ipadapter_embeds_cache_size', 10)),
            'flip_uc': raw.get('ip_flip_uc', False),
        },
        'freeu': {
            'do_freeunet': raw.get('do_freeunet', False),
            'apply_freeu_after_control': raw.get('apply_freeu_after_control', False),
            'b1': float(raw.get('b1', 1.2)),
            'b2': float(raw.get('b2', 1.4)),
            's1': float(raw.get('s1', 0.9)),
            's2': float(raw.get('s2', 0.2)),
        },
        'scene': {
            'analyze_video': raw.get('analyze_video', False),
            'diff_function': raw.get('diff_function', 'rmse+lpips'),
            'make_schedules': raw.get('make_schedules', False),
            'respect_sched': raw.get('respect_sched', True),
            'diff_override': raw.get('diff_override', []) or [],
            'flow_blend_template': raw.get('flow_blend_template', [0.8, 0.0, 0.51, 2]),
            'cc_masked_template': raw.get('cc_masked_template', [0.7, 0, 0.51, 2]),
            'steps_template': raw.get('steps_template', None),
            'style_strength_template': raw.get('style_strength_template', None),
            'cfg_scale_template': raw.get('cfg_scale_template', None),
            'latent_scale_template': raw.get('latent_scale_template', None),
            'init_scale_template': raw.get('init_scale_template', None),
            'image_scale_template': raw.get('image_scale_template', None),
        },
        'captions': {
            'make_captions': raw.get('make_captions', False),
            'keyframe_source': raw.get('keyframe_source', 'Every n-th frame'),
            'nth_frame': int(raw.get('nth_frame', 10)),
            'user_defined_keyframes': raw.get('user_defined_keyframes', []),
            'diff_thresh': float(raw.get('diff_thresh', 0.33)),
            'offset_mode': raw.get('offset_mode', 'Fixed'),
            'fixed_offset': int(raw.get('fixed_offset', 0)),
            'blip_question': raw.get('blip_question') or '',
        },
        'mask': {
            'use_background_mask': raw.get('use_background_mask', False),
            'invert_mask': raw.get('invert_mask', False),
            'apply_mask_after_warp': raw.get('apply_mask_after_warp', True),
            'background': raw.get('background', 'init_video'),
            'background_source': raw.get('background_source', 'red'),
            'mask_clip_low': int(raw.get('mask_clip_low', 0)),
            'mask_clip_high': int(raw.get('mask_clip_high', 255)),
        },
        'vae': {
            'use_tiled_vae': raw.get('use_tiled_vae', False),
            'num_tiles': raw.get('num_tiles', None),
            'tile_size': raw.get('tile_size', 128),
            'vae_stride': raw.get('vae_stride', 96),
            'vae_padding': raw.get('vae_padding', None),
        },
        'video_assembly': {
            'keep_audio': raw.get('keep_audio', True),
            'use_deflicker': raw.get('use_deflicker', False),
            'blend_mode': raw.get('blend_mode', 'optical flow'),
            'blend': float(raw.get('blend', 0.5)),
            'missed_consistency_weight': float(
                raw.get('missed_consistency_weight', 1.0)),
            'overshoot_consistency_weight': float(
                raw.get('overshoot_consistency_weight', 1.0)),
            'edges_consistency_weight': float(
                raw.get('edges_consistency_weight', 1.0)),
            'upscale_ratio': int(raw.get('upscale_ratio', 1)),
            'upscale_model': raw.get('upscale_model', 'realesr-animevideov3'),
            'upscale_model_path': raw.get('upscale_model_path', ''),
            'use_background_mask_video': raw.get('use_background_mask_video', False),
            'invert_mask_video': raw.get('invert_mask_video', False),
            'background_video': raw.get('background_video', 'init_video'),
            'background_source_video': raw.get('background_source_video', ''),
            'mask_clip_low': int(raw.get('mask_clip_low', 0)),
            'mask_clip_high': int(raw.get('mask_clip_high', 255)),
        },
        'animatediff': {
            'enabled': 'animatediff' in raw.get('model_version', ''),
            'batch_length': raw.get('batch_length', 16),
            'batch_overlap': raw.get('batch_overlap', 1),
            'context_length': raw.get('context_length', 16),
            'context_overlap': raw.get('context_overlap', 4),
            'stop_early': raw.get('stop_early', 0),
            'blend_batch_outputs': raw.get('blend_batch_outputs', True),
            'overlap_stylized': raw.get('overlap_stylized', True),
        },
        'reconstruction_noise': {
            # Notebook: noise_mode dropdown sets use_predicted_noise internally.
            # noise_mode='reconstructed' → enabled, anything else → disabled.
            # Older settings files only have use_predicted_noise (no noise_mode).
            'enabled': (raw['noise_mode'] == 'reconstructed') if 'noise_mode' in raw
                       else raw.get('use_predicted_noise', False),
            'randomness': raw.get('rec_randomness', 0.0),
            'cfg_scale': raw.get('rec_cfg', 1.0),
            'steps_pct': raw.get('rec_steps_pct', 1.0),
            'source': raw.get('rec_source', 'init'),
            'steps_cutoff': raw.get('rec_steps_cutoff', 0),
            'noise_scale': raw.get('rec_noise_scale', 1.0),
            'separate_uncond': raw.get('rec_separate_uc', True),
            'disable_controlnets': raw.get('disable_rec_noise_controlnets', False),
            'batched': raw.get('batched_adiff_rec_noise', False),
            'prompts': rec_prompts,  # None = use main text_prompts
            'neg_prompts': rec_neg_prompts,  # None = use main negative_prompts
        },
    }

    return config


def settings_to_dict(config) -> Dict[str, Any]:
    """Convert a RunConfig dataclass to a flat settings dict for serialization.

    Args:
        config: a RunConfig instance (or any dataclass)

    Returns:
        Flat dict of settings.
    """
    from dataclasses import asdict, fields
    import dataclasses

    result = {}

    # Collect names of fields that are themselves dataclasses
    nested_dc_fields = set()
    for f in fields(config):
        if dataclasses.is_dataclass(f.type if isinstance(f.type, type) else None):
            nested_dc_fields.add(f.name)

    raw = asdict(config)

    def _flatten(d, prefix='', depth=0):
        for k, v in d.items():
            key = k if not prefix else f"{prefix}_{k}"
            if isinstance(v, dict) and depth == 0 and k in nested_dc_fields:
                # Nested dataclass — flatten its fields
                _flatten(v, key, depth + 1)
            elif isinstance(v, dict):
                # User-supplied dict (e.g. text_prompts) — keep as-is
                result[key] = v
            else:
                result[key] = v

    _flatten(raw)
    return result
