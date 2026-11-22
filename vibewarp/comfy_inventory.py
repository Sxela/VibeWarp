"""Server-reported Comfy model inventory and local download destinations."""
from pathlib import Path
from urllib.parse import urlsplit
import requests

COMFY_FAMILIES = ('flux', 'hidream', 'qwen', 'mage', 'comfy_image')
MODEL_FIELD_CATEGORIES = {'comfy_unet_name': 'diffusion_models', 'comfy_clip_name': 'text_encoders',
    'comfy_vae_name': 'vae', 'comfy_checkpoint_name': 'checkpoints',
    'comfy_lora_name': 'loras', 'comfy_edit_lora_name': 'loras'}


def diffusion_model_category(name):
    return 'unet_gguf' if name.lower().endswith('.gguf') else 'diffusion_models'


def diffusion_loader(name):
    if diffusion_model_category(name) == 'unet_gguf':
        return {'class_type': 'UnetLoaderGGUF', 'inputs': {'unet_name': name}}
    return {'class_type': 'UNETLoader', 'inputs': {'unet_name': name, 'weight_dtype': 'default'}}


def model_fields(config):
    from vibewarp.ui_layout import model_family_for_version
    from vibewarp.config import (flux_model_files, qwen_model_files, mage_model_defaults, comfy_image_defaults,
        FLUX_MODEL_DEFAULTS, QWEN_MODEL_DEFAULTS, MAGE_MODEL_DEFAULTS, COMFY_IMAGE_DEFAULTS)
    family = model_family_for_version(config.model_version)
    if family not in COMFY_FAMILIES or (family == 'flux' and config.flux.backend != 'comfy'):
        return None, []
    section = getattr(config, family)
    values = vars(section).copy()
    presets = {'flux': FLUX_MODEL_DEFAULTS, 'qwen': QWEN_MODEL_DEFAULTS,
               'mage': MAGE_MODEL_DEFAULTS, 'comfy_image': COMFY_IMAGE_DEFAULTS}
    resolvers = {'flux': flux_model_files, 'qwen': qwen_model_files,
                 'mage': mage_model_defaults, 'comfy_image': comfy_image_defaults}
    if family in resolvers: values.update(resolvers[family](config))
    suggested = {key: getattr(type(section), key) for key in MODEL_FIELD_CATEGORIES if hasattr(type(section), key)}
    suggested.update(presets.get(family, {}).get(config.model_version, {}))
    result = []
    for key, category in MODEL_FIELD_CATEGORIES.items():
        if key not in values: continue
        name = str(values[key])
        required = bool(name)
        if key == 'comfy_lora_name': required &= getattr(section, 'use_lightning_lora', True)
        if key == 'comfy_edit_lora_name':
            required &= config.model_version == 'moss_image' and section.mode == 'edit'
        if category == 'diffusion_models':
            category = diffusion_model_category(name)
        result.append(dict(section=family, field=key, category=category, name=name,
                           suggested=str(suggested.get(key, '')), required=required))
    return section.comfy_server_url.rstrip('/'), result


def model_specs(config):
    url, fields = model_fields(config)
    return url, [(row['category'], row['name']) for row in fields if row['required']]


def download_folders(config, url):
    if config.comfy_models_dir:
        return {category: Path(config.comfy_models_dir)/category
                for category in set(MODEL_FIELD_CATEGORIES.values()) | {'unet'}}
    # Server paths are writable here only for a loopback ComfyUI on this machine.
    if urlsplit(url).hostname not in ('localhost', '127.0.0.1', '::1'): return {}
    try:
        response = requests.get(f'{url}/experiment/models', timeout=5)
        response.raise_for_status()
        entries = response.json()
        if not isinstance(entries, list): return {}
        folders = {}
        for entry in entries:
            if not isinstance(entry, dict): continue
            category = entry.get('name')
            if category not in set(MODEL_FIELD_CATEGORIES.values()) | {'unet', 'unet_gguf'}: continue
            for folder in entry.get('folders', []):
                if isinstance(folder, str) and Path(folder).is_absolute() and Path(folder).is_dir():
                    folders[category] = Path(folder)
                    if category == 'unet_gguf': folders['unet'] = Path(folder)
                    break
        return folders
    except (requests.RequestException, ValueError, OSError):
        return {}


def comfy_model_catalog(config):
    from vibewarp.comfy_assets import COMFY_SOURCES
    url, fields = model_fields(config)
    inventories = {}
    rows = []
    folders = None
    for field in fields:
        category, name = field['category'], field['name']
        if category not in inventories:
            try:
                response = requests.get(f'{url}/models/{category}', timeout=5)
                response.raise_for_status()
                names = response.json()
                if not isinstance(names, list) or any(not isinstance(item, str) for item in names):
                    raise ValueError('Invalid ComfyUI model listing')
                inventories[category] = (names, '')
            except (requests.RequestException, ValueError) as exc:
                inventories[category] = (None, f'Cannot query ComfyUI {category}: {exc}')
        names, error = inventories[category]
        normalized = name.replace(chr(92), '/')
        present = names is not None and normalized in {item.replace(chr(92), '/') for item in names}
        destination = None
        known = COMFY_SOURCES.get(name)
        if names is not None and not present and known:
            if folders is None: folders = download_folders(config, url)
            folder = folders.get(known[0])
            if folder is not None: destination = str((folder/name).resolve())
        rows.append({**field, 'path': f'{category}/{normalized}',
            'status': 'unknown' if names is None else 'present' if present else 'missing',
            'available': names or [], 'total': None, 'downloadable': destination is not None,
            'download_path': destination, 'source': 'comfy', 'server_url': url, 'error': error})
    return rows


def required_comfy_files(config):
    return [row for row in comfy_model_catalog(config) if row['required']]
