"""Resolve only the weight files required by the selected configuration."""
from dataclasses import dataclass
from pathlib import Path
import json

from vibewarp.acquisition import acquire
from vibewarp.config import RunConfig, MODEL_URLS
from vibewarp.controlnet_catalog import CONTROLNET_CATALOG
from vibewarp.ipadapter_catalog import IPADAPTER_CATALOG


@dataclass(frozen=True)
class ModelAsset:
    url: str
    path: str
    sha256: str = ''

    def __post_init__(self):
        object.__setattr__(self, 'path', str(Path(self.path).resolve()))


def models_root(config: RunConfig) -> Path:
    path = Path(config.model_path) if config.model_path else Path(config.root_dir) / 'models'
    return path.parent if path.is_file() else path


def required_model_files(config: RunConfig) -> list[dict]:
    """Local presence preview only: no transfers, Hub requests or GPU loading."""
    files = {}
    def add(path, name=None, *, downloadable=False, present=None, size=None):
        path = Path(path).expanduser().resolve()
        if present is None:
            present = path.is_file() and path.stat().st_size > 0
        if size is None and path.is_file():
            size = path.stat().st_size
        files[str(path)] = dict(name=name or path.name, path=str(path),
            status='present' if present else 'missing', total=size,
            downloadable=downloadable)
    for asset in selected_assets(config):
        add(asset.path, downloadable=True)
    if config.model_version.startswith('disco_'):
        d = config.disco
        if d.checkpoint_path:
            add(d.checkpoint_path,'Disco checkpoint')
        if d.quality_lora_enabled and d.quality_lora_path:
            add(d.quality_lora_path,'Quality LoRA')
        if d.sampler == 'distilled':
            if d.few_step_lora_path:
                path = Path(d.few_step_lora_path).expanduser().resolve()
                present = None
                if path.is_dir():
                    try:
                        generation = json.loads((path/'best.json').read_text())['checkpoint']
                        present = (isinstance(generation,str) and generation not in ('','.','..')
                            and '/' not in generation and '\\' not in generation
                            and (path/'best'/generation/'best-ema.safetensors').is_file()
                            and (path/'best'/generation/'COMPLETE.json').is_file())
                    except (OSError,ValueError,KeyError,TypeError):
                        present = False
                add(path,'Distilled LoRA',present=present)
        if d.sampler != 'distilled' or d.distilled_guided:
            if d.use_secondary_model and d.secondary_path:
                add(d.secondary_path,'Secondary denoiser')
            from vibewarp.disco_catalog import MODERN_ENCODERS
            for name in d.encoders:
                if name not in MODERN_ENCODERS:
                    continue
                root = models_root(config)/'disco'/name
                required = [root/value for value in ('config.json','preprocessor_config.json','tokenizer_config.json')]
                tokenizer = [root/value for value in ('tokenizer.json','tokenizer.model')]
                weights = [root/'model.safetensors']
                index = root/'model.safetensors.index.json'
                if index.is_file():
                    try:
                        names = set(json.loads(index.read_text())['weight_map'].values())
                        weights = [root/value for value in names]
                        if any(not p.resolve().is_relative_to(root.resolve()) for p in weights):
                            weights = []
                    except (OSError,ValueError,KeyError,TypeError):
                        weights = []
                present = (bool(weights) and all(p.is_file() and p.stat().st_size > 0 for p in required+weights)
                           and any(p.is_file() and p.stat().st_size > 0 for p in tokenizer))
                size = sum(p.stat().st_size for p in required+weights+tokenizer if p.is_file())
                add(root,f'{name} (model and tokenizer)',downloadable=True,present=present,size=size or None)
    else:
        if config.sd_checkpoint_path and config.model_version.startswith(('control_','v1_','v2_')):
            add(config.sd_checkpoint_path,'Diffusion checkpoint')
        if config.controlnet.enabled:
            for name,entry in config.controlnet.models.items():
                if entry.weight and entry.path:
                    add(entry.path,name)
        if config.ipadapter.enabled:
            for name,entry in config.ipadapter.models.items():
                if entry.weight and entry.path:
                    add(entry.path,name)
    return list(files.values())


def selected_assets(config: RunConfig) -> list[ModelAsset]:
    """Catalog weights with known sources; custom files are never substituted."""
    from vibewarp.ui_layout import model_family_for_version
    family = model_family_for_version(config.model_version)
    root = models_root(config)
    assets = []
    if config.comfy_models_dir and family in ('flux', 'hidream', 'qwen', 'mage'):
        from vibewarp.comfy_assets import COMFY_SOURCES
        from vibewarp.config import flux_model_files, qwen_model_files, mage_model_defaults
        section = getattr(config, family)
        files = ({key: value for key, value in vars(section).items() if key.startswith('comfy_') and key.endswith('_name')})
        if family == 'flux':
            files.update(flux_model_files(config))
        if family == 'qwen':
            files.update(qwen_model_files(config))
            if not config.qwen.use_lightning_lora:
                files.pop('comfy_lora_name', None)
        if family == 'mage':
            files.update(mage_model_defaults(config))
        for name in files.values():
            if name in COMFY_SOURCES:
                folder, url = COMFY_SOURCES[name]
                assets.append(ModelAsset(url, str(Path(config.comfy_models_dir) / folder / name)))
    if family == 'disco':
        from vibewarp.core.disco import disco_assets
        assets.extend(disco_assets(config))
    if family == 'sd':
        if not config.sd_checkpoint_path:
            from vibewarp.core.download import DEFAULT_CKPT_REPO, DEFAULT_CKPT_FILE
            if 'sdxl' not in config.model_version and 'v21' not in config.model_version:
                assets.append(ModelAsset(
                    f'https://huggingface.co/{DEFAULT_CKPT_REPO}/resolve/main/{DEFAULT_CKPT_FILE}',
                    str(root / 'checkpoints' / DEFAULT_CKPT_FILE)))
        cn_dir = Path(config.controlnet.model_dir) if config.controlnet.model_dir else root / 'ControlNet'
        if config.controlnet.enabled:
            for key, entry in config.controlnet.models.items():
                spec = CONTROLNET_CATALOG.get(key)
                if entry.weight and spec and spec.url and not entry.path:
                    assets.append(ModelAsset(spec.url, str(cn_dir / spec.filename)))
        if config.ipadapter.enabled:
            for key, entry in config.ipadapter.models.items():
                spec = IPADAPTER_CATALOG.get(entry.model_key or key)
                if not entry.weight or not spec:
                    continue
                if not entry.path:
                    assets.append(ModelAsset(spec.url, str(cn_dir / spec.filename)))
                variant = 'clip_vision_vit_bigg' if spec.clip_variant == 'ViT-bigG' else 'clip_vision_vit_h'
                if not config.ipadapter.clip_vision_model_path:
                    assets.append(ModelAsset(MODEL_URLS[variant], str(cn_dir / f'{variant}.safetensors')))
    # RAFT is shared by every renderer and uses torchvision's cache.
    if config.flow.flow_warp:
        import torch.hub
        from torchvision.models.optical_flow import Raft_Large_Weights
        url = Raft_Large_Weights.C_T_SKHT_V1.url
        assets.append(ModelAsset(url, str(Path(torch.hub.get_dir()) / 'checkpoints' / url.rsplit('/', 1)[-1])))
    return list(dict.fromkeys(assets))


def ensure_selected_assets(config: RunConfig):
    """Acquire known weights and fill only missing paths on this run's config."""
    for asset in selected_assets(config):
        acquire(asset.url, asset.path, sha256=asset.sha256)
    from vibewarp.ui_layout import model_family_for_version
    if model_family_for_version(config.model_version) == 'disco':
        from vibewarp.core.disco import ensure_guidance_assets
        ensure_guidance_assets(config)
    if model_family_for_version(config.model_version) == 'sd':
        root = models_root(config)
        if not config.sd_checkpoint_path:
            from vibewarp.core.download import DEFAULT_CKPT_FILE
            candidate = root / 'checkpoints' / DEFAULT_CKPT_FILE
            if candidate.is_file():
                config.sd_checkpoint_path = str(candidate)
        if not config.controlnet.model_dir:
            # Explicit relative paths used the working directory before the
            # shared directory was inferred. Retain that meaning for loaders.
            for section in (config.controlnet, config.ipadapter):
                for entry in section.models.values():
                    if entry.path and not Path(entry.path).is_absolute():
                        entry.path = str(Path(entry.path).resolve())
            config.controlnet.model_dir = str(root / 'ControlNet')
        if config.ipadapter.enabled and not config.ipadapter.clip_vision_model_path:
            config.ipadapter.clip_vision_model_path = config.controlnet.model_dir
