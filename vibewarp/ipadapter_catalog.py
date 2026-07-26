"""Canonical catalog of IP-Adapter models exposed by VibeWarp's UI."""

import os
from dataclasses import dataclass
from typing import Dict, List

from vibewarp.controlnet_catalog import normalize_model_version


_HF = "https://huggingface.co/h94/IP-Adapter/resolve/main"


@dataclass(frozen=True)
class IPAdapterSpec:
    """One regular/Plus IP-Adapter checkpoint the render path supports."""

    key: str
    label: str
    model_version: str
    clip_variant: str
    filename: str
    url: str


def _spec(key, label, version, clip_variant, filename, remote_path):
    return IPAdapterSpec(
        key, label, version, clip_variant, filename,
        f"{_HF}/{remote_path}",
    )


IPADAPTER_CATALOG: Dict[str, IPAdapterSpec] = {
    spec.key: spec for spec in [
        _spec("ipadapter_sd15", "Base", "sd15", "ViT-H",
              "ip-adapter_sd15.safetensors",
              "models/ip-adapter_sd15.safetensors"),
        _spec("ipadapter_sd15_light", "Light", "sd15", "ViT-H",
              "ip-adapter_sd15_light.safetensors",
              "models/ip-adapter_sd15_light.safetensors"),
        _spec("ipadapter_sd15_plus", "Plus", "sd15", "ViT-H",
              "ip-adapter-plus_sd15.safetensors",
              "models/ip-adapter-plus_sd15.safetensors"),
        _spec("ipadapter_sd15_plus_face", "Plus Face", "sd15", "ViT-H",
              "ip-adapter-plus-face_sd15.safetensors",
              "models/ip-adapter-plus-face_sd15.safetensors"),
        _spec("ipadapter_sd15_full_face", "Full Face", "sd15", "ViT-H",
              "ip-adapter-full-face_sd15.safetensors",
              "models/ip-adapter-full-face_sd15.safetensors"),
        _spec("ipadapter_sd15_vit_G", "Base (ViT-bigG)", "sd15", "ViT-bigG",
              "ip-adapter_sd15_vit-G.safetensors",
              "models/ip-adapter_sd15_vit-G.safetensors"),
        _spec("ipadapter_sdxl", "Base (ViT-bigG)", "sdxl", "ViT-bigG",
              "ip-adapter_sdxl.bin",
              "sdxl_models/ip-adapter_sdxl.bin"),
        _spec("ipadapter_sdxl_vit_h", "Base (ViT-H)", "sdxl", "ViT-H",
              "ip-adapter_sdxl_vit-h.bin",
              "sdxl_models/ip-adapter_sdxl_vit-h.bin"),
        _spec("ipadapter_sdxl_plus_vit_h", "Plus (ViT-H)", "sdxl", "ViT-H",
              "ip-adapter-plus_sdxl_vit-h.bin",
              "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"),
        _spec("ipadapter_sdxl_plus_face_vit_h", "Plus Face (ViT-H)",
              "sdxl", "ViT-H",
              "ip-adapter-plus-face_sdxl_vit-h.bin",
              "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin"),
    ]
}


def specs_for_version(model_version: str = "") -> List[IPAdapterSpec]:
    """Return selectable adapters compatible with ``model_version``."""
    version = normalize_model_version(model_version)
    specs = list(IPADAPTER_CATALOG.values())
    if version:
        specs = [spec for spec in specs if spec.model_version == version]
    return sorted(specs, key=lambda spec: spec.label)


def ipadapter_urls() -> Dict[str, str]:
    return {key: spec.url for key, spec in IPADAPTER_CATALOG.items()}


def resolve_ipadapter_path(
    model_key: str,
    configured_path: str = "",
    model_dir: str = "",
) -> str:
    """Resolve an instance path using the same catalog layout as the UI."""
    if configured_path:
        if os.path.isabs(configured_path) or not model_dir:
            return configured_path
        return os.path.join(model_dir, configured_path)
    spec = IPADAPTER_CATALOG.get(model_key)
    if spec is not None and model_dir:
        return os.path.join(model_dir, spec.filename)
    return ""
