"""Model download helpers — checkpoint and ControlNet weights from HuggingFace."""

import json
import os
from typing import Optional, Tuple

from huggingface_hub import hf_hub_download

from vibewarp.controlnet_catalog import CONTROLNET_CATALOG, controlnet_filenames


DEFAULT_CKPT_REPO = "digiplay/AbsoluteReality_v1.8.1"
DEFAULT_CKPT_FILE = "absolutereality_v181.safetensors"
DEFAULT_CKPT_SAVE_AS = "revAnimated_v122.safetensors"

CN_REPO = "lllyasviel/ControlNet-v1-1"

# CN key -> checkpoint filename on disk, for every net in the catalog (SD1.5,
# SD2.1, SDXL — not just the lllyasviel ones). Also drives ControlNet path
# inference from model_dir in pipeline._resolve_controlnet_path.
CONTROL_MODEL_FILES = controlnet_filenames()

# Default set when no settings file is given: the SD1.5 ControlNet v1.1 suite.
# The catalog also holds SDXL and third-party nets, but grabbing all of them
# unprompted would be tens of GB.
DEFAULT_CN_KEYS = [key for key, spec in CONTROLNET_CATALOG.items()
                   if CN_REPO in spec.url]


def _parse_hf_url(url: str) -> Optional[Tuple[str, str, str]]:
    """Split a HuggingFace resolve URL into (repo_id, filename, repo_type).

    Handles both model repos and Spaces:
        https://huggingface.co/<repo>/resolve/main/<path>
        https://huggingface.co/spaces/<repo>/resolve/main/<path>
    """
    marker = "huggingface.co/"
    if marker not in url or "/resolve/" not in url:
        return None
    tail = url.split(marker, 1)[1]
    repo_part, file_part = tail.split("/resolve/", 1)
    repo_type = "model"
    if repo_part.startswith("spaces/"):
        repo_type, repo_part = "space", repo_part[len("spaces/"):]
    # Strip the revision ("main/...") from the file path.
    filename = file_part.split("/", 1)[1] if "/" in file_part else file_part
    return repo_part, filename, repo_type


def _hf_download(repo_id: str, filename: str, local_dir: str, save_as: str = None,
                 repo_type: str = "model") -> str:
    """Download a file from HuggingFace; skip if already present. Returns local path."""
    dest_name = save_as or os.path.basename(filename)
    dest_path = os.path.join(local_dir, dest_name)
    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / 1024 / 1024
        print(f"  already exists ({size_mb:.0f} MB): {dest_path}", flush=True)
        return dest_path

    print(f"  downloading {repo_id}/{filename} -> {dest_path}", flush=True)
    tmp_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        repo_type=repo_type,
    )
    tmp_path = os.path.realpath(tmp_path)
    flat_dest = os.path.join(local_dir, os.path.basename(tmp_path))
    if flat_dest != dest_path:
        os.rename(flat_dest, dest_path)
    size_mb = os.path.getsize(dest_path) / 1024 / 1024
    print(f"  saved ({size_mb:.0f} MB): {dest_path}", flush=True)
    return dest_path


def cn_keys_from_settings(settings_path: str) -> list:
    """Return the list of CN key names referenced in a WarpFusion settings file."""
    with open(settings_path, encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("controlnet_multimodel", "{}")
    multimodel = json.loads(raw) if isinstance(raw, str) else raw
    return list(multimodel.keys())


def download_checkpoint(models_root: str, save_as: str = DEFAULT_CKPT_SAVE_AS) -> str:
    ckpt_dir = os.path.join(models_root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"\n[checkpoint] {DEFAULT_CKPT_REPO} -> checkpoints/{save_as}", flush=True)
    return _hf_download(DEFAULT_CKPT_REPO, DEFAULT_CKPT_FILE, ckpt_dir, save_as=save_as)


def download_controlnets(models_root: str, cn_keys: list) -> list:
    """Download the ControlNet checkpoints for the given key list. Returns local paths.

    Each net carries its own source repo (lllyasviel, CiaraRowles, xinsir, ...),
    so the repo comes from the catalog rather than a single hardcoded one. Many
    of them publish as `diffusion_pytorch_model.safetensors`, hence save_as: the
    files would otherwise overwrite each other in ControlNet/.
    """
    cn_dir = os.path.join(models_root, "ControlNet")
    os.makedirs(cn_dir, exist_ok=True)
    print(f"\n[controlnets] -> ControlNet/", flush=True)
    paths = []
    for key in cn_keys:
        spec = CONTROLNET_CATALOG.get(key)
        if spec is None:
            print(f"  WARNING: unknown CN key '{key}', skipping", flush=True)
            continue
        source = _parse_hf_url(spec.url) if spec.url else None
        if source is None:
            print(f"  WARNING: no download URL for '{key}' — supply the file "
                  f"manually in ControlNet/, skipping", flush=True)
            continue
        repo_id, filename, repo_type = source
        paths.append(_hf_download(repo_id, filename, cn_dir,
                                  save_as=spec.filename, repo_type=repo_type))
    return paths


def download_models(models_root: str, settings_path: str = None,
                    ckpt_only: bool = False, cn_only: bool = False,
                    ckpt_save_as: str = DEFAULT_CKPT_SAVE_AS,
                    cn_keys: list = None) -> None:
    """Download checkpoint and/or ControlNets.

    If settings_path is given, CN keys are read from its controlnet_multimodel field.
    If cn_keys is given explicitly, it overrides settings_path for CNs.
    """
    print(f"Models root: {models_root}", flush=True)

    if cn_keys is None:
        if settings_path:
            cn_keys = cn_keys_from_settings(settings_path)
            print(f"  CN keys from settings: {cn_keys}", flush=True)
        else:
            cn_keys = list(DEFAULT_CN_KEYS)

    total = 0

    if not cn_only:
        path = download_checkpoint(models_root, save_as=ckpt_save_as)
        total += os.path.getsize(path)

    if not ckpt_only:
        paths = download_controlnets(models_root, cn_keys)
        for p in paths:
            total += os.path.getsize(p)

    print(f"\nDone. Total: {total / 1024 / 1024 / 1024:.1f} GB", flush=True)
