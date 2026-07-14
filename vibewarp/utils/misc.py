"""Miscellaneous utility functions."""

import json
import os
import random

import numpy as np
import piexif
import torch


def create_path(filepath):
    """Create directory path if it doesn't exist."""
    os.makedirs(filepath, exist_ok=True)


def seed_everything(seed, deterministic=False):
    """Set random seeds for reproducibility across all frameworks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def json2exif(settings):
    """Encode a settings dict as EXIF UserComment data."""
    settings_str = json.dumps(settings)
    exif_ifd = {piexif.ExifIFD.UserComment: settings_str.encode()}
    exif_dict = {"Exif": exif_ifd}
    exif_dat = piexif.dump(exif_dict)
    return exif_dat


def release_vram(*, log: bool = True) -> float:
    """Drop cached CUDA memory and run a GC pass. Returns the GiB still allocated.

    A finished render leaves the UNet, VAE, ControlNets, annotators and IP-Adapter
    embeddings alive in the module-level caches, plus whatever the allocator is holding
    in its own pool. The server is long-lived and renders back to back, so without this
    the second job starts against a VRAM ceiling set by the first one — and an OOM there
    looks like "the settings are too big" rather than "we never let go of the last run".

    Only the ALLOCATOR's cache is dropped; models stay in their caches so the next job
    does not pay the reload. Call this at job boundaries, not between frames.
    """
    import gc

    gc.collect()
    try:
        import torch
    except ImportError:
        return 0.0
    if not torch.cuda.is_available():
        return 0.0

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    allocated = torch.cuda.memory_allocated() / 1024 ** 3
    reserved = torch.cuda.memory_reserved() / 1024 ** 3
    if log:
        print(f'VRAM after cleanup: {allocated:.2f} GiB allocated, '
              f'{reserved:.2f} GiB reserved by the allocator')
    return allocated
