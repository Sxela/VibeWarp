"""Vendored sgm (Stability generative-models, Sxela fork) — SDXL inference subset.

Excluded vs the fork: data/, inference/, lr_scheduler.py,
modules/autoencoding/{losses,lpips}/ and modules/diffusionmodules/loss.py
(training-only; loss.py pulls the LPIPS/torchvision chain).
Patches: pytorch-lightning stripped from models/ (nn.Module base);
kornia and fsspec imports made lazy. Everything else is fork-verbatim.
"""
from .models import AutoencodingEngine, DiffusionEngine
from .util import get_configs_path, instantiate_from_config

__version__ = "0.1.0"
