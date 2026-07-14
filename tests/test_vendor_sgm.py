"""Tests for the vendored sgm SDXL subset (vendor/sgm/).

Verifies that every instantiate_from_config target in the shipped SDXL configs
resolves from the vendored tree, that the pytorch-lightning strip left the
models as plain nn.Modules, and that the light components instantiate and run.
No network, no checkpoint downloads — CLIP/OpenCLIP embedders are resolved but
not instantiated.
"""

import importlib
import os

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from vibewarp.core.model_loader import _ensure_vendor_on_path

_ensure_vendor_on_path()

CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'configs')
SDXL_CONFIGS = ['sd_xl_base.yaml', 'sd_xl_refiner.yaml']


def _collect_targets(node, found):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == 'target' and isinstance(v, str):
                found.add(v)
            else:
                _collect_targets(v, found)
    elif isinstance(node, list):
        for v in node:
            _collect_targets(v, found)


class TestConfigTargetsResolve:
    @pytest.mark.parametrize('config_name', SDXL_CONFIGS)
    def test_config_exists(self, config_name):
        assert os.path.exists(os.path.join(CONFIG_DIR, config_name))

    @pytest.mark.parametrize('config_name', SDXL_CONFIGS)
    def test_all_targets_importable(self, config_name):
        cfg = OmegaConf.to_container(
            OmegaConf.load(os.path.join(CONFIG_DIR, config_name)))
        targets = set()
        _collect_targets(cfg, targets)
        assert targets, "config has no instantiate targets?"
        for t in sorted(targets):
            mod, cls = t.rsplit('.', 1)
            obj = getattr(importlib.import_module(mod), cls)
            assert obj is not None, t

    def test_sgm_resolves_to_vendored_copy(self):
        import sgm
        assert 'vibewarp' in sgm.__file__.replace('\\', '/')


class TestPlStrip:
    def test_diffusion_engine_is_nn_module(self):
        from sgm.models.diffusion import DiffusionEngine
        assert issubclass(DiffusionEngine, nn.Module)
        assert 'pytorch_lightning' not in str(DiffusionEngine.__mro__)

    def test_autoencoder_is_nn_module(self):
        from sgm.models.autoencoder import AutoencoderKLInferenceWrapper
        assert issubclass(AutoencoderKLInferenceWrapper, nn.Module)


class TestLightComponents:
    def test_legacy_ddpm_discretization(self):
        from sgm.modules.diffusionmodules.discretizer import LegacyDDPMDiscretization
        disc = LegacyDDPMDiscretization()
        sigmas = disc(10, do_append_zero=False)
        assert len(sigmas) == 10
        assert torch.isfinite(torch.as_tensor(sigmas)).all()

    def test_discrete_denoiser_from_config_subtree(self):
        """Instantiate the denoiser exactly as the shipped sd_xl_base.yaml does."""
        from sgm.util import instantiate_from_config
        cfg = OmegaConf.load(os.path.join(CONFIG_DIR, 'sd_xl_base.yaml'))
        denoiser_cfg = cfg.model.params.denoiser_config
        denoiser = instantiate_from_config(denoiser_cfg)
        sigma = denoiser.idx_to_sigma(torch.tensor([500]))
        assert torch.isfinite(sigma).all()

    def test_concat_timestep_embedder(self):
        from sgm.modules.encoders.modules import ConcatTimestepEmbedderND
        emb = ConcatTimestepEmbedderND(outdim=256)
        out = emb(torch.tensor([[1024.0], [512.0]]))
        assert out.shape == (2, 256)

    def test_diagonal_gaussian_regularizer(self):
        from sgm.modules.autoencoding.regularizers import DiagonalGaussianRegularizer
        reg = DiagonalGaussianRegularizer(sample=False)
        z, log = reg(torch.randn(1, 8, 4, 4))
        assert z.shape == (1, 4, 4, 4)  # mean half of the 2*C channels

    def test_openai_wrapper(self):
        from sgm.modules.diffusionmodules.wrappers import OpenAIWrapper

        class TinyNet(nn.Module):
            def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
                return x

        wrapper = OpenAIWrapper(TinyNet(), compile_model=False)
        x = torch.randn(1, 4, 8, 8)
        out = wrapper(x, torch.tensor([1.0]), c={})
        assert out.shape == x.shape

    def test_lit_ema(self):
        from sgm.modules.ema import LitEma
        net = nn.Linear(4, 4)
        ema = LitEma(net)
        ema(net)  # one update step
        assert len(list(ema.buffers())) > 0

    def test_util_functions_present(self):
        """models/diffusion.py + encoders/modules.py import these from sgm.util —
        the previously-trimmed vendored util.py was missing several."""
        from sgm import util
        for fn in ['default', 'disabled_train', 'get_obj_from_str',
                   'instantiate_from_config', 'log_txt_as_img', 'append_dims',
                   'count_params', 'expand_dims_like', 'autocast',
                   'instantiate_from_config']:
            assert callable(getattr(util, fn)), fn
