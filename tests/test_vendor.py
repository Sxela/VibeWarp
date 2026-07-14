"""Tests for vendored packages — import smoke tests."""

import pytest
import torch


class TestResizeRight:
    def test_import(self):
        from vibewarp.vendor.resize_right import resize
        assert callable(resize)

    def test_resize_tensor(self):
        from vibewarp.vendor.resize_right import resize
        x = torch.randn(1, 3, 16, 16)
        out = resize(x, out_shape=(1, 3, 32, 32))
        assert out.shape == (1, 3, 32, 32)

    def test_interp_methods(self):
        from vibewarp.vendor.resize_right import interp_methods
        assert hasattr(interp_methods, 'cubic')
        assert hasattr(interp_methods, 'lanczos2')
        assert hasattr(interp_methods, 'lanczos3')


class TestColorTransfer:
    def test_import(self):
        from vibewarp.vendor.python_color_transfer import ColorTransfer, Regrain
        assert callable(ColorTransfer)
        assert callable(Regrain)

    def test_instantiate(self):
        from vibewarp.vendor.python_color_transfer import ColorTransfer
        ct = ColorTransfer()
        assert ct.eps == 1e-6


class TestKDiffusion:
    def test_import_sampling(self):
        from vibewarp.vendor.k_diffusion import sampling
        assert hasattr(sampling, 'get_sigmas_karras')
        assert hasattr(sampling, 'sample_dpm_2')
        assert hasattr(sampling, 'sample_dpmpp_sde')

    def test_import_external(self):
        from vibewarp.vendor.k_diffusion import external
        assert hasattr(external, 'CompVisDenoiser')
        assert hasattr(external, 'CompVisVDenoiser')

    def test_import_utils(self):
        from vibewarp.vendor.k_diffusion import utils
        assert hasattr(utils, 'append_dims')

    def test_append_dims(self):
        from vibewarp.vendor.k_diffusion.utils import append_dims
        x = torch.randn(4)
        result = append_dims(x, 3)
        assert result.shape == (4, 1, 1)

    def test_get_sigmas_karras(self):
        from vibewarp.vendor.k_diffusion.sampling import get_sigmas_karras
        sigmas = get_sigmas_karras(10, 0.01, 14.6)
        assert sigmas.shape[0] == 11  # n + 1 (appended zero)
        assert sigmas[-1] == 0.0


class TestLDM:
    def test_import_util(self):
        from vibewarp.vendor.ldm.util import instantiate_from_config
        assert callable(instantiate_from_config)

    def test_import_distributions(self):
        from vibewarp.vendor.ldm.modules.distributions.distributions import DiagonalGaussianDistribution
        assert callable(DiagonalGaussianDistribution)

    def test_import_diffusion_util(self):
        from vibewarp.vendor.ldm.modules.diffusionmodules.util import GroupNorm32, timestep_embedding
        assert callable(timestep_embedding)

    def test_import_attention(self):
        from vibewarp.vendor.ldm.modules.attention import BasicTransformerBlock
        assert callable(BasicTransformerBlock)

    def test_import_openaimodel(self):
        from vibewarp.vendor.ldm.modules.diffusionmodules.openaimodel import (
            TimestepEmbedSequential, TimestepBlock
        )
        assert TimestepEmbedSequential is not None

    def test_timestep_embedding(self):
        from vibewarp.vendor.ldm.modules.diffusionmodules.util import timestep_embedding
        t = torch.tensor([1.0, 2.0, 3.0])
        emb = timestep_embedding(t, 64)
        assert emb.shape == (3, 64)


class TestSGM:
    def test_import_discretizer(self):
        from vibewarp.vendor.sgm.modules.diffusionmodules.discretizer import LegacyDDPMDiscretization
        assert callable(LegacyDDPMDiscretization)

    def test_import_util(self):
        from vibewarp.vendor.sgm.modules.diffusionmodules.util import GroupNorm32
        assert callable(GroupNorm32)

    def test_import_openaimodel(self):
        from vibewarp.vendor.sgm.modules.diffusionmodules.openaimodel import (
            TimestepEmbedSequential, TimestepBlock
        )
        assert TimestepEmbedSequential is not None


class TestCLDM:
    def test_import_model(self):
        from vibewarp.vendor.cldm.model import load_state_dict, create_model
        assert callable(load_state_dict)
        assert callable(create_model)
