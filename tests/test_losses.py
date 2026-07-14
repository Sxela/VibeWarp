"""Tests for vibewarp.core.losses."""

import torch

from vibewarp.core.losses import range_loss, softcap, spherical_dist_loss, tv_loss


class TestSoftcap:
    """Notebook-exact softcap: values within ±thresh untouched; beyond,
    linearly remapped so |x| == quantile(|arr|, q) lands at ±1."""

    def test_within_threshold_untouched(self):
        arr = torch.tensor([0.1, -0.5, 0.85, -0.89])
        out = softcap(arr, thresh=0.9, q=1.0)
        assert torch.allclose(out, arr)

    def test_max_maps_to_one(self):
        # cap = max(|arr|) = 1.5 (q=1); ratio = (1-0.9)/(1.5-0.9) = 1/6
        # 1.5 → 0.9 + 0.6/6 = 1.0
        arr = torch.tensor([0.5, 1.5])
        out = softcap(arr, thresh=0.9, q=1.0)
        assert torch.isclose(out[0], torch.tensor(0.5))
        assert torch.isclose(out[1], torch.tensor(1.0))

    def test_linear_compression_between_thresh_and_cap(self):
        # 1.2 → 0.9 + 0.3 * (1/6) = 0.95
        arr = torch.tensor([1.2, 1.5])
        out = softcap(arr, thresh=0.9, q=1.0)
        assert torch.isclose(out[0], torch.tensor(0.95))

    def test_negative_symmetric(self):
        arr = torch.tensor([-1.5, 1.5])
        out = softcap(arr, thresh=0.9, q=1.0)
        assert torch.isclose(out[0], torch.tensor(-1.0))
        assert torch.isclose(out[1], torch.tensor(1.0))

    def test_quantile_below_one(self):
        # q=0.5 on [0, 1.0, 2.0]: cap = 1.0; ratio = (1-0.9)/(1.0-0.9) = 1.0
        # → values above thresh keep their distance (identity mapping)
        arr = torch.tensor([0.0, 1.0, 2.0])
        out = softcap(arr, thresh=0.9, q=0.5)
        assert torch.allclose(out, arr)

    def test_preserves_shape(self):
        arr = torch.randn(3, 64, 64) * 2
        out = softcap(arr, thresh=0.9, q=1.0)
        assert out.shape == arr.shape

    def test_monotonic(self):
        arr = torch.linspace(-3, 3, 101)
        out = softcap(arr, thresh=0.9, q=1.0)
        assert (out[1:] >= out[:-1]).all()

    def test_config_and_settings(self, tmp_path):
        import json
        from vibewarp.config import DiffusionConfig
        from vibewarp.settings import load_warpfusion_settings
        c = DiffusionConfig()
        assert c.do_softcap is False
        assert c.softcap_thresh == 0.9
        assert c.softcap_q == 1.0
        p = tmp_path / 's.txt'
        p.write_text(json.dumps({'do_softcap': True, 'softcap_thresh': 0.8,
                                 'softcap_q': 0.95}))
        cfg = load_warpfusion_settings(str(p))
        assert cfg['diffusion']['do_softcap'] is True
        assert cfg['diffusion']['softcap_thresh'] == 0.8
        assert cfg['diffusion']['softcap_q'] == 0.95


class TestSphericalDistLoss:
    def test_identical_vectors(self):
        x = torch.randn(4, 128)
        loss = spherical_dist_loss(x, x)
        assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-5)

    def test_opposite_vectors(self):
        x = torch.randn(4, 128)
        loss = spherical_dist_loss(x, -x)
        # Should be close to pi^2/2 ≈ 4.935
        assert (loss > 4.0).all()

    def test_output_shape(self):
        x = torch.randn(8, 64)
        y = torch.randn(8, 64)
        loss = spherical_dist_loss(x, y)
        assert loss.shape == (8,)


class TestTvLoss:
    def test_constant_image_zero_loss(self):
        img = torch.ones(1, 3, 32, 32)
        loss = tv_loss(img)
        assert torch.allclose(loss, torch.zeros_like(loss))

    def test_noisy_image_positive_loss(self):
        img = torch.randn(1, 3, 32, 32)
        loss = tv_loss(img)
        assert (loss > 0).all()

    def test_output_shape(self):
        img = torch.randn(4, 3, 16, 16)
        loss = tv_loss(img)
        assert loss.shape == (4,)


class TestRangeLoss:
    def test_in_range_zero_loss(self):
        x = torch.zeros(1, 3, 8, 8)
        loss = range_loss(x)
        assert torch.allclose(loss, torch.zeros_like(loss))

    def test_out_of_range_positive_loss(self):
        x = torch.ones(1, 3, 8, 8) * 5.0
        loss = range_loss(x)
        assert (loss > 0).all()
