"""Tests for the vendored controlmodel_ipadapter (PlugableIPAdapter).

Exercises the real vendored module on CPU with a synthetic SD1.5-style
adapter state dict and a minimal fake UNet built from real (vendored) ldm
CrossAttention blocks — verifying hook installation, the hacked attention
forward with IP contribution, and clean unhooking.
"""

import pytest
import torch
import torch.nn as nn

from vibewarp.core.model_loader import _ensure_vendor_on_path

_ensure_vendor_on_path()

from vibewarp.vendor.controlmodel_ipadapter import (  # noqa: E402
    PlugableIPAdapter,
    clear_all_ip_adapter,
    clip_vision_h_uc,
    clip_vision_vith_uc,
)

CROSS_DIM = 768   # SD1.5 cross-attention dim (read from '1.to_k_ip.weight')
CLIP_DIM = 1024   # ViT-H image_embeds dim
TOKENS = 4        # non-plus adapter context tokens
INNER = 64        # attention inner dim in the fake UNet
HEADS = 4


def make_sd15_adapter_state():
    """Synthetic non-plus SD1.5 IP-Adapter state dict (16 attn layers)."""
    torch.manual_seed(0)
    image_proj = {
        'proj.weight': torch.randn(TOKENS * CROSS_DIM, CLIP_DIM) * 0.02,
        'proj.bias': torch.zeros(TOKENS * CROSS_DIM),
        'norm.weight': torch.ones(CROSS_DIM),
        'norm.bias': torch.zeros(CROSS_DIM),
    }
    ip_adapter = {}
    for n in range(16):
        key = n * 2 + 1
        ip_adapter[f'{key}.to_k_ip.weight'] = torch.randn(INNER, CROSS_DIM) * 0.02
        ip_adapter[f'{key}.to_v_ip.weight'] = torch.randn(INNER, CROSS_DIM) * 0.02
    return {'image_proj': image_proj, 'ip_adapter': ip_adapter}


def make_fake_unet():
    """Minimal UNet skeleton matching the hook's block indexing.

    get_block(model, flag)[id][1].transformer_blocks[0].attn2 must be a
    (vendored) ldm CrossAttention — the same class the hook targets.
    """
    from ldm.modules.attention import CrossAttention

    def spatial():
        holder = nn.Module()
        blk = nn.Module()
        blk.attn2 = CrossAttention(
            query_dim=INNER, context_dim=CROSS_DIM,
            heads=HEADS, dim_head=INNER // HEADS,
        )
        holder.transformer_blocks = nn.ModuleList([blk])
        return holder

    unet = nn.Module()
    # input ids used: 1,2,4,5,7,8 → need 9 entries; output ids: 3..11 → 12
    unet.input_blocks = nn.ModuleList(
        [nn.ModuleList([nn.Identity(), spatial()]) for _ in range(9)])
    unet.output_blocks = nn.ModuleList(
        [nn.ModuleList([nn.Identity(), spatial()]) for _ in range(12)])
    unet.middle_block = nn.ModuleList([nn.Identity(), spatial()])
    return unet


class TestVendoredModule:
    def test_uncond_embeds_loaded(self):
        """The .data files must ship with the vendor dir and load on CPU."""
        assert isinstance(clip_vision_h_uc, torch.Tensor)
        assert isinstance(clip_vision_vith_uc, torch.Tensor)
        assert clip_vision_h_uc.device.type == 'cpu'

    def test_adapter_flags_sd15_nonplus(self):
        adapter = PlugableIPAdapter(make_sd15_adapter_state())
        assert adapter.is_plus is False
        assert adapter.is_full is False
        assert adapter.is_faceid is False
        assert adapter.sdxl is False
        assert adapter.ipadapter.clip_extra_context_tokens == TOKENS

    def test_image_embeds_shapes(self):
        adapter = PlugableIPAdapter(make_sd15_adapter_state())
        clip_out = {'image_embeds': torch.randn(1, CLIP_DIM)}
        cond, uncond = adapter.ipadapter.get_image_embeds(clip_out)
        assert cond.shape == (1, TOKENS, CROSS_DIM)
        assert uncond.shape == (1, TOKENS, CROSS_DIM)
        assert not torch.allclose(cond, uncond)


def sdpa_reference(attn, x, ctx):
    """Independent plain-attention reference (matches attn_forward_hacked
    minus the IP contribution). The vendored ldm CrossAttention.forward
    itself is CUDA-only on this box, so we can't call it on CPU."""
    b, s, d = x.shape
    h = attn.heads
    hd = d // h
    q, k, v = attn.to_q(x), attn.to_k(ctx), attn.to_v(ctx)
    q, k, v = [t.view(b, -1, h, hd).transpose(1, 2) for t in (q, k, v)]
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    out = out.transpose(1, 2).reshape(b, -1, h * hd)
    return attn.to_out(out)


class TestHookForwardUnhook:
    def teardown_method(self):
        clear_all_ip_adapter()

    def _hook(self, unet, adapter, weight=1.0, start=0.0, end=1.0):
        adapter.hook(
            model=unet,
            clip_vision_output={'image_embeds': torch.randn(1, CLIP_DIM)},
            weight=weight, start=start, end=end,
            weight_type='linear', embeds_scaling='V only',
        )

    def test_hook_replaces_forwards(self):
        adapter = PlugableIPAdapter(make_sd15_adapter_state())
        unet = make_fake_unet()
        self._hook(unet, adapter)
        attn = unet.input_blocks[1][1].transformer_blocks[0].attn2
        assert hasattr(attn, 'ipadapter_hacks')
        assert len(attn.ipadapter_hacks) == 1
        # 6 input + 9 output + 1 middle = 16 hooked blocks
        hooked = sum(
            1 for m in unet.modules() if getattr(m, 'ipadapter_hacks', None))
        assert hooked == 16

    def test_forward_adds_ip_contribution_and_unhook_restores(self):
        torch.manual_seed(1)
        adapter = PlugableIPAdapter(make_sd15_adapter_state())
        unet = make_fake_unet()
        attn = unet.input_blocks[1][1].transformer_blocks[0].attn2
        original_forward = attn.forward

        x = torch.randn(2, 8, INNER)
        ctx = torch.randn(2, 77, CROSS_DIM)
        with torch.no_grad():
            plain = sdpa_reference(attn, x, ctx)

        self._hook(unet, adapter, weight=1.0)
        # CFGDenoiser sets this per batch: 0 = uncond row, 1 = cond row
        unet.uc_mask_shape = torch.tensor([0.0, 1.0])
        with torch.no_grad():
            after = attn(x, context=ctx)

        assert after.shape == plain.shape
        assert not torch.allclose(plain, after, atol=1e-4)

        clear_all_ip_adapter()
        assert attn.forward == original_forward
        assert attn.ipadapter_hacks == []

    def test_zero_weight_matches_plain_attention(self):
        """weight=0 → IP contribution is zero; hacked forward must equal
        plain attention."""
        torch.manual_seed(2)
        adapter = PlugableIPAdapter(make_sd15_adapter_state())
        unet = make_fake_unet()
        attn = unet.input_blocks[1][1].transformer_blocks[0].attn2

        x = torch.randn(2, 8, INNER)
        ctx = torch.randn(2, 77, CROSS_DIM)

        self._hook(unet, adapter, weight=0.0)
        unet.uc_mask_shape = torch.tensor([0.0, 1.0])
        with torch.no_grad():
            hacked = attn(x, context=ctx)
            plain = sdpa_reference(attn, x, ctx)
        assert torch.allclose(hacked, plain, atol=1e-5)

    def test_start_end_gating_defaults_to_active(self):
        """Notebook never sets current_sampling_percent → getattr default 0.5.
        start=0.6 gates the hook OFF (0.5 < 0.6): output equals plain attention."""
        torch.manual_seed(3)
        adapter = PlugableIPAdapter(make_sd15_adapter_state())
        unet = make_fake_unet()
        attn = unet.input_blocks[1][1].transformer_blocks[0].attn2

        x = torch.randn(2, 8, INNER)
        ctx = torch.randn(2, 77, CROSS_DIM)

        self._hook(unet, adapter, weight=1.0, start=0.6, end=1.0)
        unet.uc_mask_shape = torch.tensor([0.0, 1.0])
        with torch.no_grad():
            gated = attn(x, context=ctx)
            plain = sdpa_reference(attn, x, ctx)
        assert torch.allclose(gated, plain, atol=1e-5)
