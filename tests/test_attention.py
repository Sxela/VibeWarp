"""Tests for the SDPA-based CrossAttention drop-in in vendored ldm."""

import pytest
import torch

from ldm.modules.attention import (
    CrossAttention,
    SDPCrossAttention,
    BasicTransformerBlock,
    SDPA_IS_AVAILABLE,
)


def test_sdpa_available_in_torch_2():
    """PyTorch 2.0+ must expose F.scaled_dot_product_attention."""
    assert SDPA_IS_AVAILABLE is True


class TestSDPCrossAttentionEquivalence:
    """SDPA attention should produce numerically-close output to the split-attention reference.

    These use the same random weights copied across both implementations.
    """

    def _copy_weights(self, src: CrossAttention, dst: SDPCrossAttention):
        dst.to_q.weight.data.copy_(src.to_q.weight.data)
        dst.to_k.weight.data.copy_(src.to_k.weight.data)
        dst.to_v.weight.data.copy_(src.to_v.weight.data)
        dst.to_out[0].weight.data.copy_(src.to_out[0].weight.data)
        dst.to_out[0].bias.data.copy_(src.to_out[0].bias.data)

    def test_self_attention_matches(self):
        if not torch.cuda.is_available():
            pytest.skip("vanilla CrossAttention queries CUDA memory — needs CUDA")
        torch.manual_seed(0)
        dim, heads, dim_head = 320, 8, 40
        vanilla = CrossAttention(query_dim=dim, heads=heads, dim_head=dim_head).eval().cuda()
        sdpa = SDPCrossAttention(query_dim=dim, heads=heads, dim_head=dim_head).eval().cuda()
        self._copy_weights(vanilla, sdpa)

        x = torch.randn(2, 16, dim, device='cuda')
        with torch.no_grad():
            out_v = vanilla(x)
            out_s = sdpa(x)

        assert out_v.shape == out_s.shape == (2, 16, dim)
        assert torch.allclose(out_v, out_s, atol=1e-4, rtol=1e-3)

    def test_cross_attention_matches(self):
        if not torch.cuda.is_available():
            pytest.skip("vanilla CrossAttention queries CUDA memory — needs CUDA")
        torch.manual_seed(1)
        query_dim, ctx_dim, heads, dim_head = 320, 768, 8, 40
        vanilla = CrossAttention(
            query_dim=query_dim, context_dim=ctx_dim, heads=heads, dim_head=dim_head,
        ).eval().cuda()
        sdpa = SDPCrossAttention(
            query_dim=query_dim, context_dim=ctx_dim, heads=heads, dim_head=dim_head,
        ).eval().cuda()
        self._copy_weights(vanilla, sdpa)

        x = torch.randn(1, 64, query_dim, device='cuda')
        ctx = torch.randn(1, 77, ctx_dim, device='cuda')
        with torch.no_grad():
            out_v = vanilla(x, context=ctx)
            out_s = sdpa(x, context=ctx)

        assert out_v.shape == out_s.shape
        assert torch.allclose(out_v, out_s, atol=1e-4, rtol=1e-3)


class TestDefaultAttentionSelection:
    """SDPA is the normal default; parity mode matches the notebook."""

    def test_uses_sdpa_by_default(self, monkeypatch):
        # Without xformers, normal renders use the faster SDPA backend.
        from ldm.modules.attention import XFORMERS_IS_AVAILBLE
        if XFORMERS_IS_AVAILBLE:
            pytest.skip("xformers is available; SDPA selection bypassed")
        monkeypatch.delenv('VIBEWARP_PARITY_MODE', raising=False)
        block = BasicTransformerBlock(dim=320, n_heads=8, d_head=40)
        assert isinstance(block.attn1, SDPCrossAttention)
        assert isinstance(block.attn2, SDPCrossAttention)

    def test_parity_mode_uses_notebook_attention(self, monkeypatch):
        from ldm.modules.attention import XFORMERS_IS_AVAILBLE
        monkeypatch.setenv('VIBEWARP_PARITY_MODE', '1')
        block = BasicTransformerBlock(dim=320, n_heads=8, d_head=40)
        assert isinstance(block.attn1, CrossAttention)
        assert isinstance(block.attn2, CrossAttention)
