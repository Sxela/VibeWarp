"""Attention dispatch preserving OpenAI's legacy head packing and gradients."""
import torch
import torch.nn.functional as F
from torch import nn


class PixelAttention(nn.Module):
    def __init__(self, heads, *, legacy=True, backend='sdpa'):
        super().__init__()
        self.n_heads, self.legacy, self.backend = heads, legacy, backend

    def forward(self, qkv):
        batch, width, length = qkv.shape
        dim = width // (3 * self.n_heads)
        if self.legacy:
            q, k, v = qkv.reshape(batch, self.n_heads, 3 * dim, length).split(dim, dim=2)
        else:
            q, k, v = [x.reshape(batch, self.n_heads, dim, length) for x in qkv.chunk(3, dim=1)]
        q, k, v = [x.transpose(-1, -2).contiguous() for x in (q, k, v)]
        needs_grad = torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v))
        if self.backend == 'flash' and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
            try:
                from flash_attn import flash_attn_func
            except ImportError:
                pass
            else:
                out = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=0.0)
                return out.transpose(1, 2).transpose(-1, -2).reshape(batch, -1, length)
        if self.backend == 'sage' and not needs_grad and q.is_cuda:
            try:
                from sageattention import sageattn
            except ImportError:
                pass
            else:
                out = sageattn(q, k, v, is_causal=False)
                return out.transpose(-1, -2).reshape(batch, -1, length)
        # SDPA selects a gradient-capable fused kernel or its portable math path.
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return out.transpose(-1, -2).reshape(batch, -1, length)


def configure_attention(model, backend='sdpa'):
    from vibewarp.vendor.guided_diffusion.unet import QKVAttention, QKVAttentionLegacy
    if backend == 'manual':
        return
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, (QKVAttention, QKVAttentionLegacy)):
                setattr(module, name, PixelAttention(child.n_heads,
                    legacy=isinstance(child, QKVAttentionLegacy), backend=backend))
