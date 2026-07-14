"""FreeU (notebook cell 23 — Fourier_filter / apply_freeu in cldm_forward).

Scales UNet backbone features and Fourier-dampens skip connections in the
output blocks: at the 1280-channel stage the first 640 backbone channels are
multiplied by b1 and the skip is low-frequency-scaled by s1; at 640 channels,
b2 / s2 on the first 320. `apply_freeu_after_control` chooses whether this
happens before or after the ControlNet residual is added to the skip.

Notebook parity note: the notebook implements FreeU inside its control_multi
UNet forward (cldm_forward / sdxl_cn_forward). VibeWarp uses that unified
forward for every render, including an empty ControlNet model dictionary, so
`diffusion_model._freeu` also works when no ControlNet is selected.
"""

from typing import Any, Optional

import torch
import torch.fft as fft


def Fourier_filter(x: torch.Tensor, threshold: int, scale: float) -> torch.Tensor:
    """Notebook-exact Fourier low-frequency scaling (mask built on x.device
    instead of hardcoded .cuda() — behavior-identical)."""
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered


def apply_freeu(h: torch.Tensor, hs: torch.Tensor,
                b1: float = 1.2, b2: float = 1.4,
                s1: float = 0.9, s2: float = 0.2):
    """Notebook-exact apply_freeu: modulate backbone (h) and skip (hs)
    features by channel stage. Returns (h, hs)."""
    if h.shape[1] == 1280:
        h[:, :640] = h[:, :640] * b1
        hs = Fourier_filter(hs.float(), threshold=1, scale=s1)
    if h.shape[1] == 640:
        h[:, :320] = h[:, :320] * b2
        hs = Fourier_filter(hs.float(), threshold=1, scale=s2)
    return h, hs


def set_freeu(diffusion_model: Any, config) -> None:
    """Enable FreeU on a (ControlNet-patched) UNet from a FreeUConfig."""
    diffusion_model._freeu = {
        'after_control': config.apply_freeu_after_control,
        'b1': config.b1, 'b2': config.b2,
        's1': config.s1, 's2': config.s2,
    }


def clear_freeu(diffusion_model: Any) -> None:
    if hasattr(diffusion_model, '_freeu'):
        del diffusion_model._freeu
