"""GPU PDF color transfer.

Direct port of the WarpFusion notebook's `pdf_transfer_torch` monkey-patch
(cell at refs/notebook_dump.txt lines 4648-4768). Uses `torch.histc`,
`torch.cumsum`, and a custom linear-interp helper — exact numerical parity
with the original pipeline.
"""

from typing import Optional

import numpy as np
import torch


# Cache rotation matrices as torch tensors on GPU — matches notebook's
# `PT.rotation_matrices_torch = [torch.from_numpy(o).float().cuda() for o in ...]`.
_ROTATIONS: Optional[list] = None
_ROTATIONS_DEVICE: Optional[str] = None


def _get_rotations_torch(device: torch.device) -> list:
    """Load optimal rotation matrices as (3,3) tensors on `device`."""
    global _ROTATIONS, _ROTATIONS_DEVICE
    if _ROTATIONS is not None and _ROTATIONS_DEVICE == str(device):
        return _ROTATIONS
    from vibewarp.vendor.python_color_transfer.utils import Rotations
    mats = Rotations.optimal_rotations()
    _ROTATIONS = [torch.from_numpy(m).float().to(device) for m in mats]
    _ROTATIONS_DEVICE = str(device)
    return _ROTATIONS


def _torch_interp1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """1D linear interpolation, matching numpy.interp semantics."""
    idxs = torch.searchsorted(xp, x)
    idxs = idxs.clamp(1, len(xp) - 1)
    left = xp[idxs - 1]
    right = xp[idxs]
    denom = (right - left).clamp_min(1e-12)
    alpha = (x - left) / denom
    return fp[idxs - 1] + alpha * (fp[idxs] - fp[idxs - 1])


def _pdf_transfer_1d_torch(
    arr_in: torch.Tensor,
    arr_ref: torch.Tensor,
    n: int = 300,
    eps: float = 1e-6,
) -> torch.Tensor:
    """1-D PDF transfer — direct torch port of the numpy reference."""
    arr = torch.cat((arr_in, arr_ref))
    min_v = (arr.min() - eps).item()
    max_v = (arr.max() + eps).item()
    xs = torch.linspace(min_v, max_v, steps=n + 1, device=arr.device)

    hist_in = torch.histc(arr_in, bins=n, min=min_v, max=max_v)
    hist_ref = torch.histc(arr_ref, bins=n, min=min_v, max=max_v)
    xs = xs[:-1]

    cum_in = torch.cumsum(hist_in, dim=0)
    cum_ref = torch.cumsum(hist_ref, dim=0)
    d_in = cum_in / cum_in[-1].clamp_min(eps)
    d_ref = cum_ref / cum_ref[-1].clamp_min(eps)

    t_d_in = _torch_interp1d(d_in, d_ref, xs)
    t_d_in = torch.where(d_in <= d_ref[0], torch.tensor(min_v, device=arr.device), t_d_in)
    t_d_in = torch.where(d_in >= d_ref[-1], torch.tensor(max_v, device=arr.device), t_d_in)

    arr_out = _torch_interp1d(arr_in, xs, t_d_in)
    return arr_out


def _pdf_transfer_nd_torch(
    arr_in: torch.Tensor,
    arr_ref: torch.Tensor,
    rotations: list,
    step_size: float = 1.0,
) -> torch.Tensor:
    """N-D PDF transfer via rotations — matches numpy reference."""
    arr_out = arr_in.clone()
    for R in rotations:
        rot_in = R @ arr_out
        rot_ref = R @ arr_ref

        rot_out = torch.zeros_like(rot_in)
        for i in range(rot_out.shape[0]):
            rot_out[i] = _pdf_transfer_1d_torch(rot_in[i], rot_ref[i])

        delta = rot_out - rot_in
        arr_out = arr_out + step_size * (R.t() @ delta)
    return arr_out


@torch.inference_mode()
def pdf_transfer_gpu(
    img_arr_in: np.ndarray,
    img_arr_ref: np.ndarray,
    regrain: bool = False,
    device: Optional[str] = None,
) -> np.ndarray:
    """GPU PDF color transfer. Drop-in for ColorTransfer.pdf_transfer (no regrain).

    Args:
        img_arr_in: BGR uint8 input image, shape (H, W, 3).
        img_arr_ref: BGR uint8 reference image, shape (H, W, 3).
        regrain: ignored (CPU-only regrain path lives in the numpy vendor module).
        device: 'cuda' or 'cpu'. Defaults to 'cuda' when available.

    Returns:
        BGR uint8 transferred image, shape (H, W, 3).
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(device)

    t_in = torch.from_numpy(img_arr_in).to(dev, dtype=torch.float32) / 255.0
    t_ref = torch.from_numpy(img_arr_ref).to(dev, dtype=torch.float32) / 255.0

    h, w, c = t_in.shape
    assert c == 3, f"expected 3 channels, got {c}"

    reshape_in = t_in.view(-1, c).permute(1, 0).contiguous()
    reshape_ref = t_ref.view(-1, c).permute(1, 0).contiguous()

    rotations = _get_rotations_torch(dev)
    reshape_out = _pdf_transfer_nd_torch(reshape_in, reshape_ref, rotations)

    reshape_out.clamp_(0.0, 1.0)
    out = (255.0 * reshape_out).to(torch.uint8)
    img_out = out.permute(1, 0).view(h, w, c).detach().cpu().numpy()
    return img_out
