"""Loss functions for diffusion guidance."""

import torch
import torch.nn.functional as F


def spherical_dist_loss(x, y):
    """Spherical distance loss between normalized vectors."""
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input):
    """Penalize values outside [-1, 1]."""
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


def softcap(arr, thresh=0.8, q=0.95):
    """Softly compress values beyond ±thresh (notebook cell 20 `softcap`).

    Applied to the decoded image tensor ([-1, 1] range) before the final
    clamp — reduces the cross-frame feedback loop from oversaturated values.

    cap = quantile(|arr|, q); values beyond ±thresh are linearly remapped
    with ratio (1-thresh)/(cap-thresh), so |arr| == cap lands exactly at ±1.
    Values within ±thresh are untouched. Notebook-exact (including no guard
    for cap <= thresh — the notebook has none either).
    """
    cap = torch.quantile(arr.abs().float(), q)
    cap_ratio = (1 - thresh) / (cap - thresh)
    arr = torch.where(arr > thresh, thresh + (arr - thresh) * cap_ratio, arr)
    arr = torch.where(arr < -thresh, -thresh + (arr + thresh) * cap_ratio, arr)
    return arr
