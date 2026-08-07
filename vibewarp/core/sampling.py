"""Sampling utilities - noise schedules, sigma computation, diffusion helpers."""

import math

import numpy as np
import torch
import torch.nn.functional as F


def append_dims(x, n):
    """Append dimensions to tensor x until it has n dimensions."""
    return x[(Ellipsis, *(None,) * (n - x.ndim))]


def expand_to_planes(x, shape):
    """Expand a 1D tensor to match a given spatial shape."""
    return append_dims(x, len(shape)).repeat([1, 1, *shape[2:]])


def alpha_sigma_to_t(alpha, sigma):
    """Convert (alpha, sigma) noise schedule to continuous timestep t."""
    return torch.atan2(sigma, alpha) * 2 / math.pi


def t_to_alpha_sigma(t):
    """Convert continuous timestep t to (alpha, sigma) noise schedule."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


def get_premature_sigma_min(steps, sigma_min, sigma_max, rho=7.0, stop_pct=1.0):
    """Compute a premature sigma_min for early stopping in karras schedule.

    Args:
        steps: total number of steps
        sigma_min: minimum sigma
        sigma_max: maximum sigma
        rho: rho parameter for karras schedule
        stop_pct: fraction of steps to actually run (1.0 = full schedule)

    Returns:
        Adjusted sigma_min value.
    """
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    ramp = np.linspace(0, 1, steps)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    stop_idx = int(len(sigmas) * stop_pct)
    if stop_idx >= len(sigmas):
        return sigma_min
    return float(sigmas[stop_idx])


def high_frequency_loss(image1, image2):
    """Compute loss between high-frequency components of two images.

    Uses Sobel filters to extract edges, then computes MSE between them.

    Args:
        image1: tensor (B, C, H, W)
        image2: tensor (B, C, H, W)

    Returns:
        Scalar loss tensor.
    """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=image1.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32, device=image1.device).view(1, 1, 3, 3)

    # Apply per-channel
    loss = 0
    for c in range(image1.shape[1]):
        ch1 = image1[:, c:c + 1]
        ch2 = image2[:, c:c + 1]

        hf1_x = F.conv2d(ch1, sobel_x, padding=1)
        hf1_y = F.conv2d(ch1, sobel_y, padding=1)
        hf2_x = F.conv2d(ch2, sobel_x, padding=1)
        hf2_y = F.conv2d(ch2, sobel_y, padding=1)

        loss += F.mse_loss(hf1_x, hf2_x) + F.mse_loss(hf1_y, hf2_y)

    return loss / image1.shape[1]


def torch_interp1d(x, xp, fp):
    """1D linear interpolation for tensors (like np.interp).

    Args:
        x: query points
        xp: reference x coordinates (sorted)
        fp: reference y values

    Returns:
        Interpolated values at x.
    """
    slopes = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    idx = torch.searchsorted(xp[1:], x)
    idx = idx.clamp(0, len(slopes) - 1)
    return fp[idx] + slopes[idx] * (x - xp[idx])


# ---- AnimateDiff context-aware sampling ----

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ---- Per-step spatial tiled sampling ----

def compute_tile_starts(total_size: int, n_tiles: int, overlap: int) -> Tuple[List[int], int]:
    """Return evenly distributed tile starts and their common size."""
    if n_tiles <= 1 or total_size <= 1:
        return [0], total_size
    tile_size = min(
        math.ceil((total_size + (n_tiles - 1) * max(0, overlap)) / n_tiles),
        total_size,
    )
    if n_tiles == 2:
        return [0, max(0, total_size - tile_size)], tile_size
    stride = (total_size - tile_size) / (n_tiles - 1)
    return [int(round(i * stride)) for i in range(n_tiles)], tile_size


def _tile_window_1d(
    size: int,
    fade_left: int,
    fade_right: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Trapezoidal blend weights with non-zero samples at tile boundaries."""
    window = torch.ones(size, dtype=dtype, device=device)
    if fade_left > 0:
        count = min(fade_left, size)
        window[:count] = torch.linspace(
            0, 1, count + 2, dtype=dtype, device=device)[1:-1]
    if fade_right > 0:
        count = min(fade_right, size)
        window[-count:] = torch.linspace(
            1, 0, count + 2, dtype=dtype, device=device)[1:-1]
    return window


def build_tile_specs(
    height: int,
    width: int,
    tiles_h: int,
    tiles_w: int,
    overlap_h: float,
    overlap_w: float,
) -> List[Tuple[int, int, int, int, torch.Tensor]]:
    """Build spatial tile bounds and CPU float32 blend windows.

    Overlap values are percentages of the resulting tile dimensions.
    """
    def overlap_px(total: int, count: int, percent: float) -> int:
        return 0 if count <= 1 else max(0, round(math.ceil(total / count) * percent / 100))

    overlap_h_px = overlap_px(height, tiles_h, overlap_h)
    overlap_w_px = overlap_px(width, tiles_w, overlap_w)
    h_starts, tile_h = compute_tile_starts(height, tiles_h, overlap_h_px)
    w_starts, tile_w = compute_tile_starts(width, tiles_w, overlap_w_px)

    # Recompute once from the real tile size, matching the configured percentage.
    h_starts, tile_h = compute_tile_starts(
        height, tiles_h, round(tile_h * overlap_h / 100))
    w_starts, tile_w = compute_tile_starts(
        width, tiles_w, round(tile_w * overlap_w / 100))

    specs = []
    for hi, top in enumerate(h_starts):
        bottom = min(top + tile_h, height)
        fade_top = max(0, h_starts[hi - 1] + tile_h - top) if hi else 0
        fade_bottom = (
            max(0, top + tile_h - h_starts[hi + 1])
            if hi + 1 < len(h_starts) else 0)
        for wi, left in enumerate(w_starts):
            right = min(left + tile_w, width)
            fade_left = max(0, w_starts[wi - 1] + tile_w - left) if wi else 0
            fade_right = (
                max(0, left + tile_w - w_starts[wi + 1])
                if wi + 1 < len(w_starts) else 0)
            win_h = _tile_window_1d(
                bottom - top, fade_top, fade_bottom,
                dtype=torch.float32, device=torch.device('cpu'))
            win_w = _tile_window_1d(
                right - left, fade_left, fade_right,
                dtype=torch.float32, device=torch.device('cpu'))
            specs.append((top, bottom, left, right, win_h[:, None] * win_w[None, :]))
    return specs


def tile_count_for_size(
    total_size: int,
    tile_size: int,
    split_threshold_percent: float = 0.0,
) -> int:
    """How many tiles a dimension needs.

    Plain `ceil(total / tile)` splits on the tiniest leftover: 1080px against a 1024px
    target asks for 2 tiles. And because the tile size is DERIVED from the count
    (compute_tile_starts), those two come out ~640px each — you request 1024px tiles,
    silently get 640px ones, and pay two model evals instead of one.

    So only add a tile when the leftover past the ones you already have is worth it:

        n = ceil(total / tile - threshold)

    `split_threshold_percent` is that leftover, as a fraction of a tile. At 20%, a 1024px
    target absorbs up to ~1229px in a single tile:
      1080 -> 1 tile    1300 -> 2 tiles
    and with a 512px target:
      1024 -> 2 tiles   1080 -> 2 tiles   1920 -> 4 tiles

    `0` reduces exactly to the old ceil(): split as soon as the tile is exceeded at all.

    NOTE this is deliberately independent of the overlap. Overlap is the seam cross-fade;
    tying the count to it (as a first attempt did) makes tiles *multiply* — with 35%
    overlap a 1920px side needed 5 tiles to be covered by "512px" tiles, which is slower
    than the problem it was meant to fix.
    """
    tile_size = max(1, int(tile_size))
    total_size = max(1, int(total_size))
    slack = max(float(split_threshold_percent), 0.0) / 100.0
    return max(1, math.ceil(total_size / tile_size - slack))


def _crop_spatial_tensor(
    value: Any,
    bounds: Tuple[int, int, int, int],
    full_size: Tuple[int, int],
) -> Any:
    """Crop a spatial tensor at latent resolution or an integer multiple of it."""
    if not torch.is_tensor(value) or value.ndim < 2:
        return value
    top, bottom, left, right = bounds
    height, width = full_size
    value_h, value_w = value.shape[-2:]
    if value_h % height or value_w % width:
        return value
    scale_h, scale_w = value_h // height, value_w // width
    if scale_h < 1 or scale_w < 1:
        return value
    return value[..., top * scale_h:bottom * scale_h,
                 left * scale_w:right * scale_w].contiguous()


def _resize_spatial_tensor(
    value: Any,
    full_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> Any:
    """Resize a tensor whose trailing axes track the sampler latent spatially."""
    if not torch.is_tensor(value) or value.ndim < 3:
        return value
    height, width = full_size
    value_h, value_w = value.shape[-2:]
    if value_h % height or value_w % width:
        return value
    scale_h, scale_w = value_h // height, value_w // width
    if scale_h < 1 or scale_w < 1:
        return value
    flat = value.float().reshape(-1, 1, value_h, value_w)
    resized = F.interpolate(
        flat,
        size=(target_size[0] * scale_h, target_size[1] * scale_w),
        mode='bilinear',
        align_corners=False,
    )
    return resized.reshape(
        *value.shape[:-2], resized.shape[-2], resized.shape[-1],
    ).to(dtype=value.dtype)


class TiledModelFn:
    """Evaluate a denoiser in spatial tiles and blend its full prediction.

    Wrapping the denoiser rather than the sampler means this runs for every model
    evaluation, including samplers that evaluate twice per diffusion step.
    """

    def __init__(
        self,
        model_fn: Callable,
        *,
        tiles_h: int = 2,
        tiles_w: int = 2,
        tile_size: Optional[int] = None,
        overlap_h: float = 25,
        overlap_w: float = 25,
        split_threshold: float = 0.0,
        sd_model: Any = None,
        scale_schedule: Optional[Dict[int, float]] = None,
        enable_tiling: bool = True,
        min_scale_size: int = 0,
    ):
        self.model_fn = model_fn
        self.tiles_h = max(1, int(tiles_h))
        self.tiles_w = max(1, int(tiles_w))
        self.tile_size = max(1, int(tile_size)) if tile_size is not None else None
        self.overlap_h = float(overlap_h)
        self.overlap_w = float(overlap_w)
        # How far a dimension may exceed tile_size before another tile is worth it.
        self.split_threshold = float(split_threshold)
        self.sd_model = sd_model
        self.scale_schedule = dict(sorted((scale_schedule or {0: 100.0}).items()))
        self.enable_tiling = bool(enable_tiling)
        self.min_scale_size = max(0, int(min_scale_size))
        self._sigma_schedule: Sequence[float] = ()
        self._specs: Dict[Tuple[int, int], Sequence[Tuple[int, int, int, int, torch.Tensor]]] = {}

    def set_sigma_schedule(self, sigmas: torch.Tensor) -> None:
        """Bind schedule step numbers to the sigma range used by this sample."""
        self._sigma_schedule = tuple(float(s) for s in sigmas.detach().cpu().flatten())

    def _step_for_sigma(self, sigma: torch.Tensor) -> int:
        if not self._sigma_schedule:
            return 0
        current = float(sigma.detach().float().flatten()[0].cpu())
        # Schedules descend. An intermediate second-order evaluation belongs to
        # the interval that contains it; an exact next-step sigma belongs to that
        # next step.
        for index in range(len(self._sigma_schedule) - 1):
            next_sigma = self._sigma_schedule[index + 1]
            tolerance = max(abs(next_sigma), 1.0) * 1e-6
            if current > next_sigma + tolerance:
                return index
        return max(0, len(self._sigma_schedule) - 2)

    def _scale_for_sigma(self, sigma: torch.Tensor) -> float:
        step = self._step_for_sigma(sigma)
        scale = next(iter(self.scale_schedule.values()))
        for key, value in self.scale_schedule.items():
            if key > step:
                break
            scale = value
        return float(scale)

    def _scaled_size(self, height: int, width: int, scale: float) -> Tuple[int, int]:
        ratio = scale / 100.0
        if self.min_scale_size:
            final_longest_px = max(height, width) * 8
            ratio = max(ratio, self.min_scale_size / final_longest_px)
        ratio = min(ratio, 1.0)

        def aligned(value: int) -> int:
            # SD UNets downsample the latent internally three times. Keeping the
            # temporary axes on an 8-latent-unit boundary avoids skip-connection
            # shape mismatches at fractional scales.
            return min(value, max(8, int(round(value * ratio / 8.0)) * 8))

        return aligned(height), aligned(width)

    def _call_scaled(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        scale: float,
        kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """Whole-frame coarse pass. Spatial tiling is never used below 100%."""
        full_size = x.shape[-2:]
        target_size = self._scaled_size(*full_size, scale)
        scaled_x = F.interpolate(
            x.float(), size=target_size, mode='bilinear', align_corners=False,
        ).to(dtype=x.dtype)
        spatial_kwargs = {'image_cond', 'c_concat', 'denoise_mask', 'mask'}
        scaled_kwargs = {
            key: (_resize_spatial_tensor(value, full_size, target_size)
                  if key in spatial_kwargs else value)
            for key, value in kwargs.items()
        }
        original_hints = (
            dict(getattr(self.sd_model, '_cn_hints', {}) or {})
            if self.sd_model is not None else {})
        try:
            if self.sd_model is not None and original_hints:
                self.sd_model._cn_hints = {
                    key: _resize_spatial_tensor(hint, full_size, target_size)
                    for key, hint in original_hints.items()
                }
            prediction = self.model_fn(scaled_x, sigma, **scaled_kwargs)
        finally:
            if self.sd_model is not None and original_hints:
                self.sd_model._cn_hints = original_hints
        if prediction.shape != scaled_x.shape:
            raise RuntimeError(
                f"Multiscale denoiser returned {tuple(prediction.shape)} for "
                f"scaled latent {tuple(scaled_x.shape)}")
        return F.interpolate(
            prediction.float(), size=full_size, mode='bilinear', align_corners=False,
        ).to(dtype=x.dtype)

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.ndim != 4:
            return self.model_fn(x, sigma, **kwargs)

        height, width = x.shape[-2:]
        scale = self._scale_for_sigma(sigma)
        if scale < 100.0:
            return self._call_scaled(x, sigma, scale, kwargs)
        if not self.enable_tiling:
            return self.model_fn(x, sigma, **kwargs)

        tiles_h, tiles_w = self.tiles_h, self.tiles_w
        if self.tile_size is not None:
            # split_threshold decides whether the leftover is worth another tile.
            # The overlap does NOT feed the count -- see tile_count_for_size.
            tiles_h = tile_count_for_size(height, self.tile_size, self.split_threshold)
            tiles_w = tile_count_for_size(width, self.tile_size, self.split_threshold)
        if tiles_h == 1 and tiles_w == 1:
            return self.model_fn(x, sigma, **kwargs)
        specs = self._specs.setdefault(
            (height, width),
            build_tile_specs(
                height, width, tiles_h, tiles_w,
                self.overlap_h, self.overlap_w),
        )
        accum = torch.zeros_like(x, dtype=torch.float32)
        weights = torch.zeros(
            (1, 1, height, width), device=x.device, dtype=torch.float32)
        original_hints = (
            dict(getattr(self.sd_model, '_cn_hints', {}) or {})
            if self.sd_model is not None else {})

        try:
            for top, bottom, left, right, window_cpu in specs:
                bounds = (top, bottom, left, right)
                spatial_kwargs = {'image_cond', 'c_concat', 'denoise_mask', 'mask'}
                tile_kwargs = {
                    key: (_crop_spatial_tensor(value, bounds, (height, width))
                          if key in spatial_kwargs else value)
                    for key, value in kwargs.items()
                }
                if self.sd_model is not None and original_hints:
                    self.sd_model._cn_hints = {
                        key: _crop_spatial_tensor(hint, bounds, (height, width))
                        for key, hint in original_hints.items()
                    }
                tile = x[..., top:bottom, left:right].contiguous()
                prediction = self.model_fn(tile, sigma, **tile_kwargs)
                if prediction.shape != tile.shape:
                    raise RuntimeError(
                        f"Tiled denoiser returned {tuple(prediction.shape)} for tile "
                        f"{tuple(tile.shape)}")
                window = window_cpu.to(device=x.device, dtype=torch.float32)[None, None]
                accum[..., top:bottom, left:right] += prediction.float() * window
                weights[..., top:bottom, left:right] += window
        finally:
            if self.sd_model is not None and original_hints:
                self.sd_model._cn_hints = original_hints

        if not bool((weights > 0).all()):
            raise RuntimeError("Tiled sampler left part of the latent uncovered")
        return (accum / weights).to(dtype=x.dtype)

# NOTE: AnimateDiffSamplerWrapper / get_context_sampler lived here and were dead
# code — they wrapped the SINGLE-FRAME sampler, so the latent batch was always 1
# and their first line (`if batch_size <= 1: return model(...)`) short-circuited
# every call. AnimateDiff's context windows now run inside
# CFGDenoiser._forward_animatediff, over a real batch of frames.
