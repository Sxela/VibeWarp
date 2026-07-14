"""AnimateDiff motion module injection, ejection, and context scheduling.

Extracted from notebook cells handling AnimateDiff integration:
- inject_motion_module_to_unet / eject_motion_module_from_unet
- make_ctx_sched (context window scheduling for temporal attention)
- load_motion_module (MotionWrapper loading from checkpoint)

The motion modules are injected into the UNet's transformer blocks to add
temporal attention across frames. During sampling, frames are processed in
overlapping context windows to maintain temporal coherence.
"""

import copy
import gc
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from vibewarp.config import AnimateDiffConfig


# ---- Motion module loading ----

def load_motion_module(
    path: str,
    model_version: str = 'control_multi_v15',
    device: str = 'cuda',
) -> Any:
    """Load an AnimateDiff MotionWrapper from checkpoint.

    Uses the vendored motion-module implementation
    (vibewarp/vendor/animatediff_mm.py — the notebook's MotionWrapper with
    SD1.5 / SDXL / HotshotXL / v3 support). Failure raises — no silent
    fallback.

    Args:
        path: path to motion module checkpoint (.safetensors or .pth)
        model_version: SD model version (for SDXL detection)
        device: target device

    Returns:
        MotionWrapper module.
    """
    from vibewarp.vendor.cldm.model import load_state_dict
    from vibewarp.vendor.animatediff_mm import MotionWrapper

    sd = load_state_dict(path, location='cpu')

    is_sdxl = 'sdxl' in model_version.lower()
    is_v3 = 'v3' in os.path.basename(path).lower()

    mm = MotionWrapper.from_state_dict(
        sd, mm_type=os.path.basename(path),
        is_sdxl=is_sdxl, is_v3=is_v3,
    )
    mm = mm.half().to(device)
    del sd
    gc.collect()
    return mm


class _SimpleMotionModule:
    """Minimal motion module stand-in (tests / dry runs).

    Stores the state dict and provides set_video_length() as a no-op.
    Real checkpoints always load as the vendored MotionWrapper; inject/eject
    treat this class as a mark-only no-op.
    """

    def __init__(self, state_dict: dict, device: str = 'cpu'):
        self.state_dict_data = state_dict
        self.device = device
        self.video_length = 16
        self.injected = False

    def set_video_length(self, length: int):
        self.video_length = length

    def half(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def cuda(self):
        return self.to('cuda')

    def cpu(self):
        return self.to('cpu')


# ---- Injection / Ejection ----

def inject_motion_modules(
    sd_model: Any,
    motion_module: Any,
) -> None:
    """Inject AnimateDiff motion modules into the UNet's transformer blocks.

    After injection, each spatial transformer block in the UNet will have
    a temporal attention module appended. The UNet's forward pass will
    automatically process the temporal dimension.

    Args:
        sd_model: loaded SD model (with model.diffusion_model)
        motion_module: loaded MotionWrapper (from load_motion_module)

    Side effects:
        - Sets sd_model.model.diffusion_model.mm_injected = True
        - Modifies input_blocks, output_blocks (and middle_block for v2) in-place
        - For v1 (hack_gn) modules, patches GroupNorm32.forward
    """
    unet = sd_model.model.diffusion_model

    if getattr(unet, 'mm_injected', False):
        return  # Already injected

    if hasattr(motion_module, 'down_blocks'):
        # Real MotionWrapper — notebook-faithful injection
        from vibewarp.vendor.animatediff_mm import inject_motion_module_to_unet
        inject_motion_module_to_unet(unet, motion_module)
    else:
        # _SimpleMotionModule etc. — mark only
        unet.mm_injected = True

    if hasattr(motion_module, 'injected'):
        motion_module.injected = True


def eject_motion_modules(
    sd_model: Any,
    motion_module: Any = None,
) -> None:
    """Remove AnimateDiff motion modules from the UNet.

    Reverses inject_motion_modules() — removes temporal attention modules
    from all transformer blocks.

    Args:
        sd_model: loaded SD model
        motion_module: optional motion module reference (for state tracking)

    Side effects:
        - Sets sd_model.model.diffusion_model.mm_injected = False
        - Restores original block structure
    """
    unet = sd_model.model.diffusion_model

    if not getattr(unet, 'mm_injected', False):
        return  # Not injected

    if motion_module is not None and hasattr(motion_module, 'down_blocks'):
        # Real MotionWrapper — notebook-faithful ejection
        from vibewarp.vendor.animatediff_mm import eject_motion_module_from_unet
        eject_motion_module_from_unet(unet, motion_module)
    else:
        unet.mm_injected = False

    if motion_module is not None and hasattr(motion_module, 'injected'):
        motion_module.injected = False


# ---- Context scheduling ----

def make_context_schedule(
    total_length: int = 32,
    context_length: int = 16,
    overlap: int = 4,
    steps: int = 15,
) -> List[List[List[int]]]:
    """Generate context window schedules for AnimateDiff temporal attention.

    Creates a schedule of frame indices for each diffusion step. Each step
    has multiple context windows that cover all frames with overlap.

    The notebook calls this `make_ctx_sched`. Each diffusion step gets a
    shuffled list of context windows to ensure varied temporal groupings.

    Args:
        total_length: total number of frames in the batch
        context_length: number of frames per context window
        overlap: overlap between adjacent context windows
        steps: number of diffusion steps

    Returns:
        List of length `steps`, where each element is a list of context
        windows (each window is a list of frame indices).
    """
    import math
    import random

    idxs = list(range(total_length))
    stride = max(1, context_length - overlap)
    schedule = []

    for step in range(steps):
        # Always cover the head and the tail of the batch.
        windows = [idxs[:context_length]]
        if idxs[-context_length:] not in windows:
            windows.append(idxs[-context_length:])

        # Shift the window grid a little on each step so the seams between windows
        # land in different places over the course of sampling.
        start_offset = max(-stride, -((step * 2) % stride))

        for j in range(math.ceil(len(idxs) / stride)):
            start = j * stride + start_offset
            end = j * stride + context_length + start_offset
            if end > len(idxs):
                # Clamp to the last full window rather than padding a short one —
                # every window must be exactly context_length frames, because that
                # is the temporal length the motion module was given.
                start, end = -context_length, None

            window = idxs[start:end]
            if window and window not in windows:
                windows.append(window)

        # Shuffled from the global RNG (seeded per batch), as the notebook does.
        random.shuffle(windows)
        schedule.append(windows)

    return schedule


# k-diffusion samplers that evaluate the model twice per step: each consumes two
# entries from the context schedule, so it has to be doubled (notebook repeat_list).
TWO_EVAL_SAMPLERS = (
    'sample_dpm_2',
    'sample_dpm_2_ancestral',
    'sample_dpmpp_2s_ancestral',
    'sample_dpmpp_sde',
)

# Samplers that carry state between steps (old_denoised, an internal noise sampler).
# reinject_stylized has to mutate x in place — the sampler carries that same tensor
# forward, which is exactly how the anchor steers the trajectory — and that
# invalidates such state, producing streaked, degraded frames. Verified: SDXL/Hotshot
# with sample_dpmpp_sde + reinject smears badly; sample_euler (stateless) is clean.
STATEFUL_SAMPLERS = (
    'sample_dpmpp_sde',
    'sample_dpmpp_2m',
    'sample_dpmpp_2m_sde',
    'sample_lms',
)


def repeat_schedule_for_sampler(schedule, sampler_name: str):
    """Double each step's windows for samplers that call the model twice per step."""
    if sampler_name not in TWO_EVAL_SAMPLERS:
        return schedule
    doubled = []
    for windows in schedule:
        doubled.append(windows)
        doubled.append([list(w) for w in windows])
    return doubled


def split_into_batches(
    total_frames: int,
    batch_length: int = 32,
    batch_overlap: int = 8,
    start_frame: int = 0,
) -> List[List[int]]:
    """Split a frame range into overlapping batches for AnimateDiff processing.

    Args:
        total_frames: total number of frames to process
        batch_length: frames per batch
        batch_overlap: overlap between consecutive batches
        start_frame: starting frame index

    Returns:
        List of batches, each a list of frame indices.

    DELIBERATE DIVERGENCE FROM THE NOTEBOOK (do_run_adiff), owner's call 2026-07-13.

    The notebook computes `max_batches = ceil(total_length / stride)` and runs exactly
    that many, without stopping once the frames are covered. That over-counts: the last
    batch needs to START no later than total-batch_length, so the correct count is
    `ceil((total - batch_length) / stride) + 1`. With 40 frames / length 16 / overlap 4
    (stride 12), batch 2 already ends on frame 39, yet the notebook runs a 4th batch of
    [36,37,38,39, 39,39,...] -- frame 39 repeated twelve times as padding. The motion
    module then sees a frozen tail and OVERWRITES frame 39 with a degraded render
    (measured: frame 39 MAE 0.25 vs ~0.11 for its neighbours).

    We stop when the frames are covered. The notebook's tail batch is a bug and is not
    replicated; the reference is patched to match in parity runs (notebook_reference).
    """
    if total_frames <= batch_length:
        return [list(range(start_frame, start_frame + total_frames))]

    stride = max(1, batch_length - batch_overlap)
    batches = []
    pos = 0

    while pos < total_frames:
        end = min(pos + batch_length, total_frames)
        batch = list(range(start_frame + pos, start_frame + end))
        # Pad short batches by repeating last frame
        while len(batch) < batch_length:
            batch.append(batch[-1])
        batches.append(batch)
        if end >= total_frames:
            break
        pos += stride

    return batches


def blend_batch_overlap(
    prev_frames: List[Any],
    curr_frames: List[Any],
    overlap: int,
) -> List[Any]:
    """Blend overlapping frames between two consecutive AnimateDiff batches.

    Uses linear interpolation in the overlap region.

    Args:
        prev_frames: frames from previous batch (list of tensors or PIL Images)
        curr_frames: frames from current batch (list of tensors or PIL Images)
        overlap: number of overlapping frames

    Returns:
        Blended frames for the overlap region.
    """
    if overlap <= 0 or not prev_frames or not curr_frames:
        return curr_frames[:overlap] if curr_frames else []

    blended = []
    for j in range(min(overlap, len(prev_frames), len(curr_frames))):
        alpha = min(j / max(overlap, 1), 1.0)

        if isinstance(prev_frames[j], torch.Tensor) and isinstance(curr_frames[j], torch.Tensor):
            mixed = (1 - alpha) * prev_frames[j] + alpha * curr_frames[j]
        else:
            # PIL Image blending
            from PIL import Image
            import numpy as np
            p = np.array(prev_frames[j]).astype(np.float32)
            c = np.array(curr_frames[j]).astype(np.float32)
            mixed = Image.fromarray(((1 - alpha) * p + alpha * c).astype(np.uint8))

        blended.append(mixed)

    return blended
