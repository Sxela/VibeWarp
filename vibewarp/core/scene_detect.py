"""Content-aware scheduling (notebook cells 26–28).

Analyzes per-frame differences (LPIPS / rmse / rmse+lpips) to detect scene
changes, then rewrites schedules at cut boundaries via templates
``[normal_val, new_scene_val, diff_thresh, falloff_frames]`` — so each new
scene gets a fresh start (e.g. flow_blend drops to 0 at a cut and fades back).

Notebook-faithful quirks preserved:
- ``rmse(x, y)`` is actually ``|mean(x - y)|`` (the notebook's naming).
- Images are loaded /127 and normalized with CLIP stats before diffing.
- Frame 0 is always treated as a scene start.
- The cc_masked template is applied on top of the ALREADY-adjusted
  flow_blend schedule, and image_scale uses the cfg_scale template
  (both as in the notebook; both schedules are currently unsupported in
  vibewarp and warn-skip — see apply_content_schedules).

LPIPS requires the optional ``lpips`` package (``pip install lpips``).
"""

import os
from glob import glob
from typing import Any, List, Optional

import numpy as np
import torch

from vibewarp.utils.scheduling import get_scheduled_arg


_lpips_model = None

_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def load_img_lpips(path: str, size=(512, 512), device: str = 'cuda') -> torch.Tensor:
    """Notebook's load_img_lpips: RGB, LANCZOS to 512², /127, CLIP-stats norm."""
    from PIL import Image
    from torchvision import transforms as T
    image = Image.open(path).convert('RGB')
    image = image.resize(size, resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 127
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD)(image)
    return image.to(device)


def rmse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Notebook's 'rmse' — actually |mean(x - y)|. Kept as-is for parity."""
    return torch.abs(torch.mean(x - y))


def init_lpips(device: str = 'cuda'):
    """Load LPIPS (vgg net, as the notebook). Requires the `lpips` package."""
    global _lpips_model
    if _lpips_model is None:
        try:
            import lpips
        except ImportError as e:
            raise ImportError(
                "Content-aware scheduling with an lpips diff function requires "
                "the 'lpips' package: pip install lpips "
                "(or use diff_function='rmse')") from e
        _lpips_model = lpips.LPIPS(net='vgg').to(device)
    return _lpips_model


def analyze_video_frames(
    video_frames_folder: str,
    diff_function: str = 'rmse+lpips',
    device: str = 'cuda',
    size=(512, 512),
) -> List[float]:
    """Per-frame difference analysis (notebook analyze_video).

    Returns diff list with diff[0] == 0 and diff[i] = d(frame[i-1], frame[i]).
    Prints the notebook's suggested threshold (97th percentile).
    """
    frames = sorted(glob(os.path.join(video_frames_folder, '*.jpg')))
    if len(frames) < 2:
        return [0.0] * len(frames)

    if diff_function == 'lpips':
        model = init_lpips(device)
        diff_func = model
    elif diff_function == 'rmse+lpips':
        model = init_lpips(device)
        diff_func = lambda x, y: rmse(x, y) * model(x, y)
    else:  # 'rmse'
        diff_func = rmse

    from tqdm import trange
    diff = [0.0]
    prev = load_img_lpips(frames[0], size=size, device=device)
    for i in trange(1, len(frames), desc='Analyzing video'):
        cur = load_img_lpips(frames[i], size=size, device=device)
        with torch.no_grad():
            d = diff_func(prev, cur).sum().mean().detach().cpu().numpy()
        diff.append(float(d))
        prev = cur

    calc_thresh = np.percentile(np.array(diff), 97)
    print(f'suggested threshold: {round(float(calc_thresh), 2)}')
    return diff


def find_scene_splits(diff: List[float], thresh: float) -> List[int]:
    """Frame indices whose difference exceeds thresh (notebook cell 27 peaks)."""
    return [i for i, d in enumerate(diff) if d > thresh]


def adjust_schedule(diff, normal_val, new_scene_val, thresh, falloff_frames, sched=None):
    """Notebook-exact adjust_schedule (cell 28)."""
    diff_array = np.array(diff)

    diff_new = np.zeros_like(diff_array)
    diff_new = diff_new + normal_val

    for i in range(len(diff_new)):
        el = diff_array[i]
        if sched is not None:
            diff_new[i] = get_scheduled_arg(i, sched)
        if el > thresh or i == 0:
            diff_new[i] = new_scene_val
            if falloff_frames > 0:
                for j in range(falloff_frames):
                    if i + j > len(diff_new) - 1:
                        break
                    falloff_val = normal_val
                    if sched is not None:
                        falloff_val = get_scheduled_arg(i + falloff_frames, sched)
                    diff_new[i + j] = (new_scene_val * (falloff_frames - j) / falloff_frames
                                       + falloff_val * j / falloff_frames)
    return diff_new


def check_and_adjust_sched(sched, template, diff, respect_sched=True):
    """Notebook-exact: apply a template to a schedule, or pass through when
    the template is empty."""
    if template is None or template == '' or template == []:
        return sched
    normal_val, new_scene_val, thresh, falloff_frames = template
    sched_source = None
    if respect_sched:
        sched_source = sched
    return list(adjust_schedule(diff, normal_val, new_scene_val, thresh,
                                falloff_frames, sched_source).astype('float').round(3))


def apply_content_schedules(config, diff: List[float]) -> None:
    """Apply the configured templates to the run's schedules (notebook's
    make_schedules block), mutating config in place.

    Supported here: steps / cfg_scale / style_strength (diffusion) and
    flow_blend (warp). The notebook additionally adjusts latent_scale /
    init_scale / image_scale / cc_masked_diffusion schedules — those are
    not per-frame-schedulable in vibewarp yet, so their templates warn-skip.
    """
    sc = config.scene
    respect = sc.respect_sched

    d = config.diffusion
    d.steps_schedule = check_and_adjust_sched(
        d.steps_schedule, sc.steps_template, diff, respect)
    d.cfg_scale_schedule = check_and_adjust_sched(
        d.cfg_scale_schedule, sc.cfg_scale_template, diff, respect)
    d.style_strength_schedule = check_and_adjust_sched(
        d.style_strength_schedule, sc.style_strength_template, diff, respect)
    config.warp.flow_blend_schedule = check_and_adjust_sched(
        config.warp.flow_blend_schedule, sc.flow_blend_template, diff, respect)

    for name, template in [('latent_scale', sc.latent_scale_template),
                           ('init_scale', sc.init_scale_template),
                           ('image_scale', sc.image_scale_template),
                           ('cc_masked', sc.cc_masked_template)]:
        if template not in (None, '', []):
            print(f"  WARNING: {name}_template set but {name} is not "
                  f"per-frame-schedulable in vibewarp yet — template ignored")

    print('Applied content-aware schedules:')
    for sched, name in [(d.steps_schedule, 'steps_schedule'),
                        (d.cfg_scale_schedule, 'cfg_scale_schedule'),
                        (d.style_strength_schedule, 'style_strength_schedule'),
                        (config.warp.flow_blend_schedule, 'flow_blend_schedule')]:
        if isinstance(sched, list) and len(sched) > 2:
            print(f'  {name}: {sched[:100]}')
