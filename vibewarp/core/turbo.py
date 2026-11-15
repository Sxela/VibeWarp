"""Two-history flow turbo with per-frame, lossless resume checkpoints."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def blend_turbo(new, old, factor, *, device=None):
    """Blend on CUDA when available, with saturating nearest-even byte rounding."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    new_pixels = torch.tensor(np.asarray(new), dtype=torch.float32, device=device)
    old_pixels = torch.tensor(np.asarray(old), dtype=torch.float32, device=device)
    pixels = (new_pixels * factor + old_pixels * (1-factor)).round().clamp_(0, 255)
    return Image.fromarray(pixels.to(dtype=torch.uint8, device='cpu').numpy())


class TurboState:
    def __init__(self, config, batch_folder, sequence_start):
        self.config = config
        self.root = Path(batch_folder) / 'temp' / 'turbo'
        self.sequence_start = sequence_start
        self.feedback = self.old = None
        self.current_init = None
        self.active = False
        self.intermediate = False
        self.scene_cut = False
        # Do not replay histories under changed render settings on resume.
        def canonical(value):
            # Config import coerces float-typed fields such as 0 to 0.0.
            # Those spellings describe the same run and must hash identically.
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, dict):
                # Percentage schedule keys can load as 0.0 instead of 0 too.
                return {(str(int(key)) if isinstance(key, (int, float)) and float(key).is_integer()
                         else str(key)): canonical(item) for key, item in value.items()}
            if isinstance(value, list):
                return [canonical(item) for item in value]
            return value
        # Version histories when their pixel/feedback semantics change.
        data = {'version': 2, 'config': canonical(asdict(config))}
        self.signature = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def warp_params(self, frame, schedule, *, old=False):
        """Notebook do_3d_step overrides, shared by both warped histories."""
        schedule = dict(schedule)
        cfg = self.config.turbo
        if old:
            schedule['soften_consistency_mask'] = cfg.consistency_soften
        if frame % cfg.interval != 0:
            # Notebook uses a truthy override: zero retains the caller's value.
            if cfg.consistency_soften:
                schedule['soften_consistency_mask'] = cfg.consistency_soften
            if cfg.disable_cc:
                schedule['check_consistency'] = False
        return schedule

    def restore(self, frame):
        folder = self.root / f'{frame:06d}'
        try:
            metadata = json.loads((folder / 'state.json').read_text())
            if metadata['signature'] != self.signature:
                raise ValueError('Turbo settings changed or checkpoint version is outdated; regenerate the Turbo frames')
            with Image.open(folder / 'feedback.png') as image:
                self.feedback = image.convert('RGB').copy()
            if metadata['has_old']:
                with Image.open(folder / 'old.png') as image:
                    self.old = image.convert('RGB').copy()
        except (OSError, KeyError, ValueError) as exc:
            raise ValueError(f'Cannot resume turbo at frame {frame+1}: {exc}') from exc

    def prepare(self, ctx, state):
        from vibewarp.core.diffusion import warp_between_frames, get_frame_schedule
        self.scene_cut = False
        frame = state.frame_num
        cfg = self.config.turbo
        self.active = frame > self.sequence_start + cfg.preroll
        self.intermediate = self.active and frame % cfg.interval != 0
        if state.init_image:
            with Image.open(state.init_image) as image:
                self.current_init = image.convert('RGB').copy()
        else:
            self.current_init = None
        if frame > self.sequence_start and cfg.scene_cut_threshold > 0:
            folder = Path(ctx.video_frames_folder)
            paths = [folder / f'{number+1:06d}.jpg' for number in (frame-1, frame)]
            if all(p.is_file() for p in paths):
                arrays = []
                for path in paths:
                    with Image.open(path) as image:
                        arrays.append(np.asarray(image.convert('RGB').resize((64, 64)), dtype=np.float32)/255)
                self.scene_cut = float(np.abs(arrays[0]-arrays[1]).mean()) >= cfg.scene_cut_threshold
        if self.scene_cut:
            from vibewarp.core.diffusion import _masked_init_path
            from vibewarp.video.raw_processing import prepared_raw_path
            self.old = None
            self.active = self.intermediate = False
            # Same preparation as a sequence-start init: RVM/raw colour, then the
            # 'before diffusion' background mask. The bare frame would pop both back in.
            state.init_image = _masked_init_path(ctx, prepared_raw_path(ctx, frame), frame)
            state.guidance_image = None
            ctx._disco_consistency_mask = None
            ctx._temporal_consistency_mask = None
            self.current_init = None
            return
        if frame == self.sequence_start + cfg.preroll and self.current_init is not None:
            self.old = self.current_init.copy()
        if self.warp_only:
            # Only warp-only frames keep the warped history; every other active frame
            # replaces it with the current init below, so warping it would be wasted.
            if self._crossfade(frame) >= 1:
                # Last turbo frame of the cycle: the old history has weight 0 and the
                # next frame is an anchor that reseeds it, so drop it instead of warping.
                self.old = None
            elif self.old is not None:
                schedule = self.warp_params(frame, get_frame_schedule(frame, self.config), old=True)
                self.old = warp_between_frames(ctx, self.old, frame-1,
                    flow_blend=schedule['flow_blend'], consistency_params=schedule)
                # The old history is its own image for this frame: mask it once, like
                # the init, so the turbo crossfade does not bring back unmasked warp
                # artifacts. A no-op unless the stage is 'before diffusion'.
                from vibewarp.core.diffusion import _apply_background_stage
                self.old = _apply_background_stage(self.old, ctx, frame, 'before diffusion')
        elif self.active:
            self.old = self.current_init.copy() if self.current_init else None

    @property
    def warp_only(self):
        return self.intermediate and self.config.turbo.intermediate_strength == 0

    def _crossfade(self, frame):
        """Weight of the new history in a warp-only frame (notebook blend_factor)."""
        interval = self.config.turbo.interval
        return ((frame % interval)+1) / interval

    def finish(self, image):
        cfg = self.config.turbo
        if self.warp_only:
            self.feedback = self.current_init.copy()
            return blend_turbo(self.feedback, self.old or self.feedback, self._crossfade(self._frame))
        self.feedback = image.copy()
        if self._frame > self.sequence_start and self.current_init is not None:
            return blend_turbo(image, self.current_init, 1/cfg.interval)
        return image

    def save(self, frame):
        folder = self.root / f'{frame:06d}'
        folder.mkdir(parents=True, exist_ok=True)
        self.feedback.save(folder / 'feedback.png')
        if self.old is not None:
            self.old.save(folder / 'old.png')
        metadata = {'frame': frame, 'signature': self.signature, 'has_old': self.old is not None}
        temporary = folder / 'state.tmp'
        temporary.write_text(json.dumps(metadata))
        temporary.replace(folder / 'state.json')
