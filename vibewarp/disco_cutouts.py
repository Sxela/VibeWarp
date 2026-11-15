"""Executed-step cutout schedules and one-time migration of older noise axes.

This module loads no models and does not import torch: settings migration is
also used by the lightweight web configuration API.
"""
import math
import struct

CUT_SCHEDULE_VERSION = 2
CUT_SCHEDULE_FIELDS = ('cut_overview', 'cut_innercut', 'cut_icgray_schedule')


def inference_progress(step_index, executed_steps):
    return 100 * step_index / max(executed_steps - 1, 1)


def held_value(schedule, progress):
    return next((schedule[key] for key in sorted(schedule, reverse=True)
                 if key <= progress or math.isclose(key, progress, rel_tol=0,
                     abs_tol=4 * max(math.ulp(key), math.ulp(progress)))), 0)


def legacy_timestep_trace(config):
    """Mapped original times for the first configured frame, after skipping."""
    from vibewarp.utils.scheduling import get_scheduled_arg
    d = config.disco
    frame = (config.frame_range or [0])[0]
    if config.keyframes_relative_to_frame_range:
        frame = 0

    def resolve(schedule, fallback):
        value = fallback if schedule is None else get_scheduled_arg(frame, schedule, blend_json_schedules=True)
        while isinstance(value, list):
            value = value[0] if value else fallback
        return value

    steps = int(resolve(d.steps_schedule, d.steps))
    strength = float(resolve(d.style_strength_schedule, d.style_strength))
    if not 2 <= steps <= 1000 or not 0 < strength <= 1:
        raise ValueError('Cutout migration requires valid sampling steps and style strength')
    if d.timestep_grid:
        from vibewarp.core.disco_repeated import validate_grid
        grid = validate_grid(d.timestep_grid, steps)[:-1]
    elif d.sampler == 'distilled' or d.quality_lora_enabled:
        # Both supported trained adapter grids use this exact integer VP grid.
        grid = [1000 * (steps - index) // steps - 1 for index in range(steps)]
    else:
        # Mirror the original non-adapter DDIM respacing and time rescaling.
        stride = 1000 // steps
        original_steps = stride * steps
        f32 = lambda value: struct.unpack('f', struct.pack('f', value))[0]
        grid = [f32(index * stride * f32(1000 / original_steps)) for index in reversed(range(steps))]
    skip = min(steps - 1, int(steps * (1 - strength) + 1e-8))
    return grid[skip:]


def migrate_cutout_schedules(config, original_time=False):
    """Preserve legacy per-step selections, then retire the old schedule axis."""
    times = legacy_timestep_trace(config)
    positions = [999 - time if original_time else 100 * (1 - time / 1000) for time in times]
    for name in CUT_SCHEDULE_FIELDS:
        schedule = getattr(config.disco, name)
        if schedule is None:
            continue
        migrated = {}
        previous = object()
        for index, position in enumerate(positions):
            # Legacy selection used a strict comparison, without percentage tolerance.
            value = next((schedule[key] for key in sorted(schedule, reverse=True) if key <= position), 0)
            if value != previous:
                key = inference_progress(index, len(times))
                # JSON/JS round trips must never push a boundary past its step.
                if 0 < key < 100:
                    key = math.nextafter(key, -math.inf)
                migrated[key] = value
                previous = value
        setattr(config.disco, name, migrated)
    config.disco.cut_schedule_version = CUT_SCHEDULE_VERSION
