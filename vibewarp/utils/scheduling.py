"""Schedule interpolation utilities for keyframe-based parameter control."""


def get_sched_from_json(frame_num, sched_json, blend=False):
    """Look up a scheduled value from a keyframe dict.

    Args:
        frame_num: current frame index
        sched_json: dict mapping frame numbers (int or str keys) to values
        blend: if True, linearly interpolate between adjacent keyframes

    Returns:
        The scheduled value at frame_num.
    """
    frame_num = int(frame_num)
    frame_num = max(frame_num, 0)

    # Normalize keys to int
    sched_int = {}
    for key in sched_json.keys():
        sched_int[int(key)] = sched_json[key]
    sched_json = sched_int

    keys = sorted(list(sched_json.keys()))
    if not keys:
        return 0

    if frame_num < 0:
        frame_num = max(keys)
    frame_num = min(frame_num, max(keys))

    if frame_num in keys:
        return sched_json[frame_num]

    # Before first keyframe — clamp to first key's value
    if frame_num < keys[0]:
        return sched_json[keys[0]]

    for i in range(len(keys) - 1):
        k1 = keys[i]
        k2 = keys[i + 1]
        if k1 < frame_num < k2:
            if not blend:
                return sched_json[k1]
            total_dist = k2 - k1
            dist_from_k1 = frame_num - k1
            return sched_json[k1] * (1 - dist_from_k1 / total_dist) + sched_json[k2] * (dist_from_k1 / total_dist)

    # After last keyframe — clamp to last key's value
    return sched_json[keys[-1]]


def interpolate_array(array, new_steps):
    """Interpolate an array of values to a new number of steps.

    Used by WarpFusion to create per-step CFG schedules from keypoint arrays
    like [3, 7, 3] -> ramp from 3 to 7 to 3 across diffusion steps.

    Returns the array in reversed order (highest step first) so values can
    be popped from the end during sampling.
    """
    if not isinstance(array, list):
        return [array] * new_steps
    if len(array) == 1:
        return [array[0]] * new_steps
    new_steps = max(new_steps, 1)
    result = []
    old_steps = len(array) - 1
    for i in range(new_steps - 1):
        lerp = (old_steps / (new_steps - 1)) * i
        index = int(lerp)
        blend = lerp - index
        if index + 1 < len(array):
            value = array[index] * (1 - blend) + array[index + 1] * blend
        else:
            value = array[-1]
        result.append(value)
    result.append(array[-1])
    result.reverse()
    return result


def get_scheduled_arg(frame_num, schedule, blend_json_schedules=False):
    """Get a scheduled argument value, supporting both list and dict schedules.

    Args:
        frame_num: current frame index
        schedule: either a list (indexed by frame) or a dict (keyframe schedule)
        blend_json_schedules: whether to blend between keyframes for dict schedules
    """
    if isinstance(schedule, list):
        return schedule[frame_num] if frame_num < len(schedule) else schedule[-1]
    if isinstance(schedule, dict):
        return get_sched_from_json(frame_num, schedule, blend=blend_json_schedules)
    return schedule
