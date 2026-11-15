"""Compatibility for the retired, independent video frame-count cap."""


def imported_frame_range(selected, legacy_count=0):
    """An explicit range wins; otherwise translate the old count once on import.

    [0, 0] retains its historical meaning of all frames. A one-item range [N]
    selects exactly frame N, including frame zero, without a second limit.
    """
    if selected is not None and selected != [0, 0]:
        return selected
    if type(legacy_count) is int and legacy_count > 0:
        return [0] if legacy_count == 1 else [0, legacy_count - 1]
    return selected if selected is not None else [0, 0]
