"""Frame-keyed schedules can use coordinates relative to the selected range."""

from vibewarp.config import RunConfig
from vibewarp.core.diffusion import (
    _get_prompt_for_frame,
    _schedule_frame_number,
    get_frame_schedule,
)


def test_relative_coordinate_starts_at_zero_for_selected_range():
    config = RunConfig(
        frame_range=[60, 70],
        keyframes_relative_to_frame_range=True,
    )

    assert _schedule_frame_number(60, config) == 0
    assert _schedule_frame_number(61, config) == 1
    assert _schedule_frame_number(70, config) == 10


def test_absolute_coordinate_remains_default():
    config = RunConfig(frame_range=[60, 70])

    assert _schedule_frame_number(60, config) == 60


def test_numeric_keyframes_are_resolved_relative_to_range():
    config = RunConfig(
        frame_range=[60, 70],
        keyframes_relative_to_frame_range=True,
    )
    config.diffusion.style_strength_schedule = {0: 1.0, 1: 0.7}

    assert get_frame_schedule(60, config)['style_strength'] == 1.0
    assert get_frame_schedule(61, config)['style_strength'] == 0.7


def test_per_frame_lists_use_the_same_relative_coordinate():
    config = RunConfig(
        frame_range=[60, 70],
        keyframes_relative_to_frame_range=True,
    )
    config.warp.flow_blend_schedule = [0.9, 0.6]

    assert get_frame_schedule(60, config)['flow_blend'] == 0.9
    assert get_frame_schedule(61, config)['flow_blend'] == 0.6


def test_prompt_keyframes_use_relative_coordinate():
    config = RunConfig(
        frame_range=[60, 70],
        keyframes_relative_to_frame_range=True,
    )
    prompts = {0: 'first', 1: 'second'}

    first = _get_prompt_for_frame(
        _schedule_frame_number(60, config), prompts)
    second = _get_prompt_for_frame(
        _schedule_frame_number(61, config), prompts)

    assert first == 'first'
    assert second == 'second'
