"""The three schedule shapes mean different things.

The UI's chip editor exposes them as Constant / Per-frame / Keyframes and must
never silently convert between them. These tests pin the semantics its hint text
promises, so a change to scheduling can't quietly make the UI lie.
"""

import pytest

from vibewarp.config_io import config_from_dict
from vibewarp.utils.scheduling import get_scheduled_arg


class TestScheduleSemantics:
    def test_constant_single_element_list(self):
        """'Same value on every frame.'"""
        for frame in (0, 5, 500):
            assert get_scheduled_arg(frame, [0.7]) == 0.7

    def test_per_frame_list_is_indexed_by_frame(self):
        """'One value per frame, indexed from 0.'"""
        schedule = [10, 20, 30]
        assert get_scheduled_arg(0, schedule) == 10
        assert get_scheduled_arg(1, schedule) == 20
        assert get_scheduled_arg(2, schedule) == 30

    def test_per_frame_list_clamps_past_the_end(self):
        """'Frames past the last entry reuse it.'"""
        assert get_scheduled_arg(99, [10, 20, 30]) == 30

    def test_keyframes_hold_until_the_next_one(self):
        """'Frame -> value.' Without blending, a keyframe holds."""
        schedule = {0: 10, 10: 20}
        assert get_scheduled_arg(0, schedule) == 10
        assert get_scheduled_arg(5, schedule) == 10
        assert get_scheduled_arg(10, schedule) == 20
        assert get_scheduled_arg(50, schedule) == 20

    def test_keyframes_interpolate_when_blending(self):
        """'...interpolated when schedule blending is on.'"""
        mid = get_scheduled_arg(5, {0: 0.0, 10: 1.0}, blend_json_schedules=True)
        assert mid == pytest.approx(0.5, abs=0.05)

    def test_list_and_dict_are_not_interchangeable(self):
        """The reason the editor never converts silently: at frame 1 the list
        gives its second element, the dict still holds its frame-0 value."""
        assert get_scheduled_arg(1, [10, 20, 30]) == 20
        assert get_scheduled_arg(1, {0: 10, 10: 20}) == 10

    def test_scalar_passes_through(self):
        assert get_scheduled_arg(7, 0.5) == 0.5


class TestConfigAcceptsEveryEditorShape:
    """Whatever the chip editor emits must round-trip through the config layer."""

    @pytest.mark.parametrize("schedule", [
        None,             # 'Off' — the static field is used
        [0.7],            # Constant
        [10, 20, 30],     # Per-frame
        {0: 10, 10: 20},  # Keyframes
        [[3, 7, 3]],      # JSON fallback (per-step CFG ramp)
    ])
    def test_diffusion_schedules_round_trip(self, schedule):
        config = config_from_dict({"diffusion": {"cfg_scale_schedule": schedule}})
        assert config.diffusion.cfg_scale_schedule == schedule

    def test_frame_range_is_a_two_element_list(self):
        """The UI now edits this as two number inputs, not JSON."""
        config = config_from_dict({"frame_range": [10, 40]})
        assert config.frame_range == [10, 40]

    def test_every_schedule_field_defaults_to_off(self):
        """The editor shows 'Off' for None, meaning the static value is used —
        so None must be the default for anything named *_schedule."""
        from dataclasses import fields, is_dataclass

        from vibewarp.config import RunConfig

        def walk(cls, seen=None):
            seen = seen if seen is not None else set()
            if cls in seen:
                return
            seen.add(cls)
            for item in fields(cls):
                if item.name.endswith('_schedule'):
                    default = item.default
                    assert default is None, f"{cls.__name__}.{item.name} defaults to {default!r}"
                if is_dataclass(item.type):
                    walk(item.type, seen)

        config = RunConfig()
        for item in fields(config):
            value = getattr(config, item.name)
            if is_dataclass(value):
                walk(type(value))
