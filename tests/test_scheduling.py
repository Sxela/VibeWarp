"""Tests for vibewarp.utils.scheduling."""

import pytest

from vibewarp.utils.scheduling import get_sched_from_json, get_scheduled_arg


class TestGetSchedFromJson:
    def test_exact_frame(self):
        sched = {0: 1.0, 10: 2.0, 20: 3.0}
        assert get_sched_from_json(0, sched) == 1.0
        assert get_sched_from_json(10, sched) == 2.0
        assert get_sched_from_json(20, sched) == 3.0

    def test_between_frames_no_blend(self):
        sched = {0: 1.0, 10: 2.0}
        assert get_sched_from_json(5, sched, blend=False) == 1.0

    def test_between_frames_blend(self):
        sched = {0: 1.0, 10: 2.0}
        result = get_sched_from_json(5, sched, blend=True)
        assert abs(result - 1.5) < 1e-6

    def test_blend_quarter(self):
        sched = {0: 0.0, 100: 100.0}
        result = get_sched_from_json(25, sched, blend=True)
        assert abs(result - 25.0) < 1e-6

    def test_string_keys(self):
        sched = {"0": 1.0, "10": 2.0}
        assert get_sched_from_json(0, sched) == 1.0

    def test_clamp_to_max(self):
        sched = {0: 1.0, 10: 2.0}
        assert get_sched_from_json(100, sched) == 2.0

    def test_negative_frame(self):
        sched = {0: 1.0, 10: 2.0}
        result = get_sched_from_json(-5, sched)
        assert result == 1.0  # clamped to 0

    def test_empty_schedule(self):
        sched = {}
        assert get_sched_from_json(5, sched) == 0


class TestGetScheduledArg:
    def test_list_in_range(self):
        schedule = [10, 20, 30, 40]
        assert get_scheduled_arg(0, schedule) == 10
        assert get_scheduled_arg(2, schedule) == 30

    def test_list_past_end(self):
        schedule = [10, 20, 30]
        assert get_scheduled_arg(100, schedule) == 30

    def test_dict_schedule(self):
        schedule = {0: 1.0, 10: 2.0}
        assert get_scheduled_arg(5, schedule) == 1.0

    def test_dict_with_blend(self):
        schedule = {0: 0.0, 10: 10.0}
        result = get_scheduled_arg(5, schedule, blend_json_schedules=True)
        assert abs(result - 5.0) < 1e-6

    def test_scalar_passthrough(self):
        assert get_scheduled_arg(5, 42.0) == 42.0
