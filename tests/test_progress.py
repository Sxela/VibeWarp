"""Tests for structured progress reporting."""

import time
import pytest

from vibewarp.progress import (
    Stage,
    ProgressEvent,
    ProgressReporter,
)


class TestProgressEvent:
    def test_defaults(self):
        e = ProgressEvent(stage=Stage.INIT)
        assert e.stage == Stage.INIT
        assert e.frame == 0
        assert e.total_frames == 0
        assert e.pct == 0.0

    def test_pct_calculation(self):
        e = ProgressEvent(stage=Stage.RENDERING, frame=50, total_frames=100)
        assert e.pct == 50.0

    def test_pct_clamped_to_100(self):
        e = ProgressEvent(stage=Stage.RENDERING, frame=150, total_frames=100)
        assert e.pct == 100.0

    def test_pct_zero_total(self):
        e = ProgressEvent(stage=Stage.RENDERING, frame=5, total_frames=0)
        assert e.pct == 0.0

    def test_with_data(self):
        e = ProgressEvent(stage=Stage.COMPLETE, data={'output_dir': '/tmp/out'})
        assert e.data['output_dir'] == '/tmp/out'


class TestProgressReporter:
    def test_emit_triggers_listener(self):
        events = []
        reporter = ProgressReporter(on_event=lambda e: events.append(e))
        reporter.stage(Stage.INIT, 'Starting')
        assert len(events) == 1
        assert events[0].stage == Stage.INIT
        assert events[0].message == 'Starting'

    def test_multiple_listeners(self):
        count = [0, 0]
        reporter = ProgressReporter()
        reporter.on_event(lambda e: count.__setitem__(0, count[0] + 1))
        reporter.on_event(lambda e: count.__setitem__(1, count[1] + 1))
        reporter.stage(Stage.LOAD_MODELS)
        assert count == [1, 1]

    def test_stage_listener(self):
        stages = []
        reporter = ProgressReporter(on_stage=lambda s: stages.append(s))
        reporter.stage(Stage.EXTRACT_FRAMES)
        reporter.stage(Stage.LOAD_MODELS)
        assert stages == [Stage.EXTRACT_FRAMES, Stage.LOAD_MODELS]

    def test_frame_progress(self):
        frames = []
        reporter = ProgressReporter(on_frame=lambda f, t: frames.append((f, t)))
        reporter.frame_progress(0, 100)
        reporter.frame_progress(50, 100)
        assert frames == [(0, 100), (50, 100)]

    def test_frame_progress_compatible_with_ctx(self):
        """frame_progress() signature matches ctx.progress_fn(frame, total)."""
        reporter = ProgressReporter()
        # Should be callable with exactly two args
        reporter.frame_progress(10, 50)
        assert len(reporter.events) == 1
        assert reporter.events[0].frame == 10
        assert reporter.events[0].total_frames == 50

    def test_batch_progress(self):
        reporter = ProgressReporter()
        reporter.batch_progress(batch=1, total_batches=4, frame=32, total_frames=128)
        e = reporter.events[0]
        assert e.stage == Stage.RENDER_BATCH
        assert e.batch == 1
        assert e.total_batches == 4

    def test_error_reporting(self):
        reporter = ProgressReporter()
        reporter.error('CUDA OOM', device='cuda:0')
        assert reporter.events[0].stage == Stage.ERROR
        assert reporter.events[0].message == 'CUDA OOM'
        assert reporter.events[0].data['device'] == 'cuda:0'

    def test_complete(self):
        reporter = ProgressReporter()
        reporter.complete(total_frames=100, output_dir='/out')
        e = reporter.events[0]
        assert e.stage == Stage.COMPLETE
        assert e.frame == 100
        assert e.data['output_dir'] == '/out'

    def test_elapsed_increases(self):
        reporter = ProgressReporter()
        t1 = reporter.elapsed
        # Just check it's a non-negative float
        assert t1 >= 0

    def test_events_log(self):
        reporter = ProgressReporter()
        reporter.stage(Stage.INIT)
        reporter.stage(Stage.LOAD_MODELS)
        reporter.frame_progress(0, 10)
        assert len(reporter.events) == 3

    def test_event_elapsed_populated(self):
        reporter = ProgressReporter()
        reporter.stage(Stage.INIT)
        assert reporter.events[0].elapsed >= 0


class TestStageEnum:
    def test_all_stages_are_strings(self):
        for stage in Stage:
            assert isinstance(stage.value, str)

    def test_key_stages_exist(self):
        assert Stage.INIT == 'init'
        assert Stage.RENDERING == 'rendering'
        assert Stage.COMPLETE == 'complete'
        assert Stage.ERROR == 'error'
        assert Stage.RECONSTRUCTION_NOISE == 'reconstruction_noise'
