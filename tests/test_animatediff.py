"""Tests for vibewarp.core.animatediff — motion module management and context scheduling."""

import torch
import pytest

from vibewarp.core.animatediff import (
    _SimpleMotionModule,
    make_context_schedule,
    split_into_batches,
    blend_batch_overlap,
    inject_motion_modules,
    eject_motion_modules,
)


class TestSimpleMotionModule:
    def test_create(self):
        mm = _SimpleMotionModule({}, device='cpu')
        assert mm.device == 'cpu'
        assert mm.video_length == 16

    def test_set_video_length(self):
        mm = _SimpleMotionModule({})
        mm.set_video_length(32)
        assert mm.video_length == 32

    def test_half_returns_self(self):
        mm = _SimpleMotionModule({})
        assert mm.half() is mm

    def test_to_device(self):
        mm = _SimpleMotionModule({})
        mm.to('cuda')
        assert mm.device == 'cuda'
        mm.cpu()
        assert mm.device == 'cpu'

    def test_stores_state_dict(self):
        sd = {'key': torch.zeros(1)}
        mm = _SimpleMotionModule(sd)
        assert 'key' in mm.state_dict_data


class TestMakeContextSchedule:
    def test_single_window(self):
        """When total_length <= context_length, one window covers all frames."""
        sched = make_context_schedule(total_length=8, context_length=16, overlap=4, steps=3)
        assert len(sched) == 3
        assert sched[0] == [list(range(8))]

    def test_multiple_windows(self):
        """Multiple windows with overlap."""
        sched = make_context_schedule(total_length=32, context_length=16, overlap=4, steps=5)
        assert len(sched) == 5
        # Each step should have multiple windows
        for step_windows in sched:
            assert len(step_windows) >= 2
            # Each window should have context_length frames
            for window in step_windows:
                assert len(window) == 16

    def test_all_frames_covered(self):
        """Every frame index should appear in at least one window per step."""
        sched = make_context_schedule(total_length=24, context_length=16, overlap=4, steps=3)
        for step_windows in sched:
            covered = set()
            for window in step_windows:
                covered.update(window)
            for i in range(24):
                assert i in covered, f"Frame {i} not covered"

    def test_overlap_between_windows(self):
        """Adjacent windows should have overlapping frame indices."""
        sched = make_context_schedule(total_length=32, context_length=16, overlap=8, steps=1)
        # Sort windows by first element for this test
        windows = sorted(sched[0], key=lambda w: w[0])
        if len(windows) >= 2:
            set0 = set(windows[0])
            set1 = set(windows[1])
            assert len(set0 & set1) > 0

    def test_steps_match(self):
        sched = make_context_schedule(total_length=32, context_length=16, overlap=4, steps=10)
        assert len(sched) == 10

    def test_minimal_overlap(self):
        sched = make_context_schedule(total_length=32, context_length=16, overlap=0, steps=2)
        assert len(sched) == 2
        for step_windows in sched:
            assert len(step_windows) >= 2

    def test_large_overlap(self):
        """Overlap close to context_length should still work."""
        sched = make_context_schedule(total_length=32, context_length=16, overlap=15, steps=2)
        assert len(sched) == 2
        # Should have many windows due to stride=1
        assert len(sched[0]) >= 16


class TestSplitIntoBatches:
    def test_single_batch(self):
        batches = split_into_batches(total_frames=16, batch_length=32)
        assert len(batches) == 1
        assert batches[0] == list(range(16))

    def test_multiple_batches(self):
        batches = split_into_batches(total_frames=50, batch_length=32, batch_overlap=8)
        assert len(batches) >= 2

    def test_batch_length(self):
        batches = split_into_batches(total_frames=50, batch_length=32, batch_overlap=8)
        for batch in batches:
            assert len(batch) == 32

    def test_start_frame_offset(self):
        batches = split_into_batches(total_frames=16, batch_length=32, start_frame=10)
        assert batches[0][0] == 10

    def test_all_frames_covered(self):
        batches = split_into_batches(total_frames=50, batch_length=32, batch_overlap=8)
        covered = set()
        for batch in batches:
            covered.update(batch)
        for i in range(50):
            assert i in covered

    def test_no_overlap(self):
        batches = split_into_batches(total_frames=64, batch_length=32, batch_overlap=0)
        assert len(batches) == 2


class TestBlendBatchOverlap:
    def test_tensor_blending(self):
        prev = [torch.ones(3, 4, 4) * 0.0 for _ in range(4)]
        curr = [torch.ones(3, 4, 4) * 1.0 for _ in range(4)]
        blended = blend_batch_overlap(prev, curr, overlap=4)
        assert len(blended) == 4
        # First frame (alpha=0): should be all prev (0.0)
        assert blended[0].mean().item() == pytest.approx(0.0, abs=0.01)
        # Last frame (alpha=0.75): should lean toward curr
        assert blended[3].mean().item() > 0.5

    def test_pil_blending(self):
        from PIL import Image
        import numpy as np
        prev = [Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)) for _ in range(3)]
        curr = [Image.fromarray(np.full((8, 8, 3), 255, dtype=np.uint8)) for _ in range(3)]
        blended = blend_batch_overlap(prev, curr, overlap=3)
        assert len(blended) == 3
        # First frame should be close to black
        assert np.array(blended[0]).mean() < 10

    def test_empty_input(self):
        result = blend_batch_overlap([], [], overlap=4)
        assert result == []

    def test_zero_overlap(self):
        prev = [torch.ones(3, 4, 4)]
        curr = [torch.ones(3, 4, 4)]
        result = blend_batch_overlap(prev, curr, overlap=0)
        assert result == []


class TestInjectEjectMotionModules:
    def _make_mock_unet(self):
        """Create a minimal mock UNet with input/output blocks."""
        class MockUNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.input_blocks = torch.nn.ModuleList([
                    torch.nn.Linear(1, 1) for _ in range(12)
                ])
                self.output_blocks = torch.nn.ModuleList([
                    torch.nn.Linear(1, 1) for _ in range(12)
                ])
                self.mm_injected = False

        return MockUNet()

    def _make_mock_sd_model(self):
        class MockModel(torch.nn.Module):
            def __init__(self, unet):
                super().__init__()
                self.diffusion_model = unet

        class MockSDModel:
            def __init__(self):
                self.model = MockModel(self._make_mock_unet())
            def _make_mock_unet(inner_self):
                return self._make_mock_unet()

        return MockSDModel()

    def test_inject_sets_flag(self):
        sd_model = self._make_mock_sd_model()
        mm = _SimpleMotionModule({})
        inject_motion_modules(sd_model, mm)
        assert sd_model.model.diffusion_model.mm_injected is True
        assert mm.injected is True

    def test_double_inject_is_noop(self):
        sd_model = self._make_mock_sd_model()
        mm = _SimpleMotionModule({})
        inject_motion_modules(sd_model, mm)
        inject_motion_modules(sd_model, mm)  # should not error
        assert sd_model.model.diffusion_model.mm_injected is True

    def test_eject_clears_flag(self):
        sd_model = self._make_mock_sd_model()
        mm = _SimpleMotionModule({})
        inject_motion_modules(sd_model, mm)
        eject_motion_modules(sd_model, mm)
        assert sd_model.model.diffusion_model.mm_injected is False
        assert mm.injected is False

    def test_eject_without_inject_is_noop(self):
        sd_model = self._make_mock_sd_model()
        mm = _SimpleMotionModule({})
        eject_motion_modules(sd_model, mm)  # should not error
        assert sd_model.model.diffusion_model.mm_injected is False
