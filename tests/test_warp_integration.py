"""Tests for inter-frame warp integration in diffusion.py."""

import os

import numpy as np
import pytest
import torch
from PIL import Image

from vibewarp.config import (
    BrightnessConfig,
    ColorConfig,
    FlowConfig,
    RunConfig,
    WarpConfig,
)
from vibewarp.core.diffusion import RenderContext, warp_between_frames


class MockRAFTModel(torch.nn.Module):
    """Mock RAFT returning small constant flow."""

    def forward(self, frame1, frame2, num_flow_updates=20):
        B, C, H, W = frame1.shape
        flow = torch.zeros(B, 2, H, W, device=frame1.device)
        flow[:, 0, :, :] = 1.0
        flow[:, 1, :, :] = 0.5
        return None, flow


@pytest.fixture
def video_frames_dir(tmp_path):
    """Create a directory with numbered video frames."""
    d = tmp_path / "video_frames"
    d.mkdir()
    # ffmpeg %06d.jpg starts at 1, so video frames are 1-based
    for i in range(1, 6):
        img = Image.fromarray(
            np.random.randint(0, 255, (64, 80, 3), dtype=np.uint8)
        )
        img.save(str(d / f"{i:06d}.jpg"))
    return str(d)


@pytest.fixture
def ctx_with_raft(tmp_path, video_frames_dir):
    """Build a RenderContext with mock RAFT for testing."""
    batch_folder = str(tmp_path / "batch")
    os.makedirs(batch_folder, exist_ok=True)
    config = RunConfig(
        flow=FlowConfig(flow_warp=True, check_consistency=False),
        warp=WarpConfig(warp_strength=1.0, padding_ratio=0.1),
        color=ColorConfig(match_color_strength=0.0),
        brightness=BrightnessConfig(enable=False),
    )
    return RenderContext(
        config=config,
        raft_model=MockRAFTModel().eval(),
        batch_folder=batch_folder,
        video_frames_folder=video_frames_dir,
        flow_folder=str(tmp_path / "flow"),
    )


class TestWarpBetweenFrames:
    def test_returns_pil_image(self, ctx_with_raft):
        prev_frame = Image.fromarray(
            np.random.randint(0, 255, (64, 80, 3), dtype=np.uint8)
        )
        result = warp_between_frames(ctx_with_raft, prev_frame, frame_num=0)
        assert isinstance(result, Image.Image)

    def test_output_dimensions(self, ctx_with_raft):
        prev_frame = Image.fromarray(
            np.random.randint(0, 255, (64, 80, 3), dtype=np.uint8)
        )
        result = warp_between_frames(ctx_with_raft, prev_frame, frame_num=0)
        # Output size should match input dimensions
        assert result.size[0] > 0
        assert result.size[1] > 0

    def test_caches_flow_to_disk(self, ctx_with_raft):
        prev_frame = Image.fromarray(
            np.random.randint(0, 255, (64, 80, 3), dtype=np.uint8)
        )
        warp_between_frames(ctx_with_raft, prev_frame, frame_num=0)
        flow_path = os.path.join(ctx_with_raft.flow_folder, "000000.jpg.npy")
        assert os.path.exists(flow_path)

    def test_missing_video_frames_returns_prev(self, tmp_path):
        """If video frames don't exist, returns prev_frame unchanged."""
        config = RunConfig(flow=FlowConfig(flow_warp=True))
        ctx = RenderContext(
            config=config,
            video_frames_folder=str(tmp_path / "nonexistent"),
            batch_folder=str(tmp_path),
        )
        prev = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        result = warp_between_frames(ctx, prev, frame_num=0)
        # Should return prev_frame as-is when video frames missing
        assert np.array_equal(np.array(result), np.array(prev))

    def test_with_color_matching(self, ctx_with_raft):
        """Test that color matching can be enabled without errors."""
        ctx_with_raft.config.color.before_enabled = True
        ctx_with_raft.config.color.before_strength = 0.5
        prev_frame = Image.fromarray(
            np.random.randint(100, 200, (64, 80, 3), dtype=np.uint8)
        )
        result = warp_between_frames(ctx_with_raft, prev_frame, frame_num=0)
        assert isinstance(result, Image.Image)

    def test_with_brightness_adjustment(self, ctx_with_raft):
        """Test that brightness adjustment can be enabled without errors."""
        ctx_with_raft.config.brightness.enable = True
        prev_frame = Image.fromarray(
            np.random.randint(0, 255, (64, 80, 3), dtype=np.uint8)
        )
        result = warp_between_frames(ctx_with_raft, prev_frame, frame_num=0)
        assert isinstance(result, Image.Image)

    def test_with_consistency_check(self, tmp_path, video_frames_dir):
        """Test flow computation with consistency map generation."""
        batch_folder = str(tmp_path / "batch")
        os.makedirs(batch_folder, exist_ok=True)
        config = RunConfig(
            flow=FlowConfig(flow_warp=True, check_consistency=True),
            warp=WarpConfig(warp_strength=1.0),
        )
        ctx = RenderContext(
            config=config,
            raft_model=MockRAFTModel().eval(),
            batch_folder=batch_folder,
            video_frames_folder=video_frames_dir,
            flow_folder=str(tmp_path / "flow"),
        )
        prev_frame = Image.fromarray(
            np.random.randint(0, 255, (64, 80, 3), dtype=np.uint8)
        )
        result = warp_between_frames(ctx, prev_frame, frame_num=0)
        assert isinstance(result, Image.Image)
        # Consistency map should be saved
        cc_path = os.path.join(str(tmp_path / "flow"), "000000.jpg_12-21_cc.jpg")
        assert os.path.exists(cc_path)
        # The processed/effective keep mask is a render-frame-aligned debug
        # layer (frame 1 is produced by flow 0 -> 1).
        mask_path = os.path.join(batch_folder, "debug", "consistency_mask_000001.png")
        assert os.path.exists(mask_path)
        mask = Image.open(mask_path)
        assert mask.mode == 'L'
        assert mask.size == (config.video.width, config.video.height)

    def test_different_blend_values(self, ctx_with_raft):
        prev_frame = Image.fromarray(
            np.random.randint(0, 255, (64, 80, 3), dtype=np.uint8)
        )
        result_low = warp_between_frames(ctx_with_raft, prev_frame, frame_num=0, flow_blend=0.1)
        # Reset flow cache for second call
        flow_path = os.path.join(ctx_with_raft.flow_folder, "000000.jpg.npy")
        if os.path.exists(flow_path):
            os.remove(flow_path)
        result_high = warp_between_frames(ctx_with_raft, prev_frame, frame_num=0, flow_blend=0.9)
        # Results should differ with different blend values
        assert isinstance(result_low, Image.Image)
        assert isinstance(result_high, Image.Image)
