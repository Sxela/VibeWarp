"""Tests for per-frame IP-Adapter re-hooking."""

import os
import pytest
import numpy as np
import torch
from PIL import Image

from vibewarp.config import RunConfig
from vibewarp.core.diffusion import (
    RenderContext,
    FrameState,
    _update_ipadapter_for_frame,
)


class TestUpdateIpadapterForFrame:
    def test_noop_without_ipadapters(self):
        """Should be a no-op when no IP-Adapters loaded."""
        ctx = RenderContext(config=RunConfig())
        state = FrameState(frame_num=0)
        _update_ipadapter_for_frame(ctx, state)  # should not error

    def test_noop_with_static_source(self):
        """Should skip static source images (not a directory or dict)."""
        ctx = RenderContext(
            config=RunConfig(),
            loaded_ipadapters={
                'ipa1': {
                    'source_image': '/path/to/static.png',
                    'weight': 1.0,
                },
            },
        )
        state = FrameState(frame_num=5)
        _update_ipadapter_for_frame(ctx, state)  # should not error, no re-hook

    def test_directory_source_resolves(self, tmp_path):
        """Should resolve frame-numbered image from directory source."""
        # Create frame images
        src_dir = str(tmp_path / 'ipa_frames')
        os.makedirs(src_dir)
        for i in range(5):
            img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
            img.save(os.path.join(src_dir, f"{i:06d}.png"))

        ctx = RenderContext(
            config=RunConfig(),
            loaded_ipadapters={
                'ipa1': {
                    'source_image': src_dir,
                    'weights': {'image_proj': {}, 'ip_adapter': {}},
                    'weight': 0.5,
                    'start': 0.0,
                    'end': 1.0,
                },
            },
            # No clip_vision_model, so re-hook won't actually happen
            clip_vision_model=None,
        )
        state = FrameState(frame_num=2)
        # Should not crash even without clip model
        _update_ipadapter_for_frame(ctx, state)

    def test_noop_without_sd_model(self):
        """Should be a no-op when sd_model is None."""
        ctx = RenderContext(
            config=RunConfig(),
            sd_model=None,
            loaded_ipadapters={'ipa1': {'source_image': '/some/dir', 'weight': 1.0}},
        )
        state = FrameState(frame_num=0)
        _update_ipadapter_for_frame(ctx, state)  # should not error

    def test_missing_frame_in_directory(self, tmp_path):
        """Should skip gracefully when frame image doesn't exist in directory."""
        src_dir = str(tmp_path / 'ipa_frames')
        os.makedirs(src_dir)
        # Only frame 0 exists
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(
            os.path.join(src_dir, "000000.png")
        )

        ctx = RenderContext(
            config=RunConfig(),
            loaded_ipadapters={
                'ipa1': {
                    'source_image': src_dir,
                    'weights': {},
                    'weight': 1.0,
                },
            },
        )
        state = FrameState(frame_num=99)  # doesn't exist
        _update_ipadapter_for_frame(ctx, state)  # should not error

    def test_empty_source_skipped(self):
        """Empty source_image should be skipped."""
        ctx = RenderContext(
            config=RunConfig(),
            loaded_ipadapters={
                'ipa1': {
                    'source_image': '',
                    'weight': 1.0,
                },
            },
        )
        state = FrameState(frame_num=0)
        _update_ipadapter_for_frame(ctx, state)  # should not error
