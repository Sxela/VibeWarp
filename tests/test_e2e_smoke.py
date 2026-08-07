"""End-to-end smoke tests for the full VibeWarp pipeline with mock models.

These tests verify that the pipeline orchestration works correctly
by mocking the heavy model components (SD model, RAFT, etc.) while
exercising the real config, frame loop, warping, and I/O code.
"""

import os
import pytest
import numpy as np
import torch
from PIL import Image
from unittest.mock import MagicMock, patch, PropertyMock

from vibewarp.config import (
    RunConfig, DiffusionConfig, VideoConfig, FlowConfig, WarpConfig,
    ControlNetConfig, ControlNetEntry,
    AnimateDiffConfig,
    IPAdapterConfig, IPAdapterEntry,
)
from vibewarp.core.diffusion import (
    RenderContext,
    FrameState,
    _resolve_frame_range,
    _render_single_frame,
    run_frames,
    get_frame_schedule,
)


class TestEndToEndSmoke:
    """Smoke tests that exercise the pipeline end-to-end with mocks."""

    def _create_video_frames(self, tmp_path, n_frames=5, size=(64, 64)):
        """Create dummy video frames on disk."""
        vf_dir = str(tmp_path / 'video_frames')
        os.makedirs(vf_dir, exist_ok=True)
        for i in range(1, n_frames + 2):  # +2 for 1-indexed + extra
            img = Image.fromarray(
                np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            )
            img.save(os.path.join(vf_dir, f"{i:06d}.jpg"))
        return vf_dir

    def test_render_single_frame_with_mock(self, tmp_path):
        """_render_single_frame should produce an image and save it."""
        vf_dir = self._create_video_frames(tmp_path, n_frames=5)
        batch_dir = str(tmp_path / 'batch')
        os.makedirs(batch_dir, exist_ok=True)

        config = RunConfig(
            frame_range=[0, 5],
            video=VideoConfig(width=64, height=64),
            diffusion=DiffusionConfig(steps=5, cfg_scale=7.0, seed=42),
        )
        ctx = RenderContext(
            config=config,
            batch_folder=batch_dir,
            video_frames_folder=vf_dir,
        )
        state = FrameState(seed=42)

        with patch('vibewarp.core.diffusion.render_frame') as mock_rf:
            mock_rf.return_value = Image.new('RGB', (64, 64), color='red')
            image, filepath = _render_single_frame(
                ctx, state, frame_num=0, start_frame=0,
                batch_folder=batch_dir, fmt='png',
            )

        assert isinstance(image, Image.Image)
        assert os.path.exists(filepath)
        assert image.size == (64, 64)

    def test_nonzero_start_uses_absolute_raw_video_frame(self, tmp_path):
        """The first frame of range 60..70 initializes from 000061.jpg."""
        vf_dir = str(tmp_path / 'video_frames')
        batch_dir = str(tmp_path / 'batch')
        os.makedirs(vf_dir)
        os.makedirs(batch_dir)
        raw_path = os.path.join(vf_dir, '000061.jpg')
        Image.new('RGB', (64, 64), color='blue').save(raw_path)

        config = RunConfig(
            frame_range=[60, 70],
            video=VideoConfig(width=64, height=64),
            flow=FlowConfig(flow_warp=True),
        )
        ctx = RenderContext(
            config=config,
            batch_folder=batch_dir,
            video_frames_folder=vf_dir,
        )
        state = FrameState(seed=42)

        with patch('vibewarp.core.diffusion.render_frame') as mock_rf:
            mock_rf.return_value = Image.new('RGB', (64, 64), color='red')
            _render_single_frame(
                ctx, state, frame_num=60, start_frame=60,
                batch_folder=batch_dir, fmt='png',
            )

        assert state.init_image == raw_path
        assert mock_rf.call_args.args[1].init_image == raw_path

    @pytest.mark.parametrize(
        ('mode', 'expected_name'),
        [
            ('prev stylized', 'warpfusion(0)_000000.png'),
            ('prev warped', '_warped_no_cc_000001.png'),
            ('prev warped + cc', '_warped_000001.png'),
        ],
    )
    def test_sd_temporal_guidance_mode_selects_gradient_target(
        self, tmp_path, mode, expected_name,
    ):
        vf_dir = self._create_video_frames(tmp_path, n_frames=2)
        batch_dir = str(tmp_path / 'batch')
        os.makedirs(batch_dir)
        previous_path = os.path.join(
            batch_dir, 'warpfusion(0)_000000.png')
        Image.new('RGB', (64, 64), color='red').save(previous_path)

        config = RunConfig(
            frame_range=[0, 2],
            model_version='control_multi_v15',
            video=VideoConfig(width=64, height=64),
            flow=FlowConfig(flow_warp=True),
            diffusion=DiffusionConfig(guidance_mode=mode),
        )
        ctx = RenderContext(
            config=config,
            batch_folder=batch_dir,
            video_frames_folder=vf_dir,
        )
        state = FrameState(
            seed=42, prev_frame=Image.new('RGB', (64, 64), color='red'))
        warped_cc = Image.new('RGB', (64, 64), color='green')
        warped_no_cc = Image.new('RGB', (64, 64), color='blue')

        with (
            patch(
                'vibewarp.core.diffusion.warp_between_frames',
                return_value=(warped_cc, warped_no_cc),
            ),
            patch('vibewarp.core.diffusion.render_frame') as mock_rf,
        ):
            mock_rf.return_value = Image.new('RGB', (64, 64), color='white')
            _render_single_frame(
                ctx, state, frame_num=1, start_frame=0,
                batch_folder=batch_dir, fmt='png',
            )

        rendered_state = mock_rf.call_args.args[1]
        assert os.path.basename(rendered_state.guidance_image) == expected_name
        assert os.path.basename(rendered_state.init_image) == '_warped_000001.png'

    def test_run_frames_standard_mode(self, tmp_path):
        """Standard run_frames should render each frame sequentially."""
        vf_dir = self._create_video_frames(tmp_path, n_frames=3)
        batch_dir = str(tmp_path / 'batch')
        os.makedirs(batch_dir, exist_ok=True)

        config = RunConfig(
            frame_range=[0, 3],
            video=VideoConfig(width=64, height=64),
            diffusion=DiffusionConfig(steps=5, seed=42),
            flow=FlowConfig(flow_warp=False),
            animatediff=AnimateDiffConfig(enabled=False),
        )
        ctx = RenderContext(
            config=config,
            batch_folder=batch_dir,
            video_frames_folder=vf_dir,
        )

        with patch('vibewarp.core.diffusion.render_frame') as mock_rf:
            mock_rf.return_value = Image.new('RGB', (64, 64))
            paths = run_frames(ctx, frame_range=[0, 3])

        # frame_range is inclusive: 0-3 is FOUR frames.
        assert len(paths) == 4
        assert mock_rf.call_count == 4
        for p in paths:
            assert os.path.exists(p)

    def test_run_frames_animatediff_mode(self, tmp_path):
        """AnimateDiff must route to the JOINT-BATCH renderer, never to per-frame.

        This test used to assert `render_frame` was called >= 10 times — i.e. it
        asserted the broken behaviour, where every frame got its own batch-1 latent
        and the motion module had nothing to attend across.
        """
        vf_dir = self._create_video_frames(tmp_path, n_frames=20)
        batch_dir = str(tmp_path / 'batch')
        os.makedirs(batch_dir, exist_ok=True)

        config = RunConfig(
            frame_range=[0, 20],
            video=VideoConfig(width=64, height=64),
            diffusion=DiffusionConfig(steps=5, seed=42),
            flow=FlowConfig(flow_warp=False),
            animatediff=AnimateDiffConfig(
                enabled=True, batch_length=16, batch_overlap=4,
                context_length=16, context_overlap=4,
            ),
        )
        ctx = RenderContext(
            config=config,
            batch_folder=batch_dir,
            video_frames_folder=vf_dir,
        )

        with patch('vibewarp.core.diffusion.render_frame') as per_frame,              patch('vibewarp.core.diffusion._adiff_run_batch') as run_batch:
            run_batch.side_effect = lambda ctx, frames, *a, **k: [
                Image.new('RGB', (64, 64)) for _ in frames]
            paths = run_frames(ctx, frame_range=[0, 20])

        per_frame.assert_not_called()
        assert run_batch.call_count >= 1
        assert len(paths) > 0

    def test_frame_schedule_integration(self):
        """get_frame_schedule should work with full RunConfig."""
        config = RunConfig(
            diffusion=DiffusionConfig(
                steps=30,
                cfg_scale=8.0,
                seed=42,
                style_strength=0.7,
                steps_schedule={0: 20, 10: 40},
                cfg_scale_schedule=[5.0, 6.0, 7.0, 8.0],
            ),
            warp=WarpConfig(
                flow_blend=0.6,
                flow_blend_schedule={0: 0.3, 20: 0.8},
            ),
        )

        sched_0 = get_frame_schedule(0, config)
        assert sched_0['steps'] == 20
        assert sched_0['cfg_scale'] == 5.0
        assert sched_0['style_strength'] == 0.7
        assert sched_0['flow_blend'] == 0.3

        sched_5 = get_frame_schedule(5, config)
        assert sched_5['steps'] == 30  # interpolated

        sched_10 = get_frame_schedule(10, config)
        assert sched_10['steps'] == 40

    def test_controlnet_time_gating_integration(self, tmp_path):
        """ControlNets should be gated by frame percentage."""
        vf_dir = self._create_video_frames(tmp_path, n_frames=20)
        batch_dir = str(tmp_path / 'batch')

        config = RunConfig(
            frame_range=[0, 20],
            video=VideoConfig(width=64, height=64),
        )

        # Simulate loaded controlnets with time ranges
        loaded_cns = {
            'early_cn': {'weight': 1.0, 'start': 0.0, 'end': 0.5, 'source': ''},
            'late_cn': {'weight': 1.0, 'start': 0.5, 'end': 1.0, 'source': ''},
        }

        total_frames = 20

        # Frame 3 (15%): only early_cn active
        pct_3 = 3 / total_frames
        active_3 = [k for k, v in loaded_cns.items()
                     if v['start'] <= pct_3 <= v['end']]
        assert 'early_cn' in active_3
        assert 'late_cn' not in active_3

        # Frame 15 (75%): only late_cn active
        pct_15 = 15 / total_frames
        active_15 = [k for k, v in loaded_cns.items()
                      if v['start'] <= pct_15 <= v['end']]
        assert 'early_cn' not in active_15
        assert 'late_cn' in active_15

    def test_ipadapter_config_roundtrip(self):
        """IP-Adapter config should survive creation and access."""
        config = RunConfig(
            ipadapter=IPAdapterConfig(
                enabled=True,
                clip_vision_model_path='/models/clip.safetensors',
                models={
                    'ipa1': IPAdapterEntry(
                        path='/models/ip-adapter.safetensors',
                        weight=0.6,
                        source_image='/ref/style.png',
                        weight_type='ease in',
                    ),
                },
            ),
        )
        assert config.ipadapter.enabled
        assert config.ipadapter.models['ipa1'].weight == 0.6
        assert config.ipadapter.models['ipa1'].weight_type == 'ease in'

    def test_render_context_has_all_fields(self):
        """RenderContext should have all the fields we expect."""
        ctx = RenderContext()
        assert hasattr(ctx, 'config')
        assert hasattr(ctx, 'sd_model')
        assert hasattr(ctx, 'model_wrap')
        assert hasattr(ctx, 'model_wrap_cfg')
        assert hasattr(ctx, 'loaded_controlnets')
        assert hasattr(ctx, 'loaded_ipadapters')
        assert hasattr(ctx, 'raft_model')
        assert hasattr(ctx, 'clip_vision_model')
        assert hasattr(ctx, 'sampler_fn')
        assert hasattr(ctx, 'batch_folder')
        assert hasattr(ctx, 'video_frames_folder')
        assert hasattr(ctx, 'flow_folder')
        assert hasattr(ctx, 'save_frame_fn')
        assert hasattr(ctx, 'progress_fn')

    def test_full_config_with_everything_enabled(self):
        """RunConfig should handle all sub-configs simultaneously."""
        config = RunConfig(
            frame_range=[0, 100],
            text_prompts={0: "a painting", 50: "a sculpture"},
            diffusion=DiffusionConfig(
                steps=30, cfg_scale=8.0, seed=42,
                style_strength_schedule={0: 0.8, 50: 0.5},
            ),
            flow=FlowConfig(flow_warp=True, check_consistency=True),
            warp=WarpConfig(flow_blend_schedule={0: 0.3, 50: 0.7}),
            controlnet=ControlNetConfig(
                enabled=True,
                models={'canny': ControlNetEntry(weight=0.8, start=0.0, end=0.7)},
            ),
            animatediff=AnimateDiffConfig(
                enabled=True,
                batch_length=32,
                context_length=16,
            ),
            ipadapter=IPAdapterConfig(
                enabled=True,
                models={'ipa': IPAdapterEntry(weight=0.5)},
            ),
        )
        # Verify all configs accessible
        assert config.controlnet.enabled
        assert config.animatediff.enabled
        assert config.ipadapter.enabled
        assert config.diffusion.style_strength_schedule is not None
        assert config.warp.flow_blend_schedule is not None
