"""Tests for content-aware scheduling (notebook cells 26–28): frame-diff
analysis (rmse path, CPU), scene-split detection, notebook-exact schedule
adjustment, template application to config, and settings mapping."""

import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from vibewarp.config import RunConfig, SceneConfig
from vibewarp.core.scene_detect import (
    adjust_schedule,
    analyze_video_frames,
    apply_content_schedules,
    check_and_adjust_sched,
    find_scene_splits,
    load_img_lpips,
    rmse,
)


class TestDiffFunctions:
    def test_rmse_is_abs_mean_diff(self):
        # notebook quirk: 'rmse' = |mean(x-y)|, NOT root-mean-square
        x = torch.zeros(1, 3, 4, 4)
        y = torch.full((1, 3, 4, 4), 2.0)
        assert torch.isclose(rmse(x, y), torch.tensor(2.0))
        # signed diffs cancel under mean
        y2 = torch.zeros(1, 3, 4, 4)
        y2[0, 0] = 1.0
        y2[0, 1] = -1.0
        assert torch.isclose(rmse(x, y2), torch.tensor(0.0))

    def test_load_img_lpips_normalization(self, tmp_path):
        p = str(tmp_path / 'f.jpg')
        Image.fromarray(np.full((16, 16, 3), 127, dtype=np.uint8)).save(p)
        t = load_img_lpips(p, size=(8, 8), device='cpu')
        assert t.shape == (1, 3, 8, 8)
        # /127 puts mid-gray at ~1.0, then CLIP-normalize: (1-0.481)/0.269 ≈ 1.9
        assert 1.5 < t[0, 0].mean().item() < 2.3


class TestAnalyzeVideo:
    def _frames(self, tmp_path, colors):
        folder = tmp_path / 'frames'
        folder.mkdir()
        for i, c in enumerate(colors, start=1):
            Image.fromarray(np.full((16, 16, 3), c, dtype=np.uint8)).save(
                str(folder / f'{i:06d}.jpg'))
        return str(folder)

    def test_rmse_analysis_detects_cut(self, tmp_path):
        # frames: dark, dark, BRIGHT (cut), bright
        folder = self._frames(tmp_path, [10, 10, 240, 240])
        diff = analyze_video_frames(folder, diff_function='rmse', device='cpu')
        assert len(diff) == 4
        assert diff[0] == 0.0
        assert diff[2] > diff[1] * 5  # the cut dominates
        assert diff[2] > diff[3] * 5

    def test_single_frame(self, tmp_path):
        folder = self._frames(tmp_path, [10])
        assert analyze_video_frames(folder, diff_function='rmse', device='cpu') == [0.0]


class TestFindSceneSplits:
    def test_peaks(self):
        assert find_scene_splits([0, 0.1, 0.9, 0.2, 0.7], 0.5) == [2, 4]


class TestAdjustSchedule:
    def test_frame0_always_scene_start(self):
        out = adjust_schedule([0, 0, 0], normal_val=0.8, new_scene_val=0.0,
                              thresh=0.5, falloff_frames=0)
        assert out[0] == 0.0
        assert out[1] == 0.8 and out[2] == 0.8

    def test_cut_with_falloff(self):
        # cut at index 3, falloff over 2 frames: [new, new*1/2 + normal*1/2]
        diff = [0, 0, 0, 0.9, 0, 0]
        out = adjust_schedule(diff, normal_val=0.8, new_scene_val=0.0,
                              thresh=0.5, falloff_frames=2)
        assert out[3] == 0.0
        assert out[4] == pytest.approx(0.4)  # 0*1/2 + 0.8*1/2
        assert out[5] == pytest.approx(0.8)

    def test_respects_existing_schedule(self):
        # sched value used off-peak; peak overridden
        diff = [0, 0, 0.9, 0]
        sched = [0.5]  # constant schedule
        out = adjust_schedule(diff, normal_val=0.8, new_scene_val=0.1,
                              thresh=0.5, falloff_frames=0, sched=sched)
        assert out[1] == pytest.approx(0.5)   # from sched, not normal_val
        assert out[2] == pytest.approx(0.1)   # peak
        assert out[3] == pytest.approx(0.5)

    def test_falloff_clipped_at_end(self):
        diff = [0, 0.9]
        out = adjust_schedule(diff, normal_val=1.0, new_scene_val=0.0,
                              thresh=0.5, falloff_frames=5)
        assert len(out) == 2  # no out-of-bounds writes


class TestCheckAndAdjustSched:
    def test_empty_template_passthrough(self):
        sched = [1, 2, 3]
        for template in (None, '', []):
            assert check_and_adjust_sched(sched, template, [0, 0.9, 0]) is sched

    def test_returns_rounded_list(self):
        out = check_and_adjust_sched(None, [0.8, 0.0, 0.5, 3], [0, 0, 0.9, 0, 0],
                                     respect_sched=False)
        assert isinstance(out, list)
        assert len(out) == 5
        assert out[2] == 0.0
        assert out[3] == pytest.approx(0.267)  # 0*(2/3) + 0.8*(1/3), rounded


class TestApplyContentSchedules:
    def test_applies_supported_templates(self, capsys):
        cfg = RunConfig()
        cfg.scene = SceneConfig(
            make_schedules=True,
            flow_blend_template=[0.8, 0.0, 0.5, 2],
            steps_template=[25, 40, 0.5, 0],
            cfg_scale_template=None,
            latent_scale_template=[1, 2, 0.5, 0],  # unsupported → warn
        )
        diff = [0, 0, 0.9, 0]
        apply_content_schedules(cfg, diff)
        fb = cfg.warp.flow_blend_schedule
        # frame 0 is a scene start with falloff: [0.0, blend, cut→0.0, blend]
        assert isinstance(fb, list)
        assert fb[0] == 0.0 and fb[2] == 0.0
        assert fb[1] == pytest.approx(0.4) and fb[3] == pytest.approx(0.4)
        st = cfg.diffusion.steps_schedule
        assert st[2] == 40 and st[1] == 25
        assert cfg.diffusion.cfg_scale_schedule is None  # untouched
        assert 'latent_scale' in capsys.readouterr().out  # warned

    def test_respect_sched_keeps_existing_offpeak(self):
        cfg = RunConfig()
        cfg.diffusion.steps_schedule = [30]
        cfg.scene = SceneConfig(make_schedules=True, respect_sched=True,
                                flow_blend_template=None,
                                cc_masked_template=None,
                                steps_template=[25, 50, 0.5, 0])
        apply_content_schedules(cfg, [0, 0, 0.9, 0])
        st = cfg.diffusion.steps_schedule
        assert st[1] == 30  # existing schedule respected off-peak
        assert st[2] == 50  # peak overridden


class TestCaptioningIntegration:
    def test_content_aware_keyframes_from_diffs(self):
        from vibewarp.core.captioning import select_keyframes
        diffs = [0, 0.1, 0.9, 0.1, 0.6]
        assert select_keyframes(
            100, 'Content-aware scheduling keyframes',
            diffs=diffs, diff_thresh=0.5) == [1, 3, 5]


class TestConfigAndSettings:
    def test_defaults_match_notebook(self):
        c = SceneConfig()
        assert c.analyze_video is False
        assert c.diff_function == 'rmse+lpips'
        assert c.make_schedules is False
        assert c.respect_sched is True
        assert c.flow_blend_template == [0.8, 0.0, 0.51, 2]
        assert c.cc_masked_template == [0.7, 0, 0.51, 2]
        assert c.steps_template is None
        assert isinstance(RunConfig().scene, SceneConfig)

    def test_settings_mapping(self, tmp_path):
        from vibewarp.settings import load_warpfusion_settings
        p = tmp_path / 's.txt'
        p.write_text(json.dumps({
            'analyze_video': True,
            'diff_function': 'rmse',
            'make_schedules': True,
            'respect_sched': False,
            'diff_override': [0, 0.9],
            'flow_blend_template': [0.9, 0.1, 0.4, 3],
            'steps_template': [25, 40, 0.4, 1],
        }))
        cfg = load_warpfusion_settings(str(p))
        s = cfg['scene']
        assert s['analyze_video'] is True
        assert s['diff_function'] == 'rmse'
        assert s['make_schedules'] is True
        assert s['respect_sched'] is False
        assert s['diff_override'] == [0, 0.9]
        assert s['flow_blend_template'] == [0.9, 0.1, 0.4, 3]
        assert s['steps_template'] == [25, 40, 0.4, 1]
