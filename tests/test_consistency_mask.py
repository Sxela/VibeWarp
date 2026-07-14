"""Tests for consistency mask post-processing — load_cc and per-frame scheduling."""

import numpy as np
import pytest


# ---- load_cc component weights ----

class TestLoadCcWeights:
    """load_cc correctly separates missed/overshoot/edge layers and weights them."""

    def _make_cc(self, r, g, b):
        """Make a 4x4 uint8 CC map with constant RGB channels."""
        cc = np.zeros((4, 4, 3), dtype=np.uint8)
        cc[..., 0] = r
        cc[..., 1] = g
        cc[..., 2] = b
        return cc

    def test_all_white_all_weights_one_gives_ones(self):
        from vibewarp.flow.consistency import load_cc
        cc = self._make_cc(255, 255, 255)
        w = load_cc(cc, missed_consistency_weight=1.0,
                    overshoot_consistency_weight=1.0, edges_consistency_weight=1.0,
                    blur=0, dilate=0)
        assert w.shape == (4, 4, 3)
        assert np.allclose(w, 1.0)

    def test_all_black_all_weights_one_gives_zeros(self):
        from vibewarp.flow.consistency import load_cc
        cc = self._make_cc(0, 0, 0)
        w = load_cc(cc, missed_consistency_weight=1.0,
                    overshoot_consistency_weight=1.0, edges_consistency_weight=1.0,
                    blur=0, dilate=0)
        assert np.allclose(w, 0.0)

    def test_zero_missed_weight_ignores_r_channel(self):
        """missed_consistency_weight=0 → R channel has no effect → weights stay 1."""
        from vibewarp.flow.consistency import load_cc
        cc = self._make_cc(0, 255, 255)  # R=0 (all inconsistent), G/B=255 (all good)
        # With weight=0, R channel clipped to [1,1] → no effect on weights
        w = load_cc(cc, missed_consistency_weight=0.0,
                    overshoot_consistency_weight=1.0, edges_consistency_weight=1.0,
                    blur=0, dilate=0)
        assert np.allclose(w, 1.0)

    def test_full_missed_weight_makes_black_r_zero(self):
        """missed_consistency_weight=1 → black R channel → weight=0 after threshold."""
        from vibewarp.flow.consistency import load_cc
        cc = self._make_cc(0, 255, 255)  # R=0 (missed)
        w = load_cc(cc, missed_consistency_weight=1.0,
                    overshoot_consistency_weight=1.0, edges_consistency_weight=1.0,
                    blur=0, dilate=0)
        assert np.allclose(w, 0.0)

    def test_zero_overshoot_weight_ignores_g_channel(self):
        from vibewarp.flow.consistency import load_cc
        cc = self._make_cc(255, 0, 255)  # G=0 but we ignore it
        w = load_cc(cc, missed_consistency_weight=1.0,
                    overshoot_consistency_weight=0.0, edges_consistency_weight=1.0,
                    blur=0, dilate=0)
        assert np.allclose(w, 1.0)

    def test_zero_edges_weight_ignores_b_channel(self):
        from vibewarp.flow.consistency import load_cc
        cc = self._make_cc(255, 255, 0)  # B=0 but we ignore it
        w = load_cc(cc, missed_consistency_weight=1.0,
                    overshoot_consistency_weight=1.0, edges_consistency_weight=0.0,
                    blur=0, dilate=0)
        assert np.allclose(w, 1.0)

    def test_blur_applied(self):
        """Blur > 0 should soften sharp edges in the mask."""
        from vibewarp.flow.consistency import load_cc
        cc = np.zeros((10, 10, 3), dtype=np.uint8)
        cc[5:, :] = 255  # hard edge
        w_no_blur = load_cc(cc, blur=0, dilate=0)
        w_blur = load_cc(cc, blur=2, dilate=0)
        # Blurred version should have intermediate values near the edge
        assert not np.allclose(w_no_blur, w_blur)

    def test_output_shape_3channel(self):
        from vibewarp.flow.consistency import load_cc
        cc = self._make_cc(128, 128, 128)
        w = load_cc(cc, blur=0, dilate=0)
        assert w.ndim == 3
        assert w.shape[2] == 3


# ---- soften_consistency_mask in warp_frame ----

class TestSoftenConsistencyMask:
    """forward_clip clips the minimum value of CC weights in warp_frame."""

    def _make_warp_inputs(self, h=8, w=8):
        import numpy as np
        from PIL import Image
        frame1 = Image.fromarray(np.full((h, w, 3), 100, dtype=np.uint8))
        frame2 = Image.fromarray(np.full((h, w, 3), 200, dtype=np.uint8))
        flow = np.zeros((h, w, 2), dtype=np.float32)
        weights = np.zeros((h, w, 3), dtype=np.float32)  # all inconsistent
        return frame1, frame2, flow, weights

    def test_zero_clip_uses_raw_weights(self):
        """With forward_clip=0, zero weights → output = frame2 (no warped contribution)."""
        from vibewarp.flow.warp import warp_frame
        import numpy as np
        from PIL import Image
        frame1, frame2, flow, weights = self._make_warp_inputs()
        # blend=1 means: warped*weights + frame2*(1-weights); weights=0 → frame2 only
        result = warp_frame(frame1, frame2, flow, blend=1.0, weights=weights, forward_clip=0.0)
        arr = np.array(result)
        assert np.allclose(arr, 200, atol=2)

    def test_clip_at_one_ignores_weights_completely(self):
        """forward_clip=1 → weights.clip(1,1)=1 → output = warped frame (frame1 value)."""
        from vibewarp.flow.warp import warp_frame
        import numpy as np
        frame1, frame2, flow, weights = self._make_warp_inputs()
        result = warp_frame(frame1, frame2, flow, blend=1.0, weights=weights, forward_clip=1.0)
        arr = np.array(result)
        # When weights=1 everywhere: blended = frame2*(1-blend) + blend*(warped*1 + frame2*0)
        #                          = frame2*0 + warped = frame1 ≈ 100
        assert np.allclose(arr, 100, atol=2)

    def test_partial_clip_raises_weight_floor(self):
        """forward_clip=0.5 → weights.clip(0.5,1) → blended between frame1 and frame2."""
        from vibewarp.flow.warp import warp_frame
        import numpy as np
        frame1, frame2, flow, weights = self._make_warp_inputs()
        result_0 = warp_frame(frame1, frame2, flow, blend=1.0, weights=weights, forward_clip=0.0)
        result_1 = warp_frame(frame1, frame2, flow, blend=1.0, weights=weights, forward_clip=1.0)
        result_half = warp_frame(frame1, frame2, flow, blend=1.0, weights=weights, forward_clip=0.5)
        arr_0 = np.array(result_0).mean()
        arr_1 = np.array(result_1).mean()
        arr_half = np.array(result_half).mean()
        # Higher clip → more warped (frame1=100), lower value; lower clip → more frame2 (200)
        assert arr_1 < arr_half < arr_0


# ---- get_frame_schedule includes consistency params ----

class TestConsistencyScheduleInFrameSchedule:
    def _make_config(self, **flow_kwargs):
        from vibewarp.config import RunConfig, FlowConfig
        cfg = RunConfig()
        for k, v in flow_kwargs.items():
            setattr(cfg.flow, k, v)
        return cfg

    def test_static_values_passed_through(self):
        from vibewarp.core.diffusion import get_frame_schedule
        cfg = self._make_config(
            missed_consistency_weight=0.5,
            overshoot_consistency_weight=0.3,
            edges_consistency_weight=0.7,
            consistency_blur=2,
            consistency_dilate=5,
            soften_consistency_mask=0.1,
        )
        sched = get_frame_schedule(0, cfg)
        assert sched['missed_consistency_weight'] == pytest.approx(0.5)
        assert sched['overshoot_consistency_weight'] == pytest.approx(0.3)
        assert sched['edges_consistency_weight'] == pytest.approx(0.7)
        assert sched['consistency_blur'] == 2
        assert sched['consistency_dilate'] == 5
        assert sched['soften_consistency_mask'] == pytest.approx(0.1)

    def test_schedule_overrides_static(self):
        from vibewarp.core.diffusion import get_frame_schedule
        # Dict schedule: {0: 0.8} overrides static missed_consistency_weight=0.5
        cfg = self._make_config(
            missed_consistency_weight=0.5,
            missed_consistency_schedule={0: 0.8, 30: 0.2},
        )
        sched_0 = get_frame_schedule(0, cfg)
        sched_30 = get_frame_schedule(30, cfg)
        assert sched_0['missed_consistency_weight'] == pytest.approx(0.8)
        assert sched_30['missed_consistency_weight'] == pytest.approx(0.2)

    def test_soften_schedule_resolved(self):
        from vibewarp.core.diffusion import get_frame_schedule
        cfg = self._make_config(
            soften_consistency_mask=0.0,
            soften_consistency_schedule={0: 0.0, 10: 0.5},
        )
        sched = get_frame_schedule(10, cfg)
        assert sched['soften_consistency_mask'] == pytest.approx(0.5)

    def test_default_soften_is_zero(self):
        from vibewarp.config import FlowConfig
        fc = FlowConfig()
        assert fc.soften_consistency_mask == 0.0


# ---- settings mapping ----

class TestConsistencySettingsMapping:
    def test_soften_consistency_mapped(self, tmp_path):
        import json
        from vibewarp.settings import load_warpfusion_settings
        path = str(tmp_path / "s.txt")
        with open(path, 'w') as f:
            json.dump({
                "model_path": "m.safetensors",
                "soften_consistency_mask": 0.3,
                "text_prompts": {"0": ["a painting"]},
                "negative_prompts": {"0": [""]},
            }, f)
        cfg = load_warpfusion_settings(path)
        assert cfg['flow']['soften_consistency_mask'] == pytest.approx(0.3)

    def test_consistency_schedules_mapped(self, tmp_path):
        import json
        from vibewarp.settings import load_warpfusion_settings
        path = str(tmp_path / "s.txt")
        sched = {"0": 1.0, "30": 0.5}
        with open(path, 'w') as f:
            json.dump({
                "model_path": "m.safetensors",
                "missed_consistency_schedule": sched,
                "soften_consistency_schedule": [0.1],
                "text_prompts": {"0": ["a painting"]},
                "negative_prompts": {"0": [""]},
            }, f)
        cfg = load_warpfusion_settings(path)
        assert cfg['flow']['missed_consistency_schedule'] == sched
        assert cfg['flow']['soften_consistency_schedule'] == [0.1]

    def test_default_soften_when_absent(self, tmp_path):
        import json
        from vibewarp.settings import load_warpfusion_settings
        path = str(tmp_path / "s.txt")
        with open(path, 'w') as f:
            json.dump({
                "model_path": "m.safetensors",
                "text_prompts": {"0": ["a painting"]},
                "negative_prompts": {"0": [""]},
            }, f)
        cfg = load_warpfusion_settings(path)
        assert cfg['flow']['soften_consistency_mask'] == 0.0
        assert cfg['flow']['missed_consistency_schedule'] is None
