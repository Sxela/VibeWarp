"""Tests for the new annotator routes: temporalnet, inpaint conditioning,
depth-detector selection (Midas/Zoe/depth_anything) and the depth_anything
CN heatmap. Detector inference is stubbed — no model downloads."""

import numpy as np
import pytest
import torch

import vibewarp.core.controlnet as cn
from vibewarp.core.controlnet import (
    ANNOTATOR_MAP,
    INPAINT_CN_KEYS,
    annotate_image,
    map_to_conditioning,
)


IMG = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)


class TestMapEntries:
    def test_temporalnet_mapped(self):
        assert ANNOTATOR_MAP['control_sd15_temporalnet'] == 'temporalnet'
        assert ANNOTATOR_MAP['control_sdxl_temporalnet_v1'] == 'temporalnet'

    def test_depth_anything_mapped_to_depth(self):
        assert ANNOTATOR_MAP['control_sd15_depth_anything'] == 'depth'

    def test_inpaint_keys(self):
        assert 'control_sd15_inpaint' in INPAINT_CN_KEYS
        assert 'control_sdxl_inpaint' in INPAINT_CN_KEYS
        assert ANNOTATOR_MAP['control_sdxl_inpaint'] == 'inpaint'


class TestTemporalnet:
    def test_passthrough(self):
        """temporalnet conditioning IS the (previous) frame — no detector."""
        out = annotate_image(IMG, 'temporalnet')
        assert out is IMG


class TestInpaint:
    def test_requires_mask(self):
        with pytest.raises(ValueError, match='inpaint_mask'):
            annotate_image(IMG, 'inpaint')

    def test_keep_all_mask_has_no_markers(self):
        mask = np.ones((64, 96), dtype=np.float32)
        out = annotate_image(IMG, 'inpaint', inpaint_mask=mask)
        assert out.dtype == np.float32
        assert (out >= 0).all()
        assert np.allclose(out, IMG.astype(np.float32))

    def test_regenerate_all_mask_is_all_markers(self):
        mask = np.zeros((64, 96), dtype=np.float32)
        out = annotate_image(IMG, 'inpaint', inpaint_mask=mask)
        assert (out == -255.0).all()

    def test_partial_mask_marks_region(self):
        mask = np.ones((64, 96), dtype=np.float32)
        mask[:32, :] = 0.0  # regenerate top half
        out = annotate_image(IMG, 'inpaint', inpaint_mask=mask)
        assert (out[:30, :] == -255.0).all()   # inside regen region
        assert (out[34:, :] >= 0).all()        # keep region untouched

    def test_uint8_and_hwc_masks_accepted(self):
        mask8 = np.full((64, 96, 3), 255, dtype=np.uint8)
        out = annotate_image(IMG, 'inpaint', inpaint_mask=mask8)
        assert (out >= 0).all()

    def test_marker_survives_conditioning(self):
        """-255 marker must become -1.0 in the conditioning tensor
        (notebook: 'use -1 as inpaint value')."""
        mask = np.zeros((64, 96), dtype=np.float32)
        out = annotate_image(IMG, 'inpaint', inpaint_mask=mask)
        cond = map_to_conditioning(out, 64, 96)
        assert torch.isclose(cond.min(), torch.tensor(-1.0))
        assert cond.shape == (1, 3, 64, 96)


class TestDepthDetectorSelection:
    def _stub(self, monkeypatch):
        calls = []
        monkeypatch.setattr(cn, '_run_detector',
                            lambda key, loader, img, res: calls.append(key) or
                            np.full((64, 96, 3), 128, dtype=np.uint8))
        monkeypatch.setattr(cn, '_annotate_depth_anything',
                            lambda img, res: calls.append('depth_anything') or
                            np.full((64, 96, 3), 128, dtype=np.uint8))
        return calls

    def test_midas_default(self, monkeypatch):
        calls = self._stub(monkeypatch)
        annotate_image(IMG, 'depth')
        assert calls == ['depth']

    def test_zoe(self, monkeypatch):
        calls = self._stub(monkeypatch)
        annotate_image(IMG, 'depth', depth_detector='Zoe')
        assert calls == ['depth_zoe']

    def test_depth_anything(self, monkeypatch):
        calls = self._stub(monkeypatch)
        annotate_image(IMG, 'depth', depth_detector='depth_anything')
        assert calls == ['depth_anything']

    def test_colored_depth_applies_inferno(self, monkeypatch):
        """depth_anything CN models get a heatmapped depth map (notebook:
        colored = 'depth_anything' in control_key), regardless of detector."""
        self._stub(monkeypatch)
        gray = annotate_image(IMG, 'depth')
        colored = annotate_image(IMG, 'depth', colored_depth=True)
        assert colored.shape == gray.shape
        assert colored.dtype == np.uint8
        # A flat gray map heatmaps to a single non-gray INFERNO color
        assert not np.array_equal(colored, gray)
        r, g, b = colored[0, 0]
        assert not (r == g == b)


class TestSegDetectorSelection:
    def _stub(self, monkeypatch):
        calls = []
        monkeypatch.setattr(cn, '_annotate_oneformer',
                            lambda img, res, dataset: calls.append(dataset) or
                            np.zeros((64, 96, 3), dtype=np.uint8))
        monkeypatch.setattr(cn, '_run_detector',
                            lambda key, loader, img, res: calls.append(key) or
                            np.zeros((64, 96, 3), dtype=np.uint8))
        return calls

    def test_default_ufade20k_routes_to_oneformer_ade20k(self, monkeypatch):
        """UniFormer isn't on PyPI — Seg_UFADE20K maps to OneFormer-ADE20K
        (same 150 classes, same palette)."""
        calls = self._stub(monkeypatch)
        annotate_image(IMG, 'seg')
        assert calls == ['ade20k']

    def test_ofade20k(self, monkeypatch):
        calls = self._stub(monkeypatch)
        annotate_image(IMG, 'seg', seg_detector='Seg_OFADE20K')
        assert calls == ['ade20k']

    def test_ofcoco(self, monkeypatch):
        calls = self._stub(monkeypatch)
        annotate_image(IMG, 'seg', seg_detector='Seg_OFCOCO')
        assert calls == ['coco']

    def test_sam_override(self, monkeypatch):
        calls = self._stub(monkeypatch)
        annotate_image(IMG, 'seg', seg_detector='Seg_SAM')
        assert calls == ['seg_sam']


class TestSegPalettes:
    def test_palette_sizes_and_reference_values(self):
        from vibewarp.vendor.seg_palettes import ADE20K_PALETTE, COCO_PALETTE
        assert len(ADE20K_PALETTE) == 150
        assert len(COCO_PALETTE) == 133
        # Spot-check against the reference clone's category lists
        assert ADE20K_PALETTE[0] == (120, 120, 120)   # wall
        assert ADE20K_PALETTE[2] == (6, 230, 230)     # sky
        assert COCO_PALETTE[0] == (220, 20, 60)       # person
        assert COCO_PALETTE[2] == (0, 0, 142)         # car

    def test_palette_rendering(self):
        """palette[class_map] produces the flat color-coded map the seg CN
        was trained on."""
        from vibewarp.vendor.seg_palettes import ADE20K_PALETTE
        palette = np.array(ADE20K_PALETTE, dtype=np.uint8)
        class_map = np.zeros((4, 4), dtype=np.int64)
        class_map[2:, :] = 2  # bottom half = sky
        colored = palette[class_map]
        assert colored.shape == (4, 4, 3)
        assert tuple(colored[0, 0]) == (120, 120, 120)
        assert tuple(colored[3, 0]) == (6, 230, 230)


class TestPrepareHintsSkipLogic:
    def test_inpaint_skipped_without_mask(self, monkeypatch, tmp_path):
        """prepare_cn_hints_for_frame drops inpaint CNs when there is no mask
        for the frame (notebook removes the model from the list)."""
        from PIL import Image as PILImage
        src = tmp_path / 'src.jpg'
        PILImage.fromarray(IMG).save(str(src))

        class FakeModel:
            pass

        sd_model = FakeModel()
        loaded = {'control_sd15_inpaint': {'weight': 1.0}}
        cn.prepare_cn_hints_for_frame(
            sd_model, loaded, {'control_sd15_inpaint': str(src)},
            target_h=64, target_w=96, inpaint_mask=None,
        )
        assert sd_model._cn_hints == {}


class TestSettingsMapping:
    def test_new_controlnet_settings(self, tmp_path):
        import json
        from vibewarp.settings import load_warpfusion_settings
        raw = {
            'control_sd15_depth_detector': 'Zoe',
            'control_sd15_seg_detector': 'Seg_OFCOCO',
            'temporalnet_source': 'stylized',
            'temporalnet_skip_1st_frame': False,
            'control_sd15_inpaint_mask_source': 'none',
        }
        p = tmp_path / 'settings.txt'
        p.write_text(json.dumps(raw))
        cfg = load_warpfusion_settings(str(p))
        assert cfg['controlnet']['depth_detector'] == 'Zoe'
        assert cfg['controlnet']['seg_detector'] == 'Seg_OFCOCO'
        assert cfg['controlnet']['temporalnet_source'] == 'stylized'
        assert cfg['controlnet']['temporalnet_skip_1st_frame'] is False
        assert cfg['controlnet']['inpaint_mask_source'] == 'none'

    def test_defaults(self, tmp_path):
        import json
        from vibewarp.settings import load_warpfusion_settings
        p = tmp_path / 'settings.txt'
        p.write_text(json.dumps({}))
        cfg = load_warpfusion_settings(str(p))
        assert cfg['controlnet']['depth_detector'] == 'Midas'
        assert cfg['controlnet']['seg_detector'] == 'Seg_UFADE20K'
        assert cfg['controlnet']['temporalnet_source'] == 'init'
        assert cfg['controlnet']['temporalnet_skip_1st_frame'] is True
        assert cfg['controlnet']['inpaint_mask_source'] == 'consistency_mask'

    def test_config_dataclass_fields(self):
        from vibewarp.config import ControlNetConfig
        c = ControlNetConfig()
        assert c.depth_detector == 'Midas'
        assert c.temporalnet_source == 'init'
        assert c.temporalnet_skip_1st_frame is True
        assert c.inpaint_mask_source == 'consistency_mask'


class TestAnnotatorFollowUps:
    """2026-07-11 batch: scribble post-pass, normalbae flip, canny/mlsd
    thresholds, qr/tile mask options."""

    def test_normalbae_output_is_flipped(self, monkeypatch):
        base = np.zeros((8, 8, 3), dtype=np.uint8)
        base[..., 0] = 10  # R channel marker
        monkeypatch.setattr(cn, '_run_detector',
                            lambda key, loader, img, res, **kw: base.copy())
        out = annotate_image(IMG, 'normalbae')
        # notebook [:, :, ::-1]: R marker ends up in B
        assert out[0, 0, 2] == 10 and out[0, 0, 0] == 0

    def test_canny_thresholds_forwarded(self, monkeypatch):
        seen = {}

        def fake_canny(img, res, low=100, high=200):
            seen['low'], seen['high'] = low, high
            return np.zeros((8, 8, 3), dtype=np.uint8)

        monkeypatch.setattr(cn, '_annotate_canny', fake_canny)
        annotate_image(IMG, 'canny', opts={'canny_low_threshold': 50,
                                           'canny_high_threshold': 150})
        assert (seen['low'], seen['high']) == (50, 150)

    def test_mlsd_thresholds_forwarded(self, monkeypatch):
        seen = {}

        def fake_run(key, loader, img, res, **kw):
            seen.update(kw)
            return np.zeros((8, 8, 3), dtype=np.uint8)

        monkeypatch.setattr(cn, '_run_detector', fake_run)
        annotate_image(IMG, 'mlsd', opts={'mlsd_value_threshold': 0.2,
                                          'mlsd_distance_threshold': 0.3})
        assert seen == {'thr_v': 0.2, 'thr_d': 0.3}

    def test_scribble_postpass_binarizes(self, monkeypatch):
        # fake detector returns a soft-edge map; post-pass must yield a
        # strictly binary {0, 255} map
        soft = np.zeros((32, 32, 3), dtype=np.uint8)
        soft[4:28, 12:22, :] = 255  # a thick vertical stroke (survives sigma-3 blur)
        monkeypatch.setattr(cn, '_run_detector',
                            lambda key, loader, img, res, **kw: soft)
        out = annotate_image(IMG, 'scribble')
        vals = np.unique(out)
        assert set(vals.tolist()) <= {0, 255}
        assert (out == 255).any()  # the line survives

    def test_scribble_detector_select(self, monkeypatch):
        calls = []
        monkeypatch.setattr(cn, '_run_detector',
                            lambda key, loader, img, res, **kw: calls.append(key) or
                            np.zeros((16, 16, 3), dtype=np.uint8))
        annotate_image(IMG, 'scribble')
        annotate_image(IMG, 'scribble', opts={'scribble_detector': 'HED'})
        assert calls == ['scribble', 'scribble_hed']

    def test_softedge_detector_select(self, monkeypatch):
        calls = []
        monkeypatch.setattr(cn, '_run_detector',
                            lambda key, loader, img, res, **kw: calls.append(key) or
                            np.zeros((16, 16, 3), dtype=np.uint8))
        annotate_image(IMG, 'softedge')
        annotate_image(IMG, 'softedge', opts={'softedge_detector': 'HED'})
        assert calls == ['softedge', 'softedge_hed']


class TestTileQrOptions:
    def test_default_is_value_passthrough(self):
        out = annotate_image(IMG, 'tile')
        assert np.array_equal(out, IMG)

    def test_invert(self):
        out = annotate_image(IMG, 'tile', opts={'qr_cn_mask_invert': True})
        assert np.array_equal(out, 255 - IMG)

    def test_grayscale_is_3ch(self):
        out = annotate_image(IMG, 'tile', opts={'qr_cn_mask_grayscale': True})
        assert out.shape[2] == 3
        assert np.array_equal(out[..., 0], out[..., 1])

    def test_thresh_binarizes(self):
        out = annotate_image(IMG, 'tile', opts={'qr_cn_mask_thresh': 128})
        assert set(np.unique(out).tolist()) <= {0, 255}

    def test_clip(self):
        out = annotate_image(IMG, 'tile', opts={'qr_cn_mask_clip_low': 50,
                                                'qr_cn_mask_clip_high': 200})
        assert out.min() >= 50 and out.max() <= 200

    def test_downscale_only_for_tile_models(self):
        big = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        # non-tile (qr) key: no downscale
        out_qr = annotate_image(big, 'tile', opts={'downscale_tile': 4,
                                                   'is_tile_model': False})
        assert out_qr.shape == big.shape
        # tile model: downscaled (blur-via-downscale), mask ops discarded
        out_tile = annotate_image(big, 'tile', opts={'downscale_tile': 4,
                                                     'is_tile_model': True,
                                                     'qr_cn_mask_invert': True})
        assert out_tile.shape[0] < big.shape[0]
        # downscale branch uses the RAW input (notebook quirk): not inverted
        assert out_tile.mean() == pytest.approx(big.mean(), abs=25)

    def test_qr_keys_mapped_to_tile(self):
        for k in ['control_sd15_qr', 'control_sd15_monster_qr',
                  'control_sd21_qr', 'control_sd15_gif',
                  'control_sdxl_xinsir_tile']:
            assert ANNOTATOR_MAP[k] == 'tile'
        from vibewarp.core.controlnet import TILE_CN_KEYS
        assert 'control_sd15_qr' not in TILE_CN_KEYS
        assert 'control_sd15_tile' in TILE_CN_KEYS


class TestAnnotatorOptsSettings:
    def test_settings_mapping(self, tmp_path):
        import json
        from vibewarp.settings import load_warpfusion_settings
        p = tmp_path / 's.txt'
        p.write_text(json.dumps({
            'control_sd15_scribble_detector': 'HED',
            'control_sd15_softedge_detector': 'HED',
            'low_threshold': 50, 'high_threshold': 150,
            'value_threshold': 0.2, 'distance_threshold': 0.3,
            'qr_cn_mask_invert': True, 'qr_cn_mask_thresh': 100,
            'downscale_tile': 2,
        }))
        c = load_warpfusion_settings(str(p))['controlnet']
        assert c['scribble_detector'] == 'HED'
        assert c['softedge_detector'] == 'HED'
        assert c['canny_low_threshold'] == 50
        assert c['canny_high_threshold'] == 150
        assert c['mlsd_value_threshold'] == 0.2
        assert c['mlsd_distance_threshold'] == 0.3
        assert c['qr_cn_mask_invert'] is True
        assert c['qr_cn_mask_thresh'] == 100
        assert c['downscale_tile'] == 2

    def test_config_defaults(self):
        from vibewarp.config import ControlNetConfig
        c = ControlNetConfig()
        assert c.scribble_detector == 'PIDI'
        assert c.softedge_detector == 'PIDI'
        assert (c.canny_low_threshold, c.canny_high_threshold) == (100, 200)
        assert (c.mlsd_value_threshold, c.mlsd_distance_threshold) == (0.1, 0.1)
        assert c.downscale_tile == 4
        assert c.qr_cn_mask_thresh == 0
