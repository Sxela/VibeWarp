"""Tests for vibewarp.core.controlnet — annotation dispatch, caching, conditioning."""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import torch
from PIL import Image

from vibewarp.core.controlnet import (
    ANNOTATOR_MAP,
    _PASSTHROUGH_ANNOTATORS,
    _annotator_cache,
    _split_text_and_vector_conditioning,
    annotate_image,
    apply_layer_weights,
    clear_annotator_cache,
    get_controlnet_conditioning,
    map_to_conditioning,
    normalize_controlnet_weights,
)


class TestTextAndVectorConditioning:
    def test_sdxl_dict_preserves_explicit_vector(self):
        sd_model = MagicMock()
        del sd_model.conditioner
        cross = torch.randn(2, 77, 16)
        vector = torch.randn(2, 8)
        text, pooled = _split_text_and_vector_conditioning(
            sd_model, {'c_crossattn': [cross], 'y': vector})
        assert torch.equal(text, cross)
        assert pooled is vector

    def test_sdxl_tensor_uses_conditioner_vector(self):
        class Conditioner:
            vector_in = torch.randn(2, 8)
        class Model:
            conditioner = Conditioner()
        cross = torch.randn(2, 77, 16)
        text, pooled = _split_text_and_vector_conditioning(Model(), cross)
        assert text is cross
        assert pooled is Model.conditioner.vector_in

    def test_sd15_has_no_vector(self):
        text = torch.randn(2, 77, 16)
        resolved_text, vector = _split_text_and_vector_conditioning(object(), text)
        assert resolved_text is text
        assert vector is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_image(h=128, w=192):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _fake_detector(output_shape=(128, 128, 3)):
    """Return a mock detector that returns a PIL image of the given shape."""
    mock = MagicMock()
    mock.return_value = Image.fromarray(
        np.random.randint(0, 255, output_shape, dtype=np.uint8)
    )
    return mock


def _fake_loader(detector):
    """Return a loader function that yields the given detector mock."""
    return MagicMock(return_value=detector)


# ---------------------------------------------------------------------------
# ANNOTATOR_MAP
# ---------------------------------------------------------------------------

class TestAnnotatorMap:
    def test_has_sd15_types(self):
        expected = ['canny', 'depth', 'openpose', 'softedge', 'scribble',
                    'lineart', 'lineart_anime', 'mlsd', 'seg', 'shuffle',
                    'inpaint', 'normalbae', 'tile', 'ip2p']
        values = set(ANNOTATOR_MAP.values())
        for t in expected:
            assert t in values, f"'{t}' missing from ANNOTATOR_MAP values"

    def test_has_sdxl_keys(self):
        assert 'control_sdxl_canny' in ANNOTATOR_MAP
        assert 'control_sdxl_depth' in ANNOTATOR_MAP

    def test_all_values_are_strings(self):
        for k, v in ANNOTATOR_MAP.items():
            assert isinstance(v, str)

    def test_passthrough_annotators_are_annotator_types(self):
        all_types = set(ANNOTATOR_MAP.values()) | _PASSTHROUGH_ANNOTATORS
        for t in _PASSTHROUGH_ANNOTATORS:
            assert t in all_types


# ---------------------------------------------------------------------------
# annotate_image — no-model paths
# ---------------------------------------------------------------------------

class TestAnnotateImagePassthrough:
    def test_canny_returns_3ch_uint8(self):
        img = _dummy_image()
        result = annotate_image(img, 'canny', resolution=128)
        assert result.ndim == 3
        assert result.shape[2] == 3
        assert result.dtype == np.uint8

    def test_canny_binary_values(self):
        """Canny output should be 0 or 255 only."""
        img = _dummy_image()
        result = annotate_image(img, 'canny', resolution=64)
        unique = set(result.flatten())
        assert unique <= {0, 255}

    def test_tile_returns_source_values_by_default(self):
        # tile is passthrough-by-default but now runs through _annotate_tile
        # (qr/tile mask options) — value-identical copy with default opts
        img = _dummy_image()
        result = annotate_image(img, 'tile')
        assert np.array_equal(result, img)

    def test_inpaint_requires_mask(self):
        # inpaint is no longer passthrough: it builds the notebook's
        # -255-marker conditioning and requires a mask.
        img = _dummy_image()
        with pytest.raises(ValueError, match='inpaint_mask'):
            annotate_image(img, 'inpaint')

    def test_temporalnet_returns_source_unchanged(self):
        img = _dummy_image()
        result = annotate_image(img, 'temporalnet')
        assert result is img

    def test_ip2p_returns_source_unchanged(self):
        img = _dummy_image()
        result = annotate_image(img, 'ip2p')
        assert result is img

    def test_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown annotator type"):
            annotate_image(_dummy_image(), 'nonexistent')


# ---------------------------------------------------------------------------
# annotate_image — model-backed paths (mocked)
# ---------------------------------------------------------------------------

class TestAnnotateImageModelBacked:
    """All model-backed annotators are mocked — no weights downloaded."""

    def _run(self, annotator_type, loader_path, **kwargs):
        """Patch the loader, run annotate_image, return (result, loader_mock)."""
        detector = _fake_detector()
        loader = _fake_loader(detector)
        clear_annotator_cache()
        with patch(loader_path, loader):
            result = annotate_image(_dummy_image(), annotator_type, resolution=128, **kwargs)
        return result, loader, detector

    def test_depth_dispatches_to_midas(self):
        result, loader, detector = self._run(
            'depth', 'vibewarp.core.controlnet._load_midas')
        loader.assert_called_once()
        detector.assert_called_once()
        assert result.ndim == 3

    def test_softedge_dispatches_to_pidinet(self):
        result, loader, _ = self._run(
            'softedge', 'vibewarp.core.controlnet._load_pidinet')
        loader.assert_called_once()

    def test_scribble_dispatches_to_pidinet_by_default(self):
        # notebook default control_sd15_scribble_detector = 'PIDI'
        result, loader, _ = self._run(
            'scribble', 'vibewarp.core.controlnet._load_pidinet')
        loader.assert_called_once()

    def test_scribble_hed_when_selected(self):
        result, loader, _ = self._run(
            'scribble', 'vibewarp.core.controlnet._load_hed',
            opts={'scribble_detector': 'HED'})
        loader.assert_called_once()

    def test_lineart_dispatches_to_lineart(self):
        result, loader, _ = self._run(
            'lineart', 'vibewarp.core.controlnet._load_lineart')
        loader.assert_called_once()

    def test_lineart_anime_dispatches(self):
        result, loader, _ = self._run(
            'lineart_anime', 'vibewarp.core.controlnet._load_lineart_anime')
        loader.assert_called_once()

    def test_mlsd_dispatches(self):
        result, loader, _ = self._run(
            'mlsd', 'vibewarp.core.controlnet._load_mlsd')
        loader.assert_called_once()

    def test_openpose_dispatches(self):
        result, loader, detector = self._run(
            'openpose', 'vibewarp.core.controlnet._load_openpose')
        loader.assert_called_once()
        assert detector.call_args.kwargs['include_body'] is True
        assert detector.call_args.kwargs['include_hand'] is False
        assert detector.call_args.kwargs['include_face'] is False

    def test_openpose_forwards_body_hand_and_face_settings(self):
        _, _, detector = self._run(
            'openpose', 'vibewarp.core.controlnet._load_openpose',
            opts={
                'pose_include_body': False,
                'pose_include_hand': True,
                'pose_include_face': True,
            })
        assert detector.call_args.kwargs['include_body'] is False
        assert detector.call_args.kwargs['include_hand'] is True
        assert detector.call_args.kwargs['include_face'] is True

    def test_dwpose_dispatches(self):
        expected = np.zeros((64, 64, 3), dtype=np.uint8)
        with patch('vibewarp.core.controlnet._run_dwpose',
                   return_value=expected) as run:
            result = annotate_image(
                _dummy_image(), 'dwpose', resolution=128,
                opts={
                    'pose_include_body': False,
                    'pose_include_hand': True,
                    'pose_include_face': True,
                })
        assert result is expected
        assert run.call_args.kwargs == {
            'include_body': False,
            'include_hand': True,
            'include_face': True,
        }

    def test_shuffle_dispatches(self):
        result, loader, _ = self._run(
            'shuffle', 'vibewarp.core.controlnet._load_shuffle')
        loader.assert_called_once()

    def test_normalbae_dispatches(self):
        result, loader, _ = self._run(
            'normalbae', 'vibewarp.core.controlnet._load_normalbae')
        loader.assert_called_once()

    def test_seg_sam_dispatches_only_when_selected(self):
        # SAM is a non-notebook override; default seg goes to OneFormer
        result, loader, _ = self._run(
            'seg', 'vibewarp.core.controlnet._load_sam', seg_detector='Seg_SAM')
        loader.assert_called_once()

    def test_all_results_are_numpy_uint8(self):
        """Every model-backed annotator must return HWC uint8."""
        cases = [
            ('depth',         'vibewarp.core.controlnet._load_midas'),
            ('softedge',      'vibewarp.core.controlnet._load_pidinet'),
            ('scribble',      'vibewarp.core.controlnet._load_pidinet'),
            ('lineart',       'vibewarp.core.controlnet._load_lineart'),
            ('lineart_anime', 'vibewarp.core.controlnet._load_lineart_anime'),
            ('mlsd',          'vibewarp.core.controlnet._load_mlsd'),
            ('openpose',      'vibewarp.core.controlnet._load_openpose'),
            ('shuffle',       'vibewarp.core.controlnet._load_shuffle'),
            ('normalbae',     'vibewarp.core.controlnet._load_normalbae'),
        ]
        for ann_type, loader_path in cases:
            result, _, _ = self._run(ann_type, loader_path)
            assert result.ndim == 3, f"{ann_type}: expected 3D array"
            assert result.dtype == np.uint8, f"{ann_type}: expected uint8"


# ---------------------------------------------------------------------------
# Caching — loader called only once across multiple frames
# ---------------------------------------------------------------------------

class TestAnnotatorCaching:
    def test_loader_called_once_across_frames(self):
        """Detector should be instantiated once; reused on subsequent calls."""
        detector = _fake_detector()
        loader = _fake_loader(detector)
        clear_annotator_cache()

        with patch('vibewarp.core.controlnet._load_midas', loader):
            annotate_image(_dummy_image(), 'depth', resolution=64)
            annotate_image(_dummy_image(), 'depth', resolution=64)
            annotate_image(_dummy_image(), 'depth', resolution=64)

        loader.assert_called_once()          # loaded once
        assert detector.call_count == 3      # used three times

    def test_different_types_load_separately(self):
        """Each annotator type gets its own cached detector."""
        det_depth = _fake_detector()
        det_soft = _fake_detector()
        loader_midas = _fake_loader(det_depth)
        loader_pidi = _fake_loader(det_soft)
        clear_annotator_cache()

        with patch('vibewarp.core.controlnet._load_midas', loader_midas), \
             patch('vibewarp.core.controlnet._load_pidinet', loader_pidi):
            annotate_image(_dummy_image(), 'depth', resolution=64)
            annotate_image(_dummy_image(), 'softedge', resolution=64)

        loader_midas.assert_called_once()
        loader_pidi.assert_called_once()
        assert det_depth.call_count == 1
        assert det_soft.call_count == 1

    def test_clear_cache_forces_reload(self):
        """After clear_annotator_cache(), loader is called again."""
        detector = _fake_detector()
        loader = _fake_loader(detector)
        clear_annotator_cache()

        with patch('vibewarp.core.controlnet._load_midas', loader):
            annotate_image(_dummy_image(), 'depth', resolution=64)
            clear_annotator_cache()
            annotate_image(_dummy_image(), 'depth', resolution=64)

        assert loader.call_count == 2


# ---------------------------------------------------------------------------
# map_to_conditioning
# ---------------------------------------------------------------------------

class TestMapToConditioning:
    def test_output_shape(self):
        detected = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)
        tensor = map_to_conditioning(detected, target_h=512, target_w=512)
        assert tensor.shape == (1, 3, 512, 512)

    def test_output_range(self):
        detected = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)
        tensor = map_to_conditioning(detected, target_h=64, target_w=64)
        assert tensor.min().item() >= 0.0
        assert tensor.max().item() <= 1.0

    def test_grayscale_input_broadcast_to_3ch(self):
        detected = np.random.randint(0, 255, (64, 96), dtype=np.uint8)
        tensor = map_to_conditioning(detected, target_h=64, target_w=64)
        assert tensor.shape == (1, 3, 64, 64)

    def test_single_channel_input(self):
        detected = np.random.randint(0, 255, (64, 96, 1), dtype=np.uint8)
        tensor = map_to_conditioning(detected, target_h=64, target_w=64)
        assert tensor.shape == (1, 3, 64, 64)

    def test_black_image_zeros(self):
        detected = np.zeros((64, 64, 3), dtype=np.uint8)
        tensor = map_to_conditioning(detected, target_h=64, target_w=64)
        assert tensor.sum().item() == 0.0

    def test_white_image_ones(self):
        detected = np.full((64, 64, 3), 255, dtype=np.uint8)
        tensor = map_to_conditioning(detected, target_h=64, target_w=64)
        assert tensor.min().item() == pytest.approx(1.0, abs=0.01)

    def test_returns_float_tensor(self):
        detected = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        tensor = map_to_conditioning(detected, target_h=32, target_w=32)
        assert tensor.dtype == torch.float32


# ---------------------------------------------------------------------------
# get_controlnet_conditioning
# ---------------------------------------------------------------------------

class TestGetControlnetConditioning:
    def test_empty_sources_returns_none(self):
        result = get_controlnet_conditioning(
            source_images={}, controlnet_config={},
            target_h=64, target_w=64,
        )
        assert result is None

    def test_numpy_source_canny(self):
        source = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)
        result = get_controlnet_conditioning(
            source_images={'control_sd15_canny': source},
            controlnet_config={'control_sd15_canny': {'weight': 1.0}},
            target_h=64, target_w=64,
        )
        assert result is not None
        assert result.shape == (1, 3, 64, 64)
        assert result.min().item() >= 0.0
        assert result.max().item() <= 1.0

    def test_pil_source(self):
        source = Image.fromarray(np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8))
        result = get_controlnet_conditioning(
            source_images={'control_sd15_tile': source},
            controlnet_config={'control_sd15_tile': {'weight': 1.0}},
            target_h=64, target_w=64,
        )
        assert result is not None
        assert result.shape == (1, 3, 64, 64)

    def test_file_path_source(self, tmp_path):
        img = Image.fromarray(np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8))
        path = str(tmp_path / 'source.png')
        img.save(path)
        result = get_controlnet_conditioning(
            source_images={'control_sd15_canny': path},
            controlnet_config={'control_sd15_canny': {'weight': 1.0}},
            target_h=64, target_w=64,
        )
        assert result is not None
        assert result.shape == (1, 3, 64, 64)

    def test_missing_file_skipped(self):
        result = get_controlnet_conditioning(
            source_images={'control_sd15_canny': '/nonexistent/file.png'},
            controlnet_config={'control_sd15_canny': {'weight': 1.0}},
            target_h=64, target_w=64,
        )
        assert result is None

    def test_weight_scaling(self):
        source = np.full((64, 64, 3), 128, dtype=np.uint8)
        full = get_controlnet_conditioning(
            source_images={'control_sd15_tile': source},
            controlnet_config={'control_sd15_tile': {'weight': 1.0}},
            target_h=64, target_w=64,
        )
        half = get_controlnet_conditioning(
            source_images={'control_sd15_tile': source},
            controlnet_config={'control_sd15_tile': {'weight': 0.5}},
            target_h=64, target_w=64,
        )
        assert half.mean().item() < full.mean().item()

    def test_multiple_sources_combined_and_clamped(self):
        s1 = np.full((64, 64, 3), 200, dtype=np.uint8)
        s2 = np.full((64, 64, 3), 200, dtype=np.uint8)
        result = get_controlnet_conditioning(
            source_images={
                'control_sd15_tile': s1,
                'cn_b': s2,
            },
            controlnet_config={
                'control_sd15_tile': {'weight': 1.0},
                'cn_b': {'weight': 1.0, 'annotator': 'tile'},
            },
            target_h=64, target_w=64,
        )
        assert result is not None
        assert result.max().item() <= 1.0  # clamped

    def test_annotator_override_in_config(self):
        """Config 'annotator' key overrides ANNOTATOR_MAP lookup."""
        source = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = get_controlnet_conditioning(
            source_images={'unknown_model': source},
            controlnet_config={'unknown_model': {'weight': 1.0, 'annotator': 'canny'}},
            target_h=64, target_w=64,
        )
        assert result is not None

    def test_default_weight_one(self):
        """Missing config defaults to weight=1.0."""
        source = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = get_controlnet_conditioning(
            source_images={'control_sd15_canny': source},
            controlnet_config={},
            target_h=64, target_w=64,
        )
        assert result is not None


# ---------------------------------------------------------------------------
# apply_layer_weights / normalize_controlnet_weights
# ---------------------------------------------------------------------------

class TestApplyLayerWeights:
    def test_uniform_global_weight(self):
        outputs = [torch.ones(1, 8, 4, 4) * 2.0 for _ in range(3)]
        scaled = apply_layer_weights(outputs, global_weight=0.5)
        for t in scaled:
            assert t.mean().item() == pytest.approx(1.0)

    def test_per_layer_weights(self):
        outputs = [torch.ones(1, 1, 1, 1) for _ in range(3)]
        weights = [1.0, 2.0, 3.0]
        scaled = apply_layer_weights(outputs, layer_weights=weights)
        assert scaled[0].item() == pytest.approx(1.0)
        assert scaled[1].item() == pytest.approx(2.0)
        assert scaled[2].item() == pytest.approx(3.0)

    def test_layer_weights_shorter_than_outputs(self):
        outputs = [torch.ones(1, 1, 1, 1) for _ in range(4)]
        weights = [2.0, 3.0]  # shorter
        scaled = apply_layer_weights(outputs, layer_weights=weights, global_weight=0.5)
        assert scaled[0].item() == pytest.approx(2.0)
        assert scaled[1].item() == pytest.approx(3.0)
        assert scaled[2].item() == pytest.approx(0.5)  # falls back to global
        assert scaled[3].item() == pytest.approx(0.5)

    def test_empty_outputs(self):
        assert apply_layer_weights([]) == []


class TestNormalizeControlnetWeights:
    def test_two_equal_weights_normalize_to_half(self):
        configs = {
            'cn_a': {'weight': 1.0},
            'cn_b': {'weight': 1.0},
        }
        norm = normalize_controlnet_weights(configs, num_layers=3)
        for k in ('cn_a', 'cn_b'):
            for v in norm[k]:
                assert v == pytest.approx(0.5)

    def test_empty_config(self):
        assert normalize_controlnet_weights({}) == {}

    def test_per_layer_weights_normalized(self):
        configs = {
            'cn_a': {'weight': 1.0, 'layer_weights': [1.0, 2.0, 3.0]},
            'cn_b': {'weight': 1.0, 'layer_weights': [3.0, 2.0, 1.0]},
        }
        norm = normalize_controlnet_weights(configs, num_layers=3)
        for layer_idx in range(3):
            total = norm['cn_a'][layer_idx] + norm['cn_b'][layer_idx]
            assert total == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CN source selection — tests for the per-CN and global cond_image_src logic
# ---------------------------------------------------------------------------

def _resolve_cn_source(cn_data: dict, cond_image_src: str,
                        init_image: str = None, video_frame: str = None) -> str:
    """Pure-Python extraction of the source-resolution logic from render_frame.

    Mirrors exactly what diffusion.py does so we can unit-test it without
    spinning up the full render pipeline.
    """
    source_setting = cn_data.get('source', 'global')
    if source_setting in ('global', '', None):
        source_setting = cond_image_src

    if source_setting in ('init', 'raw_frame'):
        return video_frame
    elif source_setting == 'stylized':
        if init_image:
            return init_image
        else:
            return video_frame  # first-frame fallback
    return None


class TestCnSourceSelection:
    """Unit tests for per-CN source resolution logic (mirrors render_frame)."""

    def test_source_init_returns_video_frame(self):
        result = _resolve_cn_source(
            {'source': 'init'},
            cond_image_src='init',
            init_image='/prev_render.png',
            video_frame='/video/000001.jpg',
        )
        assert result == '/video/000001.jpg'

    def test_source_stylized_returns_init_image(self):
        result = _resolve_cn_source(
            {'source': 'stylized'},
            cond_image_src='init',
            init_image='/prev_render.png',
            video_frame='/video/000001.jpg',
        )
        assert result == '/prev_render.png'

    def test_source_stylized_frame0_falls_back_to_video(self):
        # On frame 0, init_image is None (no previous render yet)
        result = _resolve_cn_source(
            {'source': 'stylized'},
            cond_image_src='init',
            init_image=None,
            video_frame='/video/000001.jpg',
        )
        assert result == '/video/000001.jpg'

    def test_source_global_with_cond_image_src_init(self):
        result = _resolve_cn_source(
            {'source': 'global'},
            cond_image_src='init',
            init_image='/prev_render.png',
            video_frame='/video/000005.jpg',
        )
        assert result == '/video/000005.jpg'

    def test_source_global_with_cond_image_src_stylized(self):
        result = _resolve_cn_source(
            {'source': 'global'},
            cond_image_src='stylized',
            init_image='/prev_render.png',
            video_frame='/video/000005.jpg',
        )
        assert result == '/prev_render.png'

    def test_source_empty_string_treated_as_global(self):
        result = _resolve_cn_source(
            {'source': ''},
            cond_image_src='stylized',
            init_image='/prev_render.png',
            video_frame='/video.jpg',
        )
        assert result == '/prev_render.png'

    def test_source_missing_treated_as_global(self):
        result = _resolve_cn_source(
            {},  # no 'source' key
            cond_image_src='init',
            init_image='/prev_render.png',
            video_frame='/video.jpg',
        )
        assert result == '/video.jpg'

    def test_example_settings_inpaint_uses_stylized(self):
        # control_sd15_inpaint has source='stylized' in the example settings
        result = _resolve_cn_source(
            {'source': 'stylized'},
            cond_image_src='init',  # global is 'init' in example settings
            init_image='/warped_000010.png',
            video_frame='/video/000011.jpg',
        )
        assert result == '/warped_000010.png'

    def test_example_settings_depth_uses_video_frame(self):
        # control_sd15_depth has source='global', cond_image_src='init'
        result = _resolve_cn_source(
            {'source': 'global'},
            cond_image_src='init',
            init_image='/warped_000010.png',
            video_frame='/video/000011.jpg',
        )
        assert result == '/video/000011.jpg'


class TestDetectResolutionPrepare:
    """Test that detect_resolution from CN config is used in annotation."""

    def test_detect_resolution_minus_one_uses_max_dim(self, tmp_path):
        # detect_resolution=-1 → resolution = max(H, W)
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        source_img_path = str(tmp_path / 'source.jpg')
        Image.fromarray(img).save(source_img_path)

        called_with = {}

        def fake_annotate(image, annotator_type, resolution=512, **kwargs):
            called_with['resolution'] = resolution
            return image

        with patch('vibewarp.core.controlnet.annotate_image', fake_annotate):
            from vibewarp.core.controlnet import prepare_cn_hints_for_frame
            cn_data = {
                'model': MagicMock(),
                'weight': 1.0, 'start': 0.0, 'end': 1.0,
                'annotator': 'tile',
                'source': 'init',
                'detect_resolution': -1,
            }
            sd_model = MagicMock()
            sd_model._cn_hints = {}

            prepare_cn_hints_for_frame(
                sd_model,
                {'control_sd15_inpaint': cn_data},
                {'control_sd15_inpaint': source_img_path},
                target_h=100, target_w=200,
            )
        # max(100, 200) = 200
        assert called_with['resolution'] == 200

    def test_detect_resolution_explicit_used(self, tmp_path):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        source_img_path = str(tmp_path / 'source.jpg')
        Image.fromarray(img).save(source_img_path)

        called_with = {}

        def fake_annotate(image, annotator_type, resolution=512, **kwargs):
            called_with['resolution'] = resolution
            return image

        with patch('vibewarp.core.controlnet.annotate_image', fake_annotate):
            from vibewarp.core.controlnet import prepare_cn_hints_for_frame
            cn_data = {
                'model': MagicMock(),
                'weight': 1.0, 'start': 0.0, 'end': 1.0,
                'annotator': 'tile',
                'source': 'init',
                'detect_resolution': 512,
            }
            sd_model = MagicMock()
            sd_model._cn_hints = {}

            prepare_cn_hints_for_frame(
                sd_model,
                {'control_sd15_inpaint': cn_data},
                {'control_sd15_inpaint': source_img_path},
                target_h=100, target_w=200,
            )
        assert called_with['resolution'] == 512
