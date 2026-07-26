"""Tests for vibewarp.settings and vibewarp.utils.scheduling."""

import json
import os
import tempfile

import pytest

from vibewarp.config import RunConfig, DiffusionConfig
from vibewarp.settings import (
    generate_file_hash,
    is_vibewarp_settings,
    load_settings,
    load_warpfusion_settings,
    save_settings,
    settings_to_dict,
)


class TestSettingsFormatDetection:
    def test_notebook_diffusion_steps_is_not_vibewarp_marker(self):
        assert is_vibewarp_settings({'diffusion_steps': 1000}) is False

    def test_prefixed_vibewarp_key_is_marker(self):
        assert is_vibewarp_settings({'flow_flow_warp': True}) is True
from vibewarp.utils.scheduling import get_sched_from_json, get_scheduled_arg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_settings(tmp_path, data: dict) -> str:
    path = str(tmp_path / "settings.txt")
    with open(path, 'w') as f:
        json.dump(data, f)
    return path


def _load(tmp_path, data: dict) -> dict:
    return load_warpfusion_settings(_write_settings(tmp_path, data))


# ---------------------------------------------------------------------------
# generate_file_hash
# ---------------------------------------------------------------------------

class TestGenerateFileHash:
    def test_produces_hex_string(self, tmp_path):
        f = os.path.join(str(tmp_path), 'test.txt')
        with open(f, 'w') as fp:
            fp.write('hello')
        h = generate_file_hash(f)
        assert isinstance(h, str)
        assert len(h) == 64  # sha256 hex

    def test_deterministic(self, tmp_path):
        f = os.path.join(str(tmp_path), 'test.txt')
        with open(f, 'w') as fp:
            fp.write('hello')
        assert generate_file_hash(f) == generate_file_hash(f)


# ---------------------------------------------------------------------------
# save / load roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoadSettings:
    def test_roundtrip(self, tmp_path):
        settings = {'prompt': 'a painting', 'steps': 25, 'cfg': 7.5}
        path = save_settings(settings, str(tmp_path), 'test', 0)
        assert os.path.exists(path)
        loaded = load_settings(path)
        assert loaded['prompt'] == 'a painting'
        assert loaded['steps'] == 25

    def test_no_overwrite(self, tmp_path):
        settings = {'v': 1}
        p1 = save_settings(settings, str(tmp_path), 'test', 0)
        p2 = save_settings(settings, str(tmp_path), 'test', 0)
        assert p1 != p2
        assert os.path.exists(p1) and os.path.exists(p2)

    def test_with_subpath(self, tmp_path):
        path = save_settings({'v': 1}, str(tmp_path), 'test', 0, path='sub')
        assert 'sub' in path
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# settings_to_dict
# ---------------------------------------------------------------------------

class TestSettingsToDict:
    def test_basic(self):
        cfg = RunConfig(batch_name='myrun')
        d = settings_to_dict(cfg)
        assert 'batch_name' in d
        assert d['batch_name'] == 'myrun'

    def test_nested_steps_accessible(self):
        cfg = RunConfig(diffusion=DiffusionConfig(steps=50))
        d = settings_to_dict(cfg)
        steps_val = next((v for k, v in d.items() if 'steps' in k and v == 50), None)
        assert steps_val == 50

    def test_prompts_preserved(self):
        cfg = RunConfig(text_prompts={0: 'hello', 10: 'world'})
        d = settings_to_dict(cfg)
        assert d['text_prompts'] == {0: 'hello', 10: 'world'}


# ---------------------------------------------------------------------------
# get_sched_from_json — scheduling corner cases
# ---------------------------------------------------------------------------

class TestGetSchedFromJson:
    def test_exact_key(self):
        assert get_sched_from_json(0, {0: 10, 10: 20}) == 10
        assert get_sched_from_json(10, {0: 10, 10: 20}) == 20

    def test_before_first_key_returns_first_value(self):
        # frames before first keyframe should clamp to first key's value, not 0
        assert get_sched_from_json(0, {5: 25, 20: 30}) == 25
        assert get_sched_from_json(3, {5: 25}) == 25

    def test_after_last_key_returns_last_value(self):
        assert get_sched_from_json(100, {0: 10, 20: 30}) == 30

    def test_between_keys_no_blend(self):
        assert get_sched_from_json(5, {0: 10, 10: 20}, blend=False) == 10

    def test_between_keys_with_blend(self):
        result = get_sched_from_json(5, {0: 10, 10: 20}, blend=True)
        assert abs(result - 15.0) < 1e-6

    def test_string_keys_normalized(self):
        assert get_sched_from_json(0, {'0': 42}) == 42


class TestGetScheduledArg:
    def test_list_in_bounds(self):
        assert get_scheduled_arg(0, [10, 20, 30]) == 10
        assert get_scheduled_arg(2, [10, 20, 30]) == 30

    def test_list_out_of_bounds_clamps_last(self):
        assert get_scheduled_arg(99, [10, 20]) == 20

    def test_dict_schedule(self):
        assert get_scheduled_arg(0, {0: 5, 10: 15}) == 5

    def test_scalar_passthrough(self):
        assert get_scheduled_arg(0, 42) == 42
        assert get_scheduled_arg(5, 3.14) == 3.14


# ---------------------------------------------------------------------------
# load_warpfusion_settings — field mapping
# ---------------------------------------------------------------------------

class TestLoadInvalidFile:
    def test_nonexistent_file(self):
        with pytest.raises(Exception):
            load_settings('/nonexistent/settings.txt')


class TestPromptMapping:
    def test_text_prompts_list_joined(self, tmp_path):
        cfg = _load(tmp_path, {'text_prompts': {'0': ['hello', 'world']}})
        assert cfg['text_prompts'] == {0: 'hello, world'}

    def test_text_prompts_string_preserved(self, tmp_path):
        cfg = _load(tmp_path, {'text_prompts': {'0': 'a painting'}})
        assert cfg['text_prompts'] == {0: 'a painting'}

    def test_negative_prompts(self, tmp_path):
        cfg = _load(tmp_path, {'negative_prompts': {'0': ['ugly'], '10': ['blurry']}})
        assert cfg['negative_prompts'] == {0: 'ugly', 10: 'blurry'}

    def test_empty_prompts_produce_empty_dicts(self, tmp_path):
        cfg = _load(tmp_path, {})
        assert cfg['text_prompts'] == {}
        assert cfg['negative_prompts'] == {}


class TestDiffusionMapping:
    def test_steps_from_single_entry_schedule(self, tmp_path):
        cfg = _load(tmp_path, {'steps_schedule': {'0': 20}})
        assert cfg['diffusion']['steps'] == 20

    def test_steps_from_multi_entry_schedule_uses_first_frame(self, tmp_path):
        # multi-entry: first frame's value is used as scalar
        cfg = _load(tmp_path, {'steps_schedule': {'0': 15, '10': 30}})
        assert cfg['diffusion']['steps'] == 15

    def test_steps_first_key_not_zero(self, tmp_path):
        # first key at frame 10 → scalar should be 25
        cfg = _load(tmp_path, {'steps_schedule': {'10': 25}})
        assert cfg['diffusion']['steps'] == 25

    def test_steps_default_when_no_schedule(self, tmp_path):
        cfg = _load(tmp_path, {})
        assert cfg['diffusion']['steps'] == 25

    def test_cfg_scale_scalar_from_list(self, tmp_path):
        # [[3, 7, 3]] WarpFusion format → scalar = 3
        cfg = _load(tmp_path, {'cfg_scale_schedule': [[3, 7, 3]]})
        assert cfg['diffusion']['cfg_scale'] == 3.0

    def test_cfg_scale_scalar_plain(self, tmp_path):
        cfg = _load(tmp_path, {'cfg_scale_schedule': [7.5]})
        assert cfg['diffusion']['cfg_scale'] == 7.5

    def test_style_strength_from_schedule(self, tmp_path):
        cfg = _load(tmp_path, {'style_strength_schedule': [0.75]})
        assert cfg['diffusion']['style_strength'] == 0.75

    def test_sampler_mapped(self, tmp_path):
        cfg = _load(tmp_path, {'sampler': 'sample_dpm_2'})
        assert cfg['diffusion']['sampler'] == 'sample_dpm_2'

    def test_seed(self, tmp_path):
        cfg = _load(tmp_path, {'seed': 12345})
        assert cfg['diffusion']['seed'] == 12345

    def test_fixed_seed(self, tmp_path):
        cfg = _load(tmp_path, {'fixed_seed': True})
        assert cfg['diffusion']['fixed_seed'] is True

    def test_clip_skip(self, tmp_path):
        cfg = _load(tmp_path, {'clip_skip': 2})
        assert cfg['diffusion']['clip_skip'] == 2

    def test_sd_batch_size(self, tmp_path):
        cfg = _load(tmp_path, {'sd_batch_size': 4})
        assert cfg['diffusion']['sd_batch_size'] == 4

    def test_use_karras_noise(self, tmp_path):
        cfg = _load(tmp_path, {'use_karras_noise': False})
        assert cfg['diffusion']['use_karras_noise'] is False

    def test_dynamic_thresh(self, tmp_path):
        cfg = _load(tmp_path, {'dynamic_thresh': 15.0})
        assert cfg['diffusion']['dynamic_thresh'] == 15.0

    def test_dynamic_thresh_default(self, tmp_path):
        cfg = _load(tmp_path, {})
        assert cfg['diffusion']['dynamic_thresh'] == 30.0

    def test_code_randomness(self, tmp_path):
        cfg = _load(tmp_path, {'code_randomness': 0.3})
        assert cfg['diffusion']['code_randomness'] == 0.3

    def test_code_randomness_default(self, tmp_path):
        cfg = _load(tmp_path, {})
        assert cfg['diffusion']['code_randomness'] == 0.0

    def test_schedules_passed_through(self, tmp_path):
        cfg = _load(tmp_path, {
            'steps_schedule': {'0': 20, '30': 25},
            'style_strength_schedule': [0.6, 0.7],
            'flow_blend_schedule': [1.0],
        })
        assert cfg['diffusion']['steps_schedule'] == {0: 20, 30: 25}
        assert cfg['diffusion']['style_strength_schedule'] == [0.6, 0.7]
        assert cfg['warp']['flow_blend_schedule'] == [1.0]


class TestVideoMapping:
    def test_video_path(self, tmp_path):
        cfg = _load(tmp_path, {'video_init_path': '/path/to/video.mp4'})
        assert cfg['video']['video_init_path'] == '/path/to/video.mp4'

    def test_resolution(self, tmp_path):
        cfg = _load(tmp_path, {'width': 1024, 'height': 768})
        assert cfg['video']['width'] == 1024
        assert cfg['video']['height'] == 768

    def test_extract_nth_frame(self, tmp_path):
        cfg = _load(tmp_path, {'extract_nth_frame': 3})
        assert cfg['video']['extract_nth_frame'] == 3


class TestFlowMapping:
    def test_flow_warp(self, tmp_path):
        cfg = _load(tmp_path, {'flow_warp': False})
        assert cfg['flow']['flow_warp'] is False

    def test_check_consistency(self, tmp_path):
        cfg = _load(tmp_path, {'check_consistency': False})
        assert cfg['flow']['check_consistency'] is False

    def test_num_flow_updates_default(self, tmp_path):
        # notebook default is 20
        cfg = _load(tmp_path, {})
        assert cfg['flow']['num_flow_updates'] == 20

    def test_num_flow_updates_from_settings(self, tmp_path):
        cfg = _load(tmp_path, {'num_flow_updates': 12})
        assert cfg['flow']['num_flow_updates'] == 12

    def test_consistency_weights(self, tmp_path):
        cfg = _load(tmp_path, {
            'missed_consistency_weight': 0.5,
            'overshoot_consistency_weight': 0.8,
            'edges_consistency_weight': 0.2,
        })
        assert cfg['flow']['missed_consistency_weight'] == 0.5
        assert cfg['flow']['overshoot_consistency_weight'] == 0.8
        assert cfg['flow']['edges_consistency_weight'] == 0.2
        assert cfg['video_assembly']['missed_consistency_weight'] == 0.5
        assert cfg['video_assembly']['overshoot_consistency_weight'] == 0.8
        assert cfg['video_assembly']['edges_consistency_weight'] == 0.2

    def test_consistency_blur_dilate(self, tmp_path):
        cfg = _load(tmp_path, {'consistency_blur': 3, 'consistency_dilate': 5})
        assert cfg['flow']['consistency_blur'] == 3
        assert cfg['flow']['consistency_dilate'] == 5


class TestWarpMapping:
    def test_warp_strength(self, tmp_path):
        cfg = _load(tmp_path, {'warp_strength': 0.8})
        assert cfg['warp']['warp_strength'] == 0.8

    def test_warp_num_k(self, tmp_path):
        cfg = _load(tmp_path, {'warp_num_k': 128})
        assert cfg['warp']['warp_num_k'] == 128

    def test_warp_forward(self, tmp_path):
        cfg = _load(tmp_path, {'warp_forward': False})
        assert cfg['warp']['warp_forward'] is False

    def test_padding(self, tmp_path):
        cfg = _load(tmp_path, {'padding_ratio': 0.3, 'padding_mode': 'zeros'})
        assert cfg['warp']['padding_ratio'] == 0.3
        assert cfg['warp']['padding_mode'] == 'zeros'

    def test_flow_blend_schedule(self, tmp_path):
        cfg = _load(tmp_path, {'flow_blend_schedule': [0.8, 0.9]})
        assert cfg['warp']['flow_blend_schedule'] == [0.8, 0.9]


class TestColorMapping:
    def test_match_color_strength(self, tmp_path):
        cfg = _load(tmp_path, {'match_color_strength': 0.5})
        assert cfg['color']['match_color_strength'] == 0.5
        assert cfg['color']['after_enabled'] is True
        assert cfg['color']['after_strength'] == 0.5
        assert cfg['color']['before_enabled'] is False

    def test_colormatch_method(self, tmp_path):
        cfg = _load(tmp_path, {'colormatch_method': 'LAB'})
        assert cfg['color']['colormatch_method'] == 'LAB'

    def test_colormatch_regrain_none_becomes_false(self, tmp_path):
        cfg = _load(tmp_path, {'colormatch_regrain': None})
        assert cfg['color']['colormatch_regrain'] is False

    def test_colormatch_after(self, tmp_path):
        cfg = _load(tmp_path, {
            'colormatch_after': False,
            'match_color_strength': 0.5,
        })
        assert cfg['color']['colormatch_after'] is False
        assert cfg['color']['colormatch_mode'] == 'before'
        assert cfg['color']['before_enabled'] is True
        assert cfg['color']['after_enabled'] is False

    def test_legacy_colormatch_after_true_maps_to_after_mode(self, tmp_path):
        cfg = _load(tmp_path, {
            'colormatch_after': True,
            'match_color_strength': 0.5,
        })
        assert cfg['color']['colormatch_mode'] == 'after'
        assert cfg['color']['before_enabled'] is False
        assert cfg['color']['after_enabled'] is True

    def test_explicit_colormatch_mode_supports_both(self, tmp_path):
        cfg = _load(tmp_path, {
            'colormatch_after': False,
            'colormatch_mode': 'both',
            'match_color_strength': 0.5,
        })
        assert cfg['color']['colormatch_mode'] == 'both'
        assert cfg['color']['before_enabled'] is True
        assert cfg['color']['after_enabled'] is True

    def test_colormatch_turbo(self, tmp_path):
        cfg = _load(tmp_path, {'colormatch_turbo': True})
        assert cfg['color']['colormatch_turbo'] is True


class TestBrightnessMapping:
    def test_enable_adjust(self, tmp_path):
        cfg = _load(tmp_path, {'enable_adjust_brightness': True})
        assert cfg['brightness']['enable'] is True

    def test_thresholds(self, tmp_path):
        cfg = _load(tmp_path, {
            'high_brightness_threshold': 200,
            'low_brightness_threshold': 30,
            'max_brightness_threshold': 250,
            'min_brightness_threshold': 2,
        })
        assert cfg['brightness']['high_brightness_threshold'] == 200
        assert cfg['brightness']['low_brightness_threshold'] == 30
        assert cfg['brightness']['max_brightness_threshold'] == 250
        assert cfg['brightness']['min_brightness_threshold'] == 2

    def test_adjust_ratios(self, tmp_path):
        cfg = _load(tmp_path, {
            'high_brightness_adjust_ratio': 0.95,
            'low_brightness_adjust_ratio': 1.05,
        })
        assert cfg['brightness']['high_brightness_adjust_ratio'] == 0.95
        assert cfg['brightness']['low_brightness_adjust_ratio'] == 1.05


class TestControlNetMapping:
    def _make_cn_settings(self, cn_dict: dict) -> dict:
        return {'controlnet_multimodel': json.dumps(cn_dict)}

    def test_cn_keys_present(self, tmp_path):
        cfg = _load(tmp_path, self._make_cn_settings({
            'control_sd15_depth': {'weight': 0.8, 'start': 0.0, 'end': 1.0,
                                   'preprocess': 'depth', 'source': 'global'}
        }))
        assert 'control_sd15_depth' in cfg['controlnet']['models']

    def test_cn_weight_start_end(self, tmp_path):
        cfg = _load(tmp_path, self._make_cn_settings({
            'control_sd15_canny': {'weight': 0.5, 'start': 0.1, 'end': 0.9,
                                   'preprocess': 'global', 'source': 'init'}
        }))
        m = cfg['controlnet']['models']['control_sd15_canny']
        assert m['weight'] == 0.5
        assert m['start'] == 0.1
        assert m['end'] == 0.9

    def test_cn_source_mapped(self, tmp_path):
        cfg = _load(tmp_path, self._make_cn_settings({
            'control_sd15_softedge': {'weight': 1.0, 'start': 0.0, 'end': 1.0,
                                      'preprocess': 'global', 'source': 'stylized'}
        }))
        assert cfg['controlnet']['models']['control_sd15_softedge']['source'] == 'stylized'

    def test_cn_preprocess_global_maps_to_empty_annotator(self, tmp_path):
        # 'global' preprocess means auto-detect from model key → annotator = ''
        cfg = _load(tmp_path, self._make_cn_settings({
            'control_sd15_depth': {'weight': 1.0, 'start': 0.0, 'end': 1.0,
                                   'preprocess': 'global', 'source': 'global'}
        }))
        assert cfg['controlnet']['models']['control_sd15_depth']['annotator'] == ''

    def test_cn_preprocess_explicit_maps_to_annotator(self, tmp_path):
        cfg = _load(tmp_path, self._make_cn_settings({
            'control_sd15_depth': {'weight': 1.0, 'start': 0.0, 'end': 1.0,
                                   'preprocess': 'depth_anything', 'source': 'global'}
        }))
        assert cfg['controlnet']['models']['control_sd15_depth']['annotator'] == 'depth'

    def test_cn_preprocess_pidi_maps_to_softedge(self, tmp_path):
        cfg = _load(tmp_path, self._make_cn_settings({
            'control_sd15_softedge': {'weight': 1.0, 'start': 0.0, 'end': 1.0,
                                      'preprocess': 'PIDI', 'source': 'global'}
        }))
        assert cfg['controlnet']['models']['control_sd15_softedge']['annotator'] == 'softedge'

    def test_cn_preprocess_dw_pose_maps_to_dwpose(self, tmp_path):
        cfg = _load(tmp_path, self._make_cn_settings({
            'control_sd15_openpose': {'weight': 1.0, 'start': 0.0, 'end': 1.0,
                                      'preprocess': 'dw_pose', 'source': 'global'}
        }))
        assert cfg['controlnet']['models']['control_sd15_openpose']['annotator'] == 'dwpose'

    def test_cn_detect_resolution(self, tmp_path):
        cfg = _load(tmp_path, self._make_cn_settings({
            'control_sd15_depth': {'weight': 1.0, 'start': 0.0, 'end': 1.0,
                                   'preprocess': 'global', 'source': 'global',
                                   'detect_resolution': 512}
        }))
        assert cfg['controlnet']['models']['control_sd15_depth']['detect_resolution'] == 512

    def test_cn_detect_resolution_default_minus_one(self, tmp_path):
        cfg = _load(tmp_path, self._make_cn_settings({
            'control_sd15_depth': {'weight': 1.0, 'start': 0.0, 'end': 1.0,
                                   'preprocess': 'global', 'source': 'global'}
        }))
        assert cfg['controlnet']['models']['control_sd15_depth']['detect_resolution'] == -1

    def test_cn_detect_resolution_global_fallback(self, tmp_path):
        # Notebook: per-model -1/'global' falls back to the GLOBAL
        # detect_resolution setting (not the render size).
        settings = self._make_cn_settings({
            'control_sd15_softedge': {'weight': 1.0, 'start': 0.0, 'end': 1.0,
                                      'preprocess': 'global', 'source': 'global',
                                      'detect_resolution': -1},
            'control_sd15_depth': {'weight': 1.0, 'start': 0.0, 'end': 1.0,
                                   'preprocess': 'global', 'source': 'global',
                                   'detect_resolution': 640},
        })
        settings['detect_resolution'] = 1920
        cfg = _load(tmp_path, settings)
        models = cfg['controlnet']['models']
        assert models['control_sd15_softedge']['detect_resolution'] == 1920
        assert models['control_sd15_depth']['detect_resolution'] == 640

    def test_cn_enabled_when_models_present(self, tmp_path):
        cfg = _load(tmp_path, self._make_cn_settings({
            'control_sd15_tile': {'weight': 0.5, 'start': 0.0, 'end': 1.0,
                                  'preprocess': 'global', 'source': 'global'}
        }))
        assert cfg['controlnet']['enabled'] is True

    def test_cn_enabled_false_when_empty(self, tmp_path):
        cfg = _load(tmp_path, {})
        assert cfg['controlnet']['enabled'] is False

    def test_cn_cond_image_src(self, tmp_path):
        cfg = _load(tmp_path, {'cond_image_src': 'stylized'})
        assert cfg['controlnet']['cond_image_src'] == 'stylized'

    def test_cn_normalize_weights(self, tmp_path):
        cfg = _load(tmp_path, {'normalize_cn_weights': True})
        assert cfg['controlnet']['normalize_weights'] is True

    def test_cn_multimodel_mode(self, tmp_path):
        cfg = _load(tmp_path, {'controlnet_multimodel_mode': 'external'})
        assert cfg['controlnet']['mode'] == 'external'

    def test_cn_file_name_resolved(self, tmp_path):
        cfg = _load(tmp_path, self._make_cn_settings({
            'control_sd15_inpaint': {'weight': 0.2, 'start': 0.0, 'end': 0.9,
                                     'preprocess': 'global', 'source': 'stylized'}
        }))
        m = cfg['controlnet']['models']['control_sd15_inpaint']
        # Without models_root, just the filename
        assert 'inpaint' in m['path']


class TestAnimateDiffMapping:
    def test_enabled_for_animatediff_version(self, tmp_path):
        cfg = _load(tmp_path, {'model_version': 'control_multi_animatediff'})
        assert cfg['animatediff']['enabled'] is True

    def test_disabled_for_sd15_version(self, tmp_path):
        cfg = _load(tmp_path, {'model_version': 'control_multi'})
        assert cfg['animatediff']['enabled'] is False

    def test_batch_length(self, tmp_path):
        cfg = _load(tmp_path, {'batch_length': 16})
        assert cfg['animatediff']['batch_length'] == 16

    def test_batch_overlap(self, tmp_path):
        cfg = _load(tmp_path, {'batch_overlap': 4})
        assert cfg['animatediff']['batch_overlap'] == 4

    def test_context_length_overlap(self, tmp_path):
        cfg = _load(tmp_path, {'context_length': 24, 'context_overlap': 6})
        assert cfg['animatediff']['context_length'] == 24
        assert cfg['animatediff']['context_overlap'] == 6

    def test_stop_early(self, tmp_path):
        cfg = _load(tmp_path, {'stop_early': 3})
        assert cfg['animatediff']['stop_early'] == 3

    def test_blend_batch_outputs(self, tmp_path):
        cfg = _load(tmp_path, {'blend_batch_outputs': False})
        assert cfg['animatediff']['blend_batch_outputs'] is False

    def test_blend_batch_outputs_default(self, tmp_path):
        cfg = _load(tmp_path, {})
        assert cfg['animatediff']['blend_batch_outputs'] is True

    def test_overlap_stylized(self, tmp_path):
        cfg = _load(tmp_path, {'overlap_stylized': False})
        assert cfg['animatediff']['overlap_stylized'] is False

    def test_overlap_stylized_default(self, tmp_path):
        cfg = _load(tmp_path, {})
        assert cfg['animatediff']['overlap_stylized'] is True


class TestNoiseModeMapping:
    def test_noise_mode_random_disables_reconstruction(self, tmp_path):
        cfg = _load(tmp_path, {'noise_mode': 'random', 'use_predicted_noise': True})
        assert cfg['reconstruction_noise']['enabled'] is False

    def test_noise_mode_reconstructed_enables(self, tmp_path):
        cfg = _load(tmp_path, {'noise_mode': 'reconstructed', 'use_predicted_noise': False})
        assert cfg['reconstruction_noise']['enabled'] is True

    def test_no_noise_mode_falls_back_to_use_predicted_noise(self, tmp_path):
        cfg = _load(tmp_path, {'use_predicted_noise': True})
        assert cfg['reconstruction_noise']['enabled'] is True

    def test_no_noise_mode_no_predicted_noise_defaults_false(self, tmp_path):
        cfg = _load(tmp_path, {})
        assert cfg['reconstruction_noise']['enabled'] is False

    def test_rec_cfg(self, tmp_path):
        cfg = _load(tmp_path, {'rec_cfg': 2.0})
        assert cfg['reconstruction_noise']['cfg_scale'] == 2.0

    def test_rec_noise_scale(self, tmp_path):
        cfg = _load(tmp_path, {'rec_noise_scale': 0.8})
        assert cfg['reconstruction_noise']['noise_scale'] == 0.8

    def test_rec_randomness(self, tmp_path):
        cfg = _load(tmp_path, {'rec_randomness': 0.3})
        assert cfg['reconstruction_noise']['randomness'] == 0.3

    def test_rec_steps_pct(self, tmp_path):
        cfg = _load(tmp_path, {'rec_steps_pct': 0.7})
        assert cfg['reconstruction_noise']['steps_pct'] == 0.7

    def test_rec_source(self, tmp_path):
        cfg = _load(tmp_path, {'rec_source': 'stylized'})
        assert cfg['reconstruction_noise']['source'] == 'stylized'

    def test_rec_separate_uc(self, tmp_path):
        cfg = _load(tmp_path, {'rec_separate_uc': False})
        assert cfg['reconstruction_noise']['separate_uncond'] is False

    def test_disable_rec_noise_controlnets(self, tmp_path):
        cfg = _load(tmp_path, {'disable_rec_noise_controlnets': True})
        assert cfg['reconstruction_noise']['disable_controlnets'] is True

    def test_batched_adiff_rec_noise(self, tmp_path):
        cfg = _load(tmp_path, {'batched_adiff_rec_noise': True})
        assert cfg['reconstruction_noise']['batched'] is True


class TestRecPromptMapping:
    def test_rec_prompts_parsed(self, tmp_path):
        cfg = _load(tmp_path, {'rec_prompts': {'0': ['a landscape'], '30': ['a portrait']}})
        assert cfg['reconstruction_noise']['prompts'] == {0: 'a landscape', 30: 'a portrait'}

    def test_rec_neg_prompts_parsed(self, tmp_path):
        cfg = _load(tmp_path, {'rec_neg_prompts': {'0': ['blurry, ugly']}})
        assert cfg['reconstruction_noise']['neg_prompts'] == {0: 'blurry, ugly'}

    def test_rec_prompts_none_when_missing(self, tmp_path):
        cfg = _load(tmp_path, {})
        assert cfg['reconstruction_noise']['prompts'] is None
        assert cfg['reconstruction_noise']['neg_prompts'] is None

    def test_rec_neg_empty_string(self, tmp_path):
        cfg = _load(tmp_path, {'rec_neg_prompts': {'0': ['']}})
        assert cfg['reconstruction_noise']['neg_prompts'] == {0: ''}


class TestModelPathMapping:
    def test_model_version_control_multi_maps_to_v15(self, tmp_path):
        cfg = _load(tmp_path, {'model_version': 'control_multi'})
        assert cfg['model_version'] == 'control_multi_v15'

    def test_model_version_control_multi_animatediff(self, tmp_path):
        cfg = _load(tmp_path, {'model_version': 'control_multi_animatediff'})
        assert cfg['model_version'] == 'control_multi_v15'

    def test_model_path_passed_through(self, tmp_path):
        cfg = _load(tmp_path, {'model_path': '/models/ckpt.safetensors'})
        assert cfg['sd_checkpoint_path'] == '/models/ckpt.safetensors'

    def test_batch_name(self, tmp_path):
        cfg = _load(tmp_path, {'batch_name': 'mytest'})
        assert cfg['batch_name'] == 'mytest'

    def test_frame_range(self, tmp_path):
        cfg = _load(tmp_path, {'frame_range': [5, 50]})
        assert cfg['frame_range'] == [5, 50]


class TestModelsRootResolution:
    def test_cn_path_resolved_with_models_root(self, tmp_path):
        # Create a fake ControlNet file
        cn_dir = tmp_path / 'ControlNet'
        cn_dir.mkdir()
        cn_file = cn_dir / 'control_v11p_sd15_inpaint.pth'
        cn_file.write_text('fake')

        # Create a fake checkpoint to anchor models_root inference
        ckpt_dir = tmp_path / 'checkpoints'
        ckpt_dir.mkdir()
        ckpt_file = ckpt_dir / 'model.safetensors'
        ckpt_file.write_text('fake')

        settings = {
            'model_path': str(ckpt_file),
            'controlnet_multimodel': json.dumps({
                'control_sd15_inpaint': {'weight': 0.2, 'start': 0.0, 'end': 0.9,
                                         'preprocess': 'global', 'source': 'stylized'}
            })
        }
        cfg = load_warpfusion_settings(_write_settings(tmp_path, settings))
        m = cfg['controlnet']['models']['control_sd15_inpaint']
        assert m['path'] == str(cn_file)

    def test_models_root_inferred_from_model_path(self, tmp_path):
        # When models_root not passed, infer from model_path parent's parent
        ckpt_dir = tmp_path / 'checkpoints'
        ckpt_dir.mkdir()
        ckpt_file = ckpt_dir / 'model.safetensors'
        ckpt_file.write_text('fake')

        cn_dir = tmp_path / 'ControlNet'
        cn_dir.mkdir()
        cn_file = cn_dir / 'control_v11p_sd15_inpaint.pth'
        cn_file.write_text('fake')

        settings = {
            'model_path': str(ckpt_file),
            'controlnet_multimodel': json.dumps({
                'control_sd15_inpaint': {'weight': 0.2, 'start': 0.0, 'end': 0.9,
                                         'preprocess': 'global', 'source': 'stylized'}
            })
        }
        # No models_root passed — should be inferred
        cfg = load_warpfusion_settings(_write_settings(tmp_path, settings))
        assert cfg['controlnet']['models']['control_sd15_inpaint']['path'] == str(cn_file)

    def test_resolves_from_cwd_models_when_abs_path_absent(self, tmp_path, monkeypatch):
        # Fresh-machine flow: the settings file carries an absolute checkpoint
        # path that does NOT exist, but models were downloaded into ./models
        # under the working directory. Both checkpoint and CN must resolve
        # there with no --models-root passed and no hardcoded C:\models.
        work = tmp_path / 'work'
        (work / 'models' / 'checkpoints').mkdir(parents=True)
        (work / 'models' / 'ControlNet').mkdir(parents=True)
        (work / 'models' / 'checkpoints' / 'model.safetensors').write_text('fake')
        (work / 'models' / 'ControlNet' / 'control_v11p_sd15_inpaint.pth').write_text('fake')

        settings = {
            'model_path': r'C:\models\checkpoints\model.safetensors',  # absent
            'controlnet_multimodel': json.dumps({
                'control_sd15_inpaint': {'weight': 0.2, 'start': 0.0, 'end': 0.9,
                                         'preprocess': 'global', 'source': 'stylized'}
            })
        }
        settings_file = _write_settings(tmp_path, settings)
        monkeypatch.chdir(work)
        cfg = load_warpfusion_settings(settings_file)  # no models_root
        assert cfg['sd_checkpoint_path'] == str(work / 'models' / 'checkpoints' / 'model.safetensors')
        assert cfg['controlnet']['models']['control_sd15_inpaint']['path'] == \
            str(work / 'models' / 'ControlNet' / 'control_v11p_sd15_inpaint.pth')


class TestExampleSettingsFile:
    """Smoke-test that the real example settings file parses without errors and
    key values land in the right places."""

    SETTINGS_PATH = os.path.join(
        os.path.dirname(__file__), '..', 'refs', 'examples',
        'stable_warpfusion_0.34.0_light(3)_settings.txt'
    )

    @pytest.fixture(autouse=True)
    def skip_if_missing(self):
        if not os.path.exists(self.SETTINGS_PATH):
            pytest.skip("Example settings file not present")

    def test_parses_without_error(self):
        cfg = load_warpfusion_settings(self.SETTINGS_PATH)
        assert isinstance(cfg, dict)

    def test_steps_is_20(self):
        cfg = load_warpfusion_settings(self.SETTINGS_PATH)
        assert cfg['diffusion']['steps'] == 20

    def test_cfg_scale_is_3(self):
        cfg = load_warpfusion_settings(self.SETTINGS_PATH)
        assert cfg['diffusion']['cfg_scale'] == 3.0

    def test_sampler(self):
        cfg = load_warpfusion_settings(self.SETTINGS_PATH)
        assert cfg['diffusion']['sampler'] == 'sample_euler'

    def test_clip_skip(self):
        cfg = load_warpfusion_settings(self.SETTINGS_PATH)
        assert cfg['diffusion']['clip_skip'] == 2

    def test_sd_batch_size(self):
        cfg = load_warpfusion_settings(self.SETTINGS_PATH)
        assert cfg['diffusion']['sd_batch_size'] == 2

    def test_use_karras_noise_false(self):
        cfg = load_warpfusion_settings(self.SETTINGS_PATH)
        assert cfg['diffusion']['use_karras_noise'] is False

    def test_dynamic_thresh_30(self):
        cfg = load_warpfusion_settings(self.SETTINGS_PATH)
        assert cfg['diffusion']['dynamic_thresh'] == 30.0

    def test_code_randomness_zero(self):
        cfg = load_warpfusion_settings(self.SETTINGS_PATH)
        assert cfg['diffusion']['code_randomness'] == 0.0

    def test_reconstruction_noise_enabled(self):
        cfg = load_warpfusion_settings(self.SETTINGS_PATH)
        assert cfg['reconstruction_noise']['enabled'] is True

    def test_rec_prompts_present(self):
        cfg = load_warpfusion_settings(self.SETTINGS_PATH)
        assert cfg['reconstruction_noise']['prompts'] is not None
        assert 0 in cfg['reconstruction_noise']['prompts']

    def test_five_controlnets(self):
        cfg = load_warpfusion_settings(self.SETTINGS_PATH)
        models = cfg['controlnet']['models']
        assert set(models.keys()) == {
            'control_sd15_inpaint', 'control_sd15_depth',
            'control_sd15_softedge', 'control_sd15_ip2p', 'control_sd15_tile'
        }

    def test_inpaint_source_is_stylized(self):
        cfg = load_warpfusion_settings(self.SETTINGS_PATH)
        # settings: source='stylized' for inpaint
        assert cfg['controlnet']['models']['control_sd15_inpaint']['source'] == 'stylized'

    def test_cn_annotators_are_empty_for_global_preprocess(self):
        cfg = load_warpfusion_settings(self.SETTINGS_PATH)
        for key, m in cfg['controlnet']['models'].items():
            # All use preprocess='global' → annotator should be ''
            assert m['annotator'] == '', f"{key} annotator should be '' (auto-detect)"

    def test_flow_blend_schedule(self):
        cfg = load_warpfusion_settings(self.SETTINGS_PATH)
        assert cfg['warp']['flow_blend_schedule'] == [1]

    def test_warp_num_k(self):
        cfg = load_warpfusion_settings(self.SETTINGS_PATH)
        assert cfg['warp']['warp_num_k'] == 128

    def test_num_flow_updates(self):
        cfg = load_warpfusion_settings(self.SETTINGS_PATH)
        assert cfg['flow']['num_flow_updates'] == 20


class TestCrossPlatformPaths:
    r"""Settings files carry Windows paths even when VibeWarp runs on Linux.

    REGRESSION: the models-root fallback used os.path.basename(), and on POSIX a backslash
    is not a separator — so basename(r'C:\models\checkpoints\model.safetensors') returned
    the WHOLE string and the fallback hunted for a file with that literal name. It never
    resolved. Invisible on Windows; CI caught it the first time the suite ran on Linux.

    These assertions hold on BOTH host OSes, which is the point: a bug that only appears on
    the platform you don't develop on needs a test that doesn't care which platform it is.
    """

    WINDOWS = r'C:\models\checkpoints\model.safetensors'
    POSIX = '/home/me/models/checkpoints/model.safetensors'

    def test_basename_understands_either_separator(self):
        from vibewarp.settings import path_basename
        assert path_basename(self.WINDOWS) == 'model.safetensors'
        assert path_basename(self.POSIX) == 'model.safetensors'
        assert path_basename('model.safetensors') == 'model.safetensors'
        assert path_basename('') == ''

    def test_absoluteness_understands_either_style(self):
        from vibewarp.settings import path_is_absolute
        assert path_is_absolute(self.WINDOWS) is True
        assert path_is_absolute(self.POSIX) is True
        assert path_is_absolute('models/model.safetensors') is False
        assert path_is_absolute('') is False

    def test_parent_strips_in_the_paths_own_style(self):
        from vibewarp.settings import path_parent
        assert path_parent(self.WINDOWS, 2) == r'C:\models'
        assert path_parent(self.POSIX, 2) == '/home/me/models'

    def test_forward_slash_windows_path(self):
        # WarpFusion settings sometimes carry C:/models/... too.
        from vibewarp.settings import path_basename, path_is_absolute
        mixed = 'C:/models/checkpoints/model.safetensors'
        assert path_basename(mixed) == 'model.safetensors'
        assert path_is_absolute(mixed) is True
