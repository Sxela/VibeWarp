import json

import pytest

from vibewarp.config_io import config_from_dict, config_from_settings, config_schema, validate_config
from vibewarp.core.diffusion import RenderContext, _resolve_frame_range
from vibewarp.settings import settings_to_dict


@pytest.mark.parametrize('selected', [[0, 10], [5, 15], [5, 0]])
def test_explicit_range_overrides_retired_cap(selected):
    config = config_from_dict({'frame_range': selected, 'video': {'max_frames': 1}})
    assert config.frame_range == selected
    assert not hasattr(config.video, 'max_frames')
    assert 'video_max_frames' not in settings_to_dict(config)
    if selected[1]:
        assert _resolve_frame_range(RenderContext(config=config), None, 0) == (selected[0], selected[1] + 1)


@pytest.mark.parametrize(('count', 'expected'), [(1, [0]), (5, [0, 4]), (0, [0, 0])])
def test_legacy_count_becomes_a_range_once(count, expected, tmp_path):
    config = config_from_dict({'frame_range': [0, 0], 'video': {'max_frames': count}})
    assert config.frame_range == expected
    assert not validate_config(config, require_inputs=False)
    path = tmp_path / 'settings.txt'
    path.write_text(json.dumps(settings_to_dict(config)))
    assert config_from_settings(path).frame_range == expected
    config.frame_range = [0, 10]
    assert _resolve_frame_range(RenderContext(config=config), None, 0) == (0, 11)


def test_flat_snapshot_with_range_and_old_cap_imports_the_range(tmp_path):
    path = tmp_path / 'old-settings.txt'
    path.write_text(json.dumps({'frame_range': [0, 10], 'video_max_frames': 1, 'flow_flow_warp': False}))
    config = config_from_settings(path)
    assert _resolve_frame_range(RenderContext(config=config), None, 0) == (0, 11)
    assert 'max_frames' not in config_schema()['properties']['video']['properties']


def test_single_frame_has_a_finite_range_without_extracted_files():
    config = config_from_dict({'frame_range': [0]})
    assert _resolve_frame_range(RenderContext(config=config), None, 0) == (0, 1)
    assert _resolve_frame_range(RenderContext(config=config), None, 1) == (1, 1)
