"""Independent before/after color-match routing."""

import numpy as np
from PIL import Image

from vibewarp.config import RunConfig
from vibewarp.config_io import config_from_dict
from vibewarp.core import diffusion


def test_before_and_after_phases_are_independently_enabled():
    config = RunConfig()
    config.color.before_enabled = True
    config.color.before_strength = 0.4
    config.color.after_enabled = False
    config.color.after_strength = 0.8

    assert diffusion._uses_colormatch(config, 'before')
    assert not diffusion._uses_colormatch(config, 'after')

    config.color.after_enabled = True
    assert diffusion._uses_colormatch(config, 'after')


def test_zero_strength_disables_only_its_phase():
    config = RunConfig()
    config.color.before_enabled = True
    config.color.before_strength = 0.0
    config.color.after_enabled = True
    config.color.after_strength = 0.5

    assert not diffusion._uses_colormatch(config, 'before')
    assert diffusion._uses_colormatch(config, 'after')


def test_legacy_shared_mode_migrates_to_independent_phases():
    config = config_from_dict({
        'color': {
            'match_color_strength': 0.6,
            'colormatch_method': 'LAB',
            'colormatch_regrain': True,
            'colormatch_mode': 'both',
        },
    })

    assert config.color.before_enabled
    assert config.color.after_enabled
    assert config.color.before_strength == 0.6
    assert config.color.after_strength == 0.6
    assert config.color.before_method == 'LAB'
    assert config.color.after_method == 'LAB'
    assert config.color.before_regrain
    assert config.color.after_regrain


def test_phases_use_separate_strengths(monkeypatch):
    config = RunConfig()
    config.color.before_strength = 0.2
    config.color.after_strength = 0.8
    seen = []

    monkeypatch.setattr(
        'vibewarp.color.matching.match_color_var',
        lambda reference, target, **kwargs: (
            seen.append(kwargs['opacity']) or np.asarray(target)))

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    diffusion._match_color_array(image, image, config, 'before')
    diffusion._match_color_array(image, image, config, 'after')

    assert seen == [0.2, 0.8]


def test_post_colormatch_uses_warped_reference(monkeypatch, tmp_path):
    reference_path = tmp_path / 'warped.png'
    Image.new('RGB', (8, 8), 'red').save(reference_path)
    output = Image.new('RGB', (8, 8), 'blue')
    config = RunConfig()
    config.color.after_enabled = True
    config.color.after_strength = 0.5
    seen = {}

    def fake_match(reference, target, cfg, phase):
        seen['reference'] = np.asarray(reference).copy()
        seen['target'] = np.asarray(target).copy()
        seen['phase'] = phase
        return np.full((8, 8, 3), (0, 255, 0), dtype=np.uint8)

    monkeypatch.setattr(diffusion, '_match_color_array', fake_match)
    result = diffusion._apply_post_colormatch(output, str(reference_path), config)

    assert tuple(seen['reference'][0, 0]) == (255, 0, 0)
    assert tuple(seen['target'][0, 0]) == (0, 0, 255)
    assert seen['phase'] == 'after'
    assert result.getpixel((0, 0)) == (0, 255, 0)
