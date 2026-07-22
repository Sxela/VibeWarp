"""Before/after/both color-match routing."""

import numpy as np
import pytest
from PIL import Image

from vibewarp.config import RunConfig
from vibewarp.core import diffusion


@pytest.mark.parametrize(
    ('mode', 'before', 'after'),
    [
        ('before', True, False),
        ('after', False, True),
        ('both', True, True),
    ],
)
def test_colormatch_mode_routes_phases(mode, before, after):
    config = RunConfig()
    config.color.match_color_strength = 0.5
    config.color.colormatch_mode = mode
    assert diffusion._uses_colormatch(config, 'before') is before
    assert diffusion._uses_colormatch(config, 'after') is after


def test_zero_strength_disables_both_phases():
    config = RunConfig()
    config.color.match_color_strength = 0.0
    config.color.colormatch_mode = 'both'
    assert not diffusion._uses_colormatch(config, 'before')
    assert not diffusion._uses_colormatch(config, 'after')


def test_post_colormatch_uses_warped_reference(monkeypatch, tmp_path):
    reference_path = tmp_path / 'warped.png'
    Image.new('RGB', (8, 8), 'red').save(reference_path)
    output = Image.new('RGB', (8, 8), 'blue')
    config = RunConfig()
    config.color.match_color_strength = 0.5
    config.color.colormatch_mode = 'after'
    seen = {}

    def fake_match(reference, target, cfg):
        seen['reference'] = np.asarray(reference).copy()
        seen['target'] = np.asarray(target).copy()
        return np.full((8, 8, 3), (0, 255, 0), dtype=np.uint8)

    monkeypatch.setattr(diffusion, '_match_color_array', fake_match)
    result = diffusion._apply_post_colormatch(output, str(reference_path), config)

    assert tuple(seen['reference'][0, 0]) == (255, 0, 0)
    assert tuple(seen['target'][0, 0]) == (0, 0, 255)
    assert result.getpixel((0, 0)) == (0, 255, 0)
