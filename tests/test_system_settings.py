"""System-tier settings must describe the MACHINE, never the render.

Before this, importing someone else's settings file silently repointed your checkpoint
directory, your ControlNet model dir and your VRAM tuning — the thing the System tab was
split out to prevent.
"""

import json

import pytest
from dataclasses import asdict

from vibewarp import system_settings
from vibewarp.config import RunConfig


@pytest.fixture(autouse=True)
def install_dir(tmp_path, monkeypatch):
    """Keep the tests off the developer's real system.json."""
    monkeypatch.setenv('VIBEWARP_HOME', str(tmp_path))
    return tmp_path


def test_settings_live_next_to_the_install(install_dir):
    assert system_settings.settings_path() == install_dir / 'system.json'


def test_capture_takes_only_the_system_tier():
    config = asdict(RunConfig())
    config['controlnet']['model_dir'] = 'D:/models/cn'
    config['diffusion']['seed'] = 1234
    config['text_prompts'] = {0: 'a cat'}

    captured = system_settings.capture(config)

    assert captured['controlnet']['model_dir'] == 'D:/models/cn'
    # ...and nothing that changes what the render looks like
    assert 'diffusion' not in captured or 'seed' not in captured['diffusion']
    assert 'text_prompts' not in captured.get('main', {})


def test_import_cannot_clobber_local_machine_paths(install_dir):
    """THE bug this exists for."""
    mine = asdict(RunConfig())
    mine['controlnet']['model_dir'] = 'C:/my/models'
    mine['flow']['flow_threads'] = 16
    system_settings.save(mine)

    # A settings file from someone else's machine, with their paths and their prompt.
    theirs = asdict(RunConfig())
    theirs['controlnet']['model_dir'] = '/home/them/models'
    theirs['flow']['flow_threads'] = 2
    theirs['text_prompts'] = {0: 'their prompt'}

    merged = system_settings.overlay(theirs)

    assert merged['controlnet']['model_dir'] == 'C:/my/models'   # mine wins
    assert merged['flow']['flow_threads'] == 16                  # mine wins
    assert merged['text_prompts'] == {0: 'their prompt'}         # theirs is the point


def test_export_strips_the_system_tier():
    config = asdict(RunConfig())
    config['controlnet']['model_dir'] = 'C:/secret/path'
    config['diffusion']['seed'] = 7

    exported = system_settings.strip(config)

    assert 'model_dir' not in exported['controlnet']
    assert exported['diffusion']['seed'] == 7      # creative settings survive
    assert config['controlnet']['model_dir'] == 'C:/secret/path'   # input untouched


def test_save_then_load_round_trips(install_dir):
    config = asdict(RunConfig())
    config['ipadapter']['clip_vision_model_path'] = 'C:/clip.safetensors'
    system_settings.save(config)

    assert json.loads((install_dir / 'system.json').read_text())['ipadapter'] \
        ['clip_vision_model_path'] == 'C:/clip.safetensors'
    assert system_settings.load()['ipadapter']['clip_vision_model_path'] == 'C:/clip.safetensors'


def test_missing_or_corrupt_file_is_not_fatal(install_dir):
    assert system_settings.load() == {}
    (install_dir / 'system.json').write_text('{not json')
    assert system_settings.load() == {}
    # and overlaying with nothing stored leaves the config alone
    config = asdict(RunConfig())
    assert system_settings.overlay(config) == config


def test_stale_file_cannot_smuggle_in_creative_settings(install_dir):
    """A system.json written by an older version might hold keys that are no longer
    system-tier. Only the current system tier is applied."""
    (install_dir / 'system.json').write_text(json.dumps({
        'controlnet': {'model_dir': 'C:/mine'},
        'diffusion': {'seed': 999, 'sampler': 'sample_lms'},
    }))
    config = asdict(RunConfig())
    merged = system_settings.overlay(config)

    assert merged['controlnet']['model_dir'] == 'C:/mine'
    assert merged['diffusion']['seed'] == config['diffusion']['seed']      # NOT 999
    assert merged['diffusion']['sampler'] == config['diffusion']['sampler']
