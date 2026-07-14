"""The UI tab layout is declared in the backend, so it must stay in sync with the config.

The nav is built from `tier`/`group` in the schema. A config field with no classification
renders in NO tab — it silently disappears from the UI while still affecting the render.
These tests make that a build failure instead.
"""

import pytest

from vibewarp.config import RunConfig
from vibewarp.config_io import config_schema
from vibewarp.ui_layout import (LAYOUT, TIERS, all_config_keys, classify,
                                system_keys, unclassified)


def test_every_config_field_has_a_tab():
    missing = unclassified(RunConfig)
    assert not missing, (
        'These config fields are in no UI tab, so they would silently vanish from the '
        'form. Classify them in vibewarp/ui_layout.py:\n  ' + '\n  '.join(missing))


def test_layout_has_no_keys_the_config_dropped():
    known = set(all_config_keys(RunConfig))
    stale = sorted(key for key in LAYOUT if key not in known)
    assert not stale, (
        'ui_layout.py classifies fields that no longer exist on RunConfig (renamed or '
        'deleted?):\n  ' + '\n  '.join(stale))


def test_tiers_are_known():
    allowed = set(TIERS) | {'hidden'}
    bad = {key: tier for key, (tier, _) in LAYOUT.items() if tier not in allowed}
    assert not bad, f'unknown tier(s): {bad}'


def test_schema_carries_tier_and_group():
    schema = config_schema()
    # top-level field
    assert schema['properties']['batch_name']['tier'] == 'project'
    # nested section field
    diffusion = schema['properties']['diffusion']['properties']
    assert diffusion['steps_schedule']['tier'] == 'render'
    assert diffusion['steps_schedule']['group'] == 'Diffusion'
    # the scalar superseded by that schedule is hidden, not deleted
    assert diffusion['steps']['tier'] == 'hidden'
    assert [t['id'] for t in schema['tiers']] == list(TIERS)


@pytest.mark.parametrize('section,field,tier', [
    ('video', 'video_init_path', 'project'),
    ('main', 'text_prompts', 'render'),        # prompts are the thing you tweak every run
    ('main', 'model_version', 'project'),       # ...the model is picked once and left alone
    ('main', 'animation_mode', 'hidden'),       # only one value the engine supports
    ('brightness', 'enable', 'hidden'),         # legacy knob
    ('controlnet', 'models', 'render'),
    ('controlnet', 'model_dir', 'system'),      # a PATH -> system, not render
    ('flow', 'flow_threads', 'system'),
    ('animatediff', 'motion_module_path', 'system'),
    ('freeu', 'b1', 'advanced'),
])
def test_representative_fields_land_where_expected(section, field, tier):
    placed = classify(section, field)
    assert placed is not None and placed[0] == tier


def test_per_net_controlnet_settings_stay_on_their_card():
    """Detectors/thresholds must be reachable from the net they configure.

    ControlNetEditor draws the whole 'ControlNet' group itself, from the catalog's
    per-net `detectors` tuple — the generic Field grid never renders that group. So a
    controlnet field classified there but NOT listed on any net's card is invisible in
    the UI while still affecting the render. (And hoisting them to an Advanced tab, as a
    first pass did, means leaving the ControlNet screen just to switch the depth model.)
    """
    from vibewarp.controlnet_catalog import CONTROLNET_CATALOG

    on_a_card = {name for spec in CONTROLNET_CATALOG.values()
                 for name in (spec.detectors or ())}
    base = {'enabled', 'models', 'mode', 'cond_image_src', 'normalize_weights'}
    classified = {key.split('.', 1)[1] for key, (tier, group) in LAYOUT.items()
                  if key.startswith('controlnet.') and group == 'ControlNet'}
    orphans = sorted(classified - base - on_a_card)
    assert not orphans, (
        'ControlNet fields that no net exposes — they render nowhere:\n  '
        + '\n  '.join(orphans))


def test_system_tier_is_paths_and_performance_only():
    """System settings must be machine-level: they do not travel with a settings file."""
    keys = system_keys()
    assert 'controlnet.model_dir' in keys
    assert 'ipadapter.clip_vision_model_path' in keys
    # ...and must NOT contain anything that changes what the render looks like
    for creative in ('diffusion.steps_schedule', 'diffusion.cfg_scale_schedule',
                     'main.text_prompts', 'controlnet.models', 'diffusion.seed'):
        assert creative not in keys


def test_hoisted_fields_get_a_meaningful_label():
    """A field shown outside its own section must not be titled from its bare name.

    animatediff.enabled sits on the Render/Model card (in the notebook the motion module IS
    the model version), where a checkbox labelled just "Enabled" says nothing.
    """
    from vibewarp.ui_layout import label

    assert label('animatediff', 'enabled') == 'AnimateDiff (motion module)'
    schema = config_schema()
    assert schema['properties']['animatediff']['properties']['enabled']['label'] \
        == 'AnimateDiff (motion module)'
    # Fields that stay in their own section need no override.
    assert label('controlnet', 'enabled') is None
    assert 'label' not in schema['properties']['diffusion']['properties']['seed']


def test_labels_only_name_fields_that_exist():
    from vibewarp.ui_layout import LABELS
    known = set(all_config_keys(RunConfig))
    stale = sorted(key for key in LABELS if key not in known)
    assert not stale, f'LABELS names fields that no longer exist: {stale}'
