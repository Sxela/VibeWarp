"""ControlNet catalog: the single source of truth for keys, files, and modes."""

import pytest
from fastapi.testclient import TestClient

from vibewarp.controlnet_catalog import (
    CONTROLNET_CATALOG,
    CONTROLNET_MODES,
    MODE_PRESETS,
    annotator_map,
    controlnet_filenames,
    resolve_mode,
    specs_for_version,
)
from vibewarp.web import create_app
from vibewarp.web_jobs import JobManager


def client():
    return TestClient(create_app(JobManager(runner=lambda *_args, **_kwargs: [])))


# ---- Catalog integrity ----

class TestCatalog:
    def test_annotator_map_matches_engine_dispatch(self):
        """core.controlnet derives ANNOTATOR_MAP from the catalog — no drift."""
        from vibewarp.core.controlnet import ANNOTATOR_MAP
        assert ANNOTATOR_MAP == annotator_map()
        assert len(ANNOTATOR_MAP) == 29

    def test_every_net_has_a_known_annotator(self):
        from vibewarp.core.controlnet import ANNOTATOR_MAP
        for key, spec in CONTROLNET_CATALOG.items():
            assert spec.annotator, f"{key} has no annotator"
            assert ANNOTATOR_MAP[key] == spec.annotator

    def test_sdxl_nets_are_offered(self):
        """The old hardcoded UI list was SD1.5-only, so SDXL was unreachable."""
        keys = {spec.key for spec in specs_for_version('sdxl')}
        assert 'control_sdxl_canny' in keys
        assert 'control_sdxl_temporalnet_v1' in keys
        assert not any(key.startswith('control_sd15') for key in keys)

    def test_version_filter_families(self):
        assert {s.model_version for s in specs_for_version('control_multi_v15')} == {'sd15'}
        assert {s.model_version for s in specs_for_version('v1_5')} == {'sd15'}
        assert {s.model_version for s in specs_for_version('sdxl')} == {'sdxl'}
        # No version = everything.
        assert len(specs_for_version('')) == len(CONTROLNET_CATALOG)

    def test_lineart_anime_filename_is_the_real_one(self):
        """Regression: the file is sd15s2, not sd15 — the old maps could never
        resolve or download it."""
        spec = CONTROLNET_CATALOG['control_sd15_lineart_anime']
        assert spec.filename == 'control_v11p_sd15s2_lineart_anime.pth'
        assert spec.url.endswith('control_v11p_sd15s2_lineart_anime.pth')

    def test_filenames_are_unique(self):
        """Several nets publish as diffusion_pytorch_model.safetensors upstream;
        on disk they must not collide."""
        names = list(controlnet_filenames().values())
        assert len(names) == len(set(names))

    def test_detectors_reference_real_config_fields(self):
        from vibewarp.config import ControlNetConfig
        valid = {f.name for f in ControlNetConfig.__dataclass_fields__.values()}
        for key, spec in CONTROLNET_CATALOG.items():
            for name in spec.detectors:
                assert name in valid, f"{key} references unknown field {name}"


# ---- Mode presets (notebook parity) ----

class TestModes:
    def test_presets_match_notebook(self):
        """refs/notebook_dump.txt: balanced=[1]*13; controlnet/prompt=0.825**(12-i);
        zero_uncond only for 'controlnet'."""
        assert resolve_mode('balanced') == ([1.0] * 13, False)
        curve = [0.825 ** float(12 - i) for i in range(13)]
        assert resolve_mode('controlnet') == (curve, True)
        assert resolve_mode('prompt') == (curve, False)

    def test_custom_keeps_hand_set_weights(self):
        assert resolve_mode('custom') is None
        assert resolve_mode('nonsense') is None

    def test_custom_is_offered_but_is_not_a_preset(self):
        assert 'custom' in CONTROLNET_MODES
        assert 'custom' not in MODE_PRESETS

    def test_schema_offers_every_mode(self):
        from vibewarp.config_io import config_schema
        entry = config_schema()['properties']['controlnet']['properties']['models']
        assert list(CONTROLNET_MODES) == ['balanced', 'controlnet', 'prompt', 'custom']
        # The mode choices reach the UI via FIELD_CHOICES.
        from vibewarp.config_io import FIELD_CHOICES
        assert FIELD_CHOICES['ControlNetEntry.mode'] == list(CONTROLNET_MODES)

    def test_returned_weights_are_copies(self):
        """Callers mutate layer_weights; presets must not be shared state."""
        first, _ = resolve_mode('balanced')
        first[0] = 99.0
        second, _ = resolve_mode('balanced')
        assert second[0] == 1.0


# ---- mode is honoured outside the WarpFusion importer ----

class TestModeDerivation:
    def test_pipeline_derives_weights_from_mode(self, monkeypatch, tmp_path):
        """Regression: `mode` was written into the CN dict but never read, and the
        derivation lived only in the WarpFusion importer — so a UI/JSON config with
        mode='controlnet' silently rendered as 'balanced'."""
        from vibewarp import pipeline
        from vibewarp.config import ControlNetEntry, RunConfig

        checkpoint = tmp_path / "cn.pth"
        checkpoint.write_bytes(b"x")

        config = RunConfig(sd_checkpoint_path=str(tmp_path / "sd.safetensors"))
        config.controlnet.enabled = True
        config.controlnet.models = {
            'control_sd15_depth': ControlNetEntry(path=str(checkpoint), mode='controlnet'),
        }

        captured = {}
        monkeypatch.setattr(pipeline, 'load_sd_model', lambda *a, **k: object())
        monkeypatch.setattr(pipeline, 'wrap_model_for_kdiffusion', lambda *a, **k: (None, None))
        monkeypatch.setattr(pipeline, 'load_controlnet', lambda *a, **k: 'cn-model')

        result = pipeline.load_models(config, device='cpu')
        captured = result['controlnets']['control_sd15_depth']

        assert captured['zero_uncond'] is True
        assert captured['layer_weights'] == [0.825 ** float(12 - i) for i in range(13)]

    def test_pipeline_keeps_custom_weights(self, monkeypatch, tmp_path):
        from vibewarp import pipeline
        from vibewarp.config import ControlNetEntry, RunConfig

        checkpoint = tmp_path / "cn.pth"
        checkpoint.write_bytes(b"x")
        hand_set = [0.5] * 13

        config = RunConfig(sd_checkpoint_path=str(tmp_path / "sd.safetensors"))
        config.controlnet.enabled = True
        config.controlnet.models = {
            'control_sd15_depth': ControlNetEntry(
                path=str(checkpoint), mode='custom', layer_weights=hand_set, zero_uncond=True),
        }
        monkeypatch.setattr(pipeline, 'load_sd_model', lambda *a, **k: object())
        monkeypatch.setattr(pipeline, 'wrap_model_for_kdiffusion', lambda *a, **k: (None, None))
        monkeypatch.setattr(pipeline, 'load_controlnet', lambda *a, **k: 'cn-model')

        entry = pipeline.load_models(config, device='cpu')['controlnets']['control_sd15_depth']
        assert entry['layer_weights'] == hand_set
        assert entry['zero_uncond'] is True


# ---- /api/controlnet/catalog ----

class TestCatalogEndpoint:
    def test_lists_nets_and_presets(self):
        body = client().get("/api/controlnet/catalog").json()
        keys = {net['key'] for net in body['nets']}
        assert keys == set(CONTROLNET_CATALOG)
        assert body['modes'] == list(CONTROLNET_MODES)
        assert body['mode_presets']['balanced']['layer_weights'] == [1.0] * 13
        assert body['mode_presets']['controlnet']['zero_uncond'] is True

    def test_filters_by_model_version(self):
        body = client().get("/api/controlnet/catalog", params={"model_version": "sdxl"}).json()
        assert {net['model_version'] for net in body['nets']} == {'sdxl'}

    def test_discovers_checkpoints_on_disk(self, tmp_path):
        (tmp_path / "control_v11f1p_sd15_depth.pth").write_bytes(b"x")
        (tmp_path / "notes.txt").write_text("ignored")
        body = client().get("/api/controlnet/catalog",
                            params={"model_dir": str(tmp_path)}).json()
        assert len(body['files']) == 1
        depth = next(n for n in body['nets'] if n['key'] == 'control_sd15_depth')
        canny = next(n for n in body['nets'] if n['key'] == 'control_sd15_canny')
        assert depth['resolved_path'] == body['files'][0]
        assert canny['resolved_path'] is None  # surfaces as "Missing" in the UI

    def test_missing_model_dir_is_not_an_error(self):
        body = client().get("/api/controlnet/catalog",
                            params={"model_dir": "/nope/does/not/exist"}).json()
        assert body['files'] == []
        assert body['nets']


class TestSDXLInTheUI:
    """SDXL renders as of 2026-07-12, so the UI offers it — which is also what
    makes the SDXL ControlNets reachable, since the panel filters by model_version."""

    def test_model_version_offers_sdxl(self):
        from vibewarp.config_io import FIELD_CHOICES
        choices = FIELD_CHOICES['RunConfig.model_version']
        assert 'control_multi_sdxl' in choices
        assert 'control_multi_v15' in choices
        # AnimateDiff's render path is broken — don't offer a version that needs it.
        assert 'control_multi_animatediff_sdxl' not in choices

    def test_selecting_sdxl_switches_the_controlnet_panel(self):
        body = client().get("/api/controlnet/catalog",
                            params={"model_version": "control_multi_sdxl"}).json()
        keys = {net['key'] for net in body['nets']}
        assert 'control_sdxl_canny' in keys
        assert not any(k.startswith('control_sd15') for k in keys)

    def test_mismatched_controlnet_is_rejected(self, tmp_path):
        """Switching model_version with nets already configured would otherwise
        crash at load with a shape mismatch."""
        from vibewarp.config import ControlNetEntry, RunConfig
        from vibewarp.config_io import validate_config

        config = RunConfig(model_version='control_multi_sdxl',
                           sd_checkpoint_path=str(tmp_path / 'sdxl.safetensors'))
        config.video.video_init_path = str(tmp_path / 'v.mp4')
        config.controlnet.enabled = True
        config.controlnet.models = {'control_sd15_depth': ControlNetEntry(path='x.pth')}

        errors = validate_config(config)
        assert any(e['path'] == 'controlnet.models.control_sd15_depth'
                   and 'sd15' in e['message'] for e in errors)

    def test_matching_controlnet_is_accepted(self, tmp_path):
        from vibewarp.config import ControlNetEntry, RunConfig
        from vibewarp.config_io import validate_config

        config = RunConfig(model_version='control_multi_sdxl',
                           sd_checkpoint_path=str(tmp_path / 'sdxl.safetensors'))
        config.video.video_init_path = str(tmp_path / 'v.mp4')
        config.controlnet.enabled = True
        config.controlnet.models = {'control_sdxl_canny': ControlNetEntry(path='x.safetensors')}

        errors = validate_config(config)
        assert not [e for e in errors if e['path'].startswith('controlnet.models')]

    def test_disabled_controlnets_are_not_checked(self, tmp_path):
        from vibewarp.config import ControlNetEntry, RunConfig
        from vibewarp.config_io import validate_config

        config = RunConfig(model_version='control_multi_sdxl',
                           sd_checkpoint_path=str(tmp_path / 'sdxl.safetensors'))
        config.video.video_init_path = str(tmp_path / 'v.mp4')
        config.controlnet.enabled = False
        config.controlnet.models = {'control_sd15_depth': ControlNetEntry(path='x.pth')}

        errors = validate_config(config)
        assert not [e for e in errors if e['path'].startswith('controlnet.models')]
