import json
import os
from dataclasses import asdict

import pytest

from vibewarp.config import ControlNetEntry, IPAdapterEntry, RunConfig
from vibewarp.config_io import ConfigError, apply_path_defaults, config_from_dict, config_from_json, config_from_settings, config_schema, validate_config


def test_full_config_round_trip_preserves_nested_entries(tmp_path):
    original = RunConfig()
    original.text_prompts = {0: "first", 20: "second"}
    original.controlnet.models = {"canny": ControlNetEntry(path="canny.pth", layer_weights=[1.0] * 13)}
    original.ipadapter.models = {"portrait": IPAdapterEntry(path="ip.bin", weight=0.7)}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(asdict(original)), encoding="utf-8")
    loaded = config_from_json(path)
    assert loaded == original
    assert isinstance(loaded.controlnet.models["canny"], ControlNetEntry)
    assert isinstance(loaded.ipadapter.models["portrait"], IPAdapterEntry)


def test_unknown_fields_are_rejected_with_path():
    with pytest.raises(ConfigError, match="config.diffusion has unknown field.*surprise"):
        config_from_dict({"diffusion": {"surprise": True}})


def test_schema_contains_every_dataclass_group_and_defaults():
    schema = config_schema()
    assert schema["properties"]["vae"]["properties"]["tile_size"]["default"] == 128
    assert schema["properties"]["video_assembly"]["type"] == "dataclass"
    assert schema["properties"]["ipadapter"]["properties"]["models"]["type"] == "object"
    assert schema["properties"]["controlnet"]["properties"]["model_dir"]["type"] == "string"


def test_schema_marks_booleans_and_limited_values_for_ui_controls():
    schema = config_schema()["properties"]
    assert schema["flow"]["properties"]["flow_warp"]["type"] == "boolean"
    assert schema["animatediff"]["properties"]["enabled"]["type"] == "boolean"
    assert schema["diffusion"]["properties"]["noise_mode"]["choices"] == [
        "default", "fixed", "reconstructed"
    ]
    assert schema["diffusion"]["properties"]["guidance_mode"]["choices"] == [
        "prev stylized", "prev warped", "prev warped + cc"
    ]
    assert "sample_lcm" in schema["diffusion"]["properties"]["sampler"]["choices"]
    assert schema["diffusion"]["properties"]["sampler_tile_size"]["choices"] == [
        256, 512, 768, 1024
    ]
    assert schema["video_assembly"]["properties"]["upscale_ratio"]["choices"] == [1, 2, 4]
    entry = schema["ipadapter"]["properties"]["models"]["additional"]
    assert entry["properties"]["combine_embeds"]["choices"] == [
        "concat", "add", "subtract", "average", "norm average"]


def test_validation_reports_inputs_and_resolution():
    config = RunConfig()
    config.video.width = 513
    errors = validate_config(config)
    paths = {error["path"] for error in errors}
    assert {"sd_checkpoint_path", "video.video_init_path", "video.width"} <= paths


@pytest.mark.parametrize(
    'model_version',
    ['flux2_klein_edit', 'flux2_klein_9b_edit', 'hidream_o1_edit'])
def test_edit_models_ignore_hidden_sd_validation(model_version, tmp_path):
    config = RunConfig(model_version=model_version)
    config.video.video_init_path = str(tmp_path / 'video.mp4')
    config.sd_checkpoint_path = ''
    config.model_path = str(tmp_path / 'missing-model-root')
    config.lora_dir = str(tmp_path / 'missing-loras')
    config.controlnet.enabled = True
    config.animatediff.enabled = True
    config.diffusion.steps = 0
    config.diffusion.style_strength = 2
    config.diffusion.sampler_tile_size = 3

    paths = {e['path'] for e in validate_config(
        config, require_inputs=True, check_paths=True)}

    assert 'sd_checkpoint_path' not in paths
    assert 'model_path' not in paths
    assert 'lora_dir' not in paths
    assert not any(path.startswith('animatediff.') for path in paths)
    assert not any(path.startswith('diffusion.') for path in paths)


def test_validation_reports_invalid_tiled_sampler_geometry():
    config = RunConfig()
    config.diffusion.sampler_tile_size = 510
    config.diffusion.sampler_tile_overlap = 51
    paths = {error["path"] for error in validate_config(config, require_inputs=False)}
    assert "diffusion.sampler_tile_size" in paths
    assert "diffusion.sampler_tile_overlap" in paths


def test_validation_rejects_reference_label_opacity_outside_fraction():
    config = RunConfig(model_version='flux2_klein_edit')
    config.reference_label_opacity = 1.1
    paths = {error["path"] for error in validate_config(
        config, require_inputs=False)}
    assert "reference_label_opacity" in paths


def test_ipadapter_rejects_raw_source_and_missing_fixed_image():
    config = RunConfig()
    config.ipadapter.enabled = True
    config.ipadapter.models = {
        'raw': IPAdapterEntry(source_image={
            'source': 'raw_frame', 'image_path': ''}),
        'fixed': IPAdapterEntry(source_image={
            'source': 'upload', 'image_path': ''}),
    }
    messages = [
        error['message'] for error in validate_config(
            config, require_inputs=False)]
    assert any('raw-frame input is not supported' in message for message in messages)
    assert any('choose a fixed reference image' in message for message in messages)


def test_validation_rejects_color_match_strengths_outside_fraction():
    config = RunConfig()
    config.color.before_strength = -0.1
    config.color.after_strength = 1.1

    paths = {error["path"] for error in validate_config(
        config, require_inputs=False)}

    assert {'color.before_strength', 'color.after_strength'} <= paths


def test_validation_can_check_local_input_paths(tmp_path):
    config = RunConfig(
        sd_checkpoint_path=str(tmp_path / "missing-model.safetensors")
    )
    config.video.video_init_path = str(tmp_path / "missing-video.mp4")
    errors = validate_config(config, check_paths=True)
    assert {error["path"] for error in errors} == {
        "sd_checkpoint_path", "video.video_init_path"
    }


def test_validation_rejects_nonexistent_input_directories(tmp_path):
    config = RunConfig(root_dir=str(tmp_path))
    config.lora_dir = str(tmp_path / "missing-loras")
    config.controlnet.model_dir = str(tmp_path / "missing-controlnets")
    config.model_path = str(tmp_path / "missing-model-root")
    errors = validate_config(config, require_inputs=False, check_paths=True)
    assert {error["path"] for error in errors} == {
        "lora_dir", "controlnet.model_dir", "model_path"
    }


def test_blank_writable_paths_use_defaults():
    config = RunConfig(root_dir="", output_dir="")
    apply_path_defaults(config)
    assert config.root_dir == os.getcwd()
    assert config.output_dir == "images_out"


def test_structured_settings_file_loads(tmp_path):
    path = tmp_path / "settings.txt"
    path.write_text(json.dumps({"diffusion": {"steps": 44}, "video": {"width": 768}}), encoding="utf-8")
    config = config_from_settings(path)
    assert config.diffusion.steps == 44
    assert config.video.width == 768


def test_vibewarp_snapshot_loads_new_feature_groups(tmp_path):
    snapshot = {
        "flow_flow_warp": True,
        "freeu_do_freeunet": True,
        "mask_use_background_mask": True,
        "captions_make_captions": True,
        "scene_analyze_video": True,
        "lora_merge_precision": "fp32",
    }
    path = tmp_path / "snapshot.txt"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    config = config_from_settings(path)
    assert config.freeu.do_freeunet is True
    assert config.mask.use_background_mask is True
    assert config.captions.make_captions is True
    assert config.scene.analyze_video is True
    assert config.lora_merge_precision == "fp32"
