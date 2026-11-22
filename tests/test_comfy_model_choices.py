from dataclasses import asdict
from unittest.mock import Mock

import pytest

from vibewarp.config import comfy_image_defaults, flux_model_files
from vibewarp.config_io import config_from_dict, config_schema
from vibewarp.comfy_inventory import comfy_model_catalog, model_fields


def config_for(version='flux3_klein', **kwargs):
    config = config_from_dict({'model_version': version, 'comfy_image': kwargs})
    config.flow.flow_warp = False
    return config


def mock_inventory(monkeypatch, config, folders=None, missing_field=None):
    _, fields = model_fields(config)
    def get(url, **kwargs):
        if url.endswith('/experiment/models'):
            data = [{'name': key, 'folders': [str(value)]} for key, value in (folders or {}).items()]
        else:
            category = url.rsplit('/', 1)[-1]
            data = [row['name'] for row in fields if row['category'] == category and row['name'] and row['field'] != missing_field]
            data += ['unrelated.safetensors', 'QWEN_alternative.safetensors']
        return Mock(raise_for_status=lambda: None, json=lambda: data)
    monkeypatch.setattr('vibewarp.comfy_inventory.requests.get', get)


def test_explicit_full_vae_is_not_replaced_with_the_moss_preset():
    config = config_for('moss_image', comfy_vae_name='flux2-vae.safetensors')
    assert comfy_image_defaults(config)['comfy_vae_name'] == 'flux2-vae.safetensors'
    loaded = config_from_dict(asdict(config))
    assert comfy_image_defaults(loaded)['comfy_vae_name'] == 'flux2-vae.safetensors'
    assert '_explicit_fields' not in asdict(config)['comfy_image']
    assert config_schema()['comfy_model_suggestions']['moss_image']['comfy_vae_name'] == 'flux2-vae_small.safetensors'


def test_gguf_inventory_uses_loader_specific_listing_but_legacy_download_folder(tmp_path, monkeypatch):
    config = config_for('qwen_image_edit_2511_gguf')
    folder = tmp_path/'unet'
    folder.mkdir()
    config.comfy_models_dir = str(tmp_path)
    name = config.qwen.comfy_unet_name
    calls = []
    present = False
    def get(url, **kwargs):
        calls.append(url)
        assert not url.endswith('/models/unet')
        names = [name] if present and url.endswith('/models/unet_gguf') else []
        return Mock(raise_for_status=lambda: None, json=lambda: names)
    monkeypatch.setattr('vibewarp.comfy_inventory.requests.get', get)
    row = next(r for r in comfy_model_catalog(config) if r['field'] == 'comfy_unet_name')
    assert row['category'] == 'unet_gguf'
    assert row['status'] == 'missing' and row['downloadable']
    assert row['download_path'] == str(folder/name)
    present = True
    row = next(r for r in comfy_model_catalog(config) if r['field'] == 'comfy_unet_name')
    assert row['status'] == 'present'
    assert any(url.endswith('/models/unet_gguf') for url in calls)


def test_gguf_download_folder_is_discovered_from_loader_alias(tmp_path, monkeypatch):
    config = config_for('qwen_image_edit_2511_gguf')
    folder = tmp_path/'models'/'unet'
    folder.mkdir(parents=True)
    mock_inventory(monkeypatch, config, {'unet_gguf': folder}, missing_field='comfy_unet_name')
    row = next(r for r in comfy_model_catalog(config) if r['field'] == 'comfy_unet_name')
    assert row['status'] == 'missing' and row['downloadable']
    assert row['download_path'] == str(folder/row['name'])


def test_explicit_alternative_from_another_flux_preset_is_preserved():
    config = config_from_dict({'model_version': 'flux2_klein_9b_edit',
        'flux': {'comfy_unet_name': 'flux-2-klein-4b-fp8.safetensors'}})
    assert flux_model_files(config)['comfy_unet_name'] == 'flux-2-klein-4b-fp8.safetensors'


def test_catalog_keeps_suggestion_separate_from_selected_file(monkeypatch):
    config = config_for(comfy_clip_name='qwen3-vl-4b-heretic_int8_dynamic_convrot.safetensors')
    mock_inventory(monkeypatch, config)
    row = next(row for row in comfy_model_catalog(config) if row['field'] == 'comfy_clip_name')
    assert row['status'] == 'present'
    assert row['suggested'] == 'qwen3vl_4b_int8_convrot.safetensors'
    assert config.comfy_image.comfy_clip_name in row['available']
    assert 'QWEN_alternative.safetensors' in row['available']


def test_local_server_reported_folder_enables_missing_suggestion_download(tmp_path, monkeypatch):
    config = config_for()
    folder = tmp_path/'comfy'/'text_encoders'
    folder.mkdir(parents=True)
    mock_inventory(monkeypatch, config, {'text_encoders': folder}, missing_field='comfy_clip_name')
    row = next(row for row in comfy_model_catalog(config) if row['field'] == 'comfy_clip_name')
    assert row['name'] == row['suggested'] and row['status'] == 'missing'
    assert row['downloadable'] and row['download_path'] == str(folder/row['name'])
    from vibewarp.web_downloads import DownloadManager
    specs = DownloadManager()._specs(config)
    assert len(specs) == 1
    assert specs[0][2] == row['download_path']
    assert specs[0][0][1].endswith('/text_encoders/'+row['suggested'])
    assert not list(folder.iterdir())  # listing/queue specification does not download


def test_remote_server_does_not_treat_reported_paths_as_local(tmp_path, monkeypatch):
    config = config_for()
    config.comfy_image.comfy_server_url = 'http://comfy.example:8188'
    folder = tmp_path/'encoders'; folder.mkdir()
    mock_inventory(monkeypatch, config, {'text_encoders': folder}, missing_field='comfy_clip_name')
    row = next(row for row in comfy_model_catalog(config) if row['field'] == 'comfy_clip_name')
    assert row['status'] == 'missing' and not row['downloadable']
    config.comfy_models_dir = str(tmp_path/'shared-mount')
    row = next(row for row in comfy_model_catalog(config) if row['field'] == 'comfy_clip_name')
    assert row['downloadable']
    assert row['download_path'] == str(tmp_path/'shared-mount'/'text_encoders'/row['name'])


def test_model_endpoint_returns_field_choices_and_missing_download(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from vibewarp.web import create_app
    config = config_for()
    folder = tmp_path/'text_encoders'; folder.mkdir()
    mock_inventory(monkeypatch, config, {'text_encoders': folder}, missing_field='comfy_clip_name')
    response = TestClient(create_app()).post('/api/models/required', json=asdict(config))
    assert response.status_code == 200
    row = next(row for row in response.json()['comfy_models'] if row['field'] == 'comfy_clip_name')
    assert row['suggested'] not in row['available']
    assert row['downloadable']
    required = next(row for row in response.json()['files'] if row['field'] == 'comfy_clip_name')
    assert required['download_path'] == str(folder/required['name'])
