"""Native still-image graphs and remote Comfy inventory, without loading weights."""
from dataclasses import asdict
from unittest.mock import Mock

import pytest
import requests
from PIL import Image

from vibewarp.config import COMFY_IMAGE_DEFAULTS, ComfyImageConfig, RunConfig, comfy_image_defaults
from vibewarp.config_io import config_from_dict, config_from_settings, config_schema, validate_config
from vibewarp.core.comfy_image import ComfyImageClient
from vibewarp.comfy_inventory import model_specs, required_comfy_files
from vibewarp.settings import save_settings, settings_to_dict


def config_for(version, mode='edit'):
    config = config_from_dict({'model_version': version, 'comfy_image': {'mode': mode}})
    config.flow.flow_warp = False
    return config


def client_for(config):
    client = ComfyImageClient.__new__(ComfyImageClient)
    client.config = config
    client.base_url = 'http://comfy.example:8188'
    client.client_id = 'test'
    return client


def test_saved_comfy_settings_restore_the_entire_section(tmp_path):
    config = config_from_dict({'model_version': 'flux3_klein', 'comfy_image': {
        'steps': 7, 'mode': 'text', 'fixed_seed': False,
        'comfy_unet_name': 'custom/klein.safetensors', 'references': [{'source': 'warped'}]}})
    path = save_settings(settings_to_dict(config), str(tmp_path), 'klein', 0)
    assert config_from_settings(path).comfy_image == config.comfy_image


def test_explicit_sampling_values_reach_moss_graph_and_survive_reload(tmp_path):
    config = config_from_dict({'model_version': 'moss_image', 'comfy_image': {
        'steps': 6, 'guidance_scale': 1.0, 'scheduler': 'simple', 'mode': 'text'}})
    saved = save_settings(settings_to_dict(config), str(tmp_path), 'moss', 0)
    for current in (config, config_from_settings(saved)):
        graph = client_for(current)._build_prompt([], 'test', '', 1, 512, 512)
        sampler = graph['30']['inputs']
        assert (sampler['steps'], sampler['cfg'], sampler['scheduler']) == (6, 1.0, 'simple')
        assert graph['1']['inputs']['unet_name'] == COMFY_IMAGE_DEFAULTS['moss_image']['comfy_unet_name']

    direct = RunConfig(model_version='moss_image')
    assert direct.comfy_image.steps == 30
    direct.comfy_image.steps = 6
    assert comfy_image_defaults(direct)['steps'] == 6
    section = ComfyImageConfig(steps=50, scheduler='simple')
    direct = RunConfig(model_version='moss_image', comfy_image=section)
    assert (direct.comfy_image.steps, direct.comfy_image.scheduler) == (50, 'simple')
    assert direct.comfy_image.comfy_unet_name == COMFY_IMAGE_DEFAULTS['moss_image']['comfy_unet_name']


@pytest.mark.parametrize('version', COMFY_IMAGE_DEFAULTS)
@pytest.mark.parametrize('mode', ['edit', 'text'])
def test_native_graphs_keep_text_mode_free_of_image_conditioning(version, mode):
    config = config_for(version, mode)
    graph = client_for(config)._build_prompt(['first.png', 'second.png'], 'paint it', 'blur', 99, 512, 512)
    assert graph['1']['inputs']['unet_name'] == COMFY_IMAGE_DEFAULTS[version]['comfy_unet_name']
    sampler = graph['30']['inputs']
    assert sampler['steps'] == COMFY_IMAGE_DEFAULTS[version]['steps']
    assert sampler['seed'] == 99 and sampler['denoise'] == 1
    assert sum(n['class_type'] == 'LoadImage' for n in graph.values()) == (2 if mode == 'edit' else 0)
    if version == 'moss_image':
        assert (graph['4']['class_type'] == 'TextEncodeMOSSImageEdit') == (mode == 'edit')
        assert any(n['class_type'] == 'LoraLoaderModelOnly' for n in graph.values()) == (mode == 'edit')
        if mode == 'edit':
            for node in ('4', '5'):
                assert graph[node]['inputs']['images.image_0'] == ['10', 0]
                assert graph[node]['inputs']['images.image_1'] == ['11', 0]
                assert 'images.image_2' not in graph[node]['inputs']
            assert graph['20']['inputs']['strength_model'] == 1
    if version == 'flux3_klein':
        assert graph['20']['inputs']['base_shift'] == pytest.approx(2.7725887)
        assert ('21' in graph) == (mode == 'edit')
        if mode == 'edit': assert graph['21']['inputs']['images.image0'] == ['10', 0]
    if version == 'minimax_h3_image':
        assert sampler['latent_image'] == ['4', 1]
        assert graph['5']['class_type'] == 'ConditioningZeroOut'
        assert ('images.image_1' in graph['4']['inputs']) == (mode == 'edit')
    assert graph['32']['class_type'] == 'PreviewImage'
    assert not validate_config(config, require_inputs=False, check_paths=False)


@pytest.mark.parametrize('version', COMFY_IMAGE_DEFAULTS)
def test_schema_routes_new_models_to_the_comfy_family_and_keeps_custom_settings(version):
    config = config_for(version)
    config.comfy_image.steps = 79
    config.comfy_image.comfy_unet_name = 'custom/nested.safetensors'
    loaded = config_from_dict(asdict(config))
    assert loaded.comfy_image.steps == 79
    assert comfy_image_defaults(loaded)['comfy_unet_name'] == 'custom/nested.safetensors'
    schema = config_schema()
    assert schema['model_family_by_version'][version] == 'comfy_image'
    assert schema['comfy_image_model_defaults'][version] == COMFY_IMAGE_DEFAULTS[version]
    from vibewarp.core.model_loader import is_edit_model, is_flux_model
    assert is_edit_model(version)
    assert not is_flux_model(version)


def test_inventory_uses_server_names_even_with_wrong_or_missing_local_mount(tmp_path, monkeypatch):
    config = config_for('moss_image')
    config.comfy_models_dir = str(tmp_path/'not-mounted')
    calls = []
    _, specs = model_specs(config)
    def get(url, **kwargs):
        calls.append(url)
        category = url.rsplit('/', 1)[-1]
        return Mock(raise_for_status=lambda: None, json=lambda: [name for folder, name in specs if folder == category])
    monkeypatch.setattr('vibewarp.comfy_inventory.requests.get', get)
    from vibewarp.model_assets import required_model_files, ensure_selected_assets
    rows = required_model_files(config)
    assert len(rows) == 4 and all(row['status'] == 'present' for row in rows)
    assert all(row['source'] == 'comfy' and not row['path'].startswith(str(tmp_path)) for row in rows)
    acquire = Mock(side_effect=AssertionError('Existing remote weights must not download locally'))
    monkeypatch.setattr('vibewarp.model_assets.acquire', acquire)
    ensure_selected_assets(config)
    acquire.assert_not_called()
    assert all('/models/' in url for url in calls)


@pytest.mark.parametrize('version', ['flux2_klein_edit','hidream_o1_edit','qwen_image_edit_2511','mage_flow_edit'])
def test_existing_comfy_models_also_use_remote_inventory(version, monkeypatch):
    config = RunConfig(model_version=version)
    config.flow.flow_warp = False
    _, specs = model_specs(config)
    def get(url, **kw):
        return Mock(raise_for_status=lambda: None,
                    json=lambda: [name for category, name in specs if url.endswith('/'+category)])
    monkeypatch.setattr('vibewarp.comfy_inventory.requests.get', get)
    from vibewarp.model_assets import required_model_files
    assert all(row['status'] == 'present' for row in required_model_files(config))


def test_unknown_connection_is_not_reported_as_missing_and_no_local_fallback(tmp_path, monkeypatch):
    config = config_for('flux3_klein')
    config.comfy_models_dir = str(tmp_path)
    path = tmp_path/'diffusion_models'/config.comfy_image.comfy_unet_name
    path.parent.mkdir(); path.write_bytes(b'local but not visible remotely')
    monkeypatch.setattr('vibewarp.comfy_inventory.requests.get', Mock(side_effect=requests.ConnectionError('offline')))
    rows = required_comfy_files(config)
    assert all(row['status'] == 'unknown' and not row['downloadable'] for row in rows)
    assert all('offline' in row['error'] for row in rows)


def test_text_mode_does_not_require_or_upload_references(monkeypatch):
    config = config_for('moss_image', 'text')
    config.comfy_image.references = []
    assert not validate_config(config, require_inputs=False, check_paths=False)
    assert all(category != 'loras' for category, _ in model_specs(config)[1])
    client = client_for(config)
    client._upload_image = Mock(side_effect=AssertionError('text mode uploaded an image'))
    client._request = Mock(return_value=Mock(json=lambda: {'prompt_id': 'task'}))
    client._wait_for_output = Mock(return_value=Image.new('RGB', (8, 8)))
    monkeypatch.setattr('vibewarp.core.comfy_image.ComfyProgressMonitor', Mock())
    client.render(config, Image.new('RGB', (8, 8)), 'cat', '', 1, 512, 512)
    client._upload_image.assert_not_called()


def test_preflight_reports_missing_native_nodes_before_queueing():
    config = config_for('minimax_h3_image')
    client = client_for(config)
    client._request = Mock(return_value=Mock(json=lambda: {}))
    with pytest.raises(RuntimeError, match='update and restart ComfyUI'):
        client._validate_server()
    assert all(call.args[0] == 'GET' for call in client._request.call_args_list)


@pytest.mark.parametrize('version', COMFY_IMAGE_DEFAULTS)
@pytest.mark.parametrize('mode', ['edit', 'text'])
def test_shared_frame_pipeline_dispatches_new_models_without_sd(tmp_path, version, mode):
    from vibewarp.core.diffusion import RenderContext, FrameState, render_frame_edit
    config = config_for(version, mode)
    config.video.width = config.video.height = 64
    path = tmp_path/'raw.png'
    Image.new('RGB', (64,64), 'red').save(path)
    if mode == 'text': config.comfy_image.references = []
    backend = Mock()
    expected = Image.new('RGB', (64,64), 'blue')
    backend.render.return_value = expected
    ctx = RenderContext(config=config, edit_backend=backend)
    state = FrameState(init_image=str(path))
    result = render_frame_edit(ctx, state)
    assert result is expected and state.prev_frame is expected
    assert backend.render.call_count == 1
    assert backend.render.call_args.kwargs['config'] is config


@pytest.mark.parametrize('filename,category,loader', [
    ('custom/model.safetensors', 'diffusion_models', 'UNETLoader'),
    ('custom/model.GGUF', 'unet_gguf', 'UnetLoaderGGUF'),
])
def test_preflight_and_graph_use_the_same_loader_and_server_spelling(filename, category, loader):
    config = config_for('flux3_klein', 'text')
    config.comfy_image.comfy_unet_name = filename
    client = client_for(config)
    _, specs = model_specs(config)
    assert (category, filename) in specs
    def request(method, path):
        if path.startswith('/object_info/'):
            node = path.rsplit('/', 1)[-1]
            return Mock(json=lambda: {node: {'input': {'required': {'type': [['flux3']]}}}})
        if path.startswith('/models/'):
            category = path.rsplit('/', 1)[-1]
            return Mock(json=lambda: [name.replace('/', '\\') for folder, name in specs if folder == category])
        return Mock(json=lambda: {})
    client._request = Mock(side_effect=request)
    client._validate_server()
    client._request.assert_any_call('GET', f'/object_info/{loader}')
    client._request.assert_any_call('GET', f'/models/{category}')
    inputs = {'unet_name': filename.replace('/', '\\')}
    if loader == 'UNETLoader':
        inputs['weight_dtype'] = 'default'
    assert client._build_prompt([], 'test', '', 1, 512, 512)['1'] == {'class_type': loader, 'inputs': inputs}
