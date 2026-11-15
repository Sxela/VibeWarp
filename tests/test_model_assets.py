from dataclasses import asdict
from pathlib import Path
import time

import pytest
from fastapi.testclient import TestClient

from vibewarp.config import RunConfig, ControlNetEntry, IPAdapterEntry
from vibewarp.model_assets import selected_assets, ensure_selected_assets
from vibewarp.web_downloads import DownloadManager


def test_relative_asset_paths_share_the_same_progress_identity(tmp_path, monkeypatch):
    from vibewarp.model_assets import ModelAsset
    monkeypatch.chdir(tmp_path)
    assert ModelAsset('https://example.org/a', 'models/a.pt').path == str(tmp_path/'models/a.pt')


def test_required_list_is_read_only_includes_custom_paths_and_siglip(tmp_path,monkeypatch):
    from vibewarp.model_assets import required_model_files
    from vibewarp.web import create_app
    config=RunConfig(model_version='disco_512',root_dir=str(tmp_path))
    config.flow.flow_warp=False
    config.disco.checkpoint_path=str(tmp_path/'custom.pt')
    config.disco.encoders=['siglip2-base']
    def forbidden(*args,**kwargs):
        raise AssertionError('Listing must not acquire assets or start a download task')
    monkeypatch.setattr('vibewarp.model_assets.acquire',forbidden)
    monkeypatch.setattr('vibewarp.web.download_manager.prepare',forbidden)
    client=TestClient(create_app())
    reply=client.post('/api/models/required',json=asdict(config))
    assert reply.status_code==200
    files=reply.json()['files']
    custom=next(file for file in files if file['name']=='Disco checkpoint')
    modern=next(file for file in files if file['name'].startswith('siglip2-base'))
    assert custom['status']=='missing' and not custom['downloadable']
    assert modern['status']=='missing' and modern['downloadable']
    assert not tmp_path.joinpath('models').exists()
    (tmp_path/'custom.pt').write_bytes(b'checkpoint')
    files=required_model_files(config)
    assert next(file for file in files if file['name']=='Disco checkpoint')['status']=='present'


def test_siglip_preview_checks_local_bundle_without_contacting_hub(tmp_path):
    from vibewarp.model_assets import required_model_files
    config=RunConfig(model_version='disco_512',root_dir=str(tmp_path))
    config.flow.flow_warp=False
    config.disco.encoders=['siglip2-base']
    root=tmp_path/'models'/'disco'/'siglip2-base'
    root.mkdir(parents=True)
    for name in ('config.json','preprocessor_config.json','tokenizer_config.json','tokenizer.json','model.safetensors'):
        (root/name).write_bytes(b'local')
    files=required_model_files(config)
    assert next(file for file in files if file['name'].startswith('siglip2-base'))['status']=='present'


def test_only_selected_nonzero_models_are_acquired(tmp_path, monkeypatch):
    config = RunConfig(root_dir=str(tmp_path), sd_checkpoint_path='custom.safetensors')
    config.flow.flow_warp = False
    config.controlnet.enabled = True
    config.controlnet.models = {
        'control_sd15_canny': ControlNetEntry(),
        'control_sd15_depth': ControlNetEntry(weight=0),
    }
    config.ipadapter.enabled = True
    config.ipadapter.models = {'ipadapter_sd15': IPAdapterEntry()}
    assets = selected_assets(config)
    assert len(assets) == 3  # one CN, one adapter, one shared vision encoder
    assert not any('depth' in asset.path for asset in assets)
    assert any('canny' in asset.path for asset in assets)
    calls = []
    monkeypatch.setattr('vibewarp.model_assets.acquire', lambda url, path, **kwargs: calls.append(path))
    ensure_selected_assets(config)
    assert len(calls) == 3
    assert config.controlnet.model_dir == str(tmp_path/'models'/'ControlNet')
    assert config.ipadapter.clip_vision_model_path == config.controlnet.model_dir


@pytest.mark.parametrize('shared_directory', [False, True])
def test_preparation_preserves_custom_relative_model_paths(tmp_path, monkeypatch, shared_directory):
    from vibewarp.pipeline import _resolve_controlnet_path
    from vibewarp.ipadapter_catalog import resolve_ipadapter_path
    monkeypatch.chdir(tmp_path)
    directory = tmp_path/'custom-models' if shared_directory else tmp_path
    directory.mkdir(exist_ok=True)
    for name in ('canny.safetensors', 'adapter.safetensors'):
        (directory/name).write_bytes(b'custom weights')
    config = RunConfig(root_dir=str(tmp_path), sd_checkpoint_path='checkpoint.safetensors')
    config.flow.flow_warp = False
    config.controlnet.model_dir = str(directory) if shared_directory else ''
    config.controlnet.models = {'control_sd15_canny': ControlNetEntry(path='canny.safetensors')}
    config.ipadapter.enabled = True
    config.ipadapter.models = {'ipadapter_sd15': IPAdapterEntry(path='adapter.safetensors')}
    config.ipadapter.clip_vision_model_path = 'vision.safetensors'
    monkeypatch.setattr('vibewarp.model_assets.acquire',
                        lambda *a, **kw: pytest.fail('Custom weights must not be downloaded'))
    ensure_selected_assets(config)
    resolved = [
        _resolve_controlnet_path('control_sd15_canny', config.controlnet.models['control_sd15_canny'].path,
                                 config.controlnet.model_dir),
        resolve_ipadapter_path('ipadapter_sd15', config.ipadapter.models['ipadapter_sd15'].path,
                               config.controlnet.model_dir),
    ]
    assert [Path(path).resolve() for path in resolved] == [directory/'canny.safetensors', directory/'adapter.safetensors']
    assert all(Path(path).is_file() for path in resolved)
    ensure_selected_assets(config)
    assert config.controlnet.model_dir == str(directory if shared_directory else tmp_path/'models'/'ControlNet')


def test_comfy_assets_target_configured_server_model_directory(tmp_path):
    config = RunConfig(model_version='mage_flow_edit_turbo', comfy_models_dir=str(tmp_path))
    config.flow.flow_warp = False
    assets = selected_assets(config)
    assert len(assets) == 3
    assert any('mage_flow_edit_turbo_int8' in asset.path for asset in assets)
    assert not any('/mage_flow_edit_int8' in asset.url for asset in assets)
    assert all(asset.path.startswith(str(tmp_path)) for asset in assets)


def test_model_preparation_endpoint_reports_downloads_without_input_video(tmp_path, monkeypatch):
    from vibewarp.web import create_app
    from vibewarp.acquisition import DownloadEvent, report
    from vibewarp.web_jobs import JobManager
    manager = DownloadManager()
    monkeypatch.setattr('vibewarp.web.download_manager', manager)
    from vibewarp.model_assets import ModelAsset
    monkeypatch.setattr('vibewarp.web_downloads.selected_assets', lambda c: [ModelAsset('https://example.org/weights.pt',str(tmp_path/'weights.pt'))])
    def prepare(*args, **kwargs):
        report(DownloadEvent('weights.pt', str(tmp_path/'weights.pt'), 5, 10))
        report(DownloadEvent('weights.pt', str(tmp_path/'weights.pt'), 10, 10, 'ready'))
    monkeypatch.setattr('vibewarp.web_downloads.acquire', prepare)
    client = TestClient(create_app(JobManager()))
    config = RunConfig(root_dir=str(tmp_path))
    config.flow.flow_warp = False
    response = client.post('/api/models/prepare', json=asdict(config))
    assert response.status_code == 200
    task_id = response.json()['id']
    deadline = time.monotonic()+2
    while time.monotonic() < deadline:
        result = client.get(f'/api/models/prepare/{task_id}').json()
        if result['state'] == 'completed':
            break
        time.sleep(.01)
    assert result['state'] == 'completed'
    assert result['files'][0]['downloaded'] == result['files'][0]['total'] == 10
    assert client.get('/api/models/prepare/no-such-id').status_code == 404


def test_first_render_download_events_reach_job_monitor(tmp_path):
    from vibewarp.web_jobs import JobManager
    from vibewarp.acquisition import report, DownloadEvent
    def runner(config, progress_fn):
        report(DownloadEvent('weights.pt', str(tmp_path/'weights.pt'), 10, 10, 'ready'))
        return []
    job = JobManager(runner=runner).submit(RunConfig())
    deadline = time.monotonic()+2
    while job.state not in ('completed', 'failed') and time.monotonic() < deadline:
        time.sleep(.01)
    assert job.state == 'completed'
    assert job.snapshot()['downloads'][0]['status'] == 'ready'
