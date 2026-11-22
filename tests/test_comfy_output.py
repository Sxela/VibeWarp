"""Exercise history-to-image retrieval; graph construction alone missed this bug."""
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from vibewarp.core.flux_comfy import ComfyFluxClient, ComfyFluxError
from vibewarp.core.comfy_image import ComfyImageClient
from vibewarp.core.mage_comfy import ComfyMageFlowEditClient
from vibewarp.core.qwen_comfy import ComfyQwenEditClient
from vibewarp.core.hidream_comfy import ComfyHiDreamClient


@pytest.mark.parametrize('client_type,node', [
    (ComfyImageClient, '32'), (ComfyFluxClient, '37'),
    (ComfyMageFlowEditClient, '37'), (ComfyQwenEditClient, '37'), (ComfyHiDreamClient, '9')])
def test_history_reader_uses_the_backend_output_node(client_type, node):
    client = client_type.__new__(client_type)
    client.timeout = 1
    data = io.BytesIO()
    Image.new('RGB', (7, 9), 'red').save(data, format='PNG')
    calls = []
    def request(method, path, **kwargs):
        calls.append((method, path, kwargs))
        if path == '/history/task':
            return SimpleNamespace(json=lambda: {'task': {
                'status': {'status_str': 'success', 'completed': True},
                'outputs': {'other': {'images': [{'filename': 'wrong.png'}]},
                    node: {'images': [{'filename': 'result.png', 'subfolder': 'references', 'type': 'temp'}]}}}})
        assert path == '/view'
        assert kwargs['params'] == {'filename': 'result.png', 'subfolder': 'references', 'type': 'temp'}
        return SimpleNamespace(content=data.getvalue())
    client._request = request
    result = client._wait_for_output('task')
    assert result.size == (7, 9) and result.getpixel((0, 0)) == (255, 0, 0)
    assert all(method == 'GET' for method, _, _ in calls)


def test_missing_output_reports_expected_and_returned_nodes():
    client = ComfyImageClient.__new__(ComfyImageClient)
    client.timeout = 1
    client._request = lambda *args, **kwargs: SimpleNamespace(json=lambda: {'task': {
        'status': {'status_str': 'success', 'completed': True},
        'outputs': {'37': {'images': [{'filename': 'unrelated.png'}]}}}})
    with pytest.raises(ComfyFluxError, match="output node 32; returned output nodes: \\['37'\\]"):
        client._wait_for_output('task')


def test_execution_error_is_not_misreported_as_missing_image():
    client = ComfyImageClient.__new__(ComfyImageClient)
    client.timeout = 1
    client._request = lambda *args, **kwargs: SimpleNamespace(json=lambda: {'task': {
        'status': {'status_str': 'error', 'completed': False,
            'messages': [['execution_error', {'exception_message': 'actual Comfy failure'}]]}}})
    with pytest.raises(ComfyFluxError, match='actual Comfy failure'):
        client._wait_for_output('task')


WORKFLOWS = [
    ('flux3_klein','edit'), ('flux3_klein','text'),
    ('moss_image','edit'), ('moss_image','text'),
    ('minimax_h3_image','edit'), ('minimax_h3_image','text'),
    ('flux2_klein_edit','edit'), ('flux2_klein_9b_edit','edit'),
    ('qwen_image_edit_2511','edit'), ('qwen_image_edit_2511_gguf','edit'),
    ('mage_flow_edit','edit'), ('mage_flow_edit_turbo','edit'),
    ('hidream_o1_edit','edit'),
]


@pytest.mark.parametrize('version,mode', WORKFLOWS)
def test_every_workflow_submits_and_reads_its_generated_output(version, mode, monkeypatch):
    """Fake only HTTP, not the output reader: catch graph/reader integration drift."""
    from unittest.mock import Mock
    from vibewarp.config_io import config_from_dict
    from vibewarp.config import flux_model_files, qwen_model_files, mage_model_defaults
    from vibewarp.ui_layout import model_family_for_version
    family = model_family_for_version(version)
    config = config_from_dict({'model_version': version, 'comfy_image': {'mode': mode}})
    config.hidream.use_trained_resolution = False
    clients = {'comfy_image': ComfyImageClient, 'flux': ComfyFluxClient,
               'qwen': ComfyQwenEditClient, 'mage': ComfyMageFlowEditClient,
               'hidream': ComfyHiDreamClient}
    client = clients[family].__new__(clients[family])
    client.config = config
    client.base_url = 'http://comfy.example:8188'
    client.client_id = 'client'
    client.timeout = 1
    section = getattr(config, family)
    if family in ('flux','qwen','mage'):
        resolver = {'flux': flux_model_files, 'qwen': qwen_model_files, 'mage': mage_model_defaults}[family]
        values = {**vars(section), **resolver(config)}
        for attr, key in [('unet_name','comfy_unet_name'), ('clip_name','comfy_clip_name'), ('vae_name','comfy_vae_name')]:
            setattr(client, attr, values[key])
    if family == 'qwen':
        client.lora_name = section.comfy_lora_name
        client.use_lightning_lora = section.use_lightning_lora
    if family == 'hidream': client.checkpoint_name = section.comfy_checkpoint_name
    client._upload_image = lambda image, slot: f'reference-{slot}.png'
    module = clients[family].__module__
    monkeypatch.setattr(f'{module}.ComfyProgressMonitor', Mock())
    output = io.BytesIO()
    Image.new('RGB', (16,16), 'blue').save(output, format='PNG')
    submitted = {}
    calls = []
    def request(method, path, **kwargs):
        calls.append((method, path))
        if method == 'POST' and path == '/prompt':
            graph = kwargs['json']['prompt']
            outputs = [node for node, value in graph.items() if value['class_type'] == 'PreviewImage']
            assert len(outputs) == 1
            submitted.update(graph=graph, output=outputs[0])
            return SimpleNamespace(json=lambda: {'prompt_id': 'task'})
        if path == '/history/task':
            return SimpleNamespace(json=lambda: {'task': {
                'status': {'status_str': 'success', 'completed': True},
                'outputs': {submitted['output']: {'images': [{'filename': 'result.png'}]}}}})
        assert path == '/view' and kwargs['params']['filename'] == 'result.png'
        return SimpleNamespace(content=output.getvalue())
    client._request = request
    args = dict(config=config, edit_image=Image.new('RGB',(16,16),'red'),
        prompt='test', seed=123, width=512, height=512)
    if family != 'flux': args['negative_prompt'] = 'blur'
    result = client.render(**args)
    assert result.size == (16,16) and result.getpixel((0,0)) == (0,0,255)
    assert submitted['output'] == client.output_node_id
    assert calls == [('POST','/prompt'),('GET','/history/task'),('GET','/view')]
    assert client._active_progress is None
