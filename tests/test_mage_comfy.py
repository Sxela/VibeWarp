"""Mage-Flow-Edit configuration, presets, dispatch, and Comfy graph tests."""

from types import SimpleNamespace

from PIL import Image

from vibewarp.config import MageFlowEditConfig, RunConfig
from vibewarp.config_io import FIELD_CHOICES, config_from_dict, config_to_dict
from vibewarp.core.edit import render_edit_frame
from vibewarp.core.mage_comfy import ComfyMageFlowEditClient
from vibewarp.core.model_loader import is_edit_model, is_mage_flow_edit_model


def _client_without_network():
    client = ComfyMageFlowEditClient.__new__(ComfyMageFlowEditClient)
    client.unet_name = 'mage_flow_edit_int8_convrot.safetensors'
    client.clip_name = 'qwen3vl_4b_bf16.safetensors'
    client.vae_name = 'mage_flow_vae_bf16.safetensors'
    client.client_id = 'test-client'
    return client


def test_mage_config_surface_and_round_trip():
    config = RunConfig(model_version='mage_flow_edit')

    assert isinstance(config.mage, MageFlowEditConfig)
    assert config.mage.steps == 30
    assert config.mage.guidance_scale == 5.0
    assert config.mage.comfy_unet_name == \
        'mage_flow_edit_int8_convrot.safetensors'
    assert config.mage.comfy_clip_name == 'qwen3vl_4b_bf16.safetensors'
    assert config.mage.comfy_vae_name == 'mage_flow_vae_bf16.safetensors'
    assert 'mage_flow_edit' in FIELD_CHOICES['RunConfig.model_version']
    assert config_from_dict(config_to_dict(config)) == config
    assert is_mage_flow_edit_model(config.model_version)
    assert is_edit_model(config.model_version)


def test_mage_turbo_model_preset():
    config = config_from_dict({'model_version': 'mage_flow_edit_turbo'})

    assert config.mage.comfy_unet_name == \
        'mage_flow_edit_turbo_int8_convrot.safetensors'
    assert config.mage.steps == 4
    assert config.mage.guidance_scale == 1.0
    assert is_mage_flow_edit_model(config.model_version)


def test_mage_generic_edit_dispatch_forwards_negative_prompt():
    calls = {}

    class Backend:
        def render(self, **kwargs):
            calls.update(kwargs)
            return Image.new('RGB', (8, 8))

    config = RunConfig(model_version='mage_flow_edit')
    render_edit_frame(
        Backend(), config, Image.new('RGB', (8, 8)), 'positive', 'negative',
        1, 8, 8, reference_images=[Image.new('RGB', (8, 8))])

    assert calls['prompt'] == 'positive'
    assert calls['negative_prompt'] == 'negative'
    assert len(calls['reference_images']) == 1


def test_mage_graph_uses_native_multi_image_conditioner():
    graph = _client_without_network()._build_prompt(
        ['content.png', 'style.png', 'detail.png'],
        'apply the style', 'blurred', seed=42, width=768, height=512,
        steps=30, guidance_scale=5.0, sampler='euler',
        scheduler='simple')

    assert graph['1']['inputs']['unet_name'] == \
        'mage_flow_edit_int8_convrot.safetensors'
    assert graph['2']['inputs']['type'] == 'mage'
    assert graph['30']['class_type'] == 'TextEncodeMageFlowEdit'
    assert graph['30']['inputs']['prompt'] == 'apply the style'
    assert graph['30']['inputs']['negative_prompt'] == 'blurred'
    assert graph['30']['inputs']['images.image_1'] == ['10', 0]
    assert graph['30']['inputs']['images.image_2'] == ['11', 0]
    assert graph['30']['inputs']['images.image_3'] == ['12', 0]
    assert graph['30']['inputs']['width'] == 768
    assert graph['30']['inputs']['height'] == 512
    assert graph['35']['inputs']['positive'] == ['30', 0]
    assert graph['35']['inputs']['negative'] == ['30', 1]
    assert graph['35']['inputs']['latent_image'] == ['30', 2]
    assert graph['35']['inputs']['steps'] == 30
    assert graph['35']['inputs']['cfg'] == 5.0


def test_mage_graph_caps_native_autogrow_references_at_sixteen():
    graph = _client_without_network()._build_prompt(
        [f'{index}.png' for index in range(20)],
        'edit', '', seed=1, width=512, height=512, steps=4,
        guidance_scale=1.0, sampler='euler', scheduler='simple')

    assert graph['30']['inputs']['images.image_16'] == ['25', 0]
    assert 'images.image_17' not in graph['30']['inputs']


def test_mage_render_adds_role_instruction_and_debug_images(tmp_path):
    client = _client_without_network()
    uploaded = []
    submitted = {}
    client._upload_image = lambda image, slot: (
        uploaded.append((image.getpixel((0, 0)), slot)) or f'ref-{slot}.png')
    client._request = lambda method, path, **kwargs: (
        submitted.update(kwargs.get('json', {}))
        or SimpleNamespace(json=lambda: {'prompt_id': 'p1'}))
    expected = Image.new('RGB', (8, 8), 'blue')
    client._wait_for_output = lambda prompt_id: expected
    references = [
        Image.new('RGB', (8, 8), color)
        for color in ('green', 'red', 'white')
    ]

    result = client.render(
        RunConfig(model_version='mage_flow_edit'),
        references[0], 'user prompt', 'negative', 1, 8, 8,
        reference_images=references, debug_dir=str(tmp_path), frame_num=7)

    assert len(uploaded) == 3
    assert submitted['prompt']['30']['inputs']['prompt'].startswith(
        'Use @Image1')
    assert submitted['prompt']['30']['inputs']['negative_prompt'] == 'negative'
    assert result is expected
    assert (tmp_path / 'mage_reference_3_000007.png').is_file()
