"""Qwen Image Edit 2511 configuration and Comfy graph tests."""

from types import SimpleNamespace

from PIL import Image

from vibewarp.config import QwenEditConfig, RunConfig
from vibewarp.config_io import FIELD_CHOICES, config_from_dict, config_to_dict
from vibewarp.core.qwen_comfy import ComfyQwenEditClient
from vibewarp.core.edit import render_edit_frame
from vibewarp.core.model_loader import is_edit_model, is_qwen_edit_model


def _client_without_network():
    client = ComfyQwenEditClient.__new__(ComfyQwenEditClient)
    client.unet_name = 'qwen_image_edit_2511_int8_convrot.safetensors'
    client.clip_name = 'qwen_2.5_vl_7b_fp8_scaled.safetensors'
    client.vae_name = 'qwen_image_vae.safetensors'
    client.lora_name = (
        'Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors')
    client.client_id = 'test-client'
    return client


def test_qwen_config_surface_and_round_trip():
    config = RunConfig(model_version='qwen_image_edit_2511')
    assert isinstance(config.qwen, QwenEditConfig)
    assert config.qwen.steps == 8
    assert config.qwen.guidance_scale == 1.0
    assert config.qwen.use_lightning_lora is True
    assert config.qwen.lora_strength == 1.0
    assert 'Lightning-8steps' in config.qwen.comfy_lora_name
    assert config.qwen.sampling_shift == 3.1
    assert config.qwen.comfy_unet_name.endswith('int8_convrot.safetensors')
    assert config.qwen.comfy_vae_name == 'qwen_image_vae.safetensors'
    assert 'qwen_image_edit_2511' in FIELD_CHOICES['RunConfig.model_version']
    assert config_from_dict(config_to_dict(config)) == config
    assert is_qwen_edit_model(config.model_version)
    assert is_edit_model(config.model_version)


def test_qwen_gguf_model_preset():
    config = config_from_dict({'model_version': 'qwen_image_edit_2511_gguf'})

    assert config.qwen.comfy_unet_name == \
        'Qwen-Image-Edit-2511-Q5_K_M.gguf'
    assert config.qwen.comfy_clip_name == \
        'qwen_2.5_vl_7b_fp8_scaled.safetensors'
    assert 'qwen_image_edit_2511_gguf' in \
        FIELD_CHOICES['RunConfig.model_version']
    assert is_qwen_edit_model(config.model_version)


def test_qwen_generic_edit_dispatch_forwards_negative_prompt():
    calls = {}

    class Backend:
        def render(self, **kwargs):
            calls.update(kwargs)
            return Image.new('RGB', (8, 8))

    config = RunConfig(model_version='qwen_image_edit_2511')
    render_edit_frame(
        Backend(), config, Image.new('RGB', (8, 8)), 'positive', 'negative',
        1, 8, 8, reference_images=[Image.new('RGB', (8, 8))])

    assert calls['prompt'] == 'positive'
    assert calls['negative_prompt'] == 'negative'
    assert len(calls['reference_images']) == 1


def test_qwen_graph_matches_native_three_image_workflow():
    graph = _client_without_network()._build_prompt(
        ['content.png', 'style.png', 'detail.png', 'ignored.png'],
        'apply the style', 'blurred', seed=42, width=768, height=512,
        steps=8, guidance_scale=1.0, sampler='euler',
        scheduler='simple', sampling_shift=3.1,
        use_lightning_lora=True, lora_strength=1.0)

    assert graph['1']['inputs']['unet_name'].endswith('int8_convrot.safetensors')
    assert graph['2']['inputs']['type'] == 'qwen_image'
    assert graph['20']['inputs']['image'] == ['10', 0]
    assert graph['21']['inputs']['pixels'] == ['20', 0]
    assert graph['30']['inputs']['prompt'] == 'apply the style'
    assert graph['31']['inputs']['prompt'] == 'blurred'
    for node_id in ('30', '31'):
        assert graph[node_id]['inputs']['image1'] == ['20', 0]
        assert graph[node_id]['inputs']['image2'] == ['11', 0]
        assert graph[node_id]['inputs']['image3'] == ['12', 0]
        assert 'image4' not in graph[node_id]['inputs']
    assert graph['32']['inputs']['reference_latents_method'] == \
        'index_timestep_zero'
    assert graph['34']['inputs']['shift'] == 3.1
    assert graph['38']['inputs'] == {
        'model': ['34', 0], 'strength': 1.0, 'pre_cfg': False}
    assert graph['39']['inputs'] == {
        'model': ['38', 0],
        'lora_name':
            'Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors',
        'strength_model': 1.0,
    }
    assert graph['35']['inputs']['model'] == ['39', 0]
    assert graph['35']['inputs']['latent_image'] == ['21', 0]
    assert graph['35']['inputs']['steps'] == 8
    assert graph['35']['inputs']['cfg'] == 1.0


def test_qwen_graph_can_disable_lightning_lora():
    graph = _client_without_network()._build_prompt(
        ['content.png'], 'edit', '', seed=1, width=512, height=512,
        steps=40, guidance_scale=4.0, sampler='euler',
        scheduler='simple', sampling_shift=3.1,
        use_lightning_lora=False, lora_strength=1.0)

    assert '39' not in graph
    assert graph['35']['inputs']['model'] == ['38', 0]


def test_qwen_gguf_graph_uses_custom_loader_without_dtype_input():
    client = _client_without_network()
    client.unet_name = 'Qwen-Image-Edit-2511-Q5_K_M.gguf'

    graph = client._build_prompt(
        ['content.png'], 'edit', '', seed=1, width=512, height=512,
        steps=8, guidance_scale=1.0, sampler='euler',
        scheduler='simple', sampling_shift=3.1,
        use_lightning_lora=True, lora_strength=1.0)

    assert graph['1'] == {
        'class_type': 'UnetLoaderGGUF',
        'inputs': {'unet_name': 'Qwen-Image-Edit-2511-Q5_K_M.gguf'},
    }


def test_qwen_render_caps_references_and_adds_role_instruction(tmp_path):
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
        for color in ('green', 'red', 'white', 'blue')
    ]

    result = client.render(
        RunConfig(model_version='qwen_image_edit_2511'),
        references[0], 'user prompt', 'negative', 1, 8, 8,
        reference_images=references, debug_dir=str(tmp_path), frame_num=7)

    assert len(uploaded) == 3
    assert submitted['prompt']['30']['inputs']['prompt'].startswith(
        'Preserve the content')
    assert submitted['prompt']['31']['inputs']['prompt'] == 'negative'
    assert result is expected
    assert (tmp_path / 'qwen_reference_3_000007.png').is_file()
