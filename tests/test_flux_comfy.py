"""ComfyUI FLUX graph construction and adapter behavior without a live server."""

from types import SimpleNamespace

from PIL import Image

from vibewarp.config import RunConfig
from vibewarp.core.flux_comfy import ComfyFluxClient


def _client_without_network():
    client = ComfyFluxClient.__new__(ComfyFluxClient)
    client.unet_name = 'klein.safetensors'
    client.clip_name = 'qwen.safetensors'
    client.vae_name = 'flux2-vae.safetensors'
    client.client_id = 'test-client'
    return client


def test_comfy_graph_matches_distilled_reference_workflow():
    graph = _client_without_network()._build_prompt(
        ['first.png'], 'make it painted', seed=42,
        width=768, height=512, steps=4, guidance_scale=1.0)

    assert graph['1']['class_type'] == 'UNETLoader'
    assert graph['2']['inputs']['type'] == 'flux2'
    assert graph['12']['class_type'] == 'ReferenceLatent'
    assert graph['13']['class_type'] == 'ReferenceLatent'
    assert graph['30']['class_type'] == 'EmptyFlux2LatentImage'
    assert graph['32']['inputs']['sampler_name'] == 'euler'
    assert graph['33']['inputs'] == {'steps': 4, 'width': 768, 'height': 512}
    assert graph['34']['inputs']['cfg'] == 1.0
    assert graph['35']['inputs']['latent_image'] == ['30', 0]
    assert graph['36']['inputs']['samples'] == ['35', 0]


def test_comfy_graph_chains_multiple_references_on_both_conditionings():
    graph = _client_without_network()._build_prompt(
        ['warped.png', 'raw.png'], 'p', seed=0,
        width=512, height=512, steps=4, guidance_scale=1.0)

    assert graph['16']['inputs']['conditioning'] == ['12', 0]
    assert graph['17']['inputs']['conditioning'] == ['13', 0]
    assert graph['34']['inputs']['positive'] == ['16', 0]
    assert graph['34']['inputs']['negative'] == ['17', 0]


def test_comfy_render_preserves_arbitrary_reference_order(monkeypatch):
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
        RunConfig(model_version='flux2_klein_edit'),
        edit_image=references[0],
        reference_images=references,
        prompt='p', seed=1, width=8, height=8,
    )

    assert [pixel for pixel, _ in uploaded] == [
        (0, 128, 0), (255, 0, 0), (255, 255, 255), (0, 0, 255)]
    assert submitted['prompt']['34']['inputs']['positive'] == ['24', 0]
    assert submitted['prompt']['4']['inputs']['text'].startswith(
        'Use reference image 1')
    assert result is expected


def test_comfy_render_can_use_one_reference(monkeypatch):
    client = _client_without_network()
    uploaded = []
    submitted = {}
    client._upload_image = lambda image, slot: (
        uploaded.append(image.getpixel((0, 0))) or f'ref-{slot}.png')
    client._request = lambda method, path, **kwargs: (
        submitted.update(kwargs.get('json', {}))
        or SimpleNamespace(json=lambda: {'prompt_id': 'p1'}))
    client._wait_for_output = lambda prompt_id: Image.new('RGB', (8, 8))

    image = Image.new('RGB', (8, 8), 'green')
    client.render(
        RunConfig(model_version='flux2_klein_edit'), image, 'user prompt', 1, 8, 8,
        reference_images=[image])

    assert uploaded == [(0, 128, 0)]
    assert submitted['prompt']['4']['inputs']['text'] == 'user prompt'


def test_comfy_debug_images_match_passed_references(tmp_path):
    client = _client_without_network()
    uploaded = []
    client._upload_image = lambda image, slot: (
        uploaded.append(image.copy()) or f'ref-{slot}.png')
    client._request = lambda *args, **kwargs: SimpleNamespace(
        json=lambda: {'prompt_id': 'p1'})
    client._wait_for_output = lambda prompt_id: Image.new('RGB', (8, 8))
    references = [
        Image.new('RGB', (32, 24), 'green'),
        Image.new('RGB', (32, 24), 'red'),
    ]

    client.render(
        RunConfig(model_version='flux2_klein_edit'),
        references[0], 'p', 1, 32, 24,
        reference_images=references, debug_dir=str(tmp_path), frame_num=7)

    for index, image in enumerate(uploaded, start=1):
        saved = Image.open(
            tmp_path / f'flux_reference_{index}_000007.png').convert('RGB')
        assert list(saved.getdata()) == list(image.getdata())
