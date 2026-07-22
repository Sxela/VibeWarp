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
    client = _client_without_network()
    graph = client._build_prompt(
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
    client = _client_without_network()
    graph = client._build_prompt(
        ['warped.png', 'raw.png'], 'p', seed=0,
        width=512, height=512, steps=4, guidance_scale=1.0)

    assert graph['16']['inputs']['conditioning'] == ['12', 0]
    assert graph['17']['inputs']['conditioning'] == ['13', 0]
    assert graph['34']['inputs']['positive'] == ['16', 0]
    assert graph['34']['inputs']['negative'] == ['17', 0]


def test_comfy_render_orders_raw_then_warped_style_reference(monkeypatch):
    client = _client_without_network()
    uploaded = []
    submitted = {}

    def upload(image, slot):
        uploaded.append((image.getpixel((0, 0)), slot))
        return f'ref-{slot}.png'

    client._upload_image = upload
    client._request = lambda method, path, **kwargs: (
        submitted.update(kwargs.get('json', {}))
        or SimpleNamespace(json=lambda: {'prompt_id': 'p1'}))
    expected = Image.new('RGB', (8, 8), 'blue')
    client._wait_for_output = lambda prompt_id: expected

    config = RunConfig(model_version='flux2_klein_edit')
    result = client.render(
        config,
        edit_image=Image.new('RGB', (8, 8), 'red'),
        context_image=Image.new('RGB', (8, 8), 'green'),
        prompt='p', seed=1, width=8, height=8,
    )

    assert uploaded == [((0, 128, 0), 0), ((255, 0, 0), 1)]
    assert submitted['prompt']['34']['inputs']['positive'] == ['16', 0]
    assert submitted['prompt']['4']['inputs']['text'].startswith(
        'Transform reference image 1')
    assert result is expected


def test_comfy_render_orders_raw_warped_mask_and_previous_references(monkeypatch):
    client = _client_without_network()
    uploaded = []
    submitted = {}

    def upload(image, slot):
        uploaded.append((image.getpixel((0, 0)), slot))
        return f'ref-{slot}.png'

    client._upload_image = upload
    client._request = lambda method, path, **kwargs: (
        submitted.update(kwargs.get('json', {}))
        or SimpleNamespace(json=lambda: {'prompt_id': 'p1'}))
    client._wait_for_output = lambda prompt_id: Image.new('RGB', (8, 8))

    config = RunConfig(model_version='flux2_klein_edit')
    config.flux.feed_consistency_mask_as_context = True
    config.flux.feed_prev_frame_as_context = True
    client.render(
        config,
        edit_image=Image.new('RGB', (8, 8), 'red'),
        mask_context_image=Image.new('RGB', (8, 8), 'white'),
        prev_context_image=Image.new('RGB', (8, 8), 'blue'),
        context_image=Image.new('RGB', (8, 8), 'green'),
        prompt='p', seed=1, width=8, height=8,
    )

    assert [pixel for pixel, _ in uploaded] == [
        (0, 128, 0), (255, 0, 0), (255, 255, 255), (0, 0, 255)]
    assert submitted['prompt']['34']['inputs']['positive'] == ['24', 0]


def test_comfy_render_can_use_raw_only_reference(monkeypatch):
    client = _client_without_network()
    uploaded = []
    client._upload_image = lambda image, slot: (
        uploaded.append(image.getpixel((0, 0))) or f'ref-{slot}.png')
    client._request = lambda *args, **kwargs: SimpleNamespace(
        json=lambda: {'prompt_id': 'p1'})
    client._wait_for_output = lambda prompt_id: Image.new('RGB', (8, 8))
    config = RunConfig(model_version='flux2_klein_edit')
    config.flux.reference_mode = 'raw only'

    client.render(
        config, Image.new('RGB', (8, 8), 'red'), 'p', 1, 8, 8,
        context_image=Image.new('RGB', (8, 8), 'green'))

    assert uploaded == [(0, 128, 0)]


def test_comfy_style_label_and_debug_match_uploaded_references(tmp_path):
    client = _client_without_network()
    uploaded = []
    client._upload_image = lambda image, slot: (
        uploaded.append(image.copy()) or f'ref-{slot}.png')
    client._request = lambda *args, **kwargs: SimpleNamespace(
        json=lambda: {'prompt_id': 'p1'})
    client._wait_for_output = lambda prompt_id: Image.new('RGB', (8, 8))
    config = RunConfig(model_version='flux2_klein_edit')
    config.flux.label_raw_reference = True
    config.flux.label_style_reference = True

    client.render(
        config, Image.new('RGB', (320, 240), 'red'), 'p', 1, 320, 240,
        context_image=Image.new('RGB', (320, 240), 'green'),
        debug_dir=str(tmp_path), frame_num=7)

    # Sample the label band at its LEFT edge, not the centre: the label text is centred,
    # so an interior pixel lands on an anti-aliased glyph whose exact value is font- and
    # platform-dependent (0,0,0 with the default font, ~28,28,28 with DejaVuSans-Bold on CI).
    assert uploaded[0].getpixel((0, 230)) == (0, 0, 0)
    assert uploaded[0].getpixel((0, 0)) == (0, 128, 0)
    assert uploaded[1].getpixel((0, 230)) == (0, 0, 0)
    assert list(Image.open(tmp_path / 'flux_reference_1_000007.png').getdata()) == \
        list(uploaded[0].getdata())
    assert list(Image.open(tmp_path / 'flux_reference_2_000007.png').getdata()) == \
        list(uploaded[1].getdata())


def test_comfy_can_label_first_raw_edit_target(tmp_path):
    client = _client_without_network()
    uploaded = []
    client._upload_image = lambda image, slot: (
        uploaded.append(image.copy()) or f'ref-{slot}.png')
    client._request = lambda *args, **kwargs: SimpleNamespace(
        json=lambda: {'prompt_id': 'p1'})
    client._wait_for_output = lambda prompt_id: Image.new('RGB', (8, 8))
    config = RunConfig(model_version='flux2_klein_edit')
    config.flux.label_raw_reference = True

    client.render(
        config, Image.new('RGB', (320, 240), 'green'), 'p', 1, 320, 240,
        debug_dir=str(tmp_path), frame_num=0)

    assert len(uploaded) == 1
    assert uploaded[0].getpixel((0, 0)) == (0, 128, 0)
    assert uploaded[0].getpixel((0, 230)) == (0, 0, 0)   # band, left edge (see note above)
    assert list(Image.open(tmp_path / 'flux_reference_1_000000.png').getdata()) == \
        list(uploaded[0].getdata())
