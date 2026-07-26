"""HiDream-O1 config, native Comfy graph, and generalized edit dispatch."""

from types import SimpleNamespace

from PIL import Image

from vibewarp.config import EditReferenceConfig, HiDreamConfig, RunConfig
from vibewarp.config_io import FIELD_CHOICES, config_from_dict, config_to_dict
from vibewarp.core.hidream_comfy import (
    ComfyHiDreamClient, nearest_trained_resolution)
from vibewarp.settings import _VIBEWARP_SECTIONS, settings_to_dict


def _client_without_network():
    client = ComfyHiDreamClient.__new__(ComfyHiDreamClient)
    client.checkpoint_name = 'hidream_o1_image_fp8_scaled.safetensors'
    client.client_id = 'test-client'
    return client


def test_hidream_config_surface_and_round_trip():
    config = RunConfig(model_version='hidream_o1_edit')
    assert isinstance(config.hidream, HiDreamConfig)
    assert config.hidream.steps == 40
    assert config.hidream.guidance_scale == 5.0
    assert config.hidream.sampler == 'dpmpp_2m_sde_gpu'
    assert config.hidream.scheduler == 'normal'
    assert config.hidream.noise_scale == 8.0
    assert config.hidream.use_trained_resolution is True
    assert config.hidream.fixed_seed is True
    assert [ref.source for ref in config.hidream.references] == ['raw', 'none']
    assert 'reference image 1' in config.hidream.multi_reference_instruction
    assert 'hidream_o1_edit' in FIELD_CHOICES['RunConfig.model_version']
    assert 'hidream' in _VIBEWARP_SECTIONS

    config.hidream.steps = 32
    config.hidream.references = [
        EditReferenceConfig('raw', True),
        EditReferenceConfig('upload', image_path='style.png'),
        EditReferenceConfig('none'),
    ]
    loaded = config_from_dict(config_to_dict(config))
    assert loaded.hidream == config.hidream
    assert settings_to_dict(config)['hidream_steps'] == 32


def test_hidream_migrates_legacy_reference_settings():
    loaded = config_from_dict({
        'model_version': 'hidream_o1_edit',
        'hidream': {'feed_raw_frame_as_reference': False},
    })
    assert [ref.source for ref in loaded.hidream.references] == ['warped', 'none']

    loaded = config_from_dict({
        'hidream': {
            'feed_raw_frame_as_reference': False,
            'reference_mode': 'raw only',
        },
    })
    assert [ref.source for ref in loaded.hidream.references] == ['raw', 'none']


def test_hidream_fields_are_classified():
    from vibewarp.ui_layout import classify

    for field in (
        'comfy_server_url', 'comfy_checkpoint_name', 'comfy_timeout', 'steps',
        'guidance_scale', 'sampler', 'scheduler', 'noise_scale', 'fixed_seed',
        'use_trained_resolution', 'references', 'multi_reference_instruction',
        'patch_seam_smoothing', 'patch_seam_start', 'patch_seam_passes',
        'patch_seam_blend',
    ):
        assert classify('hidream', field) is not None, field
    assert classify('hidream', 'comfy_checkpoint_name')[0] == 'system'


def test_native_hidream_graph_uses_exactly_one_edit_reference():
    graph = _client_without_network()._build_prompt(
        ['warped.png'], 'paint it', 'blurry', seed=42,
        width=768, height=512, steps=40, guidance_scale=5.0,
        sampler='dpmpp_2m_sde_gpu', scheduler='normal')

    assert graph['1']['class_type'] == 'CheckpointLoaderSimple'
    assert graph['4']['inputs']['image'] == 'warped.png'
    assert graph['5']['inputs']['images.image_1'] == ['4', 0]
    assert 'images.image_2' not in graph['5']['inputs']
    assert graph['6']['inputs']['width'] == 768
    assert graph['7']['inputs']['noise_scale'] == 8.0
    assert graph['11']['inputs']['scheduler'] == 'normal'
    assert graph['12']['inputs']['sampler_name'] == 'dpmpp_2m_sde_gpu'
    assert graph['13']['inputs']['positive'] == ['5', 0]
    assert graph['13']['inputs']['model'] == ['10', 0]
    assert graph['10']['class_type'] == 'HiDreamO1PatchSeamSmoothing'


def test_native_hidream_graph_accepts_arbitrary_ordered_references():
    graph = _client_without_network()._build_prompt(
        ['content.png', 'style.png', 'previous.png'],
        'paint it', 'blurry', seed=42, width=2048, height=2048, steps=40,
        guidance_scale=5.0, sampler='dpmpp_2m_sde_gpu', scheduler='normal')

    assert graph['4']['inputs']['image'] == 'content.png'
    assert graph['14']['inputs']['image'] == 'style.png'
    assert graph['15']['inputs']['image'] == 'previous.png'
    assert graph['5']['inputs']['images.image_1'] == ['4', 0]
    assert graph['5']['inputs']['images.image_2'] == ['14', 0]
    assert graph['5']['inputs']['images.image_3'] == ['15', 0]


def test_hidream_render_preserves_references_and_debug_images(tmp_path):
    client = _client_without_network()
    uploaded = []
    submitted = {}
    client._upload_image = lambda image, filename='edit_target.png': (
        uploaded.append((filename, image.copy())) or filename)
    client._request = lambda method, path, **kwargs: (
        submitted.update(kwargs.get('json', {}))
        or SimpleNamespace(json=lambda: {'prompt_id': 'p1'}))
    expected = Image.new('RGB', (8, 8), 'blue')
    client._wait_for_output = lambda prompt_id: expected
    config = RunConfig(model_version='hidream_o1_edit')
    config.hidream.use_trained_resolution = False
    references = [
        Image.new('RGB', (8, 8), 'green'),
        Image.new('RGB', (8, 8), 'red'),
    ]

    result = client.render(
        config, references[0], 'user style prompt', 'n', 1, 8, 8,
        reference_images=references, debug_dir=str(tmp_path), frame_num=7)

    assert [name for name, _ in uploaded] == [
        'reference_1_000007.png', 'reference_2_000007.png']
    assert [image.getpixel((0, 0)) for _, image in uploaded] == [
        (0, 128, 0), (255, 0, 0)]
    assert submitted['prompt']['5']['inputs']['images.image_2'] == ['14', 0]
    assert submitted['prompt']['2']['inputs']['text'].startswith(
        'Use reference image 1')
    for index, (_, image) in enumerate(uploaded, start=1):
        saved = Image.open(
            tmp_path / f'hidream_reference_{index}_000007.png').convert('RGB')
        assert list(saved.getdata()) == list(image.getdata())
    assert result is expected


def test_hidream_single_reference_does_not_add_role_instruction():
    client = _client_without_network()
    submitted = {}
    client._upload_image = lambda image, filename='edit_target.png': filename
    client._request = lambda method, path, **kwargs: (
        submitted.update(kwargs.get('json', {}))
        or SimpleNamespace(json=lambda: {'prompt_id': 'p1'}))
    client._wait_for_output = lambda prompt_id: Image.new('RGB', (8, 8))
    config = RunConfig(model_version='hidream_o1_edit')
    config.hidream.use_trained_resolution = False
    image = Image.new('RGB', (8, 8), 'green')

    client.render(
        config, image, 'user prompt', 'n', 1, 8, 8,
        reference_images=[image])

    assert submitted['prompt']['2']['inputs']['text'] == 'user prompt'


def test_hidream_trained_resolution_tracks_output_aspect_ratio():
    assert nearest_trained_resolution(1280, 720) == (2560, 1440)
    assert nearest_trained_resolution(720, 1280) == (1440, 2560)
    assert nearest_trained_resolution(1024, 1024) == (2048, 2048)


def test_hidream_can_explicitly_use_output_resolution():
    client = _client_without_network()
    submitted = {}
    client._upload_image = lambda image, filename='edit_target.png': 'edit.png'
    client._request = lambda method, path, **kwargs: (
        submitted.update(kwargs.get('json', {}))
        or SimpleNamespace(json=lambda: {'prompt_id': 'p1'}))
    client._wait_for_output = lambda prompt_id: Image.new('RGB', (8, 8))
    config = RunConfig(model_version='hidream_o1_edit')
    config.hidream.use_trained_resolution = False

    client.render(
        config, Image.new('RGB', (640, 360)), 'p', 'n', 1, 640, 360)

    assert submitted['prompt']['6']['inputs']['width'] == 640
    assert submitted['prompt']['6']['inputs']['height'] == 360


def test_generic_hidream_dispatch_forwards_unified_references():
    from vibewarp.core.edit import render_edit_frame

    class Backend:
        def render(self, **kwargs):
            self.received = kwargs
            return kwargs['edit_image']

    backend = Backend()
    config = RunConfig(model_version='hidream_o1_edit')
    image = Image.new('RGB', (8, 8))
    references = [image, Image.new('RGB', (8, 8), 'blue')]
    result = render_edit_frame(
        backend, config, image, 'p', 'n', 1, 8, 8,
        reference_images=references)

    assert backend.received['reference_images'] is references
    assert 'context_image' not in backend.received
    assert result is image


def test_hidream_frame_loop_resolves_ordered_temporal_references(tmp_path):
    from vibewarp.core.diffusion import FrameState, RenderContext, render_frame_edit

    frames = tmp_path / 'frames'
    frames.mkdir()
    Image.new('RGB', (16, 16), 'green').save(frames / '000002.jpg')
    warped_path = tmp_path / 'warped.png'
    Image.new('RGB', (16, 16), 'red').save(warped_path)

    class Backend:
        def render(self, **kwargs):
            self.received = kwargs
            return Image.new('RGB', (16, 16), 'black')

    backend = Backend()
    config = RunConfig(model_version='hidream_o1_edit')
    config.video.width = config.video.height = 16
    config.hidream.references = [
        EditReferenceConfig('raw'),
        EditReferenceConfig('previous'),
        EditReferenceConfig('warped'),
        EditReferenceConfig('none'),
    ]
    previous = Image.new('RGB', (16, 16), 'blue')
    state = FrameState(
        frame_num=1, init_image=str(warped_path), prev_frame=previous)

    render_frame_edit(RenderContext(
        config=config, edit_backend=backend,
        video_frames_folder=str(frames) + '/', batch_folder=str(tmp_path)), state)

    refs = backend.received['reference_images']
    assert refs[0].getpixel((0, 0))[1] > 100
    assert refs[1].getpixel((0, 0)) == (0, 0, 255)
    assert refs[2].getpixel((0, 0)) == (255, 0, 0)
    assert backend.received['edit_image'] is refs[0]
    assert state.prev_frame.getpixel((0, 0)) == (0, 0, 0)
