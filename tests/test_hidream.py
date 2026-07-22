"""HiDream-O1 config, native Comfy graph, and generalized edit dispatch."""

from types import SimpleNamespace

from PIL import Image

from vibewarp.config import HiDreamConfig, RunConfig
from vibewarp.config_io import FIELD_CHOICES, config_from_dict, config_to_dict
from vibewarp.core.hidream_comfy import (ComfyHiDreamClient,
                                         add_style_reference_label,
                                         nearest_trained_resolution)
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
    assert config.hidream.reference_mode == 'raw + warped'
    assert config.hidream.use_unwarped_style_reference is False
    assert config.hidream.label_raw_reference is False
    assert config.hidream.label_style_reference is False
    assert 'reference image 1' in config.hidream.multi_reference_instruction
    assert 'hidream_o1_edit' in FIELD_CHOICES['RunConfig.model_version']
    assert 'hidream' in _VIBEWARP_SECTIONS

    config.hidream.steps = 32
    loaded = config_from_dict(config_to_dict(config))
    assert loaded.hidream == config.hidream
    assert settings_to_dict(config)['hidream_steps'] == 32


def test_hidream_migrates_previous_raw_reference_boolean():
    loaded = config_from_dict({
        'model_version': 'hidream_o1_edit',
        'hidream': {'feed_raw_frame_as_reference': False},
    })
    assert loaded.hidream.reference_mode == 'warped only'

    loaded = config_from_dict({
        'hidream': {
            'feed_raw_frame_as_reference': False,
            'reference_mode': 'raw only',
        },
    })
    assert loaded.hidream.reference_mode == 'raw only'


def test_hidream_fields_are_classified():
    from vibewarp.ui_layout import classify

    for field in (
        'comfy_server_url', 'comfy_checkpoint_name', 'comfy_timeout', 'steps',
        'guidance_scale', 'sampler', 'scheduler', 'noise_scale', 'fixed_seed',
        'use_trained_resolution', 'reference_mode', 'label_raw_reference',
        'label_style_reference',
        'use_unwarped_style_reference',
        'multi_reference_instruction',
        'patch_seam_smoothing', 'patch_seam_start', 'patch_seam_passes',
        'patch_seam_blend',
    ):
        assert classify('hidream', field) is not None, field
    assert classify('hidream', 'comfy_checkpoint_name')[0] == 'system'


def test_native_hidream_graph_uses_exactly_one_edit_reference():
    graph = _client_without_network()._build_prompt(
        'warped.png', 'paint it', 'blurry', seed=42,
        width=768, height=512, steps=40, guidance_scale=5.0,
        sampler='dpmpp_2m_sde_gpu', scheduler='normal')

    assert graph['1']['class_type'] == 'CheckpointLoaderSimple'
    assert graph['5']['class_type'] == 'HiDreamO1ReferenceImages'
    assert graph['5']['inputs']['images.image_1'] == ['4', 0]
    assert 'images' not in graph['5']['inputs']
    assert graph['6']['class_type'] == 'EmptyHiDreamO1LatentImage'
    assert graph['6']['inputs']['width'] == 768
    assert graph['7']['class_type'] == 'ModelNoiseScale'
    assert graph['7']['inputs']['noise_scale'] == 8.0
    assert graph['11']['class_type'] == 'BasicScheduler'
    assert graph['11']['inputs']['model'] == ['7', 0]
    assert graph['11']['inputs']['scheduler'] == 'normal'
    assert graph['11']['inputs']['steps'] == 40
    assert graph['12']['class_type'] == 'KSamplerSelect'
    assert graph['12']['inputs']['sampler_name'] == 'dpmpp_2m_sde_gpu'
    assert graph['13']['class_type'] == 'SamplerCustom'
    assert graph['13']['inputs']['positive'] == ['5', 0]
    assert graph['13']['inputs']['negative'] == ['5', 1]
    assert graph['13']['inputs']['sigmas'] == ['11', 0]
    assert graph['13']['inputs']['latent_image'] == ['6', 0]
    assert graph['8']['inputs']['vae'] == ['1', 2]
    assert graph['8']['inputs']['samples'] == ['13', 0]
    assert graph['10']['class_type'] == 'HiDreamO1PatchSeamSmoothing'
    assert graph['10']['inputs']['start_percent'] == 0.8
    assert graph['10']['inputs']['model'] == ['7', 0]
    assert graph['10']['inputs']['passes'] == 'ramp_2_4'
    assert graph['10']['inputs']['blend'] == 'median'
    assert graph['13']['inputs']['model'] == ['10', 0]


def test_native_hidream_graph_accepts_ordered_second_reference():
    graph = _client_without_network()._build_prompt(
        'content.png', 'paint it', 'blurry', seed=42,
        width=2048, height=2048, steps=40, guidance_scale=5.0,
        sampler='dpmpp_2m_sde_gpu', scheduler='normal',
        second_image_name='style.png')

    assert graph['4']['inputs']['image'] == 'content.png'
    assert graph['14']['class_type'] == 'LoadImage'
    assert graph['14']['inputs']['image'] == 'style.png'
    assert graph['5']['inputs']['images.image_1'] == ['4', 0]
    assert graph['5']['inputs']['images.image_2'] == ['14', 0]


def test_hidream_render_uploads_only_the_prepared_edit_target(tmp_path):
    client = _client_without_network()
    uploaded = []
    submitted = {}
    client._upload_image = lambda image, filename='edit_target.png': (
        uploaded.append(image.getpixel((0, 0))) or 'edit.png')
    client._request = lambda method, path, **kwargs: (
        submitted.update(kwargs.get('json', {}))
        or SimpleNamespace(json=lambda: {'prompt_id': 'p1'}))
    expected = Image.new('RGB', (8, 8), 'blue')
    client._wait_for_output = lambda prompt_id: expected

    result = client.render(
        RunConfig(model_version='hidream_o1_edit'),
        edit_image=Image.new('RGB', (8, 8), 'red'),
        prompt='p', negative_prompt='n', seed=1, width=8, height=8,
        debug_dir=str(tmp_path), frame_num=0)

    assert uploaded == [(255, 0, 0)]
    assert submitted['prompt']['5']['inputs']['images.image_1'] == ['4', 0]
    assert submitted['prompt']['6']['inputs']['width'] == 2048
    assert submitted['prompt']['6']['inputs']['height'] == 2048
    assert Image.open(tmp_path / 'hidream_reference_1_000000.png').size == \
        (2048, 2048)
    assert result is expected


def test_hidream_render_assigns_content_and_style_reference_roles():
    client = _client_without_network()
    uploaded = []
    submitted = {}
    client._upload_image = lambda image, filename='edit_target.png': (
        uploaded.append((filename, image.getpixel((0, 0)))) or filename)
    client._request = lambda method, path, **kwargs: (
        submitted.update(kwargs.get('json', {}))
        or SimpleNamespace(json=lambda: {'prompt_id': 'p1'}))
    client._wait_for_output = lambda prompt_id: Image.new('RGB', (8, 8))
    config = RunConfig(model_version='hidream_o1_edit')
    config.hidream.use_trained_resolution = False

    client.render(
        config, Image.new('RGB', (8, 8), 'red'), 'user style prompt', 'n',
        1, 8, 8, context_image=Image.new('RGB', (8, 8), 'green'))

    assert uploaded == [
        ('content_frame.png', (0, 128, 0)),
        ('style_reference.png', (255, 0, 0)),
    ]
    graph = submitted['prompt']
    assert graph['5']['inputs']['images.image_1'] == ['4', 0]
    assert graph['5']['inputs']['images.image_2'] == ['14', 0]
    assert graph['2']['inputs']['text'].endswith('\n\nuser style prompt')
    assert graph['2']['inputs']['text'].startswith('Transform reference image 1')


def test_hidream_raw_and_style_labels_match_uploaded_references(tmp_path):
    client = _client_without_network()
    uploaded = []
    client._upload_image = lambda image, filename='edit_target.png': (
        uploaded.append((filename, image.copy())) or filename)
    client._request = lambda method, path, **kwargs: SimpleNamespace(
        json=lambda: {'prompt_id': 'p1'})
    client._wait_for_output = lambda prompt_id: Image.new('RGB', (8, 8))
    config = RunConfig(model_version='hidream_o1_edit')
    config.hidream.use_trained_resolution = False
    config.hidream.label_raw_reference = True
    config.hidream.label_style_reference = True

    client.render(
        config, Image.new('RGB', (320, 240), 'red'), 'p', 'n', 1, 320, 240,
        context_image=Image.new('RGB', (320, 240), 'green'),
        debug_dir=str(tmp_path), frame_num=7)

    content = uploaded[0][1]
    style = uploaded[1][1]
    assert uploaded[0][0] == 'content_frame_000007.png'
    assert uploaded[1][0] == 'style_reference_000007.png'
    assert content.getpixel((160, 230)) == (0, 0, 0)
    assert content.getpixel((0, 0)) == (0, 128, 0)
    assert style.getpixel((0, 230)) == (0, 0, 0)
    assert style.getpixel((0, 0)) == (255, 0, 0)
    assert style.size == content.size == (320, 240)
    saved_content = Image.open(tmp_path / 'hidream_reference_1_000007.png')
    saved_style = Image.open(tmp_path / 'hidream_reference_2_000007.png')
    assert list(saved_content.getdata()) == list(content.getdata())
    assert list(saved_style.getdata()) == list(style.getdata())


def test_hidream_can_label_first_raw_edit_target(tmp_path):
    client = _client_without_network()
    uploaded = []
    client._upload_image = lambda image, filename='edit_target.png': (
        uploaded.append(image.copy()) or filename)
    client._request = lambda *args, **kwargs: SimpleNamespace(
        json=lambda: {'prompt_id': 'p1'})
    client._wait_for_output = lambda prompt_id: Image.new('RGB', (8, 8))
    config = RunConfig(model_version='hidream_o1_edit')
    config.hidream.use_trained_resolution = False
    config.hidream.label_raw_reference = True

    client.render(
        config, Image.new('RGB', (320, 240), 'green'), 'p', 'n', 1, 320, 240,
        debug_dir=str(tmp_path), frame_num=0)

    assert len(uploaded) == 1
    assert uploaded[0].getpixel((0, 0)) == (0, 128, 0)
    assert uploaded[0].getpixel((160, 230)) == (0, 0, 0)
    assert list(Image.open(tmp_path / 'hidream_reference_1_000000.png').getdata()) == \
        list(uploaded[0].getdata())


def test_style_reference_label_keeps_dimensions():
    labeled = add_style_reference_label(Image.new('RGB', (128, 96), 'blue'))
    assert labeled.size == (128, 96)
    assert labeled.getpixel((0, 0)) == (0, 0, 255)
    assert labeled.getpixel((0, 95)) == (0, 0, 0)


def test_hidream_render_can_use_raw_frame_as_only_reference():
    client = _client_without_network()
    uploaded = []
    submitted = {}
    client._upload_image = lambda image, filename='edit_target.png': (
        uploaded.append((filename, image.getpixel((0, 0)))) or filename)
    client._request = lambda method, path, **kwargs: (
        submitted.update(kwargs.get('json', {}))
        or SimpleNamespace(json=lambda: {'prompt_id': 'p1'}))
    client._wait_for_output = lambda prompt_id: Image.new('RGB', (8, 8))
    config = RunConfig(model_version='hidream_o1_edit')
    config.hidream.use_trained_resolution = False
    config.hidream.reference_mode = 'raw only'

    client.render(
        config, Image.new('RGB', (8, 8), 'red'), 'user prompt', 'n', 1, 8, 8,
        context_image=Image.new('RGB', (8, 8), 'green'))

    assert uploaded == [('content_frame.png', (0, 128, 0))]
    graph = submitted['prompt']
    assert graph['5']['inputs']['images.image_1'] == ['4', 0]
    assert 'images.image_2' not in graph['5']['inputs']
    assert graph['2']['inputs']['text'] == 'user prompt'


def test_hidream_render_can_use_warped_frame_as_only_reference():
    client = _client_without_network()
    uploaded = []
    submitted = {}
    client._upload_image = lambda image, filename='edit_target.png': (
        uploaded.append((filename, image.getpixel((0, 0)))) or filename)
    client._request = lambda method, path, **kwargs: (
        submitted.update(kwargs.get('json', {}))
        or SimpleNamespace(json=lambda: {'prompt_id': 'p1'}))
    client._wait_for_output = lambda prompt_id: Image.new('RGB', (8, 8))
    config = RunConfig(model_version='hidream_o1_edit')
    config.hidream.use_trained_resolution = False
    config.hidream.reference_mode = 'warped only'

    client.render(
        config, Image.new('RGB', (8, 8), 'red'), 'user prompt', 'n', 1, 8, 8,
        context_image=Image.new('RGB', (8, 8), 'green'))

    assert uploaded == [('edit_target.png', (255, 0, 0))]
    assert 'images.image_2' not in submitted['prompt']['5']['inputs']


def test_hidream_trained_resolution_tracks_output_aspect_ratio():
    assert nearest_trained_resolution(1024, 1024) == (2048, 2048)
    assert nearest_trained_resolution(1920, 1080) == (2560, 1440)
    assert nearest_trained_resolution(1080, 1920) == (1440, 2560)


def test_hidream_can_explicitly_use_output_resolution():
    client = _client_without_network()
    uploaded_sizes = []
    submitted = {}
    client._upload_image = lambda image, filename='edit_target.png': (
        uploaded_sizes.append(image.size) or 'edit.png')
    client._request = lambda method, path, **kwargs: (
        submitted.update(kwargs.get('json', {}))
        or SimpleNamespace(json=lambda: {'prompt_id': 'p1'}))
    client._wait_for_output = lambda prompt_id: Image.new('RGB', (8, 8))
    config = RunConfig(model_version='hidream_o1_edit')
    config.hidream.use_trained_resolution = False

    client.render(
        config, Image.new('RGB', (640, 480)), 'p', 'n', 1, 640, 480)

    assert uploaded_sizes == [(640, 480)]
    assert submitted['prompt']['6']['inputs']['width'] == 640
    assert submitted['prompt']['6']['inputs']['height'] == 480


def test_hidream_frame_downsamples_native_result_to_video_size(tmp_path):
    from vibewarp.core.diffusion import FrameState, RenderContext, render_frame_edit

    init_path = tmp_path / 'frame.png'
    Image.new('RGB', (32, 32), 'red').save(init_path)

    class Backend:
        def render(self, **kwargs):
            return Image.new('RGB', (2048, 2048), 'blue')

    config = RunConfig(model_version='hidream_o1_edit')
    config.video.width = config.video.height = 32
    state = FrameState(frame_num=0, init_image=str(init_path))

    result = render_frame_edit(
        RenderContext(config=config, edit_backend=Backend()), state)

    assert result.size == (32, 32)
    assert state.prev_frame.size == (32, 32)


def test_generic_hidream_dispatch_passes_raw_but_drops_flux_only_references():
    from vibewarp.core.edit import render_edit_frame

    class Backend:
        def render(self, **kwargs):
            self.received = kwargs
            return kwargs['edit_image']

    backend = Backend()
    edit = Image.new('RGB', (8, 8), 'red')
    render_edit_frame(
        backend, RunConfig(model_version='hidream_o1_edit'), edit, 'p', 'n',
        1, 8, 8,
        context_image=Image.new('RGB', (8, 8), 'green'),
        prev_context_image=Image.new('RGB', (8, 8), 'blue'),
        mask_context_image=Image.new('RGB', (8, 8), 'white'))

    assert backend.received['edit_image'] is edit
    assert backend.received['context_image'].getpixel((0, 0)) == (0, 128, 0)
    assert 'prev_context_image' not in backend.received
    assert 'mask_context_image' not in backend.received


def test_hidream_frame_loop_supplies_raw_current_frame_as_content(tmp_path):
    from vibewarp.core.diffusion import FrameState, RenderContext, render_frame_edit

    frames = tmp_path / 'frames'
    frames.mkdir()
    raw_path = frames / '000002.jpg'
    Image.new('RGB', (16, 16), 'green').save(raw_path)
    edit_path = tmp_path / 'warped.png'
    Image.new('RGB', (16, 16), 'red').save(edit_path)

    class Backend:
        def render(self, **kwargs):
            self.received = kwargs
            return kwargs['edit_image']

    backend = Backend()
    config = RunConfig(model_version='hidream_o1_edit')
    config.video.width = config.video.height = 16
    state = FrameState(frame_num=1, init_image=str(edit_path))
    render_frame_edit(RenderContext(
        config=config, edit_backend=backend,
        video_frames_folder=str(frames) + '/', batch_folder=str(tmp_path)), state)

    assert backend.received['edit_image'].getpixel((0, 0)) == (255, 0, 0)
    raw = backend.received['context_image'].getpixel((0, 0))
    assert raw[1] > raw[0] and raw[1] > raw[2]
    assert backend.received['debug_dir'] == str(tmp_path / 'debug')
    assert backend.received['frame_num'] == 1


def test_hidream_can_replace_warped_style_with_unwarped_previous_frame(tmp_path):
    from vibewarp.core.diffusion import FrameState, RenderContext, render_frame_edit

    frames = tmp_path / 'frames'
    frames.mkdir()
    Image.new('RGB', (16, 16), 'green').save(frames / '000002.jpg')
    edit_path = tmp_path / 'warped.png'
    Image.new('RGB', (16, 16), 'red').save(edit_path)

    class Backend:
        def render(self, **kwargs):
            self.received = kwargs
            return Image.new('RGB', (16, 16), 'black')

    backend = Backend()
    config = RunConfig(model_version='hidream_o1_edit')
    config.video.width = config.video.height = 16
    config.hidream.use_unwarped_style_reference = True
    previous = Image.new('RGB', (16, 16), 'blue')
    state = FrameState(
        frame_num=1, init_image=str(edit_path), prev_frame=previous)

    render_frame_edit(RenderContext(
        config=config, edit_backend=backend,
        video_frames_folder=str(frames) + '/'), state)

    assert backend.received['edit_image'].getpixel((0, 0)) == (0, 0, 255)
    assert backend.received['context_image'].getpixel((0, 0))[1] > 100
    # The backend result still becomes the next temporal state.
    assert state.prev_frame.getpixel((0, 0)) == (0, 0, 0)
