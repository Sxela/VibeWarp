"""Flux 2 Klein Edit — config plumbing, dispatch seam, and the diffusers adapter.

The actual pipeline call needs a GPU and the real weights, so it is not exercised
here. These tests cover the parts that ARE deterministic on CPU: that the config
serializes and round-trips through every codec, that the new model version is
offered and classified in the UI, and that the diffusers adapter passes only the
keyword arguments the pipeline declares (the seam that keeps us robust to the
model's still-evolving API).
"""

import json
import sys
import types
from dataclasses import asdict

import pytest

from vibewarp.config import (
    EditReferenceConfig, FluxConfig, RunConfig, flux_model_files)
from vibewarp.config_io import (
    FIELD_CHOICES,
    config_from_dict,
    config_from_json,
    config_to_dict,
)
from vibewarp.settings import (
    _VIBEWARP_SECTIONS,
    is_vibewarp_settings,
    load_vibewarp_settings,
    settings_to_dict,
)


# ---- config surface --------------------------------------------------------

def test_flux_config_defaults():
    fx = RunConfig().flux
    assert isinstance(fx, FluxConfig)
    assert fx.backend == 'comfy'
    assert fx.steps == 4
    assert fx.guidance_scale == 1.0
    assert fx.fixed_seed is True
    assert fx.denoise == 1.0
    assert [ref.source for ref in fx.references] == ['raw', 'none']
    assert all(ref.label is False for ref in fx.references)
    assert 'reference image 1' in fx.multi_reference_instruction
    assert fx.model_repo  # a non-empty default repo id


def test_model_version_choice_is_offered():
    assert 'flux2_klein_edit' in FIELD_CHOICES['RunConfig.model_version']
    assert 'flux2_klein_9b_edit' in FIELD_CHOICES['RunConfig.model_version']
    assert FIELD_CHOICES['FluxConfig.backend'] == ['comfy', 'diffusers']


def test_flux_9b_uses_its_matching_model_and_text_encoder_defaults():
    config = RunConfig(model_version='flux2_klein_9b_edit')
    files = flux_model_files(config)
    assert files['comfy_unet_name'] == 'flux-2-klein-9b-fp8.safetensors'
    assert files['comfy_clip_name'] == 'qwen_3_8b_fp8mixed.safetensors'
    assert files['comfy_vae_name'] == 'flux2-vae.safetensors'
    assert files['model_repo'] == 'black-forest-labs/FLUX.2-klein-9B'

    config.flux.comfy_clip_name = 'custom-qwen.safetensors'
    assert flux_model_files(config)['comfy_clip_name'] == \
        'custom-qwen.safetensors'

    loaded = config_from_dict({'model_version': 'flux2_klein_9b_edit'})
    assert loaded.flux.comfy_unet_name == \
        'flux-2-klein-9b-fp8.safetensors'
    assert loaded.flux.comfy_clip_name == \
        'qwen_3_8b_fp8mixed.safetensors'

    loaded.model_version = 'flux2_klein_edit'
    assert flux_model_files(loaded)['comfy_unet_name'] == \
        'flux-2-klein-4b-fp8.safetensors'


def test_flux_json_codec_round_trip():
    c = RunConfig(model_version='flux2_klein_edit')
    c.flux.denoise = 0.7
    c.flux.model_repo = 'org/flux-klein-edit'
    c.flux.references = [EditReferenceConfig(source='warped')]
    loaded = config_from_dict(config_to_dict(c))
    assert loaded.flux == c.flux
    assert loaded.model_version == 'flux2_klein_edit'


def test_flux_json_file_round_trip(tmp_path):
    c = RunConfig(model_version='flux2_klein_edit')
    c.flux.steps = 12
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(asdict(c)), encoding='utf-8')
    assert config_from_json(path).flux.steps == 12


def test_flux_migrates_previous_raw_reference_boolean():
    loaded = config_from_dict({
        'model_version': 'flux2_klein_edit',
        'flux': {'feed_init_as_context': True},
    })
    assert [ref.source for ref in loaded.flux.references] == [
        'raw', 'warped', 'none']

    loaded = config_from_dict({
        'flux': {
            'feed_init_as_context': False,
            'reference_mode': 'raw only',
        },
    })
    assert [ref.source for ref in loaded.flux.references] == ['raw', 'none']


# ---- VibeWarp's own flat saved format --------------------------------------

def test_flux_registered_as_settings_section():
    assert 'flux' in _VIBEWARP_SECTIONS


def test_flux_flat_settings_round_trip():
    c = RunConfig(model_version='flux2_klein_edit')
    c.flux.denoise = 0.65
    c.flux.model_repo = 'my/repo'
    flat = settings_to_dict(c)
    assert flat['flux_denoise'] == 0.65
    assert flat['flux_model_repo'] == 'my/repo'
    assert is_vibewarp_settings(flat)

    nested = load_vibewarp_settings_from_dict(flat)
    assert nested['flux']['denoise'] == 0.65
    assert nested['flux']['model_repo'] == 'my/repo'
    assert nested['model_version'] == 'flux2_klein_edit'


def load_vibewarp_settings_from_dict(flat, tmp_path=None):
    """Helper: write the flat dict to a temp file and un-flatten it."""
    import os
    import tempfile
    fh = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False)
    json.dump(flat, fh)
    fh.close()
    try:
        return load_vibewarp_settings(fh.name)
    finally:
        os.unlink(fh.name)


# ---- UI layout -------------------------------------------------------------

def test_flux_fields_are_classified():
    from vibewarp.ui_layout import classify
    for field in ('denoise', 'guidance_scale', 'steps', 'references',
                  'multi_reference_instruction',
                  'max_sequence_length'):
        assert classify('flux', field) is not None, field
    # model_repo is a machine path — must live in the System tier so it does not
    # travel with a shared settings file.
    tier, _ = classify('flux', 'model_repo')
    assert tier == 'system'


# ---- diffusers adapter seam (no torch/diffusers/GPU needed) -----------------

class _FakePipe:
    """Stand-in whose __call__ declares only a subset of the candidate kwargs."""

    def __init__(self, **declared_defaults):
        self._declared = declared_defaults
        self.received = None

    def __call__(self, image=None, prompt=None, num_inference_steps=None,
                 guidance_scale=None, strength=None, image_2=None, generator=None):
        self.received = dict(
            image=image, prompt=prompt, num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, strength=strength, image_2=image_2,
        )
        out = types.SimpleNamespace(images=[image])
        return out


def _install_fake_torch(monkeypatch):
    """render_flux_frame does `import torch` for Generator + inference_mode."""
    fake = types.ModuleType('torch')

    class _Gen:
        def manual_seed(self, s):
            return self

    fake.Generator = lambda device=None: _Gen()

    class _NoCtx:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    fake.inference_mode = lambda: _NoCtx()
    monkeypatch.setitem(sys.modules, 'torch', fake)
    return fake


def test_render_flux_frame_passes_only_supported_kwargs(monkeypatch):
    from PIL import Image

    from vibewarp.core.flux import render_flux_frame

    _install_fake_torch(monkeypatch)

    pipe = _FakePipe()
    cfg = RunConfig(model_version='flux2_klein_edit')
    cfg.flux.denoise = 0.5
    cfg.flux.steps = 7

    edit = Image.new('RGB', (64, 64), (10, 20, 30))
    ctx = Image.new('RGB', (64, 64), (200, 100, 50))
    out = render_flux_frame(
        pipe, cfg, edit_image=edit, prompt='p', negative_prompt='n',
        seed=1, width=64, height=64, reference_images=[ctx, edit],
    )

    # The fake pipeline does not declare `max_sequence_length`/`height`/`width`/
    # `negative_prompt`, so those must have been dropped, not passed.
    assert pipe.received['num_inference_steps'] == 7
    assert pipe.received['strength'] == 0.5
    # Raw content is primary; warped style is the ordered second reference.
    assert pipe.received['image'].getpixel((0, 0)) == (200, 100, 50)
    assert isinstance(pipe.received['image_2'], Image.Image)
    assert pipe.received['image_2'].size == ctx.size
    assert pipe.received['image_2'].getpixel((0, 0)) == (10, 20, 30)
    assert isinstance(out, Image.Image)


def test_render_flux_frame_warped_only_omits_raw_context(monkeypatch):
    from PIL import Image

    from vibewarp.core.flux import render_flux_frame

    _install_fake_torch(monkeypatch)

    pipe = _FakePipe()
    cfg = RunConfig(model_version='flux2_klein_edit')
    cfg.flux.denoise = 1.0

    edit = Image.new('RGB', (32, 32), (0, 0, 0))
    ctx = Image.new('RGB', (32, 32), (255, 255, 255))
    render_flux_frame(
        pipe, cfg, edit_image=edit, prompt='p', negative_prompt='',
        seed=0, width=32, height=32, reference_images=[edit],
    )
    assert pipe.received['image_2'] is None


@pytest.mark.parametrize(('fixed_seed', 'expected'), [(True, 42), (False, 45)])
def test_flux_frame_seed_policy_is_independent_from_sd(monkeypatch, tmp_path,
                                                       fixed_seed, expected):
    from PIL import Image

    from vibewarp.core.diffusion import FrameState, RenderContext, render_frame_flux
    import vibewarp.core.flux as flux_module

    init_path = tmp_path / 'warped.png'
    Image.new('RGB', (32, 32), 'gray').save(init_path)
    captured = {}

    def fake_render(pipe, config, **kwargs):
        captured['seed'] = kwargs['seed']
        return Image.new('RGB', (32, 32), 'blue')

    monkeypatch.setattr(flux_module, 'render_flux_frame', fake_render)
    config = RunConfig(model_version='flux2_klein_edit')
    config.video.width = config.video.height = 32
    config.diffusion.seed = 42
    config.diffusion.fixed_seed = False  # must not control the Flux policy
    config.flux.fixed_seed = fixed_seed
    ctx = RenderContext(config=config, flux_pipe=object())
    state = FrameState(frame_num=3, init_image=str(init_path))

    render_frame_flux(ctx, state)
    assert captured['seed'] == expected


def test_flux_frame_resolves_ordered_raw_previous_and_warped_references(
        monkeypatch, tmp_path):
    from PIL import Image

    from vibewarp.core.diffusion import FrameState, RenderContext, render_frame_flux
    import vibewarp.core.flux as flux_module

    frames = tmp_path / 'frames'
    frames.mkdir()
    Image.new('RGB', (32, 32), 'green').save(frames / '000002.jpg')
    edit_path = tmp_path / 'warped.png'
    Image.new('RGB', (32, 32), 'red').save(edit_path)
    captured = {}

    def fake_render(pipe, config, **kwargs):
        captured.update(kwargs)
        return Image.new('RGB', (32, 32), 'green')

    monkeypatch.setattr(flux_module, 'render_flux_frame', fake_render)
    config = RunConfig(model_version='flux2_klein_edit')
    config.video.width = config.video.height = 32
    config.flux.references = [
        EditReferenceConfig(source='raw'),
        EditReferenceConfig(source='previous'),
        EditReferenceConfig(source='warped'),
        EditReferenceConfig(source='none'),
    ]
    previous = Image.new('RGB', (32, 32), 'blue')
    state = FrameState(frame_num=1, init_image=str(edit_path), prev_frame=previous)

    render_frame_flux(RenderContext(
        config=config, flux_pipe=object(),
        video_frames_folder=str(frames) + '/'), state)

    pixels = [image.getpixel((0, 0))
              for image in captured['reference_images']]
    assert pixels[0][1] > pixels[0][0] and pixels[0][1] > pixels[0][2]
    assert pixels[1:] == [(0, 0, 255), (255, 0, 0)]
    assert captured['edit_image'] is captured['reference_images'][0]
    assert state.prev_frame.getpixel((0, 0)) == (0, 128, 0)


def test_flux_first_frame_omits_optional_temporal_references(
        monkeypatch, tmp_path):
    from PIL import Image

    from vibewarp.core.diffusion import FrameState, RenderContext, render_frame_flux
    import vibewarp.core.flux as flux_module

    frames = tmp_path / 'frames'
    frames.mkdir()
    raw_path = frames / '000001.jpg'
    Image.new('RGB', (32, 32), 'green').save(raw_path)
    captured = {}

    def fake_render(pipe, config, **kwargs):
        captured.update(kwargs)
        return Image.new('RGB', (32, 32), 'green')

    monkeypatch.setattr(flux_module, 'render_flux_frame', fake_render)
    config = RunConfig(model_version='flux2_klein_edit')
    config.video.width = config.video.height = 32
    config.flux.references = [
        EditReferenceConfig(source='raw'),
        EditReferenceConfig(source='previous'),
        EditReferenceConfig(source='warped'),
        EditReferenceConfig(source='none'),
    ]
    state = FrameState(frame_num=0, init_image=str(raw_path), prev_frame=None)

    render_frame_flux(RenderContext(
        config=config, flux_pipe=object(),
        video_frames_folder=str(frames) + '/'), state)

    assert len(captured['reference_images']) == 1
    assert captured['reference_images'][0].getpixel((0, 0))[1] > 100
