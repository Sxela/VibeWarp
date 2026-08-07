import torch

from vibewarp.core.controlnet import patch_model_for_controlnets
from vibewarp.core.unet_acceleration import (
    begin_unet_evaluation,
    compile_auxiliary_models,
    configure_persistent_compile_cache,
    configure_unet_acceleration,
    reset_unet_cache,
)


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, h, emb, context):
        self.calls += 1
        return h + 0.01


class _Out(torch.nn.Module):
    def forward(self, h):
        return h[:, :1]


class _UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_channels = 1
        self.dtype = torch.float32
        self.num_classes = None
        self.time_embed = torch.nn.Identity()
        self.input_blocks = torch.nn.ModuleList([_Block() for _ in range(4)])
        self.middle_block = _Block()
        self.output_blocks = torch.nn.ModuleList([_Block() for _ in range(4)])
        self.out = _Out()


class _SD:
    def __init__(self):
        self.model = type('Container', (), {})()
        self.model.diffusion_model = _UNet()


def _call(sd, x):
    begin_unet_evaluation(sd)
    return sd.apply_model(
        x, torch.tensor([10.0]), torch.zeros(1, 1, 1))


def test_deepcache_skips_inner_blocks_on_intermediate_evaluation():
    sd = _SD()
    patch_model_for_controlnets(sd, {})
    configure_unet_acceleration(
        sd, cache_mode='deepcache', cache_interval=2)
    x = torch.zeros(1, 1, 8, 8)

    _call(sd, x)
    middle_calls = sd.model.diffusion_model.middle_block.calls
    _call(sd, x)

    runtime = sd.model.diffusion_model._vw_cache_runtime
    assert middle_calls == 1
    assert sd.model.diffusion_model.middle_block.calls == 1
    assert runtime['hits'] == 1
    assert runtime['misses'] == 1


def test_first_block_cache_reuses_output_for_identical_first_activation():
    sd = _SD()
    patch_model_for_controlnets(sd, {})
    configure_unet_acceleration(
        sd, cache_mode='first_block', cache_threshold=0.0)
    x = torch.zeros(1, 1, 8, 8)

    first = _call(sd, x)
    middle_calls = sd.model.diffusion_model.middle_block.calls
    second = _call(sd, x)

    runtime = sd.model.diffusion_model._vw_cache_runtime
    assert torch.equal(first, second)
    assert sd.model.diffusion_model.middle_block.calls == middle_calls
    assert runtime['hits'] == 1
    assert runtime['misses'] == 1


def test_cache_reset_drops_frame_state_and_statistics():
    sd = _SD()
    patch_model_for_controlnets(sd, {})
    configure_unet_acceleration(sd, cache_mode='deepcache')
    _call(sd, torch.zeros(1, 1, 8, 8))

    reset_unet_cache(sd)

    assert sd.model.diffusion_model._vw_cache_runtime == {
        'evaluation': -1,
        'slot_cursor': 0,
        'slots': {},
        'hits': 0,
        'misses': 0,
    }


def test_compile_cache_uses_stable_directory_and_enables_fx_cache(
        tmp_path, monkeypatch):
    monkeypatch.delenv('TORCHINDUCTOR_CACHE_DIR', raising=False)
    monkeypatch.delenv('TORCHINDUCTOR_FX_GRAPH_CACHE', raising=False)
    target = tmp_path / 'inductor'

    result = configure_persistent_compile_cache(str(target))

    assert result == str(target.resolve())
    assert target.is_dir()
    assert __import__('os').environ['TORCHINDUCTOR_FX_GRAPH_CACHE'] == '1'


def test_auxiliary_compile_wraps_each_unique_controlnet_and_vae(monkeypatch):
    wrapped = []

    def fake_compile(callable_, **kwargs):
        wrapped.append((callable_, kwargs))
        return callable_

    monkeypatch.setattr(torch, 'compile', fake_compile)

    class _Aux(torch.nn.Module):
        def forward(self, value):
            return value

    class _VAE:
        def encode(self, value):
            return value

        def decode(self, value):
            return value

    sd = type('SD', (), {'first_stage_model': _VAE()})()
    controlnet = _Aux()
    compile_auxiliary_models(sd, {
        'one': {'model': controlnet},
        'same-checkpoint': {'model': controlnet},
    })

    assert len(wrapped) == 3
    assert controlnet._vw_compiled is True
    assert sd.first_stage_model._vw_compiled is True
