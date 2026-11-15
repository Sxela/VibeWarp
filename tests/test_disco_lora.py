import json

import numpy as np
import pytest
import torch
from torch import nn
from safetensors.torch import save_file

from vibewarp.config import DISCO_DISTILLED_PRESET
from vibewarp.config_io import config_from_dict, validate_config, config_schema
from vibewarp.core.disco_lora import (AdapterLayer, GRIDS, TEACHER_SHA256, VPSchedule,
    adapter_config, inject_adapters, load_adapter_state, resolve_adapter, rollout, sha256, use_adapter)


def metadata(**changes):
    return dict(teacher_sha256=TEACHER_SHA256, grid=GRIDS[4], size=512, rank=2,
        alpha=4, objective_version='supervised-clean-sqrt-snr-v1', **changes)


def write_adapter(path, changes=None):
    config = metadata()
    config.update(changes or {})
    save_file({'sample':torch.ones(1)}, str(path), metadata={'config':json.dumps(config)})
    return path


@pytest.mark.parametrize('base,shape', [
    (nn.Linear(3,5),(2,3)),
    (nn.Conv1d(3,5,3,stride=2,padding=2,dilation=2),(2,3,13)),
    (nn.Conv2d(3,5,3,stride=2,padding=2,dilation=2),(2,3,13,15)),
])
def test_adapter_all_layer_types_scale_and_disabled_base_parity(base,shape):
    torch.manual_seed(42)
    model = nn.Sequential(base)
    x = torch.randn(shape)
    expected = model(x).detach()
    inject_adapters(model,2,4)
    layer = model[0]
    nn.init.normal_(layer.adapters['student'][1].weight)
    torch.testing.assert_close(model(x),expected,rtol=0,atol=0)
    with use_adapter(model):
        torch.testing.assert_close(model(x),expected+2*layer.adapters['student'](x))
        with use_adapter(model,None):
            torch.testing.assert_close(model(x),expected,rtol=0,atol=0)
        assert layer.active == 'student'
    assert layer.active is None
    with pytest.raises(ValueError,match='already injected'):
        inject_adapters(model,2,4)


def test_adapter_loading_rejects_extra_missing_bad_shape_and_nonfinite_weights():
    model = nn.Sequential(nn.Linear(3,5))
    inject_adapters(model,2,4)
    state = {f'0.adapters.student.{key}':value.clone() for key,value in model[0].adapters['student'].state_dict().items()}
    load_adapter_state(model,state)
    for invalid in ({}, {**state,'extra':torch.ones(1)},
                    {**state,'0.adapters.student.0.weight':torch.zeros(1)},
                    {**state,'0.adapters.student.0.weight':torch.full((2,3),float('nan'))}):
        with pytest.raises(ValueError):
            load_adapter_state(model,invalid)


def test_strength_scales_adapter_and_zero_is_exact_base_with_nested_restore():
    torch.manual_seed(7)
    model=nn.Sequential(nn.Linear(3,5))
    x=torch.randn(2,3)
    base=model(x).detach()
    inject_adapters(model,2,4)
    nn.init.normal_(model[0].adapters['student'][1].weight)
    with use_adapter(model,strength=1):
        trained=model(x).detach()
        with use_adapter(model,strength=.5):
            torch.testing.assert_close(model(x),base+(trained-base)*.5)
            with use_adapter(model,strength=0):
                assert torch.equal(model(x),base)
            assert model[0].strength==.5
            with pytest.raises(RuntimeError):
                with use_adapter(model,strength=2):
                    raise RuntimeError('cancel render')
            assert model[0].strength==.5
        assert model[0].strength==1
    assert model[0].active is None and model[0].strength==1


@pytest.mark.parametrize('strength',[-1,float('nan'),float('inf')])
def test_bad_lora_strength_is_rejected(strength):
    config=config_from_dict(DISCO_DISTILLED_PRESET)
    config.disco.few_step_lora_path='adapter.safetensors'
    config.disco.lora_strength=strength
    assert any(e['path']=='disco.lora_strength' for e in validate_config(config,require_inputs=False))
    with pytest.raises(ValueError,match='strength'):
        with use_adapter(nn.Linear(2,2),strength=strength):
            pass


def test_reference_render_passes_configured_strength_to_rollout(monkeypatch):
    from PIL import Image
    from vibewarp.core.disco_lora import DistilledDiscoBackend
    backend=DistilledDiscoBackend.__new__(DistilledDiscoBackend)
    config=config_from_dict(DISCO_DISTILLED_PRESET)
    config.disco.lora_strength=.35
    calls=[]
    def sample(seed,**kwargs):
        calls.append(kwargs['lora_strength'])
        return torch.zeros(1,3,512,512)
    monkeypatch.setattr(backend,'sample_tensor',sample)
    backend.render(config,Image.new('RGB',(512,512)),'','',42)
    assert calls==[.35]


@pytest.mark.parametrize('changes', [
    {'teacher_sha256':'wrong'}, {'grid':[999,666,333,-1]}, {'grid':[999.,749,499,249,-1]},
    {'size':256}, {'rank':0}, {'alpha':float('nan')}, {'fp32_output_head':False},
    {'objective_version':'endpoint-dsm-pseudohuber-v3'},
])
def test_rejects_incompatible_adapter_metadata(tmp_path, changes):
    with pytest.raises(ValueError):
        adapter_config(write_adapter(tmp_path/'test.safetensors',changes))


def test_resolves_current_retained_best_without_pinning_a_checksum(tmp_path):
    selected = tmp_path/'best'/'selected-1010'
    selected.mkdir(parents=True)
    weights = write_adapter(selected/'best-ema.safetensors')
    (tmp_path/'best.json').write_text(json.dumps({'checkpoint':selected.name}))
    manifest = selected/'COMPLETE.json'
    manifest.write_text(json.dumps({'files':{'best-ema.safetensors':sha256(weights)}}))
    assert resolve_adapter(tmp_path) == weights
    assert adapter_config(weights)['grid'] == GRIDS[4]
    # Safetensors cannot carry code, so a stale digest is not an error...
    manifest.write_text(json.dumps({'files':{'best-ema.safetensors':'0'*64}}))
    assert resolve_adapter(tmp_path) == weights
    # ...but an unfinished training generation still is.
    manifest.unlink()
    with pytest.raises(ValueError,match='incomplete'):
        resolve_adapter(tmp_path)


def test_saved_settings_with_retired_lora_checksums_still_load():
    from vibewarp.config_io import config_from_dict
    config = config_from_dict({'disco': {'few_step_lora_sha256': 'a'*64, 'quality_lora_sha256': 'b'*64}})
    assert not hasattr(config.disco, 'few_step_lora_sha256')


@pytest.mark.parametrize('pointer',['../escape',r'..\escape','.','..',''])
def test_rejects_unsafe_best_pointer(tmp_path,pointer):
    (tmp_path/'best.json').write_text(json.dumps({'checkpoint':pointer}))
    with pytest.raises(ValueError,match='pointer'):
        resolve_adapter(tmp_path)


@pytest.mark.parametrize('steps',[4,6])
def test_saved_grid_original_timesteps_fp32_math_clean_endpoint_and_no_clipping(steps):
    schedule = VPSchedule(np.linspace(.9999,.00004,1000),'cpu')
    calls = []
    class Model(nn.Module):
        def forward(self,x,t):
            calls.append(int(t[0]))
            assert t.dtype == torch.long
            a,s = schedule.coefficients(t,x)
            eps = (x-a*2.5)/s
            return torch.cat((eps,torch.full_like(eps,float('nan'))),dim=1)
    result = list(rollout(Model(),schedule,torch.randn(1,3,4,4),GRIDS[steps]))
    assert calls == GRIDS[steps][:-1]
    assert len(result) == steps and result[-1].dtype == torch.float32
    torch.testing.assert_close(result[-1],torch.full_like(result[-1],2.5),atol=2e-5,rtol=2e-5)


def test_preset_validation_ui_schema_and_only_base_assets():
    config = config_from_dict(DISCO_DISTILLED_PRESET)
    config.disco.few_step_lora_path = 'adapter.safetensors'
    assert not validate_config(config,require_inputs=False)
    from vibewarp.model_assets import selected_assets
    assets = selected_assets(config)
    assert len(assets) == 1 and assets[0].sha256 == TEACHER_SHA256
    schema = config_schema()
    assert 'distilled' in schema['properties']['disco']['properties']['sampler']['choices']
    assert schema['disco_distilled_preset']['disco']['distilled_guided'] is True
    assert 'steps' not in schema['disco_distilled_preset']['disco']


@pytest.mark.parametrize('section,field,value', [
    ('disco','use_secondary_model',True),('disco','clip_guidance_scale',1000),
    ('disco','steps',5),('disco','style_strength',.5),('disco','inpaint',True),
    ('diffusion','tiled_sampler',True),('flow','flow_warp',True),('video','width',1024),
])
def test_distilled_validation_blocks_unvalidated_workflows(section,field,value):
    config = config_from_dict(DISCO_DISTILLED_PRESET)
    config.disco.few_step_lora_path = 'adapter.safetensors'
    setattr(getattr(config,section),field,value)
    assert any(error['path'] == f'{section}.{field}' for error in validate_config(config,require_inputs=False))
