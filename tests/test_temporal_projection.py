from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn import functional as F

from vibewarp.config import RunConfig
from vibewarp.config_io import config_schema, validate_config
from vibewarp.core.comfy_temporal_guidance import VibeWarpTemporalGuidance, warp_target
from vibewarp.core.diffusion import FrameState, RenderContext
from vibewarp.core.edit import load_edit_backend, render_edit_frame
from vibewarp.core.temporal_projection import (
    ComfyFrameGuidance, native_target, save_native_history, wrap_sd_projection,
)


def test_packed_warp_moves_one_feature_cell_across_normalized_phases():
    image = torch.arange(64.).reshape(1, 1, 8, 8)
    mean = torch.tensor([1., -2., 3., -4.])
    std = torch.tensor([1., 2., 3., 4.])
    ae = SimpleNamespace(ps=[2, 2], bn_eps=0, bn=SimpleNamespace(
        running_mean=mean, running_var=std.square()))
    previous = (F.pixel_unshuffle(image, 2) - mean[:, None, None]) / std[:, None, None]
    flow = torch.zeros(16, 16, 2)
    flow[..., 0] = 2
    cc = torch.ones(16, 16)
    cc[2, 5] = 0
    corrected, mask = warp_target(previous, flow, cc, ae)
    restored = F.pixel_shuffle(corrected * std[:, None, None] + mean[:, None, None], 2)
    torch.testing.assert_close(restored[..., :-1], image[..., 1:])
    assert not mask[..., -1].any() and not mask[0, 0, 0, 1] and mask[0, 0, 1, 1]


@pytest.mark.parametrize('steps,count', [(4, 2), (8, 5), (16, 11)])
@pytest.mark.parametrize('video', [False, True])
def test_post_cfg_projects_only_trusted_predictions_and_scheduled_steps(steps, count, video):
    model = Mock(load_device='cpu')
    model.model.process_latent_in.side_effect = lambda x: x * 2
    previous = torch.ones(1, 4, 3, 4, dtype=torch.bfloat16)
    if video:
        previous = previous.unsqueeze(2)
    motion = torch.zeros(1, 3, 12, 16)
    motion[:, 2, :, 4:] = 1
    vae = SimpleNamespace(first_stage_model=None)
    node = VibeWarpTemporalGuidance()
    out, = node.patch(model, vae, {'samples': previous}, {'samples': motion}, .2, .1, .85,
                      previous={'samples': previous})
    assert out is model.clone.return_value
    correct, = out.set_model_sampler_post_cfg_function.call_args.args
    clean = torch.zeros_like(previous)
    sigmas = torch.linspace(1, 0, steps + 1)
    opts = {'transformer_options': {'sample_sigmas': sigmas}}
    results = [correct({'denoised': clean, 'sigma': sigma[None], 'model_options': opts})
               for sigma in sigmas[:-1]]
    changed = [r for r in results if r is not clean]
    assert len(changed) == count
    for r in changed:
        assert torch.equal(r[..., 0], clean[..., 0])
        assert (r[..., 1:] == .4).all() and r.dtype == clean.dtype
    model.reset_mock()
    motion.zero_()
    assert node.patch(model, vae, {'samples': previous}, {'samples': motion}, .2, 0, 1,
                      previous={'samples': previous}) == (model,)
    model.clone.assert_not_called()


def test_native_resume_and_partial_sd_schedule(tmp_path):
    ctx = RenderContext(batch_folder=str(tmp_path))
    ctx._sequence_start_frame = 7
    ctx.temporal_motion = (np.zeros((8, 8, 2)), np.ones((8, 8)))
    ctx.config.flow.temporal_guidance_strength = .2
    state = FrameState(frame_num=7)
    assert native_target(ctx, state, None) is None
    history = torch.full((1, 4, 2, 2), 5.)
    save_native_history(ctx, state, history)
    state.frame_num += 1
    target = native_target(ctx, state, None)
    torch.testing.assert_close(target[0], history)
    clean = torch.zeros_like(history)
    model = Mock(return_value=clean)
    schedule = torch.tensor([.6, .5, .2, .1, 0])
    wrapper = wrap_sd_projection(model, target, schedule, ctx.config.flow)
    outputs = [wrapper(clean, sigma[None]) for sigma in schedule[:-1]]
    assert model.call_count == 4 and outputs[0] is clean and outputs[-1] is clean
    assert (outputs[1] == 1).all() and (outputs[2] == 1).all()
    state.frame_num += 1
    with pytest.raises(ValueError, match='Missing temporal latent'):
        native_target(ctx, state, None)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA interpolation regression')
def test_restored_native_history_uses_same_warp_kernel(tmp_path):
    ctx = RenderContext(batch_folder=str(tmp_path))
    rng = np.random.default_rng(42)
    ctx.temporal_motion = (rng.normal(size=(80, 64, 2)).astype('float32'), np.ones((80, 64)))
    state = FrameState()
    state.prev_latent = torch.from_numpy(rng.normal(size=(1, 3, 80, 64)).astype('float32')).cuda()
    save_native_history(ctx, state, state.prev_latent)
    state.frame_num = 1
    live = native_target(ctx, state, None, 'cuda')
    state.prev_latent = None
    restored = native_target(ctx, state, None, 'cuda')
    assert all(torch.equal(a, b) for a, b in zip(live, restored))


@pytest.mark.parametrize('version', ['flux2_klein_edit', 'flux2_klein_9b_edit', 'hidream_o1_edit',
    'qwen_image_edit_2511', 'qwen_image_edit_2511_gguf', 'mage_flow_edit', 'mage_flow_edit_turbo',
    'flux3_klein', 'moss_image', 'minimax_h3_image'])
def test_edit_dispatch_captures_and_reuses_sampled_latent(version, tmp_path, monkeypatch):
    from vibewarp.core import comfy_image, flux_comfy, hidream_comfy, qwen_comfy, mage_comfy
    modules = [comfy_image, flux_comfy, hidream_comfy, qwen_comfy, mage_comfy]
    for module in modules:
        monkeypatch.setattr(module, 'ComfyProgressMonitor', Mock())
    for cls in (comfy_image.ComfyImageClient, flux_comfy.ComfyFluxClient, hidream_comfy.ComfyHiDreamClient,
                qwen_comfy.ComfyQwenEditClient, mage_comfy.ComfyMageFlowEditClient):
        monkeypatch.setattr(cls, '_validate_server', lambda self: None)
    cfg = RunConfig(model_version=version)
    cfg.flow.temporal_guidance_strength = .2
    ctx = RenderContext(config=cfg, batch_folder=str(tmp_path))
    client = load_edit_backend(cfg)
    reference = Image.new('RGB', (32, 32))
    graphs = []
    def request(method, path, **kw):
        if path == '/prompt':
            graphs.append(kw['json']['prompt'])
            data = {'prompt_id': 'job'}
        elif path == '/upload/image':
            data = {'subfolder': 'test', 'name': kw['files']['image'][0]}
        elif path.startswith('/object_info/'):
            data = {'VibeWarpTemporalGuidance': True}
        else:
            data = {'job': {'outputs': {'vw_history': {'latents': [
                {'subfolder': 'test', 'filename': 'sample_00001_.latent'}]}}}}
        return Mock(json=lambda: data)
    client._request = request
    client._wait_for_output = lambda prompt: reference
    state = FrameState()
    for i in range(2):
        state.frame_num = i
        frame = ComfyFrameGuidance.for_frame(ctx, state)
        render_edit_frame(client, cfg, reference, 'paint', '', 123, 32, 32,
                          frame_num=i, temporal_frame=frame)
        ctx.temporal_motion = (np.zeros((32, 32, 2)), np.ones((32, 32)))
    assert 'vw_guidance' not in graphs[0]
    g = graphs[1]
    decode = next(n['inputs'] for n in g.values() if n['class_type'] == 'VAEDecode')
    assert g['vw_history']['inputs']['samples'] == decode['samples']
    assert g['vw_previous']['inputs']['latent'] == 'test/sample_00001_.latent [output]'
    sampler = g[decode['samples'][0]]['inputs']
    owner = g[sampler['guider'][0]]['inputs'] if 'guider' in sampler else sampler
    assert owner['model'] == ['vw_guidance', 0]
    assert g['vw_guidance']['inputs']['canvas'] == sampler['latent_image']
    assert g['vw_guidance']['inputs']['vae'] == decode['vae']


def test_guidance_is_opt_in_and_validation_names_incompatible_settings():
    c = RunConfig()
    field = config_schema()['properties']['flow']['properties']['temporal_guidance_strength']
    assert field['default'] == 0 and field['workflows'] == ['consistency']
    c.flow.temporal_guidance_strength = .2
    c.flow.check_consistency = False
    c.flow.temporal_guidance_end = 0
    c.animatediff.enabled = True
    errors = {e['path'] for e in validate_config(c, require_inputs=False)}
    assert {'flow.check_consistency', 'flow.temporal_guidance_start', 'flow.temporal_guidance_strength'} <= errors
