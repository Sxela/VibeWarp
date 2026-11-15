from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vibewarp.config import RunConfig
from vibewarp.core.diffusion import RenderContext, run_frames


def make_context(tmp_path):
    video = tmp_path / 'frames'
    video.mkdir(exist_ok=True)
    batch = tmp_path / 'out'
    batch.mkdir(exist_ok=True)
    for frame in range(8):
        Image.new('RGB', (64, 64), (50, 60, 70)).save(video/f'{frame+1:06d}.jpg')
    config = RunConfig(model_version='disco_512')
    config.video.width = config.video.height = 64
    config.turbo.enabled, config.turbo.preroll = True, 1
    config.turbo.interval = 3
    config.color.before_enabled = config.color.after_enabled = False
    return RenderContext(config=config, batch_folder=str(batch), video_frames_folder=str(video))


def install_fakes(monkeypatch, rendered):
    def warp(ctx, previous, frame, **kwargs):
        pixels = np.minimum(np.asarray(previous, dtype=np.uint16)+1, 255).astype(np.uint8)
        image = Image.fromarray(pixels)
        return (image, image.copy()) if kwargs.get('return_warped') else image
    def render(ctx, state):
        rendered.append(state.frame_num)
        image = Image.new('RGB', (64, 64), (30+state.frame_num*12, 40, 50))
        state.prev_frame = image
        return image
    monkeypatch.setattr('vibewarp.core.diffusion.warp_between_frames', warp)
    monkeypatch.setattr('vibewarp.core.diffusion.render_frame', render)


def test_turbo_skips_diffusion_but_outputs_every_frame(tmp_path, monkeypatch):
    rendered = []
    install_fakes(monkeypatch, rendered)
    ctx = make_context(tmp_path)
    paths = run_frames(ctx, frame_range=[0, 7])
    assert rendered == [0, 1, 3, 6]
    assert len(paths) == 8 and all(Path(p).is_file() for p in paths)
    assert (Path(ctx.batch_folder)/'temp/turbo/000007/state.json').is_file()


def test_turbo_resume_matches_uninterrupted_output(tmp_path, monkeypatch):
    rendered = []
    install_fakes(monkeypatch, rendered)
    ctx = make_context(tmp_path)
    full = run_frames(ctx, frame_range=[0, 7])
    expected = [np.asarray(Image.open(p)).copy() for p in full[5:]]
    # Resume from a saved intermediate frame, rather than an anchor.
    resumed_ctx = make_context(tmp_path)
    rendered.clear()
    resumed = run_frames(resumed_ctx, frame_range=[0, 7], resume_from=5)
    assert rendered == [6]
    for path, pixels in zip(resumed, expected):
        np.testing.assert_array_equal(np.asarray(Image.open(path)), pixels)


def test_light_turbo_renders_intermediates_and_restores_strength(tmp_path, monkeypatch):
    rendered = []
    install_fakes(monkeypatch, rendered)
    ctx = make_context(tmp_path)
    ctx.config.turbo.intermediate_strength = .1
    run_frames(ctx, frame_range=[0, 4])
    assert rendered == list(range(5))
    assert ctx.config.diffusion.style_strength_schedule is None


def test_scene_cut_forces_diffusion_and_resets_histories(tmp_path, monkeypatch):
    rendered = []
    install_fakes(monkeypatch, rendered)
    ctx = make_context(tmp_path)
    Image.new('RGB', (64, 64), 'white').save(Path(ctx.video_frames_folder)/'000003.jpg')
    run_frames(ctx, frame_range=[0, 4])
    assert 2 in rendered


def test_resume_with_changed_turbo_settings_fails_clearly(tmp_path, monkeypatch):
    install_fakes(monkeypatch, [])
    ctx = make_context(tmp_path)
    run_frames(ctx, frame_range=[0, 4])
    ctx.config.turbo.interval = 4
    with pytest.raises(ValueError, match='settings changed'):
        run_frames(ctx, frame_range=[0, 4], resume_from=3)


def test_turbo_resume_survives_saved_settings_numeric_coercion(tmp_path, monkeypatch):
    from vibewarp.settings import save_settings, settings_to_dict
    from vibewarp.config_io import config_from_settings
    install_fakes(monkeypatch, [])
    ctx = make_context(tmp_path)
    ctx.config.disco.init_scale = 0
    full = run_frames(ctx, frame_range=[0, 7])
    expected = [Path(p).read_bytes() for p in full[5:]]
    settings = save_settings(settings_to_dict(ctx.config), ctx.batch_folder, 'saved', 0)
    restored = make_context(tmp_path)
    restored.config = config_from_settings(settings)
    paths = run_frames(restored, frame_range=[0, 7], resume_from=5)
    assert [Path(p).read_bytes() for p in paths] == expected


@pytest.mark.parametrize('skip,strength', [(None, 0), (.75, .25), ('75%', .25),
    ('100% (don`t diffuse turbo frames, fastest)', 0), (0, 1), (1, 0)])
def test_notebook_turbo_controls_import_and_roundtrip(tmp_path, skip, strength):
    import json
    from vibewarp.config_io import config_from_settings, config_from_dict
    from vibewarp.disco_settings import import_disco_notebook
    from vibewarp.settings import settings_to_dict
    config = config_from_dict(import_disco_notebook(dict(
        turbo_mode=True, turbo_frame_skips_steps=skip,
        forward_weights_clip=.2, forward_weights_clip_turbo_step=.8,
        disable_cc_for_turbo_frames=True)))
    assert config.turbo.intermediate_strength == strength
    assert config.turbo.consistency_soften == .8
    assert config.turbo.disable_cc
    assert config.flow.soften_consistency_mask == .2
    path = tmp_path/'settings.txt'
    path.write_text(json.dumps(settings_to_dict(config)))
    assert config_from_settings(path).turbo == config.turbo


def test_notebook_turbo_softness_widget_alias():
    from vibewarp.disco_settings import import_disco_notebook
    assert import_disco_notebook({'soften_consistency_mask_for_turbo_frames': .7})['turbo']['consistency_soften'] == .7


def test_zero_turbo_softness_preserves_notebook_fallback(tmp_path):
    from vibewarp.core.turbo import TurboState
    config = RunConfig()
    turbo = TurboState(config, tmp_path, 0)
    schedule = {'soften_consistency_mask': .2}
    assert turbo.warp_params(1, schedule)['soften_consistency_mask'] == .2
    assert turbo.warp_params(1, schedule, old=True)['soften_consistency_mask'] == 0
    assert schedule['soften_consistency_mask'] == .2


@pytest.mark.parametrize('strength', [0, .25])
def test_both_histories_use_turbo_softness_and_cc_override(tmp_path, monkeypatch, strength):
    install_fakes(monkeypatch, [])
    ctx = make_context(tmp_path)
    ctx.config.turbo.intermediate_strength = strength
    ctx.config.turbo.consistency_soften = .8
    ctx.config.turbo.disable_cc = True
    ctx.config.flow.soften_consistency_mask = .2
    calls = {}
    def warp(ctx, image, frame, **kw):
        calls.setdefault(frame+1, []).append(kw['consistency_params'].copy())
        return (image, image.copy()) if kw.get('return_warped') else image
    monkeypatch.setattr('vibewarp.core.diffusion.warp_between_frames', warp)
    run_frames(ctx, frame_range=[0, 4])
    # The old history is only warped where it is used (warp-only frames); the
    # notebook also warps it on diffused frames, then discards the result.
    # Frame 2 is the last turbo frame of its cycle (old weight 0): no old-history warp.
    assert [s['soften_consistency_mask'] for s in calls[2]] == [.8]
    # Frame 4 (weight 2/3) warps both histories with the Turbo softness.
    assert [s['soften_consistency_mask'] for s in calls[4]] == ([.8, .8] if strength == 0 else [.8])
    assert all(s['check_consistency'] is False for s in calls[2])
    # Anchor keeps normal softness and does not disable consistency.
    assert [s['soften_consistency_mask'] for s in calls[3]] == [.2]
    assert all(s.get('check_consistency', True) for s in calls[3])
    assert ctx.config.flow.check_consistency
    assert ctx.config.flow.soften_consistency_mask == .2


def test_disabled_consistency_ignores_even_a_cached_mask(tmp_path, monkeypatch):
    from vibewarp.core.diffusion import warp_between_frames
    ctx = make_context(tmp_path)
    seen = {}
    def flow(**kw):
        seen['cc_path'] = kw['cc_path']
        return np.zeros((64, 64, 2), dtype=np.float32), np.zeros((64, 64, 3), dtype=np.uint8)
    def warp(**kw):
        seen['weights'] = kw['weights']
        return kw['frame1'], kw['frame1'].copy()
    monkeypatch.setattr('vibewarp.flow.flow_utils.get_flow_and_cc', flow)
    monkeypatch.setattr('vibewarp.flow.warp.warp_frame', warp)
    warp_between_frames(ctx, Image.new('RGB', (64, 64)), 1,
        consistency_params={'check_consistency': False}, return_warped=True)
    assert seen == {'cc_path': '', 'weights': None}
    assert ctx._disco_consistency_mask is None
    assert ctx._temporal_consistency_mask is None


@pytest.mark.parametrize('strength', [0, .25])
def test_after_color_corrects_anchor_and_intermediate_feedback_and_resume(tmp_path, monkeypatch, strength):
    from vibewarp.core import diffusion
    install_fakes(monkeypatch, [])
    ctx = make_context(tmp_path)
    ctx.config.turbo.intermediate_strength = strength
    ctx.config.color.after_enabled = True
    seen = {}
    def color(image, reference, config):
        return Image.fromarray(np.minimum(np.asarray(image, dtype=np.uint16)+10, 255).astype(np.uint8))
    def warp(ctx, image, frame, **kw):
        if kw.get('return_warped'):
            seen[frame+1] = image.copy()
            return image, image.copy()
        return image
    monkeypatch.setattr(diffusion, '_apply_post_colormatch', color)
    monkeypatch.setattr(diffusion, 'warp_between_frames', warp)
    full = run_frames(ctx, frame_range=[0, 7])
    root = Path(ctx.batch_folder)/'temp/turbo'
    for frame in (2, 3):  # warp-only/light intermediate and diffusion anchor
        feedback = np.asarray(Image.open(root/f'{frame:06d}'/'feedback.png'))
        np.testing.assert_array_equal(np.asarray(seen[frame+1]), feedback)
        if frame == 3 or strength:
            assert tuple(feedback[0, 0]) == (30+frame*12+10, 50, 60)
        else:
            previous = np.asarray(Image.open(root/f'{frame-1:06d}'/'feedback.png'))
            np.testing.assert_array_equal(feedback, previous+10)
    expected = [Path(p).read_bytes() for p in full[5:]]
    resumed = make_context(tmp_path)
    resumed.config = ctx.config
    assert [Path(p).read_bytes() for p in run_frames(resumed, frame_range=[0, 7], resume_from=5)] == expected


@pytest.mark.parametrize('factor', [.5, 1/3, 2/3, .25, .75])
def test_tensor_blending_uses_nearest_rounding_like_notebook(factor):
    import cv2
    from vibewarp.core.turbo import blend_turbo
    values = np.arange(256, dtype=np.uint8)
    new = np.broadcast_to(values[:, None, None], (256, 256, 3)).copy()
    old = np.broadcast_to(values[None, :, None], (256, 256, 3)).copy()
    expected = cv2.addWeighted(new, factor, old, 1-factor, 0)
    actual = blend_turbo(Image.fromarray(new), Image.fromarray(old), factor, device='cpu')
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.gpu
def test_cuda_turbo_blend_matches_cpu():
    import torch
    from vibewarp.core.turbo import blend_turbo
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    values = np.arange(256, dtype=np.uint8)
    new = Image.fromarray(np.broadcast_to(values[:, None, None], (256, 256, 3)).copy())
    old = Image.fromarray(np.broadcast_to(values[None, :, None], (256, 256, 3)).copy())
    for factor in (.5, 1/3, 2/3, .25, .75):
        np.testing.assert_array_equal(np.asarray(blend_turbo(new, old, factor, device='cuda')),
            np.asarray(blend_turbo(new, old, factor, device='cpu')))


def test_history_is_warped_only_on_warp_only_frames(tmp_path, monkeypatch):
    install_fakes(monkeypatch, [])
    history_warps = []
    flow_warp = __import__('vibewarp.core.diffusion', fromlist=['x']).warp_between_frames
    def counting(ctx, previous, frame, **kwargs):
        if not kwargs.get('return_warped'):
            history_warps.append(frame + 1)
        return flow_warp(ctx, previous, frame, **kwargs)
    monkeypatch.setattr('vibewarp.core.diffusion.warp_between_frames', counting)
    run_frames(make_context(tmp_path), frame_range=[0, 7])
    # interval 3, preroll 1: frames 3 and 6 are diffused, so their history is replaced;
    # frames 2 and 5 give the old history weight 0, so it is dropped, not warped.
    assert history_warps == [4, 7]
    history_warps.clear()
    (tmp_path / 'light').mkdir()
    ctx = make_context(tmp_path / 'light')
    ctx.config.turbo.intermediate_strength = .1
    run_frames(ctx, frame_range=[0, 4])
    assert history_warps == []


def test_scene_cut_init_keeps_raw_preparation_and_background_mask(tmp_path, monkeypatch):
    calls = []
    install_fakes(monkeypatch, [])
    def prepared(ctx, frame, reference=None, **kwargs):
        calls.append(('prepared', frame))
        return str(Path(ctx.video_frames_folder) / f'{frame+1:06d}.jpg')
    def masked(ctx, path, frame):
        calls.append(('masked', frame, Path(path).name))
        return path
    monkeypatch.setattr('vibewarp.video.raw_processing.prepared_raw_path', prepared)
    monkeypatch.setattr('vibewarp.core.diffusion._masked_init_path', masked)
    ctx = make_context(tmp_path)
    Image.new('RGB', (64, 64), 'white').save(Path(ctx.video_frames_folder)/'000003.jpg')
    run_frames(ctx, frame_range=[0, 4])
    assert ('prepared', 2) in calls
    assert calls[calls.index(('prepared', 2)) + 1] == ('masked', 2, '000003.jpg')


def test_before_diffusion_mask_also_masks_the_warped_old_history(tmp_path, monkeypatch):
    # Without it, the 1/3 old-history share of a turbo crossfade brings back an
    # unmasked ghost of the previous pose (measured on a real render, 2026-11-15).
    install_fakes(monkeypatch, [])
    masked = []
    def stage(image, ctx, frame, which):
        if which == 'before diffusion':
            masked.append(frame)
        return image
    monkeypatch.setattr('vibewarp.core.diffusion._apply_background_stage', stage)
    ctx = make_context(tmp_path)
    ctx.config.mask.use_background_mask = True
    ctx.config.mask.render_mask_stage = 'before diffusion'
    run_frames(ctx, frame_range=[0, 7])
    # Every init is masked once; warp-only frames that use the old history (4, 7) mask it too.
    assert sorted(masked) == sorted(list(range(8)) + [4, 7])
