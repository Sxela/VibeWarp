"""Background mask sources: RVM matting, a user mask video, or a user image folder.

Every source is materialized into <run>/video_frames_alpha, which render-time and
assembly-time masking both read (review findings 1 and 6, 2026-11-15).
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vibewarp.config import RunConfig
from vibewarp.video import masks


def _run(tmp_path, frames=(0, 1, 2), size=(8, 8)):
    run = tmp_path / 'run'
    video_frames = run / 'video_frames'
    video_frames.mkdir(parents=True)
    for index in frames:
        Image.new('RGB', size, 'white').save(video_frames / f'{index + 1:06d}.jpg')
    cfg = RunConfig()
    cfg.video.width, cfg.video.height = size
    cfg.mask.use_background_mask = True
    cfg.disco.pad_to_multiple_64 = False
    return cfg, run, video_frames


def _mask_folder(tmp_path, values, size=(4, 4)):
    folder = tmp_path / 'user_masks'
    folder.mkdir()
    for index, value in enumerate(values):
        # Natural sort: mask_10 comes after mask_9.
        Image.new('L', size, value).save(folder / f'mask_{index}.png')
    return folder


def _materialized(video_frames, index):
    with Image.open(Path(masks.run_mask_folder(video_frames)) / f'{index + 1:06d}.png') as image:
        return image.size, image.getpixel((0, 0))


def test_image_folder_is_one_mask_per_source_frame_and_follows_the_stride(tmp_path):
    cfg, run, video_frames = _run(tmp_path, frames=(1, 2))
    cfg.mask.mask_source, cfg.video.extract_nth_frame = masks.FOLDER, 2
    cfg.mask.mask_path = str(_mask_folder(tmp_path, [10 * i for i in range(12)]))
    masks.materialize(cfg, video_frames, run)
    # Extracted frame k is source frame 2k: frames 1, 2 -> images 2, 4 (natural order).
    assert _materialized(video_frames, 1) == ((8, 8), 20)
    assert _materialized(video_frames, 2) == ((8, 8), 40)


def test_image_folder_with_too_few_masks_fails_before_rendering(tmp_path):
    cfg, run, video_frames = _run(tmp_path, frames=(0, 1, 2))
    cfg.mask.mask_source = masks.FOLDER
    cfg.mask.mask_path = str(_mask_folder(tmp_path, [0, 255]))
    with pytest.raises(ValueError, match='has 2 images.*frame 3'):
        masks.materialize(cfg, video_frames, run)


def test_mask_video_is_extracted_with_the_input_stride_and_numbering(tmp_path, monkeypatch):
    cfg, run, video_frames = _run(tmp_path, frames=(3, 4))
    cfg.mask.mask_source, cfg.mask.mask_path = masks.VIDEO, str(tmp_path / 'matte.mp4')
    cfg.video.extract_nth_frame = 3
    calls = []
    def extract(**kwargs):
        calls.append(kwargs)
        Path(kwargs['output_dir']).mkdir(parents=True, exist_ok=True)
        for offset, value in enumerate((60, 90)):
            Image.new('L', (4, 4), value).save(
                Path(kwargs['output_dir']) / f"{kwargs['start_number'] + offset:06d}.jpg")
    monkeypatch.setattr('vibewarp.video.input.extract_frames', extract)
    masks.materialize(cfg, video_frames, run)
    assert calls[0]['nth_frame'] == 3
    assert (calls[0]['start_frame'], calls[0]['end_frame'], calls[0]['start_number']) == (9, 12, 4)
    assert abs(_materialized(video_frames, 3)[1] - 60) <= 2  # JPEG round trip
    assert abs(_materialized(video_frames, 4)[1] - 90) <= 2


def test_short_mask_video_fails_clearly(tmp_path, monkeypatch):
    cfg, run, video_frames = _run(tmp_path, frames=(0, 1))
    cfg.mask.mask_source, cfg.mask.mask_path = masks.VIDEO, str(tmp_path / 'matte.mp4')
    def extract(**kwargs):
        Path(kwargs['output_dir']).mkdir(parents=True, exist_ok=True)
        Image.new('L', (4, 4), 255).save(Path(kwargs['output_dir']) / '000001.jpg')
    monkeypatch.setattr('vibewarp.video.input.extract_frames', extract)
    with pytest.raises(ValueError, match='no frame for render frame 2'):
        masks.materialize(cfg, video_frames, run)


def test_rvm_materialization_matches_the_direct_cached_mask_bytes(tmp_path):
    from vibewarp.video import raw_processing as rp
    cfg, run, video_frames = _run(tmp_path, frames=(0,))
    alpha = tmp_path / 'alpha'
    alpha.mkdir()
    np.save(alpha / '000000.npy', np.linspace(0, 1, 16, dtype=np.float16).reshape(4, 4))
    cache = SimpleNamespace(alpha=alpha)
    masks.materialize(cfg, video_frames, run, cache)
    cfg.mask.background, cfg.mask.background_source = 'color', 'black'
    direct = rp.apply_cached_mask(Image.new('RGB', (8, 8), 'white'), cfg, cache, 0, str(video_frames))
    from vibewarp.core.diffusion import _apply_render_bg_mask
    via_folder = _apply_render_bg_mask(Image.new('RGB', (8, 8), 'white'), 0, cfg, str(video_frames))
    assert np.array_equal(np.asarray(direct), np.asarray(via_folder))


def test_nothing_is_materialized_without_masking(tmp_path):
    cfg, run, video_frames = _run(tmp_path)
    cfg.mask.use_background_mask = False
    assert masks.materialize(cfg, video_frames, run) is None
    assert not Path(masks.run_mask_folder(video_frames)).exists()


@pytest.mark.parametrize('source', [masks.VIDEO, masks.FOLDER])
@pytest.mark.parametrize('render,assembly', [(True, False), (False, True)])
def test_user_mask_sources_never_need_rvm_or_cuda(source, render, assembly):
    from vibewarp.video.raw_processing import enabled, needs_alpha
    cfg = RunConfig()
    cfg.mask.mask_source = source
    cfg.mask.use_background_mask = render
    cfg.video_assembly.use_background_mask_video = assembly
    assert not needs_alpha(cfg) and not enabled(cfg)
    cfg.mask.mask_source = masks.RVM
    assert needs_alpha(cfg)  # assembly-only masking also generates RVM masks now
    cfg.mask.mask_source = source
    cfg.mask.rvm_enabled = True  # masked tint is RVM-only
    assert needs_alpha(cfg)


def test_mask_source_validation(tmp_path):
    from vibewarp.config_io import validate_config
    def errors(cfg):
        return {e['path']: e['message'] for e in validate_config(cfg, require_inputs=False, check_paths=True)}
    cfg = RunConfig()
    cfg.mask.use_background_mask, cfg.mask.mask_source = True, masks.FOLDER
    assert 'mask.mask_path' in errors(cfg)
    cfg.mask.mask_path = str(tmp_path / 'missing')
    assert 'does not exist' in errors(cfg)['mask.mask_path']
    cfg.mask.mask_path = str(tmp_path)
    assert 'mask.mask_path' not in errors(cfg)
    cfg.mask.mask_source = masks.VIDEO
    assert 'Mask video does not exist' in errors(cfg)['mask.mask_path']
    cfg.mask.mask_source = 'init_video'
    assert 'mask.mask_source' in errors(cfg)
    cfg.mask.mask_source, cfg.mask.use_background_mask = masks.FOLDER, False
    cfg.mask.mask_path = ''
    assert 'mask.mask_path' not in errors(cfg)  # unused unless some stage masks


def test_schema_puts_mask_source_and_path_next_to_the_mask_switch():
    from vibewarp.config_io import config_schema
    props = config_schema()['properties']['mask']['properties']
    assert props['mask_source']['choices'] == list(masks.MASK_SOURCES)
    for name in ('mask_source', 'mask_path'):
        assert props[name]['tier'] == 'render'
        assert props[name]['group'] == props['use_background_mask']['group']


def test_assembly_reads_materialized_png_masks_and_warns_when_there_are_none(tmp_path, capsys, monkeypatch):
    from vibewarp.video.output import apply_video_mask, create_video
    cfg, run, video_frames = _run(tmp_path, frames=(0,))
    folder = Path(masks.run_mask_folder(video_frames))
    folder.mkdir()
    Image.new('L', (8, 8), 0).save(folder / '000001.png')
    out = apply_video_mask(Image.new('RGB', (8, 8), 'white'), 0, 'color', 'black', str(video_frames))
    assert out.getpixel((4, 4)) == (0, 0, 0)
    (folder / '000001.png').unlink()
    Image.new('RGB', (8, 8), 'white').save(run / 'clip(0)_000000.png')

    # Exercise mask handling without depending on an installed FFmpeg or codec.
    commands = []
    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        return SimpleNamespace(returncode=0, stderr='',
                               stdout='libx264' if '-encoders' in cmd else '')
    monkeypatch.setattr('vibewarp.video.input._find_ffmpeg', lambda: 'ffmpeg')
    monkeypatch.setattr('vibewarp.video.output.subprocess.run', fake_run)

    output_path = str(run / 'clip.mp4')
    assert create_video(frames_dir=str(run), output_path=output_path, batch_name='clip',
                        img_format='png', blend_mode='None', use_background_mask_video=True,
                        video_frames_folder=str(video_frames), keep_audio=False) == output_path
    assert 'has no masks' in capsys.readouterr().out
    assert [cmd[-1] for cmd in commands if '-c:v' in cmd] == [output_path]
    with Image.open(run / 'blend_temp' / '000000.png') as frame:
        assert frame.getpixel((4, 4)) == (255, 255, 255)


@pytest.mark.parametrize('raw,source,path', [
    ({}, masks.RVM, ''),
    ({'mask_source': 'init_video'}, masks.RVM, ''),
    ({'mask_source': 'mask_video', 'mask_video_path': 'C:/matte.mp4'}, masks.VIDEO, 'C:/matte.mp4'),
    ({'mask_source': 'mask_video'}, masks.RVM, ''),
])
def test_notebook_mask_source_import(tmp_path, raw, source, path):
    import json
    from vibewarp.settings import load_warpfusion_settings
    settings = tmp_path / 'settings.txt'
    settings.write_text(json.dumps(raw))
    mask = load_warpfusion_settings(str(settings))['mask']
    assert (mask['mask_source'], mask['mask_path']) == (source, path)
