import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.gpu_validation import (compare_jobs, deep_merge, expand_tests,
                                  export_final_frames, image_metrics,
                                  load_base_config, _latest_paired_dir,
                                  _paired_test_id)
from tools.notebook_reference import (build_headless_notebook,
                                      to_notebook_frame_range,
                                      to_notebook_settings,
                                      verify_applied_settings)
from vibewarp.core.model_loader import _cache_path


def test_deep_merge_preserves_siblings():
    base = {'diffusion': {'steps': 25, 'seed': 1}, 'frame_range': [0, 4]}
    merged = deep_merge(base, {'diffusion': {'steps': 8}})
    assert merged == {'diffusion': {'steps': 8, 'seed': 1}, 'frame_range': [0, 4]}
    assert base['diffusion']['steps'] == 25


def test_model_cache_separates_attention_variants(tmp_path):
    checkpoint = str(tmp_path / 'model.safetensors')
    normal = _cache_path(checkpoint, None, True)
    parity = _cache_path(checkpoint, None, True, 'parity-manual-attention')
    assert normal != parity


def test_model_cache_lives_with_install_not_checkpoint(tmp_path):
    # The pickle is code-version specific, so it must live with the install,
    # never in the shared checkpoint/models directory.
    checkpoint = str(tmp_path / 'checkpoints' / 'model.safetensors')
    path = _cache_path(checkpoint, None, True)
    assert '.vibewarp_cache' in path
    assert str(tmp_path) not in path


def test_model_cache_key_includes_code_fingerprint(monkeypatch):
    # A changed vendored-model fingerprint must produce a different cache path
    # so an updated install never reuses an incompatible pickle.
    import vibewarp.core.model_loader as ml
    checkpoint = 'anywhere/model.safetensors'
    monkeypatch.setattr(ml, '_code_fingerprint', lambda: 'aaaaaaaa')
    before = ml._cache_path(checkpoint, None, True)
    monkeypatch.setattr(ml, '_code_fingerprint', lambda: 'bbbbbbbb')
    after = ml._cache_path(checkpoint, None, True)
    assert before != after


def test_hf_logging_quiet_by_default_verbose_in_debug(monkeypatch):
    # The CLIP full-checkpoint load prints a big 'UNEXPECTED keys' table at
    # transformers' default verbosity; hide it unless VIBEWARP_DEBUG is set.
    hf = pytest.importorskip('transformers.utils.logging')
    import vibewarp.core.model_loader as ml
    monkeypatch.delenv('VIBEWARP_DEBUG', raising=False)
    ml._configure_hf_logging()
    assert hf.get_verbosity() == hf.ERROR
    monkeypatch.setenv('VIBEWARP_DEBUG', '1')
    ml._configure_hf_logging()
    assert hf.get_verbosity() == hf.INFO


def test_image_metrics_exact_and_changed(tmp_path):
    a = tmp_path / 'a.png'
    b = tmp_path / 'b.png'
    c = tmp_path / 'c.png'
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(a)
    Image.fromarray(pixels).save(b)
    pixels[0, 0] = 255
    Image.fromarray(pixels).save(c)
    assert image_metrics(a, b)['exact'] is True
    changed = image_metrics(a, c)
    assert changed['exact'] is False
    assert changed['max_abs'] == 1.0


def test_compare_jobs_enforces_thresholds(tmp_path):
    for side in ('left', 'right'):
        folder = tmp_path / side / 'artifacts'
        folder.mkdir(parents=True)
        Image.new('RGB', (2, 2), 'black').save(folder / '000.png')
    result = compare_jobs(tmp_path / 'left', tmp_path / 'right',
                          'artifacts/*.png', {'exact': True})
    assert result['status'] == 'pass'
    Image.new('RGB', (2, 2), 'white').save(
        tmp_path / 'right' / 'artifacts' / '000.png')
    result = compare_jobs(tmp_path / 'left', tmp_path / 'right',
                          'artifacts/*.png', {'max_abs': 0.0})
    assert result['status'] == 'fail'


def test_compare_jobs_accepts_distinct_artifact_globs(tmp_path):
    left = tmp_path / 'left' / 'artifacts' / 'run' / '0'
    right = tmp_path / 'right' / 'artifacts' / 'reference'
    left.mkdir(parents=True)
    right.mkdir(parents=True)
    Image.new('RGB', (2, 2), 'black').save(left / 'run.png')
    Image.new('RGB', (2, 2), 'black').save(right / 'reference.jpg')
    result = compare_jobs(
        tmp_path / 'left', tmp_path / 'right', 'artifacts/run/*/*.png',
        {'exact': True}, right_pattern='artifacts/reference/*.jpg')
    assert result['status'] == 'pass'


def test_example_manifest_lists_required_parity_cases():
    path = Path(__file__).parents[1] / 'configs' / 'gpu_validation.example.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    ids = {job['id'] for job in manifest['jobs']}
    assert {'empty-cn-stock', 'empty-cn-patched', 'controlnet-annotators',
            'sample-lcm', 'softcap', 'background-mask', 'captioning',
            'ip-adapter', 'animatediff', 'sdxl'} <= ids


def test_public_manifests_are_valid_and_sanitized():
    configs = Path(__file__).parents[1] / 'configs'
    validation = json.loads(
        (configs / 'gpu_validation.example.json').read_text(encoding='utf-8'))
    parity = json.loads(
        (configs / 'gpu_parity.example.json').read_text(encoding='utf-8'))

    assert validation['jobs']
    assert parity['tests']
    published = json.dumps([validation, parity])
    assert r'C:\code\warp' not in published
    assert r'C:\models' not in published


def test_load_base_config_accepts_vibewarp_saved_settings(tmp_path):
    saved = tmp_path / 'saved.txt'
    saved.write_text(json.dumps({
        'sd_checkpoint_path': 'model.safetensors',
        'diffusion_steps': 20,
        'video_width': 384,
    }), encoding='utf-8')
    loaded = load_base_config({
        'base_config': str(saved),
        'base_config_kind': 'vibewarp',
    }, tmp_path)
    assert loaded['diffusion']['steps'] == 20
    assert loaded['video']['width'] == 384


def test_notebook_settings_translation_disables_controlnet():
    translated = to_notebook_settings({
        'diffusion': {'steps': 20, 'cfg_scale': 5, 'sampler': 'sample_euler'},
        'controlnet': {'enabled': False, 'models': {'should_not_survive': {}}},
    }, {'notebook_only_key': 123, 'sampler': 'old'})
    assert translated['notebook_only_key'] == 123
    assert translated['sampler'] == 'sample_euler'
    assert translated['model_version'] == 'control_multi'
    assert translated['steps_schedule'] == {0: 20}
    assert translated['cfg_scale_schedule'] == [5]
    assert translated['width_height'] == [512, 512]
    assert json.loads(translated['controlnet_multimodel']) == {}


def test_notebook_settings_reuses_resolved_controlnet_directory(tmp_path):
    model_dir = tmp_path / 'ControlNet'
    translated = to_notebook_settings({
        'controlnet': {
            'enabled': True,
            'models': {
                'depth': {'path': str(model_dir / 'depth.pth')},
                'tile': {'path': str(model_dir / 'tile.pth')},
            },
        },
    })
    assert translated['controlnet_models_dir'] == str(model_dir.resolve())


def test_notebook_settings_schedules_never_scalar():
    # The notebook GUI loader raises on bare scalars and silently keeps the
    # previous widget value (cfg 5.0 ran as [15] before this was fixed).
    translated = to_notebook_settings({
        'diffusion': {'cfg_scale': 5.0, 'cfg_scale_schedule': 5.0,
                      'style_strength': 0.8},
    })
    assert translated['cfg_scale_schedule'] == [5.0]
    assert translated['style_strength_schedule'] == [0.8]
    listed = to_notebook_settings({
        'diffusion': {'cfg_scale_schedule': [2.0, 3.0]},
    })
    assert listed['cfg_scale_schedule'] == [2.0, 3.0]


def test_verify_applied_settings_flags_silent_gui_rejection():
    translated = {'cfg_scale_schedule': [5.0], 'sampler': 'sample_euler',
                  'seed': 42, 'controlnet_multimodel':
                  json.dumps({'control_sd15_depth': {'weight': 1.0}})}
    snapshot = {'cfg_scale_schedule': [15], 'sampler': 'sample_euler',
                'seed': 42, 'controlnet_multimodel':
                json.dumps({'control_sd15_depth': {'weight': 1.0}})}
    mismatches = verify_applied_settings(translated, snapshot)
    assert len(mismatches) == 1 and 'cfg_scale_schedule' in mismatches[0]
    snapshot['cfg_scale_schedule'] = [5]
    assert verify_applied_settings(translated, snapshot) == []
    # scalar requested == single-entry schedule executed
    assert verify_applied_settings({'clip_skip': 2}, {'clip_skip': 2.0}) == []


def test_notebook_settings_enable_animatediff_via_model_version():
    """The notebook has no animatediff on/off flag.

    It switches on `'animatediff' in model_version`. Emitting animatediff_* keys
    (as an earlier translation did) set NOTHING, so the reference would render
    frame-by-frame with no motion module -- a parity run comparing AnimateDiff
    against no-AnimateDiff, which silently "passes" as a giant fake regression.
    """
    translated = to_notebook_settings({
        'model_version': 'control_multi_v15',
        'animatediff': {'enabled': True, 'motion_module_path': 'C:/mm/v15.ckpt',
                        'batch_length': 16, 'context_length': 16},
    })
    assert translated['model_version'] == 'control_multi_animatediff'
    assert translated['motion_module_path'] == 'C:/mm/v15.ckpt'
    assert translated['batch_length'] == 16
    assert not [key for key in translated if key.startswith('animatediff_')]
    # Unset keys must still be pinned to VibeWarp's effective defaults, or each
    # side falls back to its own (notebook cell 47 defaults differ from ours).
    assert translated['reinject_stylized'] is True
    assert translated['lerp_reinject_stylized_frames'] == 0
    assert translated['looped_noise'] is False

    sdxl = to_notebook_settings({
        'model_version': 'control_multi_sdxl',
        'animatediff': {'enabled': True, 'motion_module_path': 'C:/mm/hsxl.st'},
    })
    assert sdxl['model_version'] == 'control_multi_animatediff_sdxl'

    # Disabled AnimateDiff must not touch model_version.
    off = to_notebook_settings({'model_version': 'control_multi_v15',
                                'animatediff': {'enabled': False}})
    assert off['model_version'] == 'control_multi'
    assert 'motion_module_path' not in off


def test_notebook_frame_range_matches_each_notebook_path():
    """VibeWarp's frame_range is inclusive; the notebook agrees with itself only in adiff.

    * do_run_adiff: `total = end - start + 1` -- INCLUSIVE, same as us, no adjustment.
    * frame-by-frame: `range(start, max_frames)` -- EXCLUSIVE, so it needs end + 1 to
      cover the same frames.

    Under AnimateDiff one extra frame takes total_length from 16 to 17, which flips
    max_batches 1 -> 2; the second batch then overwrites the overlap frames, so even the
    frames both sides rendered would not be comparable.
    """
    ad = {'enabled': True, 'motion_module_path': 'C:/mm.ckpt'}
    # AnimateDiff is inclusive like us -> unchanged.
    assert to_notebook_frame_range(
        {'frame_range': [0, 15], 'animatediff': ad}) == [0, 15]
    assert to_notebook_frame_range(
        {'frame_range': [4, 20], 'animatediff': ad}) == [4, 20]
    # The frame-by-frame path is exclusive -> needs one more.
    assert to_notebook_frame_range(
        {'frame_range': [0, 15], 'animatediff': {'enabled': False}}) == [0, 16]
    assert to_notebook_frame_range({'frame_range': [0, 15]}) == [0, 16]
    # [0, 0] is the "everything" sentinel -- do not touch it.
    assert to_notebook_frame_range(
        {'frame_range': [0, 0], 'animatediff': ad}) == [0, 0]
    assert to_notebook_frame_range({'frame_range': [0, 0]}) == [0, 0]


def test_notebook_settings_ignore_frame_keyed_noise():
    """frame_keyed_noise needs no notebook key -- the notebook already does it.

    do_run_adiff slices a single `big_noise` by absolute frame index, which IS
    frame-keyed noise. (It emits no such setting, so there is nothing to translate.)
    """
    translated = to_notebook_settings({
        'model_version': 'control_multi_v15',
        'animatediff': {'enabled': True, 'motion_module_path': 'C:/mm.ckpt',
                        'frame_keyed_noise': True},
    })
    assert 'frame_keyed_noise' not in translated
    assert translated['model_version'] == 'control_multi_animatediff'


def test_expand_tests_runs_notebook_first_for_animatediff():
    """AnimateDiff needs the reference's noise, so the reference must run first.

    The notebook's big_noise comes from an UNSEEDED cpu generator, so the two sides
    cannot arrive at the same noise on their own (two same-seed notebook runs differ
    at MAE 0.30). VibeWarp is fed the noise the reference dumped; that only works if
    the notebook job has already run.
    """
    manifest = {
        'defaults': {'base_config': 'x.txt'},
        'tests': [
            {'id': 'plain', 'settings': 'x.txt'},
            {'id': 'ad', 'settings': 'x.txt',
             'overrides': {'animatediff': {'enabled': True}}},
        ],
    }
    jobs, _ = expand_tests(manifest)
    order = [job['id'] for job in jobs]
    # Non-AnimateDiff keeps the original order and takes no noise injection.
    assert order.index('plain-vibewarp') < order.index('plain-notebook')
    assert 'adiff_noise_from' not in jobs[order.index('plain-vibewarp')]
    # AnimateDiff flips it, and the vibewarp side declares where its noise comes from.
    assert order.index('ad-notebook') < order.index('ad-vibewarp')
    assert jobs[order.index('ad-vibewarp')]['adiff_noise_from'] == 'ad/notebook'


def test_expand_tests_builds_paired_jobs_and_comparison():
    manifest = {
        'defaults': {'thresholds': {'mae': 0.03}},
        'tests': [{'id': 'hyper-sd-cn',
                   'settings': 'refs/examples/hyper_sd_8step_test.txt',
                   'overrides': {'controlnet': {'enabled': False}}}],
    }
    jobs, comparisons = expand_tests(manifest)
    by_id = {job['id']: job for job in jobs}
    vibewarp = by_id['hyper-sd-cn-vibewarp']
    notebook = by_id['hyper-sd-cn-notebook']
    assert vibewarp['dir'] == 'hyper-sd-cn/vibewarp'
    assert vibewarp['env']['VIBEWARP_PARITY_MODE'] == '1'
    assert notebook['dir'] == 'hyper-sd-cn/notebook'
    assert notebook['reference_notebook'] is True
    assert notebook['settings_template'] == vibewarp['base_config']
    assert vibewarp['batch_name'] == notebook['batch_name'] == 'hyper-sd-cn'
    assert vibewarp['export_frames'] and notebook['export_frames']
    comp = comparisons[-1]
    assert comp['id'] == 'hyper-sd-cn'
    assert comp['pattern'] == 'frame_*.*'
    assert comp['thresholds'] == {'mae': 0.03}


def test_expand_tests_carries_single_controlnet_filter():
    jobs, _ = expand_tests({
        'tests': [{'id': 'depth-only', 'settings': 'base.txt',
                   'only_controlnets': ['control_sd15_depth']}],
    })
    assert all(job['only_controlnets'] == ['control_sd15_depth'] for job in jobs)


def test_timestamped_pair_directory_discovery(tmp_path):
    job = {'dir': 'parity-basic-no-cn/vibewarp'}
    assert _paired_test_id(job) == 'parity-basic-no-cn'
    older = tmp_path / '2026-07-11_14-00-00-000001_parity-basic-no-cn'
    newer = tmp_path / '2026-07-11_14-00-01-000001_parity-basic-no-cn'
    unrelated = tmp_path / '2026-07-11_14-00-02-000001_parity-basic-cn'
    for path in (newer, unrelated, older):
        path.mkdir()
    assert _latest_paired_dir(tmp_path, 'parity-basic-no-cn') == newer


def test_export_final_frames_selects_newest_run_and_skips_debug(tmp_path):
    art = tmp_path / 'artifacts' / 'test' / '1'
    art.mkdir(parents=True)
    debug = tmp_path / 'artifacts' / 'test' / 'controlnetDebug'
    debug.mkdir()
    black = Image.new('RGB', (2, 2), 'black')
    black.save(art / 'test(0)_000000.png')
    black.save(art / 'test(1)_000000.png')
    black.save(art / 'test(1)_000001.png')
    black.save(debug / 'test(1)_control_sd15_depth_000000.jpg')
    black.save(art / '_warped_000001.png')
    (tmp_path / 'frame_999999.png').write_bytes(b'stale')
    count = export_final_frames(tmp_path, 'test')
    assert count == 2
    exported = sorted(p.name for p in tmp_path.glob('frame_*.*'))
    assert exported == ['frame_000000.png', 'frame_000001.png']


def test_build_headless_notebook_patches_boundaries(tmp_path):
    cells = [{'cell_type': 'code', 'metadata': {}, 'outputs': [],
              'execution_count': None, 'source': ['pass\n']} for _ in range(56)]
    cells[18]['source'] = ["warp_installer.pull_repos(is_colab)\n",
                           "outDirPath = os.path.join(root_path,'images_out')\n"]
    cells[20]['source'] = [
        'mp_drawing = mp.solutions.drawing_utils\n',
        'mp_drawing_styles = mp.solutions.drawing_styles\n',
        'mp_face_detection = mp.solutions.face_detection\n',
        'mp_face_mesh = mp.solutions.face_mesh\n',
        'mp_face_connections = mp.solutions.face_mesh_connections.FACEMESH_TESSELATION\n',
        'mp_hand_connections = mp.solutions.hands_connections.HAND_CONNECTIONS\n',
        'mp_body_connections = mp.solutions.pose_connections.POSE_CONNECTIONS\n',
        'DrawingSpec = mp.solutions.drawing_styles.DrawingSpec\n',
        'PoseLandmark = mp.solutions.drawing_styles.PoseLandmark\n',
        'right_iris_draw = None\n',
        'right_eye_draw = None\n',
        'right_eyebrow_draw = None\n',
        'left_iris_draw = None\n',
        'left_eye_draw = None\n',
        'left_eyebrow_draw = None\n',
        'mouth_draw = None\n',
        'head_draw = None\n',
        'face_connection_spec = {}\n',
        'iris_landmark_spec = {468: right_iris_draw, 473: left_iris_draw}\n',
    ]
    cells[21]['source'] = ["video_init_path = 'old.mp4' #@param {'type':'string'}\n"]
    cells[22]['source'] = [
        '        batch_size = sd_batch_size\n',
        '        cond_in = torch.cat([uncond, cond])\n',
        '                    control = loaded_controlnets[key].cuda()(x=x_noisy, hint=cond_hint,\n',
        '                                                                timesteps=t, context=cond_txt, y=y)\n',
        'x = x * sigmas[0]\n',
        "print('xi', xi.shape)\n",
    ]
    cells[53]['source'] = ["settings_path = '-1'\n"]
    cells[55]['source'] = [
        'frame_range = [0,15]\n',
        '  user_settings = get_settings_from_gui(user_settings_keys, guis)\n',
        '  for key in user_settings.keys():\n',
        '    globals()[key] = user_settings[key]\n',
    ]
    source = tmp_path / 'source.ipynb'
    destination = tmp_path / 'headless.ipynb'
    settings = tmp_path / 'settings.json'
    settings.write_text(json.dumps({'video_init_path': 'paired.mp4'}), encoding='utf-8')
    source.write_text(json.dumps({'cells': cells, 'metadata': {}}), encoding='utf-8')
    build_headless_notebook(source, destination, settings,
                            tmp_path / 'out', [0, 4], tmp_path / 'reference')
    built = json.loads(destination.read_text(encoding='utf-8'))
    combined = '\n'.join(''.join(c['source']) for c in built['cells'])
    assert 'pull_repos disabled' in combined
    assert "hasattr(mp, 'solutions')" in combined
    assert 'frame_range = [0, 4]' in combined
    assert combined.count('diag_rng.txt') == 2
    assert combined.count('diag_model_inputs.json') == 1
    assert combined.count('diag_controlnet.json') == 1
    assert repr(str(tmp_path / 'settings.json')) in combined
    assert "video_init_path = 'paired.mp4'" in combined
