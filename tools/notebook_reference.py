"""Build and execute a headless WarpFusion reference notebook.

This deliberately uses the original notebook cells and reference installation;
it only patches environment/setup boundaries needed for unattended execution.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def to_notebook_settings(config: dict, template: dict | None = None) -> dict:
    """Translate a resolved VibeWarp config to notebook settings keys."""
    out: dict = dict(template or {})
    model_version = config.get('model_version', 'control_multi_v15')
    notebook_model_versions = {
        'control_multi_v15': 'control_multi',
        'control_multi_v21': 'control_multi_v2',
    }
    # The notebook has no "enable animatediff" flag: it switches on the presence
    # of 'animatediff' in model_version (see `if 'animatediff' in model_version`
    # throughout cell 20). Emitting animatediff_* keys, as we used to, set nothing
    # at all -- the reference would have rendered frame-by-frame with the motion
    # module absent, i.e. a parity run comparing AnimateDiff against no-AnimateDiff.
    if config.get('animatediff', {}).get('enabled'):
        notebook_model_versions = {
            'control_multi_v15': 'control_multi_animatediff',
            'control_multi_v21': 'control_multi_animatediff',
            'control_multi_sdxl': 'control_multi_animatediff_sdxl',
        }
    out.update({
        'model_path': config.get('sd_checkpoint_path', ''),
        'vae_ckpt': config.get('vae_ckpt_path', ''),
        'model_version': notebook_model_versions.get(model_version, model_version),
        'batch_name': config.get('batch_name', 'reference'),
        'text_prompts': {str(k): [v] if isinstance(v, str) else v
                         for k, v in config.get('text_prompts', {}).items()},
        'negative_prompts': {str(k): [v] if isinstance(v, str) else v
                             for k, v in config.get('negative_prompts', {}).items()},
    })
    # An empty lora_dir crashes the notebook's lora cell (os.makedirs('')).
    # Emit only real paths; otherwise the notebook keeps its own default dir
    # (harmless: it only lists loras there, application needs prompt tags).
    if config.get('lora_dir'):
        out['lora_dir'] = config['lora_dir']
    else:
        out.pop('lora_dir', None)
    video = config.get('video', {})
    out.update({
        'video_init_path': video.get('video_init_path', ''),
        'extract_nth_frame': video.get('extract_nth_frame', 1),
        'width': video.get('width', 512), 'height': video.get('height', 512),
        'width_height': [video.get('width', 512), video.get('height', 512)],
        'save_img_format': video.get('save_img_format', 'png'),
    })
    diffusion = config.get('diffusion', {})
    # The notebook GUI loader (load_settings) json-dumps lists/dicts into text
    # widgets but raises on bare scalars, silently keeping the widget's old
    # value. Every schedule therefore MUST be emitted as a list or dict.
    def as_schedule(value, default):
        if value is None:
            value = default
        return value if isinstance(value, (list, dict)) else [value]
    out.update({
        'steps_schedule': diffusion.get('steps_schedule') or
                          {0: diffusion.get('steps', 25)},
        'cfg_scale_schedule': as_schedule(
            diffusion.get('cfg_scale_schedule') or diffusion.get('cfg_scale'), 7.5),
        'style_strength_schedule': as_schedule(
            diffusion.get('style_strength_schedule'),
            diffusion.get('style_strength', 0.65)),
        'sampler': diffusion.get('sampler', 'sample_euler'),
        'seed': diffusion.get('seed', -1),
        # Notebook derives use_predicted_noise/fixed_code from this dropdown.
        'noise_mode': {'default': 'random', 'fixed': 'fixed',
                       'reconstructed': 'reconstructed'}.get(
                           diffusion.get('noise_mode', 'default'), 'random'),
        'fixed_code': diffusion.get('noise_mode') == 'fixed',
        'code_randomness': diffusion.get('code_randomness', 0.0),
        'clip_skip': diffusion.get('clip_skip', 1),
        'use_karras_noise': diffusion.get('use_karras_noise', True),
        'dynamic_thresh': diffusion.get('dynamic_thresh', 30.0),
        'do_softcap': diffusion.get('do_softcap', False),
        'softcap_thresh': diffusion.get('softcap_thresh', 0.9),
        'softcap_q': diffusion.get('softcap_q', 1.0),
    })
    for section in ('flow', 'warp', 'brightness', 'color', 'mask'):
        out.update(config.get(section, {}))
    vae = config.get('vae', {})
    out.update(vae)
    cn = config.get('controlnet', {})
    models = {}
    cn_paths = [entry.get('path', '') for entry in cn.get('models', {}).values()
                if entry.get('path')]
    if cn_paths:
        cn_dirs = [os.path.dirname(os.path.abspath(path)) for path in cn_paths]
        out['controlnet_models_dir'] = os.path.commonpath(cn_dirs)
    if cn.get('enabled', True):
        for key, entry in cn.get('models', {}).items():
            models[key] = {
                'weight': entry.get('weight', 1.0),
                'start': entry.get('start', 0.0),
                'end': entry.get('end', 1.0),
                'preprocess': entry.get('annotator') or 'global',
                'mode': 'global',
                'detect_resolution': entry.get('detect_resolution', -1),
                'source': entry.get('source', 'global'),
            }
    out['controlnet_multimodel'] = json.dumps(models)
    out['controlnet_multimodel_mode'] = cn.get('mode', 'internal')
    out['cond_image_src'] = cn.get('cond_image_src', 'init')
    out['normalize_cn_weights'] = cn.get('normalize_weights', False)

    ad = config.get('animatediff', {})
    if ad.get('enabled'):
        # Emit the EFFECTIVE values, not just the keys the job happened to set.
        # An unset key means VibeWarp falls back to AnimateDiffConfig's default
        # while the notebook falls back to cell 47's -- and those differ (e.g.
        # notebook looped_noise=True vs ours False). Pinning both sides to the
        # same numbers is the whole point of the paired harness.
        from vibewarp.config import AnimateDiffConfig
        effective = AnimateDiffConfig(**ad)
        # motion_module_path is a cell-23 #@param (rewritten from early_settings);
        # the rest are plain globals with GUI widgets, so they arrive via the
        # settings loader. All are bare keys -- there is no animatediff_ prefix.
        out['motion_module_path'] = effective.motion_module_path
        for key in ('batch_length', 'batch_overlap', 'context_length',
                    'context_overlap', 'looped_noise', 'overlap_stylized',
                    'blend_batch_outputs', 'reinject_stylized',
                    'lerp_reinject_stylized_frames', 'pingpong_noise',
                    'batched_adiff_rec_noise', 'rec_sliding_ctx', 'stop_early'):
            out[key] = getattr(effective, key)
        # frame_keyed_noise has no notebook key, and needs none: the notebook slices a
        # single big_noise by absolute frame index, which is the same thing. Its noise
        # VALUES are unreproducible (nothing seeds the CPU generator before big_noise),
        # so an image-level AnimateDiff comparison is only meaningful when both sides
        # are fed the same noise -- see VIBEWARP_ADIFF_NOISE / _adiff_injected_noise.
    ipa = config.get('ipadapter', {})
    out.update({f'ipadapter_{k}': v for k, v in ipa.items() if k != 'models'})
    captions = config.get('captions', {})
    out.update(captions)
    scene = config.get('scene', {})
    out.update(scene)
    return out


def to_notebook_frame_range(config: dict) -> list[int]:
    """Translate VibeWarp's frame_range to the one the notebook needs.

    VibeWarp's frame_range is INCLUSIVE at both ends (0-15 = sixteen frames), matching
    what the two numbers in the UI say. The notebook is inconsistent with ITSELF:

    * do_run_adiff: `end_frame = args.max_frames`, `total = end - start + 1` -- INCLUSIVE,
      so it agrees with us and needs no adjustment.
    * frame-by-frame: `range(start_frame, max_frames)` -- EXCLUSIVE, so it renders one
      frame FEWER than we do and needs `end + 1` to cover the same set.

    Getting this wrong is not cosmetic. Under AnimateDiff a single extra frame takes
    total_length from 16 to 17, which flips max_batches from 1 to 2: the notebook then
    runs a second batch over the tail and OVERWRITES the overlap frames, so even the
    frames both sides rendered would not be comparable.
    """
    frame_range = list(config.get('frame_range') or [0, 0])
    if len(frame_range) != 2:
        return frame_range
    start, end = frame_range
    if end <= start:
        return frame_range              # [0, 0] = "everything"; leave the sentinel alone
    if config.get('animatediff', {}).get('enabled'):
        return [start, end]             # do_run_adiff is inclusive, like us
    return [start, end + 1]             # the frame-by-frame path is exclusive


VERIFIED_KEYS = (
    'cfg_scale_schedule', 'style_strength_schedule', 'steps_schedule',
    'flow_blend_schedule', 'sampler', 'clip_skip', 'use_karras_noise',
    'fixed_code', 'width', 'height', 'do_softcap', 'dynamic_thresh',
    'text_prompts', 'negative_prompts', 'model_path', 'noise_mode',
    # model_version is the AnimateDiff on/off switch -- if it fails to apply, the
    # reference silently renders frame-by-frame and parity measures nothing.
    'model_version',
    'batch_length', 'batch_overlap', 'context_length', 'context_overlap',
    'looped_noise', 'overlap_stylized', 'blend_batch_outputs',
    'reinject_stylized', 'lerp_reinject_stylized_frames',
)


def _norm_value(value):
    if isinstance(value, dict):
        return {str(k): _norm_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_norm_value(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value


def verify_applied_settings(translated: dict, snapshot: dict) -> list[str]:
    """Compare the settings the notebook SAVED against what we asked for.

    The GUI loader swallows per-key exceptions and keeps the previous widget
    value, so a translation format error renders with defaults while looking
    successful. The saved snapshot is the ground truth of the executed run.
    """
    mismatches = []
    for key in VERIFIED_KEYS:
        if key not in translated or key not in snapshot:
            continue
        want, got = translated[key], snapshot[key]
        # scalars count as single-entry schedules
        if isinstance(want, (int, float)) and not isinstance(want, bool) \
                and isinstance(got, list):
            want = [want]
        if _norm_value(want) != _norm_value(got):
            mismatches.append(f'{key}: requested {want!r}, notebook ran {got!r}')
    seed = translated.get('seed')
    if seed not in (None, -1) and 'seed' in snapshot \
            and _norm_value(seed) != _norm_value(snapshot['seed']):
        mismatches.append(f"seed: requested {seed!r}, notebook ran {snapshot['seed']!r}")
    want_cn = translated.get('controlnet_multimodel')
    got_cn = snapshot.get('controlnet_multimodel')
    if want_cn is not None and got_cn is not None:
        if isinstance(want_cn, str):
            want_cn = json.loads(want_cn)
        if isinstance(got_cn, str):
            got_cn = json.loads(got_cn)
        if sorted(want_cn) != sorted(got_cn):
            mismatches.append(
                f'controlnet_multimodel models: requested {sorted(want_cn)}, '
                f'notebook ran {sorted(got_cn)}')
        else:
            for name, entry in want_cn.items():
                for field in ('weight', 'start', 'end', 'detect_resolution'):
                    if field in entry and field in got_cn[name] \
                            and _norm_value(entry[field]) != _norm_value(got_cn[name][field]):
                        mismatches.append(
                            f'{name}.{field}: requested {entry[field]!r}, '
                            f'notebook ran {got_cn[name][field]!r}')
    return mismatches


def build_headless_notebook(source: Path, destination: Path, settings: Path,
                            output: Path, frame_range: list[int],
                            workdir: Path | None = None,
                            early_settings: dict | None = None) -> None:
    notebook = json.loads(source.read_text(encoding='utf-8'))
    cells = notebook['cells'][18:56]
    settings_data = (early_settings if early_settings is not None else
                     json.loads(settings.read_text(encoding='utf-8')))
    # Video extraction and model loading happen before the GUI settings loader.
    # Rewrite Colab parameter assignments in the generated copy so these early
    # side effects receive the same paired-case settings.
    substituted: set[str] = set()
    for cell in cells:
        if cell.get('cell_type') != 'code':
            continue
        source_text = ''.join(cell.get('source', []))
        for key, value in settings_data.items():
            if not re.match(r'^[A-Za-z_]\w*$', key):
                continue
            pattern = rf'(?m)^(\s*){re.escape(key)}\s*=\s*([^\n]*?)(\s*#@param[^\n]*)$'
            source_text, count = re.subn(
                pattern,
                lambda match, k=key, v=value:
                    f"{match.group(1)}{k} = {v!r}{match.group(3)}",
                source_text)
            if count:
                substituted.add(key)
        cell['source'] = source_text.splitlines(keepends=True)
    # motion_module_path is the one AnimateDiff setting the notebook never writes
    # to its settings snapshot, so verify_applied_settings cannot cover it. If the
    # #@param rewrite silently misses, the notebook keeps its own hardcoded path
    # (a HotshotXL module, in the shipped copy) or downloads a default -- i.e. it
    # would render with the WRONG motion module and still look like a clean run.
    if settings_data.get('motion_module_path') and 'motion_module_path' not in substituted:
        raise RuntimeError('could not patch notebook motion_module_path')
    # save_img_format is a PLAIN global (cell 47 assigns it twice, 'png' then 'jpg' --
    # the last one wins). It has no #@param and no GUI widget, so the rewrite above
    # cannot reach it and it never lands in the settings snapshot either. The notebook
    # therefore always writes JPEG, then re-reads those lossy frames as the stylized
    # overlap init AND as the reinject_stylized latents -- so the JPEG loss feeds back
    # into the next batch and COMPOUNDS. Measured: batch0 MAE 0.025, batch1 0.070,
    # batch2 0.119 against our lossless PNG. Force the two sides onto one format.
    image_format = settings_data.get('save_img_format')
    if image_format:
        format_pattern = r"(?m)^(\s*)save_img_format\s*=\s*'[^']*'"
        format_applied = 0
        for cell in cells:
            if cell.get('cell_type') != 'code':
                continue
            text = ''.join(cell['source'])
            text, count = re.subn(
                format_pattern,
                lambda match: f"{match.group(1)}save_img_format = {image_format!r}",
                text)
            if count:
                cell['source'] = text.splitlines(keepends=True)
                format_applied += count
        if not format_applied:
            raise RuntimeError('could not patch notebook save_img_format')
    # Setup cell: never install/update during validation and isolate output.
    setup = ''.join(cells[0]['source'])
    for installer_call in (
        'uninstall_pytorch', 'install_torch_windows', 'install_torch_colab',
        'pull_repos', 'install_dependencies_colab',
    ):
        setup = re.sub(
            rf'warp_installer\.{installer_call}\([^\n]*\)',
            f"print('Reference validation: {installer_call} disabled')", setup)
    if workdir is not None:
        setup = setup.replace('root_path = os.getcwd()', f'root_path = {str(workdir)!r}')
        setup = setup.replace('root_dir = os.getcwd()', f'root_dir = {str(workdir)!r}')
        setup = setup.replace('PROJECT_DIR = os.path.abspath(os.getcwd())',
                              f'PROJECT_DIR = {str(workdir)!r}')
    setup += (
        "\n# Compatibility for BasicSR with torchvision >= 0.17.\n"
        "import torchvision.transforms.functional as _tv_functional\n"
        "sys.modules.setdefault('torchvision.transforms.functional_tensor', _tv_functional)\n")
    setup = setup.replace("outDirPath = os.path.join(root_path,'images_out')",
                          f"outDirPath = {str(output)!r}")
    cells[0]['source'] = setup.splitlines(keepends=True)
    # mediapipe 0.10.33 removed the legacy solutions namespace used only by
    # optional face/hand guidance. Keep non-face reference cases executable;
    # those optional cases still fail loudly if they try to consume the Nones.
    imports_index = 20 - 18
    imports = ''.join(cells[imports_index]['source'])
    legacy_mp = '\n'.join([
        "if hasattr(mp, 'solutions'):",
        "  mp_drawing = mp.solutions.drawing_utils",
        "  mp_drawing_styles = mp.solutions.drawing_styles",
        "  mp_face_detection = mp.solutions.face_detection",
        "  mp_face_mesh = mp.solutions.face_mesh",
        "  mp_face_connections = mp.solutions.face_mesh_connections.FACEMESH_TESSELATION",
        "  mp_hand_connections = mp.solutions.hands_connections.HAND_CONNECTIONS",
        "  mp_body_connections = mp.solutions.pose_connections.POSE_CONNECTIONS",
        "else:",
        "  print('Reference validation: legacy mediapipe solutions unavailable; face/hand cases disabled')",
        "  mp_drawing = mp_drawing_styles = mp_face_detection = mp_face_mesh = None",
        "  mp_face_connections = mp_hand_connections = mp_body_connections = None",
    ])
    imports, count = re.subn(
        r"mp_drawing = mp\.solutions\.drawing_utils.*?mp_body_connections = mp\.solutions\.pose_connections\.POSE_CONNECTIONS",
        legacy_mp, imports, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError('could not patch legacy mediapipe compatibility block')
    drawing_pattern = (r"DrawingSpec = mp\.solutions\.drawing_styles\.DrawingSpec.*?"
                       r"iris_landmark_spec = \{468: right_iris_draw, 473: left_iris_draw\}")
    drawing_match = re.search(drawing_pattern, imports, flags=re.DOTALL)
    if not drawing_match:
        raise RuntimeError('could not patch legacy mediapipe drawing block')
    original_drawing = drawing_match.group(0)
    guarded_drawing = (
        "if hasattr(mp, 'solutions'):\n" +
        '\n'.join('  ' + line for line in original_drawing.splitlines()) +
        "\nelse:\n"
        "  DrawingSpec = PoseLandmark = None\n"
        "  right_iris_draw = right_eye_draw = right_eyebrow_draw = None\n"
        "  left_iris_draw = left_eye_draw = left_eyebrow_draw = None\n"
        "  mouth_draw = head_draw = None\n"
        "  face_connection_spec = {}\n"
        "  iris_landmark_spec = {}")
    imports = imports[:drawing_match.start()] + guarded_drawing + imports[drawing_match.end():]
    cells[imports_index]['source'] = imports.splitlines(keepends=True)
    # GUI cell: load the generated, notebook-native settings file.
    gui_index = 53 - 18
    gui = ''.join(cells[gui_index]['source'])
    gui, count = re.subn(
        r"settings_path\s*=\s*'-1'",
        lambda _match: f"settings_path = {str(settings)!r}", gui, count=1)
    if count != 1:
        raise RuntimeError('could not patch notebook settings_path')
    cells[gui_index]['source'] = gui.splitlines(keepends=True)
    # Capture the first CFG/UNet inputs. Hash float32-normalized tensors so
    # dtype metadata and numeric content can be compared across environments.
    cfg_probe = '''
        if not getattr(self, '_vibewarp_input_probed', False):
          import hashlib as _diag_hashlib, json as _diag_json
          def _diag_tensor(value):
            cpu = value.detach().float().cpu().contiguous()
            return {'shape': list(value.shape), 'dtype': str(value.dtype),
                    'sha256_f32': _diag_hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
                    'first8': [round(v, 6) for v in cpu.flatten()[:8].tolist()]}
          _diag = {'x': _diag_tensor(x_in), 'sigma': _diag_tensor(sigma_in),
                   'uc': _diag_tensor(uncond), 'c': _diag_tensor(cond),
                   'sd_batch_size': batch_size}
          # SDXL: the pooled/size vector drives style as hard as the crossattn does,
          # so a mismatch here shows up as "same composition, different look".
          # Read it off the conditioner (same place VibeWarp's probe reads it), since
          # the local `y` is not in scope at every injection point.
          _vec = getattr(getattr(sd_model, 'conditioner', None), 'vector_in', None)
          if _vec is not None:
            _diag['y'] = _diag_tensor(_vec)
          open(os.path.join(outDirPath, 'diag_model_inputs.json'), 'w').write(
              _diag_json.dumps(_diag, sort_keys=True, indent=2))
          self._vibewarp_input_probed = True
'''
    cfg_probes_applied = 0
    for cell in cells:
        if cell.get('cell_type') != 'code':
            continue
        text = ''.join(cell['source'])
        marker = '        batch_size = sd_batch_size\n'
        target = '        cond_in = torch.cat([uncond, cond])\n'
        marker_at = text.find(marker)
        target_at = text.find(target, marker_at if marker_at >= 0 else len(text))
        if marker_at >= 0 and target_at >= 0:
            insert_at = target_at + len(target)
            text = text[:insert_at] + cfg_probe + text[insert_at:]
            cell['source'] = text.splitlines(keepends=True)
            cfg_probes_applied += 1
            break
    if cfg_probes_applied != 1:
        raise RuntimeError('expected one CFG model-input probe')
    cn_target = '''                    control = loaded_controlnets[key].cuda()(x=x_noisy, hint=cond_hint,
                                                                timesteps=t, context=cond_txt, y=y)
'''
    cn_probe = '''                    if not globals().get('_vibewarp_cn_input_probed', False):
                      import hashlib as _cn_hashlib, json as _cn_json
                      def _cn_diag_tensor(value):
                        cpu = value.detach().float().cpu().contiguous()
                        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                                'sha256_f32': _cn_hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
                                'first8': [round(v, 6) for v in cpu.flatten()[:8].tolist()]}
                      _cn_diag = {'key': key, 'x': _cn_diag_tensor(x_noisy),
                                  'hint': _cn_diag_tensor(cond_hint),
                                  'timestep': _cn_diag_tensor(t),
                                  'context': _cn_diag_tensor(cond_txt),
                                  'residuals': [_cn_diag_tensor(value) for value in control]}
                      open(os.path.join(outDirPath, 'diag_controlnet.json'), 'w').write(
                          _cn_json.dumps(_cn_diag, sort_keys=True, indent=2))
                      torch.save(cond_hint.detach().cpu(),
                                 os.path.join(outDirPath, 'diag_controlnet_hint.pt'))
                      torch.save(cond_txt.detach().cpu(),
                                 os.path.join(outDirPath, 'diag_controlnet_context.pt'))
                      globals()['_vibewarp_cn_input_probed'] = True
'''
    cn_probes_applied = 0
    for cell in cells:
        if cell.get('cell_type') != 'code':
            continue
        text = ''.join(cell['source'])
        if cn_target in text:
            text = text.replace(cn_target, cn_target + cn_probe, 1)
            cell['source'] = text.splitlines(keepends=True)
            cn_probes_applied += 1
            break
    if cn_probes_applied != 1:
        raise RuntimeError('expected one ControlNet probe')
    # AnimateDiff noise probe. do_run_adiff draws `big_noise` ONCE for the whole
    # video, on CPU, and slices it by ABSOLUTE frame index -- and there is no
    # seed_everything anywhere before it, so it inherits whatever global RNG state
    # model construction left behind. Dump it (and save the tensor) so we can prove
    # whether it is seed-derived at all, and so VibeWarp can be fed the identical
    # noise to test the AnimateDiff math independently of the RNG stream.
    adiff_target = ("  big_noise = torch.randn((end_frame+1,4,H//8,W//8), "
                    "device='cpu').half()\n")
    adiff_probe = '''  if True:
    import hashlib as _ad_hashlib, json as _ad_json
    _ad_cpu = big_noise.detach().float().cpu().contiguous()
    open(os.path.join(outDirPath, 'diag_adiff_noise.json'), 'w').write(_ad_json.dumps({
        'shape': list(big_noise.shape), 'dtype': str(big_noise.dtype),
        'seed': args.seed,
        'sha256_f32': _ad_hashlib.sha256(_ad_cpu.numpy().tobytes()).hexdigest(),
        'first8': [round(v, 6) for v in _ad_cpu.flatten()[:8].tolist()],
    }, sort_keys=True, indent=2))
    torch.save(_ad_cpu, os.path.join(outDirPath, 'diag_adiff_noise.pt'))
'''
    # do_run_adiff over-counts its batches: max_batches = ceil(total_length / stride)
    # keeps going after the frames are covered. The last batch only needs to START by
    # total-batch_length, so the count should be ceil((total-batch_length)/stride)+1.
    # With 40 frames / length 16 / overlap 4, batch 2 already ends on frame 39, yet the
    # notebook runs a 4th batch of [36,37,38,39,39,39,...] -- frame 39 padded twelve
    # times -- and OVERWRITES frame 39 with a degraded render (MAE 0.25 vs ~0.11 for
    # its neighbours). Owner confirmed the notebook is wrong here and VibeWarp is right
    # (2026-07-13), so we correct the reference rather than replicate the bug; without
    # this the two sides render different frames and the tail is not comparable.
    batches_target = ("  max_batches = 1 if total_length==batch_length else "
                      "math.ceil(total_length/(batch_length-batch_overlap))\n")
    batches_fixed = ("  max_batches = 1 if total_length==batch_length else "
                     "math.ceil((total_length-batch_length)/(batch_length-batch_overlap))+1\n")
    batches_applied = 0
    for cell in cells:
        if cell.get('cell_type') != 'code':
            continue
        text = ''.join(cell['source'])
        if batches_target in text:
            text = text.replace(batches_target, batches_fixed, 1)
            cell['source'] = text.splitlines(keepends=True)
            batches_applied += 1
            break
    if 'animatediff' in str(settings_data.get('model_version', '')) \
            and batches_applied != 1:
        raise RuntimeError('could not patch notebook adiff max_batches')

    adiff_probes_applied = 0
    for cell in cells:
        if cell.get('cell_type') != 'code':
            continue
        text = ''.join(cell['source'])
        if adiff_target in text:
            text = text.replace(adiff_target, adiff_target + adiff_probe, 1)
            cell['source'] = text.splitlines(keepends=True)
            adiff_probes_applied += 1
            break
    # Only AnimateDiff runs reach do_run_adiff, so only they require the noise probe.
    if 'animatediff' in str(settings_data.get('model_version', '')) \
            and adiff_probes_applied != 1:
        raise RuntimeError('expected one AnimateDiff noise probe')
    # Diagnostic probes: record the first values of the initial noise so RNG
    # alignment with VibeWarp can be checked. do_run's stdout is swallowed by
    # the GUI's ipywidget Output (invisible to nbconvert), so write to a file
    # in the isolated output directory instead.
    probe = ("open(os.path.join(outDirPath, 'diag_rng.txt'), 'a').write("
             "'%s frame=%d x8=%s\\n' % ({tag!r}, frame_num, "
             "[round(v, 5) for v in {var}.flatten()[:8].tolist()]))")
    probes_applied = 0
    for cell in cells:
        if cell.get('cell_type') != 'code':
            continue
        text = ''.join(cell['source'])
        if 'x = x * sigmas[0]\n' in text:
            text = text.replace(
                'x = x * sigmas[0]\n',
                'x = x * sigmas[0]; ' + probe.format(tag='txt2img', var='x') + '\n', 1)
            probes_applied += 1
        if "print('xi', xi.shape)\n" in text:
            text = text.replace(
                "print('xi', xi.shape)\n",
                "print('xi', xi.shape); " + probe.format(tag='img2img', var='noise') + '\n', 1)
            probes_applied += 1
        if probes_applied:
            cell['source'] = text.splitlines(keepends=True)
            break
    if probes_applied != 2:
        raise RuntimeError(f'expected 2 RNG probes, applied {probes_applied}')
    # Render cell: bound the exact same frame interval as VibeWarp.
    run_index = 55 - 18
    run = ''.join(cells[run_index]['source'])
    run, count = re.subn(r'frame_range\s*=\s*\[[^\]]*\]',
                         f'frame_range = {frame_range!r}', run, count=1)
    if count != 1:
        raise RuntimeError('could not patch notebook frame_range')
    cells[run_index]['source'] = run.splitlines(keepends=True)
    notebook['cells'] = cells
    notebook['metadata']['kernelspec'] = {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook), encoding='utf-8')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--notebook', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-python', type=Path, required=True)
    parser.add_argument('--settings-template', type=Path)
    parser.add_argument('--timeout', type=int, default=21600)
    args = parser.parse_args(argv)
    args.notebook = args.notebook.resolve()
    args.config = args.config.resolve()
    args.workdir = args.workdir.resolve()
    args.output = args.output.resolve()
    args.reference_python = args.reference_python.resolve()
    if args.settings_template:
        args.settings_template = args.settings_template.resolve()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=True)
    batch_output = args.output / config.get('batch_name', 'reference')
    if batch_output.exists():
        shutil.rmtree(batch_output)
    template = None
    if args.settings_template:
        template = json.loads(args.settings_template.read_text(encoding='utf-8'))
    translated = to_notebook_settings(config)
    settings = args.output / 'reference_settings.json'
    settings.write_text(json.dumps(to_notebook_settings(config, template), indent=2), encoding='utf-8')
    generated = args.output / 'reference_headless.ipynb'
    executed = args.output / 'reference_executed.ipynb'
    build_headless_notebook(args.notebook, generated, settings, args.output,
                            to_notebook_frame_range(config), args.workdir,
                            early_settings=translated)
    env = os.environ.copy()
    # Keep Jupyter/IPython runtime state inside the isolated job. The user's
    # profile DB may be read-only under validation sandboxes, and `%cd` tries
    # to update its directory history on every reference run.
    ipython_dir = args.output / '.ipython'
    jupyter_dir = args.output / '.jupyter'
    ipython_dir.mkdir(exist_ok=True)
    jupyter_dir.mkdir(exist_ok=True)
    env.update(IS_DOCKER='1', IS_LOCAL_INSTALL='1', PYTHONUNBUFFERED='1',
               IPYTHONDIR=str(ipython_dir), JUPYTER_CONFIG_DIR=str(jupyter_dir))
    command = [str(args.reference_python), '-m', 'jupyter', 'nbconvert',
               '--to', 'notebook', '--execute', str(generated),
               '--output', str(executed),
               f'--ExecutePreprocessor.timeout={args.timeout}',
               '--ExecutePreprocessor.allow_errors=False']
    returncode = subprocess.run(command, cwd=args.workdir, env=env).returncode
    if returncode != 0:
        return returncode
    snapshots = sorted((batch_output / 'settings').glob('*_settings.txt'),
                       key=lambda p: p.stat().st_mtime)
    if not snapshots:
        print('reference run saved no settings snapshot; cannot verify')
        return 1
    snapshot = json.loads(snapshots[-1].read_text(encoding='utf-8'))
    mismatches = verify_applied_settings(translated, snapshot)
    if mismatches:
        print('reference notebook silently ignored requested settings:')
        for line in mismatches:
            print('  ' + line)
        return 1
    print(f'settings snapshot verified against translation: {snapshots[-1]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
