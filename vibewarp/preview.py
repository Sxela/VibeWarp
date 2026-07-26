"""Render-output introspection for the web UI's preview/debug tab.

Lets you scrub a finished (or in-progress) run frame by frame and put the init
frame, warped init, processed consistency mask, each ControlNet source/detected
map, diffusion input, and final render side by side.

Frame numbers here are always the RENDER frame number (0-based), which is what
the rest of the pipeline uses. That is not what every layer is called on disk:
extracted video frames are 1-based, so render frame N is `video_frames/{N+1}.jpg`
(see core/diffusion.py). Layers carry that offset so callers never have to think
about it — getting it wrong silently shows you the wrong frame.
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

# A run directory is a plain name ("54") directly under <output_dir>/<batch_name>.
# Anything else is rejected rather than joined into a path.
_RUN_ID = re.compile(r'^[A-Za-z0-9._-]+$')
_FRAME_SUFFIX = re.compile(r'^(?P<stem>.*?)_?(?P<frame>\d{6})\.(?P<ext>png|jpg|jpeg)$', re.I)
IMAGE_EXTS = ('.png', '.jpg', '.jpeg')
RUN_LABEL_FILE = '.vibewarp-label'
MAX_RUN_LABEL_LENGTH = 80

# Debug prefixes that aren't ControlNets.
_DEBUG_LABELS = {
    'diffusion_input': ('Diffusion input', 'Diffusion'),
    'consistency_mask': ('Consistency mask (white = keep warp)', 'Optical flow'),
    'hidream_reference_1': ('HiDream reference 1', 'Edit inputs'),
    'flux_reference_1': ('FLUX.2 reference 1', 'Edit inputs'),
    'flux_reference_2': ('FLUX.2 reference 2', 'Edit inputs'),
    'hidream_reference_2': ('HiDream reference 2 — style', 'Edit inputs'),
    'qwen_reference_1': ('Qwen reference 1', 'Edit inputs'),
    'mage_reference_1': ('Mage-Flow reference 1', 'Edit inputs'),
}


@dataclass(frozen=True)
class Layer:
    """One image series in a run, addressable by render frame number."""
    id: str
    label: str
    group: str
    directory: str          # relative to the run dir ('' = run root)
    prefix: str             # filename prefix before the 6-digit frame number
    frame_offset: int = 0   # add to the render frame number to get the on-disk number
    frames: tuple = ()

    def filename(self, frame: int, ext: str) -> str:
        return f"{self.prefix}{frame + self.frame_offset:06d}{ext}"


def runs_root(output_dir: str, batch_name: str) -> str:
    return os.path.join(output_dir or 'images_out', batch_name or 'warpfusion')


def list_runs(output_dir: str, batch_name: str) -> List[dict]:
    """Run directories, newest first."""
    root = runs_root(output_dir, batch_name)
    if not os.path.isdir(root):
        return []
    runs = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path) or not _RUN_ID.match(name):
            continue
        names = os.listdir(path)
        output = _scan(
            path, '', _output_prefix(path, batch_name, name, names), names=names)
        video = video_file(path, batch_name, names)
        resume = resume_status(path, batch_name, name, output_frames=output)
        runs.append({
            'id': name,
            'label': run_label(path),
            'modified': os.path.getmtime(path),
            'frames': len(output),
            # Newest rendered frame — the gallery uses it as the run's thumbnail.
            'last_frame': max(output) if output else None,
            'prompt': _run_prompt(path),
            # Older runs (and interrupted ones) may have kept none — the UI greys out
            # "Load settings" rather than offering a button that cannot work.
            'has_settings': settings_file(path) is not None,
            'video_available': video is not None,
            'resume_available': resume is not None,
            'resume_from': resume,
        })
    # Numeric run dirs ("2" vs "10") must not sort lexicographically.
    runs.sort(key=lambda r: (r['modified'], _numeric(r['id'])), reverse=True)
    return runs


def run_label(run: str) -> str:
    """Return a run's optional user-facing label without affecting its identity."""
    try:
        with open(os.path.join(run, RUN_LABEL_FILE), encoding='utf-8') as handle:
            return handle.read(MAX_RUN_LABEL_LENGTH + 1).strip()[:MAX_RUN_LABEL_LENGTH]
    except OSError:
        return ''


def set_run_label(output_dir: str, batch_name: str, run_id: str, label: str) -> str:
    """Persist a display-only run label; directory and artifact names stay unchanged."""
    run = run_dir(output_dir, batch_name, run_id)
    if run is None:
        raise FileNotFoundError(f'Unknown run: {run_id}')
    if not isinstance(label, str):
        raise ValueError('Run label must be text')
    cleaned = ' '.join(label.split())
    if len(cleaned) > MAX_RUN_LABEL_LENGTH:
        raise ValueError(
            f'Run label must be {MAX_RUN_LABEL_LENGTH} characters or fewer')

    path = os.path.join(run, RUN_LABEL_FILE)
    before = os.stat(run)
    try:
        if cleaned:
            temp_path = f'{path}.tmp'
            with open(temp_path, 'w', encoding='utf-8') as handle:
                handle.write(cleaned)
            os.replace(temp_path, path)
        elif os.path.exists(path):
            os.remove(path)
    finally:
        # Renaming a display label should not make an old run jump to the top of
        # the newest-first history list.
        os.utime(run, (before.st_atime, before.st_mtime))
    return cleaned


def settings_file(run: str) -> Optional[str]:
    """The settings the pipeline saved for a run, or None if it kept none.

    Every render writes `<run>/settings/<batch>(0)_settings.txt`. This is what lets you
    pull an old run's configuration back into the form and iterate on it, instead of
    reconstructing it by eye from the frames.
    """
    settings = os.path.join(run, 'settings')
    if not os.path.isdir(settings):
        return None
    files = [f for f in sorted(os.listdir(settings))
             if f.lower().endswith(('.txt', '.json'))]
    return os.path.join(settings, files[0]) if files else None


def video_file(run: str, batch_name: str,
               names: Optional[List[str]] = None) -> Optional[str]:
    """Return the assembled video for a run, preferring its canonical name."""
    preferred = os.path.join(run, f'{batch_name}.mp4')
    if os.path.isfile(preferred):
        return preferred
    try:
        available = names if names is not None else os.listdir(run)
        videos = [os.path.join(run, name) for name in available
                  if name.lower().endswith('.mp4')]
    except OSError:
        return None
    return max(videos, key=os.path.getmtime) if videos else None


def resume_status(run: str, batch_name: str, run_id: str, *,
                  output_frames: Optional[Dict[int, str]] = None,
                  extracted_frames: Optional[Dict[int, str]] = None) -> Optional[int]:
    """First missing render frame, or None when a saved run is complete/unknown."""
    path = settings_file(run)
    if path is None:
        return None
    try:
        from vibewarp.config_io import config_from_settings
        config = config_from_settings(path)
        output = output_frames
        if output is None:
            names = os.listdir(run)
            output = _scan(
                run, '', _output_prefix(run, batch_name, run_id, names), names=names)
        selected = config.frame_range or [0, 0]
        if len(selected) == 2 and selected[1] > selected[0]:
            start, end = selected[0], selected[1] + 1
        else:
            extracted = extracted_frames
            if extracted is None:
                extracted = _scan(run, 'video_frames', '', offset=1)
            if not extracted:
                return None
            start, end = min(extracted), max(extracted) + 1
        return next((frame for frame in range(start, end) if frame not in output), None)
    except (OSError, ValueError, TypeError):
        return None


def _run_prompt(run: str) -> str:
    """First prompt from the run's saved settings — what makes an old run
    identifiable in the gallery. Best effort: never fail a listing over it."""
    path = settings_file(run)
    if path is None:
        return ''
    try:
        import json
        with open(path, encoding='utf-8') as handle:
            prompts = json.load(handle).get('text_prompts') or {}
        if isinstance(prompts, dict):
            prompts = list(prompts.values())
        first = prompts[0] if prompts else ''
        if isinstance(first, list):
            first = first[0] if first else ''
        return str(first)
    except Exception:
        return ''


def _numeric(name: str) -> float:
    try:
        return float(name)
    except ValueError:
        return -1.0


def run_dir(output_dir: str, batch_name: str, run_id: str) -> Optional[str]:
    """Resolve a run directory, refusing anything outside the runs root."""
    if not run_id or not _RUN_ID.match(run_id):
        return None
    root = os.path.abspath(runs_root(output_dir, batch_name))
    path = os.path.abspath(os.path.join(root, run_id))
    if os.path.commonpath([root, path]) != root or not os.path.isdir(path):
        return None
    return path


def _output_prefix(path: str, batch_name: str, run_id: str,
                   names: Optional[List[str]] = None) -> str:
    """Rendered frames are named `<batch>(<run>)_NNNNNN.<ext>`.

    The run index in the name is not always the directory name (a resumed run
    keeps its original index), so read it off an actual file instead of assuming.
    """
    if names is None:
        names = os.listdir(path)
    expected = f"{batch_name}({run_id})_"
    if any(name.startswith(expected) for name in names):
        return expected
    for name in sorted(names):
        if name.startswith(f"{batch_name}(") and name.lower().endswith(IMAGE_EXTS):
            head = name.split(')_')[0]
            return f"{head})_"
    return expected


def _scan(run: str, directory: str, prefix: str, offset: int = 0,
          names: Optional[List[str]] = None) -> Dict[int, str]:
    """Map render frame number -> file extension, for one series."""
    folder = os.path.join(run, directory) if directory else run
    if not os.path.isdir(folder):
        return {}
    found = {}
    if names is None:
        names = os.listdir(folder)
    for name in names:
        if not name.startswith(prefix) or not name.lower().endswith(IMAGE_EXTS):
            continue
        match = _FRAME_SUFFIX.match(name)
        if not match:
            continue
        # Guard against a prefix that is a prefix of another series
        # (e.g. "control_sd15_depth_" also matching "control_sd15_depth_anything_").
        stem = match.group('stem')
        if f"{stem}_" != prefix and stem != prefix.rstrip('_'):
            continue
        found[int(match.group('frame')) - offset] = os.path.splitext(name)[1]
    return found


def describe_run(output_dir: str, batch_name: str, run_id: str) -> Optional[dict]:
    """Layers available in a run, each with the render frames it has."""
    run = run_dir(output_dir, batch_name, run_id)
    if run is None:
        return None

    root_names = os.listdir(run)
    video_dir = os.path.join(run, 'video_frames')
    video_names = os.listdir(video_dir) if os.path.isdir(video_dir) else []
    debug = os.path.join(run, 'debug')
    debug_names = os.listdir(debug) if os.path.isdir(debug) else []

    candidates = [
        # Extracted video frames are 1-based on disk: render frame N is N+1.
        Layer('init', 'Init (video frame)', 'Input', 'video_frames', '', frame_offset=1),
        Layer('warped', 'Warped init', 'Input', '', '_warped_'),
        Layer('output', 'Output', 'Render', '',
              _output_prefix(run, batch_name, run_id, root_names)),
    ]

    if debug_names:
        prefixes = set()
        for name in debug_names:
            match = _FRAME_SUFFIX.match(name)
            if match:
                prefixes.add(match.group('stem'))
        for stem in sorted(prefixes):
            label, group = _debug_label(stem)
            candidates.append(Layer(f'debug:{stem}', label, group, 'debug', f'{stem}_'))

    layers = []
    scanned = {}
    for layer in candidates:
        names = (root_names if not layer.directory else
                 video_names if layer.directory == 'video_frames' else debug_names)
        frames = _scan(
            run, layer.directory, layer.prefix, layer.frame_offset, names=names)
        scanned[layer.id] = frames
        if frames:
            layers.append({
                'id': layer.id,
                'label': layer.label,
                'group': layer.group,
                'frames': sorted(frames),
            })

    all_frames = sorted({f for layer in layers for f in layer['frames']})
    return {
        'run': run_id,
        'layers': layers,
        'frames': all_frames,
        'video_available': video_file(run, batch_name, root_names) is not None,
        'resume_from': resume_status(
            run, batch_name, run_id,
            output_frames=scanned.get('output'),
            extracted_frames=scanned.get('init')),
    }


def _debug_label(stem: str) -> tuple:
    if stem in _DEBUG_LABELS:
        return _DEBUG_LABELS[stem]
    if stem.startswith('flux_reference_'):
        return stem.replace('_', ' ').replace('flux', 'FLUX.2', 1), 'Edit inputs'
    if stem.startswith('qwen_reference_'):
        return stem.replace('_', ' ').replace('qwen', 'Qwen', 1), 'Edit inputs'
    if stem.startswith('mage_reference_'):
        return stem.replace('_', ' ').replace(
            'mage', 'Mage-Flow', 1), 'Edit inputs'
    for suffix, kind in (('_detected', 'detected map'), ('_source', 'source')):
        if stem.endswith(suffix):
            net = stem[: -len(suffix)]
            return f"{_net_label(net)} — {kind}", 'ControlNet'
    return stem.replace('_', ' '), 'Debug'


def _net_label(key: str) -> str:
    from vibewarp.controlnet_catalog import CONTROLNET_CATALOG
    spec = CONTROLNET_CATALOG.get(key)
    return spec.label if spec else key


def image_path(output_dir: str, batch_name: str, run_id: str,
               layer_id: str, frame: int) -> Optional[str]:
    """Absolute path of one layer's image for a render frame, or None."""
    run = run_dir(output_dir, batch_name, run_id)
    if run is None:
        return None

    if layer_id == 'init':
        layers = [Layer('init', '', '', 'video_frames', '', frame_offset=1)]
    elif layer_id == 'warped':
        layers = [Layer('warped', '', '', '', '_warped_')]
    elif layer_id == 'output':
        # Current runs always use (0); older layouts used the run id. Try both
        # directly so a frame seek performs only a handful of stat calls instead
        # of listing every file in a many-thousand-frame directory.
        prefixes = dict.fromkeys((f'{batch_name}(0)_', f'{batch_name}({run_id})_'))
        layers = [Layer('output', '', '', '', prefix) for prefix in prefixes]
    elif layer_id.startswith('debug:'):
        stem = layer_id.split(':', 1)[1]
        if not _RUN_ID.match(stem):
            return None
        layers = [Layer(layer_id, '', '', 'debug', f'{stem}_')]
    else:
        return None

    for layer in layers:
        folder = os.path.join(run, layer.directory) if layer.directory else run
        for ext in IMAGE_EXTS:
            candidate = os.path.join(folder, layer.filename(frame, ext))
            if os.path.isfile(candidate):
                return candidate

    # A few imported/very old runs can carry a prefix unrelated to either
    # convention. Pay for one directory scan only on that compatibility path.
    if layer_id == 'output':
        fallback = Layer(
            'output', '', '', '', _output_prefix(run, batch_name, run_id))
        for ext in IMAGE_EXTS:
            candidate = os.path.join(run, fallback.filename(frame, ext))
            if os.path.isfile(candidate):
                return candidate
    return None
