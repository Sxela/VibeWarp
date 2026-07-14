"""Resumable overnight GPU validation orchestrator.

The runner expands a base VibeWarp/WarpFusion config into independent jobs,
runs each in a fresh process (releasing VRAM between cases), compares image
artifacts, and writes JSON plus Markdown reports. It has no GPU dependency of
its own, which keeps orchestration and report tests runnable on CPU.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import glob
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp'}
RUN_STAMP_FORMAT = '%Y-%m-%d_%H-%M-%S-%f'


def deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def expand(value: str, variables: dict[str, str]) -> str:
    value = os.path.expandvars(value)
    try:
        return value.format_map(variables)
    except KeyError as exc:
        raise ValueError(f"unknown placeholder {exc} in {value!r}") from exc


def load_base_config(spec: dict, repo: Path) -> dict:
    raw_path = spec.get('base_config')
    if not raw_path:
        raise ValueError("manifest.defaults.base_config is required")
    path = Path(expand(raw_path, {'repo': str(repo)}))
    if not path.is_absolute():
        path = repo / path
    kind = spec.get('base_config_kind', 'json')
    if kind == 'json':
        with path.open(encoding='utf-8') as handle:
            return json.load(handle)
    if kind == 'warpfusion':
        from vibewarp.settings import load_warpfusion_settings
        return load_warpfusion_settings(
            str(path), models_root=spec.get('models_root'))
    if kind == 'vibewarp':
        from vibewarp.settings import load_vibewarp_settings
        return load_vibewarp_settings(str(path))
    raise ValueError(f"unsupported base_config_kind: {kind}")


def image_metrics(left: Path, right: Path) -> dict[str, Any]:
    with Image.open(left) as a_img, Image.open(right) as b_img:
        a = np.asarray(a_img.convert('RGB'), dtype=np.float32) / 255.0
        b = np.asarray(b_img.convert('RGB'), dtype=np.float32) / 255.0
    if a.shape != b.shape:
        return {'shape_left': list(a.shape), 'shape_right': list(b.shape),
                'shape_match': False}
    delta = np.abs(a - b)
    mse = float(np.mean(np.square(a - b)))
    return {
        'shape_match': True,
        'mae': float(np.mean(delta)),
        'rmse': math.sqrt(mse),
        'max_abs': float(np.max(delta)),
        'psnr': float('inf') if mse == 0 else float(-10.0 * math.log10(mse)),
        'exact': bool(np.array_equal(a, b)),
    }


def _artifact_files(job_dir: Path, pattern: str) -> list[Path]:
    return sorted(
        Path(p) for p in glob.glob(str(job_dir / pattern), recursive=True)
        if Path(p).suffix.lower() in IMAGE_SUFFIXES
    )


def compare_jobs(left_dir: Path, right_dir: Path, pattern: str,
                 thresholds: dict, right_pattern: str | None = None) -> dict[str, Any]:
    left = _artifact_files(left_dir, pattern)
    right = _artifact_files(right_dir, right_pattern or pattern)
    pairs = list(zip(left, right))
    result: dict[str, Any] = {
        'left_count': len(left), 'right_count': len(right), 'files': []}
    if not pairs or len(left) != len(right):
        result.update(status='fail', reason='artifact counts differ or are zero')
        return result
    passed = True
    for a, b in pairs:
        metrics = image_metrics(a, b)
        entry = {'left': a.name, 'right': b.name, **metrics}
        if not metrics.get('shape_match'):
            entry['passed'] = False
        else:
            entry['passed'] = all(
                metrics.get(name, float('inf')) <= limit
                for name, limit in thresholds.items()
                if name in {'mae', 'rmse', 'max_abs'}
            )
            min_psnr = thresholds.get('min_psnr')
            if min_psnr is not None:
                entry['passed'] &= metrics['psnr'] >= min_psnr
            if thresholds.get('exact'):
                entry['passed'] &= metrics['exact']
        passed &= entry['passed']
        result['files'].append(entry)
    result['status'] = 'pass' if passed else 'fail'
    return result


def expand_tests(manifest: dict) -> tuple[list[dict], list[dict]]:
    """Expand paired `tests` into vibewarp/notebook jobs plus a comparison.

    Each test uses ONE original WarpFusion settings file (refs/examples) as
    the base for BOTH sides, so untranslated keys cannot silently diverge.
    Outputs land in <run_dir>/<timestamp>_<test-id>/{vibewarp,notebook}.
    """
    jobs = list(manifest.get('jobs', []))
    comparisons = list(manifest.get('comparisons', []))
    default_thresholds = manifest.get('defaults', {}).get(
        'thresholds', {'mae': 0.015, 'min_psnr': 30.0})
    for test in manifest.get('tests', []):
        test_id = test['id']
        if '/' in test_id or '\\' in test_id:
            raise ValueError(f'test id must be a plain name: {test_id!r}')
        settings = test.get('settings')
        if not settings:
            raise ValueError(f'test {test_id} needs a settings file')
        common = {
            'enabled': test.get('enabled', True),
            'requires': list(test.get('requires', [])),
            'base_config': settings,
            'base_config_kind': test.get('settings_kind', 'warpfusion'),
            'overrides': test.get('overrides', {}),
            'batch_name': test_id,
            'export_frames': True,
            'only_controlnets': test.get('only_controlnets'),
        }
        vibewarp_job = {**common, 'id': f'{test_id}-vibewarp',
                        'dir': f'{test_id}/vibewarp',
                        'env': {'VIBEWARP_PARITY_MODE': '1', **test.get('env', {})}}
        notebook_job = {**common, 'id': f'{test_id}-notebook',
                        'dir': f'{test_id}/notebook',
                        'reference_notebook': True,
                        'settings_template': settings}
        # AnimateDiff: the notebook's initial noise is drawn from an UNSEEDED cpu
        # generator (nothing calls seed_everything before do_run_adiff's big_noise),
        # so it is not reproducible from a seed -- two notebook runs at the same seed
        # were measured to disagree at MAE 0.30, worse than notebook-vs-vibewarp.
        # Comparing images is therefore only meaningful if BOTH sides start from the
        # same noise. Run the reference first and feed VibeWarp the noise it dumped.
        adiff = bool(test.get('overrides', {})
                     .get('animatediff', {}).get('enabled'))
        if adiff and test.get('notebook', True):
            vibewarp_job['adiff_noise_from'] = f'{test_id}/notebook'
            jobs.append(notebook_job)
        jobs.append(vibewarp_job)
        if test.get('notebook', True):
            if not adiff:
                jobs.append(notebook_job)
            comparisons.append({
                'id': test_id,
                'left': f'{test_id}-vibewarp', 'right': f'{test_id}-notebook',
                'pattern': 'frame_*.*',
                'thresholds': test.get('thresholds', default_thresholds),
            })
    return jobs, comparisons


def _paired_test_id(job: dict) -> str | None:
    """Return the paired-test directory component for an expanded job."""
    raw = str(job.get('dir', '')).replace('\\', '/')
    if '/' not in raw:
        return None
    return raw.split('/', 1)[0]


def _latest_paired_dir(run_dir: Path, test_id: str) -> Path | None:
    """Find the newest timestamp-prefixed directory for a paired test."""
    matches = sorted(
        path for path in run_dir.glob(f'*_{test_id}')
        if path.is_dir() and path.name.endswith(f'_{test_id}')
    )
    return matches[-1] if matches else None


def export_final_frames(job_dir: Path, batch_name: str) -> int:
    """Copy the newest run's final frames to <job_dir>/frame_NNNNNN.<ext>.

    Matches only `<batch>(<run>)_<frame>.<ext>` — annotator debug maps, masks
    and blend temps use longer suffixes and are excluded.
    """
    import re
    pattern = re.compile(
        re.escape(batch_name) + r'\((\d+)\)_(\d+)\.(png|jpe?g|webp)$')
    frames = []
    artifacts = job_dir / 'artifacts'
    if artifacts.is_dir():
        for path in artifacts.rglob('*'):
            match = pattern.fullmatch(path.name)
            if match:
                frames.append((int(match.group(1)), int(match.group(2)), path))
    for stale in job_dir.glob('frame_*.*'):
        stale.unlink()
    if not frames:
        return 0
    newest_run = max(run for run, _, _ in frames)
    exported = 0
    for run, frame, path in frames:
        if run != newest_run:
            continue
        shutil.copy2(path, job_dir / f'frame_{frame:06d}{path.suffix.lower()}')
        exported += 1
    return exported


def _missing_requirements(job: dict, variables: dict[str, str], repo: Path) -> list[str]:
    missing = []
    for raw in job.get('requires', []):
        path = Path(expand(raw, variables))
        if not path.is_absolute():
            path = repo / path
        if not path.exists():
            missing.append(str(path))
    return missing


def _write_report(run_dir: Path, report: dict) -> None:
    (run_dir / 'report.json').write_text(
        json.dumps(report, indent=2, default=str), encoding='utf-8')
    lines = [
        '# GPU validation report', '',
        f"Started: {report['started']}",
        f"Finished: {report.get('finished', 'in progress')}", '',
        '## Jobs', '',
        '| Job | Status | Duration | Note |',
        '|---|---:|---:|---|',
    ]
    for job in report['jobs']:
        note = job.get('reason', '').replace('|', '\\|')
        lines.append(f"| `{job['id']}` | {job['status']} | "
                     f"{job.get('duration_seconds', 0):.1f}s | {note} |")
    lines += ['', '## Comparisons', '',
              '| Comparison | Status | Frames | Note |',
              '|---|---:|---:|---|']
    for comp in report.get('comparisons', []):
        count = min(comp.get('left_count', 0), comp.get('right_count', 0))
        note = comp.get('reason', '').replace('|', '\\|')
        lines.append(f"| `{comp['id']}` | {comp['status']} | {count} | {note} |")
    (run_dir / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_manifest(manifest_path: Path, output_root: Path, resume: bool = True,
                 only: set[str] | None = None) -> tuple[Path, dict]:
    repo = Path(__file__).resolve().parents[1]
    with manifest_path.open(encoding='utf-8') as handle:
        manifest = json.load(handle)
    defaults = manifest.get('defaults', {})
    run_name = manifest.get('name', 'gpu-parity')
    run_dir = output_root / run_name if run_name else output_root
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs, comparisons = expand_tests(manifest)
    base_cache: dict[str, dict] = {}

    def job_base(job: dict) -> dict:
        spec = defaults
        if job.get('base_config'):
            spec = {**defaults, 'base_config': job['base_config'],
                    'base_config_kind': job.get('base_config_kind', 'warpfusion')}
        cache_key = f"{spec.get('base_config')}::{spec.get('base_config_kind')}"
        if cache_key not in base_cache:
            base_cache[cache_key] = load_base_config(spec, repo)
        return base_cache[cache_key]

    started = dt.datetime.now().astimezone().isoformat()
    report: dict[str, Any] = {
        'manifest': str(manifest_path.resolve()), 'started': started,
        'host': platform.node(), 'python': sys.version, 'jobs': [],
        'comparisons': [],
    }
    python = expand(defaults.get('python', sys.executable), {'repo': str(repo)})
    common = defaults.get('overrides', {})
    timeout = defaults.get('timeout_seconds', 21600)

    # Both sides of a paired experiment share one timestamp-prefixed parent.
    # A normal invocation resumes the newest pair; --fresh creates a new pair
    # and therefore never overwrites evidence produced by older code.
    invocation_stamp = dt.datetime.now().strftime(RUN_STAMP_FORMAT)
    paired_dirs: dict[str, Path] = {}
    for job in jobs:
        test_id = _paired_test_id(job)
        if test_id is None or test_id in paired_dirs:
            continue
        previous = _latest_paired_dir(run_dir, test_id) if resume else None
        paired_dirs[test_id] = previous or run_dir / f'{invocation_stamp}_{test_id}'

    def resolve_job_dir(job: dict) -> Path:
        test_id = _paired_test_id(job)
        if test_id is not None:
            side = str(job['dir']).replace('\\', '/').split('/', 1)[1]
            return paired_dirs[test_id] / side
        return run_dir / job['dir'] if job.get('dir') else run_dir / 'jobs' / job['id']

    def selected(job: dict) -> bool:
        if not only:
            return True
        test_id = str(job.get('dir', '')).replace('\\', '/').split('/')[0]
        return job['id'] in only or (test_id and test_id in only)

    for job in jobs:
        job_id = job['id']
        if not selected(job):
            continue
        job_dir = resolve_job_dir(job)
        variables = {'repo': str(repo), 'run_dir': str(run_dir),
                     'job_dir': str(job_dir), 'python': python}
        result_path = job_dir / 'result.json'
        if resume and result_path.exists():
            previous = json.loads(result_path.read_text(encoding='utf-8'))
            if previous.get('status') == 'pass':
                previous['resumed'] = True
                if job.get('export_frames'):
                    export_final_frames(job_dir, job.get('batch_name', job_id))
                report['jobs'].append(previous)
                _write_report(run_dir, report)
                continue
        job_dir.mkdir(parents=True, exist_ok=True)
        missing = _missing_requirements(job, variables, repo)
        if not job.get('enabled', True) or missing:
            result = {'id': job_id, 'status': 'skip', 'duration_seconds': 0,
                      'reason': 'disabled' if not job.get('enabled', True)
                      else 'missing: ' + ', '.join(missing)}
            result_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
            report['jobs'].append(result)
            _write_report(run_dir, report)
            continue

        config = deep_merge(job_base(job), common)
        config = deep_merge(config, job.get('overrides', {}))
        only_controlnets = job.get('only_controlnets')
        if only_controlnets is not None:
            selected_controlnets = set(only_controlnets)
            models = config.get('controlnet', {}).get('models', {})
            config['controlnet']['models'] = {
                key: value for key, value in models.items()
                if key in selected_controlnets
            }
        config['root_dir'] = str(job_dir)
        config['output_dir'] = 'artifacts'
        config['batch_name'] = job.get('batch_name', job_id)
        config_path = job_dir / 'config.json'
        config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
        command = job.get('command') or [python, '-m', 'vibewarp',
                                         '--config', str(config_path)]
        if job.get('reference_notebook'):
            reference = manifest.get('reference_notebook', {})
            required = ('source', 'workdir', 'python')
            absent = [key for key in required if not reference.get(key)]
            if absent:
                raise ValueError('reference_notebook is missing: ' + ', '.join(absent))
            command = [
                python, str(repo / 'tools' / 'notebook_reference.py'),
                '--notebook', expand(reference['source'], variables),
                '--config', str(config_path),
                '--workdir', expand(reference['workdir'], variables),
                '--output', str(job_dir / 'artifacts'),
                '--reference-python', expand(reference['python'], variables),
                '--timeout', str(timeout),
            ]
            template = job.get('settings_template') or reference.get('settings_template')
            if template:
                command += ['--settings-template', expand(template, variables)]
        command = [expand(str(part), variables) for part in command]
        env = os.environ.copy()
        env.update({key: expand(str(value), variables)
                    for key, value in defaults.get('env', {}).items()})
        env.update({key: expand(str(value), variables)
                    for key, value in job.get('env', {}).items()})
        noise_from = job.get('adiff_noise_from')
        if noise_from:
            noise = (paired_dirs[noise_from.split('/')[0]] / 'notebook'
                     / 'artifacts' / 'diag_adiff_noise.pt')
            if not noise.exists():
                raise RuntimeError(
                    f'{job_id}: AnimateDiff parity needs the reference noise at '
                    f'{noise} (run the -notebook side first). Without it the two '
                    f'sides start from different noise and the comparison is void.')
            env['VIBEWARP_ADIFF_NOISE'] = str(noise)
        log_path = job_dir / 'run.log'
        started_job = time.monotonic()
        try:
            with log_path.open('w', encoding='utf-8', errors='replace') as log:
                process = subprocess.run(command, cwd=repo, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, timeout=timeout)
            status = 'pass' if process.returncode == 0 else 'fail'
            reason = '' if process.returncode == 0 else f'exit code {process.returncode}'
        except subprocess.TimeoutExpired:
            status, reason = 'fail', f'timed out after {timeout}s'
        except Exception:
            status, reason = 'fail', traceback.format_exc(limit=2)
        result = {'id': job_id, 'status': status,
                  'duration_seconds': time.monotonic() - started_job,
                  'finished_at': dt.datetime.now().astimezone().isoformat(),
                  'reason': reason, 'command': command, 'log': str(log_path)}
        # Visible freshness marker: run_<timestamp>.stamp in the job folder
        for stale_stamp in job_dir.glob('run_*.stamp'):
            stale_stamp.unlink()
        stamp = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        (job_dir / f'run_{stamp}.stamp').touch()
        if status == 'pass' and job.get('export_frames'):
            exported = export_final_frames(job_dir, job.get('batch_name', job_id))
            result['exported_frames'] = exported
            if not exported:
                result.update(status='fail', reason='no final frames to export')
        result_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
        report['jobs'].append(result)
        _write_report(run_dir, report)

    job_status = {item['id']: item['status'] for item in report['jobs']}
    job_dirs = {job['id']: resolve_job_dir(job) for job in jobs}
    for spec in comparisons:
        if only and not ({spec['left'], spec['right']} <= only
                         or spec['id'] in only):
            continue
        comp = {'id': spec['id']}
        if job_status.get(spec['left']) != 'pass' or job_status.get(spec['right']) != 'pass':
            comp.update(status='skip', reason='one or both jobs did not pass')
        else:
            compared = compare_jobs(
                job_dirs[spec['left']], job_dirs[spec['right']],
                spec.get('left_pattern', spec.get('pattern', 'artifacts/**/*.png')),
                spec.get('thresholds', {}), right_pattern=spec.get('right_pattern'))
            comp.update(compared)
        test_dir = paired_dirs.get(spec['id'], run_dir / spec['id'])
        if test_dir.is_dir():
            (test_dir / 'comparison.json').write_text(
                json.dumps(comp, indent=2, default=str), encoding='utf-8')
            for stale_stamp in test_dir.glob('compared_*.stamp'):
                stale_stamp.unlink()
            stamp = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            (test_dir / f"compared_{stamp}_{comp.get('status', 'skip')}.stamp").touch()
        report['comparisons'].append(comp)

    report['finished'] = dt.datetime.now().astimezone().isoformat()
    _write_report(run_dir, report)
    return run_dir, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('gpu_validation'))
    parser.add_argument('--fresh', action='store_true', help='rerun passing jobs')
    parser.add_argument('--only', nargs='+', help='run only these job IDs')
    parser.add_argument('--clean', action='store_true', help='delete this suite output first')
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    run_name = manifest.get('name', 'gpu-parity')
    suite_dir = args.output / run_name if run_name else args.output
    if args.clean and suite_dir.exists():
        shutil.rmtree(suite_dir)
    run_dir, report = run_manifest(args.manifest, args.output,
                                   resume=not args.fresh,
                                   only=set(args.only) if args.only else None)
    print(f"Report: {run_dir / 'report.md'}")
    failed = any(item['status'] == 'fail' for item in report['jobs'])
    failed |= any(item['status'] == 'fail' for item in report['comparisons'])
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
