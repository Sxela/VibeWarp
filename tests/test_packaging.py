"""Guards against packaging regressions that break fresh installs.

The vendored packages are the whole point of Goal 1 (self-containment): a
fresh clone must contain every file the runtime imports. A stray/unanchored
.gitignore rule once silently excluded vibewarp/vendor/ldm/models/** and
sgm/models/** from git, so clones crashed with ModuleNotFoundError even
though the files existed on the author's disk. This test catches that class
of bug at CI time instead of at a user's first run.
"""
import os
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(REPO, 'vibewarp', 'vendor')


def _git_tracked_vendor_files():
    try:
        out = subprocess.run(
            ['git', 'ls-files', 'vibewarp/vendor'],
            cwd=REPO, capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.SubprocessError):
        pytest.skip('git not available')
    if out.returncode != 0:
        pytest.skip('not a git checkout')
    return {os.path.normpath(line) for line in out.stdout.splitlines() if line.strip()}


def test_all_vendored_python_is_tracked():
    tracked = _git_tracked_vendor_files()
    untracked = []
    for dirpath, _dirs, files in os.walk(VENDOR):
        if '__pycache__' in dirpath:
            continue
        for fn in files:
            if not fn.endswith('.py'):
                continue
            rel = os.path.normpath(os.path.relpath(os.path.join(dirpath, fn), REPO))
            if rel not in tracked:
                untracked.append(rel)
    assert not untracked, (
        'Vendored source files exist on disk but are NOT tracked by git — a '
        'fresh clone would be missing them (check .gitignore for an '
        f'unanchored rule):\n  ' + '\n  '.join(sorted(untracked)))
