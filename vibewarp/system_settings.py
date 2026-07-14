"""Machine-level settings, kept OUT of the per-project settings file.

Model paths, VRAM tuning and thread counts describe the machine, not the render. If they
travel inside a settings file then importing someone else's settings silently repoints your
checkpoint directory and retunes your VRAM — which is exactly what the System tab exists to
prevent. So they live in one file next to the install and are re-applied after every import.

  overlay(defaults)  -> config with this machine's system values applied
  capture(config)    -> {"section": {"field": value}} for the system tier only
  strip(config)      -> config with the system tier removed (for export)

Which fields are "system" is declared in ui_layout.py (tier='system'), so this module never
holds a field list of its own.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict

from vibewarp.ui_layout import system_keys

FILENAME = 'system.json'


def settings_path() -> Path:
    """Where this machine's system settings live.

    Next to the install (the repo/workfolder root), not in the user's home — the owner's
    call: an install stays self-contained and portable. VIBEWARP_HOME overrides it, which
    is also what keeps the tests off the developer's real file.
    """
    home = os.environ.get('VIBEWARP_HOME')
    root = Path(home) if home else Path(__file__).resolve().parent.parent
    return root / FILENAME


def _split(key: str) -> tuple[str, str]:
    section, _, field = key.partition('.')
    return section, field


def capture(config: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the system-tier values out of a config dict."""
    out: Dict[str, Any] = {}
    for key in system_keys():
        section, field = _split(key)
        if section == 'main':
            if field in config:
                out.setdefault('main', {})[field] = config[field]
        elif isinstance(config.get(section), dict) and field in config[section]:
            out.setdefault(section, {})[field] = config[section][field]
    return out


def strip(config: Dict[str, Any]) -> Dict[str, Any]:
    """Config with the system tier removed — what we hand out on export."""
    out = copy.deepcopy(config)
    for key in system_keys():
        section, field = _split(key)
        if section == 'main':
            out.pop(field, None)
        elif isinstance(out.get(section), dict):
            out[section].pop(field, None)
    return out


def overlay(config: Dict[str, Any], stored: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Apply this machine's system settings on top of a config.

    Only keys that are actually in the system tier are copied, so a stale system.json
    written by an older version cannot smuggle in creative settings.
    """
    stored = load() if stored is None else stored
    if not stored:
        return config
    out = copy.deepcopy(config)
    for key in system_keys():
        section, field = _split(key)
        source = stored.get(section)
        if not isinstance(source, dict) or field not in source:
            continue
        if section == 'main':
            out[field] = source[field]
        elif isinstance(out.get(section), dict):
            out[section][field] = source[field]
    return out


def load() -> Dict[str, Any]:
    path = settings_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        # A corrupt system.json must not take the app down; fall back to the defaults.
        return {}
    return data if isinstance(data, dict) else {}


def save(config: Dict[str, Any]) -> Dict[str, Any]:
    """Persist the system tier of `config`. Returns what was written."""
    captured = capture(config)
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(captured, indent=2), encoding='utf-8')
    return captured
