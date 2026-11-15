"""Read the outer render ETA from tqdm output without confusing sampling bars."""
import re

_ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
_OUTER = re.compile(r'^\s*(?:Rendering frames|Contact sheets|AnimateDiff batches):')
_ETA = re.compile(r'\[[^\[\]\r\n]*?<\s*(\d+:\d{2}(?::\d{2})?)\s*[,\]]')


def render_eta(line):
    """Return (is_outer_progress, remaining_seconds); unknown estimates are None."""
    clean = _ANSI.sub('', line).strip()
    if not _OUTER.match(clean):
        return False, None
    match = _ETA.search(clean)
    if not match:
        return True, None
    parts = [int(part) for part in match[1].split(':')]
    if any(part >= 60 for part in parts[1:]):
        return True, None
    seconds = parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0] * 3600 + parts[1] * 60 + parts[2]
    return True, seconds
