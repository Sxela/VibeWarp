"""Simple append-only timing log — writes to vibewarp_profile.txt in the repo root."""

import os
import time
from datetime import datetime

_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'vibewarp_profile.txt')


def log_time(label: str, elapsed: float) -> None:
    """Print and append a timing entry."""
    msg = f"[{datetime.now().strftime('%H:%M:%S')}] {label}: {elapsed:.1f}s"
    print(f"  {msg}")
    try:
        with open(_LOG_PATH, 'a') as f:
            f.write(msg + '\n')
    except Exception:
        pass
