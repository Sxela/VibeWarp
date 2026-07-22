"""Cooperative cancellation for work running on the renderer thread."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Callable, Iterator


_state = threading.local()


@contextmanager
def cancellation_scope(check: Callable[[], bool]) -> Iterator[None]:
    """Make a job cancellation check available to deeply nested render code."""
    previous = getattr(_state, 'check', None)
    _state.check = check
    try:
        yield
    finally:
        if previous is None:
            try:
                del _state.check
            except AttributeError:
                pass
        else:
            _state.check = previous


def is_cancel_requested() -> bool:
    check = getattr(_state, 'check', None)
    return bool(check and check())


def raise_if_cancelled() -> None:
    if is_cancel_requested():
        raise KeyboardInterrupt('Cancelled by user')
