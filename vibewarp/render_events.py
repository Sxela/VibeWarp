"""Step/preview events without coupling a renderer to the web server."""
import threading
from contextlib import contextmanager

_state = threading.local()


@contextmanager
def render_event_scope(callback):
    previous = getattr(_state, 'callback', None)
    _state.callback = callback
    try:
        yield
    finally:
        _state.callback = previous


def render_event(**data):
    callback = getattr(_state, 'callback', None)
    if callback:
        callback(data)
