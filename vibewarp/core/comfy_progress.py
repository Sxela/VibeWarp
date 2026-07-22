"""Live ComfyUI execution feedback for VibeWarp's synchronous backends."""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable
from urllib.parse import quote, urlsplit, urlunsplit

try:
    import websocket
except ImportError:  # HTTP queue/elapsed feedback remains available.
    websocket = None


class ComfyProgressMonitor:
    """Mirror Comfy execution events to stdout, with an HTTP fallback.

    VibeWarp's web job runner already streams stdout into the in-app log dock,
    so these messages appear without a separate frontend transport.
    """

    def __init__(
        self,
        base_url: str,
        client_id: str,
        graph: dict[str, dict[str, Any]],
    ) -> None:
        self.base_url = base_url
        self.client_id = client_id
        self.node_names = {
            str(node_id): str(node.get('class_type', f'node {node_id}'))
            for node_id, node in graph.items()
        }
        self.prompt_id: str | None = None
        self.started_at = time.monotonic()
        self._socket: Any = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._connected = False
        self._last_message: tuple[Any, ...] | None = None
        self._last_event_at = 0.0
        self._next_queue_poll = 0.0
        self._next_heartbeat = self.started_at + 10.0

    def start(self) -> None:
        """Connect before prompt submission so early execution events are seen."""
        if websocket is None or not self.base_url:
            self._ready.set()
            return
        self._thread = threading.Thread(
            target=self._listen,
            name='vibewarp-comfy-progress',
            daemon=True,
        )
        self._thread.start()
        # Local connections normally resolve immediately. Do not let feedback
        # delay rendering when a proxy/firewall interferes with WebSockets.
        self._ready.wait(2.0)

    def set_prompt_id(self, prompt_id: str) -> None:
        self.prompt_id = prompt_id
        self.started_at = time.monotonic()
        self._next_heartbeat = self.started_at + 10.0
        print(f"ComfyUI: prompt {prompt_id[:8]} submitted", flush=True)

    def poll(self, request: Callable[..., Any]) -> None:
        """Report queue state and periodic elapsed time if events go quiet."""
        if not self.prompt_id:
            return
        now = time.monotonic()
        if now >= self._next_queue_poll and (
            not self._connected or now - self._last_event_at >= 5.0
        ):
            self._next_queue_poll = now + 3.0
            try:
                queue = request('GET', '/queue', timeout=5.0).json()
                state = self._queue_state(queue)
                if state:
                    self._emit(('queue', state), f"ComfyUI: {state}")
            except Exception:
                # History polling remains authoritative; feedback must never
                # turn a successful render into a failure.
                pass
        if now >= self._next_heartbeat:
            elapsed = round(now - self.started_at)
            self._next_heartbeat = now + 10.0
            self._emit(
                ('heartbeat', elapsed),
                f"ComfyUI: still working - {elapsed}s elapsed",
            )

    def close(self) -> None:
        self._stop.set()
        sock = self._socket
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)

    def cancel(self, request: Callable[..., Any]) -> None:
        """Remove this prompt from Comfy's queue or interrupt it if running."""
        if not self.prompt_id:
            return
        try:
            queue = request('GET', '/queue', timeout=5.0).json()
            state = self._queue_state(queue)
            if state and state.startswith('queued'):
                request(
                    'POST', '/queue', timeout=5.0,
                    json={'delete': [self.prompt_id]},
                )
                print('ComfyUI: removed cancelled prompt from queue', flush=True)
            elif state == 'running':
                # New Comfy builds target this prompt specifically. On older
                # builds the endpoint is global, but the queue check above
                # guarantees our own prompt is the one currently executing.
                request(
                    'POST', '/interrupt', timeout=5.0,
                    json={'prompt_id': self.prompt_id},
                )
                print('ComfyUI: interrupted cancelled prompt', flush=True)
        except Exception as exc:
            # Local cancellation must still complete even if Comfy disappeared.
            print(f"ComfyUI: could not stop remote prompt: {exc}", flush=True)

    def _websocket_url(self) -> str:
        parts = urlsplit(self.base_url)
        scheme = 'wss' if parts.scheme == 'https' else 'ws'
        path = f"{parts.path.rstrip('/')}/ws"
        return urlunsplit((
            scheme, parts.netloc, path,
            f"clientId={quote(self.client_id, safe='')}", '',
        ))

    def _listen(self) -> None:
        try:
            self._socket = websocket.create_connection(
                self._websocket_url(), timeout=2.0)
            self._socket.settimeout(1.0)
            self._connected = True
        except Exception:
            self._ready.set()
            return
        self._ready.set()
        while not self._stop.is_set():
            try:
                message = self._socket.recv()
            except Exception as exc:
                timeout_error = getattr(websocket, 'WebSocketTimeoutException', ())
                if timeout_error and isinstance(exc, timeout_error):
                    continue
                break
            if not isinstance(message, str):
                continue  # Binary preview frames are deliberately ignored.
            try:
                event = json.loads(message)
            except (TypeError, ValueError):
                continue
            self._handle_event(event)
        self._connected = False

    def _handle_event(self, event: dict[str, Any]) -> None:
        event_type = event.get('type')
        data = event.get('data') or {}
        prompt_id = data.get('prompt_id')
        if not self.prompt_id or (prompt_id and prompt_id != self.prompt_id):
            return
        self._last_event_at = time.monotonic()

        if event_type == 'execution_start':
            self._emit(('start',), 'ComfyUI: execution started')
        elif event_type == 'executing':
            node_id = data.get('node')
            if node_id is None:
                self._emit(('complete',), 'ComfyUI: execution complete')
            else:
                name = self.node_names.get(str(node_id), f'node {node_id}')
                self._emit(('node', str(node_id)), f"ComfyUI: running {name}")
        elif event_type == 'progress':
            self._emit_progress(data.get('node'), data.get('value'), data.get('max'))
        elif event_type == 'progress_state':
            for node_id, state in (data.get('nodes') or {}).items():
                if state.get('state') == 'running':
                    self._emit_progress(
                        state.get('display_node_id', node_id),
                        state.get('value'), state.get('max'))
        elif event_type == 'execution_error':
            detail = data.get('exception_message') or data.get('exception_type')
            suffix = f": {detail}" if detail else ''
            self._emit(('error', str(detail)), f"ComfyUI: execution failed{suffix}")

    def _emit_progress(self, node_id: Any, value: Any, maximum: Any) -> None:
        try:
            value_number = float(value)
            maximum_number = float(maximum)
        except (TypeError, ValueError):
            return
        if maximum_number <= 0:
            return
        display_value = int(value_number) if value_number.is_integer() else value_number
        display_max = int(maximum_number) if maximum_number.is_integer() else maximum_number
        percent = round(value_number / maximum_number * 100)
        name = self.node_names.get(str(node_id), f'node {node_id}')
        self._emit(
            ('progress', str(node_id), value_number, maximum_number),
            f"ComfyUI: {name} {display_value}/{display_max} ({percent}%)",
        )

    def _queue_state(self, queue: dict[str, Any]) -> str | None:
        for key, label in (('queue_running', 'running'), ('queue_pending', 'queued')):
            for position, item in enumerate(queue.get(key, []), start=1):
                if isinstance(item, dict):
                    item_prompt_id = item.get('prompt_id')
                else:
                    item_prompt_id = item[1] if len(item) > 1 else None
                if item_prompt_id == self.prompt_id:
                    return label if label == 'running' else f'queued (position {position})'
        return None

    def _emit(self, key: tuple[Any, ...], message: str) -> None:
        if key == self._last_message:
            return
        self._last_message = key
        print(message, flush=True)
