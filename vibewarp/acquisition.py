"""Resumable model acquisition and structured byte progress for CLI and UI.

Only completed files are published at their final name. Transfers to the same
destination share a lock, including selection-time prefetch and the renderer.
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import requests

from vibewarp.cancellation import raise_if_cancelled, cancellation_check, cancellation_scope


@dataclass(frozen=True)
class DownloadEvent:
    name: str
    path: str
    downloaded: int = 0
    total: int | None = None
    status: str = 'downloading'
    speed: float = 0.0
    error: str = ''

    def to_dict(self):
        return asdict(self)


_state = threading.local()
_locks: dict[str, threading.Lock] = {}
_guard = threading.Lock()
_verified = {}


def _callbacks():
    local = getattr(_state, 'callback', None)
    return [local] if local else []


def report(event: DownloadEvent):
    raise_if_cancelled()
    callback = getattr(_state, 'callback', None)
    if callback:
        callback(event)


@contextmanager
def download_scope(callback: Callable[[DownloadEvent], None]):
    previous = getattr(_state, 'callback', None)
    _state.callback = callback
    try:
        yield
    finally:
        _state.callback = previous


def _checksum(path: Path, expected: str):
    stat = path.stat()
    identity = (str(path), stat.st_size, stat.st_mtime_ns, expected)
    with _guard:
        if identity in _verified:
            return
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            raise_if_cancelled()
            digest.update(chunk)
    if digest.hexdigest().lower() != expected.lower():
        raise ValueError(f'SHA256 mismatch for {path.name}')
    with _guard:
        _verified[identity] = True


def acquire(url: str, destination: str | Path, *, sha256: str = '',
            retries: int = 3, headers: dict | None = None) -> str:
    """Download HTTP(S) weights, resuming only with a validated Content-Range.

    An ignored Range restarts the partial file. A short response is never
    treated as complete. Existing files can be verified against a pinned hash.
    """
    if not url.startswith(('https://', 'http://')):
        raise ValueError('Model downloads require an HTTP(S) URL')
    if url.startswith('https://huggingface.co/'):
        from huggingface_hub.utils import build_hf_headers
        headers = {**build_hf_headers(), **(headers or {})}
    dest = Path(destination).resolve()
    with _guard:
        lock = _locks.setdefault(os.path.normcase(str(dest)), threading.Lock())
    while not lock.acquire(timeout=0.2):
        report(DownloadEvent(dest.name, str(dest), status='waiting'))
    try:
        if dest.is_file() and dest.stat().st_size:
            size = dest.stat().st_size
            if sha256:
                report(DownloadEvent(dest.name, str(dest), size, size, 'verifying'))
                _checksum(dest, sha256)
            report(DownloadEvent(dest.name, str(dest), size, size, 'ready'))
            return str(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + '.part')
        for attempt in range(retries):
            raise_if_cancelled()
            offset = part.stat().st_size if part.exists() else 0
            request_headers = {'Accept-Encoding': 'identity', **(headers or {})}
            if offset:
                request_headers['Range'] = f'bytes={offset}-'
            try:
                with requests.get(url, headers=request_headers, stream=True,
                                  timeout=(15, 30)) as response:
                    # A stale/oversized partial is restarted on the next attempt.
                    if response.status_code == 416:
                        part.unlink(missing_ok=True)
                        raise requests.RequestException('Server rejected partial download')
                    response.raise_for_status()
                    total = None
                    if response.status_code == 206:
                        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',
                                             response.headers.get('Content-Range', ''))
                        if not match or int(match[1]) != offset:
                            part.unlink(missing_ok=True)
                            raise requests.RequestException('Invalid download Content-Range')
                        total = int(match[3])
                    else:
                        offset = 0
                        length = response.headers.get('Content-Length')
                        if length:
                            total = int(length)
                    started = last = time.monotonic()
                    downloaded = offset
                    report(DownloadEvent(dest.name, str(dest), downloaded, total))
                    with part.open('ab' if offset else 'wb') as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            raise_if_cancelled()
                            if not chunk:
                                continue
                            stream.write(chunk)
                            downloaded += len(chunk)
                            now = time.monotonic()
                            if now - last >= 0.1:
                                report(DownloadEvent(dest.name, str(dest), downloaded,
                                       total, speed=(downloaded-offset)/max(now-started, .001)))
                                last = now
                    if total is not None and downloaded != total:
                        raise requests.RequestException('Incomplete model download')
                    if not downloaded:
                        raise requests.RequestException('Empty model download')
                if sha256:
                    report(DownloadEvent(dest.name, str(dest), downloaded, total, 'verifying'))
                    try:
                        _checksum(part, sha256)
                    except ValueError:
                        part.unlink(missing_ok=True)
                        raise
                os.replace(part, dest)
                report(DownloadEvent(dest.name, str(dest), downloaded, downloaded, 'ready'))
                return str(dest)
            except (requests.RequestException, ValueError) as exc:
                if attempt + 1 == retries:
                    report(DownloadEvent(dest.name, str(dest), status='failed', error=str(exc)))
                    raise
                report(DownloadEvent(dest.name, str(dest), status='retrying', error=str(exc)))
                # Cooperative backoff so cancellation remains responsive.
                for _ in range(5 * (attempt + 1)):
                    raise_if_cancelled()
                    time.sleep(.1)
        raise RuntimeError('No download attempts configured')
    finally:
        lock.release()


def install_download_hooks():
    """Bridge third-party HF/Torch model downloads into the current job scope.

    Install once; outside a download scope each library keeps its original
    behavior. There are no process-wide UI callbacks to leak across jobs.
    """
    with _guard:
        import torch.hub
        if not getattr(torch.hub.download_url_to_file, '_vibewarp', False):
            original = torch.hub.download_url_to_file

            def torch_download(url, dst, hash_prefix=None, progress=True):
                if getattr(_state, 'callback', None) is None:
                    return original(url, dst, hash_prefix, progress)
                # Torch supplies short SHA prefixes; retain its validation by
                # downloading into a distinct file and checking before publish.
                path = acquire(url, str(dst) + '.transfer')
                if hash_prefix:
                    digest_hash = hashlib.sha256()
                    with open(path, 'rb') as stream:
                        while chunk := stream.read(4 * 1024 * 1024):
                            raise_if_cancelled()
                            digest_hash.update(chunk)
                    digest = digest_hash.hexdigest()
                    if not digest.startswith(hash_prefix):
                        Path(path).unlink(missing_ok=True)
                        raise ValueError(f'Invalid model hash for {Path(dst).name}')
                os.replace(path, dst)

            torch_download._vibewarp = True
            torch.hub.download_url_to_file = torch_download

        import huggingface_hub.file_download as hf
        if not getattr(hf.http_get, '_vibewarp', False):
            from huggingface_hub.utils import tqdm as base

            class DownloadTqdm(base):
                _vibewarp = True

                def __init__(self, *args, **kwargs):
                    self._download_callbacks = _callbacks()
                    self._download_cancel = cancellation_check()
                    self._download_last = 0.0
                    if self._download_callbacks:
                        kwargs['disable'] = False
                    super().__init__(*args, **kwargs)
                    if str(self.desc).endswith(': downloading bytes'):
                        # Xet has a second compressed-network counter. The
                        # reconstruction counter is the actual file progress.
                        self._download_callbacks = []
                    for callback in self._download_callbacks:
                        callback(DownloadEvent(str(self.desc or 'Model weights'),
                            getattr(self, '_download_path', '') or str(self.desc or 'hub'),
                            int(self.n), int(self.total) if self.total else None))

                def update(self, n=1):
                    result = super().update(n)
                    now = time.monotonic()
                    if self._download_callbacks and now - self._download_last >= .1:
                        if self._download_cancel():
                            raise KeyboardInterrupt('Cancelled by user')
                        raise_if_cancelled()
                        event = DownloadEvent(
                            str(self.desc or 'Model weights'), getattr(self, '_download_path', '') or str(self.desc or 'hub'),
                            int(self.n), int(self.total) if self.total else None)
                        for callback in self._download_callbacks:
                            callback(event)
                        self._download_last = now
                    return result

                def close(self):
                    callbacks = getattr(self, '_download_callbacks', [])
                    if callbacks and not getattr(self, '_download_closed', False):
                        self._download_closed = True
                        complete = self.total is not None and self.n >= self.total
                        event = DownloadEvent(str(self.desc or 'Model weights'),
                            getattr(self, '_download_path', '') or str(self.desc or 'hub'), int(self.n),
                            int(self.total) if self.total else None,
                            'ready' if complete else 'paused')
                        for callback in callbacks:
                            callback(event)
                    return super().close()

            def wrap_download(function):
                def download(*args, **kwargs):
                    if _callbacks():
                        target = kwargs.get('incomplete_path')
                        if target is None:
                            stream = kwargs.get('temp_file', args[1] if len(args) > 1 else None)
                            target = getattr(stream, 'name', '')
                        class TransferTqdm(DownloadTqdm):
                            _download_path = str(target or '')
                        kwargs['tqdm_class'] = TransferTqdm
                    return function(*args, **kwargs)
                download._vibewarp = True
                return download

            hf.http_get = wrap_download(hf.http_get)
            hf.xet_get = wrap_download(hf.xet_get)

        # Snapshot files run in a thread pool. Transfer only the submitting
        # job's observer/check; concurrent preparations must remain isolated.
        import huggingface_hub._snapshot_download as snapshots
        map_name = 'hf_thread_map' if hasattr(snapshots, 'hf_thread_map') else 'thread_map'
        original_map = getattr(snapshots, map_name)
        if not getattr(original_map, '_vibewarp', False):

            def scoped_map(function, *iterables, **kwargs):
                callback = getattr(_state, 'callback', None)
                if callback is None:
                    return original_map(function, *iterables, **kwargs)
                check = cancellation_check()
                def scoped_function(*args):
                    with cancellation_scope(check), download_scope(callback):
                        return function(*args)
                return original_map(scoped_function, *iterables, **kwargs)

            scoped_map._vibewarp = True
            setattr(snapshots, map_name, scoped_map)
