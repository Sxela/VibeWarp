import hashlib
import threading

import pytest
import requests

from vibewarp.acquisition import acquire, download_scope, install_download_hooks
from vibewarp.cancellation import cancellation_scope


class Response:
    def __init__(self, data, status=200, headers=None):
        self.data, self.status_code = data, status
        self.headers = headers if headers is not None else {'Content-Length': str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, size):
        yield self.data


def test_atomic_download_reports_bytes_and_checks_hash(tmp_path, monkeypatch):
    dest = tmp_path / 'model.pt'
    events = []
    def get(*args, **kwargs):
        assert not dest.exists()
        return Response(b'weights')
    monkeypatch.setattr(requests, 'get', get)
    with download_scope(events.append):
        acquire('https://example.org/model.pt', dest, sha256=hashlib.sha256(b'weights').hexdigest())
    assert dest.read_bytes() == b'weights'
    assert not dest.with_suffix('.pt.part').exists()
    assert events[-1].status == 'ready'
    assert events[-1].downloaded == events[-1].total == 7
    assert any(e.status == 'verifying' for e in events)


@pytest.mark.parametrize('resumed', [True, False])
def test_resume_handles_both_valid_range_and_ignored_range(tmp_path, monkeypatch, resumed):
    dest = tmp_path / 'model.pt'
    (tmp_path / 'model.pt.part').write_bytes(b'ab')
    def get(*args, **kwargs):
        assert kwargs['headers']['Range'] == 'bytes=2-'
        return Response(b'cd', 206, {'Content-Range': 'bytes 2-3/4'}) if resumed else Response(b'abcd')
    monkeypatch.setattr(requests, 'get', get)
    acquire('https://example.org/model.pt', dest)
    assert dest.read_bytes() == b'abcd'


def test_truncated_response_does_not_publish_model(tmp_path, monkeypatch):
    monkeypatch.setattr(requests, 'get', lambda *a, **k: Response(b'a', headers={'Content-Length': '4'}))
    dest = tmp_path / 'model.pt'
    with pytest.raises(requests.RequestException, match='Incomplete'):
        acquire('https://example.org/model.pt', dest, retries=1)
    assert not dest.exists()
    assert (tmp_path / 'model.pt.part').read_bytes() == b'a'


def test_bad_checksum_is_never_published(tmp_path, monkeypatch):
    monkeypatch.setattr(requests, 'get', lambda *a, **k: Response(b'bad'))
    with pytest.raises(ValueError, match='SHA256'):
        acquire('https://example.org/model.pt', tmp_path / 'model.pt', sha256='0'*64, retries=1)
    assert not list(tmp_path.iterdir())


def test_cancelled_transfer_keeps_partial_and_releases_lock(tmp_path, monkeypatch):
    calls = []
    def get(*args, **kwargs):
        calls.append(1)
        return Response(b'weights')
    monkeypatch.setattr(requests, 'get', get)
    with cancellation_scope(lambda: True), pytest.raises(KeyboardInterrupt):
        acquire('https://example.org/model.pt', tmp_path / 'model.pt')
    acquire('https://example.org/model.pt', tmp_path / 'model.pt')
    assert len(calls) == 1


def test_two_consumers_share_one_transfer(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []
    def get(*args, **kwargs):
        calls.append(1)
        entered.set()
        assert release.wait(2)
        return Response(b'weights')
    monkeypatch.setattr(requests, 'get', get)
    dest = tmp_path / 'model.pt'
    threads = [threading.Thread(target=acquire, args=('https://example.org/model.pt', dest)) for _ in range(2)]
    for thread in threads:
        thread.start()
    assert entered.wait(1)
    release.set()
    for thread in threads:
        thread.join(2)
        assert not thread.is_alive()
    assert len(calls) == 1
    assert dest.read_bytes() == b'weights'


def test_hf_hook_observes_progress_without_changing_torch_outside_scope(monkeypatch):
    import io
    from contextlib import contextmanager
    import httpx
    import huggingface_hub.file_download as hf
    install_download_hooks()
    events = []
    @contextmanager
    def response(**kwargs):
        yield httpx.Response(200, content=b'weights', headers={'Content-Length':'7'},
                             request=httpx.Request('GET', kwargs['url']))
    monkeypatch.setattr(hf, 'http_stream_backoff', response)
    target = io.BytesIO()
    import huggingface_hub._snapshot_download as snapshots
    with download_scope(events.append):
        snapshots.hf_thread_map(lambda url: hf.http_get(url, target, expected_size=7),
            ['https://example.org/model.bin'], max_workers=1, disable=True)
    assert target.getvalue() == b'weights'
    assert events[-1].total == events[-1].downloaded == 7
    assert events[-1].status == 'ready'
