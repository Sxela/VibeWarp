"""ComfyUI WebSocket event formatting and HTTP fallback behavior."""

from types import SimpleNamespace

from vibewarp.core.comfy_progress import ComfyProgressMonitor


def _monitor():
    monitor = ComfyProgressMonitor(
        'http://127.0.0.1:8188', 'client id',
        {
            '1': {'class_type': 'CheckpointLoaderSimple'},
            '13': {'class_type': 'SamplerCustom'},
        },
    )
    monitor.set_prompt_id('prompt-123')
    return monitor


def test_websocket_url_tracks_server_scheme_path_and_client():
    monitor = ComfyProgressMonitor(
        'https://example.test/comfy', 'client id', {})

    assert monitor._websocket_url() == \
        'wss://example.test/comfy/ws?clientId=client%20id'


def test_execution_events_report_nodes_and_sampler_steps(capsys):
    monitor = _monitor()
    capsys.readouterr()

    monitor._handle_event({
        'type': 'executing',
        'data': {'prompt_id': 'prompt-123', 'node': '1'},
    })
    monitor._handle_event({
        'type': 'progress',
        'data': {
            'prompt_id': 'prompt-123', 'node': '13',
            'value': 12, 'max': 40,
        },
    })
    monitor._handle_event({
        'type': 'executing',
        'data': {'prompt_id': 'someone-else', 'node': '13'},
    })

    output = capsys.readouterr().out
    assert 'ComfyUI: running CheckpointLoaderSimple' in output
    assert 'ComfyUI: SamplerCustom 12/40 (30%)' in output
    assert output.count('SamplerCustom') == 1


def test_new_progress_state_event_reports_running_node(capsys):
    monitor = _monitor()
    capsys.readouterr()

    monitor._handle_event({
        'type': 'progress_state',
        'data': {
            'prompt_id': 'prompt-123',
            'nodes': {
                '1': {'state': 'finished', 'value': 1, 'max': 1},
                '13': {'state': 'running', 'value': 7, 'max': 40},
            },
        },
    })

    assert 'ComfyUI: SamplerCustom 7/40 (18%)' in capsys.readouterr().out


def test_http_fallback_reports_pending_queue_position(capsys):
    monitor = _monitor()
    capsys.readouterr()
    monitor._connected = False
    response = SimpleNamespace(json=lambda: {
        'queue_running': [[1, 'other-prompt']],
        'queue_pending': [[2, 'other-pending'], [3, 'prompt-123']],
    })

    monitor.poll(lambda *args, **kwargs: response)

    assert 'ComfyUI: queued (position 2)' in capsys.readouterr().out


def test_cancel_deletes_pending_prompt(capsys):
    monitor = _monitor()
    capsys.readouterr()
    calls = []

    def request(method, path, **kwargs):
        calls.append((method, path, kwargs.get('json')))
        return SimpleNamespace(json=lambda: {
            'queue_running': [],
            'queue_pending': [[1, 'prompt-123']],
        })

    monitor.cancel(request)

    assert ('POST', '/queue', {'delete': ['prompt-123']}) in calls
    assert 'removed cancelled prompt' in capsys.readouterr().out


def test_cancel_targets_running_prompt_for_interrupt(capsys):
    monitor = _monitor()
    capsys.readouterr()
    calls = []

    def request(method, path, **kwargs):
        calls.append((method, path, kwargs.get('json')))
        return SimpleNamespace(json=lambda: {
            'queue_running': [[1, 'prompt-123']],
            'queue_pending': [],
        })

    monitor.cancel(request)

    assert ('POST', '/interrupt', {'prompt_id': 'prompt-123'}) in calls
    assert 'interrupted cancelled prompt' in capsys.readouterr().out
