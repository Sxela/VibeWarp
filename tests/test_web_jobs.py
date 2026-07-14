import time
from threading import Event

from vibewarp.config import RunConfig
from vibewarp.web_jobs import JobManager


def wait_terminal(job, timeout=2):
    deadline = time.time() + timeout
    while job.state not in {"completed", "cancelled", "failed"} and time.time() < deadline:
        time.sleep(0.01)
    return job


def test_job_runs_and_records_progress_and_artifacts(tmp_path):
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"image")
    def runner(config, progress_fn):
        print("loading model")
        progress_fn(1, 2); progress_fn(2, 2)
        return [str(frame)]
    manager = JobManager(runner=runner)
    job = wait_terminal(manager.submit(RunConfig()))
    assert job.state == "completed"
    assert job.progress == 100
    assert job.artifacts == [str(frame.resolve())]
    assert "loading model" in job.logs


def test_queued_job_can_be_cancelled():
    release = False
    def runner(config, progress_fn):
        nonlocal release
        while not release: time.sleep(0.01)
        return []
    manager = JobManager(runner=runner)
    first = manager.submit(RunConfig())
    second = manager.submit(RunConfig())
    manager.cancel(second.id)
    assert second.state == "cancelled"
    release = True
    wait_terminal(first)


def test_running_job_cancels_at_progress_boundary():
    def runner(config, progress_fn):
        for index in range(100):
            time.sleep(0.005); progress_fn(index, 100)
        return []
    manager = JobManager(runner=runner)
    job = manager.submit(RunConfig())
    deadline = time.time() + 1
    while job.state == "queued" and time.time() < deadline: time.sleep(0.005)
    manager.cancel(job.id)
    assert wait_terminal(job).state == "cancelled"


def test_preview_is_published_before_job_completes(tmp_path):
    rendered = Event()
    release = Event()
    def runner(config, progress_fn):
        print(f"  Run directory: {tmp_path}")
        frame = tmp_path / "warpfusion(0)_000000.png"
        frame.write_bytes(b"preview")
        progress_fn(0, 2)
        rendered.set()
        release.wait(1)
        return [str(frame)]
    manager = JobManager(runner=runner)
    job = manager.submit(RunConfig())
    assert rendered.wait(1)
    assert job.state == "running"
    assert job.preview_path == str((tmp_path / "warpfusion(0)_000000.png").resolve())
    assert job.snapshot()["preview_available"] is True
    release.set()
    wait_terminal(job)


def test_failed_render_logs_the_whole_traceback():
    """The status bar gets one line; the log needs the whole thing.

    A failure previously appended the traceback as ONE multi-line entry, so the collapsed
    log dock showed only its first line and the 500-entry cap could swallow it whole.
    """
    import time
    from vibewarp.config import RunConfig
    from vibewarp.web_jobs import JobManager

    def boom(config, progress_fn=None):
        print("pipeline output before the crash")
        value = None
        return value.float()

    manager = JobManager(runner=boom)
    job = manager.submit(RunConfig())
    for _ in range(200):
        if job.state in ("failed", "completed", "cancelled"):
            break
        time.sleep(0.02)

    snapshot = job.snapshot()
    assert snapshot["state"] == "failed"
    assert snapshot["message"] == (
        "Render failed: 'NoneType' object has no attribute 'float'")
    # the full traceback is its own field...
    assert "AttributeError" in snapshot["traceback"]
    assert "in boom" in snapshot["traceback"]
    # ...and is in the log, one entry per line, AFTER the output that led to it
    logs = snapshot["logs"]
    assert "pipeline output before the crash" in logs
    assert any(line.startswith("RENDER FAILED: AttributeError") for line in logs)
    assert any(line.startswith("Traceback (most recent call last)") for line in logs)
    assert logs.index("pipeline output before the crash") < len(logs) - 1


def test_vram_is_released_after_every_job(monkeypatch):
    """The server renders back to back in ONE process.

    Without a cleanup at the job boundary the next render starts against a VRAM ceiling
    set by the last one, and the resulting OOM reads as "these settings are too big"
    rather than "we never let go of the previous run". It must run on failure too --
    a job that OOMs is exactly the one whose memory you need back.
    """
    import time
    from vibewarp.config import RunConfig
    from vibewarp import web_jobs

    calls = []
    monkeypatch.setattr(web_jobs, 'release_vram', lambda *a, **k: calls.append(1))

    for runner in (lambda config, progress_fn=None: [],            # succeeds
                   lambda config, progress_fn=None: 1 / 0):        # fails
        manager = web_jobs.JobManager(runner=runner)
        job = manager.submit(RunConfig())
        for _ in range(200):
            if job.state in ("failed", "completed", "cancelled"):
                break
            time.sleep(0.02)

    assert len(calls) == 2, 'VRAM must be released after a failed job too'
