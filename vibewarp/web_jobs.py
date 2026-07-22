"""Single-GPU background job management for the local web application."""

from __future__ import annotations

import copy
import io
import os
import sys
import threading
import traceback
import uuid
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from vibewarp.config import RunConfig
from vibewarp.cancellation import cancellation_scope
from vibewarp.utils.misc import release_vram


TERMINAL_STATES = {"completed", "cancelled", "failed"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    id: str
    config: RunConfig
    state: str = "queued"
    stage: str = "queued"
    message: str = "Waiting for the renderer"
    frame: int = 0
    total_frames: int = 0
    progress: float = 0.0
    created_at: str = field(default_factory=_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    cancel_requested: bool = False
    error: Optional[str] = None
    # Full traceback of a failed render. `error` is the one-line summary the status bar
    # shows; this is what you actually need to debug it.
    traceback: Optional[str] = field(default=None, repr=False)
    artifacts: List[str] = field(default_factory=list)
    preview_path: Optional[str] = field(default=None, repr=False)
    run_dir: Optional[str] = field(default=None, repr=False)
    logs: List[str] = field(default_factory=list)
    revision: int = 0
    operation: str = "render"
    resume_from: int = 0
    condition: threading.Condition = field(default_factory=threading.Condition, repr=False)

    def snapshot(self, *, include_config: bool = False) -> dict:
        data = {
            "id": self.id, "state": self.state, "stage": self.stage,
            "message": self.message, "frame": self.frame,
            "total_frames": self.total_frames, "progress": self.progress,
            "created_at": self.created_at, "started_at": self.started_at,
            "finished_at": self.finished_at, "cancel_requested": self.cancel_requested,
            "error": self.error, "traceback": self.traceback,
            "artifacts": list(self.artifacts),
            "preview_available": bool(self.preview_path),
            "logs": list(self.logs), "revision": self.revision,
            "operation": self.operation,
        }
        if include_config:
            data["config"] = asdict(self.config)
        return data


class _LineWriter(io.TextIOBase):
    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback
        self.buffer = ""

    def write(self, text: str) -> int:
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.rstrip():
                self.callback(line.rstrip())
        return len(text)

    def flush(self) -> None:
        if self.buffer.rstrip():
            self.callback(self.buffer.rstrip())
        self.buffer = ""


class JobManager:
    """Owns a FIFO queue and one renderer thread."""

    def __init__(self, runner: Optional[Callable] = None, history_limit: int = 20):
        self._runner = runner
        self._history_limit = history_limit
        self._jobs: Dict[str, Job] = {}
        self._queue: List[str] = []
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._worker: Optional[threading.Thread] = None

    def submit(self, config: RunConfig) -> Job:
        job = Job(id=uuid.uuid4().hex, config=copy.deepcopy(config))
        with self._wake:
            self._jobs[job.id] = job
            self._queue.append(job.id)
            self._ensure_worker()
            self._wake.notify_all()
        return job

    def submit_resume(self, config: RunConfig, run_dir: str, resume_from: int) -> Job:
        job = Job(
            id=uuid.uuid4().hex,
            config=copy.deepcopy(config),
            operation="resume",
            resume_from=resume_from,
            run_dir=os.path.abspath(run_dir),
            message=f"Waiting to resume from frame {resume_from + 1}",
        )
        return self._enqueue(job)

    def submit_video(self, config: RunConfig, run_dir: str) -> Job:
        job = Job(
            id=uuid.uuid4().hex,
            config=copy.deepcopy(config),
            operation="video",
            run_dir=os.path.abspath(run_dir),
            message="Waiting to assemble video",
        )
        return self._enqueue(job)

    def _enqueue(self, job: Job) -> Job:
        with self._wake:
            self._jobs[job.id] = job
            self._queue.append(job.id)
            self._ensure_worker()
            self._wake.notify_all()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> List[dict]:
        with self._lock:
            return [job.snapshot() for job in reversed(list(self._jobs.values()))]

    def cancel(self, job_id: str) -> Optional[Job]:
        with self._wake:
            job = self._jobs.get(job_id)
            if not job or job.state in TERMINAL_STATES:
                return job
            job.cancel_requested = True
            if job.state == "queued":
                self._queue = [queued for queued in self._queue if queued != job_id]
                job.state = job.stage = "cancelled"
                job.message = "Cancelled before rendering"
                job.finished_at = _now()
            else:
                job.message = "Cancellation requested"
            self._touch(job)
            return job

    def wait_for_change(self, job: Job, revision: int, timeout: float = 15) -> dict:
        with job.condition:
            if job.revision <= revision and job.state not in TERMINAL_STATES:
                job.condition.wait(timeout)
            return job.snapshot()

    def _ensure_worker(self) -> None:
        if not self._worker or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._work, name="vibewarp-renderer", daemon=True)
            self._worker.start()

    def _touch(self, job: Job) -> None:
        with job.condition:
            job.revision += 1
            job.condition.notify_all()

    def _log(self, job: Job, line: str) -> None:
        with self._lock:
            marker = "Run directory:"
            if marker in line:
                job.run_dir = os.path.abspath(line.partition(marker)[2].strip())
            job.logs.append(line)
            job.logs[:] = job.logs[-500:]
            self._touch(job)

    def _work(self) -> None:
        while True:
            with self._wake:
                if not self._queue:
                    return
                job = self._jobs[self._queue.pop(0)]
                if job.cancel_requested:
                    continue
                job.state, job.stage = "running", "starting"
                job.message = (
                    "Starting video assembly" if job.operation == "video"
                    else "Starting resumed render" if job.operation == "resume"
                    else "Starting render")
                job.started_at = _now()
                self._touch(job)
            self._run(job)
            self._prune()

    def _run(self, job: Job) -> None:
        def update_preview(frame: int) -> None:
            if not job.run_dir:
                return
            candidate = os.path.join(
                job.run_dir,
                f"{job.config.batch_name}(0)_{frame:06d}.{job.config.video.save_img_format}",
            )
            if os.path.isfile(candidate):
                job.preview_path = os.path.abspath(candidate)

        def progress(frame: int, total: int) -> None:
            if job.cancel_requested:
                raise KeyboardInterrupt("Cancelled by user")
            with self._lock:
                job.stage = "rendering"
                job.frame, job.total_frames = frame, total
                job.progress = min(100.0, frame / total * 100) if total else 0.0
                selected = job.config.frame_range
                source_start = (job.resume_from if job.operation == "resume" else
                                selected[0] if selected and selected != [0, 0] else 0)
                # Config/source frames are zero-based; the UI describes them in
                # the same human one-based numbering used for extracted frames.
                source_frame = source_start + max(0, frame - 1) + 1
                job.message = (
                    f"Rendering source frame {source_frame} ({frame}/{total})")
                update_preview(source_frame - 1)
                self._touch(job)

        writer = _LineWriter(lambda line: self._log(job, line))
        try:
            with redirect_stdout(writer), redirect_stderr(writer):
                with cancellation_scope(lambda: job.cancel_requested):
                    if job.operation == "video":
                        from vibewarp.pipeline import assemble_video
                        with self._lock:
                            job.stage = "assembling"
                            job.message = "Assembling video from completed frames"
                            self._touch(job)
                        paths = [assemble_video(job.config, job.run_dir)]
                    else:
                        runner = self._runner
                        if runner is None:
                            from vibewarp.pipeline import run
                            runner = run
                        kwargs = {"progress_fn": progress}
                        if job.operation == "resume":
                            kwargs.update(
                                resume_from=job.resume_from,
                                existing_run_dir=job.run_dir,
                            )
                        paths = runner(job.config, **kwargs)
            writer.flush()
            with self._lock:
                job.artifacts = [os.path.abspath(path) for path in (paths or [])]
                if job.artifacts:
                    job.preview_path = job.artifacts[-1]
                job.state = job.stage = "completed"
                job.progress = 100.0
                job.message = (
                    "Video is ready" if job.operation == "video" else
                    f"Completed with {len(job.artifacts)} rendered frames")
        except KeyboardInterrupt:
            with self._lock:
                job.state = job.stage = "cancelled"
                job.message = "Render cancelled"
        except BaseException as exc:
            # Flush whatever the pipeline had buffered FIRST, so the traceback lands after
            # the output that led to it instead of jumping ahead of it.
            writer.flush()
            report = traceback.format_exc()
            with self._lock:
                job.state = job.stage = "failed"
                job.error = str(exc)
                job.message = f"Render failed: {exc}"
                job.traceback = report
                # One entry per line: as a single blob the log dock's collapsed view showed
                # only the first line, and the 500-entry cap could swallow the whole thing.
                self._log(job, "")
                self._log(job, f"RENDER FAILED: {type(exc).__name__}: {exc}")
                for line in report.rstrip().splitlines():
                    self._log(job, line)
            # ...and to the server console, which the browser log never reaches.
            print(report, file=sys.__stderr__, flush=True)
        finally:
            writer.flush()
            release_vram()
            with self._lock:
                job.finished_at = _now()
                self._touch(job)

    def _prune(self) -> None:
        with self._lock:
            terminal = [job for job in self._jobs.values() if job.state in TERMINAL_STATES]
            for job in terminal[:-self._history_limit]:
                self._jobs.pop(job.id, None)
