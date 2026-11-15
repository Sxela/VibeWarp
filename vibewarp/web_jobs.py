"""Single-GPU background job management for the local web application."""

from __future__ import annotations

import copy
import io
import os
import re
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
from vibewarp.acquisition import download_scope, install_download_hooks
from vibewarp.render_events import render_event_scope
from vibewarp.log_progress import _ANSI, render_eta
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
    eta_seconds: Optional[float] = None
    eta_updated_at: Optional[str] = None
    downloads: Dict[str, dict] = field(default_factory=dict)
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
    preview_frame: int = 0
    preview_reference_path: str = ''
    # Progress-bar description -> absolute log position of that bar's line, so a
    # carriage-return redraw replaces it in place (see _LineWriter).
    _progress_lines: Dict[Optional[str], int] = field(default_factory=dict, repr=False)
    _log_count: int = field(default=0, repr=False)
    condition: threading.Condition = field(default_factory=threading.Condition, repr=False)

    def snapshot(self, *, include_config: bool = False) -> dict:
        run_id = os.path.basename(self.run_dir) if self.run_dir else None
        data = {
            "id": self.id, "state": self.state, "stage": self.stage,
            "message": self.message, "frame": self.frame,
            "total_frames": self.total_frames, "progress": self.progress,
            "eta_seconds": self.eta_seconds if self.state == 'running' and self.stage != 'cleanup' and not self.cancel_requested else None,
            "eta_updated_at": self.eta_updated_at,
            "downloads": list(self.downloads.values()),
            "created_at": self.created_at, "started_at": self.started_at,
            "finished_at": self.finished_at, "cancel_requested": self.cancel_requested,
            "error": self.error, "traceback": self.traceback,
            "artifacts": list(self.artifacts),
            "preview_available": bool(self.preview_path),
            "logs": list(self.logs), "revision": self.revision,
            "operation": self.operation,
            # Lets History attach live progress to the run card without
            # exposing an absolute server-side output path.
            "run_id": run_id,
        }
        if include_config:
            data["config"] = asdict(self.config)
        return data


CR, LF = '\r', '\n'
_PROGRESS_TAIL = re.compile(r'\s*(?:\d+(?:\.\d+)?%|\d+/\d+).*$')


def progress_key(line: str) -> str:
    """Which progress bar a carriage-return update belongs to: its description.

    'Rendering frames: 3/101 [...]' -> 'Rendering frames:'; a bar with no
    description (' 40%|####  | 20/50 ...') -> ''.
    """
    return _PROGRESS_TAIL.sub('', line).strip()


class _LineWriter(io.TextIOBase):
    """Splits stream writes into log lines.

    A segment that is written with or ends in a carriage return is a progress-bar
    redraw (tqdm). Those go to `progress_callback`, which is expected to replace the
    bar's previous line rather than add another, so a 50-step sampler does not push
    the real log out of the 500-line window.
    """

    def __init__(self, callback: Callable[[str], None],
                 progress_callback: Optional[Callable[[str], None]] = None):
        self.callback = callback
        self.progress_callback = progress_callback or callback
        self.buffer = ""
        self._after_cr = False

    def _emit(self, line: str, progress: bool) -> None:
        line = line.rstrip()
        # tqdm prefixes redraws with an erase-line escape; that alone is not a line.
        if _ANSI.sub('', line).strip():
            (self.progress_callback if progress else self.callback)(line)

    def write(self, text: str) -> int:
        self.buffer += text
        while True:
            # Hold a trailing CR back: it may be the first half of a split CRLF.
            cut = next((i for i, char in enumerate(self.buffer)
                        if char == LF or (char == CR and i + 1 < len(self.buffer))), None)
            if cut is None:
                break
            line, char = self.buffer[:cut], self.buffer[cut]
            if char == CR and self.buffer[cut + 1] == LF:
                char, cut = LF, cut + 1
            self.buffer = self.buffer[cut + 1:]
            if char == CR:
                self._emit(line, True)
                self._after_cr = True
            else:
                # A bar's final redraw ends in LF but still follows a CR.
                self._emit(line, self._after_cr)
                self._after_cr = False
        return len(text)

    def flush(self) -> None:
        self._emit(self.buffer, self._after_cr or self.buffer.endswith(CR))
        self.buffer = ""
        self._after_cr = False


class JobManager:
    """Owns a FIFO queue and one renderer thread."""

    def __init__(self, runner: Optional[Callable] = None, history_limit: int = 20):
        self._runner = runner
        self.download_manager = None
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

    def submit_raw_preview(self, config: RunConfig, frame: int, reference_path: str = '') -> Job:
        return self._enqueue(Job(id=uuid.uuid4().hex,config=copy.deepcopy(config),
            operation='raw_preview',preview_frame=frame,preview_reference_path=reference_path,
            message='Waiting to preview raw frame preparation'))

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

    def _log(self, job: Job, line: str, *, progress: bool = False) -> None:
        with self._lock:
            marker = "Run directory:"
            if marker in line:
                job.run_dir = os.path.abspath(line.partition(marker)[2].strip())
            outer, remaining = render_eta(line)
            if outer:
                job.eta_seconds = remaining
                job.eta_updated_at = _now()
            key = progress_key(line) if progress else None
            # Absolute position (lines ever logged) survives the 500-line trim below.
            dropped = job._log_count - len(job.logs)
            position = job._progress_lines.get(key) if progress else None
            if position is not None and position - dropped >= 0:
                job.logs[position - dropped] = line
            else:
                job.logs.append(line)
                job._log_count += 1
                if progress:
                    job._progress_lines[key] = job._log_count - 1
                excess = len(job.logs) - 500
                if excess > 0:
                    del job.logs[:excess]
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
                    'Preparing raw video frame' if job.operation == 'raw_preview' else
                    "Starting video assembly" if job.operation == "video"
                    else "Starting resumed render" if job.operation == "resume"
                    else "Starting render")
                job.started_at = _now()
                self._touch(job)
            self._run(job)
            self._prune()

    def _run(self, job: Job) -> None:
        terminal_state = 'failed'
        def step_progress(event):
            with self._lock:
                job.stage = 'sampling'
                job.message = f"Frame {event['frame']+1}: denoising {event['step']}/{event['total_steps']}"
                if event.get('preview'):
                    job.preview_path = event['preview']
                self._touch(job)

        def download_progress(event):
            if self.download_manager is not None:
                self.download_manager.runtime_progress(event, job.id)
            with self._lock:
                job.downloads[event.path] = event.to_dict()
                if event.status not in ('ready', 'paused'):
                    job.stage = 'downloading'
                    job.message = f'Preparing {event.name}'
                self._touch(job)

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

        writer = _LineWriter(lambda line: self._log(job, line),
                             lambda line: self._log(job, line, progress=True))
        try:
            with redirect_stdout(writer), redirect_stderr(writer):
                with cancellation_scope(lambda: job.cancel_requested), download_scope(download_progress), render_event_scope(step_progress):
                    install_download_hooks()
                    if job.operation == 'raw_preview':
                        from vibewarp.video.raw_processing import preview_frame
                        paths = [preview_frame(job.config,job.preview_frame,job.preview_reference_path)]
                    elif job.operation == "video":
                        from vibewarp.pipeline import assemble_video
                        with self._lock:
                            job.stage = "assembling"
                            job.message = "Assembling video from completed frames"
                            self._touch(job)
                        paths = [assemble_video(job.config, job.run_dir)]
                    else:
                        runner = self._runner
                        if runner is None:
                            if self.download_manager is not None:
                                self.download_manager.wait_for_render(
                                    job.config, f'render:{job.id}', download_progress)
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
                terminal_state = 'completed'
                job.stage = 'cleanup'
                job.progress = 100.0
                job.message = (
                    'Raw frame preview is ready' if job.operation == 'raw_preview' else
                    "Video is ready" if job.operation == "video" else
                    f"Completed with {len(job.artifacts)} rendered frames")
        except KeyboardInterrupt:
            with self._lock:
                terminal_state = 'cancelled'
                job.stage = 'cleanup'
                job.message = "Render cancelled"
        except BaseException as exc:
            # Flush whatever the pipeline had buffered FIRST, so the traceback lands after
            # the output that led to it instead of jumping ahead of it.
            writer.flush()
            report = traceback.format_exc()
            with self._lock:
                terminal_state = 'failed'
                job.stage = 'cleanup'
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
            # Publish failures/cancellation before touching a potentially broken
            # CUDA context; cleanup must never strand a job as "running".
            if terminal_state in ('failed','cancelled'):
                with self._lock:
                    job.state = job.stage = terminal_state
                    job.finished_at = _now()
                    self._touch(job)
            if self.download_manager is not None:
                try:
                    self.download_manager.finish_runtime(job.id, terminal_state)
                except Exception as exc:
                    self._log(job,f'Download cleanup failed: {exc}')
            try:
                if terminal_state in ('failed','cancelled'):
                    release_vram(synchronize=False)
                else:
                    release_vram()
            except Exception as exc:
                self._log(job,f'GPU cleanup failed: {exc}')
                if terminal_state == 'completed':
                    terminal_state = 'failed'
                    job.error = str(exc)
                    job.message = f'GPU cleanup failed: {exc}'
                    job.traceback = traceback.format_exc()
            with self._lock:
                job.state = job.stage = terminal_state
                job.finished_at = _now()
                self._touch(job)

    def _prune(self) -> None:
        with self._lock:
            terminal = [job for job in self._jobs.values() if job.state in TERMINAL_STATES]
            for job in terminal[:-self._history_limit]:
                self._jobs.pop(job.id, None)
