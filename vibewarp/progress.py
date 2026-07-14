"""Structured progress reporting for GUI integration.

Provides a ProgressReporter that emits typed events at key pipeline stages,
allowing GUIs (Gradio, Streamlit, etc.) to display real-time progress.

Usage:
    reporter = ProgressReporter()
    reporter.on_event(lambda e: print(e))
    ctx.progress_fn = reporter.frame_progress

    # Or use the callback-style API:
    reporter = ProgressReporter(
        on_stage=lambda s: update_status(s),
        on_frame=lambda f, t: update_bar(f, t),
    )
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class Stage(str, Enum):
    """Pipeline stages for progress reporting."""
    INIT = 'init'
    EXTRACT_FRAMES = 'extract_frames'
    LOAD_MODELS = 'load_models'
    LOAD_CONTROLNETS = 'load_controlnets'
    LOAD_MOTION_MODULE = 'load_motion_module'
    LOAD_IPADAPTERS = 'load_ipadapters'
    LOAD_FLOW_MODEL = 'load_flow_model'
    RENDERING = 'rendering'
    RENDER_BATCH = 'render_batch'
    RENDER_FRAME = 'render_frame'
    RECONSTRUCTION_NOISE = 'reconstruction_noise'
    WARPING = 'warping'
    CREATE_VIDEO = 'create_video'
    COMPLETE = 'complete'
    ERROR = 'error'


@dataclass
class ProgressEvent:
    """A progress event emitted during pipeline execution."""
    stage: Stage
    frame: int = 0
    total_frames: int = 0
    batch: int = 0
    total_batches: int = 0
    message: str = ''
    elapsed: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)

    @property
    def pct(self) -> float:
        """Overall progress as a percentage (0-100)."""
        if self.total_frames > 0:
            return min(100.0, (self.frame / self.total_frames) * 100)
        return 0.0


class ProgressReporter:
    """Emits structured progress events during pipeline execution.

    Can be wired into RenderContext.progress_fn for frame-level updates,
    and called directly for stage-level updates.
    """

    def __init__(
        self,
        on_event: Optional[Callable[[ProgressEvent], None]] = None,
        on_stage: Optional[Callable[[Stage], None]] = None,
        on_frame: Optional[Callable[[int, int], None]] = None,
    ):
        self._listeners: List[Callable[[ProgressEvent], None]] = []
        self._stage_listeners: List[Callable[[Stage], None]] = []
        self._frame_listeners: List[Callable[[int, int], None]] = []
        self._start_time = time.time()
        self._events: List[ProgressEvent] = []
        self._current_stage: Optional[Stage] = None

        if on_event:
            self._listeners.append(on_event)
        if on_stage:
            self._stage_listeners.append(on_stage)
        if on_frame:
            self._frame_listeners.append(on_frame)

    def on_event(self, callback: Callable[[ProgressEvent], None]) -> None:
        """Register a listener for all progress events."""
        self._listeners.append(callback)

    def on_stage(self, callback: Callable[[Stage], None]) -> None:
        """Register a listener for stage transitions."""
        self._stage_listeners.append(callback)

    def on_frame(self, callback: Callable[[int, int], None]) -> None:
        """Register a listener for frame completions."""
        self._frame_listeners.append(callback)

    def emit(self, event: ProgressEvent) -> None:
        """Emit a progress event to all listeners."""
        event.elapsed = time.time() - self._start_time
        self._events.append(event)
        for listener in self._listeners:
            listener(event)

    def stage(self, stage: Stage, message: str = '', **data) -> None:
        """Report a stage transition."""
        self._current_stage = stage
        event = ProgressEvent(stage=stage, message=message, data=data)
        self.emit(event)
        for listener in self._stage_listeners:
            listener(stage)

    def frame_progress(self, frame: int, total: int) -> None:
        """Report frame-level progress (compatible with ctx.progress_fn)."""
        event = ProgressEvent(
            stage=self._current_stage or Stage.RENDERING,
            frame=frame,
            total_frames=total,
            message=f'Frame {frame}/{total}',
        )
        self.emit(event)
        for listener in self._frame_listeners:
            listener(frame, total)

    def batch_progress(self, batch: int, total_batches: int, frame: int, total_frames: int) -> None:
        """Report batch-level progress for AnimateDiff."""
        event = ProgressEvent(
            stage=Stage.RENDER_BATCH,
            frame=frame,
            total_frames=total_frames,
            batch=batch,
            total_batches=total_batches,
            message=f'Batch {batch}/{total_batches}, frame {frame}/{total_frames}',
        )
        self.emit(event)

    def error(self, message: str, **data) -> None:
        """Report an error."""
        event = ProgressEvent(stage=Stage.ERROR, message=message, data=data)
        self.emit(event)

    def complete(self, total_frames: int = 0, output_dir: str = '') -> None:
        """Report pipeline completion."""
        event = ProgressEvent(
            stage=Stage.COMPLETE,
            frame=total_frames,
            total_frames=total_frames,
            message=f'Complete: {total_frames} frames rendered',
            data={'output_dir': output_dir},
        )
        self.emit(event)

    @property
    def events(self) -> List[ProgressEvent]:
        """All emitted events (for testing/debugging)."""
        return list(self._events)

    @property
    def elapsed(self) -> float:
        """Total elapsed time since reporter creation."""
        return time.time() - self._start_time
