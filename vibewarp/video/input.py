"""Video frame extraction and dataset utilities."""

import os
import subprocess
from glob import glob

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {'jpeg', 'jpg', 'png', 'tiff', 'bmp', 'webp'}


def fit_dimensions(width: int, height: int, max_size: int, multiple: int = 8):
    """Fit dimensions to a maximum edge while preserving aspect ratio."""
    if width <= 0 or height <= 0 or max_size <= 0:
        raise ValueError("Video dimensions and max_size must be positive")
    scale = min(1.0, max_size / max(width, height))
    return (
        max(multiple, round(width * scale / multiple) * multiple),
        max(multiple, round(height * scale / multiple) * multiple),
    )


def resolve_video_dimensions(video_path: str, max_size: int):
    """Read source dimensions and resolve render dimensions from max_size."""
    import cv2
    capture = cv2.VideoCapture(video_path)
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Could not read video dimensions: {video_path}")
    return fit_dimensions(width, height, max_size)


def probe_video(video_path: str) -> dict:
    """Source metadata for a video: dimensions, fps, frame count, duration, codec.

    Prefers ffprobe, which reads the container's own index. OpenCV's CAP_PROP_FRAME_COUNT
    is an estimate for several formats (it can be derived from duration x fps), and an
    over-count means the UI would offer frames that do not exist. Falls back to OpenCV when
    ffprobe is not installed, and says which one it used.
    """
    if not video_path or not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)

    probe = _find_ffprobe()
    if probe:
        try:
            return _probe_with_ffprobe(probe, video_path)
        except (subprocess.SubprocessError, ValueError, KeyError, OSError):
            pass          # a malformed stream is not a reason to fail; fall back
    return _probe_with_opencv(video_path)


def _find_ffprobe():
    import shutil
    found = shutil.which('ffprobe')
    if found:
        return found
    # ffprobe ships alongside ffmpeg; reuse the same search.
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        return None
    candidate = os.path.join(os.path.dirname(ffmpeg),
                             'ffprobe.exe' if ffmpeg.endswith('.exe') else 'ffprobe')
    return candidate if os.path.isfile(candidate) else None


def _probe_with_ffprobe(ffprobe: str, video_path: str) -> dict:
    import json
    result = subprocess.run(
        [ffprobe, '-v', 'error', '-select_streams', 'v:0', '-of', 'json',
         '-show_entries',
         'stream=width,height,avg_frame_rate,nb_frames,codec_name:format=duration',
         video_path],
        capture_output=True, text=True, check=True, timeout=30)
    data = json.loads(result.stdout)
    stream = (data.get('streams') or [{}])[0]

    width, height = int(stream['width']), int(stream['height'])
    # avg_frame_rate is a rational string like "30000/1001".
    numerator, _, denominator = (stream.get('avg_frame_rate') or '0/1').partition('/')
    fps = float(numerator) / float(denominator) if float(denominator or 0) else 0.0
    duration = float((data.get('format') or {}).get('duration') or 0.0)

    frames = int(stream.get('nb_frames') or 0)
    if frames <= 0 and fps > 0 and duration > 0:
        # Some containers (notably .webm) do not store nb_frames.
        frames = int(round(duration * fps))

    return {
        'width': width, 'height': height,
        'fps': round(fps, 3),
        'frames': frames,
        'duration': round(duration, 2),
        'codec': stream.get('codec_name') or '',
        'source': 'ffprobe',
    }


def _probe_with_opencv(video_path: str) -> dict:
    import cv2
    capture = cv2.VideoCapture(video_path)
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Could not read video dimensions: {video_path}")
    return {
        'width': width, 'height': height,
        'fps': round(fps, 3),
        'frames': max(0, frames),
        'duration': round(frames / fps, 2) if fps > 0 else 0.0,
        'codec': '',
        'source': 'opencv',
    }


def first_visible_frame(video_path: str, max_scan: int = 120, threshold: float = 8.0):
    """(index, RGB ndarray) of the first frame that is not essentially black.

    Plenty of clips open on a fade-in or a few black frames, and a black thumbnail tells
    you nothing about the video you just selected. Scans forward until a frame's mean
    luminance clears `threshold` (0-255), and gives up after `max_scan` frames, returning
    frame 0 — a genuinely black video should still show something rather than nothing.
    """
    import cv2

    capture = cv2.VideoCapture(video_path)
    try:
        first = None
        for index in range(max(1, max_scan)):
            ok, frame = capture.read()
            if not ok:
                break
            if first is None:
                first = (0, frame)
            if float(frame.mean()) >= threshold:
                return index, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if first is None:
            raise ValueError(f"Could not read any frame from: {video_path}")
        return 0, cv2.cvtColor(first[1], cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def frame_at(video_path: str, frame_index: int):
    """Return one exact zero-based video frame as an RGB ndarray."""
    import cv2

    index = max(0, int(frame_index))
    capture = cv2.VideoCapture(video_path)
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(
                f"Could not read frame {index} from: {video_path}")
        return index, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def _find_ffmpeg():
    """Locate ffmpeg binary, checking PATH and common locations."""
    import shutil
    ff = shutil.which('ffmpeg')
    if ff:
        return ff
    # Check common locations relative to package
    pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for candidate in [
        os.path.join(pkg_dir, '..', 'ffmpeg.exe'),
        os.path.join(pkg_dir, '..', 'ffmpeg'),
        os.path.join(pkg_dir, 'ffmpeg.exe'),
        os.path.join(pkg_dir, 'ffmpeg'),
    ]:
        candidate = os.path.normpath(candidate)
        if os.path.isfile(candidate):
            return candidate
    return 'ffmpeg'  # fallback to PATH


def extract_frames(video_path, output_dir, nth_frame=1, start_frame=0, end_frame=999999999):
    """Extract frames from a video file using ffmpeg.

    Args:
        video_path: path to video file
        output_dir: directory to save extracted frames
        nth_frame: extract every nth frame
        start_frame: starting frame index
        end_frame: ending frame index
    """
    os.makedirs(output_dir, exist_ok=True)
    ffmpeg = _find_ffmpeg()
    cmd = [
        ffmpeg, '-i', video_path,
        '-vf', f'select=between(n\\,{start_frame}\\,{end_frame})*not(mod(n\\,{nth_frame}))',
        '-vsync', 'vfr',
        '-q:v', '2',
        # Extracted video frames are one-based on disk while frame_range is
        # zero-based. Preserve absolute numbering: frame 60 -> 000061.jpg.
        '-start_number', str(start_frame + 1),
        os.path.join(output_dir, '%06d.jpg')
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Check if any frames were extracted
    extracted = glob(os.path.join(output_dir, '*.jpg'))
    if not extracted:
        stderr = result.stderr.decode('utf-8', errors='replace') if result.stderr else ''
        raise RuntimeError(
            f"ffmpeg extracted 0 frames from {video_path}\n"
            f"  Command: {' '.join(cmd)}\n"
            f"  stderr: {stderr[:500]}"
        )


class FrameDataset:
    """Dataset of video frames from a directory, video file, or glob pattern.

    (c) Alex Spirin 2023

    Args:
        source_path: path to a directory of images, a video file, or a glob pattern.
            Can also be a string representation of a dict mapping frame numbers to paths.
        outdir_prefix: prefix for extracted frame output directory
        videoframes_root: root directory for extracted video frames
        hash_fn: optional callable to generate a hash from a file path
    """

    def __init__(self, source_path, outdir_prefix='', videoframes_root='', hash_fn=None):
        self.frame_paths = None

        # Handle dict-style source path (e.g., '{"0": "path1.jpg", "10": "path2.jpg"}')
        if "{" in source_path:
            import json
            try:
                source_dict = json.loads(source_path.replace("'", '"'))
            except (json.JSONDecodeError, ValueError):
                source_dict = eval(source_path)

            if isinstance(source_dict, dict):
                self.frame_paths = []
                max_frame = max(int(k) for k in source_dict.keys())
                for i in range(max(max_frame, 1)):
                    # Find the applicable frame for this index
                    keys = sorted(int(k) for k in source_dict.keys())
                    frame = source_dict[str(keys[0])]
                    for k in keys:
                        if i >= k:
                            frame = source_dict[str(k)]
                    assert os.path.exists(frame), \
                        f'Source frame {frame} does not exist.'
                    self.frame_paths.append(frame)
                return

        if not os.path.exists(source_path):
            matches = sorted(glob(source_path))
            if matches:
                self.frame_paths = matches
            else:
                raise FileNotFoundError(
                    f'Frame source not found at {source_path}')

        if os.path.exists(source_path):
            if os.path.isfile(source_path):
                ext = os.path.splitext(source_path)[1][1:].lower()
                if ext in IMAGE_EXTENSIONS:
                    self.frame_paths = [source_path]
                else:
                    # Video file - extract frames
                    if hash_fn:
                        file_hash = hash_fn(source_path)[:10]
                    else:
                        file_hash = str(hash(source_path))[:10]
                    out_path = os.path.join(videoframes_root,
                                            f'{outdir_prefix}_{file_hash}')
                    extract_frames(source_path, out_path)
                    self.frame_paths = sorted(glob(os.path.join(out_path, '*.*')))
                    if not self.frame_paths:
                        raise RuntimeError(
                            f'Could not extract frames from {source_path}')

            elif os.path.isdir(source_path):
                self.frame_paths = sorted(glob(os.path.join(source_path, '*.*')))
                if not self.frame_paths:
                    raise FileNotFoundError(
                        f'Found 0 frames in {source_path}')

        if self.frame_paths is None:
            raise FileNotFoundError(
                f'Frame source not found at {source_path}')

        # Validate extensions
        self._validate_extensions(source_path)
        self.frame_paths = sorted(self.frame_paths)

    def _validate_extensions(self, source_path):
        extensions = set()
        for f in self.frame_paths:
            ext = os.path.splitext(f)[1][1:].lower()
            if ext not in IMAGE_EXTENSIONS:
                raise ValueError(
                    f'Non-image file extension: {ext} in {source_path}')
            extensions.add(ext)
            if len(extensions) > 1:
                raise ValueError(
                    f'Multiple file extensions {extensions} in {source_path}. '
                    f'Use a glob pattern to select one type.')

    def __getitem__(self, idx):
        idx = min(idx, len(self.frame_paths) - 1)
        return self.frame_paths[idx]

    def __len__(self):
        return len(self.frame_paths)
