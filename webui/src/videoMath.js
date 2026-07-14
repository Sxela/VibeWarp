// The render size and frame count derived from a probed video.
//
// These mirror the backend exactly:
//   fitDimensions  <-> vibewarp/video/input.py :: fit_dimensions
//   renderableFrames <-> the ceil(frames / nth) in /api/video/probe
//
// They live here so changing `max_size` or `extract_nth_frame` does not re-run ffprobe on
// the server — the video itself only needs reading when the PATH changes. Both sides are
// pinned by tests (videoMath.test.js here, test_web_api.py there), so a change to one
// without the other fails.

/** Fit to a maximum edge, preserving aspect, snapped to a multiple (8 for latents). */
export function fitDimensions(width, height, maxSize, multiple = 8) {
  if (width <= 0 || height <= 0 || maxSize <= 0) return {width, height};
  const scale = Math.min(1, maxSize / Math.max(width, height));
  const snap = (n) => Math.max(multiple, Math.round((n * scale) / multiple) * multiple);
  return {width: snap(width), height: snap(height)};
}

/** Frames left after extraction keeps every nth one. */
export function renderableFrames(frames, nth) {
  const step = Math.max(1, Math.round(nth || 1));
  return frames > 0 ? Math.ceil(frames / step) : 0;
}

/** Derived view of a probed source: what this render will actually produce. */
/** How many frames a [start, end] range covers, inclusive. end = 0 means "to the end". */
export function framesInRange(range, available) {
  if (available <= 0) return 0;
  const last = available - 1;
  const start = Math.min(Math.max(Math.round(range?.[0] ?? 0), 0), last);
  const rawEnd = Math.round(range?.[1] ?? 0);
  const end = rawEnd > 0 ? Math.min(rawEnd, last) : last;
  return end >= start ? end - start + 1 : 0;
}

/**
 * Derived view of a probed source: what this render will actually produce.
 *
 * `renderCount` is the number that matters, and it is NOT the source frame count: it is
 * what survives BOTH extract_nth_frame and frame_range. Reporting the extracted count as
 * "frames to render" was misleading — with nth = 1 it merely echoed the source length, and
 * it ignored the frame range entirely, so a 12-frame render advertised itself as 559.
 */
export function derive(source, maxSize, nth, range) {
  const render = maxSize > 0
    ? fitDimensions(source.width, source.height, maxSize)
    : {width: source.width, height: source.height};
  const extracted = renderableFrames(source.frames, nth);
  return {
    render,
    extracted,
    // frame_range is INCLUSIVE at both ends, so the last valid index is n-1.
    maxFrame: Math.max(0, extracted - 1),
    extractNth: Math.max(1, Math.round(nth || 1)),
    renderCount: framesInRange(range, extracted),
  };
}
