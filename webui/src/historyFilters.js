export function normalizeMinimumFrames(value) {
  let parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? Math.max(0, parsed) : 0;
}

export function filterRunsByMinimumFrames(runs, minimumFrames) {
  let minimum = normalizeMinimumFrames(minimumFrames);
  return (runs ?? []).filter(run => Number(run.frames ?? 0) >= minimum);
}

export function latestRenderedFrame(detail) {
  let output = (detail?.layers ?? []).find(layer => layer.id === 'output');
  if (output?.frames?.length) return output.frames.at(-1);
  // Before frame 1 finishes, prefer the beginning of the selected range.
  // Extracted input frames already cover the whole range, so their final item
  // is specifically the misleading empty endpoint we want to avoid.
  return detail?.frames?.length ? detail.frames[0] : null;
}

export function frameAfterRunLoad({
  selectedFrame,
  previousRun,
  nextRun,
  previousLatest,
  nextLatest,
  selectLatest = false,
}) {
  if (selectLatest) return nextLatest;
  // Continue following an in-progress run only while the user was already on
  // that same run's latest output. A different run must never clamp or replace
  // the globally selected source-frame number.
  if (previousRun === nextRun && selectedFrame === previousLatest) {
    return nextLatest;
  }
  return selectedFrame;
}
