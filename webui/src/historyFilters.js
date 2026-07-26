export function normalizeMinimumFrames(value) {
  let parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? Math.max(0, parsed) : 0;
}

export function filterRunsByMinimumFrames(runs, minimumFrames) {
  let minimum = normalizeMinimumFrames(minimumFrames);
  return (runs ?? []).filter(run => Number(run.frames ?? 0) >= minimum);
}
