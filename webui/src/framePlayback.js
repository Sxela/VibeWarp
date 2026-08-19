// Looping playback over a run's rendered frames, so a render can be reviewed
// as motion without assembling a video first.
//
// A run's frame list is NOT necessarily contiguous — resumed and partial
// renders leave gaps — so playback walks the list of frames that actually
// exist rather than incrementing a counter, which would stall on missing
// frames.

export const PLAYBACK_FPS = 12;
export const PLAYBACK_INTERVAL_MS = Math.round(1000 / PLAYBACK_FPS);

/**
 * The frame to show after `current`, wrapping back to the first one.
 *
 * @param {number} current  frame currently displayed
 * @param {number[]} frames ascending list of frames that exist for this run
 * @returns {number} the next frame, or `current` when there is nothing to play
 */
export function nextFrame(current, frames) {
  if (!frames || frames.length === 0) return current;
  const i = frames.indexOf(current);
  if (i !== -1) return frames[(i + 1) % frames.length];
  // The frame number box lets you type a frame this run never rendered.
  // Resume at the next one that exists, or wrap around if we are past the end.
  const ahead = frames.find((f) => f > current);
  return ahead === undefined ? frames[0] : ahead;
}

/** Playback needs at least two frames to show anything as motion. */
export function canPlay(frames) {
  return !!frames && frames.length > 1;
}
