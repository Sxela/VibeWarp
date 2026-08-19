import { describe, it, expect } from 'vitest';
import {
  PLAYBACK_FPS,
  PLAYBACK_INTERVAL_MS,
  canPlay,
  nextFrame,
} from './framePlayback.js';

describe('nextFrame', () => {
  it('advances to the next rendered frame', () => {
    expect(nextFrame(0, [0, 1, 2, 3])).toBe(1);
    expect(nextFrame(2, [0, 1, 2, 3])).toBe(3);
  });

  it('wraps around at the end — playback loops', () => {
    expect(nextFrame(3, [0, 1, 2, 3])).toBe(0);
  });

  it('skips gaps rather than stalling on frames that were never rendered', () => {
    // A resumed or partial run leaves holes. Incrementing a counter would sit
    // on 3 and 4 showing nothing; walking the list steps straight to 7.
    expect(nextFrame(2, [0, 1, 2, 7, 8])).toBe(7);
    expect(nextFrame(8, [0, 1, 2, 7, 8])).toBe(0);
  });

  it('recovers when the frame box holds a number the run never rendered', () => {
    expect(nextFrame(4, [0, 1, 2, 7, 8])).toBe(7);
  });

  it('wraps when the typed frame is past the last rendered one', () => {
    expect(nextFrame(99, [0, 1, 2])).toBe(0);
  });

  it('does not move when there is nothing to play', () => {
    expect(nextFrame(5, [])).toBe(5);
    expect(nextFrame(5, undefined)).toBe(5);
  });

  it('stays put on a single-frame run instead of flickering', () => {
    expect(nextFrame(4, [4])).toBe(4);
  });

  it('walks a whole loop and returns to the start', () => {
    const frames = [0, 1, 2, 7, 8];
    const seen = [];
    let f = frames[0];
    for (let i = 0; i < frames.length; i++) {
      seen.push(f);
      f = nextFrame(f, frames);
    }
    expect(seen).toEqual(frames);
    expect(f).toBe(frames[0]);
  });
});

describe('canPlay', () => {
  it('needs at least two frames', () => {
    expect(canPlay([0, 1])).toBe(true);
    expect(canPlay([0])).toBe(false);
    expect(canPlay([])).toBe(false);
    expect(canPlay(undefined)).toBe(false);
  });
});

describe('playback rate', () => {
  it('derives a sane interval from the frame rate', () => {
    expect(PLAYBACK_INTERVAL_MS).toBe(Math.round(1000 / PLAYBACK_FPS));
    expect(PLAYBACK_INTERVAL_MS).toBeGreaterThan(0);
  });
});
