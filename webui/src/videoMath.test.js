import { describe, expect, it } from 'vitest';
import { derive, fitDimensions, framesInRange, renderableFrames } from './videoMath.js';

// These mirror vibewarp/video/input.py::fit_dimensions and the /api/video/probe maths.
// They are computed client-side so that changing max_size or extract_nth_frame does not
// re-run ffprobe on the server — but that means the two implementations must not drift.
// test_web_api.py pins the same numbers on the Python side.

describe('fitDimensions', () => {
  it('caps the long edge and snaps to a multiple of 8', () => {
    expect(fitDimensions(1280, 720, 768)).toEqual({width: 768, height: 432});
  });

  it('never upscales', () => {
    expect(fitDimensions(640, 360, 4096)).toEqual({width: 640, height: 360});
  });

  it('works on portrait', () => {
    expect(fitDimensions(1080, 1920, 768)).toEqual({width: 432, height: 768});
  });

  it('keeps at least one multiple', () => {
    expect(fitDimensions(1000, 10, 64).height).toBeGreaterThanOrEqual(8);
  });
});

describe('renderableFrames', () => {
  it('is the whole clip when every frame is kept', () => {
    expect(renderableFrames(1147, 1)).toBe(1147);
  });

  it('rounds UP — the last partial step still yields a frame', () => {
    expect(renderableFrames(1147, 4)).toBe(287);   // ceil(1147/4)
    expect(renderableFrames(10, 3)).toBe(4);       // 0,3,6,9
  });

  it('survives a nonsense step', () => {
    expect(renderableFrames(100, 0)).toBe(100);
    expect(renderableFrames(0, 5)).toBe(0);
  });
});

describe('framesInRange', () => {
  it('is inclusive at both ends', () => {
    expect(framesInRange([0, 11], 559)).toBe(12);
    expect(framesInRange([0, 0], 559)).toBe(559);     // end = 0 -> everything
    expect(framesInRange([5, 5], 559)).toBe(1);
  });

  it('clamps to what exists', () => {
    expect(framesInRange([0, 9999], 559)).toBe(559);
    expect(framesInRange([9999, 0], 559)).toBe(1);    // start pinned to the last frame
  });

  it('is zero when there is nothing to render', () => {
    expect(framesInRange([0, 0], 0)).toBe(0);
    expect(framesInRange([20, 5], 559)).toBe(0);      // end before start
  });
});

describe('derive', () => {
  const source = {width: 1280, height: 720, frames: 1147, fps: 29.99};

  it('reports the size and count the render will actually produce', () => {
    const out = derive(source, 768, 1, [0, 0]);
    expect(out.render).toEqual({width: 768, height: 432});
    expect(out.extracted).toBe(1147);
    expect(out.renderCount).toBe(1147);
    // frame_range is inclusive at both ends, so the last valid index is n-1.
    expect(out.maxFrame).toBe(1146);
  });

  it('renderCount honours the frame range', () => {
    // REGRESSION: "frames to render" used to report the EXTRACTED count, so it ignored the
    // range entirely — a 12-frame render (0-11) advertised itself as 559.
    const out = derive({...source, frames: 559}, 1280, 1, [0, 11]);
    expect(out.renderCount).toBe(12);
    expect(out.extracted).toBe(559);
  });

  it('renderCount honours extraction AND the range together', () => {
    const out = derive(source, 768, 4, [0, 9]);
    expect(out.extracted).toBe(287);       // ceil(1147/4)
    expect(out.renderCount).toBe(10);      // frames 0..9 of those
    expect(out.maxFrame).toBe(286);
  });

  it('max_size = 0 means the source size', () => {
    expect(derive(source, 0, 1, [0, 0]).render).toEqual({width: 1280, height: 720});
  });

  it('extraction lowers the frame ceiling', () => {
    const out = derive(source, 768, 4, [0, 0]);
    expect(out.extracted).toBe(287);
    expect(out.maxFrame).toBe(286);
    expect(out.extractNth).toBe(4);
    expect(out.renderCount).toBe(287);
  });
});
