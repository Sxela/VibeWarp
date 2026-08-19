import { describe, expect, it } from 'vitest';
import {
  filterRunsByMinimumFrames,
  frameAfterRunLoad,
  latestRenderedFrame,
  normalizeMinimumFrames,
  renderedFrames,
} from './historyFilters.js';

describe('renderedFrames', () => {
  const detail = {
    frames: [0, 1, 2, 3, 4, 5],   // union of all layers == the whole range
    layers: [
      { id: 'init', frames: [0, 1, 2, 3, 4, 5] },
      { id: 'output', frames: [0, 1] },
    ],
  };

  it('reports what rendered, not the range the inits cover', () => {
    expect(renderedFrames(detail)).toEqual([0, 1]);
  });

  it('is empty before the first frame finishes', () => {
    expect(renderedFrames({ frames: [0, 1, 2], layers: [{ id: 'init', frames: [0, 1, 2] }] }))
      .toEqual([]);
    expect(renderedFrames(null)).toEqual([]);
    expect(renderedFrames({})).toEqual([]);
  });

  it('agrees with latestRenderedFrame', () => {
    expect(latestRenderedFrame(detail)).toBe(renderedFrames(detail).at(-1));
  });
});

describe('History rendered-frame filtering', () => {
  const runs = [
    { id: 'failed', frames: 0 },
    { id: 'short', frames: 3 },
    { id: 'complete', frames: 24 },
  ];

  it('normalizes invalid and negative values to zero', () => {
    expect(normalizeMinimumFrames('bad')).toBe(0);
    expect(normalizeMinimumFrames(-4)).toBe(0);
    expect(normalizeMinimumFrames('5')).toBe(5);
  });

  it('filters zero-frame and short runs using an inclusive minimum', () => {
    expect(filterRunsByMinimumFrames(runs, 1).map(run => run.id))
      .toEqual(['short', 'complete']);
    expect(filterRunsByMinimumFrames(runs, 4).map(run => run.id))
      .toEqual(['complete']);
    expect(filterRunsByMinimumFrames(runs, 0)).toEqual(runs);
  });

  it('selects the latest rendered output instead of an unrendered input frame', () => {
    let detail = {
      frames: [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
      layers: [
        {id: 'init', frames: [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]},
        {id: 'output', frames: [60, 61, 62]},
      ],
    };
    expect(latestRenderedFrame(detail)).toBe(62);
  });

  it('uses the first input while an in-progress run has no output yet', () => {
    expect(latestRenderedFrame({
      frames: [60, 61, 62],
      layers: [{id: 'init', frames: [60, 61, 62]}],
    })).toBe(60);
  });

  it('preserves the selected frame when switching runs even if it is out of range', () => {
    expect(frameAfterRunLoad({
      selectedFrame: 70,
      previousRun: 'run-a',
      nextRun: 'run-b',
      previousLatest: 70,
      nextLatest: 12,
    })).toBe(70);
  });

  it('continues following the latest frame only within the same active run', () => {
    expect(frameAfterRunLoad({
      selectedFrame: 12,
      previousRun: 'run-a',
      nextRun: 'run-a',
      previousLatest: 12,
      nextLatest: 13,
    })).toBe(13);
  });

  it('selects the latest output for the initial history load', () => {
    expect(frameAfterRunLoad({
      selectedFrame: 0,
      previousRun: undefined,
      nextRun: 'run-a',
      previousLatest: null,
      nextLatest: 62,
      selectLatest: true,
    })).toBe(62);
  });
});
