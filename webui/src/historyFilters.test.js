import { describe, expect, it } from 'vitest';
import {
  filterRunsByMinimumFrames,
  normalizeMinimumFrames,
} from './historyFilters.js';

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
});
