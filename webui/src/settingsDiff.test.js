import { describe, expect, it } from 'vitest';
import { diffSettings, formatSettingValue } from './settingsDiff.js';

describe('settings diff', () => {
  it('reports changed leaves with stable dotted paths', () => {
    let rows = diffSettings(
      {diffusion: {seed: 1, steps: 20}, prompt: 'cat', refs: [1, 2]},
      {diffusion: {seed: 2, steps: 20}, prompt: 'dog', refs: [1, 3]},
    );
    expect(rows).toEqual([
      {path: 'diffusion.seed', left: 1, right: 2},
      {path: 'prompt', left: 'cat', right: 'dog'},
      {path: 'refs', left: [1, 2], right: [1, 3]},
    ]);
  });

  it('distinguishes absent, empty, and equal values', () => {
    expect(diffSettings({a: 1}, {a: 1})).toEqual([]);
    expect(diffSettings({}, {new_field: false})).toEqual([
      {path: 'new_field', left: undefined, right: false},
    ]);
    expect(formatSettingValue(undefined)).toBe('—');
    expect(formatSettingValue('')).toBe('""');
  });
});
