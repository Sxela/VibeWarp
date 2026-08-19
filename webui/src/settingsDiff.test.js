import { describe, expect, it } from 'vitest';
import {
  annotateUnsaved,
  diffSettings,
  formatSettingValue,
  unsavedOnlyRows,
  wasSaved,
} from './settingsDiff.js';

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

describe('settings a run never saved', () => {
  const left = {diffusion: {steps: 20, sampler_tile_size: 768}};
  const right = {diffusion: {steps: 20, sampler_tile_size: 512},
                 contact_sheet: {mode: 'off'}};

  it('knows whether a path was present', () => {
    expect(wasSaved(left, 'diffusion.sampler_tile_size')).toBe(true);
    expect(wasSaved(left, 'contact_sheet.mode')).toBe(false);
    expect(wasSaved(null, 'diffusion.steps')).toBe(false);
    expect(wasSaved(left, '')).toBe(false);
  });

  it('marks which side actually saved a differing setting', () => {
    const rows = annotateUnsaved(
      diffSettings({diffusion: {sampler_tile_size: 768}},
                   {diffusion: {sampler_tile_size: 512}}),
      left, right);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({leftSaved: true, rightSaved: true});
  });

  it('flags a setting one run predates, even when the loaded values agree', () => {
    // Defaults are filled in on load, so contact_sheet.mode reads 'off' for
    // both — but only one run ever had the option.
    const rows = unsavedOnlyRows(left, right, []);
    const paths = rows.map(r => r.path);
    expect(paths).toContain('contact_sheet.mode');
    const row = rows.find(r => r.path === 'contact_sheet.mode');
    expect(row.leftSaved).toBe(false);
    expect(row.rightSaved).toBe(true);
  });

  it('does not duplicate a path the value diff already reported', () => {
    const rows = diffSettings({a: 1}, {});
    expect(unsavedOnlyRows({a: 1}, {}, rows).map(r => r.path)).not.toContain('a');
  });
});
