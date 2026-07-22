import { describe, expect, it } from 'vitest';
import { comfyConnection, fieldSupportsModel, modelFamily } from './modelCompatibility.js';

const schema = {
  model_family_by_version: {
    control_multi_v15: 'sd',
    flux2_klein_edit: 'flux',
    hidream_o1_edit: 'hidream',
  },
  default_model_family: 'sd',
};

describe('model compatibility', () => {
  it('resolves declared edit families and defaults imported legacy models to SD', () => {
    expect(modelFamily('flux2_klein_edit', schema)).toBe('flux');
    expect(modelFamily('hidream_o1_edit', schema)).toBe('hidream');
    expect(modelFamily('control_multi_animatediff', schema)).toBe('sd');
  });

  it('shows common fields and only the selected model-specific fields', () => {
    expect(fieldSupportsModel({}, 'hidream')).toBe(true);
    expect(fieldSupportsModel({ model_families: ['sd'] }, 'hidream')).toBe(false);
    expect(fieldSupportsModel({ model_families: ['hidream'] }, 'hidream')).toBe(true);
    expect(fieldSupportsModel({ model_families: ['flux'] }, 'hidream')).toBe(false);
  });

  it('identifies only model selections that use a ComfyUI server', () => {
    let flux = { model_version: 'flux2_klein_edit',
      flux: { backend: 'comfy', comfy_server_url: 'http://127.0.0.1:8188' } };
    expect(comfyConnection(flux, schema)).toEqual({
      label: 'Flux.2', url: 'http://127.0.0.1:8188', path: 'flux.comfy_server_url' });
    flux.flux.backend = 'diffusers';
    expect(comfyConnection(flux, schema)).toBeNull();

    let hidream = { model_version: 'hidream_o1_edit',
      hidream: { comfy_server_url: 'http://localhost:9191' } };
    expect(comfyConnection(hidream, schema)).toEqual({
      label: 'HiDream-O1', url: 'http://localhost:9191',
      path: 'hidream.comfy_server_url' });
    expect(comfyConnection({ model_version: 'control_multi_v15' }, schema)).toBeNull();
  });
});
