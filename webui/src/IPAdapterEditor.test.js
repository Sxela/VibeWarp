import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import IPAdapterEditor from './IPAdapterEditor.svelte';

const value = {
  enabled: true,
  flip_uc: false,
  models: {
    ipadapter_sd15_plus: {
      path: 'C:/models/ip.bin', weight: 0.7, start: 0, end: 1,
      source_image: { source: 'none', image_path: '' },
      weight_type: 'linear', combine_embeds: 'concat', embeds_scaling: 'V only',
    },
  },
};

const catalog = {
  adapters: [
    {
      key: 'ipadapter_sd15', label: 'Base', model_version: 'sd15',
      clip_variant: 'ViT-H', filename: 'ip-adapter_sd15.safetensors',
      resolved_path: 'C:/models/ip-adapter_sd15.safetensors',
    },
    {
      key: 'ipadapter_sd15_plus', label: 'Plus', model_version: 'sd15',
      clip_variant: 'ViT-H', filename: 'ip-adapter-plus_sd15.safetensors',
      resolved_path: null,
    },
  ],
  files: ['C:/models/ip-adapter_sd15.safetensors'],
};

beforeEach(() => {
  global.fetch = vi.fn(() => Promise.resolve({
    ok: true, json: () => Promise.resolve(catalog),
  }));
});
afterEach(() => { vi.restoreAllMocks(); delete global.fetch; });

describe('IP-Adapter image selectors', () => {
  it('adds a compatible model from the catalog instead of requiring an internal key', async () => {
    const onchange = vi.fn();
    render(IPAdapterEditor, {
      value: { enabled: false, flip_uc: false, models: {} },
      onchange, modelVersion: 'control_multi_v15', modelDir: 'C:/models',
    });
    const picker = screen.getByText('Add IP-Adapter').closest('label').querySelector('select');
    await waitFor(() => expect(picker.options.length).toBe(3));
    expect([...picker.options].map(option => option.value)).toEqual([
      '', 'ipadapter_sd15', 'ipadapter_sd15_plus',
    ]);
    await fireEvent.change(picker, { target: { value: 'ipadapter_sd15' } });
    await fireEvent.click(screen.getByRole('button', { name: 'Add' }));
    expect(onchange.mock.calls.at(-1)[0].models.ipadapter_sd15.path)
      .toBe('C:/models/ip-adapter_sd15.safetensors');
    expect(onchange.mock.calls.at(-1)[0].models.ipadapter_sd15.model_key)
      .toBe('ipadapter_sd15');
    expect(onchange.mock.calls.at(-1)[0].enabled).toBe(true);
  });

  it('allows the same checkpoint to be added as another independent instance', async () => {
    const onchange = vi.fn();
    render(IPAdapterEditor, {
      value: {
        enabled: true, flip_uc: false,
        models: {
          ipadapter_sd15: {
            ...value.models.ipadapter_sd15_plus,
            model_key: 'ipadapter_sd15',
            path: 'C:/models/ip-adapter_sd15.safetensors',
          },
        },
      },
      onchange, modelVersion: 'control_multi_v15', modelDir: 'C:/models',
    });
    const picker = screen.getByText('Add IP-Adapter').closest('label').querySelector('select');
    await waitFor(() => expect(picker).not.toBeDisabled());
    await fireEvent.change(picker, { target: { value: 'ipadapter_sd15' } });
    await fireEvent.click(screen.getByRole('button', { name: 'Add' }));
    const models = onchange.mock.calls.at(-1)[0].models;
    expect(models.ipadapter_sd15__2.model_key).toBe('ipadapter_sd15');
    expect(models.ipadapter_sd15__2.source_image).toEqual({
      source: 'none', image_path: '',
    });
  });

  it('offers no raw-frame input and defaults each adapter to off', () => {
    render(IPAdapterEditor, {
      value, onchange: vi.fn(), frameRange: [12, 20], videoPath: 'C:/clip.mp4',
    });
    const source = screen.getByText('Image source').closest('label').querySelector('select');
    expect([...source.options].map(option => option.value)).toEqual([
      'none', 'previous', 'warped', 'upload',
    ]);
    expect(source).toHaveValue('none');
    expect(screen.getByText('No image is sent')).toBeInTheDocument();
    expect(screen.getByText(/not sent on frame 12/i)).toBeInTheDocument();
  });

  it('stores previous-frame modes in the unified source shape', async () => {
    const onchange = vi.fn();
    render(IPAdapterEditor, { value, onchange });
    const source = screen.getByText('Image source').closest('label').querySelector('select');
    await fireEvent.change(source, { target: { value: 'warped' } });
    expect(onchange.mock.calls.at(-1)[0].models.ipadapter_sd15_plus.source_image)
      .toEqual({ source: 'warped', image_path: '' });
  });

  it('uploads a fixed style reference through the shared endpoint', async () => {
    const onchange = vi.fn();
    global.fetch = vi.fn((url) => Promise.resolve({
      ok: true,
      json: () => Promise.resolve(
        String(url).startsWith('/api/ipadapter/catalog')
          ? catalog : { path: 'C:/cache/style.png' }),
    }));
    const { container } = render(IPAdapterEditor, { value, onchange });
    const file = new File(['image'], 'style.png', { type: 'image/png' });
    await fireEvent.change(container.querySelector('input[type="file"]'), {
      target: { files: [file] },
    });
    await waitFor(() => expect(onchange).toHaveBeenCalled());
    expect(onchange.mock.calls.at(-1)[0].models.ipadapter_sd15_plus.source_image)
      .toEqual({ source: 'upload', image_path: 'C:/cache/style.png' });
  });

  it('supports several images with one shared weight and combine method', async () => {
    const onchange = vi.fn();
    const multi = {
      ...value,
      models: {
        ipadapter_sd15_plus: {
          ...value.models.ipadapter_sd15_plus,
          source_images: [
            { source: 'previous', image_path: '' },
            { source: 'upload', image_path: 'C:/refs/style.png' },
          ],
          combine_embeds: 'average',
        },
      },
    };
    render(IPAdapterEditor, { value: multi, onchange });

    const selectors = screen.getAllByText('Image source')
      .map(label => label.closest('label').querySelector('select'));
    expect(selectors).toHaveLength(2);
    const combine = screen.getByText('Combine images')
      .closest('label').querySelector('select');
    expect([...combine.options].map(option => option.value)).toEqual([
      'concat', 'add', 'subtract', 'average', 'norm average',
    ]);
    expect(combine).toHaveValue('average');

    await fireEvent.change(selectors[1], { target: { value: 'warped' } });
    const entry = onchange.mock.calls.at(-1)[0].models.ipadapter_sd15_plus;
    expect(entry.source_images).toEqual([
      { source: 'previous', image_path: '' },
      { source: 'warped', image_path: '' },
    ]);
    expect(entry.weight).toBe(0.7);
  });
});
