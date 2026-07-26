import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import ReferenceImagesEditor from './ReferenceImagesEditor.svelte';

const defaults = [
  { source: 'raw', label: false, image_path: '' },
  { source: 'none', label: false, image_path: '' },
];

afterEach(() => { vi.restoreAllMocks(); delete global.fetch; });

describe('ordered reference editor', () => {
  it('always shows the required first image and one inactive next slot', () => {
    render(ReferenceImagesEditor, {
      value: defaults, onchange: vi.fn(), maxReferences: 4,
      videoPath: 'C:/clip.mp4', frameRange: [3, 8], extractNth: 2,
    });
    expect(screen.getByText('Image 1')).toBeInTheDocument();
    expect(screen.getByText('Required')).toBeInTheDocument();
    expect(screen.getByText('Image 2')).toBeInTheDocument();
    expect(screen.getByText('Not sent')).toBeInTheDocument();
    expect(screen.getByText('Bake @Image1 label')).toBeInTheDocument();
    expect(screen.getByLabelText('Reference label opacity')).toHaveValue('0.7');
    expect(screen.getByText('70%')).toBeInTheDocument();
    expect(screen.queryByText('Image 3')).not.toBeInTheDocument();
    expect(screen.getByAltText(/preview for raw frame/i).src).toContain('frame=6');

    const [first, second] = screen.getAllByRole('combobox');
    expect([...first.options].map(option => option.value)).toEqual([
      'raw', 'previous', 'warped']);
    expect([...second.options].map(option => option.value)).toEqual([
      'raw', 'previous', 'warped', 'none', 'upload']);
  });

  it('changes the shared label opacity as a numeric fraction', async () => {
    const onLabelOpacityChange = vi.fn();
    render(ReferenceImagesEditor, {
      value: defaults, onchange: vi.fn(), onLabelOpacityChange,
    });
    await fireEvent.input(screen.getByLabelText('Reference label opacity'), {
      target: { value: '0.45' },
    });
    expect(onLabelOpacityChange).toHaveBeenLastCalledWith(0.45);
  });

  it('reveals the next slot when the current one becomes active', async () => {
    const onchange = vi.fn();
    const { rerender } = render(ReferenceImagesEditor, {
      value: defaults, onchange, maxReferences: 4,
    });
    await fireEvent.change(screen.getAllByRole('combobox')[1], {
      target: { value: 'previous' },
    });
    const next = onchange.mock.calls.at(-1)[0];
    expect(next.map(ref => ref.source)).toEqual(['raw', 'previous', 'none']);
    await rerender({ value: next, onchange, maxReferences: 4 });
    expect(screen.getByText('Image 3')).toBeInTheDocument();
  });

  it('explains first-frame behavior for required and optional temporal inputs', () => {
    render(ReferenceImagesEditor, {
      value: [
        { source: 'previous', label: false, image_path: '' },
        { source: 'warped', label: false, image_path: '' },
        defaults[1],
      ],
      onchange: vi.fn(), maxReferences: 4, videoPath: 'C:/clip.mp4',
    });

    expect(screen.getByText('Raw fallback on first frame')).toBeInTheDocument();
    expect(screen.getByText('Not sent on first frame')).toBeInTheDocument();
    expect(screen.queryByText('First-frame placeholder')).not.toBeInTheDocument();
  });

  it('uploads an image from the square picker and stores its server path', async () => {
    const onchange = vi.fn();
    global.fetch = vi.fn(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ path: 'C:/cache/style.png' }),
    }));
    const uploadMode = [
      defaults[0],
      { source: 'upload', label: false, image_path: '' },
      defaults[1],
    ];
    const { container } = render(ReferenceImagesEditor, {
      value: uploadMode, onchange, maxReferences: 4,
    });
    const file = new File(['image'], 'style.png', { type: 'image/png' });
    await fireEvent.change(container.querySelector('input[type="file"]'), {
      target: { files: [file] },
    });

    await waitFor(() => expect(onchange).toHaveBeenCalled());
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/references/upload?filename=style.png',
      expect.objectContaining({ method: 'POST', body: file }));
    expect(onchange.mock.calls.at(-1)[0][1]).toEqual({
      source: 'upload', label: false, image_path: 'C:/cache/style.png',
    });
  });
});
