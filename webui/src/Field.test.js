import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import Field from './Field.svelte';

const stringField = { type: 'string' };

function mockExists(result) {
  global.fetch = vi.fn(() =>
    Promise.resolve({ json: () => Promise.resolve(result) }));
}

afterEach(() => { vi.restoreAllMocks(); delete global.fetch; });

describe('path fields', () => {
  it('flags a path that is not on disk', async () => {
    mockExists({ checked: true, exists: false });
    const onchange = vi.fn();
    render(Field, { name: 'sd_checkpoint_path', schema: stringField, value: '', onchange });

    await fireEvent.change(screen.getByRole('textbox'), { target: { value: 'C:/nope.ckpt' } });

    await waitFor(() => expect(screen.getByText('Not found on disk')).toBeInTheDocument());
    expect(onchange).toHaveBeenCalledWith('C:/nope.ckpt');
  });

  it('says nothing about a path that exists', async () => {
    mockExists({ checked: true, exists: true });
    render(Field, { name: 'sd_checkpoint_path', schema: stringField, value: '', onchange: vi.fn() });

    await fireEvent.change(screen.getByRole('textbox'), { target: { value: 'C:/real.ckpt' } });

    await waitFor(() => expect(global.fetch).toHaveBeenCalled());
    expect(screen.queryByText('Not found on disk')).not.toBeInTheDocument();
  });

  it('treats empty as optional, not missing', async () => {
    // Plenty of path fields are legitimately blank; validate_config decides what is
    // required, and the UI must not second-guess it.
    mockExists({ checked: false });
    render(Field, { name: 'lora_dir', schema: stringField, value: 'x', onchange: vi.fn() });

    await fireEvent.change(screen.getByRole('textbox'), { target: { value: '' } });

    expect(screen.queryByText('Not found on disk')).not.toBeInTheDocument();
    expect(global.fetch).not.toHaveBeenCalled();   // nothing to ask about
  });

  it('strips quotes pasted in from Explorer before checking', async () => {
    mockExists({ checked: true, exists: true });
    const onchange = vi.fn();
    render(Field, { name: 'sd_checkpoint_path', schema: stringField, value: '', onchange });

    await fireEvent.change(screen.getByRole('textbox'),
                           { target: { value: '"C:/models/sd.ckpt"' } });

    expect(onchange).toHaveBeenCalledWith('C:/models/sd.ckpt');
    await waitFor(() => expect(global.fetch).toHaveBeenCalled());
    expect(global.fetch.mock.calls[0][0]).toContain(encodeURIComponent('C:/models/sd.ckpt'));
  });

  it('does not blame the field when the server is unreachable', async () => {
    global.fetch = vi.fn(() => Promise.reject(new Error('offline')));
    render(Field, { name: 'sd_checkpoint_path', schema: stringField, value: '', onchange: vi.fn() });

    await fireEvent.change(screen.getByRole('textbox'), { target: { value: 'C:/x.ckpt' } });

    await waitFor(() => expect(global.fetch).toHaveBeenCalled());
    expect(screen.queryByText('Not found on disk')).not.toBeInTheDocument();
  });

  it('leaves non-path strings alone', async () => {
    render(Field, { name: 'batch_name', schema: stringField, value: 'run', onchange: vi.fn() });
    await fireEvent.change(screen.getByRole('textbox'), { target: { value: '"quoted"' } });
    expect(global.fetch).toBeUndefined();   // no existence check for a plain string
  });
});

describe('prompt routing', () => {
  it('sends reconstruction-noise prompts to the schedule editor too', () => {
    // Same Dict[int, str] shape as text_prompts; they used to render as a raw JSON blob.
    render(Field, { name: 'neg_prompts', schema: { type: 'object' },
                    value: { 0: 'blurry' }, onchange: vi.fn() });
    expect(screen.getByLabelText('Prompt')).toHaveValue('blurry');
  });
});

describe('validation errors', () => {
  it('shows the message next to the field, not only in the sidebar', () => {
    render(Field, { name: 'sd_checkpoint_path', schema: stringField, value: '',
                    path: 'main', error: 'A Stable Diffusion checkpoint is required',
                    onchange: vi.fn() });

    expect(screen.getByText('A Stable Diffusion checkpoint is required')).toBeInTheDocument();
    expect(screen.getByRole('textbox')).toHaveClass('invalid');
  });

  it('carries an anchor id so the sidebar can scroll to it', () => {
    const { container } = render(Field, { name: 'video_init_path', schema: stringField,
                                          value: '', path: 'video', error: 'required',
                                          onchange: vi.fn() });
    expect(container.querySelector('#field-video-video_init_path')).toBeInTheDocument();
  });

  it('an error replaces the hint rather than stacking under it', () => {
    render(Field, { name: 'steps', schema: { type: 'integer' }, value: 20, path: 'diffusion',
                    hint: 'Recommended: 20', error: 'must be positive', onchange: vi.fn() });
    expect(screen.getByText('must be positive')).toBeInTheDocument();
    expect(screen.queryByText('Recommended: 20')).not.toBeInTheDocument();
  });

  it('says nothing when the field is fine', () => {
    const { container } = render(Field, { name: 'batch_name', schema: stringField,
                                          value: 'run', path: 'main', onchange: vi.fn() });
    expect(container.querySelector('.err')).toBeNull();
    expect(screen.getByRole('textbox')).not.toHaveClass('invalid');
  });
});
