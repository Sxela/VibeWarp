import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import Preview from './Preview.svelte';

// Guards the wiring, not just the maths: framePlayback.test.js proves the
// helper loops correctly, but a preset that never reaches the DOM is exactly
// the class of bug this suite exists for.

const RUN = {
  id: '1', label: '', frames: 4, last_frame: 3,
  modified: '2026-01-01T00:00:00Z',
  has_settings: false, resume_available: false, video_available: false,
};

/**
 * @param rendered frames the run actually produced (the output layer)
 * @param range    frames the run intends to cover. The backend reports
 *                 detail.frames as the UNION of all layers, and extracted
 *                 init frames already span the whole range — so this is
 *                 normally much longer than `rendered` mid-render.
 */
function stubApi(rendered, range = rendered) {
  global.fetch = vi.fn(async (url) => {
    if (url.startsWith('/api/preview/runs?')) {
      return { ok: true, json: async () => ({
        runs: [{ ...RUN, frames: rendered.length, last_frame: rendered.at(-1) }],
        root: '/out',
      }) };
    }
    if (url.startsWith('/api/preview/runs/1?')) {
      return { ok: true, json: async () => ({
        run: '1', frames: range,
        layers: [
          { id: 'init', label: 'Init', group: 'Input', frames: range },
          { id: 'output', label: 'Output', group: 'Output', frames: rendered },
        ],
      }) };
    }
    return { ok: true, json: async () => ({}) };
  });
}

function mount() {
  return render(Preview, {
    config: { output_dir: 'images_out', batch_name: 'warpfusion' },
    job: null, onload: vi.fn(), onjob: vi.fn(),
  });
}

const frameBox = () => screen.getAllByLabelText('Frame')[0];
const playButton = () => screen.getByRole('button', { name: /play frames in a loop|pause playback/i });

afterEach(() => { vi.restoreAllMocks(); delete global.fetch; });

describe('looping frame playback', () => {
  it('plays, advances through frames, and pauses from the same button', async () => {
    stubApi([0, 1, 2, 3]);
    mount();

    // The frame box renders before the run detail arrives, so wait for the
    // button to become live rather than for the box to exist.
    await waitFor(() => expect(playButton()).toBeEnabled());
    expect(playButton()).toHaveAttribute('aria-pressed', 'false');

    await fireEvent.click(playButton());
    // Same button — it became Pause rather than growing a second control.
    expect(playButton()).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: /pause playback/i })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /play frames in a loop/i })).toBeNull();

    const started = frameBox().value;
    await waitFor(() => expect(frameBox().value).not.toBe(started), { timeout: 2000 });

    await fireEvent.click(playButton());
    expect(playButton()).toHaveAttribute('aria-pressed', 'false');

    // Paused really means paused: the interval was torn down.
    const stopped = frameBox().value;
    await new Promise((r) => setTimeout(r, 300));
    expect(frameBox().value).toBe(stopped);
  });

  it('loops back to the first frame instead of stopping at the end', async () => {
    stubApi([0, 1, 2, 3]);
    mount();
    await waitFor(() => expect(frameBox()).toBeInTheDocument());

    // The initial load selects the run's latest frame, so playback starts at
    // the end and the very next tick has to wrap.
    await waitFor(() => expect(frameBox().value).toBe('3'));
    await fireEvent.click(playButton());
    await waitFor(() => expect(frameBox().value).toBe('0'), { timeout: 2000 });
  });

  it('stepping a frame by hand pauses playback', async () => {
    stubApi([0, 1, 2, 3]);
    mount();
    await waitFor(() => expect(frameBox()).toBeInTheDocument());

    await fireEvent.click(playButton());
    expect(playButton()).toHaveAttribute('aria-pressed', 'true');

    await fireEvent.click(screen.getByLabelText('Previous frame'));
    expect(playButton()).toHaveAttribute('aria-pressed', 'false');
  });

  it('loops only the rendered frames, not the whole selected range', async () => {
    // Mid-render: 3 frames out of a 40-frame range. The extracted inits cover
    // all 40, so detail.frames does too -- playing that would crawl through 37
    // frames the run has not produced.
    stubApi([0, 1, 2], Array.from({ length: 40 }, (_, i) => i));
    mount();
    await waitFor(() => expect(playButton()).toBeEnabled());
    await waitFor(() => expect(frameBox().value).toBe('2'));

    await fireEvent.click(playButton());

    const seen = new Set();
    for (let i = 0; i < 12; i++) {
      seen.add(frameBox().value);
      await new Promise((r) => setTimeout(r, 40));
    }
    // Several full loops' worth of ticks never leave the rendered frames.
    expect([...seen].sort()).toEqual(['0', '1', '2']);
  });

  it('cannot be played when only one frame has rendered so far', async () => {
    // One rendered frame inside a long range: the range is not playable
    // material just because the inits exist.
    stubApi([0], Array.from({ length: 40 }, (_, i) => i));
    mount();
    // Wait on a layer chip: it only renders once detail has been applied, so
    // the button below is disabled on merit rather than because we looked
    // before the fetch settled.
    await screen.findByRole('button', { name: 'Init' });
    expect(playButton()).toBeDisabled();
  });

  it('cannot be played with a single rendered frame', async () => {
    // Frame 5, not 0: the box starts at 0, so seeing 5 proves the run detail
    // actually loaded and the button is disabled on merit, not on timing.
    stubApi([5]);
    mount();
    await waitFor(() => expect(frameBox().value).toBe('5'));
    expect(playButton()).toBeDisabled();
  });
});
