import { render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import Preview from './Preview.svelte';

// The run list is only refetched when a job reaches a terminal state, so an
// in-progress run's frame count used to sit frozen until a manual refresh.

const RUN = {
  id: '1', label: '', frames: 2, last_frame: 1,
  modified: '2026-01-01T00:00:00Z',
  has_settings: false, resume_available: false, video_available: false,
};

/**
 * @param listed   what /runs reports (the stale snapshot)
 * @param rendered what /runs/{id} reports (the live truth)
 */
function stubApi(listed, rendered, extraRuns = []) {
  const calls = {list: 0, detail: 0};
  global.fetch = vi.fn(async (url) => {
    if (url.startsWith('/api/preview/runs?')) {
      calls.list += 1;
      return {ok: true, json: async () => ({
        runs: [{...RUN, frames: listed.length,
                last_frame: listed.length ? listed.at(-1) : null},
               ...extraRuns],
        root: '/out',
      })};
    }
    if (url.startsWith('/api/preview/runs/1?')) {
      calls.detail += 1;
      return {ok: true, json: async () => ({
        run: '1', frames: rendered,
        layers: [{id: 'output', label: 'Output', group: 'Output',
                  frames: rendered}],
      })};
    }
    return {ok: true, json: async () => ({})};
  });
  return calls;
}

function mount(job) {
  return render(Preview, {
    config: {output_dir: 'images_out', batch_name: 'warpfusion'},
    job, onload: vi.fn(), onjob: vi.fn(),
  });
}

const runningJob = (frame, runId = '1') => ({
  id: 'job-1', state: 'running', frame, operation: 'render', run_id: runId,
});

afterEach(() => { vi.restoreAllMocks(); delete global.fetch; });

describe('active run summary', () => {
  it('shows the live frame count, not the stale listed one', async () => {
    // The list says 2; the run has actually produced 5.
    stubApi([0, 1], [0, 1, 2, 3, 4]);
    mount(runningJob(4));

    await waitFor(() => expect(screen.getByText(/5 frames/)).toBeInTheDocument());
    expect(screen.queryByText(/2 frames/)).toBeNull();
  });

  it('does not re-list every run just to update the count', async () => {
    const calls = stubApi([0, 1], [0, 1, 2, 3, 4]);
    mount(runningJob(4));
    await waitFor(() => expect(screen.getByText(/5 frames/)).toBeInTheDocument());

    // One listing on mount. Advancing frames must not trigger more: relisting
    // rescans every run on disk and warms thumbnails.
    expect(calls.list).toBe(1);
  });

  it('re-lists once when the running job is not in the list at all', async () => {
    // Job reports run 9, which the list has never heard of — a run started
    // after this view loaded. There is no row to patch, so it must relist.
    const calls = stubApi([0, 1], [0, 1, 2]);
    mount(runningJob(1, '9'));

    await waitFor(() => expect(calls.list).toBeGreaterThan(1));
  });

  it('keeps a single frame singular', async () => {
    stubApi([], [0]);
    mount(runningJob(0));
    await waitFor(() => expect(screen.getByText(/1 frame(?!s)/)).toBeInTheDocument());
  });
});
