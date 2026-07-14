import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import VideoEditor from './VideoEditor.svelte';

const video = { video_init_path: '', max_size: 768, extract_nth_frame: 1,
                width: 512, height: 512, save_img_format: 'png' };

const PROBE = {
  source: { width: 1280, height: 720, fps: 29.99, frames: 1147,
            duration: 38.26, codec: 'h264', source: 'ffprobe' },
};

/** Mock /api/video/probe. `ok:false` simulates an unreadable/missing file. */
function mockProbe(body = PROBE, ok = true, status = 200) {
  global.fetch = vi.fn(() => Promise.resolve({
    ok, status, json: () => Promise.resolve(body) }));
}

const pathInput = () => screen.getByPlaceholderText('C:\\videos\\input.mp4');

afterEach(() => { vi.restoreAllMocks(); delete global.fetch; });

describe('validation errors', () => {
  it('shows the input-video error on the field itself', () => {
    // VideoEditor hand-rolls its inputs, so Field never sees them — and "An input video
    // is required" is the most common error there is. It had nowhere to render.
    mockProbe();
    const { container } = render(VideoEditor, {
      value: video, onchange: vi.fn(),
      errors: { video_init_path: 'An input video is required' },
    });

    expect(screen.getByText('An input video is required')).toBeInTheDocument();
    expect(container.querySelector('#field-video-video_init_path input'))
      .toHaveClass('invalid');
  });

  it('anchors every field so the sidebar can scroll to any of them', () => {
    mockProbe();
    const { container } = render(VideoEditor, { value: video, onchange: vi.fn() });
    for (const field of ['video_init_path', 'max_size', 'extract_nth_frame'])
      expect(container.querySelector(`#field-video-${field}`)).toBeInTheDocument();
  });

  it('stays quiet when there is nothing wrong', () => {
    mockProbe();
    const { container } = render(VideoEditor, { value: video, onchange: vi.fn() });
    expect(container.querySelector('.err')).toBeNull();
  });
});

describe('probing the video', () => {
  it('reports the source, the REAL render size and the frame count', async () => {
    mockProbe();
    const { container } = render(VideoEditor, {
      value: { ...video, video_init_path: 'C:/clip.mp4' }, onchange: vi.fn() });

    await waitFor(() => expect(container.textContent).toContain('1280×720'));
    const text = container.textContent;
    // 1280x720 capped at max_size 768 -> 768x432. This used to be a placeholder labelled
    // "recalculated when the render starts".
    expect(text).toContain('768×432');
    expect(text).toContain('1147 frames');
    expect(text).toContain('h264');
  });

  it('re-reads the video ONLY when the path changes', async () => {
    // Changing max_size or extract_nth_frame is pure arithmetic (videoMath.js). Re-running
    // ffprobe for a slider nudge is wasted work.
    mockProbe();
    const { rerender } = render(VideoEditor, {
      value: { ...video, video_init_path: 'C:/clip.mp4' }, onchange: vi.fn() });

    await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(1));

    await rerender({ value: { ...video, video_init_path: 'C:/clip.mp4', max_size: 512 },
                     onchange: vi.fn() });
    await rerender({ value: { ...video, video_init_path: 'C:/clip.mp4', max_size: 512,
                              extract_nth_frame: 4 }, onchange: vi.fn() });
    expect(global.fetch).toHaveBeenCalledTimes(1);       // still just the one probe

    await rerender({ value: { ...video, video_init_path: 'C:/other.mp4' },
                     onchange: vi.fn() });
    await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(2));
  });

  it('recomputes the render size locally as max_size changes', async () => {
    mockProbe();
    const { rerender, container } = render(VideoEditor, {
      value: { ...video, video_init_path: 'C:/clip.mp4' }, onchange: vi.fn() });
    await waitFor(() => expect(container.textContent).toContain('768×432'));

    await rerender({ value: { ...video, video_init_path: 'C:/clip.mp4', max_size: 512 },
                     onchange: vi.fn() });

    await waitFor(() => expect(container.textContent).toContain('512×288'));
  });

  it('says so when the video cannot be read', async () => {
    mockProbe({ detail: [{ message: 'No video at that path' }] }, false, 404);
    render(VideoEditor, { value: { ...video, video_init_path: 'C:/nope.mp4' },
                          onchange: vi.fn() });

    await waitFor(() => expect(screen.getByText('No video at that path')).toBeInTheDocument());
  });

  it('prefers the server validation error over the probe message', async () => {
    // Both can be true at once; the validation error is the authoritative one.
    mockProbe({ detail: [{ message: 'No video at that path' }] }, false, 404);
    render(VideoEditor, { value: { ...video, video_init_path: 'C:/nope.mp4' },
                          onchange: vi.fn(),
                          errors: { video_init_path: 'An input video is required' } });

    await waitFor(() => expect(global.fetch).toHaveBeenCalled());
    expect(screen.getByText('An input video is required')).toBeInTheDocument();
    expect(screen.queryByText('No video at that path')).not.toBeInTheDocument();
  });

  it('strips quotes pasted in from Explorer', async () => {
    mockProbe();
    const onchange = vi.fn();
    render(VideoEditor, { value: video, onchange });

    await fireEvent.change(pathInput(), { target: { value: '"C:/clip.mp4"' } });

    expect(onchange).toHaveBeenCalledWith(
      expect.objectContaining({ video_init_path: 'C:/clip.mp4' }));
  });
});
