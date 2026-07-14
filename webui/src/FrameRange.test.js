import { render, screen, fireEvent } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import FrameRange from './FrameRange.svelte';

const inputs = () => screen.getAllByRole('spinbutton');   // [start, end]

describe('frame range', () => {
  it('is inclusive at both ends', () => {
    // 0-15 is SIXTEEN frames. Getting this wrong made AnimateDiff fail with
    // "needs at least context_length (16) frames, got 15".
    render(FrameRange, { value: [0, 15], onchange: vi.fn(), maxFrame: 100 });
    expect(screen.getByText('16 frames')).toBeInTheDocument();
  });

  it('clamps to what the video actually has', async () => {
    // The ceiling comes from probing the clip. Without it you can ask for frames that do
    // not exist and only find out mid-render.
    const onchange = vi.fn();
    render(FrameRange, { value: [0, 0], onchange, maxFrame: 40 });

    await fireEvent.change(inputs()[1], { target: { value: '999' } });

    expect(onchange).toHaveBeenCalledWith([0, 40]);
  });

  it('leaves the range alone when the video has not been probed', async () => {
    const onchange = vi.fn();
    render(FrameRange, { value: [0, 0], onchange, maxFrame: 0 });

    await fireEvent.change(inputs()[1], { target: { value: '999' } });

    expect(onchange).toHaveBeenCalledWith([0, 999]);
  });

  it('end = 0 means every frame', () => {
    render(FrameRange, { value: [0, 0], onchange: vi.fn(), maxFrame: 40 });
    expect(screen.getByText('41 frames')).toBeInTheDocument();   // 0..40 inclusive
  });

  it('warns when the end precedes the start', () => {
    render(FrameRange, { value: [20, 5], onchange: vi.fn(), maxFrame: 100 });
    expect(screen.getByText(/End must be greater than or equal to start/)).toBeInTheDocument();
  });

  it('warns when the range overruns the video', () => {
    // Reachable from a loaded settings file, which is not clamped on the way in.
    render(FrameRange, { value: [0, 500], onchange: vi.fn(), maxFrame: 40 });
    expect(screen.getByText('This video only has 41 frames (0–40).')).toBeInTheDocument();
  });
});
