import { render, screen, fireEvent } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import ScheduleEditor from './ScheduleEditor.svelte';

// Renders the editor and re-renders with whatever it emits, so `value` behaves like a real
// parent-owned prop. Without this the mode-inference bugs below are invisible: they only
// show up when the emitted value comes BACK in.
function editor(props) {
  const onchange = vi.fn();
  const view = render(ScheduleEditor, { ...props, onchange });
  const commit = async () => {
    // Switching to JSON deliberately emits nothing (it only changes the view), so there may
    // be no call to replay. Re-rendering with `undefined` would look like the parent wiping
    // the value, which is a different thing entirely.
    if (!onchange.mock.calls.length) return props.value;
    const last = onchange.mock.calls.at(-1)[0];
    await view.rerender({ ...props, value: last, onchange });
    return last;
  };
  return { onchange, commit, ...view };
}

const click = (label) => fireEvent.click(screen.getByRole('button', { name: label }));

describe('mode buttons', () => {
  it('has no Off — a schedule is always a schedule', () => {
    // The notebook has no separate scalar: `steps = get_scheduled_arg(f, steps_schedule)`,
    // and a one-element list IS the constant. Off had no meaning to express.
    editor({ name: 'steps_schedule', value: [20] });
    expect(screen.queryByRole('button', { name: 'Off' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Constant' })).toBeInTheDocument();
  });

  it('Per-frame is not a dead button', async () => {
    // REGRESSION: mode was derived purely from the value's shape, so switching a
    // single-value schedule to Per-frame emitted [v] — length 1 — which the derivation
    // read straight back as 'constant'. The button appeared to do nothing.
    const { onchange, commit } = editor({ name: 'steps_schedule', value: [20] });

    await click('Per-frame');
    expect(onchange).toHaveBeenCalledWith([20]);
    await commit();

    expect(screen.getByRole('button', { name: 'Per-frame' })).toHaveClass('on');
    expect(screen.getByRole('button', { name: 'Constant' })).not.toHaveClass('on');
    // and it must now behave like a list: you can add a frame
    expect(screen.getByRole('button', { name: /\+ frame/ })).toBeInTheDocument();
  });

  it('JSON is not a dead button', async () => {
    // REGRESSION: setMode('json') called onchange(value ?? values) — it re-emitted the
    // value it already had, so nothing changed and the mode snapped back.
    const { commit } = editor({ name: 'cfg_scale_schedule', value: [7] });

    await click('JSON');
    await commit();

    expect(screen.getByRole('button', { name: 'JSON' })).toHaveClass('on');
    expect(screen.getByPlaceholderText('[[3, 7, 3]]')).toBeInTheDocument();
  });

  it('a value replaced from outside snaps back to its natural shape', async () => {
    // The picked mode must survive OUR writes but not the parent's (a settings import).
    const onchange = vi.fn();
    const props = { name: 'steps_schedule', value: [20], onchange };
    const view = render(ScheduleEditor, props);

    await click('JSON');
    await view.rerender({ ...props, value: { 0: 10, 30: 20 } });   // settings loaded

    expect(screen.getByRole('button', { name: 'Keyframes' })).toHaveClass('on');
  });
});

describe('keyframes', () => {
  it('converts a per-frame list to keyframes losslessly', async () => {
    const { onchange } = editor({ name: 'steps_schedule', value: [10, 11, 12] });
    await click('Keyframes');
    expect(onchange).toHaveBeenCalledWith({ 0: 10, 1: 11, 2: 12 });
  });

  it('refuses to collide two keyframes on the same frame', async () => {
    // A collision would silently drop a chip.
    const { onchange } = editor({ name: 'steps_schedule', value: { 0: 10, 20: 30 } });
    const frames = screen.getAllByLabelText('Frame');
    await fireEvent.change(frames[1], { target: { value: '0' } });
    expect(onchange).not.toHaveBeenCalled();
  });
});

describe('prompt schedules (kind="text")', () => {
  it('emits a STRING, not a list', async () => {
    // REGRESSION: the editor split on newlines and emitted a list, but text_prompts is
    // Dict[int, str] — the config rejected it outright:
    //   ConfigError: config.text_prompts.0 must be str
    // Every prompt edit would have failed on Start render. Blending lives INSIDE the
    // string (`a:0.7 | b:0.3`), not in a list.
    const { onchange } = editor({ name: 'text_prompts', value: { 0: 'a cat' }, kind: 'text' });

    await fireEvent.change(screen.getByLabelText('Prompt'), { target: { value: 'a dog' } });

    expect(onchange).toHaveBeenCalledWith({ 0: 'a dog' });
    const emitted = onchange.mock.calls.at(-1)[0];
    expect(typeof emitted[0]).toBe('string');
    expect(Array.isArray(emitted[0])).toBe(false);
  });

  it('reads a list from an imported WarpFusion settings file', () => {
    // Settings files store {"0": ["prompt"]}, so a list can still arrive from an import.
    editor({ name: 'text_prompts', value: { 0: ['a cat', 'a dog'] }, kind: 'text' });
    expect(screen.getByLabelText('Prompt')).toHaveValue('a cat | a dog');
  });

  it('offers no Per-frame mode — prompts are keyed to frames, not indices', () => {
    editor({ name: 'text_prompts', value: { 0: 'a cat' }, kind: 'text' });
    expect(screen.queryByRole('button', { name: 'Per-frame' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Constant' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Keyframes' })).toBeInTheDocument();
  });
});

describe('sparkline', () => {
  const spark = () => document.querySelector('.spark');

  it('draws the shape of a multi-point schedule', () => {
    editor({ name: 'cfg_scale_schedule', value: { 0: 7, 20: 9, 40: 5 } });
    expect(spark()).toBeInTheDocument();
    expect(screen.getByText('5 – 9')).toBeInTheDocument();
  });

  it('is skipped where it would say nothing', () => {
    editor({ name: 'steps_schedule', value: [20] });        // a constant is a flat line
    expect(spark()).not.toBeInTheDocument();
  });

  it('is skipped for prompts — text has no magnitude', () => {
    editor({ name: 'text_prompts', value: { 0: 'a', 10: 'b' }, kind: 'text' });
    expect(spark()).not.toBeInTheDocument();
  });

  it('survives a flat schedule instead of dividing by zero', () => {
    editor({ name: 'steps_schedule', value: { 0: 5, 10: 5 } });
    expect(spark()).toBeInTheDocument();
    expect(spark().querySelector('path').getAttribute('d')).not.toContain('NaN');
  });
});
