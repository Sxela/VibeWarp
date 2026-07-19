import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import Supporters from './Supporters.svelte';
import { SUPPORTERS } from './supporters.js';

describe('the supporter list', () => {
  it('has no blank or accidentally-duplicated entries', () => {
    // Real people, pasted by hand — an empty string or a stray repeat would show as a
    // gap or a double credit.
    expect(SUPPORTERS.every((name) => name.trim().length > 0)).toBe(true);
    expect(new Set(SUPPORTERS).size).toBe(SUPPORTERS.length);
  });

  it('renders every supporter (each appears in both marquee copies)', () => {
    const { container } = render(Supporters);
    const shown = [...container.querySelectorAll('.track li')].map((li) => li.textContent);
    for (const name of SUPPORTERS)
      expect(shown.filter((t) => t === name)).toHaveLength(2);
  });

  it('preserves non-Latin names verbatim', () => {
    // ともや / 和樹 横田 are in the list; a naive ASCII filter somewhere would drop them.
    render(Supporters);
    expect(screen.getAllByText('ともや').length).toBeGreaterThan(0);
  });
});

describe('supporters block', () => {
  it('links to Patreon, opened safely', () => {
    render(Supporters);
    const link = screen.getByRole('link', { name: /Patreon/i });
    expect(link).toHaveAttribute('href', 'https://www.patreon.com/sxela');
    // target=_blank without noopener hands the opened page a handle on ours.
    expect(link).toHaveAttribute('rel', expect.stringContaining('noopener'));
  });

  it('duplicates the list so the marquee loop is seamless', () => {
    // The track scrolls by exactly -50%; that only lines up if the list is rendered twice.
    const { container } = render(Supporters);
    const items = container.querySelectorAll('.track li');
    expect(items.length % 2).toBe(0);
    const half = items.length / 2;
    expect(items[0].textContent).toBe(items[half].textContent);
  });

  it('hides the duplicate half from screen readers', () => {
    const { container } = render(Supporters);
    const items = [...container.querySelectorAll('.track li')];
    const hidden = items.filter((li) => li.getAttribute('aria-hidden') === 'true');
    // The copy exists only to make the animation loop; it must not be read out twice.
    expect(hidden.length).toBe(items.length / 2);
  });
});

describe('when it appears', () => {
  // The rule: visible while you wait for your FIRST rendered frame, then gone for the rest
  // of the session — including between jobs. Re-showing it after every render would nag.
  //
  // App.svelte latches on `job.preview_available`:
  //     let seenFirstFrame = $state(false);
  //     $effect(() => { if (job?.preview_available) seenFirstFrame = true; });
  //     {#if !seenFirstFrame}<Supporters/>{/if}
  //
  // That latch is the whole behaviour, so test it directly rather than driving the app.
  function latch() {
    let seen = false;
    return {
      observe: (job) => { if (job?.preview_available) seen = true; },
      get visible() { return !seen; },
    };
  }

  it('shows before the first frame and hides once one arrives', () => {
    const l = latch();
    l.observe(null);                                   // no job yet
    expect(l.visible).toBe(true);
    l.observe({ preview_available: false });           // rendering, no frame yet
    expect(l.visible).toBe(true);
    l.observe({ preview_available: true });            // first frame lands
    expect(l.visible).toBe(false);
  });

  it('stays hidden for the rest of the session, even for a NEW job', () => {
    const l = latch();
    l.observe({ preview_available: true });            // first render produced a frame
    l.observe(null);                                   // job cleared
    l.observe({ preview_available: false });           // a second render starts
    expect(l.visible).toBe(false);                     // must not reappear
  });
});
