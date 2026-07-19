<script>
  // Shown in the render monitor, in the space the first rendered frame will occupy — so it
  // is visible while you wait for that frame and then disappears for the rest of the
  // session. Deliberately not a banner, not a modal, and never in the way of the work.
  import { SUPPORTERS } from './supporters.js';

  const PATREON = 'https://www.patreon.com/sxela';

  // Rendered TWICE and scrolled by exactly -50%, so the loop is seamless — the second copy
  // is in place the instant the first scrolls out. The two halves wrap identically (same
  // content, same container width), which is what keeps the seam invisible now that the
  // names flow rather than sitting one per line.
  let names = [...SUPPORTERS, ...SUPPORTERS];
  // Speed follows the list length so a long one does not sprint, but a slow constant keeps
  // a ~90-name list from taking minutes to loop. Roughly one name per second, floor 12s.
  let duration = `${Math.max(12, Math.round(SUPPORTERS.length * 1.0))}s`;
</script>

<section class="supporters">
  <p class="head">Supported by</p>

  <div class="viewport" style={`--duration:${duration}`}>
    <ul class="track">
      {#each names as name, index}
        <li aria-hidden={index >= SUPPORTERS.length}>{name}</li>
      {/each}
    </ul>
  </div>

  <a class="patreon" href={PATREON} target="_blank" rel="noopener noreferrer">
    Support VibeWarp on Patreon
  </a>
</section>

<style>
  /* Takes whatever height is left in the aside (a flex column), instead of a fixed strip
     with dead space beneath it. min-height:0 so the marquee can actually shrink. */
  .supporters{flex:1;min-height:0;display:flex;flex-direction:column;
              margin-top:18px;padding:14px;border:1px solid #292e36;border-radius:10px;
              background:#0e1115}
  .head{margin:0 0 8px;color:#6f7883;font-size:10px;text-transform:uppercase;letter-spacing:.1em}

  /* Fixed height + hidden overflow is what makes the marquee a marquee. */
  .viewport{flex:1;min-height:88px;overflow:hidden;position:relative;
            /* Fade the ends so names emerge and dissolve instead of being chopped. */
            -webkit-mask-image:linear-gradient(180deg,transparent,#000 22%,#000 78%,transparent);
            mask-image:linear-gradient(180deg,transparent,#000 22%,#000 78%,transparent)}
  /* Names FLOW and wrap, several to a line — one per line wasted the width and showed only
     a handful of people per scroll cycle. */
  .track{margin:0;padding:0;list-style:none;
         display:flex;flex-wrap:wrap;justify-content:center;column-gap:7px;row-gap:2px;
         animation:scroll var(--duration) linear infinite}
  .viewport:hover .track{animation-play-state:paused}   /* let people actually read it */
  li{color:#aeb4bd;font-size:12px;line-height:1.55;white-space:nowrap}
  /* A separator after EVERY name, including the last: the list is duplicated, so the seam
     where the second copy begins needs one too, or two names would collide there. */
  li::after{content:'·';margin-left:7px;color:#4b525b}

  /* Exactly -50%: the list is duplicated, so half a loop lands the copy where the original
     started and the seam is invisible. */
  @keyframes scroll{
    from{transform:translateY(0)}
    to{transform:translateY(-50%)}
  }
  /* Respect the OS setting — a perpetually moving element is a genuine problem for some. */
  @media(prefers-reduced-motion:reduce){
    .track{animation:none}
    .viewport{flex:0 0 auto;min-height:0;overflow-y:auto;mask-image:none;-webkit-mask-image:none}
  }

  .patreon{flex:0 0 auto;display:block;margin-top:12px;padding:9px;border:1px solid #3a4a26;border-radius:8px;
           background:#1a2113;color:#c8d3ad;text-align:center;text-decoration:none;font-size:12px}
  .patreon:hover{border-color:#8ea834;background:#20290f;color:#d8ff55}
</style>
