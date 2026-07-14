<script>
  // `errors` is {field: message} for the video section. This editor hand-rolls its inputs
  // instead of using Field, so it carries the anchor ids and error slots itself.
  //
  // It also probes the video (ffprobe, falling back to OpenCV) as soon as a path is set:
  // the source metadata, the size the render will ACTUALLY be, and the real frame ceiling
  // were all previously invisible until the render started.
  let { value, onchange, errors = {}, frameRange, onFrameRange } = $props();
  import FrameRange from './FrameRange.svelte';
  import { stripQuotes } from './paths.js';
  import { derive } from './videoMath.js';

  function patch(changes){ onchange({...value, ...changes}); }

  let source = $state(null);      // the probed video, keyed on PATH alone
  let probing = $state(false);
  let probeError = $state('');

  // Read the video ONLY when the path changes. max_size and extract_nth_frame just move
  // arithmetic (see videoMath.js) -- re-running ffprobe for a slider nudge is wasted work.
  let path = $derived(value.video_init_path || '');
  $effect(() => {
    let current = path;
    if (!current) { source = null; probeError = ''; return; }
    let cancelled = false;
    probing = true;
    fetch(`/api/video/probe?path=${encodeURIComponent(current)}`)
      .then(async (r) => {
        if (cancelled) return;
        let d = await r.json();
        if (!r.ok) { source = null; probeError = d.detail?.[0]?.message || 'Could not read that video'; }
        else { source = d.source; probeError = ''; }
      })
      .catch(() => { if (!cancelled) { source = null; probeError = 'Could not reach the server'; } })
      .finally(() => { if (!cancelled) probing = false; });
    return () => { cancelled = true; };
  });

  // Recomputed locally as you tweak — no server round-trip, no video decode.
  let probe = $derived(source
    ? derive(source, value.max_size ?? 0, value.extract_nth_frame ?? 1, frameRange)
    : null);

  const setPath = (e) => patch({video_init_path: stripQuotes(e.target.value)});
  const pretty = (n) => n >= 60 ? `${Math.floor(n / 60)}m ${Math.round(n % 60)}s` : `${n.toFixed(1)}s`;
  let thumb = $derived(path && source
    ? `/api/video/thumbnail?path=${encodeURIComponent(path)}` : '');
</script>

<div class="grid">
  <label class="wide" id="field-video-video_init_path"><span>Input video path</span>
    <input class:invalid={!!errors.video_init_path || (!!probeError && !probing)}
           value={value.video_init_path} placeholder="C:\videos\input.mp4" onchange={setPath}/>
    {#if errors.video_init_path}<small class="err">{errors.video_init_path}</small>
    {:else if probeError}<small class="err">{probeError}</small>
    {:else if probing}<small>Reading video…</small>{/if}
  </label>

  {#if source && probe}
    <div class="meta wide">
      <!-- The first non-black frame: clips that fade in would otherwise show a black
           thumbnail, which tells you nothing about the video you just picked. -->
      <img src={thumb} alt="First frame of the input video"/>
      <dl>
        <div><dt>Source</dt>
             <dd>{source.width}×{source.height}
                 {#if source.codec}<em>{source.codec}</em>{/if}</dd></div>
        <div><dt>Length</dt>
             <dd>{source.frames} frames · {source.fps} fps
                 {#if source.duration}· {pretty(source.duration)}{/if}</dd></div>
        <div><dt>Renders at</dt>
             <dd class="hi">{probe.render.width}×{probe.render.height}</dd></div>
        <!-- What survives BOTH extract_nth_frame and frame_range. Showing the extracted
             count here merely echoed the source length when nth = 1, and ignored the frame
             range entirely — a 12-frame render advertised itself as 559. -->
        <div><dt>Frames to render</dt>
             <dd class="hi">{probe.renderCount}
               {#if probe.extractNth > 1 || probe.renderCount !== source.frames}
                 <em>of {source.frames}{#if probe.extractNth > 1}, every {probe.extractNth}th{/if}</em>
               {/if}</dd></div>
      </dl>
    </div>
  {/if}

  <label id="field-video-max_size"><span>Max size</span>
    <input class:invalid={!!errors.max_size} type="number" min="8" step="8"
           value={value.max_size} onchange={(e)=>patch({max_size: Number(e.target.value)})}/>
    {#if errors.max_size}<small class="err">{errors.max_size}</small>
    {:else}<small>Longest edge. The render size above is the exact result, rounded to /8.</small>{/if}
  </label>

  <label id="field-video-extract_nth_frame"><span>Extract every Nth frame</span>
    <input class:invalid={!!errors.extract_nth_frame} type="number" min="1" step="1"
           value={value.extract_nth_frame}
           onchange={(e)=>patch({extract_nth_frame: Math.max(1, Number(e.target.value))})}/>
    {#if errors.extract_nth_frame}<small class="err">{errors.extract_nth_frame}</small>{/if}
  </label>

  <FrameRange value={frameRange} onchange={onFrameRange} maxFrame={probe?.maxFrame ?? 0}/>
</div>

<style>
  .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.wide{grid-column:1/-1}
  input,select{width:100%;border:1px solid #303640;background:#0c0f13;color:#eef0f2;border-radius:8px;padding:10px 11px;outline:none;font:12px Consolas,monospace}
  input:focus,select:focus{border-color:#8ea834}
  small{color:#68717d;line-height:1.4}
  label{display:flex;flex-direction:column;gap:7px;color:#aeb4bd;font-size:12px}
  .invalid{border-color:#c0505a}
  .err{display:block;margin-top:6px;color:#f18b91;font:11px/1.4 system-ui,sans-serif}

  .meta{display:flex;gap:16px;align-items:center;padding:12px;border:1px solid #292e36;
        border-radius:10px;background:#0e1115}
  .meta img{flex:0 0 132px;width:132px;aspect-ratio:16/9;object-fit:cover;
            border-radius:7px;background:#08090b}
  .meta dl{margin:0;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px 18px;flex:1}
  .meta dl div{display:flex;flex-direction:column;gap:2px;min-width:0}
  .meta dt{color:#6f7883;font-size:10px;text-transform:uppercase;letter-spacing:.09em}
  .meta dd{margin:0;color:#dfe3e8;font:12px Consolas,monospace}
  .meta dd.hi{color:#d8ff55}
  .meta em{margin-left:5px;color:#68717d;font-style:normal;font-size:11px}
  @media(max-width:700px){.grid{grid-template-columns:1fr}.wide{grid-column:1}
                          .meta{flex-direction:column;align-items:stretch}
                          .meta img{width:100%;flex:none}}
</style>
