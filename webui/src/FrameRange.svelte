<script>
  import HelpTip from './HelpTip.svelte';
  // frame_range is [start, end] — two numbers, not something anyone should be typing as
  // JSON. Both ends are INCLUSIVE: 0-15 renders sixteen frames. end = 0 means "to the end".
  //
  // `maxFrame` comes from probing the video (last renderable index, after extract_nth_frame).
  // Without it you can ask for frames that do not exist and only find out mid-render.
  let { value, onchange, maxFrame = 0 } = $props();

  let start = $derived(Array.isArray(value) ? (value[0] ?? 0) : 0);
  let single = $derived(value?.length===1);
  let end = $derived(single?start:Array.isArray(value) ? (value[1] ?? 0) : 0);

  const num = (raw, fallback) => {
    let n = Math.round(Number(raw));
    return Number.isFinite(n) && n >= 0 ? n : fallback;
  };
  const clamp = (n) => maxFrame > 0 ? Math.min(n, maxFrame) : n;
  function set(nextStart, nextEnd){ onchange(single&&nextStart===nextEnd?[clamp(nextStart)]:[clamp(nextStart),clamp(nextEnd)]); }

  // Inclusive on both ends, and 0 means "everything".
  let count = $derived(
    single ? 1 : end > start ? end - start + 1
    : end === 0 && maxFrame > 0 ? maxFrame - start + 1
    : 0);
  let overrun = $derived(maxFrame > 0 && (start > maxFrame || end > maxFrame));
</script>

<div class="range">
  <span class="title">Frame Range
    {#if count > 0}<small>{count} frame{count === 1 ? '' : 's'}</small>{/if}
  </span>
  <div class="pair">
    <label><span>Start</span>
      <input type="number" min="0" max={maxFrame || null} step="1" value={start}
             onchange={(e)=>{const next=num(e.target.value,start);set(next,single?next:end);}}/>
    </label>
    <label><span>End {#if single}<small>Single frame</small>{:else}<HelpTip label="Help: End frame" text={`0 = all frames${maxFrame ? `, max ${maxFrame}` : ''}. Both range ends are inclusive.`}/>{/if}</span>
      <input type="number" min="0" max={maxFrame || null} step="1" value={end}
             onchange={(e)=>set(start, num(e.target.value, end))}/>
    </label>
  </div>
  {#if single}<button class="all-frames" onclick={()=>onchange([start,0])}>Use remaining frames</button>{/if}
  {#if end > 0 && end < start}
    <p class="warn">End must be greater than or equal to start (or 0 to render every frame).</p>
  {:else if overrun}
    <p class="warn">This video only has {maxFrame + 1} frames (0–{maxFrame}).</p>
  {/if}
</div>

<style>
  .range{grid-column:1/-1;display:flex;flex-direction:column;gap:9px}
  .title{color:var(--muted);font-size:var(--text-label)}
  .title small{margin-left:6px;color:var(--accent-muted);font-family:Consolas,monospace}
  .pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
  label{display:flex;flex-direction:column;gap:7px;color:var(--muted);font-size:var(--text-label)}
  small{color:var(--muted)}
  input{width:100%;border:1px solid var(--input-border);background:var(--input-bg);color:var(--ink);border-radius:var(--radius);padding:10px 11px;font-size:var(--text-value)}
  input:focus{border-color:var(--accent-muted)}
  .warn{margin:0;color:var(--warning);font-size:var(--text-help)}
  @media(max-width:700px){.pair{grid-template-columns:1fr}}
</style>
