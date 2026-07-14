<script>
  // frame_range is [start, end] — two numbers, not something anyone should be typing as
  // JSON. Both ends are INCLUSIVE: 0-15 renders sixteen frames. end = 0 means "to the end".
  //
  // `maxFrame` comes from probing the video (last renderable index, after extract_nth_frame).
  // Without it you can ask for frames that do not exist and only find out mid-render.
  let { value, onchange, maxFrame = 0 } = $props();

  let start = $derived(Array.isArray(value) ? (value[0] ?? 0) : 0);
  let end = $derived(Array.isArray(value) ? (value[1] ?? 0) : 0);

  const num = (raw, fallback) => {
    let n = Math.round(Number(raw));
    return Number.isFinite(n) && n >= 0 ? n : fallback;
  };
  const clamp = (n) => maxFrame > 0 ? Math.min(n, maxFrame) : n;
  function set(nextStart, nextEnd){ onchange([clamp(nextStart), clamp(nextEnd)]); }

  // Inclusive on both ends, and 0 means "everything".
  let count = $derived(
    end > start ? end - start + 1
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
             onchange={(e)=>set(num(e.target.value, start), end)}/>
    </label>
    <label><span>End <small>0 = all frames{maxFrame ? `, max ${maxFrame}` : ''}</small></span>
      <input type="number" min="0" max={maxFrame || null} step="1" value={end}
             onchange={(e)=>set(start, num(e.target.value, end))}/>
    </label>
  </div>
  {#if end > 0 && end < start}
    <p class="warn">End must be greater than or equal to start (or 0 to render every frame).</p>
  {:else if overrun}
    <p class="warn">This video only has {maxFrame + 1} frames (0–{maxFrame}).</p>
  {/if}
</div>

<style>
  .range{grid-column:1/-1;display:flex;flex-direction:column;gap:9px}
  .title{color:#aeb4bd;font-size:12px}
  .title small{margin-left:6px;color:#8ea834;font-family:Consolas,monospace}
  .pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
  label{display:flex;flex-direction:column;gap:7px;color:#aeb4bd;font-size:12px}
  small{color:#68717d}
  input{width:100%;border:1px solid #303640;background:#0c0f13;color:#eef0f2;border-radius:8px;padding:10px 11px;outline:none;font:12px Consolas,monospace}
  input:focus{border-color:#8ea834}
  .warn{margin:0;color:#e0c169;font-size:11px}
  @media(max-width:700px){.pair{grid-template-columns:1fr}}
</style>
