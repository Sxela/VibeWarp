<script>
  // Frame-by-frame comparison of a run's layers: init frame, warped init, each
  // ControlNet's source + detected map, the diffusion input, and the output.
  //
  // This is a MODE, not a settings tab, so it owns the whole window: runs down the left,
  // the viewer in the middle, layer chips pinned to the bottom. Nothing scrolls except the
  // runs list — the images scale into whatever height is left rather than pushing the page.
  let { config, job, onload } = $props();

  const STORAGE_KEY = 'vibewarp.preview.layers.v1';
  const DEFAULT_LAYERS = ['init', 'output'];

  let runs = $state([]);
  let runId = $state('');
  let detail = $state(null);
  let frame = $state(0);
  let selected = $state(load());
  let loading = $state(false);
  let error = $state('');
  let zoom = $state(null);   // layer id shown in the lightbox
  let loadingSettings = $state(false);
  let loaded = $state('');   // transient confirmation

  // Pull an old run's settings back into the form. The point of keeping history is being
  // able to iterate on a run you liked, not just look at it.
  async function loadSettings(){
    let run = runs.find(r => r.id === runId);
    if (!run?.has_settings) return;
    loadingSettings = true; error = '';
    try {
      let r = await fetch(`/api/preview/runs/${runId}/settings?${params}`, {method: 'POST'});
      let d = await r.json();
      if (!r.ok) { error = d.detail?.[0]?.message || 'Could not load those settings'; return; }
      onload?.(d.config);
      loaded = `Loaded settings from run #${runId}`;
      setTimeout(() => { loaded = ''; }, 4000);
    } catch { error = 'Could not load those settings'; }
    finally { loadingSettings = false; }
  }
  let current = $derived(runs.find(r => r.id === runId));

  function pick(id){
    runId = id;
    loadRun();
  }
  function when(seconds){
    let d = new Date(seconds * 1000);
    let mins = Math.round((Date.now() - d) / 60000);
    if (mins < 1) return 'just now';
    if (mins < 60) return `${mins}m ago`;
    if (mins < 60 * 24) return `${Math.round(mins / 60)}h ago`;
    return d.toLocaleDateString();
  }

  function load(){
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) ?? DEFAULT_LAYERS; }
    catch { return DEFAULT_LAYERS; }
  }
  $effect(() => { localStorage.setItem(STORAGE_KEY, JSON.stringify(selected)); });

  let params = $derived(new URLSearchParams({
    output_dir: config?.output_dir || 'images_out',
    batch_name: config?.batch_name || 'warpfusion',
  }));

  async function loadRuns(keep = true){
    loading = true; error = '';
    try {
      let r = await fetch(`/api/preview/runs?${params}`);
      let d = await r.json();
      runs = d.runs ?? [];
      if (!runs.length) { detail = null; error = `No runs under ${d.root}`; return; }
      if (!keep || !runs.some(run => run.id === runId)) runId = runs[0].id;
      await loadRun();
    } catch { error = 'Could not list runs'; }
    finally { loading = false; }
  }

  async function loadRun(){
    if (!runId) return;
    error = '';
    try {
      let r = await fetch(`/api/preview/runs/${runId}?${params}`);
      if (!r.ok) { error = 'Could not read that run'; detail = null; return; }
      detail = await r.json();
      if (!detail.frames.length) { error = 'That run has no frames yet'; return; }
      // Keep the current frame if it still exists, else clamp into range.
      if (!detail.frames.includes(frame)) frame = detail.frames.at(-1);
    } catch { error = 'Could not read that run'; }
  }

  // Initial load, and reload when the output dir / batch name change.
  $effect(() => { params; loadRuns(); });

  // While a render is running, new frames keep appearing — refresh as it advances.
  let lastJobFrame = $state(-1);
  $effect(() => {
    if (job?.state === 'running' && job.frame !== lastJobFrame) {
      lastJobFrame = job.frame;
      loadRun();
    }
  });

  let groups = $derived(groupBy(detail?.layers ?? []));
  function groupBy(layers){
    let out = new Map();
    for (let layer of layers) {
      if (!out.has(layer.group)) out.set(layer.group, []);
      out.get(layer.group).push(layer);
    }
    return [...out];
  }
  let shown = $derived((detail?.layers ?? []).filter(l => selected.includes(l.id)));
  let frames = $derived(detail?.frames ?? []);
  let minFrame = $derived(frames.length ? frames[0] : 0);
  let maxFrame = $derived(frames.length ? frames.at(-1) : 0);

  function toggle(id){
    selected = selected.includes(id) ? selected.filter(x => x !== id) : [...selected, id];
  }
  function step(delta){
    frame = Math.min(maxFrame, Math.max(minFrame, frame + delta));
  }
  function src(layer){
    return `/api/preview/runs/${runId}/image?${params}&layer=${encodeURIComponent(layer)}&frame=${frame}`;
  }
  const has = (layer) => layer.frames.includes(frame);

  function onKey(e){
    if (e.key === 'Escape') { zoom = null; return; }
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.key === 'ArrowLeft') step(-1);
    if (e.key === 'ArrowRight') step(1);
  }
</script>

<svelte:window onkeydown={onKey}/>

<main class="preview-mode">
  <aside class="runs">
    <div class="runs-head">
      <p>Runs</p>
      <button class="link" onclick={()=>loadRuns()} disabled={loading}>
        {loading ? 'Loading…' : 'Refresh'}
      </button>
    </div>
    <div class="run-list">
      {#each runs as run (run.id)}
        <button class="run-card" class:on={run.id === runId} onclick={()=>pick(run.id)} title={run.prompt}>
          <div class="thumb">
            {#if run.last_frame !== null}
              <img src={`/api/preview/runs/${run.id}/image?${params}&layer=output&frame=${run.last_frame}`}
                   alt={`Run ${run.id}`} loading="lazy"/>
            {:else}<span>none</span>{/if}
          </div>
          <div class="meta">
            <b>#{run.id}</b>
            <span>{run.frames} frame{run.frames === 1 ? '' : 's'} · {when(run.modified)}</span>
            {#if run.prompt}<em>{run.prompt}</em>{/if}
          </div>
        </button>
      {:else}
        <p class="none">No runs under {config?.output_dir || 'images_out'}/{config?.batch_name || 'warpfusion'}</p>
      {/each}
    </div>
  </aside>

  <section class="viewer">
    <div class="bar">
      <div class="stepper">
        <button onclick={()=>step(-1)} disabled={frame <= minFrame} aria-label="Previous frame">‹</button>
        <input type="number" min={minFrame} max={maxFrame} bind:value={frame} aria-label="Frame"/>
        <button onclick={()=>step(1)} disabled={frame >= maxFrame} aria-label="Next frame">›</button>
      </div>
      <input class="scrub" type="range" min={minFrame} max={maxFrame} step="1"
             bind:value={frame} disabled={!frames.length} aria-label="Frame"/>
      <small>{minFrame}–{maxFrame} · ← → to step · click to zoom</small>
      <button class="load" onclick={loadSettings}
              disabled={!current?.has_settings || loadingSettings}
              title={current?.has_settings
                ? `Load run #${runId}'s settings into the render form`
                : 'This run saved no settings'}>
        {loadingSettings ? 'Loading…' : 'Load settings'}
      </button>
    </div>
    {#if loaded}<div class="loaded">{loaded}</div>{/if}

    <div class="stage">
      {#if error}
        <div class="empty">{error}</div>
      {:else if !shown.length}
        <div class="empty">Pick a layer below to compare.</div>
      {:else}
        <div class="frames" style={`--cols:${Math.min(shown.length, 3)}`}>
          {#each shown as layer (layer.id)}
            <figure>
              <figcaption>{layer.label}</figcaption>
              {#if has(layer)}
                <button class="shot" onclick={()=>zoom = layer.id} aria-label={`Zoom ${layer.label}`}>
                  <img src={src(layer.id)} alt={`${layer.label}, frame ${frame}`}/>
                </button>
              {:else}
                <div class="missing">Not produced for frame {frame}</div>
              {/if}
            </figure>
          {/each}
        </div>
      {/if}
    </div>

    <!-- Pinned: the chips ARE the controls for this view, so they never scroll away. -->
    <div class="layers">
      {#each groups as [group, items]}
        <div class="group">
          <span>{group}</span>
          {#each items as layer}
            <button class="chip" class:on={selected.includes(layer.id)} onclick={()=>toggle(layer.id)}>
              {layer.label}
            </button>
          {/each}
        </div>
      {/each}
    </div>
  </section>
</main>

{#if zoom}
  <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
  <div class="lightbox" onclick={()=>zoom = null}>
    <img src={src(zoom)} alt={`Frame ${frame}`}/>
    <button class="close" aria-label="Close zoom">×</button>
  </div>
{/if}

<style>
  /* NOT `.preview`: style.css already owns that class for the render monitor's thumbnail,
     and its `max-height:320px` was clamping this whole shell to 320px -- the layout cut off
     mid-screen, the image was cropped and the chips were pushed out of view.
     `min-height`/`padding-bottom` are reset because style.css sets them on every <main>. */
  .preview-mode{display:grid;grid-template-columns:345px minmax(0,1fr);
           /* The row MUST be bounded. Left as `auto` it sized itself to the images, which
              made .viewer taller than the shell and pushed the layer chips below the fold,
              where overflow:hidden swallowed them. `flex:1` on .stage can only shrink if
              its parent's height is actually constrained. */
           grid-template-rows:minmax(0,1fr);
           height:calc(100vh - 78px - var(--log-height, 282px));
           max-height:none;min-height:0;padding-bottom:0;overflow:hidden}

  .runs{display:flex;flex-direction:column;min-height:0;height:100%;
         border-right:1px solid #252930;background:#0e1014}
  .runs-head{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:16px 14px 10px}
  .runs-head p{margin:0;color:#7f8792;font-size:11px;text-transform:uppercase;letter-spacing:.1em}
  .link{border:0;background:none;color:#8ea834;font-size:11px;cursor:pointer;padding:4px}
  .link:disabled{color:#4b525b;cursor:default}
  .run-list{flex:1;min-height:0;overflow-y:auto;display:flex;flex-direction:column;gap:8px;padding:0 12px 16px}
  .none{color:#656d78;font-size:11px;line-height:1.5;padding:8px 2px}
  .run-card{flex:0 0 auto;display:flex;align-items:stretch;gap:9px;padding:0;
            border:1px solid #292e36;border-radius:9px;background:#12151a;overflow:hidden;
            cursor:pointer;text-align:left}
  .run-card:hover{border-color:#4b525b}
  .run-card.on{border-color:#d8ff55;box-shadow:0 0 0 1px #d8ff55}
  .thumb{flex:0 0 102px;width:102px;height:102px;display:grid;place-items:center;background:#08090b}
  .thumb img{width:100%;height:100%;object-fit:cover}
  .thumb span{color:#4b525b;font-size:9px}
  .meta{display:flex;flex-direction:column;justify-content:center;gap:3px;
        padding:9px 10px 9px 0;min-width:0}
  .meta b{color:#e7eaee;font:600 13px Consolas,monospace}
  .meta span{color:#8b929c;font-size:11px}
  .meta em{color:#68717d;font-size:10px;font-style:normal;line-height:1.4;
           display:-webkit-box;-webkit-line-clamp:3;line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}

  .viewer{display:flex;flex-direction:column;min-height:0;height:100%;min-width:0}

  .bar{flex:0 0 auto;display:flex;align-items:center;gap:14px;padding:12px 20px;border-bottom:1px solid #252930}
  .scrub{flex:1;accent-color:#d8ff55;background:transparent}
  small{color:#68717d;font-size:11px;white-space:nowrap}
  .load{flex:0 0 auto;border:1px solid #3a4a26;background:#1a2113;color:#c8d3ad;
        border-radius:8px;padding:7px 13px;font-size:12px;cursor:pointer;white-space:nowrap}
  .load:hover:not(:disabled){border-color:#8ea834;background:#20290f;color:#d8ff55}
  .load:disabled{border-color:#2b3038;background:#12151a;color:#4b525b;cursor:default}
  .loaded{flex:0 0 auto;padding:8px 20px;background:#1a2113;border-bottom:1px solid #3a4a26;
          color:#d8ff55;font-size:11px}
  .stepper{display:flex;gap:5px;align-items:center}
  .stepper input{width:70px;text-align:center;border:1px solid #303640;background:#0c0f13;
                 color:#eef0f2;border-radius:7px;padding:6px 8px;font:12px Consolas,monospace}
  .stepper button{border:1px solid #353a42;background:#191c21;color:#d9dce0;border-radius:7px;
                  padding:6px 11px;cursor:pointer;line-height:1}
  .stepper button:disabled{color:#4b525b;cursor:default}

  /* The images take whatever height is left and scale INTO it -- they never push the page.
     Every link in this chain needs min-height:0, or an intrinsically-sized <img> wins and
     the container grows to the image's natural height instead of the other way round. */
  .stage{flex:1;min-height:0;padding:16px 20px;display:flex;overflow:hidden}
  .frames{flex:1;min-height:0;display:grid;
          grid-template-columns:repeat(var(--cols),minmax(0,1fr));gap:12px}
  figure{margin:0;display:flex;flex-direction:column;min-height:0;min-width:0;
         border:1px solid #292e36;border-radius:10px;background:#0e1115;overflow:hidden}
  figcaption{flex:0 0 auto;padding:8px 11px;background:#12161b;color:#dfe3e8;font-size:11px}
  .shot{flex:1;min-height:0;padding:0;border:0;background:#08090b;cursor:zoom-in;
        display:flex;align-items:center;justify-content:center;overflow:hidden}
  .shot img{max-width:100%;max-height:100%;width:auto;height:auto;object-fit:contain;display:block}
  .missing{flex:1;display:grid;place-items:center;color:#656d78;font-size:11px}
  .empty{margin:auto;color:#656d78;font-size:12px;text-align:center}

  .layers{flex:0 0 auto;display:flex;flex-wrap:wrap;gap:10px 16px;padding:14px 20px;
          border-top:1px solid #252930;background:#0e1014}
  .group{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
  .group>span{color:#5f6771;font-size:10px;text-transform:uppercase;letter-spacing:.1em}
  .chip{border:1px solid #303640;background:#12151a;color:#aeb4bd;border-radius:20px;
        padding:9px 15px;font-size:12px;line-height:1.2;cursor:pointer}
  .chip:hover{border-color:#4b525b;color:#e7eaee}
  .chip.on{background:#d8ff55;border-color:#d8ff55;color:#111;font-weight:600}

  /* Click any image for full size; click anywhere (or Esc) to dismiss. */
  .lightbox{position:fixed;inset:0;z-index:20;background:#05070aee;display:grid;place-items:center;
            padding:32px;cursor:zoom-out;overflow:auto}
  .lightbox img{max-width:100%;max-height:100%;object-fit:contain}
  .close{position:fixed;top:18px;right:24px;width:36px;height:36px;border:1px solid #353a42;
         border-radius:50%;background:#12151a;color:#d9dce0;font-size:19px;line-height:1;cursor:pointer}
  .close:hover{background:#c0505a;color:#fff;border-color:#c0505a}
</style>
