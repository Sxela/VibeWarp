<script>
  import { diffSettings, formatSettingValue } from './settingsDiff.js';
  // Frame-by-frame comparison of a run's layers: init frame, warped init, each
  // ControlNet's source + detected map, the diffusion input, and the output.
  //
  // This is a MODE, not a settings tab, so it owns the whole window: runs down the left,
  // the viewer in the middle, layer chips pinned to the bottom. Nothing scrolls except the
  // runs list — the images scale into whatever height is left rather than pushing the page.
  let { config, job, onload, onjob } = $props();

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
  let renaming = $state(false);
  let draftLabel = $state('');
  let savingLabel = $state(false);
  let action = $state('');
  let showVideo = $state(false);
  let compareIds = $state([]);
  let diffRows = $state([]);
  let diffLoading = $state(false);
  let diffError = $state('');
  let showDiff = $state(false);

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

  function pick(id, event){
    if (event?.ctrlKey || event?.metaKey) {
      toggleRunComparison(id);
      return;
    }
    runId = id;
    renaming = false;
    showVideo = false;
    loadRun();
  }
  function runName(id){
    let run = runs.find(item => item.id === id);
    return run?.label || `Run #${id}`;
  }
  function toggleRunComparison(id){
    let next;
    if (compareIds.includes(id)) next = compareIds.filter(item => item !== id);
    else if (compareIds.length < 2) next = [...compareIds, id];
    else next = [compareIds[1], id];
    compareIds = next;
    showDiff = next.length === 2;
    diffRows = []; diffError = '';
    if (next.length === 2) loadSettingsDiff(next);
  }
  async function loadSettingsDiff(ids){
    let key = ids.join(':');
    diffLoading = true; diffError = '';
    try {
      let responses = await Promise.all(ids.map(id =>
        fetch(`/api/preview/runs/${id}/settings?${params}`, {method: 'POST'})));
      let bodies = await Promise.all(responses.map(response => response.json()));
      if (compareIds.join(':') !== key) return;
      let failed = responses.findIndex(response => !response.ok);
      if (failed >= 0) {
        let detail = bodies[failed].detail;
        diffError = detail?.[0]?.message || detail || `${runName(ids[failed])} has no readable settings`;
        return;
      }
      diffRows = diffSettings(bodies[0].config, bodies[1].config);
    } catch {
      if (compareIds.join(':') === key) diffError = 'Could not load settings for comparison';
    } finally {
      if (compareIds.join(':') === key) diffLoading = false;
    }
  }
  async function runAction(kind){
    if (!current || action || ['queued', 'running'].includes(job?.state)) return;
    action = kind; error = '';
    try {
      let r = await fetch(`/api/preview/runs/${runId}/${kind}?${params}`, {method: 'POST'});
      let d = await r.json();
      if (!r.ok) {
        error = typeof d.detail === 'string' ? d.detail : 'Could not start that action';
        return;
      }
      onjob?.(d);
      loaded = kind === 'resume'
        ? `Resuming run #${runId} from frame ${current.resume_from + 1}`
        : `Building video for run #${runId}`;
    } catch {
      error = `Could not ${kind === 'resume' ? 'resume the run' : 'build the video'}`;
    } finally { action = ''; }
  }
  function beginRename(){
    draftLabel = current?.label || '';
    renaming = true;
  }
  async function saveLabel(){
    if (!current || savingLabel) return;
    savingLabel = true; error = '';
    try {
      let r = await fetch(`/api/preview/runs/${runId}/label?${params}`, {
        method: 'PUT', headers: {'content-type': 'application/json'},
        body: JSON.stringify({label: draftLabel}),
      });
      let d = await r.json();
      if (!r.ok) { error = d.detail || 'Could not rename that run'; return; }
      runs = runs.map(run => run.id === runId ? {...run, label: d.label} : run);
      renaming = false;
    } catch { error = 'Could not rename that run'; }
    finally { savingLabel = false; }
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
  $effect(() => { params; compareIds = []; showDiff = false; loadRuns(); });

  // While a render is running, new frames keep appearing — refresh as it advances.
  let lastJobFrame = $state(-1);
  let lastTerminalJob = $state('');
  $effect(() => {
    if (job?.state === 'running' && job.frame !== lastJobFrame) {
      lastJobFrame = job.frame;
      loadRun();
    }
    if (job?.id && ['completed', 'cancelled', 'failed'].includes(job.state)
        && job.id !== lastTerminalJob) {
      lastTerminalJob = job.id;
      loadRuns();
      if (job.operation === 'video' && job.state === 'completed') showVideo = true;
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
    if (e.key === 'Escape') { zoom = null; showDiff = false; return; }
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.key === 'ArrowLeft') step(-1);
    if (e.key === 'ArrowRight') step(1);
  }
</script>

<svelte:window onkeydown={onKey}/>

<main class="preview-mode">
  <aside class="runs">
    <div class="runs-head">
      <div><p>Runs</p><span>Ctrl/Cmd-click two to diff</span></div>
      <button class="link" onclick={()=>loadRuns()} disabled={loading}>
        {loading ? 'Loading…' : 'Refresh'}
      </button>
    </div>
    <div class="run-list">
      {#each runs as run (run.id)}
        <button class="run-card" class:on={run.id === runId}
                class:compared={compareIds.includes(run.id)}
                onclick={(event)=>pick(run.id,event)} title={run.prompt}>
          {#if compareIds.includes(run.id)}
            <i class="compare-badge">{compareIds.indexOf(run.id) === 0 ? 'A' : 'B'}</i>
          {/if}
          <div class="thumb">
            {#if run.last_frame !== null}
              <img src={`/api/preview/runs/${run.id}/image?${params}&layer=output&frame=${run.last_frame}`}
                   alt={`Run ${run.id}`} loading="lazy"/>
            {:else}<span>none</span>{/if}
          </div>
          <div class="meta">
            <b>{run.label || `#${run.id}`}</b>
            <span>{run.label ? `#${run.id} · ` : ''}{run.frames} frame{run.frames === 1 ? '' : 's'} · {when(run.modified)}</span>
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
      {#if renaming}
        <form class="rename" onsubmit={(e)=>{e.preventDefault();saveLabel()}}>
          <input bind:value={draftLabel} maxlength="80" placeholder={`Run #${runId} label`}
                 aria-label="Run display label"/>
          <button type="submit" disabled={savingLabel}>{savingLabel ? 'Saving…' : 'Save'}</button>
          <button type="button" onclick={()=>renaming=false}>Cancel</button>
        </form>
      {:else}
        <button class="rename-action" onclick={beginRename} disabled={!current}>Rename run</button>
      {/if}
      <button class="load" onclick={loadSettings}
              disabled={!current?.has_settings || loadingSettings}
              title={current?.has_settings
                ? `Load run #${runId}'s settings into the render form`
                : 'This run saved no settings'}>
        {loadingSettings ? 'Loading…' : 'Load settings'}
      </button>
      <button class="history-action" onclick={()=>runAction('resume')}
              disabled={!current?.resume_available || action || ['queued','running'].includes(job?.state)}
              title={current?.resume_available
                ? `Continue at source frame ${current.resume_from + 1}`
                : 'This run is complete or has no resumable settings'}>
        {action === 'resume' ? 'Queuing…' : 'Resume'}
      </button>
      <button class="history-action" onclick={()=>runAction('video')}
              disabled={!current?.frames || action || ['queued','running'].includes(job?.state)}>
        {action === 'video' ? 'Queuing…' : current?.video_available ? 'Rebuild video' : 'Make video'}
      </button>
      {#if current?.video_available}
        <button class="history-action play" class:on={showVideo}
                onclick={()=>showVideo=!showVideo}>{showVideo ? 'View frames' : 'Play video'}</button>
      {/if}
    </div>
    {#if loaded}<div class="loaded">{loaded}</div>{/if}

    <div class="stage">
      {#if error}
        <div class="empty">{error}</div>
      {:else if showVideo && current?.video_available}
        <video class="video-player" controls preload="metadata"
               src={`/api/preview/runs/${runId}/video?${params}&v=${current.modified}`}>
          <track kind="captions"/>
        </video>
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

{#if showDiff}
  <div class="diff-backdrop" role="presentation" onclick={(event)=>{if(event.target===event.currentTarget)showDiff=false}}>
    <div class="diff-dialog" role="dialog" aria-modal="true" aria-label="Run settings difference">
      <header>
        <div>
          <p>Settings diff</p>
          <h2><span>A</span> {runName(compareIds[0])} <b>versus</b> <span>B</span> {runName(compareIds[1])}</h2>
        </div>
        <button onclick={()=>showDiff=false} aria-label="Close settings diff">×</button>
      </header>
      <div class="diff-body">
        {#if diffLoading}<div class="diff-state">Loading saved settings…</div>
        {:else if diffError}<div class="diff-state error">{diffError}</div>
        {:else if !diffRows.length}<div class="diff-state">These runs have identical settings.</div>
        {:else}
          <table>
            <thead><tr><th>Setting</th><th><i>A</i> {runName(compareIds[0])}</th><th><i>B</i> {runName(compareIds[1])}</th></tr></thead>
            <tbody>{#each diffRows as row (row.path)}
              <tr><th>{row.path}</th><td><code>{formatSettingValue(row.left)}</code></td><td><code>{formatSettingValue(row.right)}</code></td></tr>
            {/each}</tbody>
          </table>
        {/if}
      </div>
    </div>
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
  .runs-head span{display:block;margin-top:3px;color:#515966;font-size:9px}
  .link{border:0;background:none;color:#8ea834;font-size:11px;cursor:pointer;padding:4px}
  .link:disabled{color:#4b525b;cursor:default}
  .run-list{flex:1;min-height:0;overflow-y:auto;display:flex;flex-direction:column;gap:8px;padding:0 12px 16px}
  .none{color:#656d78;font-size:11px;line-height:1.5;padding:8px 2px}
  .run-card{flex:0 0 auto;display:flex;align-items:stretch;gap:9px;padding:0;
            border:1px solid #292e36;border-radius:9px;background:#12151a;overflow:hidden;
            cursor:pointer;text-align:left;position:relative}
  .run-card:hover{border-color:#4b525b}
  .run-card.on{border-color:#d8ff55;box-shadow:0 0 0 1px #d8ff55}
  .run-card.compared{box-shadow:0 0 0 2px #6ba8ff inset}
  .compare-badge{position:absolute;z-index:2;top:6px;left:6px;width:22px;height:22px;
        display:grid;place-items:center;border-radius:50%;background:#6ba8ff;color:#07111f;
        box-shadow:0 2px 8px #0009;font:700 11px Consolas,monospace;font-style:normal}
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

  .bar{flex:0 0 auto;display:flex;align-items:center;gap:8px;padding:12px 20px;
       border-bottom:1px solid #252930;overflow-x:auto}
  .scrub{flex:1;accent-color:#d8ff55;background:transparent}
  small{color:#68717d;font-size:11px;white-space:nowrap}
  .load{flex:0 0 auto;border:1px solid #3a4a26;background:#1a2113;color:#c8d3ad;
        border-radius:8px;padding:7px 13px;font-size:12px;cursor:pointer;white-space:nowrap}
  .load:hover:not(:disabled){border-color:#8ea834;background:#20290f;color:#d8ff55}
  .load:disabled{border-color:#2b3038;background:#12151a;color:#4b525b;cursor:default}
  .rename-action{flex:0 0 auto;border:1px solid #353a42;background:#171a1f;color:#aeb4bd;
        border-radius:8px;padding:7px 11px;font-size:12px;cursor:pointer;white-space:nowrap}
  .rename-action:hover:not(:disabled){border-color:#68717d;background:#20242a;color:#fff}
  .rename-action:disabled{opacity:.45;cursor:default}
  .history-action{flex:0 0 auto;border:1px solid #353a42;background:#171a1f;color:#cbd0d6;
        border-radius:8px;padding:7px 10px;font-size:12px;cursor:pointer;white-space:nowrap}
  .history-action:hover:not(:disabled){border-color:#8ea834;color:#d8ff55;background:#20290f}
  .history-action:disabled{opacity:.4;cursor:default}
  .history-action.play,.history-action.play.on{border-color:#4d5d2a;color:#d8ff55}
  .rename{display:flex;align-items:center;gap:5px;min-width:260px}
  .rename input{min-width:120px;padding:7px 9px;font-size:12px}
  .rename button{border:1px solid #3a414b;background:#191c21;color:#cbd0d6;border-radius:7px;
        padding:7px 9px;font-size:11px;cursor:pointer}
  .rename button[type="submit"]{border-color:#4d5d2a;color:#d8ff55}
  .rename button:hover:not(:disabled){background:#252a31}
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
  .video-player{width:100%;height:100%;object-fit:contain;background:#08090b;border-radius:10px}
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

  .diff-backdrop{position:fixed;inset:0;z-index:30;display:grid;place-items:center;
        padding:36px;background:#05070add;backdrop-filter:blur(5px)}
  .diff-dialog{width:min(1180px,96vw);height:min(780px,90vh);display:flex;flex-direction:column;
        overflow:hidden;border:1px solid #353b45;border-radius:13px;background:#0d1014;
        box-shadow:0 24px 80px #000c}
  .diff-dialog>header{flex:0 0 auto;display:flex;align-items:center;justify-content:space-between;
        gap:20px;padding:17px 20px;border-bottom:1px solid #292e36;background:#12161b}
  .diff-dialog header p{margin:0 0 5px;color:#69727d;font-size:10px;text-transform:uppercase;letter-spacing:.12em}
  .diff-dialog h2{margin:0;color:#e8ebef;font-size:15px;font-weight:600}
  .diff-dialog h2 span,.diff-dialog thead i{display:inline-grid;place-items:center;width:20px;height:20px;
        margin-right:5px;border-radius:50%;background:#6ba8ff;color:#07111f;
        font:700 10px Consolas,monospace;font-style:normal}
  .diff-dialog h2 b{margin:0 9px;color:#65707b;font-size:11px;font-weight:400}
  .diff-dialog>header>button{width:34px;height:34px;border:1px solid #353b45;border-radius:50%;
        background:#181c22;color:#cdd2d8;font-size:20px;cursor:pointer}
  .diff-dialog>header>button:hover{border-color:#c0505a;background:#c0505a;color:#fff}
  .diff-body{flex:1;min-height:0;overflow:auto}
  .diff-state{height:100%;display:grid;place-items:center;color:#7e8792;font-size:13px}
  .diff-state.error{color:#f18b91}
  .diff-dialog table{width:100%;border-collapse:collapse;table-layout:fixed}
  .diff-dialog thead{position:sticky;top:0;z-index:1;background:#14181e}
  .diff-dialog th,.diff-dialog td{padding:10px 13px;border-bottom:1px solid #242a32;
        text-align:left;vertical-align:top}
  .diff-dialog thead th{color:#aeb5bf;font-size:11px;font-weight:600}
  .diff-dialog thead th:first-child,.diff-dialog tbody th{width:28%}
  .diff-dialog tbody th{color:#8ea834;font:11px/1.45 Consolas,monospace;overflow-wrap:anywhere}
  .diff-dialog td{width:36%;background:#0a0d11}
  .diff-dialog td code{display:block;color:#d9dde2;font:11px/1.5 Consolas,monospace;
        white-space:pre-wrap;overflow-wrap:anywhere}
</style>
