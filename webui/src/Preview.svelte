<script>
  import {tick} from 'svelte';
  import {closeOnSelection} from './closeOnSelection.js';
  import LayerViewer from './LayerViewer.svelte';
  import ImageZoom from './ImageZoom.svelte';
  import {derive as deriveVideo} from './videoMath.js';
  import {
    annotateUnsaved,
    diffSettings,
    formatSettingValue,
    unsavedOnlyRows,
  } from './settingsDiff.js';
  import {
    filterRunsByMinimumFrames,
    frameAfterRunLoad,
    latestRenderedFrame,
    normalizeMinimumFrames,
    renderedFrames,
  } from './historyFilters.js';
  import {
    PLAYBACK_FPS,
    PLAYBACK_INTERVAL_MS,
    canPlay,
    nextFrame,
  } from './framePlayback.js';
  // Frame-by-frame comparison of a run's layers: init frame, warped init, each
  // ControlNet's source + detected map, the diffusion input, and the output.
  //
  // This is a MODE, not a settings tab, so it owns the whole window: runs down the left,
  // the viewer in the middle, layer chips pinned to the bottom. Nothing scrolls except the
  // runs list — the images scale into whatever height is left rather than pushing the page.
  let { config, job, onload, onjob, onsource=()=>{}, history=true, view=$bindable('inspect') } = $props();
  let focusedLayer=$state(null), sourceInfo=$state(null), sourceError=$state(''), showRuns=$state(false);
  let sourcePath=$derived(config?.video?.video_init_path||'');
  $effect(()=>{
    const path=sourcePath;
    sourceInfo=null;sourceError='';
    if(!path)return;
    const controller=new AbortController();
    fetch(`/api/video/probe?path=${encodeURIComponent(path)}`,{signal:controller.signal})
      .then(async response=>{const result=await response.json();if(controller.signal.aborted)return;if(!response.ok)throw new Error(result.detail?.[0]?.message||'Could not read the input video');sourceInfo=result.source;})
      .catch(error=>{if(error.name!=='AbortError')sourceError=error.message;});
    return()=>controller.abort();
  });
  let sourceProbe=$derived(sourceInfo?deriveVideo(sourceInfo,config?.video?.max_size||0,config?.video?.extract_nth_frame||1,config?.frame_range):null);
  let sourceFirst=$derived(Math.min(sourceProbe?.maxFrame??Infinity,Math.max(0,Number(config?.frame_range?.[0]||0))));
  let sourceLast=$derived(config?.frame_range?.length===1?sourceFirst:Math.max(sourceFirst,Math.min(sourceProbe?.maxFrame||0,config?.frame_range?.[1]>0?config.frame_range[1]:sourceProbe?.maxFrame||0)));

  const STORAGE_KEY = 'vibewarp.preview.layers.v1';
  const MIN_FRAMES_STORAGE_KEY = 'vibewarp.preview.minimumFrames.v1';
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
  let cancelling = $state(false);
  let showVideo = $state(false);
  let renameInput=$state(null);
  let compareIds = $state([]);
  let diffRows = $state([]);
  let diffLoading = $state(false);
  let diffError = $state('');
  let showDiff = $state(false);
  let minimumFrames = $state(loadMinimumFrames());
  let sourceUrl=$derived(!history&&sourceInfo&&frame>=sourceFirst&&frame<=sourceLast?`/api/video/thumbnail?path=${encodeURIComponent(sourcePath)}&frame=${frame*Math.max(1,config?.video?.extract_nth_frame||1)}`:'');
  $effect(()=>{if(!history&&sourceInfo&&(frame<sourceFirst||frame>sourceLast))frame=sourceFirst;});
  let liveUrl=$derived(!history&&job?.preview_available&&frame===job.frame&&(!job.run_id||job.run_id===runId)?`/api/jobs/${job.id}/preview?v=${job.revision}`:'');

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
  let activeJob = $derived(
    job && ['queued', 'running'].includes(job.state) ? job : null);

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
    const existing=compareIds.indexOf(id);
    assignComparison(id,existing>=0?existing:!compareIds[0]?0:1);
  }
  function assignComparison(id, slot){
    let next=[...compareIds];
    if(next[1-slot]===id)next[1-slot]=undefined;
    next[slot]=next[slot]===id?undefined:id;
    compareIds=next;
    showDiff=!!next[0]&&!!next[1];diffRows=[];diffError='';
    if(showDiff)loadSettingsDiff(next);
  }
  async function loadSettingsDiff(ids){
    let key = ids.join(':');
    diffLoading = true; diffError = '';
    try {
      // overlay_system=false: with the overlay on, BOTH runs are stamped with
      // this machine's current system tier, so any difference in multiscale,
      // the tiled sampler, caching and so on silently reads as "no change" —
      // even though those visibly alter the render.
      let responses = await Promise.all(ids.map(id =>
        fetch(`/api/preview/runs/${id}/settings?${params}&overlay_system=false`,
              {method: 'POST'})));
      let bodies = await Promise.all(responses.map(response => response.json()));
      if (compareIds.join(':') !== key) return;
      let failed = responses.findIndex(response => !response.ok);
      if (failed >= 0) {
        let detail = bodies[failed].detail;
        diffError = detail?.[0]?.message || detail || `${runName(ids[failed])} has no readable settings`;
        return;
      }
      diffRows = [
        ...annotateUnsaved(
          diffSettings(bodies[0].config, bodies[1].config),
          bodies[0].saved, bodies[1].saved),
        ...unsavedOnlyRows(bodies[0].saved, bodies[1].saved,
                           diffSettings(bodies[0].config, bodies[1].config)),
      ];
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
  async function cancelActiveJob(){
    if (!activeJob || cancelling || activeJob.cancel_requested) return;
    cancelling = true; error = '';
    try {
      let r = await fetch(`/api/jobs/${activeJob.id}/cancel`, {method: 'POST'});
      let d = await r.json();
      if (!r.ok) {
        error = typeof d.detail === 'string' ? d.detail : 'Could not cancel the render';
        return;
      }
      onjob?.(d);
    } catch {
      error = 'Could not cancel the render';
    } finally {
      cancelling = false;
    }
  }
  async function beginRename(){
    draftLabel = current?.label || '';
    renaming = true;
    await tick();
    renameInput?.focus();
    renameInput?.select();
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
    let d = new Date(typeof seconds==='number'?seconds*1000:seconds);
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
  function loadMinimumFrames(){
    try {
      return normalizeMinimumFrames(
        localStorage.getItem(MIN_FRAMES_STORAGE_KEY) ?? 0);
    } catch {
      return 0;
    }
  }
  let visibleRuns = $derived(
    filterRunsByMinimumFrames(runs, minimumFrames));
  function setMinimumFrames(value){
    minimumFrames = normalizeMinimumFrames(value);
    try {
      localStorage.setItem(MIN_FRAMES_STORAGE_KEY, String(minimumFrames));
    } catch {}
    let available = filterRunsByMinimumFrames(runs, minimumFrames);
    let availableIds = new Set(available.map(run => run.id));
    compareIds = compareIds.map(id => availableIds.has(id)?id:undefined);
    if (!compareIds[0]||!compareIds[1]) showDiff = false;
    if (!available.length) {
      runId = '';
      detail = null;
      error = minimumFrames
        ? `No runs rendered at least ${minimumFrames} frame${minimumFrames===1?'':'s'}`
        : '';
      return;
    }
    error = '';
    if (!available.some(run => run.id === runId)) {
      runId = available[0].id;
      loadRun();
    }
  }

  let params = $derived(new URLSearchParams({
    output_dir: config?.output_dir || 'images_out',
    batch_name: config?.batch_name || 'warpfusion',
  }));

  let listRevision=0, detailRevision=0;
  async function loadRuns(keep = true){
    const revision=++listRevision;
    detailRevision+=1;
    loading = true; error = '';
    try {
      let r = await fetch(`/api/preview/runs?${params}`);
      let d = await r.json();
      if(revision!==listRevision)return;
      runs = d.runs ?? [];
      if (!runs.length) { detail = null; runId='';error = history?`No runs under ${d.root}`:''; return; }
      let available = filterRunsByMinimumFrames(runs, minimumFrames);
      if (!available.length) {
        runId = ''; detail = null;
        error = `No runs rendered at least ${minimumFrames} frame${minimumFrames===1?'':'s'}`;
        return;
      }
      let switched = !keep || !available.some(run => run.id === runId);
      if (switched) runId = !history&&job?.run_id&&available.some(run=>run.id===job.run_id)?job.run_id:available[0].id;
      await loadRun(!detail);
    } catch { if(revision===listRevision)error = 'Could not list runs'; }
    finally { if(revision===listRevision)loading = false; }
  }

  async function loadRun(selectLatestOutput = false){
    if (!runId) return;
    const revision=++detailRevision, requestedRun=runId;
    error = '';
    try {
      let previousRun = detail?.run;
      let previousLatest = latestRenderedFrame(detail);
      let r = await fetch(`/api/preview/runs/${runId}?${params}`);
      if(revision!==detailRevision||requestedRun!==runId)return;
      if (!r.ok) { error = 'Could not read that run'; detail = null; return; }
      const next=await r.json();
      if(revision!==detailRevision||requestedRun!==runId)return;
      detail = next;
      syncActiveRunSummary();
      if (!detail.frames.length) return;
      let latest = latestRenderedFrame(detail);
      // The frame number is global across history runs. Only the initial load
      // selects the latest output; an in-progress run keeps following its own
      // latest frame until the user scrubs away from it.
      const nextFrame = frameAfterRunLoad({
        selectedFrame: frame,
        previousRun,
        nextRun: runId,
        previousLatest,
        nextLatest: latest,
        selectLatest: selectLatestOutput,
      });
      if(history||frame!==job?.frame)frame=nextFrame??frame;
    } catch { if(revision===detailRevision)error = 'Could not read that run'; }
  }

  // The sidebar card's frame count and thumbnail come from the runs LIST,
  // which is only refetched when a job reaches a terminal state — so a
  // rendering run's count sat frozen at whatever it was when the list loaded
  // until you refreshed. `detail` is already refetched on every frame, so
  // patch the row from it rather than re-listing every run once per frame.
  function syncActiveRunSummary(){
    if (!runId || !detail) return;
    let rendered = renderedFrames(detail);
    runs = runs.map(run => run.id === runId
      ? {...run,
         frames: rendered.length,
         last_frame: rendered.length ? rendered.at(-1) : null}
      : run);
  }

  // Initial load, and reload when the output dir / batch name change.
  $effect(() => { params; compareIds = []; showDiff = false; detail=null;runs=[];runId='';loadRuns(false); });

  // A run that started after this list loaded has no row to patch. Bring it in
  // once — not on every frame, which would rescan every run on disk — and only
  // after the initial listing settles, or the empty list mid-load looks like a
  // missing run. Latched per run id so a run whose directory does not exist yet
  // cannot trigger a relist per frame.
  let relistedFor = $state('');
  $effect(() => {
    if (loading || job?.state !== 'running' || !job.run_id) return;
    if (relistedFor === job.run_id) return;
    if (runs.some(run => run.id === job.run_id)) return;
    relistedFor = job.run_id;
    loadRuns();
  });

  // While a render is running, new frames keep appearing — refresh as it advances.
  let lastJobFrame = $state(-1);
  let lastTerminalJob = $state('');
  $effect(() => {
    if (job?.state === 'running' && job.frame !== lastJobFrame) {
      if(!history&&(frame===lastJobFrame||lastJobFrame<0))frame=job.frame;
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
  $effect(()=>{
    if(!history&&job?.run_id&&runs.some(run=>run.id===job.run_id)&&runId!==job.run_id){runId=job.run_id;loadRun();}
  });

  let inspectLayers=$derived([...(detail?.layers||[]),...(detail?.missing_layers||[])]);
  let frames = $derived(detail?.frames ?? []);
  let minFrame = $derived(!history&&sourceInfo?sourceFirst:frames.length?frames[0]:sourceFirst);
  let maxFrame = $derived(!history&&sourceInfo?sourceLast:frames.length?frames.at(-1):sourceLast);

  // Loop the rendered frames in place, so a run can be judged as motion
  // without assembling a video first. One button toggles play and pause.
  //
  // Deliberately NOT `frames`: that is the union of every layer, and extracted
  // inputs already span the whole selected range, so playing it would crawl
  // through frames the run has not rendered yet.
  let playing = $state(false);
  let playbackFrames = $derived(renderedFrames(detail).filter(index=>index>=minFrame&&index<=maxFrame));
  let playable = $derived(canPlay(playbackFrames)||(!history&&!playbackFrames.length&&sourceLast>sourceFirst));

  function step(delta){
    playing = false;   // a manual move means you want to look at that frame
    frame = Math.min(maxFrame, Math.max(minFrame, frame + delta));
  }

  // `playing`/`playable` are read synchronously so they are the effect's only
  // dependencies; `frame` and `playbackFrames` are read inside the timer
  // callback, which runs outside the tracked pass, so ticking does not restart
  // the interval (and a still-rendering run picks up new frames as they land).
  $effect(() => {
    if (!playing || !playable) return;
    const timer = setInterval(
      () => { frame = playbackFrames.length?nextFrame(frame, playbackFrames):frame>=sourceLast?sourceFirst:frame+1; },
      PLAYBACK_INTERVAL_MS);
    return () => clearInterval(timer);
  });

  // Switching runs, or losing the frame list, must not leave playback running
  // against frames that no longer exist.
  $effect(() => { if (!playable) playing = false; });
  $effect(() => { runId; playing = false; });
  function src(layer){
    if(layer==='init'&&sourceUrl)return sourceUrl;
    if(layer==='output'&&liveUrl)return liveUrl;
    return `/api/preview/runs/${runId}/image?${params}&layer=${encodeURIComponent(layer)}&frame=${frame}`;
  }
  let downloadableLayer=$derived(view==='source'?'init':view==='output'?'output':view==='inspect'?focusedLayer:detail?.layers?.find(layer=>layer.id==='output')?.frames.includes(frame)||liveUrl?'output':'init');
  let canDownload=$derived(downloadableLayer==='init'&&sourceUrl||downloadableLayer==='output'&&liveUrl||detail?.layers?.some(layer=>layer.id===downloadableLayer&&layer.frames.includes(frame)));
  let stripFrames=$derived.by(()=>{
    if(!history&&sourceInfo){
      const count=Math.min(12,sourceLast-sourceFirst+1);
      return Array.from({length:count},(_,index)=>Math.round(sourceFirst+index*(sourceLast-sourceFirst)/Math.max(1,count-1)));
    }
    const available=playbackFrames.length?playbackFrames:frames;
    if(available.length){const count=Math.min(12,available.length);return Array.from({length:count},(_,index)=>available[Math.round(index*(available.length-1)/Math.max(1,count-1))]);}
    if(!sourceInfo)return [];
    const count=Math.min(12,sourceLast-sourceFirst+1);
    return Array.from({length:count},(_,index)=>Math.round(sourceFirst+index*(sourceLast-sourceFirst)/Math.max(1,count-1)));
  });
  function stripImage(index){
    const output=detail?.layers?.find(layer=>layer.id==='output');
    if(output?.frames.includes(index))return `/api/preview/runs/${runId}/thumbnail?${params}&layer=output&frame=${index}`;
    if(detail?.layers?.find(layer=>layer.id==='init')?.frames.includes(index))return `/api/preview/runs/${runId}/thumbnail?${params}&layer=init&frame=${index}`;
    return `/api/video/thumbnail?path=${encodeURIComponent(sourcePath)}&frame=${index*Math.max(1,config?.video?.extract_nth_frame||1)}`;
  }

  function onKey(e){
    if (e.key === 'Escape') { zoom = null; showDiff = false; return; }
    if (['INPUT','SELECT','TEXTAREA','BUTTON','SUMMARY'].includes(e.target.tagName)||e.target.isContentEditable) return;
    if (e.key === 'ArrowLeft') step(-1);
    if (e.key === 'ArrowRight') step(1);
  }
  function trapDialog(node){
    const previous=document.activeElement;
    node.querySelector('button')?.focus();
    const onTab=event=>{
      if(event.key!=='Tab')return;
      const targets=[...node.querySelectorAll('button:not(:disabled),a[href],input,select,textarea,[tabindex="0"]')];
      const first=targets[0],last=targets.at(-1);
      if(event.shiftKey&&document.activeElement===first){event.preventDefault();last?.focus();}
      else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first?.focus();}
    };
    node.addEventListener('keydown',onTab);
    return {destroy(){node.removeEventListener('keydown',onTab);previous?.focus();}};
  }
</script>
<svelte:window onkeydown={onKey}/>
<div class="preview-mode" class:with-history={history} class:show-runs={showRuns}>
  {#if history}
    <aside class="runs" aria-label="Render history">
      <div class="runs-head"><h2>Render history</h2><button onclick={()=>loadRuns()} disabled={loading}>{loading?'Loading…':'Refresh'}</button></div>
      <label class="frame-filter"><span>Minimum rendered frames</span><input type="number" min="0" step="1" value={minimumFrames} onchange={(event)=>setMinimumFrames(event.currentTarget.value)} aria-label="Minimum rendered frames"/><small>{visibleRuns.length} of {runs.length} runs</small></label>
      <div class="run-list">
        {#each visibleRuns as run (run.id)}
          <article class="run-card" class:on={run.id===runId}>
            <button class="run-pick" onclick={(event)=>{pick(run.id,event);showRuns=false;}} title={run.prompt}>
              <div class="thumb">{#if run.last_frame!==null}<img src={`/api/preview/runs/${run.id}/thumbnail?${params}&layer=output&frame=${run.last_frame}`} alt={`Run ${run.id}`} loading="lazy"/>{:else}<span>No frames</span>{/if}</div>
              <div class="meta"><b>{run.label||`Run #${run.id}`}</b><span>{run.frames} frame{run.frames===1?'':'s'} · {when(run.modified)}</span>{#if run.prompt}<em>{run.prompt}</em>{/if}{#if activeJob?.run_id===run.id}<strong>{activeJob.message} · {Math.round(activeJob.progress||0)}%</strong>{/if}</div>
            </button>
            <div class="run-comparison"><button class:on={compareIds[0]===run.id} aria-pressed={compareIds[0]===run.id} aria-label={`Select ${run.label||`run ${run.id}`} as A`} onclick={()=>assignComparison(run.id,0)}>A {compareIds[0]===run.id?'✓':''}</button><button class:on={compareIds[1]===run.id} aria-pressed={compareIds[1]===run.id} aria-label={`Select ${run.label||`run ${run.id}`} as B`} onclick={()=>assignComparison(run.id,1)}>B {compareIds[1]===run.id?'✓':''}</button></div>
          </article>
        {:else}<p class="none">{runs.length?`No runs rendered at least ${minimumFrames} frames.`:`No runs under ${config?.output_dir||'images_out'}/${config?.batch_name||'warpfusion'}`}</p>{/each}
      </div>
      <small class="comparison-help">Compare settings: choose runs A and B. The viewer shows the selected run.</small>
    </aside>
  {/if}
  <section class="viewer" aria-label="Frame viewer">
    {#if history}
      <div class="run-toolbar">
        <button class="mobile-runs" onclick={()=>showRuns=!showRuns} aria-expanded={showRuns}>{showRuns?'Back to viewer':'Choose run'}</button>
        <button class="run-name" onclick={beginRename} disabled={!current} title="Rename run" aria-label={`Rename ${current?.label||`Run #${runId}`}`}>{current?.label|| (runId?`Run #${runId}`:'Render history')}</button>
        <div class="run-actions" role="group" aria-label="Run actions">
          <button onclick={loadSettings} disabled={!current?.has_settings||loadingSettings}>{loadingSettings?'Loading…':'Load settings'}</button>
          <button onclick={()=>runAction('resume')} disabled={!current?.resume_available||!!action||!!activeJob}>{action==='resume'?'Queuing…':'Resume'}</button>
        </div>
      </div>
      {#if renaming}<form class="rename" onsubmit={(event)=>{event.preventDefault();saveLabel();}}><label><span class="visually-hidden">Run display label</span><input bind:this={renameInput} bind:value={draftLabel} maxlength="80" aria-label="Run display label"/></label><button type="submit" disabled={savingLabel}>{savingLabel?'Saving…':'Save'}</button><button type="button" onclick={()=>renaming=false}>Cancel</button></form>{/if}
    {/if}
    {#if activeJob&&history}<div class="live-job" role="status"><span>{activeJob.message||activeJob.stage}</span><strong>{Math.round(activeJob.progress||0)}%</strong><button onclick={cancelActiveJob} disabled={cancelling||activeJob.cancel_requested}>{activeJob.cancel_requested?'Cancelling…':'Cancel render'}</button></div>{/if}
    {#if loaded}<div class="viewer-notice" role="status">{loaded}</div>{/if}
    {#if error}<div class="viewer-notice error" role="alert">{error}</div>{/if}
    {#if sourceError&&!history&&!detail}<div class="viewer-notice error" role="status">{sourceError}</div>{/if}
    {#if showVideo&&current?.video_available}<div class="video-stage"><video controls preload="metadata" src={`/api/preview/runs/${runId}/video?${params}&v=${current.video_modified??current.modified}`}><track kind="captions"/></video></div>
    {:else}<LayerViewer layers={inspectLayers} {frame} bind:view bind:selected bind:focus={focusedLayer} {sourceUrl} {liveUrl}
      sourcePending={!history&&!!sourcePath&&!sourceInfo&&!sourceError} sourceEmpty={!history&&!sourcePath} sourceError={!history?sourceError:''}
      outputPending={!history&&!!activeJob} imageUrl={src} onzoom={(id)=>zoom=id} onchoose={onsource}/>{/if}
    <div class="timeline">
      <div class="timeline-controls">
        <div class="timeline-source">
        {#if current?.video_available}<fieldset class="player-mode" aria-label="Player mode"><label><input type="radio" name="player-mode" checked={!showVideo} onchange={()=>showVideo=false}/><span>View frames</span></label><label><input type="radio" name="player-mode" checked={showVideo} onchange={()=>{playing=false;showVideo=true;}}/><span>Play video</span></label></fieldset>{/if}
        <small class="timeline-meta">{sourceInfo&&!history?`${sourceInfo.fps} fps source`:runId?`Run #${runId}`:'Source preview'}</small>
        </div>
        <div class="stepper"><button onclick={()=>step(-1)} disabled={frame<=minFrame} aria-label="Previous frame">‹</button><button class="loop" onclick={()=>playing=!playing} disabled={!playable} aria-pressed={playing} aria-label={playing?'Pause playback':'Play frames in a loop'} title={`Loop rendered frames at ${PLAYBACK_FPS} fps`}>{playing?'Pause':'Play'}</button><button onclick={()=>step(1)} disabled={frame>=maxFrame} aria-label="Next frame">›</button><input type="number" min={minFrame} max={maxFrame} bind:value={frame} aria-label="Frame"/><span>/ {!history&&!sourceInfo&&!frames.length?'…':maxFrame}</span></div>
        {#if canDownload||current?.video_available||current?.frames}<div class="output-actions">
          <details class="download-menu" use:closeOnSelection><summary aria-disabled={!canDownload&&!current?.video_available}>Download</summary><div>{#if canDownload}<a href={`${src(downloadableLayer)}&download=true`} download>Current frame</a>{/if}{#if current?.video_available}<a href={`/api/preview/runs/${runId}/video?${params}&download=true`} download>Rendered video</a>{/if}{#if !canDownload&&!current?.video_available}<span>No rendered image or video available.</span>{/if}</div></details>
          {#if current?.frames}<button onclick={()=>runAction('video')} disabled={!current?.has_settings||!!action||!!activeJob}>{action==='video'?'Queuing…':current?.video_available?'Rebuild video':'Make video'}</button>{/if}
        </div>{/if}
      </div>
      <input class="scrub" type="range" min={minFrame} max={maxFrame} step="1" bind:value={frame} disabled={maxFrame<=minFrame} aria-label="Frame"/>
      <div class="filmstrip" aria-label="Frame thumbnails">{#each stripFrames as index,indexKey (indexKey)}<button class:at-frame={frame===index} onclick={()=>{playing=false;showVideo=false;frame=index;}} aria-label={`Go to frame ${index}`}><img src={stripImage(index)} alt="" loading="lazy"/></button>{/each}</div>
    </div>
  </section>
</div>
{#if zoom}<ImageZoom src={src(zoom)} title={`${detail?.layers?.find(layer=>layer.id===zoom)?.label||zoom} · frame ${frame}`} onclose={()=>zoom=null}/>{/if}
{#if showDiff}
  <div class="diff-backdrop" role="presentation" onclick={(event)=>{if(event.target===event.currentTarget)showDiff=false}}>
    <div class="diff-dialog" use:trapDialog role="dialog" aria-modal="true" aria-label="Run settings difference">
      <header>
        <div>
          <p>Compare settings</p>
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
              <tr>
                <th>{row.path}</th>
                <td>{#if row.leftSaved === false}<em class="unsaved"
                       title="This run's settings file has no such key — it predates the setting, or never wrote it"
                       >not saved</em>{:else}<code>{formatSettingValue(row.left)}</code>{/if}</td>
                <td>{#if row.rightSaved === false}<em class="unsaved"
                       title="This run's settings file has no such key — it predates the setting, or never wrote it"
                       >not saved</em>{:else}<code>{formatSettingValue(row.right)}</code>{/if}</td>
              </tr>
            {/each}</tbody>
          </table>
        {/if}
      </div>
    </div>
  </div>
{/if}


<style>
  .preview-mode{display:flex;flex:1;min-width:0;min-height:0;height:100%;overflow:hidden;background:#0b1218;color:#e8eef2}.preview-mode.with-history{display:grid;grid-template-columns:228px minmax(0,1fr);grid-template-rows:minmax(0,1fr)}.runs{display:flex;flex-direction:column;min-height:0;min-width:0;background:#111a21;border-right:1px solid #344651}.runs-head{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:17px 14px 12px}.runs-head h2{font-size:14px;font-weight:600}.runs-head button{font-size:11px;padding:5px 7px;min-height:32px}.frame-filter{display:grid;grid-template-columns:minmax(0,1fr) 57px;align-items:center;gap:7px;font-size:11px;padding:10px;border:1px solid #344651;border-radius:6px;margin:0 12px 13px}.frame-filter input{min-height:32px;padding:5px 6px;font-size:12px}.frame-filter small{grid-column:1/-1;font-size:11px}.run-list{display:flex;flex-direction:column;gap:11px;flex:1;min-height:0;overflow:auto;padding:0 12px 12px}.run-card{border:1px solid #3c5160;border-radius:6px;overflow:hidden;flex-shrink:0;background:#18242d}.run-card.on{border-color:#c4dc6a}.run-pick{display:flex;flex-direction:column;width:100%;background:transparent;border:0;border-radius:0;padding:0;text-align:left;color:#e8eef2;cursor:pointer}.run-pick:hover{background:#22333e}.thumb{position:relative;flex:0 0 105px;min-height:0;overflow:hidden;height:105px;background:#080e13;display:grid;place-items:center;width:100%}.thumb img{position:absolute;inset:0;display:block;width:100%;height:100%;object-fit:cover;object-position:center 45%}.thumb span{font-size:12px;color:#a6b8c6}.meta{display:flex;flex-direction:column;gap:5px;padding:10px 11px}.meta b{font-size:13px;font-weight:500}.meta span{font-size:11px;color:#adbfcc}.meta em{font-size:11px;font-style:normal;color:#b3c3cf;display:-webkit-box;-webkit-line-clamp:2;line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}.meta strong{font-size:11px;font-weight:500;color:#d8ff55}.run-comparison{display:flex;gap:6px;padding:0 10px 10px}.run-comparison button{flex:1;font-size:12px;padding:4px 8px;min-height:30px}.run-comparison button.on{background:#2d3b22;color:#e1f3ad;border-color:#adcb55}.comparison-help{font-size:11px;padding:10px 14px;border-top:1px solid #334652}.none{padding:10px;color:#afc1cd;font-size:12px;line-height:1.6}.viewer{display:flex;flex-direction:column;flex:1;min-width:0;min-height:0;height:100%;overflow:hidden}.run-toolbar{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;border-bottom:1px solid #334652;background:#15222a;padding:8px 16px;min-height:48px;flex-shrink:0}.run-name{flex:1;min-width:0;text-align:left;border:0;background:transparent;padding:6px 0;color:inherit;font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.run-actions{display:flex;align-items:center;flex-wrap:wrap;gap:8px;max-width:100%}.download-menu{position:relative;flex-shrink:0}.download-menu summary{padding:7px 11px;border:1px solid #4b6373;background:#1c2a34;border-radius:5px;font-size:12px;color:#dfe9ee;cursor:pointer;min-height:32px;display:flex;align-items:center;justify-content:center}.download-menu>div{position:absolute;z-index:20;display:flex;flex-direction:column;gap:4px;right:0;background:#1b2a34;border:1px solid #526c7d;border-radius:6px;padding:7px;min-width:175px;box-shadow:0 12px 30px #0008}.download-menu>div{bottom:40px}.run-actions button{min-height:36px;font-size:12px;padding:7px 10px}.download-menu a{padding:8px;color:#e0eecd;text-decoration:none;font-size:12px;border-radius:4px}.download-menu a:hover{background:#324757}.download-menu span{padding:8px;color:#b3c7d4;font-size:12px;line-height:1.5}.download-menu summary[aria-disabled=true]{color:#9caebc}.mobile-runs{display:none}.rename{display:flex;gap:8px;padding:9px 16px;border-bottom:1px solid #3b5261}.rename label{flex:1}.rename input{padding:7px 10px;font-size:13px;min-height:36px}.rename button{font-size:12px;min-height:36px}.live-job{display:flex;align-items:center;gap:12px;padding:9px 16px;background:#24311e;color:#e3edcd;font-size:12px;flex-shrink:0}.live-job>span{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.live-job strong{font:12px Consolas,monospace}.live-job button{font-size:11px;min-height:32px;color:#ffc3cb;border-color:#a15b6a;background:#35232d}.viewer-notice{padding:9px 16px;font-size:12px;background:#213223;color:#e0ebcf;flex-shrink:0}.viewer-notice.error{background:#33212a;color:#ffc4cc}.timeline{container-type:inline-size;padding:11px 16px 13px;border-top:1px solid #344955;background:#14212a;flex:0 0 auto}.timeline-controls{display:grid;grid-template-columns:minmax(0,1fr) auto minmax(0,1fr);align-items:center;gap:9px;min-height:34px}.timeline-source{grid-column:1;display:flex;align-items:center;flex-wrap:wrap;gap:8px;min-width:0}.stepper{grid-column:2;justify-self:center;display:flex;align-items:center;gap:5px;font-size:12px;color:#c2d2dd}.stepper input{width:65px;padding:6px 7px;text-align:center;min-height:32px;font:12px Consolas,monospace}.stepper button{min-height:32px;padding:5px 10px;font-size:12px}.timeline-meta{font-size:11px;white-space:nowrap}.output-actions{grid-column:3;justify-self:end;display:flex;align-items:center;gap:8px}.output-actions>button{min-height:32px;padding:5px 9px;font-size:12px}.scrub{display:block;width:100%;accent-color:#d8ff55;padding:0;height:15px;min-height:15px;background:transparent;border:0;margin:7px 0}.filmstrip{display:flex;gap:3px;height:36px;overflow:hidden}.filmstrip button{padding:0;flex:1;min-width:0;border:1px solid transparent;min-height:0;border-radius:2px;overflow:hidden;background:#0a1116}.filmstrip button.at-frame{border-color:#d8ff55}.filmstrip img{height:100%;width:100%;object-fit:cover;display:block}.video-stage{flex:1;min-height:0;display:flex;flex-direction:column;padding:16px;gap:10px}.video-stage video{width:100%;height:100%;min-height:0;flex:1;object-fit:contain;background:#060b0f}.player-mode{display:flex;gap:2px;margin:0;padding:3px;border:1px solid #4b6373;border-radius:6px;min-width:0}.player-mode label{position:relative;cursor:pointer}.player-mode input{position:absolute;opacity:0;width:1px;height:1px;min-height:0}.player-mode span{display:flex;align-items:center;justify-content:center;min-height:32px;padding:5px 10px;border-radius:3px;font-size:12px;color:#c2d2dd}.player-mode input:checked+span{background:#d8ff55;color:#142000}.player-mode input:focus-visible+span{outline:2px solid #e8eef2;outline-offset:2px}
  .diff-backdrop{position:fixed;inset:0;z-index:50;display:grid;place-items:center;padding:24px;background:#02080de8}.diff-dialog{width:min(1180px,96vw);height:min(780px,90dvh);display:flex;flex-direction:column;overflow:hidden;border:1px solid #526e80;border-radius:8px;background:#111c24;box-shadow:0 24px 80px #000c}.diff-dialog>header{display:flex;align-items:center;justify-content:space-between;gap:20px;padding:16px 20px;border-bottom:1px solid #3d5667;background:#192a35}.diff-dialog header p{margin:0 0 5px;color:#b1c4d1;font-size:12px}.diff-dialog h2{margin:0;font-size:15px;font-weight:500}.diff-dialog h2 span,.diff-dialog thead i{display:inline-grid;place-items:center;width:22px;height:22px;margin-right:5px;border-radius:4px;background:#d8ff55;color:#162000;font:12px Consolas,monospace;font-style:normal}.diff-dialog h2 b{margin:0 9px;color:#afc4d2;font-size:12px;font-weight:400}.diff-dialog>header>button{min-height:36px;min-width:36px;font-size:20px;padding:4px 10px}.diff-body{flex:1;min-height:0;overflow:auto}.diff-state{height:100%;display:grid;place-items:center;color:#bfd0dc;font-size:13px;padding:24px}.diff-state.error{color:#ffbdc8}.diff-dialog table{width:100%;border-collapse:collapse;table-layout:fixed}.diff-dialog thead{position:sticky;top:0;z-index:1;background:#1c303d}.diff-dialog th,.diff-dialog td{padding:11px 13px;border-bottom:1px solid #334b5c;text-align:left;vertical-align:top;overflow-wrap:anywhere}.diff-dialog thead th{font-size:12px;font-weight:500;color:#d9e7ef}.diff-dialog th:first-child{width:28%}.diff-dialog tbody th{color:#d3e9a8;font:12px/1.5 Consolas,monospace}.diff-dialog td{width:36%;background:#101d26}.diff-dialog td code{display:block;font:12px/1.6 Consolas,monospace;color:#e1ebf1;white-space:pre-wrap;overflow-wrap:anywhere}.unsaved{font-size:12px;color:#b0c3d1;font-style:italic}
  @media(max-width:700px){.preview-mode.with-history{display:flex}.runs{display:none;width:100%}.preview-mode.show-runs .runs{display:flex}.preview-mode.show-runs .viewer{display:none}.mobile-runs{display:block;min-height:36px;font-size:12px}.run-list{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));align-content:start}.runs-head h2{font-size:16px}.run-toolbar{padding:7px 12px}.run-name{flex:1;min-width:0;text-align:left;border:0;background:transparent;padding:6px 0;color:inherit;font-size:12px}.run-actions{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));width:100%}.run-actions button{min-height:40px}.timeline{container-type:inline-size;padding:10px 12px 12px}.timeline-controls{gap:8px}.timeline-meta{font-size:10px}.stepper{gap:4px}.stepper input{width:55px}.stepper button{min-height:36px;padding:5px 9px}.output-actions{flex-basis:100%;justify-content:flex-end}.output-actions>button,.download-menu summary{min-height:36px;font-size:12px}.filmstrip{height:32px}.diff-backdrop{padding:8px}.diff-dialog>header{padding:12px;gap:9px}.diff-dialog h2{font-size:12px}.diff-dialog th,.diff-dialog td{padding:8px 7px;font-size:11px}.diff-dialog td code{font-size:11px}.diff-dialog thead th{font-size:11px}.diff-dialog h2 b{margin:0 3px}.diff-dialog h2 span{width:18px;height:18px}.rename{padding:8px 12px;gap:6px}.comparison-help{font-size:12px}}
  @container(max-width:740px){.timeline-controls{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}.timeline-source{grid-column:1;grid-row:1}.output-actions{grid-column:2;grid-row:1}.stepper{grid-column:1/-1;grid-row:2}}
  @container(max-width:480px){.timeline-controls{grid-template-columns:minmax(0,1fr)}.timeline-source{grid-column:1;grid-row:1}.output-actions{grid-column:1;grid-row:3}.stepper{grid-column:1;grid-row:2}}
  @media(pointer:coarse){.stepper button,.output-actions>button,.download-menu summary,.run-comparison button,.run-actions button,.run-name,.player-mode span{min-height:44px}}
</style>
