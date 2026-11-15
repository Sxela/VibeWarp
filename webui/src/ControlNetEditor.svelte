<script>
  import HelpTip from './HelpTip.svelte';
  import { SvelteSet } from 'svelte/reactivity';
  import Field from './Field.svelte';
  import { stripQuotes } from './paths.js';
  // `schema` is the ControlNetConfig schema — used to render the global detector
  // settings without redeclaring their choices here.
  let { value, onchange, schema, modelVersion = '' } = $props();

  const entryDefaults = {path:'',weight:1,start:0,end:1,annotator:'',source:'',detect_resolution:-1,layer_weights:null,mode:'balanced',zero_uncond:false};
  // Engine-accepted values (core/diffusion.py): '' defers to the global
  // conditioning source; anything else must be init/stylized or a real path.
  const sources = [['','Global (use conditioning source)'],['init','Init — raw video frame'],['stylized','Stylized — warped previous render']];

  let catalog = $state({nets:[],files:[],mode_presets:{}});
  let picker = $state('');
  let loading = $state(false);
  // Collapsed by default so a full stack of nets fits on one screen; weight (the
  // setting people actually tune) stays visible in the header either way.
  let expanded = $state(new SvelteSet());

  function toggleOpen(key){
    if (expanded.has(key)) expanded.delete(key);
    else expanded.add(key);
  }

  // Refetch whenever the model directory or base model changes: both alter which
  // nets are offered and which checkpoints are found on disk.
  $effect(() => {
    let params = new URLSearchParams({model_version:modelVersion||'', model_dir:value.model_dir||''});
    let cancelled = false;
    loading = true;
    fetch(`/api/controlnet/catalog?${params}`)
      .then(r => r.json())
      .then(d => { if (!cancelled) catalog = d; })
      .catch(() => {})
      .finally(() => { if (!cancelled) loading = false; });
    return () => { cancelled = true; };
  });

  let specs = $derived(new Map(catalog.nets.map(net => [net.key, net])));
  let active = $derived(Object.keys(value.models ?? {}).map(key => ({key, entry:value.models[key], net:specs.get(key)})));
  let available = $derived(catalog.nets.filter(net => !(net.key in (value.models ?? {}))));

  function patch(changes){ onchange({...value, ...changes}); }
  function entryPatch(key, changes){ patch({models:{...value.models, [key]:{...value.models[key], ...changes}}}); }

  function add(){
    if (!picker) return;
    patch({models:{...value.models, [picker]:{...entryDefaults}}, enabled:true});
    expanded.add(picker);   // you just added it — open it to configure
    picker = '';
  }
  function remove(key){
    let models = {...value.models};
    delete models[key];
    expanded.delete(key);
    patch({models, enabled:Object.keys(models).length > 0 && value.enabled});
  }

  // Layer weights and zero_uncond are DERIVED from mode (notebook: the
  // controlnet_multimodel_inferred block) — only 'custom' lets you set them.
  function weightsFor(entry){
    let preset = catalog.mode_presets?.[entry.mode];
    return preset ? preset.layer_weights : (entry.layer_weights ?? Array(13).fill(1));
  }
  function zeroUncondFor(entry){
    let preset = catalog.mode_presets?.[entry.mode];
    return preset ? preset.zero_uncond : Boolean(entry.zero_uncond);
  }
  function setMode(key, entry, mode){
    // Entering custom: seed the editable weights from what the old mode produced,
    // so the curve doesn't jump when you switch.
    if (mode === 'custom') entryPatch(key, {mode, layer_weights:weightsFor(entry), zero_uncond:zeroUncondFor(entry)});
    else entryPatch(key, {mode, layer_weights:null});
  }
  function setLayerWeight(key, entry, index, raw){
    let weights = [...weightsFor(entry)];
    weights[index] = Number(raw);
    entryPatch(key, {layer_weights:weights});
  }

  // Checkpoint status. '' means "infer from the model directory", which only
  // works if the expected filename is actually there.
  function status(key, entry){
    let net = specs.get(key);
    // The catalog is filtered by model_version, so a net that isn't in it belongs
    // to another base model — switching SD1.5 <-> SDXL with nets configured leaves
    // these behind, and they would crash the render with a shape mismatch.
    if (!net) return {kind:'error', text:'Wrong base model — remove'};
    if (entry.path) return catalog.files.includes(entry.path) ? {kind:'ok', text:'Found'} : {kind:'warn', text:'Not in model directory'};
    if (net.resolved_path) return {kind:'ok', text:`Auto: ${net.filename}`};
    if (net.filename) return {kind:'error', text:`Missing: ${net.filename}`};
    return {kind:'error', text:'No checkpoint — pick a file'};
  }
  const basename = (path) => path.split(/[\\/]/).pop();
  const number = (e) => Number(e.target.value);
</script>

<div class="base grid">
  <label class="toggle"><input type="checkbox" checked={value.enabled} onchange={(e)=>patch({enabled:e.target.checked})}/><span>Enable ControlNet processing</span></label>
  <label class="wide"><span>Model directory <HelpTip label="Help: Model directory" text="Scanned for checkpoints."/></span><input value={value.model_dir} placeholder="models/ControlNet" onchange={(e)=>patch({model_dir:stripQuotes(e.target.value)})}/></label>
  <label><span>Loading mode</span><select value={value.mode} onchange={(e)=>patch({mode:e.target.value})}><option>internal</option><option>external</option></select></label>
  <label><span>Global conditioning source</span><select value={value.cond_image_src} onchange={(e)=>patch({cond_image_src:e.target.value})}><option>init</option><option>stylized</option><option>cond_video</option></select></label>
  <label class="toggle"><input type="checkbox" checked={value.normalize_weights} onchange={(e)=>patch({normalize_weights:e.target.checked})}/><span>Normalize combined weights</span></label>
</div>

<div class="adder">
  <label><span>Add ControlNet</span>
    <select bind:value={picker} disabled={!available.length}>
      <option value="">{loading ? 'Loading…' : available.length ? 'Select a ControlNet…' : 'All available nets added'}</option>
      {#each available as net}<option value={net.key}>{net.label} — {net.key}{net.resolved_path ? '' : ' (no checkpoint found)'}</option>{/each}
    </select>
  </label>
  <button class="add" onclick={add} disabled={!picker}>Add</button>
  <p class="hint">{catalog.files.length} checkpoint{catalog.files.length===1?'':'s'} in the model directory{catalog.nets.length?` · ${catalog.nets.length} nets available for ${modelVersion||'all models'}`:''}</p>
</div>

{#if !active.length}
  <div class="none">No ControlNets configured. Add one above.</div>
{/if}

<div class="model-list">
{#each active as {key, entry, net} (key)}
  {@const state = status(key, entry)}
  {@const open = expanded.has(key)}
  <article class:stale={!net} class:open>
    <div class="model-head">
      <button class="disclose" onclick={()=>toggleOpen(key)} aria-expanded={open}
              aria-label={`${open ? 'Collapse' : 'Expand'} ${net?.label ?? key} settings`}>
        <i class:open></i>
        <span class="title"><b>{net?.label ?? key}</b><code>{key}</code></span>
      </button>
      <div class="head-right">
        <!-- Weight is the setting people actually tune, so it lives in the header. -->
        <label class="weight" title="Weight">
          <span>w</span>
          <input type="number" step="0.05" value={entry.weight}
                 onchange={(e)=>entryPatch(key,{weight:number(e)})}/>
        </label>
        <span class="badge {state.kind}">{state.text}</span>
        <button class="remove" onclick={()=>remove(key)} aria-label="Remove {key}">✕</button>
      </div>
    </div>

    {#if open}
    <div class="grid settings">
      <label class="wide"><span>Checkpoint</span>
        <select value={catalog.files.includes(entry.path) ? entry.path : (entry.path ? '__custom__' : '')}
                onchange={(e)=>entryPatch(key, {path: e.target.value === '__custom__' ? (entry.path || ' ') : e.target.value})}>
          <option value="">Auto — infer from model directory{net?.filename ? ` (${net.filename})` : ''}</option>
          {#each catalog.files as file}<option value={file}>{basename(file)}</option>{/each}
          <option value="__custom__">Custom path…</option>
        </select>
      </label>
      {#if entry.path && !catalog.files.includes(entry.path)}
        <label class="wide"><span>Custom checkpoint path</span><input value={entry.path.trim()} placeholder="C:\models\ControlNet\my_net.safetensors" onchange={(e)=>entryPatch(key, {path:stripQuotes(e.target.value)})}/></label>
      {/if}

      <label><span>Detect resolution <HelpTip label="Help: Detect resolution" text="−1 uses the global detection resolution."/></span><input type="number" step="8" value={entry.detect_resolution} onchange={(e)=>entryPatch(key,{detect_resolution:number(e)})}/></label>
      <label><span>Start <HelpTip label="Help: Step range" text="Fraction of sampling steps, from 0 to 1."/></span><input type="number" min="0" max="1" step="0.05" value={entry.start} onchange={(e)=>entryPatch(key,{start:number(e)})}/></label>
      <label><span>End <HelpTip label="Help: Step range" text="Fraction of sampling steps, from 0 to 1."/></span><input type="number" min="0" max="1" step="0.05" value={entry.end} onchange={(e)=>entryPatch(key,{end:number(e)})}/></label>
      <label><span>Source</span>
        <select value={sources.some(([id])=>id===entry.source) ? entry.source : '__path__'}
                onchange={(e)=>entryPatch(key,{source: e.target.value === '__path__' ? (entry.source || ' ') : e.target.value})}>
          {#each sources as [id,label]}<option value={id}>{label}</option>{/each}
          <option value="__path__">Custom frame folder / file…</option>
        </select>
      </label>
      <label><span>Annotator override</span><input value={entry.annotator} placeholder={net?.annotator ? `Auto (${net.annotator})` : 'Auto'} onchange={(e)=>entryPatch(key,{annotator:e.target.value})}/></label>
      {#if entry.source && !sources.some(([id])=>id===entry.source)}
        <label class="wide"><span>Custom source path</span><input value={entry.source.trim()} placeholder="C:\frames\depth" onchange={(e)=>entryPatch(key,{source:stripQuotes(e.target.value)})}/></label>
      {/if}
    </div>

    <div class="weighting">
      <div class="weighting-head">
        <label><span>Mode</span>
          <select value={entry.mode} onchange={(e)=>setMode(key, entry, e.target.value)}>
            <option value="balanced">Balanced</option>
            <option value="controlnet">ControlNet is more important</option>
            <option value="prompt">Prompt is more important</option>
            <option value="custom">Custom layer weights</option>
          </select>
        </label>
        <div class="derived">
          <span>Zero unconditional</span>
          {#if entry.mode === 'custom'}
            <label class="toggle bare"><input type="checkbox" aria-label="Zero unconditional" checked={entry.zero_uncond} onchange={(e)=>entryPatch(key,{zero_uncond:e.target.checked})}/></label>
          {:else}
            <b>{zeroUncondFor(entry) ? 'on' : 'off'}</b><HelpTip label="Help: Zero unconditional" text="Set by the loading mode. Choose Custom layer weights to edit this value."/>
          {/if}
        </div>
      </div>

      <div class="layers">
        <span class="layers-label">Layer weights <HelpTip label="Help: Layer weights" text={entry.mode === 'custom' ? '13 UNet blocks — editable' : 'Derived from the loading mode. Choose Custom layer weights to edit individual blocks.'}/></span>
        <div class="bars">
          {#each weightsFor(entry) as weight, index}
            <div class="bar" title={`Layer ${index}: ${weight.toFixed(3)}`}>
              <i style={`height:${Math.max(2, Math.min(1, weight) * 100)}%`}></i>
              {#if entry.mode === 'custom'}
                <input type="number" aria-label={`Layer ${index} weight`} min="0" max="1" step="0.05" value={Number(weight.toFixed(3))} onchange={(e)=>setLayerWeight(key, entry, index, e.target.value)}/>
              {/if}
            </div>
          {/each}
        </div>
      </div>
    </div>

    {#if net?.detectors?.length}
      <div class="detectors">
        <p>Detector settings <HelpTip label="Help: Detector settings" text={`Global — shared by every net using the ${net.annotator} annotator.`}/></p>
        <div class="grid">
          {#each net.detectors as name}
            <Field {name} schema={schema.properties[name]} value={value[name]} onchange={(v)=>patch({[name]:v})}/>
          {/each}
        </div>
      </div>
    {/if}
    {/if}
  </article>
{/each}
</div>

<style>
  .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
  .base{padding-bottom:22px;border-bottom:1px solid #292e36}
  .wide{grid-column:1/-1}
  .toggle{flex-direction:row;align-items:center;gap:10px;padding-top:22px}
  .toggle.bare{padding:0}
  /* Do NOT set width/height on .toggle input here. style.css styles it as a
     36x20 pill with an :after knob; a scoped override outranks that and
     collapses the track into a bare circle. */

  .adder{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:12px;align-items:end;margin-top:22px}
  .adder .hint{grid-column:1/-1;margin:0;color:var(--muted);font-size:var(--text-help)}
  .add{border:1px solid #4d5d2a;background:#d8ff55;color:#111;font-weight:700;padding:10px 20px;border-radius:var(--radius);cursor:pointer;height:38px}
  .add:disabled{background:#191c21;color:#5c636c;border-color:#303640;cursor:default}
  .none{margin-top:18px;padding:26px;border:1px dashed #30353d;border-radius:var(--radius);color:var(--muted);font-size:var(--text-label);text-align:center}

  .model-list{display:grid;gap:12px;margin-top:18px}
  article{border:1px solid #292e36;border-radius:var(--radius);background:#0e1115;overflow:hidden}
  article.stale{border-color:#67363a}
  .model-head{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:9px 12px 9px 8px;background:#12161b}
  .disclose{display:flex;align-items:center;gap:9px;min-width:0;flex:1;border:0;background:none;padding:5px 4px;cursor:pointer;text-align:left}
  .disclose i{flex-shrink:0;width:0;height:0;border-left:5px solid #6d7580;border-top:4px solid transparent;border-bottom:4px solid transparent;transition:transform .12s}
  .disclose i.open{transform:rotate(90deg)}
  .disclose:hover i{border-left-color:#d8ff55}
  .title{display:flex;align-items:baseline;gap:10px;min-width:0}
  .title b{color:#e7eaee;font-size:14px}
  .title code,code{font:var(--text-help) var(--font-ui);color:var(--muted)}
  .head-right{display:flex;align-items:center;gap:10px;flex-shrink:0}
  /* Weight lives in the header: it is the one setting people constantly tune. */
  .weight{flex-direction:row;align-items:center;gap:5px;padding:3px 8px 3px 9px;border:1px solid #343a43;border-radius:20px;background:#0c0f13}
  .weight span{color:var(--accent-muted);font:600 var(--text-help) var(--font-ui)}
  .weight input{width:46px;border:0;background:transparent;padding:2px 0;text-align:center;font:var(--text-value) var(--font-mono);-moz-appearance:textfield}
  .weight input::-webkit-outer-spin-button,.weight input::-webkit-inner-spin-button{-webkit-appearance:none;margin:0}
  .weight:focus-within{border-color:var(--accent-muted)}
  .badge{font-size:var(--text-help);padding:3px 8px;border-radius:20px;border:1px solid}
  .badge.ok{color:#b6d96a;border-color:#4d5d2a;background:#1a2013}
  .badge.warn{color:var(--warning);border-color:#6b5726;background:#211c11}
  .badge.error{color:var(--error);border-color:#67363a;background:#241416}
  .remove{border:1px solid #303640;background:#171a1f;color:var(--muted);width:26px;height:26px;border-radius:var(--radius);cursor:pointer;line-height:1}
  .remove:hover{border-color:#67363a;color:var(--error)}

  .settings{padding:16px;border-top:1px solid #292e36}
  .weighting{padding:16px;border-top:1px solid #292e36;background:#0c0f13}
  .weighting-head{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:16px;align-items:end}
  .derived{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:var(--text-label);padding-bottom:10px}
  .derived b{color:#e7eaee}
  .layers{margin-top:16px}
  .layers-label{display:block;color:var(--muted);font-size:var(--text-label);margin-bottom:8px}
  .bars{display:grid;grid-template-columns:repeat(13,minmax(0,1fr));gap:4px;align-items:end}
  .bar{display:flex;flex-direction:column;justify-content:flex-end;gap:4px;height:64px}
  .bar i{display:block;width:100%;background:linear-gradient(#d8ff55,#8ea834);border-radius:3px 3px 0 0;min-height:2px}
  .bar input{padding:4px 2px;text-align:center;font-size:var(--text-help)}

  .detectors{padding:14px 16px;background:#151a14;border-top:1px solid #33401f}
  .detectors p{margin:0 0 12px;color:#c8d3ad;font-size:var(--text-label)}

  input,select{width:100%;border:1px solid var(--input-border);background:var(--input-bg);color:var(--ink);border-radius:var(--radius);padding:10px 11px;font-size:var(--text-value)}
  input:focus,select:focus{border-color:var(--accent-muted)}
  select:disabled{color:#5c636c}
  label{display:flex;flex-direction:column;gap:7px;color:var(--muted);font-size:var(--text-label)}

  @media(max-width:700px){
    .grid,.weighting-head{grid-template-columns:1fr}
    .wide{grid-column:1}
    .adder{grid-template-columns:1fr}
    .title code{display:none}
    .bars{grid-template-columns:repeat(7,minmax(0,1fr))}
  }
</style>
