<script>
  import { stripQuotes } from './paths.js';

  let {
    value, onchange, schema, videoPath = '', frameRange = [0, 0],
    extractNth = 1, modelVersion = '', modelDir = '',
  } = $props();

  const defaults = {
    model_key:'', path:'', weight:1, start:0, end:1,
    source_image:{source:'none', image_path:''},
    source_images:[],
    weight_type:'linear', combine_embeds:'concat', embeds_scaling:'V only',
  };
  const sources = [
    ['none', 'Off'],
    ['previous', 'Previous stylized frame'],
    ['warped', 'Previous stylized + warp + consistency'],
    ['upload', 'Upload / fixed image'],
  ];
  let entrySchema = $derived(
    schema?.properties?.models?.additional?.properties ?? {});
  const choices = (name, fallback) => entrySchema?.[name]?.choices ?? fallback;
  let catalog = $state({adapters:[], files:[]});
  let picker = $state('');
  let loading = $state(false);
  let uploadError = $state({});
  let uploading = $state('');
  let dragOver = $state('');
  let inputs = $state({});

  let models = $derived(value?.models ?? {});
  let specs = $derived(new Map(catalog.adapters.map(adapter => [adapter.key, adapter])));
  const entryModelKey = (key, entry) => entry?.model_key || key.replace(/__\d+$/, '');
  let active = $derived(Object.entries(models).map(
    ([key, entry]) => ({key, entry, adapter:specs.get(entryModelKey(key, entry))})));
  let available = $derived(catalog.adapters);
  let sourceFrame = $derived(
    Math.max(0, Number(frameRange?.[0] ?? 0))
      * Math.max(1, Number(extractNth || 1)));
  let videoThumb = $derived(videoPath
    ? `/api/video/thumbnail?path=${encodeURIComponent(videoPath)}&frame=${sourceFrame}`
    : '');

  $effect(() => {
    let params = new URLSearchParams({
      model_version:modelVersion || '', model_dir:modelDir || '',
    });
    let cancelled = false;
    loading = true;
    fetch(`/api/ipadapter/catalog?${params}`)
      .then(response => response.json())
      .then(data => { if (!cancelled) catalog = data; })
      .catch(() => {})
      .finally(() => { if (!cancelled) loading = false; });
    return () => { cancelled = true; };
  });

  function patch(changes){ onchange({...value, ...changes}); }
  function entryPatch(key, changes){
    patch({models:{...models, [key]:{...models[key], ...changes}}});
  }
  function normalizeRef(source){
    if (source && typeof source === 'object' && !Array.isArray(source))
      return {source:source.source || 'none', image_path:source.image_path || ''};
    if (source === 'stylized' || source === 'warped')
      return {source:'warped', image_path:''};
    if (source === 'previous' || source === 'prev_frame')
      return {source:'previous', image_path:''};
    if (!source || ['none','off','raw_frame','init'].includes(source))
      return {source:'none', image_path:''};
    return {source:'upload', image_path:String(source)};
  }
  function refs(entry){
    let list = Array.isArray(entry?.source_images) && entry.source_images.length
      ? entry.source_images : [entry?.source_image];
    return list.map(normalizeRef);
  }
  function setRefs(key, next){
    entryPatch(key, {source_images:next, source_image:next[0]});
  }
  function add(){
    if (!picker) return;
    let adapter = specs.get(picker);
    let instanceKey = picker;
    let suffix = 2;
    while (models[instanceKey]) instanceKey = `${picker}__${suffix++}`;
    patch({models:{...models, [instanceKey]:{
      ...structuredClone(defaults), model_key:picker,
      path:adapter?.resolved_path || '',
    }}, enabled:true});
    picker = '';
  }
  function remove(key){
    let next = {...models};
    delete next[key];
    patch({models:next, enabled:Object.keys(next).length > 0 && value.enabled});
  }
  const slotId = (key, index) => `${key}:${index}`;
  function selectSource(key, index, mode){
    let slot = slotId(key, index);
    if (mode === 'upload') {
      inputs[slot]?.click();
      return;
    }
    let next = refs(models[key]);
    next[index] = {source:mode, image_path:''};
    setRefs(key, next);
  }
  async function upload(key, index, file){
    if (!file) return;
    let slot = slotId(key, index);
    uploading = slot;
    uploadError = {...uploadError, [slot]:''};
    try {
      let response = await fetch(
        `/api/references/upload?filename=${encodeURIComponent(file.name)}`,
        {method:'POST', headers:{'content-type':file.type || 'application/octet-stream'},
         body:file});
      let data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Could not upload image');
      let next = refs(models[key]);
      next[index] = {source:'upload', image_path:data.path};
      setRefs(key, next);
    } catch (error) {
      uploadError = {...uploadError, [slot]:error.message || 'Could not upload image'};
    } finally {
      uploading = '';
    }
  }
  function drop(key, index, event){
    event.preventDefault();
    dragOver = '';
    upload(key, index, event.dataTransfer?.files?.[0]);
  }
  function addImage(key){
    setRefs(key, [...refs(models[key]), {source:'none', image_path:''}]);
  }
  function removeImage(key, index){
    let next = refs(models[key]).filter((_, itemIndex) => itemIndex !== index);
    setRefs(key, next.length ? next : [{source:'none', image_path:''}]);
  }
  const combineHints = {
    concat:'Keep every image as a longer ordered token sequence.',
    add:'Sum all image embeddings.',
    subtract:'Image 1 minus the average of Images 2…N.',
    average:'Average all image embeddings; lowest memory.',
    'norm average':'Normalize each embedding, then average them.',
  };
  function status(adapter, entry){
    if (!adapter) return {kind:'error', text:'Wrong base model — remove'};
    if (entry.path)
      return catalog.files.includes(entry.path)
        ? {kind:'ok', text:'Found'}
        : {kind:'warn', text:'Custom checkpoint'};
    if (adapter.resolved_path) return {kind:'ok', text:`Found: ${adapter.filename}`};
    return {kind:'error', text:`Missing: ${adapter.filename}`};
  }
  const basename = (path) => path.split(/[\\/]/).pop();
  const number = (event) => Number(event.currentTarget.value);
</script>

<div class="base">
  <label class="toggle"><input type="checkbox" checked={value.enabled}
    onchange={(event)=>patch({enabled:event.currentTarget.checked})}/>
    <span>Enable IP-Adapter processing</span></label>
  <label class="toggle"><input type="checkbox" checked={value.flip_uc}
    onchange={(event)=>patch({flip_uc:event.currentTarget.checked})}/>
    <span>Flip unconditional mask</span></label>
</div>

<div class="adder">
  <label><span>Add IP-Adapter</span>
    <select bind:value={picker} disabled={!available.length}>
      <option value="">{loading ? 'Loading…' : available.length
        ? 'Select an IP-Adapter…' : 'All compatible adapters added'}</option>
      {#each available as adapter}
        <option value={adapter.key}>{adapter.label} — {adapter.clip_variant}
          {adapter.resolved_path ? '' : ' (no checkpoint found)'}</option>
      {/each}
    </select>
  </label>
  <button class="add" onclick={add} disabled={!picker}>Add</button>
  <p class="hint">Adapters may be added repeatedly with a different image and weight ·
    {catalog.files.length} checkpoint{catalog.files.length===1?'':'s'}
    in the ControlNet model directory{catalog.adapters.length
      ? ` · ${catalog.adapters.length} compatible with ${modelVersion||'this model'}` : ''}</p>
</div>

{#if !active.length}
  <div class="none">No IP-Adapters configured. Add one above.</div>
{/if}

<div class="adapter-list">
{#each active as {key, entry, adapter} (key)}
  {@const references = refs(entry)}
  {@const state = status(adapter, entry)}
  <article>
    <header>
      <div><b>{adapter?.label ?? key}</b><span>{key}{adapter ? ` · ${adapter.clip_variant}` : ''}</span></div>
      <label class="weight"><span>w</span><input type="number" step="0.05"
        value={entry.weight} onchange={(event)=>entryPatch(key,{weight:number(event)})}/></label>
      <span class="badge {state.kind}">{state.text}</span>
      <button class="remove" onclick={()=>remove(key)} aria-label={`Remove ${key}`}>✕</button>
    </header>

    <div class="content">
      <section class="source-stack">
        <div class="source-title">
          <b>Adapter images</b>
          <span>{references.length} image{references.length===1?'':'s'} · shared weight</span>
        </div>
        <div class="source-grid">
          {#each references as reference, index}
            {@const slot = slotId(key,index)}
            <div class="source-card">
              <div class="image-head">
                <b>Image {index+1}</b>
                {#if references.length > 1}
                  <button class="remove-image" onclick={()=>removeImage(key,index)}
                    aria-label={`Remove image ${index+1} from ${key}`}>Remove</button>
                {/if}
              </div>
              {#if reference.source === 'upload'}
                <button class="preview drop" class:over={dragOver===slot}
                  ondragover={(event)=>{event.preventDefault();dragOver=slot}}
                  ondragleave={()=>dragOver=''} ondrop={(event)=>drop(key,index,event)}
                  onclick={()=>inputs[slot]?.click()}
                  aria-label={`Upload image ${index+1} for ${key}`}>
                  {#if reference.image_path}
                    <img src={`/api/references/thumbnail?path=${encodeURIComponent(reference.image_path)}`}
                      alt={`Fixed reference ${index+1} for ${key}`}/><span>Replace image</span>
                  {:else}
                    <i>+</i><span>{uploading===slot?'Uploading…':'Drop image or click'}</span>
                  {/if}
                </button>
              {:else}
                <div class="preview" class:empty={!videoThumb || reference.source==='none'}>
                  {#if reference.source === 'none'}
                    <i>+</i><span>No image is sent</span>
                  {:else if videoThumb}
                    <img src={videoThumb} alt={`First range frame preview for ${key}`}/>
                    <span>Off on first range frame</span>
                  {:else}
                    <span>Choose an input video to preview</span>
                  {/if}
                </div>
              {/if}
              <input class="picker" type="file" accept="image/*" bind:this={inputs[slot]}
                onchange={(event)=>upload(key,index,event.currentTarget.files?.[0])}/>
              {#if uploadError[slot]}<small class="err">{uploadError[slot]}</small>{/if}
              <label><span>Image source</span>
                <select value={reference.source}
                  onchange={(event)=>selectSource(key,index,event.currentTarget.value)}>
                  {#each sources as [id,label]}<option value={id}>{label}</option>{/each}
                </select>
              </label>
            </div>
          {/each}
        </div>
        <button class="add-image" onclick={()=>addImage(key)}>+ Add image to this adapter</button>
        <p>Temporal sources are not sent on frame {Number(frameRange?.[0] ?? 0)}.</p>
      </section>

      <section class="settings">
        <label class="wide"><span>Adapter checkpoint</span>
          <select value={catalog.files.includes(entry.path) ? entry.path
              : (entry.path ? '__custom__' : '')}
            onchange={(event)=>entryPatch(key,{path:event.currentTarget.value==='__custom__'
              ? (entry.path || ' ') : event.currentTarget.value})}>
            <option value="">Select a checkpoint…</option>
            {#each catalog.files as file}<option value={file}>{basename(file)}</option>{/each}
            <option value="__custom__">Custom path…</option>
          </select>
        </label>
        {#if entry.path && !catalog.files.includes(entry.path)}
          <label class="wide"><span>Custom checkpoint path</span><input value={entry.path.trim()}
            placeholder="C:\models\controlnet\ip-adapter-plus_sd15.safetensors"
            onchange={(event)=>entryPatch(key,{path:stripQuotes(event.currentTarget.value)})}/></label>
        {/if}
        <label><span>Start <small>fraction of steps</small></span><input type="number"
          min="0" max="1" step="0.05" value={entry.start}
          onchange={(event)=>entryPatch(key,{start:number(event)})}/></label>
        <label><span>End <small>fraction of steps</small></span><input type="number"
          min="0" max="1" step="0.05" value={entry.end}
          onchange={(event)=>entryPatch(key,{end:number(event)})}/></label>
        <label><span>Weight type</span><select value={entry.weight_type}
          onchange={(event)=>entryPatch(key,{weight_type:event.currentTarget.value})}>
          {#each choices('weight_type',['linear']) as choice}<option value={choice}>{choice}</option>{/each}
        </select></label>
        <label><span>Combine images</span><select value={entry.combine_embeds}
          onchange={(event)=>entryPatch(key,{combine_embeds:event.currentTarget.value})}>
          {#each choices('combine_embeds',['concat','add','subtract','average','norm average']) as choice}
            <option value={choice}>{choice}</option>
          {/each}
        </select></label>
        <p class="combine-hint">{combineHints[entry.combine_embeds] ?? combineHints.concat}</p>
        <label class="wide"><span>Embedding scaling</span><select value={entry.embeds_scaling}
          onchange={(event)=>entryPatch(key,{embeds_scaling:event.currentTarget.value})}>
          {#each choices('embeds_scaling',['V only','K+V']) as choice}<option value={choice}>{choice}</option>{/each}
        </select></label>
      </section>
    </div>
  </article>
{/each}
</div>

<style>
  .base{display:flex;gap:28px;padding-bottom:18px;border-bottom:1px solid #292e36}
  .toggle{display:flex;flex-direction:row;align-items:center;gap:9px;color:#aeb4bd;font-size:12px}
  .toggle input{width:18px;height:18px}
  .adder{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:12px;align-items:end;margin-top:18px}
  .adder .hint{grid-column:1/-1;margin:0;color:#68717d;font-size:10px}
  .add{height:38px;border:1px solid #4d5d2a;background:#d8ff55;color:#111;font-weight:700;padding:10px 20px;border-radius:8px;cursor:pointer}
  .add:disabled{background:#191c21;color:#5c636c;border-color:#303640}
  .none{margin-top:18px;padding:26px;border:1px dashed #30353d;border-radius:10px;color:#656d78;text-align:center;font-size:12px}
  .adapter-list{display:grid;gap:14px;margin-top:18px}
  article{border:1px solid #292e36;border-radius:11px;background:#0e1115;overflow:hidden}
  header{display:flex;align-items:center;gap:12px;padding:11px 13px;background:#12161b}
  header>div{display:flex;flex-direction:column;flex:1;min-width:0}header b{color:#e7eaee;font-size:13px}header div span{color:#626b76;font-size:9px;text-transform:uppercase}
  .weight{display:flex;flex-direction:row;align-items:center;gap:5px;padding:3px 8px;border:1px solid #343a43;border-radius:20px}
  .weight span{color:#8ea834;font:600 11px Consolas}.weight input{width:48px;border:0;padding:2px;text-align:center}
  .remove{border:1px solid #303640;background:#171a1f;color:#8b929c;width:27px;height:27px;border-radius:7px;cursor:pointer}
  .remove:hover{border-color:#67363a;color:#f18b91}
  .badge{font-size:9px;padding:3px 7px;border-radius:20px;border:1px solid;white-space:nowrap}
  .badge.ok{color:#b6d96a;border-color:#4d5d2a;background:#1a2013}
  .badge.warn{color:#e0c169;border-color:#6b5726;background:#211c11}
  .badge.error{color:#f18b91;border-color:#67363a;background:#241416}
  .content{display:grid;grid-template-columns:minmax(300px,1.35fr) minmax(260px,1fr);gap:16px;padding:16px}
  .source-stack{min-width:0}.source-title{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px}
  .source-title b,.image-head b{color:#cfd4db;font-size:11px}.source-title span{color:#68717d;font-size:9px;text-transform:uppercase}
  .source-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:10px}
  .source-card{min-width:0;padding:10px;border:1px solid #292f37;border-radius:9px;background:#0b0e12}
  .image-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
  .remove-image{border:0;background:transparent;color:#737c87;font-size:9px;cursor:pointer;padding:3px}
  .remove-image:hover{color:#f18b91}.add-image{width:100%;margin-top:10px;padding:8px;border:1px dashed #3b434d;border-radius:8px;background:#0b0e12;color:#aeb4bd;font-size:10px;cursor:pointer}
  .add-image:hover{border-color:#8ea834;color:#d8ff55}
  .settings{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:13px;align-content:start}
  .wide{grid-column:1/-1}
  .preview{position:relative;width:100%;aspect-ratio:1;border:1px solid #292f37;border-radius:8px;background:#080a0d;overflow:hidden;margin-bottom:11px}
  .preview img{width:100%;height:100%;object-fit:cover;display:block}
  .preview>span{color:#68717d;font-size:10px;text-align:center;padding:12px}
  .preview.empty,.drop{display:grid;place-items:center;align-content:center;gap:5px}
  .preview i{color:#9daa83;font:300 32px/1 system-ui;font-style:normal}
  .preview img+span,.drop img+span{position:absolute;left:50%;bottom:8px;transform:translateX(-50%);padding:5px 8px;border-radius:6px;background:#090b0ddd;color:#dce4d0;white-space:nowrap}
  .drop{cursor:pointer}.drop:hover,.drop.over{border-color:#9fbe4d;background:#11170d}
  .picker{display:none}.source-stack>p{margin:8px 0 0;color:#68717d;font-size:10px;line-height:1.4}.err{display:block;color:#f18b91;font-size:10px;margin:-5px 0 8px}
  .combine-hint{grid-column:1/-1;margin:-6px 0 0;color:#68717d;font-size:10px;line-height:1.4}
  label{display:flex;flex-direction:column;gap:7px;color:#aeb4bd;font-size:11px}small{color:#68717d}
  input,select{width:100%;border:1px solid #303640;background:#0c0f13;color:#eef0f2;border-radius:8px;padding:10px 11px;outline:none;font:12px Consolas,monospace}
  input:focus,select:focus{border-color:#8ea834}
  @media(max-width:760px){.content{grid-template-columns:1fr}.source-grid{grid-template-columns:repeat(auto-fit,minmax(145px,1fr))}.settings{grid-template-columns:1fr}.wide,.combine-hint{grid-column:1}.base{flex-direction:column;gap:10px}}
</style>
