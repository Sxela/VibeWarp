<script>
  import { tick } from 'svelte';
  import Field from './Field.svelte';
  import ControlNetEditor from './ControlNetEditor.svelte';
  import VideoEditor from './VideoEditor.svelte';
  import Preview from './Preview.svelte';
  import Supporters from './Supporters.svelte';
  const storageKey='vibewarp.config.v1';
  let config=$state(null), schema=$state(null), selected=$state('project'), job=$state(null), errors=$state([]), busy=$state(false), initialized=$state(false), logCollapsed=$state(false), logElement=$state(null), source;
  // Tabs come from the BACKEND (vibewarp/ui_layout.py stamps tier/group onto every field
  // in the schema). The nav holds no field list of its own — one here would drift silently
  // the moment config.py changed. A field with no tier renders nowhere, and a test fails.
  const blurbs={project:'Input & output for this video',render:'What you change every run',advanced:'Set once, or never',system:'This machine — not saved with settings',preview:'Inspect'};
  // 'preview' is not a config section — it's the frame-by-frame debug view.
  // Preview is a MODE, not a settings tab: it has nothing to do with the config form and
  // wants the whole window (runs list, viewer, layer chips) rather than a column of it.
  let mode=$state('render');
  // The monitor thumbnail is small and it is the only look you get at a frame mid-render,
  // so it opens full size on click -- same gesture as the History view.
  let zoomShot=$state(null);
  // Latches once ANY frame has rendered this session, so the supporters block appears
  // while you wait for your first frame and then never interrupts again.
  let seenFirstFrame=$state(false);
  $effect(()=>{ if(job?.preview_available) seenFirstFrame=true; });
  let tabs=$derived(schema ? schema.tiers : []);

  // Flatten the schema into {section, name, schema, tier, group}, in declaration order.
  let entries=$derived.by(()=>{
    if(!schema) return [];
    let out=[];
    for(let [name,prop] of Object.entries(schema.properties)){
      if(prop.type==='dataclass'){
        for(let [child,sub] of Object.entries(prop.properties))
          if(sub.tier) out.push({section:name,name:child,schema:sub,tier:sub.tier,group:sub.group});
      } else if(prop.tier) out.push({section:'main',name,schema:prop,tier:prop.tier,group:prop.group});
    }
    return out;
  });
  // Groups for the selected tab, preserving field order within each.
  let groups=$derived.by(()=>{
    let out=[];
    for(let e of entries.filter(e=>e.tier===selected)){
      let g=out.find(g=>g.name===e.group);
      if(!g) out.push(g={name:e.group,items:[]});
      g.items.push(e);
    }
    // Order by what ui_layout DECLARED, not by the order fields sit on RunConfig -- the
    // latter buried "Input Video" behind "Output" and "Model" on the Project tab.
    let order=tabs.find(t=>t.id===selected)?.groups ?? [];
    if(order.length) out.sort((a,b)=>{
      let ia=order.indexOf(a.name), ib=order.indexOf(b.name);
      return (ia<0?999:ia)-(ib<0?999:ib);
    });
    return out;
  });

  // Search spans EVERY tab. With ~190 fields across 4 tabs and Advanced collapsed by
  // default, finding `softcap_thresh` otherwise means guessing which of 13 groups it is in.
  let query=$state('');
  const pretty=(s)=>s.replaceAll('_',' ');
  let hits=$derived.by(()=>{
    let q=query.trim().toLowerCase();
    if(!q) return [];
    return entries.filter(e=>
      pretty(e.name).toLowerCase().includes(q) ||
      e.group.toLowerCase().includes(q) ||
      e.section.toLowerCase().includes(q)
    ).slice(0,60);
  });
  const tierLabel=(id)=>tabs.find(t=>t.id===id)?.label||id;

  // Validation errors arrive as {path:'video.video_init_path'} — the same shape as the
  // layout keys, so we can send you straight to the offending field instead of making you
  // guess which of 4 tabs and 13 collapsed groups it is hiding in.
  function locate(path){
    if(!path) return null;
    let dot=path.indexOf('.');
    let section=dot<0?'main':path.slice(0,dot), name=dot<0?path:path.slice(dot+1);
    return entries.find(e=>e.section===section&&e.name===name)||null;
  }
  // 'section.field' -> message, so a field can show its OWN error instead of the sidebar
  // being the only place that knows.
  let errorFor=$derived.by(()=>{
    let map={};
    for(let e of errors){
      let hit=locate(e.path);
      if(hit) map[`${hit.section}.${hit.name}`]=e.message;
    }
    return map;
  });
  const fieldError=(item)=>errorFor[`${item.section}.${item.name}`]||'';
  // Bespoke editors (VideoEditor, ControlNetEditor) render their own inputs, so they need
  // the errors for their whole section rather than one field at a time.
  const sectionErrors=(section)=>Object.fromEntries(
    Object.entries(errorFor).filter(([k])=>k.startsWith(section+'.'))
                            .map(([k,v])=>[k.slice(section.length+1),v]));
  // How many errors each TAB holds, so a collapsed Advanced group can't hide one from you.
  let errorsPerTier=$derived.by(()=>{
    let count={};
    for(let e of errors){
      let hit=locate(e.path);
      if(hit) count[hit.tier]=(count[hit.tier]||0)+1;
    }
    return count;
  });

  async function goToError(path){
    let hit=locate(path);
    if(!hit) return;
    selected=hit.tier;
    query='';
    openGroups.add(hit.group); openGroups=new Set(openGroups);   // Advanced starts collapsed
    await tick();
    // VideoEditor carries its own anchors (see its `errors` prop) — video_init_path is the
    // most common error there is, so "just open the tab" was not good enough. ControlNet
    // still has none; if errors start landing inside it, give it anchors the same way.
    let node=document.getElementById(`field-${hit.section}-${hit.name}`);
    if(!node) return;
    node.scrollIntoView({behavior:'smooth',block:'center'});
    node.classList.remove('flash'); void node.offsetWidth; node.classList.add('flash');
    node.querySelector('input,textarea,select')?.focus({preventScroll:true});
  }
  // These sections have purpose-built editors; the generic Field grid can't express them.
  const custom={'Input Video':'video','ControlNet':'controlnet'};
  // Groups that span both columns: the custom editors, plus anything whose fields are
  // long-form (prompt textareas, keyframe chip rows) and would be cramped at half width.
  const wide=new Set(['Prompts','Scene Scheduling','Flow & Consistency']);
  // A group can point at the tab holding the rest of its settings. AnimateDiff's toggle is
  // a MODEL choice (in the notebook it IS the model version), so it sits on the Model card
  // — but its tuning lives in Advanced, and you should not have to go hunting for it.
  const seeAlso={Model:{when:()=>config?.animatediff?.enabled,tier:'advanced',group:'AnimateDiff'}};
  function openGroup(tier,group){selected=tier;query='';openGroups.add(group);openGroups=new Set(openGroups)}
  let openGroups=$state(new Set());
  function setField(name,v){config={...config,[name]:v}}
  function setIn(section,name,v){
    if(section==='main') return setField(name,v);
    config={...config,[section]:{...config[section],[name]:v}};
  }
  function fieldHint(item){
    if(item.section!=='diffusion'||item.name!=='sampler_tile_size') return '';
    let vanillaSdxl=(config?.model_version||'').toLowerCase().includes('sdxl')&&!config?.animatediff?.enabled;
    return vanillaSdxl
      ? 'Recommended: 1024 px — vanilla SDXL was trained at 1024 px.'
      : 'Recommended: 512 px — SD1.5 and AnimateDiff models were trained at 512 px.';
  }
  const enabledDot=(section)=>config?.[section]?.enabled||config?.[section]?.flow_warp||config?.[section]?.do_freeunet||config?.[section]?.make_captions||config?.[section]?.use_background_mask;
  function mergeDefaults(defaults,saved){if(!saved||typeof saved!=='object'||Array.isArray(saved))return defaults;let result={...defaults};for(let key of Object.keys(saved)){if(defaults[key]&&typeof defaults[key]==='object'&&!Array.isArray(defaults[key]))result[key]=mergeDefaults(defaults[key],saved[key]);else result[key]=saved[key]}return result}
  async function init(){let r=await fetch('/api/config');let d=await r.json();schema=d.schema;let saved=null;if(!d.prefilled){try{saved=JSON.parse(localStorage.getItem(storageKey))}catch{localStorage.removeItem(storageKey)}}config=mergeDefaults(d.defaults,saved);initialized=true} init();
  $effect(()=>{if(initialized&&config)localStorage.setItem(storageKey,JSON.stringify(config))});
  $effect(()=>{job?.revision;if(!logCollapsed&&logElement)requestAnimationFrame(()=>{logElement.scrollTop=logElement.scrollHeight})});
  function listen(id){source?.close();source=new EventSource(`/api/jobs/${id}/events`);source.onmessage=(e)=>{job=JSON.parse(e.data);if(['completed','failed','cancelled'].includes(job.state))source.close()}}
  async function run(){busy=true;errors=[];let r=await fetch('/api/jobs',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(config)});let d=await r.json();busy=false;if(!r.ok){errors=Array.isArray(d.detail)?d.detail:[{message:d.detail||'Submission failed'}];return}job=d;listen(job.id)}
  async function cancel(){if(job)await fetch(`/api/jobs/${job.id}/cancel`,{method:'POST'})}
  // System-tier fields describe the MACHINE (model paths, VRAM, threads), so they are
  // stripped from an export and never travel to someone else's install.
  let systemFields=$derived(entries.filter(e=>e.tier==='system'));
  function withoutSystem(c){
    let out=structuredClone($state.snapshot(c));
    for(let e of systemFields){
      if(e.section==='main') delete out[e.name];
      else if(out[e.section]) delete out[e.section][e.name];
    }
    return out;
  }
  function exportConfig(){let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(withoutSystem(config),null,2)],{type:'application/json'}));a.download=`${config.batch_name||'vibewarp'}.json`;a.click();URL.revokeObjectURL(a.href)}
  // Persist the system tier server-side (system.json next to the install) whenever it
  // changes, so it survives a settings import, a reset, and a new browser.
  let systemSnapshot=$state(null);
  $effect(()=>{
    if(!initialized||!config||!systemFields.length) return;
    let current=JSON.stringify(systemFields.map(e=>e.section==='main'?config[e.name]:config[e.section]?.[e.name]));
    if(systemSnapshot===null){systemSnapshot=current;return}   // don't re-save what we just loaded
    if(current===systemSnapshot) return;
    systemSnapshot=current;
    fetch('/api/system',{method:'PUT',headers:{'content-type':'application/json'},body:JSON.stringify($state.snapshot(config))}).catch(()=>{});
  });
  async function importConfig(e){let f=e.target.files[0];if(!f)return;busy=true;errors=[];let content=await f.text();let r=await fetch('/api/config/import',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({filename:f.name,content})});let d=await r.json();busy=false;if(!r.ok){errors=Array.isArray(d.detail)?d.detail:[{message:d.detail||'Import failed'}];return}config=d.config;e.target.value=''}
  async function resetConfig(){localStorage.removeItem(storageKey);let r=await fetch('/api/config');let d=await r.json();config=d.defaults;errors=[];selected='project'}
</script>
<svelte:head><title>VibeWarp</title></svelte:head>
<header>
  <div class="brand"><span class="mark">V</span><div><h1>VibeWarp</h1><p>Video diffusion workspace</p></div></div>
  <div class="modes">
    <button class:on={mode==='render'} onclick={()=>mode='render'}>Render</button>
    <button class:on={mode==='preview'} onclick={()=>mode='preview'}>History &amp; Comparison</button>
  </div>
  <div class="actions">
    {#if mode==='render'}
      <label class="file">Import settings<input type="file" accept=".json,.txt" onchange={importConfig}/></label>
      <button class="ghost" onclick={resetConfig}>Reset</button>
      <button class="ghost" onclick={exportConfig}>Export</button>
    {/if}
    <button class="run" onclick={run} disabled={busy||job?.state==='running'}>{busy?'Loading…':'Start render'}</button>
  </div>
</header>
{#if mode==='preview'}
  <div class="preview-shell" style={`--log-height:${logCollapsed?'42px':'282px'}`}>
    <Preview {config} {job} onload={(c)=>{config=c;errors=[];}}/>
  </div>
{:else}
<main class="render-shell" class:compact-log={logCollapsed} style={`--log-height:${logCollapsed?'42px':'282px'}`}>
  <nav>{#each tabs as tab}<button class:active={selected===tab.id} onclick={()=>selected=tab.id}><span>{tab.label}</span>{#if errorsPerTier[tab.id]}<em>{errorsPerTier[tab.id]}</em>{/if}</button>{/each}</nav>
  <section class="editor">
    {#if !schema}<div class="loading">Loading configuration…</div>
    {:else}
      <div class="section-head">
        <div><p>{query.trim() ? 'Across all tabs' : blurbs[selected]}</p>
             <h2>{query.trim() ? `${hits.length} match${hits.length===1?'':'es'}` : tabs.find(t=>t.id===selected)?.label}</h2></div>
        <div class="search-box">
          <input class="search" placeholder="Search settings…" bind:value={query}
                 onkeydown={(e)=>{if(e.key==='Escape')query=''}}/>
          {#if query}<button class="clear" onclick={()=>query=''} aria-label="Clear search">×</button>{/if}
        </div>
      </div>
      {#if query.trim()}
        {#if !hits.length}
          <div class="panel"><div class="empty">Nothing matches “{query}”.</div></div>
        {:else}
          <div class="panel"><div class="grid">
            {#each hits as item (item.section+'.'+item.name)}
              {#if custom[item.group]}
                <button class="jump" onclick={()=>{selected=item.tier;query=''}}>
                  <b>{pretty(item.name)}</b>
                  <span>{tierLabel(item.tier)} › {item.group} — open the tab</span>
                </button>
              {:else}
                <div class="hit">
                  <span class="where">{tierLabel(item.tier)} › {item.group}</span>
                  <Field name={item.name} schema={item.schema} path={item.section}
                         hint={fieldHint(item)} error={fieldError(item)}
                         value={item.section==='main' ? config[item.name] : config[item.section]?.[item.name]}
                         onchange={(v)=>setIn(item.section,item.name,v)}/>
                </div>
              {/if}
            {/each}
          </div></div>
        {/if}
      {:else}
      <div class="groups">
      {#each groups as group (group.name)}
        {@const only=custom[group.name]}
        {@const full=!!only || wide.has(group.name)}
        {@const link=seeAlso[group.name]}
        <details class="panel group" class:full open={selected!=='advanced' || openGroups.has(group.name)}>
          <summary><span>{group.name}</span>{#if only && enabledDot(only)}<i></i>{/if}</summary>
          <div class={only ? 'stack' : 'grid'}>
            {#if only==='video'}
              <VideoEditor value={config.video} errors={sectionErrors('video')}
                          onchange={(v)=>setField('video',v)}
                          frameRange={config.frame_range}
                          onFrameRange={(v)=>setField('frame_range',v)}/>
            {:else if only==='controlnet'}
              <ControlNetEditor value={config.controlnet} schema={schema.properties.controlnet}
                                modelVersion={config.model_version}
                                onchange={(v)=>setField('controlnet',v)}/>
            {:else}
              {#each group.items as item (item.section+'.'+item.name)}
                <Field name={item.name} schema={item.schema} path={item.section}
                       hint={fieldHint(item)} error={fieldError(item)}
                       value={item.section==='main' ? config[item.name] : config[item.section]?.[item.name]}
                       onchange={(v)=>setIn(item.section,item.name,v)}/>
              {/each}
              {#if link?.when()}
                <button class="see-also" onclick={()=>openGroup(link.tier,link.group)}>
                  {link.group} settings → {tierLabel(link.tier)}
                </button>
              {/if}
            {/if}
          </div>
        </details>
      {/each}
      </div>
      {/if}
    {/if}
  </section>
  <aside>
    <div class="monitor-head"><span class:live={job?.state==='running'}></span><div><p>Render monitor</p><h3>{job?.message||'Ready to render'}</h3></div></div>
    {#if errors.length}<div class="errors">{#each errors as error}
      {@const hit=locate(error.path)}
      {#if hit}
        <button class="err-jump" onclick={()=>goToError(error.path)}>
          <b>{pretty(hit.name)}</b> {error.message}
          <span>{tierLabel(hit.tier)} › {hit.group}</span>
        </button>
      {:else}<p><b>{error.path||'Error'}</b> {error.message}</p>{/if}
    {/each}</div>{/if}
    {#if job}
      <div class="progress-meta"><span>{job.stage}</span><span>{Math.round(job.progress)}%</span></div><div class="progress"><i style={`width:${job.progress}%`}></i></div>
      <div class="stats"><div><b>{job.frame}</b><span>Frame</span></div><div><b>{job.total_frames||'—'}</b><span>Total</span></div><div><b>{job.state}</b><span>Status</span></div></div>
      {#if job.preview_available}
        <button class="shot" onclick={()=>zoomShot=`/api/jobs/${job.id}/preview?v=${job.revision}`} aria-label="Zoom latest render">
          <img class="preview" src={`/api/jobs/${job.id}/preview?v=${job.revision}`} alt="Latest render"/>
        </button>
      {:else}<div class="empty">Rendered frames will appear here</div>{/if}
      {#if job.state==='running' || job.state==='queued'}<button class="cancel" onclick={cancel}>Cancel render</button>{/if}
    {:else}<div class="empty tall"><strong>No active job</strong><span>Configure the render and start when ready.</span></div>{/if}
    {#if !seenFirstFrame}<Supporters/>{/if}
  </aside>
</main>
{/if}
{#if zoomShot}
  <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
  <div class="lightbox" onclick={()=>zoomShot=null}>
    <img src={zoomShot} alt="Latest render"/>
    <button class="close" aria-label="Close zoom">×</button>
  </div>
{/if}
<svelte:window onkeydown={(e)=>{if(e.key==='Escape')zoomShot=null}}/>
<section class:collapsed={logCollapsed} class="log-dock">
  <button class="log-head" onclick={()=>logCollapsed=!logCollapsed} aria-expanded={!logCollapsed}>
    <span><i class:live={job?.state==='running'}></i> Output log</span>
    <code>{logCollapsed ? (job?.logs?.at(-1)||'Waiting for pipeline output…') : ''}</code>
    <b>{logCollapsed?'Expand':'Collapse'}</b>
  </button>
  {#if !logCollapsed}<pre bind:this={logElement}>{job?.logs?.join('\n')||'Waiting for pipeline output…'}</pre>{/if}
</section>
<style>
  /* A tab is a list of collapsible groups. Advanced starts collapsed — it is the "set once
     or never" tier, and 13 groups expanded is the wall of settings we are getting rid of.

     The groups FLOW into two columns (multi-column, not grid: grid rows would leave a tall
     group stranded next to a short one). break-inside keeps a group whole. Groups whose
     fields are long-form — prompts, the ControlNet cards, chip rows — span both columns. */
  .groups{column-count:2;column-gap:14px}
  .group{break-inside:avoid;-webkit-column-break-inside:avoid}
  .group.full{column-span:all}
  /* Half-width columns are too narrow for a 2-up field grid; only the full-span groups
     keep two fields per row. */
  .group:not(.full)>:global(.grid){grid-template-columns:1fr}
  @media(max-width:1500px){.groups{column-count:1}.group:not(.full)>:global(.grid){grid-template-columns:repeat(2,minmax(0,1fr))}}
  .group{padding:0;margin-bottom:14px;overflow:hidden}
  .group>summary{display:flex;align-items:center;gap:9px;padding:16px 22px;cursor:pointer;color:#dce0e5;font-size:13px;font-weight:600;list-style:none;user-select:none}
  .group>summary::-webkit-details-marker{display:none}
  .group>summary::before{content:'';width:5px;height:5px;border-right:1.5px solid #6d7580;border-bottom:1.5px solid #6d7580;transform:rotate(-45deg);transition:transform .15s;margin-right:2px}
  .group[open]>summary::before{transform:rotate(45deg)}
  .group>summary:hover{background:#171b21}
  .group>summary i{width:6px;height:6px;border-radius:50%;background:#d8ff55}
  .group>:global(.grid),.group>:global(.stack){padding:4px 22px 22px}
  /* Render vs Preview: two different jobs, so two different layouts rather than cramming
     the frame viewer into a settings column. */
  .modes{display:flex;gap:3px;padding:3px;border:1px solid #2b3038;background:#0e1116;border-radius:10px}
  .modes button{border:0;background:none;color:#8b929c;padding:7px 18px;border-radius:8px;cursor:pointer;font-size:13px}
  .modes button:hover{color:#e7eaee}
  .modes button.on{background:#232a33;color:#fff;font-weight:600}
  /* The nav stays put while the settings scroll. `aside` was already sticky; `nav` was not,
     so it scrolled off the top and you had to scroll back up to change tab. */
  :global(main.render-shell>nav){position:sticky;top:78px;align-self:start}
  /* A flex column so the supporters block can absorb whatever height is left below the
     monitor, instead of sitting in a fixed strip with dead space under it. */
  :global(main.render-shell>aside){display:flex;flex-direction:column}
  :global(.section-head){display:flex;align-items:flex-end;justify-content:space-between;gap:20px}
  .search-box{position:relative;margin-bottom:22px}
  .search{width:260px;border:1px solid #303640;background:#0c0f13;color:#eef0f2;border-radius:9px;padding:9px 30px 9px 12px;font:12px Consolas,monospace}
  .search:focus{border-color:#8ea834;outline:none}
  /* Only rendered when there is something to clear — an always-on × next to an empty box
     is just a dead control. Escape does the same thing from the keyboard. */
  .clear{position:absolute;top:50%;right:8px;transform:translateY(-50%);display:grid;place-items:center;width:18px;height:18px;border:0;border-radius:50%;background:#2a3038;color:#aeb4bd;font-size:13px;line-height:1;cursor:pointer;padding:0}
  .clear:hover{background:#c0505a;color:#fff}
  /* A search hit says where it lives, so you learn the layout instead of fighting it. */
  .hit{display:flex;flex-direction:column;gap:5px}
  .where{color:#6f7883;font-size:10px;text-transform:uppercase;letter-spacing:.09em}
  /* ControlNet / Input Video have bespoke editors, so a field can't be shown inline —
     offer the tab instead of pretending the setting isn't there. */
  .jump{display:flex;flex-direction:column;gap:4px;align-items:flex-start;text-align:left;border:1px dashed #3a414b;background:none;border-radius:9px;padding:11px 13px;cursor:pointer}
  .jump:hover{border-color:#8ea834;background:#151a14}
  .jump b{color:#e7eaee;font-size:13px}
  .jump span{color:#7f8792;font-size:11px}
  /* Only shown once the feature is on, so it points at settings you can actually use. */
  .see-also{grid-column:1/-1;justify-self:start;border:1px solid #3a414b;background:none;color:#c8d3ad;border-radius:8px;padding:8px 13px;font-size:12px;cursor:pointer}
  .see-also:hover{border-color:#8ea834;background:#151a14;color:#d8ff55}
  /* An error you can click takes you to the field. The path alone ("video.video_init_path")
     made you find the tab and group yourself. */
  .err-jump{display:block;width:100%;text-align:left;border:0;background:none;color:#f0c3c6;padding:5px 4px;border-radius:6px;cursor:pointer;font-size:11px}
  .err-jump:hover{background:#4a2429}
  /* Error count per tab: a collapsed Advanced group must not be able to hide one. */
  :global(main>nav em){display:grid;place-items:center;min-width:17px;height:17px;padding:0 4px;border-radius:9px;background:#c0505a;color:#fff;font:600 10px/1 system-ui,sans-serif;font-style:normal}
  .err-jump b{color:#fff;margin-right:5px}
  .err-jump span{display:block;margin-top:3px;color:#c78d93;font-size:10px;text-transform:uppercase;letter-spacing:.08em}
  /* The monitor image is a button so it can be zoomed; strip the button chrome. */
  .shot{display:block;width:100%;padding:0;border:0;background:none;cursor:zoom-in}
  .lightbox{position:fixed;inset:0;z-index:30;background:#05070aee;display:grid;place-items:center;padding:32px;cursor:zoom-out;overflow:auto}
  .lightbox img{max-width:100%;max-height:100%;object-fit:contain}
  .close{position:fixed;top:18px;right:24px;width:36px;height:36px;border:1px solid #353a42;border-radius:50%;background:#12151a;color:#d9dce0;font-size:19px;line-height:1;cursor:pointer}
  .close:hover{background:#c0505a;color:#fff;border-color:#c0505a}
  .log-dock{position:fixed;z-index:10;left:0;right:0;bottom:0;background:#090b0eeb;border-top:1px solid #30353d;backdrop-filter:blur(14px);box-shadow:0 -12px 35px #0008}.log-head{width:100%;height:42px;display:grid;grid-template-columns:auto minmax(0,1fr) auto;align-items:center;gap:16px;padding:0 20px;border:0;background:#12151a;color:#aeb4bd;cursor:pointer;text-align:left}.log-head span{display:flex;align-items:center;gap:9px;color:#e4e7eb;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.08em}.log-head i{width:6px;height:6px;border-radius:50%;background:#555}.log-head i.live{background:#d8ff55;box-shadow:0 0 10px #d8ff55}.log-head code{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#747d88;font:11px Consolas,monospace}.log-head b{font-size:11px;color:#7d8691}.log-dock pre{height:240px;max-height:35vh;margin:0;border-radius:0;padding:14px 20px;background:#080a0d;color:#9ca6b2;overflow:auto;font:11px/1.55 Consolas,monospace;white-space:pre-wrap}.log-dock.collapsed{height:42px}:global(main.render-shell){padding-bottom:282px}:global(main.render-shell>nav),:global(main.render-shell>aside){height:calc(100vh - 78px - var(--log-height));overflow:auto}.compact-log{padding-bottom:42px}@media(max-width:700px){.log-head{padding:0 12px}.log-head b{display:none}.log-dock pre{height:190px}:global(main.render-shell){padding-bottom:232px}.compact-log{padding-bottom:42px}}
</style>
