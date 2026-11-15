<script>
  let {layers=[], frame=0, view=$bindable('inspect'), selected=$bindable(['init','output']),
    focus=$bindable(null), sourceUrl='', liveUrl='', sourcePending=false, sourceEmpty=false, sourceError='', outputPending=false, imageUrl, onzoom=()=>{}, onchoose=()=>{}} = $props();
  const placeholder=(id,label)=>({id,label,group:'Frames',frames:[]});
  let input=$derived(layers.find(layer=>layer.id==='init')||placeholder('init','Source'));
  let output=$derived(layers.find(layer=>layer.id==='output')||placeholder('output','Output'));
  let hasOutput=$derived(output.frames.length>0||!!liveUrl);
  let shown=$derived(view==='source'?[input]:view==='output'?[output]:view==='compare'?(hasOutput?[input,output]:[input]):layers.filter(layer=>selected.includes(layer.id)));
  let groups=$derived.by(()=>{
    const result=new Map();
    for(const layer of layers){const group=layer.category||layer.group||'Frames';if(!result.has(group))result.set(group,[]);result.get(group).push(layer);}
    return [...result];
  });
  const present=layer=>layer.id==='init'&&(sourcePending||sourceError||sourceEmpty)?false:layer.id==='init'&&sourceUrl?true:layer.id==='output'&&liveUrl?true:layer.frames.includes(frame);
  const url=layer=>layer.id==='init'&&sourceUrl?sourceUrl:layer.id==='output'&&liveUrl?liveUrl:imageUrl(layer.id);
  function toggle(id){selected=selected.includes(id)?selected.filter(item=>item!==id):[...selected,id];if(selected.includes(id))focus=id;}
  $effect(()=>{if(!shown.some(layer=>layer.id===focus))focus=shown[0]?.id||null;});
</script>

<div class="viewer-tabs" role="tablist" aria-label="Frame view">
  {#each [['source','Source'],['output','Output'],['compare','Source / output'],['inspect','Inspect']] as [id,label]}
    <button role="tab" aria-selected={view===id} class:on={view===id} onclick={()=>view=id}>{label}</button>
  {/each}
  {#if shown.length>1}<span class="layer-count">{shown.length} layers · frame {frame}</span>
    <label class="mobile-layer-picker"><span class="visually-hidden">Displayed diagnostic layer</span><select bind:value={focus}>{#each shown as layer}<option value={layer.id}>{layer.label}</option>{/each}</select></label>
  {/if}
</div>
<div class="layer-stage" style={`--cols:${Math.max(1,Math.min(shown.length,3))}`}>
  {#if !shown.length}<div class="stage-empty">Select layers below to inspect this frame.</div>{/if}
  {#each shown as layer (layer.id)}
    <figure class:focused={focus===layer.id}>
      <figcaption><span>{layer.short_label||layer.label}</span><small>Frame {frame}</small></figcaption>
      {#if present(layer)}
        <button class="layer-shot" onclick={()=>{focus=layer.id;onzoom(layer.id);}} aria-label={`Enlarge ${layer.label}`}><img src={url(layer)} alt={`${layer.label}, frame ${frame}`}/></button>
      {:else}<div class="missing">
        {#if layer.id==='init'&&sourcePending}<strong>Loading source preview…</strong>
        {:else if layer.id==='init'&&sourceError}<strong>{sourceError}</strong><button onclick={onchoose}>Check input video</button>
        {:else if layer.id==='init'&&(sourceEmpty||!input.frames.length)}<button class="choose-source" onclick={onchoose}>Choose input video</button>
        {:else if layer.id==='output'&&(!hasOutput||outputPending&&frame>Math.max(-1,...output.frames))}<strong>{hasOutput?'Waiting for this output frame':'Waiting for first output'}</strong><small>Rendered frames will appear here.</small>
        {:else}<strong>Not saved for frame {frame}</strong><small>This run has no image for this layer at the selected frame.</small>{/if}
      </div>{/if}
    </figure>
  {/each}
</div>
{#if view==='inspect'}
  <div class="layer-selectors" aria-label="Diagnostic layers">
    {#each groups as [group,items]}
      <div class="layer-group"><span>{group}</span>{#each items as layer}
        <button class:on={selected.includes(layer.id)} class:unavailable={!layer.frames.includes(frame)} aria-pressed={selected.includes(layer.id)} onclick={()=>toggle(layer.id)}>{layer.short_label||layer.label}{#if !layer.frames.includes(frame)}<small>Not saved</small>{/if}</button>
      {/each}</div>
    {/each}
    {#if layers.some(layer=>layer.id==='debug:consistency_mask')}<small class="mask-key">CC mask: white keeps the warp; black marks inconsistent areas.</small>{/if}
  </div>
{/if}

<style>
  .choose-source{background:var(--accent);border-color:var(--accent);color:#152000}
  .viewer-tabs{display:flex;align-items:center;flex-wrap:wrap;gap:4px;padding:10px 16px;border-bottom:1px solid #303e48;min-height:53px;flex:0 0 auto;background:#121b21}
  .viewer-tabs button{border:1px solid transparent;background:transparent;color:#b9c7d0;font-size:12px;min-height:32px;padding:5px 10px;border-radius:5px}.viewer-tabs button:hover{color:#fff;background:#25333d}.viewer-tabs button.on{border-color:#52626e;color:#eef3f5;background:#293741}
  .layer-count{font-size:11px;color:#b8c9d3;margin-left:auto}.mobile-layer-picker{display:none}
  .layer-stage{flex:1;min-height:0;min-width:0;display:grid;grid-template-columns:repeat(var(--cols),minmax(0,1fr));grid-auto-rows:minmax(0,1fr);gap:12px;padding:16px;background:#090e12;overflow:hidden}
  figure{margin:0;display:flex;flex-direction:column;min-width:0;min-height:0;background:#0c1319;border:1px solid #354550;border-radius:6px;overflow:hidden}figcaption{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:9px 11px;background:#18232b;font-size:12px;color:#e0e9ee;flex:0 0 auto}figcaption span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}figcaption small{font:11px Consolas,monospace;color:#adbfcb;white-space:nowrap}.layer-shot{display:flex;align-items:center;justify-content:center;flex:1;min-height:0;width:100%;padding:0;border:0;border-radius:0;background:#070b0e;cursor:zoom-in;overflow:hidden}.layer-shot:hover{background:#0d151b}.layer-shot img{width:100%;height:100%;object-fit:contain;min-height:0;display:block}.missing,.stage-empty{display:flex;flex-direction:column;align-items:center;justify-content:center;flex:1;min-height:0;padding:24px;gap:9px;color:#bdcbd5;text-align:center;font-size:13px}.missing{overflow:auto;padding:12px}.missing strong{font-weight:500;color:#e0e8ed}.missing small{color:#a8b9c5;line-height:1.6}.stage-empty{grid-column:1/-1}
  .layer-selectors{flex:0 0 auto;padding:12px 16px;display:flex;flex-direction:column;gap:9px;border-top:1px solid #33434e;background:#121c23;max-height:225px;overflow:auto}.layer-group{display:flex;align-items:center;flex-wrap:wrap;gap:6px}.layer-group>span{width:84px;font-size:11px;color:#adbfcb;flex-shrink:0}.layer-group button{display:flex;align-items:center;gap:6px;font-size:12px;min-height:32px;padding:5px 10px;background:#1b2933;color:#d4dfe6;border:1px solid #455966;border-radius:5px}.layer-group button:hover{border-color:#a5b9c6}.layer-group button.on{background:#d8ff55;border-color:#d8ff55;color:#162000}.layer-group button.unavailable:not(.on){border-style:dashed;color:#a8bac7}.layer-group button small{font-size:10px;color:inherit}.mask-key{font-size:11px;color:#bfccd5}
  @media(max-width:700px){.viewer-tabs{padding:8px 12px;gap:3px}.viewer-tabs button{min-height:36px;padding:6px 9px}.layer-count{display:none}.mobile-layer-picker{display:flex;flex:0 0 100%;margin-top:5px}.mobile-layer-picker select{font-size:13px;padding:5px 8px;min-height:34px}.layer-stage{grid-template-columns:minmax(0,1fr);padding:10px;gap:8px}figure:not(.focused){display:none}.layer-selectors{padding:8px 10px;max-height:150px}.layer-group>span{width:100%}.layer-group button{min-height:40px}.layer-stage .missing{padding:14px}}
  @media(pointer:coarse){.viewer-tabs button,.layer-group button{min-height:44px}}
</style>
