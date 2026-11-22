<script>
  import HelpTip from './HelpTip.svelte';
  import {sortedComfyModels} from './comfyModels.js';
  let {title,anchor,value='',onchange,info=null,help='',error=''}=$props();
  let suggested=$derived(info?.suggested||'');
  let available=$derived(info?.available||[]);
  let known=$derived(Array.isArray(info?.available)&&!info?.error);
  let options=$derived(sortedComfyModels(available,suggested,value));
  function label(name){
    if(name===suggested)return `${name} (Suggested${known&&!available.includes(name)?' · Missing':''})`;
    if(!available.includes(name))return `${name} (Current${known?' · Missing':''})`;
    return name;
  }
</script>
<label id={anchor} class:errored={!!error}>
  <span>{title}{#if help}<HelpTip label={'Help: '+title} text={help}/>{/if}</span>
  <select {value} aria-label={title} aria-busy={!!info?.loading} onchange={event=>onchange(event.currentTarget.value)}>
    {#if !value&&!suggested}<option value="">Choose a model</option>{/if}
    {#each options as name}<option value={name}>{label(name)}</option>{/each}
  </select>
  {#if info?.error}<small class="err">{info.error}</small>{/if}
  {#if error}<small class="err">{error}</small>{/if}
</label>
<style>
  select{width:100%;min-width:0}
  .err{color:var(--error,#ed9393);overflow-wrap:anywhere}
  .errored select{border-color:#c0505a}
</style>
