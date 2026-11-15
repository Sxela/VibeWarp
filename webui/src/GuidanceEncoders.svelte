<script>
  import HelpTip from './HelpTip.svelte';
  let {value=[],choices=[],onchange,error=''}=$props();
  function toggle(name,checked){onchange(checked ? [...value,name] : value.filter(v=>v!==name));}
</script>
<fieldset id="field-disco-encoders">
  <legend>Guidance encoders <HelpTip label="Help: Guidance encoders" text="Combine classic CLIP encoders for the Disco look, or compare SigLIP 2. Each selected encoder adds guidance work and VRAM."/></legend>
  <div>
    {#each choices as name}
      <label><input type="checkbox" checked={value.includes(name)} onchange={(e)=>toggle(name,e.target.checked)}/>{name}</label>
    {/each}
  </div>
  {#if error}<small>{error}</small>{/if}
</fieldset>
<style>
  fieldset{border:0;padding:0;margin:0;grid-column:1/-1}legend{color:var(--muted);font-size:var(--text-label)}
  div{display:flex;gap:.7rem;flex-wrap:wrap}label{font-size:var(--text-label);display:flex;gap:.4rem;align-items:center}small{color:var(--error)}
</style>
