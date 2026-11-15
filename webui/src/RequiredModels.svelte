<script>
  let {files=[],loading=false,busy=false,error='',ondownload}=$props();
  let expanded=$state(true);
  let missing=$derived(files.filter(file=>!['present','ready'].includes(file.status)));
</script>
{#if loading||error||missing.length}
<section class="required-models" aria-label="Required models">
  <details bind:open={expanded}>
  <summary>Required models · {missing.length}</summary>
  {#if loading}<p role="status">Checking local model files…</p>{/if}
  <ul>
    {#each missing as file (file.path)}
      <li>
        <div><strong>{file.name}</strong><span>{file.status==='present'?'Present':'Missing'}</span></div>
        <small>{file.path}</small>
        {#if file.status!=='present' && !file.downloadable}<small>Choose an existing custom file.</small>{/if}
      </li>
    {/each}
  </ul>
  {#if error}<p class="error">{error}</p>{/if}
  </details>
  <button disabled={loading||busy||!missing.some(file=>file.downloadable)} onclick={ondownload}>
    {busy?'Downloading models…':'Download required models'}
  </button>
</section>
{/if}
<style>
  .required-models{margin:1rem 0}summary{cursor:pointer;font-weight:600;font-size:.85rem}ul{list-style:none;padding:0;margin:0}
  li{margin:.7rem 0}li div{display:flex;justify-content:space-between;gap:.7rem;font-size:.8rem}
  strong,small{overflow-wrap:anywhere}small,p,span{color:var(--muted,#999);font-size:.72rem}
  small{display:block;margin-top:.2rem}button{width:100%;margin-top:.5rem}.error{color:#ed9393}
</style>
