<script>
  let {files=[], title='Model downloads',onaction} = $props();
  let expanded=$state(true);
  let pending=$derived(files.filter(file=>!['ready','present'].includes(file.status)));
  let queued=$derived(files.filter(file=>file.status==='queued').length);
  let active=$derived(files.filter(file=>['starting','downloading','waiting','verifying','retrying','pausing','cancelling'].includes(file.status)).length);
  const bytes = value => {
    if(value == null) return 'unknown';
    if(value < 1024) return `${value} B`;
    return `${(value / 1024 ** (value >= 1024**3 ? 3 : 2)).toFixed(1)} ${value >= 1024**3 ? 'GB' : 'MB'}`;
  };
</script>
{#if pending.length}
  <details class="downloads" aria-label={title} bind:open={expanded}>
    <summary>{title} · {active} active · {queued} queued</summary>
    {#each pending as file (file.id||file.path)}
      <div class="download">
        <div><strong>{file.name}</strong><span>{file.status}</span></div>
        {#if file.active_file}<small>{file.active_file}</small>{/if}
        {#if file.total > 0}
          <progress aria-label={`Download ${file.name}`} max={file.total} value={Math.min(file.downloaded,file.total)}></progress>
        {:else if ['queued','waiting','downloading','verifying','retrying'].includes(file.status)}
          <progress aria-label={`Download ${file.name}`}></progress>
        {/if}
        <small>{bytes(file.downloaded)}{file.total ? ` / ${bytes(file.total)}` : ''}{file.speed > 0 ? ` · ${bytes(file.speed)}/s` : ''}</small>
        {#if file.error}<small class="error">{file.error}</small>{/if}
        {#if onaction}
          <div class="actions">
            {#if file.can_pause}<button aria-label={`Pause ${file.name}`} onclick={()=>onaction(file,'pause')}>Pause</button>{/if}
            {#if file.can_resume}<button aria-label={`Resume ${file.name}`} onclick={()=>onaction(file,'resume')}>{file.status==='paused'?'Resume':'Retry'}</button>{/if}
            {#if file.can_cancel}<button aria-label={`Cancel ${file.name}`} onclick={()=>onaction(file,'cancel')}>Cancel</button>{/if}
          </div>
        {/if}
      </div>
    {/each}
  </details>
{/if}
<style>
  .downloads { margin: 1rem 0; }
  summary { cursor:pointer; font-weight:600; font-size:.85rem; }
  .actions {justify-content:flex-end!important;margin-top:.3rem}
  button {font-size:.72rem;padding:.25rem .55rem}
  .download { margin: .7rem 0; }
  .download div { display: flex; justify-content: space-between; gap: .5rem; font-size: .8rem; }
  strong { overflow-wrap: anywhere; }
  span,small { color: var(--muted,#999); }
  progress { width: 100%; height: .5rem; accent-color: var(--accent,#baea8d); display: block; margin: .35rem 0; }
  small { display: block; font-size: .72rem; }
  .error { color: #ed9393; }
</style>
