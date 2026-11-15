<script>
  let {config,job,onjob}= $props();
  let frame=$state(0),requestedFrame=$state(0),reference=$state(''),pending=$state(false),error=$state(''),id=$state('');
  let current=$derived(job?.id===id?job:null);
  let active=$derived(pending||['queued','running'].includes(current?.state));
  async function preview(){
    pending=true;error='';id='';
    try{
      const response=await fetch('/api/video/raw-preview',{method:'POST',headers:{'content-type':'application/json'},
        body:JSON.stringify({config,frame,reference_path:reference||config.video?.first_frame_path||''})});
      const next=await response.json();
      if(!response.ok){
        const detail=Array.isArray(next.detail)?next.detail.map(item=>[item.path,item.message||item.msg].filter(Boolean).join(': ')).join('; '):next.detail;
        throw new Error(typeof detail==='string'&&detail?detail:'Could not prepare preview');
      }
      id=next.id;requestedFrame=frame;onjob(next);
    }catch(reason){error=reason.message;}finally{pending=false;}
  }
</script>
<section aria-label="Raw video preparation preview" class="raw-preview">
  <label>Preview frame<input aria-label="Raw preview frame" type="number" min="0" step="1" bind:value={frame}/></label>
  {#if config.mask?.raw_color_enabled}<label>Stylized reference image for preview<input aria-label="Raw preview reference image" type="text" bind:value={reference} placeholder={config.video?.first_frame_path||'Path to a stylized image'}/></label>{/if}
  <button onclick={preview} disabled={active||(!config.mask?.rvm_enabled&&!config.mask?.use_background_mask&&!config.mask?.raw_color_enabled)}>{active?'Preparing preview…':'Preview one frame'}</button>
  <small>Uses the shared video cache. Color changes affect render inputs only.</small>
  {#if error||current?.error}<p role="alert">{error||current.error}</p>{/if}
  {#if !error&&!pending&&current?.state==='completed'&&current.preview_available}<img src={`/api/jobs/${id}/preview?v=${current.revision}`} alt={`Prepared raw video frame ${requestedFrame}`}/>{/if}
</section>
<style>
  .raw-preview{display:flex;flex-direction:column;gap:12px;border-top:1px solid #344651;padding-top:16px;margin-top:16px}.raw-preview label{display:flex;flex-direction:column;gap:7px;font-size:13px}.raw-preview img{width:100%;max-height:440px;object-fit:contain;background:#080e13}.raw-preview small{color:#aebfc9;font-size:12px;line-height:1.5}.raw-preview p{color:#ffc4cc;font-size:13px}.raw-preview button{align-self:flex-start;min-height:40px}
</style>
