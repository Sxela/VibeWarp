<script>
  import {onMount} from 'svelte';
  let {src, title='Image', onclose}=$props();
  let dialog, closeButton;
  onMount(()=>{dialog.showModal?.();closeButton?.focus();});
</script>
<dialog bind:this={dialog} aria-label={title} oncancel={(event)=>{event.preventDefault();onclose();}} onclick={(event)=>{if(event.target===dialog)onclose();}}>
  <div class="zoom-inner"><header><strong>{title}</strong><button bind:this={closeButton} onclick={onclose} aria-label="Close zoom">Close</button></header><div class="zoom-image"><img {src} alt={title}/></div></div>
</dialog>
<style>
  dialog{width:min(1280px,96vw);height:92dvh;border:1px solid #566c7c;border-radius:8px;background:#0d161d;color:#eef3f5;padding:0;max-height:96dvh}dialog::backdrop{background:#03070be8}.zoom-inner{display:flex;flex-direction:column;height:100%;padding:16px;gap:15px}header{display:flex;align-items:center;justify-content:space-between;gap:20px;flex-shrink:0}strong{font-size:14px;font-weight:500}header button{background:#23323d;color:#e8f0f5;border:1px solid #5b7282;border-radius:5px;padding:8px 12px;cursor:pointer;min-height:40px}.zoom-image{display:flex;align-items:center;justify-content:center;flex:1;min-height:0}.zoom-image img{width:100%;height:100%;object-fit:contain}
</style>
