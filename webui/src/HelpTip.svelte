<script>
  import {tick,onMount,onDestroy} from 'svelte';
  let {label='Help',text='',children}= $props();
  const id=$props.id();
  let open=$state(false), button=$state(null), tip=$state(null), left=$state(8), top=$state(8);
  let hideTimer;
  const keepOpen=()=>clearTimeout(hideTimer);
  const hideSoon=()=>{hideTimer=setTimeout(()=>open=false,160);};
  onDestroy(()=>clearTimeout(hideTimer));
  onMount(()=>{const close=()=>open=false;document.addEventListener('scroll',close,true);return()=>document.removeEventListener('scroll',close,true);});
  function portal(node){document.body.append(node);return {destroy(){node.remove();}};}
  async function show(){
    keepOpen();open=true;await tick();if(!button||!tip)return;
    const rect=button.getBoundingClientRect();
    left=Math.max(8,Math.min(window.innerWidth-tip.offsetWidth-8,rect.left));
    top=rect.bottom+8+tip.offsetHeight<window.innerHeight?rect.bottom+8:Math.max(8,rect.top-tip.offsetHeight-8);
  }
</script>
<svelte:window onkeydown={event=>{if(event.key==='Escape')open=false;}} onresize={()=>open=false} onscroll={()=>open=false}/>
<span class="help-tip" role="presentation" onmouseleave={hideSoon}>
  <span role="button" tabindex="0" bind:this={button} class="help-trigger" aria-label={label} aria-describedby={open?id:undefined}
    onmouseenter={show} onfocus={show} onblur={()=>open=false}
    onkeydown={event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();event.stopPropagation();if(open)open=false;else show();}}}
    onclick={event=>{event.preventDefault();event.stopPropagation();if(open)open=false;else show();}}><span aria-hidden="true">?</span></span>
  {#if open}<span use:portal bind:this={tip} {id} role="tooltip" class="help-content" style={`left:${left}px;top:${top}px`}
    onmouseenter={keepOpen} onmouseleave={hideSoon}>{#if children}{@render children()}{:else}{text}{/if}</span>{/if}
</span>
<style>
  .help-tip{display:inline-flex;vertical-align:middle;margin-left:5px}
  .help-trigger{display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;min-height:22px;padding:0;border:1px solid var(--line);border-radius:50%;color:var(--muted);background:transparent;font:600 12px/1 var(--font-ui);cursor:help}
  .help-content{position:fixed;z-index:1000;max-width:min(370px,calc(100vw - 16px));padding:10px 12px;border:1px solid var(--input-border);border-radius:var(--radius);background:var(--surface);color:var(--ink);font:var(--text-help)/1.6 var(--font-ui);box-shadow:0 8px 24px #0007;white-space:normal;text-align:left}
  .help-content :global(code){color:var(--accent)}
</style>
