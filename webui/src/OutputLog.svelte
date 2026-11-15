<script>
  import {untrack} from 'svelte';
  let {logs=[],revision=0}=$props();
  const storageKey='vibewarp.log.height';
  function storedHeight(){try{const value=Number(localStorage.getItem(storageKey));return Number.isFinite(value)&&value>=120?value:240;}catch{return 240;}}
  let preferredHeight=$state(storedHeight()),viewportHeight=$state(window.innerHeight);
  let resizing=$state(false),startY=0,startHeight=0,activePointer=null,element=$state(null),following=$state(true);
  let maximum=$derived(Math.max(120,Math.floor(viewportHeight*.7)));
  let height=$derived(Math.max(120,Math.min(maximum,preferredHeight)));
  function setHeight(value){
    preferredHeight=Math.max(120,Math.min(maximum,value));
    try{localStorage.setItem(storageKey,String(preferredHeight));}catch{}
  }
  function start(event){
    if(event.button!==undefined&&event.button!==0)return;
    event.preventDefault();startY=event.clientY;startHeight=height;activePointer=event.pointerId;resizing=true;
    event.currentTarget.setPointerCapture?.(event.pointerId);
  }
  function move(event){if(resizing&&(activePointer==null||event.pointerId===activePointer))setHeight(startHeight+startY-event.clientY);}
  function end(event){if(activePointer==null||event.pointerId===activePointer){resizing=false;activePointer=null;}}
  function key(event){
    const next={ArrowUp:height+32,ArrowDown:height-32,Home:120,End:maximum}[event.key];
    if(next!==undefined){event.preventDefault();setHeight(next);}
  }
  $effect(()=>{
    logs;revision;height;
    if(!element||!untrack(()=>following))return;
    const frame=requestAnimationFrame(()=>{if(element)element.scrollTop=element.scrollHeight;});
    return()=>cancelAnimationFrame(frame);
  });
</script>
<svelte:window onpointermove={move} onpointerup={end} onpointercancel={end} onresize={()=>viewportHeight=window.innerHeight}/>
<section id="output-log" class="output-log" class:resizing aria-label="Output log" style:height={`${height}px`}>
  <!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions (A focusable separator resizes the adjacent pane using pointer or keyboard.) -->
  <div class="log-resizer" role="separator" tabindex="0" aria-label="Resize system log" aria-orientation="horizontal"
    aria-controls="system-log-content" aria-valuemin="120" aria-valuemax={maximum} aria-valuenow={height} aria-valuetext={`${height} pixels`}
    title="Drag to resize system log. Arrow keys resize; Home and End set limits."
    onpointerdown={start} onkeydown={key}><span></span></div>
  <pre id="system-log-content" bind:this={element} onscroll={()=>following=element.scrollHeight-element.scrollTop-element.clientHeight<24}>{logs.join('\n')||'Waiting for pipeline output…'}</pre>
</section>
<style>
  .output-log{flex:0 0 auto;min-height:120px;max-height:70dvh;display:flex;flex-direction:column;background:#0a1015;border-top:1px solid var(--line)}
  .log-resizer{flex:0 0 10px;display:flex;align-items:center;justify-content:center;cursor:row-resize;touch-action:none;background:#192228}
  .log-resizer span{width:54px;height:3px;border-radius:2px;background:var(--muted)}
  .log-resizer:hover,.log-resizer:focus-visible,.resizing .log-resizer{background:#30412b}
  .log-resizer:hover span,.resizing .log-resizer span{background:var(--accent)}
  pre{flex:1;min-height:0;overflow:auto;padding:12px 20px;margin:0;font:12px/1.6 var(--font-mono);color:#b7c5ce;white-space:pre-wrap;overflow-wrap:anywhere}
  .resizing pre{user-select:none;cursor:row-resize}
</style>
