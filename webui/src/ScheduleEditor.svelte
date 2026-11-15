<script>
  import HelpTip from './HelpTip.svelte';
  import {canonicalSchedule} from './schedules.js';
  let {name, label='', value, onchange, kind='number', axis='frame', valueLabel='Value', valueType, interpolation='blend', hint=''}=$props();
  let transfer=$state(null), transferError=$state(''), rawInvalid=$state(false);
  $effect(()=>{value;transfer=null;transferError='';rawInvalid=false;});
  const options=()=>({kind,axis,valueType});
  const blank=()=>kind==='text'?'':0;
  let parsed=$derived.by(()=>{try{return {value:canonicalSchedule(value,options()),valid:true};}catch{return {value,valid:false};}});
  let normalized=$derived(parsed.value);
  let mode=$derived(!parsed.valid||normalized!=null&&typeof normalized==='object'&&Object.values(normalized).some(v=>kind==='number'&&typeof v!=='number')?'json':'keyframes');
  function push(next){onchange(next);}
  async function copyJson(){
    const text=JSON.stringify(normalized??{0:blank()},null,2);
    transferError='';
    try{await navigator.clipboard.writeText(text);}catch{transfer={mode:'copy',text};}
  }
  function pasteJson(text){
    try{push(canonicalSchedule(JSON.parse(text),options()));transfer=null;transferError='';}
    catch(error){transferError=error.message;}
  }
  async function pasteClipboard(){
    transferError='';
    try{const text=await navigator.clipboard.readText();pasteJson(text);if(transferError)transfer={mode:'paste',text};}
    catch{transfer={mode:'paste',text:''};}
  }

  // All simple representations share frame/step/progress-keyed chips.
  let chips = $derived.by(() => {
    if (mode === 'keyframes' && normalized && typeof normalized === 'object')
      return Object.entries(normalized).map(([k, v]) => ({key: Number(k), value: v})).sort((a, b) => a.key - b.key);
    if (mode === 'keyframes') {
      let first = Array.isArray(value) ? (value[0] ?? blank())
        : (value && typeof value === 'object') ? (Object.values(value)[0] ?? blank())
        : (value ?? blank());
      return [{key: 0, value: first}];
    }
    return [];
  });

  const num = (raw, fallback = 0) => {
    let n = Number(raw);
    return Number.isFinite(n) ? n : fallback;
  };
  // A prompt frame holds ONE string: text_prompts is Dict[int, str], and multi-prompt
  // blending lives INSIDE it (`a:0.7 | b:0.3`), not in a list. Emitting a list fails
  // validation outright — "config.text_prompts.0 must be str".
  // Read defensively though: WarpFusion settings files store {"0": ["prompt"]}, so a list
  // can still arrive here from an import.
  const readText = (v) => Array.isArray(v) ? v.join(' | ') : (v ?? '');
  const coerce = (raw, fallback) => kind === 'text' ? String(raw) : num(raw, fallback);

  function emit(items){
    let out = {};
    for (let chip of items) out[chip.key] = chip.value;
    push(out);
  }

  function setValue(index, raw){
    let items = chips.map((c, i) => i === index ? {...c, value: coerce(raw, c.value)} : c);
    emit(items);
  }
  function setFrame(index, raw){
    // Keyframes only. Collisions would silently drop a chip, so block them.
    let frame = axis === 'progress' ? Math.max(0,Math.min(100,num(raw,chips[index].key)))
      : Math.max(0, Math.round(num(raw, chips[index].key)));
    if (chips.some((c, i) => i !== index && c.key === frame)) return;
    emit(chips.map((c, i) => i === index ? {...c, key: frame} : c));
  }
  function add(){
    let last = chips.at(-1);
    let next = {key:last?last.key+10:0,value:last?.value??blank()};
    if(axis==='progress'){
      next.key=Math.min(100,next.key);
      if(chips.some(chip=>chip.key===next.key)){
        const keys=[...new Set([0,...chips.map(chip=>chip.key),100])].sort((a,b)=>a-b);
        const gaps=keys.slice(1).map((key,index)=>[key-keys[index],(key+keys[index])/2]);
        next.key=gaps.sort((a,b)=>b[0]-a[0])[0]?.[1]??0;
      }
    }
    emit([...chips, next]);
  }
  function remove(index){
    if (chips.length <= 1) return;   // a schedule always has at least one value
    emit(chips.filter((_, i) => i !== index));
  }

  function applyRaw(e){
    try{push(canonicalSchedule(JSON.parse(e.target.value),options()));rawInvalid=false;transferError='';}
    catch(error){rawInvalid=true;transferError=error.message;}
  }
  let title=$derived(label||name.replaceAll('_',' ').replace(/\b\w/g,c=>c.toUpperCase()));
  const stepHelp = 'Each key is a sampling step and each value is a percentage of the final resolution. The minimum-size setting can raise that percentage so the longest side never falls below its pixel floor. Below 100%, the UNet evaluates one downscaled whole frame with no tiles. If Tiled Sampler is enabled, tiling runs only on steps whose scheduled value is 100%; otherwise those steps use a normal whole-frame evaluation. Multiscale remains active when Tiled Sampler is off. Values hold until the next keyed step.';

  // A sparkline of the schedule. `0:7, 20:9, 40:5` tells you nothing at a glance; the shape
  // does. Progress charts always cover 0–100%, including the final held value.
  // Other numeric charts need multiple points; prompts have no magnitude.
  let spark = $derived.by(() => {
    if (kind === 'text' || mode === 'json' || (axis!=='progress'&&chips.length<2)) return null;
    let points = chips.map(c => ({x: Number(c.key), y: Number(c.value)}))
                      .filter(p => Number.isFinite(p.x) && Number.isFinite(p.y)
                        && (axis!=='progress'||p.x>=0&&p.x<=100));
    if (!points.length || (axis!=='progress'&&points.length<2)) return null;
    const plotted=[...points];
    if(axis==='progress'){
      if(plotted[0].x>0)plotted.unshift({x:0,y:0});
      if(plotted.at(-1).x<100)plotted.push({x:100,y:plotted.at(-1).y});
    }
    let xs = plotted.map(p => p.x), ys = plotted.map(p => p.y);
    let x0 = axis==='progress'?0:Math.min(...xs), x1 = axis==='progress'?100:Math.max(...xs);
    let y0 = Math.min(...ys), y1 = Math.max(...ys);
    const W = 100, H = 22;
    // A flat schedule would divide by zero; pin it to the middle instead.
    const sx = (x) => x1 === x0 ? 0 : ((x - x0) / (x1 - x0)) * W;
    const sy = (y) => y1 === y0 ? H / 2 : H - ((y - y0) / (y1 - y0)) * H;
    return {
      d: plotted.map((p, i) => (axis==='progress'||axis==='step'||interpolation==='hold')&&i
        ? `H${sx(p.x).toFixed(1)}V${sy(p.y).toFixed(1)}`
        : `${i ? 'L' : 'M'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' '),
      dots: points.map(p => ({cx: sx(p.x), cy: sy(p.y)})),
      lo: Math.min(...ys), hi: Math.max(...ys), W, H,
    };
  });

  let help=$derived(hint|| (axis==='progress'
    ? `Inference progress (%) → ${valueLabel.toLowerCase()}. Executed steps run from 0% to 100%; one step uses 0%. Restarts after skipped steps. Values hold until the next key; no interpolation.`
    :axis==='step'?stepHelp:kind==='text'
    ? 'Frame → prompt. Supports a:0.7 | b:0.3 blending and <lora:name:weight>.'
    :interpolation==='hold'?'Frame → value. Values hold until the next key; no interpolation.'
    :'Frame → value. A single key at frame 0 is constant. Values between keyframes are interpolated when schedule blending is on.'));
</script>

<div class="sched" class:progress={axis==='progress'} role="group" aria-label={title}>
  <div class="head">
    <span>{title}
      <HelpTip label={'Help: '+title} text={help}/>
    </span>
    {#if spark}
      <svg class="spark" viewBox={`0 0 ${spark.W} ${spark.H}`} preserveAspectRatio="none"
           role="img" aria-label={`${axis==='progress'?'Inference progress 0–100%; ':''}Schedule from ${spark.lo} to ${spark.hi}`}>
        <path d={spark.d}/>
        {#each spark.dots as dot}<circle cx={dot.cx} cy={dot.cy} r="1.6"/>{/each}
      </svg>
      <b class="range">{spark.lo} – {spark.hi}</b>
    {/if}
    <div class="json-actions">
      <button type="button" aria-label={'Copy '+title+' JSON'} title="Copy schedule JSON" onclick={copyJson}><svg aria-hidden="true" viewBox="0 0 16 16"><rect x="5" y="5" width="8" height="9" rx="1"/><path d="M10 5V2H2v9h3"/></svg></button>
      <button type="button" aria-label={'Paste '+title+' JSON'} title="Paste schedule JSON" onclick={pasteClipboard}><svg aria-hidden="true" viewBox="0 0 16 16"><path d="M5 3H3v11h10V3h-2M6 2h4v3H6zM8 7v5m-2-2 2 2 2-2"/></svg></button>
    </div>
  </div>

  {#if transfer}<div class="json-transfer"><label>{transfer.mode==='copy'?'Copy schedule JSON':'Paste schedule JSON'}<textarea bind:value={transfer.text} readonly={transfer.mode==='copy'} aria-label="Schedule JSON"></textarea></label><div><button type="button" onclick={()=>{transfer=null;transferError='';}}>{transfer.mode==='copy'?'Done':'Cancel'}</button>{#if transfer.mode==='paste'}<button type="button" onclick={()=>pasteJson(transfer.text)}>Paste keyframes</button>{/if}</div></div>{/if}
  {#if transferError}<p class="transfer-error" role="alert">{transferError}</p>{/if}
  {#if mode === 'json'}
    <label class="advanced-json"><span>Advanced schedule JSON <HelpTip label="Help: Advanced schedule JSON" text="This representation includes values that chips cannot express. It is preserved as JSON."/></span><textarea class:invalid={rawInvalid} value={JSON.stringify(value,null,2)} onchange={applyRaw} aria-label="Advanced schedule JSON"></textarea></label>
  {:else}
    <div class="chips" class:text={kind === 'text'}>
      {#each chips as chip, index (index)}
        <span class="chip" class:text={kind === 'text'}>
            <input class="frame" type="number" min="0" max={axis==='progress'?100:undefined} step={axis==='progress'?'any':'1'} value={chip.key}
                   onchange={(e)=>setFrame(index, e.target.value)}
                   aria-label={axis === 'progress' ? 'Inference progress (%)' : axis === 'step' ? 'Step' : 'Frame'}/>
            {#if axis==='progress'}<small>%</small>{/if}
            <i>:</i>
          {#if kind === 'text'}
            <textarea class="prompt" rows="2" value={readText(chip.value)}
                      onchange={(e)=>setValue(index, e.target.value)}
                      placeholder="a portrait, comic style" aria-label="Prompt"></textarea>
          {:else}
            <input class="val" style:width={`${Math.min(18,Math.max(6,String(chip.value).length+1))}ch`} type="number" step={valueLabel==='Cut count per batch'?1:'any'} min={axis==='progress'?0:undefined} max={valueLabel==='Grayscale probability'?1:undefined} value={chip.value}
                   onchange={(e)=>setValue(index, e.target.value)} aria-label={valueLabel}/>
          {/if}
          {#if chips.length > 1}
            <button class="x" onclick={()=>remove(index)} aria-label="Remove">×</button>
          {/if}
        </span>
      {/each}
        <button class="add" onclick={add}>+
          {axis === 'step' ? 'step' : 'keyframe'}</button>
    </div>

  {/if}
</div>

<style>
  .sched{grid-column:1/-1;display:flex;flex-direction:column;gap:9px}
  .head{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}
  .head>span{color:var(--muted);font-size:var(--text-label)}
  .head{position:relative}
  .spark{width:96px;height:22px;margin-left:auto;overflow:visible}
  .spark path{fill:none;stroke:#8ea834;stroke-width:1.4;vector-effect:non-scaling-stroke;stroke-linejoin:round}
  .spark circle{fill:#d8ff55}
  .range{color:var(--muted);font:var(--text-value) var(--font-mono);font-weight:400}

  .chips{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:10px;border:1px solid #303640;border-radius:8px;background:#0c0f13;min-height:44px}
  .chip{display:inline-flex;align-items:center;gap:3px;background:#191d23;border:1px solid #343a43;border-radius:20px;padding:3px 4px 3px 9px}
  .chip i{color:var(--muted);font-style:normal}
  .chip input{border:0;background:transparent;color:#eef0f2;font:var(--text-value) var(--font-mono);outline:none;padding:2px 0;-moz-appearance:textfield}
  .chip input::-webkit-outer-spin-button,.chip input::-webkit-inner-spin-button{-webkit-appearance:none;margin:0}
  .chip .frame{width:34px;color:var(--accent-muted);text-align:right}
  .progress .chip .frame{width:82px}
  .progress .chip>small{font-size:var(--text-help);color:#a7b4bc}
  .chip .val{width:52px}
  .chip input:focus{color:#d8ff55}
  .x{border:0;background:none;color:var(--muted);cursor:pointer;font-size:14px;line-height:1;padding:0 5px;border-radius:50%}
  .x:hover{color:var(--error)}
  .add{border:1px dashed #3a414b;background:none;color:var(--muted);border-radius:20px;padding:5px 12px;font-size:var(--text-help);cursor:pointer}
  /* Prompts are long: stack them full-width instead of wrapping as inline pills. */
  .chips.text{flex-direction:column;align-items:stretch}
  .chip.text{border-radius:10px;padding:6px 6px 6px 9px;align-items:flex-start;width:100%}
  .chip.text .frame{margin-top:5px}
  .chip.text i{margin-top:5px}
  .chip .prompt{flex:1;min-width:0;border:0;background:transparent;color:#eef0f2;font:var(--text-value)/1.5 var(--font-ui);outline:none;resize:vertical;padding:3px 6px}
  .chip .prompt:focus{color:#fff}
  .add:hover{border-color:var(--accent-muted);color:#d8ff55}

  textarea.invalid{border-color:var(--error)}
  .json-actions{display:flex;gap:4px;margin-left:auto}.json-actions button{width:30px;min-height:30px;padding:6px;background:transparent;border-color:transparent;color:var(--muted)}.json-actions svg{width:16px;height:16px;fill:none;stroke:currentColor;stroke-width:1.3}
  .json-transfer{padding:12px;border:1px solid var(--line);border-radius:var(--radius)}.json-transfer textarea,.advanced-json textarea{font-family:var(--font-mono);min-height:90px}.json-transfer>div{display:flex;justify-content:flex-end;gap:6px;margin-top:8px}.transfer-error{color:var(--error);font-size:var(--text-help)}
</style>
