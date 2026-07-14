<script>
  // Schedules as editable chips instead of hand-written JSON.
  //
  // The three shapes mean DIFFERENT things (utils/scheduling.py::get_scheduled_arg):
  //   scalar / [v]      constant
  //   [v0, v1, v2]      indexed BY FRAME; frames past the end clamp to the last value
  //   {frame: value}    keyframes; interpolated between when blend_json_schedules is on
  // So the editor never silently converts between them — changing the mode is an
  // explicit act. Anything it can't model (e.g. the [[3,7,3]] per-step CFG ramp)
  // falls back to raw JSON rather than being mangled.
  // kind: 'number' (default) | 'text' — text is for prompt schedules, whose values are
  // strings ({frame: [prompt]}) rather than numbers.
  let { name, value, onchange, kind = 'number' } = $props();

  let raw = $state('');
  let rawInvalid = $state(false);

  const isNested = (v) => Array.isArray(v) && v.some(item => Array.isArray(item) || item !== null && typeof item === 'object');
  const blank = () => kind === 'text' ? '' : 0;

  // The shape the CURRENT value implies. There is no 'off': the notebook has no separate
  // scalar setting — `steps = get_scheduled_arg(frame, steps_schedule)`, and a one-element
  // list IS the constant. So a null schedule renders as a constant on the fallback scalar,
  // and we always emit a schedule.
  let shape = $derived(
    kind === 'text' ? 'keyframes'
    : value === null || value === undefined ? 'constant'
    : isNested(value) ? 'json'
    : Array.isArray(value) ? (value.length <= 1 ? 'constant' : 'list')
    : typeof value === 'object' ? 'keyframes'
    : 'constant'
  );

  // The mode the USER picked, which overrides the inferred shape.
  //
  // This state is why Per-frame and JSON used to be dead buttons: mode was derived purely
  // from the value, so switching a single-chip schedule to Per-frame emitted `[v]` — length
  // 1 — which the derivation read straight back as 'constant'. And JSON re-emitted the value
  // it already had. Both looked like no-ops.
  let picked = $state(null);
  let internal = false;
  $effect(() => {
    value;                      // track
    // Our OWN writes must not clear the picked mode (emitting [v] in Per-frame mode would
    // otherwise infer 'constant' and snap the UI back). A value replaced from OUTSIDE
    // (settings load) should render in its natural shape, so clear it then.
    if (internal) { internal = false; return; }
    picked = null;
  });
  let mode = $derived(picked ?? shape);

  // Every write goes through this, so the effect above can tell ours from the parent's.
  function push(next){ internal = true; onchange(next); }

  // Chips: [{key, value}] — key is the frame for keyframes, the index for lists.
  let chips = $derived.by(() => {
    if (mode === 'keyframes' && value && !Array.isArray(value) && typeof value === 'object')
      return Object.entries(value).map(([k, v]) => ({key: Number(k), value: v})).sort((a, b) => a.key - b.key);
    if (mode === 'list' && Array.isArray(value))
      return value.map((v, i) => ({key: i, value: v}));
    if (mode === 'constant' || mode === 'keyframes' || mode === 'list') {
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
    if (mode === 'keyframes') {
      let out = {};
      for (let chip of items) out[chip.key] = chip.value;
      push(out);
    } else push(items.map(c => c.value));
  }

  function setValue(index, raw){
    let items = chips.map((c, i) => i === index ? {...c, value: coerce(raw, c.value)} : c);
    emit(items);
  }
  function setFrame(index, raw){
    // Keyframes only. Collisions would silently drop a chip, so block them.
    let frame = Math.max(0, Math.round(num(raw, chips[index].key)));
    if (chips.some((c, i) => i !== index && c.key === frame)) return;
    emit(chips.map((c, i) => i === index ? {...c, key: frame} : c));
  }
  function add(){
    let last = chips.at(-1);
    let next = mode === 'keyframes'
      ? {key: (last?.key ?? -1) + 10, value: last?.value ?? blank()}
      : {key: chips.length, value: last?.value ?? blank()};
    emit([...chips, next]);
  }
  function remove(index){
    if (chips.length <= 1) return;   // a schedule always has at least one value
    emit(chips.filter((_, i) => i !== index));
  }

  function setMode(next){
    if (next === mode) return;
    // Snapshot the chips BEFORE touching `picked`: chips is derived from the mode, so
    // setting picked first collapses it to a single fallback chip and the conversion below
    // silently drops every value but the first.
    const from = chips;
    const fromKeyframes = mode === 'keyframes';
    const values = from.length ? from.map(c => c.value) : [blank()];
    picked = next;                   // hold the choice; the value alone can't express it
    if (next === 'constant') push([values[0]]);
    else if (next === 'list') push(values);
    else if (next === 'keyframes') {
      // From a per-frame list, index == frame, so this conversion is lossless.
      let out = {};
      from.forEach((c, i) => { out[fromKeyframes ? c.key : i] = c.value; });
      push(out);
    } else if (next === 'json') {
      raw = JSON.stringify(value ?? values);
      rawInvalid = false;
    }
  }

  function applyRaw(e){
    try {
      push(JSON.parse(e.target.value));
      rawInvalid = false;
    } catch { rawInvalid = true; }
  }

  let title = $derived(name.replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase()));

  // A sparkline of the schedule. `0:7, 20:9, 40:5` tells you nothing at a glance; the shape
  // does. Only for numeric multi-point schedules — a constant is a flat line (pointless) and
  // prompts have no magnitude.
  let spark = $derived.by(() => {
    if (kind === 'text' || mode === 'json' || chips.length < 2) return null;
    let points = chips.map(c => ({x: Number(c.key), y: Number(c.value)}))
                      .filter(p => Number.isFinite(p.x) && Number.isFinite(p.y));
    if (points.length < 2) return null;
    let xs = points.map(p => p.x), ys = points.map(p => p.y);
    let x0 = Math.min(...xs), x1 = Math.max(...xs);
    let y0 = Math.min(...ys), y1 = Math.max(...ys);
    const W = 100, H = 22;
    // A flat schedule would divide by zero; pin it to the middle instead.
    const sx = (x) => x1 === x0 ? 0 : ((x - x0) / (x1 - x0)) * W;
    const sy = (y) => y1 === y0 ? H / 2 : H - ((y - y0) / (y1 - y0)) * H;
    return {
      d: points.map((p, i) => `${i ? 'L' : 'M'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' '),
      dots: points.map(p => ({cx: sx(p.x), cy: sy(p.y)})),
      lo: Math.min(...ys), hi: Math.max(...ys), W, H,
    };
  });

  // No 'Off': a schedule is always a schedule (see the shape comment above).
  //
  // Prompts are {frame: [text, ...]} — a dict of LISTS. Per-frame (index-keyed) is
  // meaningless for them, and a lone keyframe at frame 0 already IS the constant, so
  // they get Keyframes + JSON only. That also keeps us from ever coercing the list
  // into a bare scalar.
  let modes = $derived(kind === 'text'
    ? [['keyframes','Keyframes'],['json','JSON']]
    : [['constant','Constant'],['list','Per-frame'],['keyframes','Keyframes'],['json','JSON']]);
</script>

<div class="sched">
  <div class="head">
    <span>{title}</span>
    {#if spark}
      <svg class="spark" viewBox={`0 0 ${spark.W} ${spark.H}`} preserveAspectRatio="none"
           role="img" aria-label={`Schedule from ${spark.lo} to ${spark.hi}`}>
        <path d={spark.d}/>
        {#each spark.dots as dot}<circle cx={dot.cx} cy={dot.cy} r="1.6"/>{/each}
      </svg>
      <b class="range">{spark.lo} – {spark.hi}</b>
    {/if}
    <div class="modes">
      {#each modes as [id, label]}
        <button class:on={mode === id} onclick={()=>setMode(id)}>{label}</button>
      {/each}
    </div>
  </div>

  {#if mode === 'json'}
    <input class:invalid={rawInvalid} value={raw || JSON.stringify(value)} onchange={applyRaw}
           placeholder="[[3, 7, 3]]"/>
    <p class="hint">Advanced form (e.g. the <code>[[low, high, low]]</code> per-step CFG ramp). Chips can't represent it.</p>
  {:else}
    <div class="chips" class:text={kind === 'text'}>
      {#each chips as chip, index (index)}
        <span class="chip" class:text={kind === 'text'}>
          {#if mode === 'keyframes'}
            <input class="frame" type="number" min="0" step="1" value={chip.key}
                   onchange={(e)=>setFrame(index, e.target.value)} aria-label="Frame"/>
            <i>:</i>
          {:else if mode === 'list'}
            <b>{index}</b><i>:</i>
          {/if}
          {#if kind === 'text'}
            <textarea class="prompt" rows="2" value={readText(chip.value)}
                      onchange={(e)=>setValue(index, e.target.value)}
                      placeholder="a portrait, comic style" aria-label="Prompt"></textarea>
          {:else}
            <input class="val" type="number" step="any" value={chip.value}
                   onchange={(e)=>setValue(index, e.target.value)} aria-label="Value"/>
          {/if}
          {#if chips.length > 1}
            <button class="x" onclick={()=>remove(index)} aria-label="Remove">×</button>
          {/if}
        </span>
      {/each}
      {#if mode !== 'constant'}
        <button class="add" onclick={add}>+ {mode === 'keyframes' ? 'keyframe' : 'frame'}</button>
      {/if}
    </div>
    <p class="hint">
      {#if mode === 'constant'}Same {kind === 'text' ? 'prompt' : 'value'} on every frame.
      {:else if mode === 'list'}One value per frame, indexed from 0. Frames past the last entry reuse it.
      {:else if kind === 'text'}Frame → prompt. Supports <code>a:0.7 | b:0.3</code> blending and <code>&lt;lora:name:weight&gt;</code>.
      {:else}Frame → value. Values between keyframes are interpolated when schedule blending is on.
      {/if}
    </p>
  {/if}
</div>

<style>
  .sched{grid-column:1/-1;display:flex;flex-direction:column;gap:9px}
  .head{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}
  .head>span{color:#aeb4bd;font-size:12px}
  .head{position:relative}
  .spark{width:96px;height:22px;margin-left:auto;overflow:visible}
  .spark path{fill:none;stroke:#8ea834;stroke-width:1.4;vector-effect:non-scaling-stroke;stroke-linejoin:round}
  .spark circle{fill:#d8ff55}
  .range{color:#6f7883;font:11px Consolas,monospace;font-weight:400}
  .modes{display:flex;gap:4px}
  .modes button{border:1px solid #303640;background:#12151a;color:#8b929c;border-radius:7px;padding:5px 10px;font-size:11px;cursor:pointer}
  .modes button:hover{color:#e7eaee;border-color:#4b525b}
  .modes button.on{background:#d8ff55;border-color:#d8ff55;color:#111;font-weight:600}

  .chips{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:10px;border:1px solid #303640;border-radius:8px;background:#0c0f13;min-height:44px}
  .chip{display:inline-flex;align-items:center;gap:3px;background:#191d23;border:1px solid #343a43;border-radius:20px;padding:3px 4px 3px 9px}
  .chip i{color:#5f6771;font-style:normal}
  .chip b{color:#8ea834;font:600 11px Consolas,monospace;min-width:14px;text-align:right}
  .chip input{border:0;background:transparent;color:#eef0f2;font:12px Consolas,monospace;outline:none;padding:2px 0;-moz-appearance:textfield}
  .chip input::-webkit-outer-spin-button,.chip input::-webkit-inner-spin-button{-webkit-appearance:none;margin:0}
  .chip .frame{width:34px;color:#8ea834;text-align:right}
  .chip .val{width:52px}
  .chip input:focus{color:#d8ff55}
  .x{border:0;background:none;color:#68717d;cursor:pointer;font-size:14px;line-height:1;padding:0 5px;border-radius:50%}
  .x:hover{color:#f18b91}
  .add{border:1px dashed #3a414b;background:none;color:#8b929c;border-radius:20px;padding:5px 12px;font-size:11px;cursor:pointer}
  /* Prompts are long: stack them full-width instead of wrapping as inline pills. */
  .chips.text{flex-direction:column;align-items:stretch}
  .chip.text{border-radius:10px;padding:6px 6px 6px 9px;align-items:flex-start;width:100%}
  .chip.text .frame{margin-top:5px}
  .chip.text i{margin-top:5px}
  .chip .prompt{flex:1;min-width:0;border:0;background:transparent;color:#eef0f2;font:12px/1.5 inherit;outline:none;resize:vertical;padding:3px 6px}
  .chip .prompt:focus{color:#fff}
  .add:hover{border-color:#8ea834;color:#d8ff55}

  .hint{margin:0;color:#68717d;font-size:11px}
  code{color:#8ea834;font-size:11px}
  input.invalid{border-color:#f26a6a}
  .sched>input{width:100%;border:1px solid #303640;background:#0c0f13;color:#eef0f2;border-radius:8px;padding:10px 11px;outline:none;font:12px Consolas,monospace}
  .sched>input:focus{border-color:#8ea834}
</style>
