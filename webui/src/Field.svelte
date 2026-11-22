<script>
  // `error` is the server's validation message for THIS field. It is rendered next to the
  // input rather than only in the sidebar, and the anchor id lets the sidebar scroll here.
  let { name, schema, value, onchange, path = '', hint = '', error = '', modelInfo=null } = $props();
  let help=$derived(hint||schema.help||'');
  import Field from './Field.svelte';
  import ScheduleEditor from './ScheduleEditor.svelte';
  import HelpTip from './HelpTip.svelte';
  import FrameRange from './FrameRange.svelte';
  import GuidanceEncoders from './GuidanceEncoders.svelte';
  import ComfyModelSelect from './ComfyModelSelect.svelte';
  import { isPathField, stripQuotes } from './paths.js';
  // Every *_schedule field is a keyframe/per-frame series — chips, not raw JSON.
  let isProgress = $derived(['inference_progress','denoising_progress'].includes(schema.schedule_domain));
  let isSchedule = $derived(name.endsWith('_schedule') || !!schema.schedule_domain);
  // Prompts are the same keyframe structure, only the values are text: Dict[int, str].
  // reconstruction_noise.prompts/neg_prompts are prompt schedules too (same type) — they
  // were rendering as a raw JSON blob.
  const PROMPTS = new Set(['text_prompts', 'negative_prompts', 'prompts', 'neg_prompts']);
  let isPrompts = $derived(schema.schedule_kind==='text'||PROMPTS.has(name));
  // Model reference instructions are plain text, not frame-keyed prompt schedules.
  // They still need the same roomy editor as prompts instead of a one-line input.
  const PLAIN_TEXT = new Set(['multi_reference_instruction']);
  let isPlainText = $derived(PLAIN_TEXT.has(name));
  // The backend can override the display name (ui_layout.LABELS). A field hoisted out of
  // its section needs one: `animatediff.enabled` on the Model card would just say "Enabled".
  let title = $derived(schema.label || name.replaceAll('_',' ').replace(/\b\w/g, c=>c.toUpperCase()));
  let full = $derived(path ? `${path}.${name}` : name);
  // Scroll target for "jump to the offending field" from the error list.
  let anchor = $derived(`field-${path || 'main'}-${name}`);
  let jsonText = $derived(typeof value === 'string' ? value : JSON.stringify(value ?? (schema.type==='array'?[]:{}), null, 2));
  function numberValue(e){ onchange(schema.type==='integer' ? parseInt(e.target.value||0) : parseFloat(e.target.value||0)); }

  // A typo'd path used to surface only at render time, minutes in. Check it on blur.
  // Empty is not "missing" — plenty of path fields are legitimately optional, and
  // validate_config is what decides which are required.
  let missing = $state(false);
  async function checkPath(e){
    let cleaned = stripQuotes(e.target.value);
    onchange(cleaned);
    if (!cleaned) { missing = false; return; }
    try {
      let r = await fetch(`/api/fs/exists?path=${encodeURIComponent(cleaned)}`);
      let d = await r.json();
      missing = d.checked && !d.exists;
    } catch { missing = false; }   // server unreachable is not the field's fault
  }
  function jsonValue(e){ try { onchange(JSON.parse(e.target.value)); e.target.classList.remove('invalid'); } catch { e.target.classList.add('invalid'); } }
</script>
{#if schema.comfy_model}
  <ComfyModelSelect {title} {anchor} {value} {onchange} info={modelInfo} {help} {error}/>
{:else if schema.vector_labels}
  <fieldset id={anchor} class:errored={!!error}><legend>{title}{#if help}<HelpTip label={'Help: '+title} text={help}/>{/if}</legend><div class="vector-inputs">
    {#each schema.vector_labels as axis,index}<label><span>{axis}</span><input type="number" step="any"
      aria-label={`${title} ${axis}`} value={value?.[index]??0}
      onchange={(event)=>onchange(schema.vector_labels.map((_,i)=>i===index?Number(event.currentTarget.value):value?.[i]??0))}/></label>{/each}
  </div>{#if error}<small class="err">{error}</small>{/if}</fieldset>
{:else if schema.rgb_color}
  <label class="rgb-field" id={anchor}><span>{title}{#if help}<HelpTip label={'Help: '+title} text={help}/>{/if}</span><input type="color" aria-label={title}
    value={'#'+(value||[0,0,0]).map(c=>Math.max(0,Math.min(255,c)).toString(16).padStart(2,'0')).join('')}
    oninput={(event)=>onchange([1,3,5].map(offset=>parseInt(event.currentTarget.value.slice(offset,offset+2),16)))}/>
    <small>RGB {(value||[0,0,0]).join(', ')}</small></label>
{:else if schema.slider}
  <label class:errored={!!error} id={anchor}><span>{title} <output>{Number(((value??0)*schema.slider.scale).toFixed(2))} / {schema.slider.max}</output>{#if help}<HelpTip label={'Help: '+title} text={help}/>{/if}</span>
    <input type="range" min={schema.slider.min} max={schema.slider.max} step={schema.slider.step}
      value={(value??0)*schema.slider.scale} aria-label={title}
      oninput={(event)=>onchange(Number(event.currentTarget.value)/schema.slider.scale)}/>
    {#if error}<small class="err">{error}</small>{/if}</label>
{:else if schema.encoder_options}
  <GuidanceEncoders {value} choices={schema.encoder_options} {onchange} {error}/>
{:else if isSchedule || isPrompts || name === 'frame_range'}
  <div class="wrap" class:errored={!!error} id={anchor}>
    {#if name === 'frame_range'}<FrameRange {value} {onchange}/>
    {:else}<ScheduleEditor {name} {value} {onchange} label={title} hint={help}
      kind={isPrompts ? 'text' : 'number'}
      axis={isProgress ? 'progress' : schema.schedule_domain === 'sampling_step' || name === 'sampler_scale_schedule' ? 'step' : 'frame'}
      interpolation={schema.schedule_interpolation || (isProgress || schema.schedule_domain==='sampling_step'?'hold':'blend')} valueType={schema.schedule_value}
      valueLabel={schema.schedule_value==='cut_count'?'Cut count per batch':schema.schedule_value==='probability'?'Grayscale probability':'Value'}/>{/if}
    {#if error}<small class="err">{error}</small>{/if}
  </div>
{:else if schema.type === 'dataclass'}
  <fieldset id={anchor}><legend>{title}{#if help}<HelpTip label={'Help: '+title} text={help}/>{/if}</legend><div class="grid">
    {#each Object.entries(schema.properties) as [childName, childSchema]}
      <Field name={childName} schema={childSchema} value={value?.[childName]} path={full} onchange={(v)=>onchange({...value,[childName]:v})}/>
    {/each}
  </div></fieldset>
{:else if schema.type === 'boolean'}
  <label class="toggle" class:errored={!!error} id={anchor}><input type="checkbox" checked={value} onchange={(e)=>onchange(e.target.checked)}/><span>{title}{#if help}<HelpTip label={'Help: '+title} text={help}/>{/if}</span>
    {#if error}<small class="err">{error}</small>{/if}</label>
{:else if schema.choices}
  <label class:errored={!!error} id={anchor}><span>{title}{#if help}<HelpTip label={'Help: '+title} text={help}/>{/if}</span><select class:invalid={!!error} value={value} onchange={(e)=>onchange(schema.type==='integer' ? parseInt(e.target.value) : e.target.value)}>
    {#each schema.choices as choice}<option value={choice}>{schema.choice_labels?.[choice] || choice}</option>{/each}
  </select>{#if error}<small class="err">{error}</small>{/if}</label>
{:else if schema.type === 'integer' || schema.type === 'number'}
  <label class:errored={!!error} id={anchor}><span>{title}{#if help}<HelpTip label={'Help: '+title} text={help}/>{/if}</span><input class:invalid={!!error} type="number" value={value} step={schema.type==='integer'?1:'any'} onchange={numberValue}/>
    {#if error}<small class="err">{error}</small>{/if}</label>
{:else if schema.type === 'string'}
  <label class:wide={name.includes('prompt') || isPlainText || isPathField(name)} class:errored={!!error} id={anchor}><span>{title}{#if help}<HelpTip label={'Help: '+title} text={help}/>{/if}</span>
    {#if name.includes('prompt') || isPlainText}<textarea class:invalid={!!error} value={value} rows="3" aria-label={title} onchange={(e)=>onchange(e.target.value)}></textarea>
    {:else if isPathField(name)}<input class="path-value" class:missing class:invalid={!!error} value={value} onchange={checkPath} onblur={checkPath}/>
      {#if missing}<small class="missing-note">Not found on disk</small>{/if}
    {:else}<input class:invalid={!!error} value={value} onchange={(e)=>onchange(e.target.value)}/>{/if}
    {#if error}<small class="err">{error}</small>{/if}
  </label>
{:else}
  <label class="wide" class:errored={!!error} id={anchor}><span>{title} <small>JSON</small></span><textarea class="json-value" class:invalid={!!error} value={jsonText} rows="4" onchange={jsonValue}></textarea>
    {#if error}<small class="err">{error}</small>{/if}</label>
{/if}
<style>
  select{width:100%;border:1px solid var(--input-border);background:var(--input-bg);color:var(--ink);border-radius:var(--radius);padding:10px 11px;font-size:var(--text-value)}
  select:focus{border-color:var(--accent-muted)}
  .missing{border-color:var(--warning)}
  .missing-note{display:block;margin-top:6px;color:var(--warning);font:var(--text-help)/1.5 var(--font-ui)}
  /* The server's message, next to the field it is about — not only in the sidebar. */
  .err{display:block;margin-top:6px;color:var(--error);font:var(--text-help)/1.5 var(--font-ui)}
  .errored :global(input),.errored :global(textarea),.errored select{border-color:#c0505a}
  .wrap{grid-column:1/-1}
  .vector-inputs{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
  input[type=range]{width:100%;accent-color:#d8ff55;min-height:32px;padding:0;border:0;background:transparent;cursor:pointer}
  /* Flash the field when the sidebar scrolls you to it, so you can see WHICH one it meant. */
  :global(.flash){animation:flash 1.6s ease-out}
  @keyframes flash{0%,30%{background:#4a242980;box-shadow:0 0 0 6px #4a242940}100%{background:transparent;box-shadow:none}}
</style>
