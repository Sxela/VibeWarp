<script>
  // `error` is the server's validation message for THIS field. It is rendered next to the
  // input rather than only in the sidebar, and the anchor id lets the sidebar scroll here.
  let { name, schema, value, onchange, path = '', hint = '', error = '' } = $props();
  import Field from './Field.svelte';
  import ScheduleEditor from './ScheduleEditor.svelte';
  import FrameRange from './FrameRange.svelte';
  import { isPathField, stripQuotes } from './paths.js';
  // Every *_schedule field is a keyframe/per-frame series — chips, not raw JSON.
  let isSchedule = $derived(name.endsWith('_schedule'));
  // Prompts are the same keyframe structure, only the values are text: Dict[int, str].
  // reconstruction_noise.prompts/neg_prompts are prompt schedules too (same type) — they
  // were rendering as a raw JSON blob.
  const PROMPTS = new Set(['text_prompts', 'negative_prompts', 'prompts', 'neg_prompts']);
  let isPrompts = $derived(PROMPTS.has(name));
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
{#if isSchedule || isPrompts || name === 'frame_range'}
  <div class="wrap" class:errored={!!error} id={anchor}>
    {#if name === 'frame_range'}<FrameRange {value} {onchange}/>
    {:else}<ScheduleEditor {name} {value} {onchange} kind={isPrompts ? 'text' : 'number'}/>{/if}
    {#if error}<small class="err">{error}</small>{/if}
  </div>
{:else if schema.type === 'dataclass'}
  <fieldset id={anchor}><legend>{title}</legend><div class="grid">
    {#each Object.entries(schema.properties) as [childName, childSchema]}
      <Field name={childName} schema={childSchema} value={value?.[childName]} path={full} onchange={(v)=>onchange({...value,[childName]:v})}/>
    {/each}
  </div></fieldset>
{:else if schema.type === 'boolean'}
  <label class="toggle" class:errored={!!error} id={anchor}><input type="checkbox" checked={value} onchange={(e)=>onchange(e.target.checked)}/><span>{title}</span>
    {#if error}<small class="err">{error}</small>{/if}</label>
{:else if schema.choices}
  <label class:errored={!!error} id={anchor}><span>{title}</span><select class:invalid={!!error} value={value} onchange={(e)=>onchange(schema.type==='integer' ? parseInt(e.target.value) : e.target.value)}>
    {#each schema.choices as choice}<option value={choice}>{choice}</option>{/each}
  </select>{#if error}<small class="err">{error}</small>{:else if hint}<small class="hint">{hint}</small>{/if}</label>
{:else if schema.type === 'integer' || schema.type === 'number'}
  <label class:errored={!!error} id={anchor}><span>{title}</span><input class:invalid={!!error} type="number" value={value} step={schema.type==='integer'?1:'any'} onchange={numberValue}/>
    {#if error}<small class="err">{error}</small>{:else if hint}<small class="hint">{hint}</small>{/if}</label>
{:else if schema.type === 'string'}
  <label class:wide={name.includes('prompt') || isPlainText || isPathField(name)} class:errored={!!error} id={anchor}><span>{title}</span>
    {#if name.includes('prompt') || isPlainText}<textarea class:invalid={!!error} value={value} rows="3" aria-label={title} onchange={(e)=>onchange(e.target.value)}></textarea>
    {:else if isPathField(name)}<input class:missing class:invalid={!!error} value={value} onchange={checkPath} onblur={checkPath}/>
      {#if missing}<small class="missing-note">Not found on disk</small>{/if}
    {:else}<input class:invalid={!!error} value={value} onchange={(e)=>onchange(e.target.value)}/>{/if}
    {#if error}<small class="err">{error}</small>{:else if hint}<small class="hint">{hint}</small>{/if}
  </label>
{:else}
  <label class="wide" class:errored={!!error} id={anchor}><span>{title} <small>JSON</small></span><textarea class:invalid={!!error} value={jsonText} rows="4" onchange={jsonValue}></textarea>
    {#if error}<small class="err">{error}</small>{/if}</label>
{/if}
<style>
  select{width:100%;border:1px solid #303640;background:#0c0f13;color:#eef0f2;border-radius:8px;padding:10px 11px;outline:none;font:12px Consolas,monospace}
  select:focus{border-color:#8ea834}
  .hint{display:block;margin-top:6px;color:#8f98a5;font:11px/1.4 system-ui,sans-serif}
  .missing{border-color:#c98a3c}
  .missing-note{display:block;margin-top:6px;color:#c98a3c;font:11px/1.4 system-ui,sans-serif}
  /* The server's message, next to the field it is about — not only in the sidebar. */
  .err{display:block;margin-top:6px;color:#f18b91;font:11px/1.4 system-ui,sans-serif}
  .errored :global(input),.errored :global(textarea),.errored select{border-color:#c0505a}
  .wrap{grid-column:1/-1}
  /* Flash the field when the sidebar scrolls you to it, so you can see WHICH one it meant. */
  :global(.flash){animation:flash 1.6s ease-out}
  @keyframes flash{0%,30%{background:#4a242980;box-shadow:0 0 0 6px #4a242940}100%{background:transparent;box-shadow:none}}
</style>
