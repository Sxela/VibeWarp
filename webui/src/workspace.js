export function conditionsMatch(schema,config) {
  return (schema?.visible_when||[]).every(condition=>{
    const value=condition.path.split('.').reduce((item,key)=>item?.[key],config);
    return 'equals' in condition?value===condition.equals:value!==condition.not_equals;
  });
}

export function flattenFields(schema) {
  return Object.entries(schema?.properties || {}).flatMap(([name, prop]) =>
    prop.type === 'dataclass'
      ? Object.entries(prop.properties).map(([child, sub]) => ({
          section: name, name: child, schema: sub, tier: sub.tier, group: sub.group,
        }))
      : [{section: 'main', name, schema: prop, tier: prop.tier, group: prop.group}]
  ).filter(entry => entry.tier);
}

export function migrateLegacyFrameLimit(config) {
  if (!config?.video || !('max_frames' in config.video)) return config;
  const {max_frames:count,...video}=config.video;
  const selected=config.frame_range;
  const explicit=Array.isArray(selected)&&!(selected.length===2&&selected[0]===0&&selected[1]===0);
  const frame_range=explicit?selected:Number.isInteger(count)&&count>0?(count===1?[0]:[0,count-1]):selected||[0,0];
  return {...config,video,frame_range};
}

export function normalizeRawOpacities(config) {
  if(!config?.mask)return config;
  const mask={...config.mask};
  for(const name of ['foreground_opacity','background_opacity']){
    const value=mask[name];
    // Early unrestricted inputs allowed whole percentages, but the renderer
    // has always used fractions. Repair those browser values on restoration.
    if(Number.isInteger(value)&&value>1&&value<=100)mask[name]=value/100;
  }
  return {...config,mask};
}

export function dropLegacyFrameHash(config) {
  if(!config?.video||!('first_frame_sha256' in config.video))return config;
  const {first_frame_sha256,...video}=config.video;
  return {...config,video};
}

export const fieldLocation = entry => entry.schema.workflows?.[0] || entry.tier;
export const fieldGroup = (entry,workflow) => entry.schema.workflow_groups?.[workflow] || entry.group;
export const fieldEditor = (entry,workflow) => Object.hasOwn(entry.schema.workflow_editors||{},workflow)
  ? entry.schema.workflow_editors[workflow] : entry.schema.editor;
export const inWorkflow = (entry, workflow) => entry.tier !== 'hidden' && (
  workflow === 'all' || (entry.schema.workflows || [entry.tier]).includes(workflow));

export function searchFields(entries, query) {
  const q = query.trim().toLowerCase().replaceAll('_', ' ');
  if (!q) return [];
  return entries.filter(entry => entry.tier !== 'hidden' &&
    [entry.name, entry.schema.label, entry.group, ...Object.values(entry.schema.workflow_groups||{}), entry.section, entry.schema.description]
      .filter(Boolean).join(' ').replaceAll('_', ' ').toLowerCase().includes(q)).slice(0, 60);
}

export function fieldValue(config, entry) {
  const section = entry.section === 'main' ? config : config?.[entry.section];
  if (entry.section === 'disco' && entry.name === 'cut_icgray_schedule')
    return section?.cut_icgray_schedule ?? {0: section?.cut_icgray_p};
  return section?.[entry.name] ?? (entry.schema?.schedule_fallback
    ? section?.[entry.schema.schedule_fallback] : entry.name.endsWith('_schedule')
    ? section?.[entry.name.slice(0, -9)] : undefined);
}

export function contactSheetConflict(config, family) {
  if (!['flux', 'hidream', 'qwen', 'mage'].includes(family) || !config?.contact_sheet?.mode || config.contact_sheet.mode === 'off') return '';
  if (family === 'flux' && config?.flux?.backend !== 'comfy')
    return 'Temporal contact sheets require the ComfyUI backend.';
  if (config?.[family]?.references?.slice(1).some(ref => ref.source !== 'none'))
    return 'Temporal contact sheets do not support optional reference images. Your references have been kept.';
  return '';
}
