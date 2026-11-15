// Canonicalize only representations that the keyframe chips can express losslessly.
export function canonicalSchedule(value, {kind='number', axis='frame', valueType}={}) {
  if (value == null) return value;
  const validKey = key => key.trim() !== '' && Number.isFinite(Number(key)) && Number(key)>=0
    && (axis==='progress' ? Number(key)<=100 : Number.isSafeInteger(Number(key)));
  const numeric = item => typeof item==='number' && Number.isFinite(item);
  const validNumber = item => numeric(item) && (valueType!=='cut_count'||Number.isInteger(item)&&item>=0)
    && (valueType!=='probability'||item>=0&&item<=1);
  const advancedValue = item => validNumber(item) || Array.isArray(item)&&item.length>0&&item.every(advancedValue);
  if (kind==='number' && Array.isArray(value) && value.some(item=>Array.isArray(item))) {
    if (!value.every(advancedValue)) throw Error('Schedule values must be finite numbers.');
    return value;
  }
  const entries = typeof value==='object' ? Object.entries(value) : [['0',value]];
  // Empty prompt dictionaries (negative/reconstruction prompts) are meaningful.
  if (!entries.length && kind!=='text') throw Error('A schedule needs at least one keyframe.');
  const result={};
  for (const [key, initial] of entries) {
    if (!validKey(key)) throw Error(axis==='progress'?'Use progress keys between 0 and 100.':'Use non-negative whole frame or step numbers.');
    const item=kind==='text'&&Array.isArray(initial)&&initial.every(v=>typeof v==='string')?initial.join(' | '):initial;
    if (kind==='text' ? typeof item!=='string' : !advancedValue(item))
      throw Error(kind==='text'?'Prompt values must be text.':'Schedule values are invalid for this field.');
    if (Object.hasOwn(result,Number(key))) throw Error('Two keys resolve to the same position.');
    result[Number(key)]=item;
  }
  return result;
}

export const scheduleOptions = schema => ({
  kind:schema.schedule_kind||'number',
  axis:['inference_progress','denoising_progress'].includes(schema.schedule_domain)?'progress':schema.schedule_domain==='sampling_step'?'step':'frame',
  valueType:schema.schedule_value,
});

export function normalizeSchedules(config,schema) {
  if (!config) return config;
  const result=JSON.parse(JSON.stringify(config));
  function walk(target,properties) {
    for (const [name,property] of Object.entries(properties||{})) {
      if (property.type==='dataclass') { if(target[name]) walk(target[name],property.properties); continue; }
      if (!property.schedule_domain && !property.schedule_kind) continue;
      // A visible scalar can belong to a mode where activating its optional
      // schedule is invalid (Disco's exact distilled reference mode). Only
      // materialize fallback fields which the backend marks as superseded.
      const fallback=property.schedule_fallback;
      const value=target[name]??(fallback&&properties[fallback]?.tier==='hidden'?target[fallback]:undefined);
      if (value==null) continue;
      try { target[name]=canonicalSchedule(value,scheduleOptions(property)); }
      catch { /* Keep advanced/legacy data visible for correction, never discard it. */ }
    }
  }
  walk(result,schema?.properties);
  return result;
}
