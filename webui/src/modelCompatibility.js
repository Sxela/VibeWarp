export function modelFamily(version, schema) {
  return schema?.model_family_by_version?.[version]
    ?? schema?.default_model_family
    ?? 'sd';
}

export function fieldSupportsModel(fieldSchema, family) {
  let allowed = fieldSchema?.model_families;
  return !allowed || allowed.includes(family);
}

export function applyModelDefaults(config, version, schema) {
  let presetGroups = [
    ['flux_model_defaults', 'flux'],
    ['qwen_model_defaults', 'qwen'],
    ['mage_model_defaults', 'mage'],
  ];
  for (let [presetKey, sectionKey] of presetGroups) {
    let presets = schema?.[presetKey];
    let target = presets?.[version];
    if (!target) continue;

    let known = Object.values(presets);
    let section = { ...config[sectionKey] };
    for (let [key, next] of Object.entries(target)) {
      let current = section[key];
      if (!current || known.some(preset => preset[key] === current))
        section[key] = next;
    }
    return { ...config, model_version: version, [sectionKey]: section };
  }
  return { ...config, model_version: version };
}

export function comfyConnection(config, schema) {
  let family = modelFamily(config?.model_version, schema);
  if (family === 'flux' && config?.flux?.backend === 'comfy') {
    return { label: 'Flux.2', url: config.flux.comfy_server_url,
             path: 'flux.comfy_server_url' };
  }
  if (family === 'hidream') {
    return { label: 'HiDream-O1', url: config?.hidream?.comfy_server_url,
             path: 'hidream.comfy_server_url' };
  }
  if (family === 'qwen') {
    return { label: 'Qwen Image Edit', url: config?.qwen?.comfy_server_url,
             path: 'qwen.comfy_server_url' };
  }
  if (family === 'mage') {
    return { label: 'Mage-Flow Edit', url: config?.mage?.comfy_server_url,
             path: 'mage.comfy_server_url' };
  }
  return null;
}

export function modelFieldHint(config, section, name) {
  let version = config?.model_version ?? '';

  if (section === 'mage') {
    let turbo = version === 'mage_flow_edit_turbo';
    let preset = turbo ? 'Mage-Flow Edit Turbo' : 'Mage-Flow Edit';
    let values = turbo
      ? { steps: 4, guidance_scale: 1, sampler: 'euler', scheduler: 'simple' }
      : { steps: 30, guidance_scale: 5, sampler: 'euler', scheduler: 'simple' };
    if (name in values) return `Recommended for ${preset}: ${values[name]}.`;
  }

  if (section === 'qwen') {
    let lightning = config?.qwen?.use_lightning_lora !== false;
    let mode = lightning ? 'with the 8-step Lightning LoRA' : 'without the Lightning LoRA';
    let values = lightning
      ? { steps: 8, guidance_scale: 1, lora_strength: 1,
          sampler: 'euler', scheduler: 'simple', sampling_shift: 3.1 }
      : { steps: 40, guidance_scale: 4,
          sampler: 'euler', scheduler: 'simple', sampling_shift: 3.1 };
    if (name in values) return `Recommended ${mode}: ${values[name]}.`;
  }

  let recommendations = {
    flux: {
      steps: 'Recommended for FLUX.2 Klein: 4.',
      guidance_scale: 'Recommended for FLUX.2 Klein: 1.',
    },
    hidream: {
      steps: 'Recommended for HiDream-O1 Full: 40.',
      guidance_scale: 'Recommended for HiDream-O1 Full: 5.',
      sampler: 'Recommended for HiDream-O1 Full: dpmpp_2m_sde_gpu.',
      scheduler: 'Recommended for HiDream-O1 Full: normal.',
      noise_scale: 'Required training noise scale for HiDream-O1 Full: 8.',
    },
  };
  return recommendations[section]?.[name] ?? '';
}
