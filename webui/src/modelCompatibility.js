export function modelFamily(version, schema) {
  return schema?.model_family_by_version?.[version]
    ?? schema?.default_model_family
    ?? 'sd';
}

export function fieldSupportsModel(fieldSchema, family) {
  let allowed = fieldSchema?.model_families;
  return !allowed || allowed.includes(family);
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
  return null;
}
