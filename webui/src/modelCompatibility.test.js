import { describe, expect, it } from 'vitest';
import { applyModelDefaults, rememberModelSelections, comfyConnection, fieldSupportsModel, modelFamily, modelFieldHint, switchDiscoSampler } from './modelCompatibility.js';

it('isolates custom filenames per model and restores them after serialization',()=>{
  const schema={properties:{comfy_image:{properties:{
    comfy_clip_name:{comfy_model:true},comfy_unet_name:{comfy_model:true},
    comfy_vae_name:{comfy_model:true},comfy_edit_lora_name:{comfy_model:true},
  }}},comfy_image_model_defaults:{
    flux3:{comfy_clip_name:'4b',comfy_unet_name:'flux',comfy_vae_name:'flux-vae',comfy_edit_lora_name:''},
    minimax:{comfy_clip_name:'32b',comfy_unet_name:'mini',comfy_vae_name:'mini-vae',comfy_edit_lora_name:''},
  }};
  let config={model_version:'flux3',comfy_image:{comfy_clip_name:'custom-4b',comfy_unet_name:'custom-flux',comfy_vae_name:'full-vae',comfy_edit_lora_name:'',mode:'edit'}};
  let saved=rememberModelSelections(config,schema);
  config=applyModelDefaults(config,'minimax',schema,saved);
  expect(config.comfy_image).toMatchObject(schema.comfy_image_model_defaults.minimax);
  config.comfy_image.comfy_clip_name='custom-32b';
  saved=JSON.parse(JSON.stringify(rememberModelSelections(config,schema,saved)));
  config=applyModelDefaults(config,'flux3',schema,saved);
  expect(config.comfy_image).toMatchObject({comfy_clip_name:'custom-4b',comfy_unet_name:'custom-flux',comfy_vae_name:'full-vae',comfy_edit_lora_name:'',mode:'edit'});
  expect(applyModelDefaults(config,'minimax',schema,saved).comfy_image.comfy_clip_name).toBe('custom-32b');
});

it('switches native Comfy image presets and connects to the selected server',()=>{
  const schema={model_family_by_version:{moss_image:'comfy_image'},comfy_image_model_defaults:{
    flux3_klein:{comfy_unet_name:'flux.safetensors',steps:50},moss_image:{comfy_unet_name:'moss.safetensors',steps:30}}};
  const current={comfy_image:{comfy_unet_name:'flux.safetensors',steps:50,mode:'text',comfy_server_url:'http://comfy.example:8188'}};
  const next=applyModelDefaults(current,'moss_image',schema);
  expect(next.comfy_image).toEqual({...current.comfy_image,comfy_unet_name:'moss.safetensors',steps:30});
  expect(comfyConnection(next,schema).url).toBe('http://comfy.example:8188');
  expect(applyModelDefaults({...current,comfy_image:{...current.comfy_image,steps:79}},'moss_image',schema).comfy_image.steps).toBe(79);
});

describe('Disco distilled sampler settings',()=>{
  const settings={
    properties:{disco:{type:'dataclass'},video:{type:'dataclass'}},
    disco_distilled_preset:{model_version:'disco_512',text_prompts:{0:''},
      disco:{sampler:'distilled',steps:4,encoders:[]},video:{width:512,height:512}},
    disco_guided_restore:{model_version:'disco_512',text_prompts:{0:'default'},
      disco:{sampler:'ddim',steps:50,encoders:['ViT-B-32']},video:{width:512,height:512}},
  };
  it.each([true,false])('keeps secondary and guidance choices (%s) when selecting the adapter sampler',enabled=>{
    const original={model_version:'disco_512',text_prompts:{0:'my painting'},
      disco:{sampler:'ddim',steps:120,encoders:['siglip2-base'],checkpoint_path:'base.pt',few_step_lora_path:'adapter.safetensors',use_secondary_model:enabled,distilled_guided:enabled,clip_guidance_scale:7000,eta:.25},
      video:{width:1024,height:768,video_init_path:'input.mp4'}};
    const next=switchDiscoSampler(original,'distilled',settings);
    expect(next.config).toEqual({...original,disco:{...original.disco,sampler:'distilled'}});
    const edited={...next.config,disco:{...next.config.disco,checkpoint_path:'new-base.pt',clip_guidance_scale:9000}};
    const restored=switchDiscoSampler(edited,'plms',settings,next.previous);
    expect(restored.config).toEqual({...edited,disco:{...edited.disco,sampler:'plms'}});
  });
  it('keeps imported adapter settings when changing back to DDIM',()=>{
    const config={disco:{sampler:'distilled',steps:4,encoders:[],use_secondary_model:false,distilled_guided:false,few_step_lora_path:'adapter.safetensors'},video:{width:512,height:512}};
    const next=switchDiscoSampler(config,'ddim',settings);
    expect(next.config).toEqual({...config,disco:{...config.disco,sampler:'ddim'}});
  });
});

const schema = {
  model_family_by_version: {
    control_multi_v15: 'sd',
    flux2_klein_edit: 'flux',
    flux2_klein_9b_edit: 'flux',
    hidream_o1_edit: 'hidream',
    qwen_image_edit_2511: 'qwen',
    qwen_image_edit_2511_gguf: 'qwen',
    mage_flow_edit: 'mage',
    mage_flow_edit_turbo: 'mage',
  },
  default_model_family: 'sd',
  flux_model_defaults: {
    flux2_klein_edit: {
      comfy_unet_name: 'flux-2-klein-4b-fp8.safetensors',
      comfy_clip_name: 'qwen_3_4b.safetensors',
      model_repo: 'black-forest-labs/FLUX.2-klein-4B',
    },
    flux2_klein_9b_edit: {
      comfy_unet_name: 'flux-2-klein-9b-fp8.safetensors',
      comfy_clip_name: 'qwen_3_8b_fp8mixed.safetensors',
      model_repo: 'black-forest-labs/FLUX.2-klein-9B',
    },
  },
  qwen_model_defaults: {
    qwen_image_edit_2511: {
      comfy_unet_name: 'qwen_image_edit_2511_int8_convrot.safetensors',
    },
    qwen_image_edit_2511_gguf: {
      comfy_unet_name: 'Qwen-Image-Edit-2511-Q5_K_M.gguf',
    },
  },
  mage_model_defaults: {
    mage_flow_edit: {
      comfy_unet_name: 'mage_flow_edit_int8_convrot.safetensors',
      steps: 30,
      guidance_scale: 5,
    },
    mage_flow_edit_turbo: {
      comfy_unet_name: 'mage_flow_edit_turbo_int8_convrot.safetensors',
      steps: 4,
      guidance_scale: 1,
    },
  },
};

describe('model compatibility', () => {
  it('resolves declared edit families and defaults imported legacy models to SD', () => {
    expect(modelFamily('flux2_klein_edit', schema)).toBe('flux');
    expect(modelFamily('flux2_klein_9b_edit', schema)).toBe('flux');
    expect(modelFamily('hidream_o1_edit', schema)).toBe('hidream');
    expect(modelFamily('qwen_image_edit_2511', schema)).toBe('qwen');
    expect(modelFamily('qwen_image_edit_2511_gguf', schema)).toBe('qwen');
    expect(modelFamily('mage_flow_edit', schema)).toBe('mage');
    expect(modelFamily('mage_flow_edit_turbo', schema)).toBe('mage');
    expect(modelFamily('control_multi_animatediff', schema)).toBe('sd');
  });

  it('switches the Qwen native and GGUF transformer presets', () => {
    let config = {
      model_version: 'qwen_image_edit_2511',
      qwen: {
        comfy_unet_name: 'qwen_image_edit_2511_int8_convrot.safetensors',
        comfy_clip_name: 'qwen_2.5_vl_7b_fp8_scaled.safetensors',
      },
    };
    let switched = applyModelDefaults(
      config, 'qwen_image_edit_2511_gguf', schema);
    expect(switched.model_version).toBe('qwen_image_edit_2511_gguf');
    expect(switched.qwen.comfy_unet_name).toBe(
      'Qwen-Image-Edit-2511-Q5_K_M.gguf');
    expect(switched.qwen.comfy_clip_name).toBe(
      'qwen_2.5_vl_7b_fp8_scaled.safetensors');

    switched.qwen.comfy_unet_name = 'my-qwen-q5.gguf';
    expect(applyModelDefaults(
      switched, 'qwen_image_edit_2511', schema
    ).qwen.comfy_unet_name).toBe('my-qwen-q5.gguf');
  });

  it('switches known Flux defaults while preserving custom model files', () => {
    let config = {
      model_version: 'flux2_klein_edit',
      flux: {
        comfy_unet_name: 'flux-2-klein-4b-fp8.safetensors',
        comfy_clip_name: 'my-custom-encoder.safetensors',
        model_repo: 'black-forest-labs/FLUX.2-klein-4B',
      },
    };
    let switched = applyModelDefaults(config, 'flux2_klein_9b_edit', schema);
    expect(switched.model_version).toBe('flux2_klein_9b_edit');
    expect(switched.flux.comfy_unet_name).toBe(
      'flux-2-klein-9b-fp8.safetensors');
    expect(switched.flux.comfy_clip_name).toBe(
      'my-custom-encoder.safetensors');
    expect(switched.flux.model_repo).toBe(
      'black-forest-labs/FLUX.2-klein-9B');
  });

  it('switches the Mage standard and Turbo sampling presets', () => {
    let config = {
      model_version: 'mage_flow_edit',
      mage: {
        comfy_unet_name: 'mage_flow_edit_int8_convrot.safetensors',
        steps: 30,
        guidance_scale: 5,
      },
    };
    let switched = applyModelDefaults(config, 'mage_flow_edit_turbo', schema);
    expect(switched.model_version).toBe('mage_flow_edit_turbo');
    expect(switched.mage.comfy_unet_name).toBe(
      'mage_flow_edit_turbo_int8_convrot.safetensors');
    expect(switched.mage.steps).toBe(4);
    expect(switched.mage.guidance_scale).toBe(1);
  });

  it('shows common fields and only the selected model-specific fields', () => {
    expect(fieldSupportsModel({}, 'hidream')).toBe(true);
    expect(fieldSupportsModel({ model_families: ['sd'] }, 'hidream')).toBe(false);
    expect(fieldSupportsModel({ model_families: ['hidream'] }, 'hidream')).toBe(true);
    expect(fieldSupportsModel({ model_families: ['flux'] }, 'hidream')).toBe(false);
  });

  it('shows model-aware recommended sampling values', () => {
    expect(modelFieldHint(
      { model_version: 'mage_flow_edit_turbo' }, 'mage', 'steps'
    )).toBe('Recommended for Mage-Flow Edit Turbo: 4.');
    expect(modelFieldHint(
      { model_version: 'mage_flow_edit' }, 'mage', 'guidance_scale'
    )).toBe('Recommended for Mage-Flow Edit: 5.');
    expect(modelFieldHint({
      model_version: 'qwen_image_edit_2511',
      qwen: { use_lightning_lora: true },
    }, 'qwen', 'steps')).toBe(
      'Recommended with the 8-step Lightning LoRA: 8.');
    expect(modelFieldHint({
      model_version: 'qwen_image_edit_2511',
      qwen: { use_lightning_lora: false },
    }, 'qwen', 'guidance_scale')).toBe(
      'Recommended without the Lightning LoRA: 4.');
    expect(modelFieldHint(
      { model_version: 'hidream_o1_edit' }, 'hidream', 'noise_scale'
    )).toBe('Required training noise scale for HiDream-O1 Full: 8.');
  });

  it('identifies only model selections that use a ComfyUI server', () => {
    let flux = { model_version: 'flux2_klein_edit',
      flux: { backend: 'comfy', comfy_server_url: 'http://127.0.0.1:8188' } };
    expect(comfyConnection(flux, schema)).toEqual({
      label: 'Flux.2', url: 'http://127.0.0.1:8188', path: 'flux.comfy_server_url' });
    flux.flux.backend = 'diffusers';
    expect(comfyConnection(flux, schema)).toBeNull();

    let hidream = { model_version: 'hidream_o1_edit',
      hidream: { comfy_server_url: 'http://localhost:9191' } };
    expect(comfyConnection(hidream, schema)).toEqual({
      label: 'HiDream-O1', url: 'http://localhost:9191',
      path: 'hidream.comfy_server_url' });
    let qwen = { model_version: 'qwen_image_edit_2511',
      qwen: { comfy_server_url: 'http://localhost:8288' } };
    expect(comfyConnection(qwen, schema)).toEqual({
      label: 'Qwen Image Edit', url: 'http://localhost:8288',
      path: 'qwen.comfy_server_url' });
    let mage = { model_version: 'mage_flow_edit',
      mage: { comfy_server_url: 'http://localhost:8388' } };
    expect(comfyConnection(mage, schema)).toEqual({
      label: 'Mage-Flow Edit', url: 'http://localhost:8388',
      path: 'mage.comfy_server_url' });
    expect(comfyConnection({ model_version: 'control_multi_v15' }, schema)).toBeNull();
  });
});
