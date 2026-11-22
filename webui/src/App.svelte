<script>
  import { tick, untrack, onDestroy } from 'svelte';
  import Field from './Field.svelte';
  import {normalizeRawOpacities,dropLegacyFrameHash,conditionsMatch} from './workspace.js';
  import {closeOnSelection} from './closeOnSelection.js';
  import RawVideoPreview from './RawVideoPreview.svelte';
  import ControlNetEditor from './ControlNetEditor.svelte';
  import IPAdapterEditor from './IPAdapterEditor.svelte';
  import ReferenceImagesEditor from './ReferenceImagesEditor.svelte';
  import VideoEditor from './VideoEditor.svelte';
  import Preview from './Preview.svelte';
  import Supporters from './Supporters.svelte';
  import DownloadProgress from './DownloadProgress.svelte';
  import RequiredModels from './RequiredModels.svelte';
  import { applyModelDefaults, rememberModelSelections, comfyConnection, fieldSupportsModel, modelFamily, modelFieldHint, switchDiscoSampler } from './modelCompatibility.js';
  import {flattenFields, fieldLocation, fieldGroup, fieldEditor, inWorkflow, searchFields, fieldValue as readFieldValue, contactSheetConflict, migrateLegacyFrameLimit} from './workspace.js';
  import {normalizeSchedules} from './schedules.js';
  import HelpTip from './HelpTip.svelte';
  import OutputLog from './OutputLog.svelte';
  import {remainingEta,formatEta} from './jobEta.js';
  let inspectorHidden=$state(false), snapshotLoading=$state(false), snapshotError=$state('');
  let snapshotRevision=0;
  const storageKey='vibewarp.config.v1';
  const modelSelectionsKey='vibewarp.modelSelections.v1';
  let modelSelections={};
  let config=$state(null), schema=$state(null), selected=$state('source'), job=$state(null), errors=$state([]), busy=$state(false), initialized=$state(false), logCollapsed=$state(true), source;
  let etaClock=$state(Date.now());
  let eta=$derived(remainingEta(job,etaClock));
  $effect(()=>{
    if(job?.state!=='running'||!Number.isFinite(job?.eta_seconds))return;
    etaClock=Date.now();const timer=setInterval(()=>etaClock=Date.now(),1000);
    return()=>clearInterval(timer);
  });
  let mobileView=$state('viewer'), viewerMode=$state('compare'), fileInput=$state(null), initError=$state('');
  let editingProject=$state(false),projectDraft=$state(''),projectInput=$state(null),projectTitle=$state(null);
  let inspectorWidth=$state(420), resizing=$state(false), jobConfig=$state(null);
  const workflowHelp={source:'Choose the video, frame range, and destination.',model:'Model files and backend connection.',prompts:'Prompts, sampling, and seed.',clip:'General guidance, encoders, cutouts, and gradient controls.',conditioning:'Inputs that guide this model.',background:'RVM foreground/background masks, tint and render compositing.',color:'Raw-input color preparation and matching before or after diffusion.',consistency:'Motion, warp, and consistency between frames.',export:'Assemble the finished video.',all:'Every editable setting for this model.',system:'Paths and performance for this machine.'};
  // Tabs come from the BACKEND (vibewarp/ui_layout.py stamps tier/group onto every field
  // in the schema). The nav holds no field list of its own — one here would drift silently
  // the moment config.py changed. A field with no tier renders nowhere, and a test fails.
  // History is a viewer mode rather than a settings section.
  let mode=$state('render');
  let renderActive=$derived(['queued','running'].includes(job?.state));
  let cancelling=$state(false);
  // Latches once ANY frame has rendered this session, so the supporters block appears
  // while you wait for your first frame and then never interrupts again.
  let seenFirstFrame=$state(false);
  $effect(()=>{ if(job?.preview_available) seenFirstFrame=true; });
  let tabs=$derived((schema?.workflows || schema?.tiers || []).filter(tab=>
    tab.id==='model'||entries.some(entry=>inWorkflow(entry,tab.id))));
  $effect(()=>{if(initialized&&tabs.length&&!tabs.some(tab=>tab.id===selected))selected=tabs[0].id;});
  let modelOptions=$derived(schema?.model_options || (schema?.properties?.model_version?.choices || []).map(value=>({value,label:value})));
  let modelGroups=$derived(schema?.model_groups || [{id:null,label:''}]);
  let supportedModel=$derived(!config || !modelOptions.length || modelOptions.some(option=>option.value===config.model_version));
  let viewerConfig=$derived(job ? jobConfig : config);
  let inspectorVisible=$derived(mode==='render'&&!inspectorHidden&&viewerMode!=='inspect');
  let pendingDownloads=$derived(downloadFiles.filter(file=>!['ready','present','cancelled'].includes(file.status)).length);
  let missingModels=$derived(requiredModels.filter(file=>!['present','ready'].includes(file.status)).length);
  function chooseWorkflow(id){inspectorHidden=false;selected=id;mode='render';query='';mobileView='settings';if(viewerMode==='inspect')viewerMode='compare';}
  function openModelSetup(){chooseWorkflow(tabs.some(tab=>tab.id==='model')?'model':tabs[0]?.id);}
  function changeInspectorWidth(value){
    inspectorWidth=Math.max(340,Math.min(680,value));
    try{localStorage.setItem('vibewarp.inspector.width',String(inspectorWidth));}catch{}
  }
  function resizeInspector(event){if(resizing)changeInspectorWidth(window.innerWidth-event.clientX);}
  function resizeKey(event){
    if(event.key==='ArrowLeft'||event.key==='ArrowRight'){event.preventDefault();changeInspectorWidth(inspectorWidth+(event.key==='ArrowLeft'?20:-20));}
    if(event.key==='Home'||event.key==='End'){event.preventDefault();changeInspectorWidth(event.key==='Home'?340:680);}
  }
  onDestroy(()=>source?.close());
  $effect(()=>{if(job?.state==='failed')logCollapsed=false;});

  // Flatten the schema into {section, name, schema, tier, group}, in declaration order.
  let allEntries=$derived(flattenFields(schema));
  let activeModelFamily=$derived(modelFamily(config?.model_version, schema));
  let conditioningConflict=$derived(contactSheetConflict(config,activeModelFamily));
  let comfyTarget=$derived(comfyConnection(config, schema));
  let comfyStatus=$state(null);
  let preparation=$state(null), preparationError=$state('');
  let requiredModels=$state([]), requiredLoading=$state(false), requiredError=$state('');
  let comfyModels=$state([]);
  let requiredRevision=$state(0), preparing=$state(false);
  const modelFiles=section=>Object.fromEntries(Object.entries(section||{}).filter(
    ([name])=>name.endsWith('_name')||['backend','model_repo','use_lightning_lora'].includes(name)));
  let downloadFiles=$state([]);
  async function controlDownload(file,action){
    try{
      let response=await fetch(`/api/models/downloads/${file.id}/${action}`,{method:'POST'});
      let next=await response.json();
      if(!response.ok) throw new Error(next.detail||'Could not update download');
      downloadFiles=downloadFiles.map(row=>row.id===next.id?next:row);
      preparationError='';
    }catch(error){preparationError=error.message;}
  }
  $effect(()=>{
    if(!initialized) return;
    let controller=new AbortController(),timer,stopped=false,lastReady='';
    async function refresh(){
      try{
        let response=await fetch('/api/models/downloads',{signal:controller.signal});
        if(!response.ok) throw new Error('Could not read download queue');
        let next=await response.json();
        if(stopped) return;
        downloadFiles=next.files||[];
        let current=untrack(()=>preparation);
        let task=next.tasks?.find(task=>task.id===current?.id);
        if(task) preparation=task;
        let ready=downloadFiles.filter(file=>file.status==='ready').map(file=>file.id).sort().join(',');
        if(ready!==lastReady){lastReady=ready;requiredRevision+=1;}
      }catch(error){if(!stopped&&error.name!=='AbortError') preparationError=error.message;}
      finally{if(!stopped) timer=setTimeout(refresh,500);}
    }
    refresh();
    return()=>{stopped=true;clearTimeout(timer);controller.abort();};
  });
  // Selection previews local presence only. Transfers require an explicit
  // download click or render; editing controls never starts a transfer.
  let preparationKey=$derived(config ? JSON.stringify({
    model_version:config.model_version, model_path:config.model_path,
    root_dir:config.root_dir, sd_checkpoint_path:config.sd_checkpoint_path,
    comfy_models_dir:config.comfy_models_dir,
    controlnet:config.controlnet, ipadapter:config.ipadapter,
    flow_warp:config.flow?.flow_warp,
    disco:{sampler:config.disco?.sampler,distilled_guided:config.disco?.distilled_guided,few_step_lora_path:config.disco?.few_step_lora_path,
           quality_lora_enabled:config.disco?.quality_lora_enabled,quality_lora_path:config.disco?.quality_lora_path,
           checkpoint_path:config.disco?.checkpoint_path,secondary_path:config.disco?.secondary_path,
           use_secondary_model:config.disco?.use_secondary_model,encoders:config.disco?.encoders},
    flux:modelFiles(config.flux), qwen:modelFiles(config.qwen),
    mage:modelFiles(config.mage), hidream:modelFiles(config.hidream),
    comfy_image:{...modelFiles(config.comfy_image),mode:config.comfy_image?.mode},
    comfy_urls:[config.flux?.comfy_server_url,config.qwen?.comfy_server_url,config.mage?.comfy_server_url,config.hidream?.comfy_server_url,config.comfy_image?.comfy_server_url]
  }) : '');
  $effect(()=>{
    let key=preparationKey;
    requiredRevision;
    if(!initialized||!key) return;
    let controller=new AbortController(), stopped=false;
    let payload=untrack(()=>JSON.parse(JSON.stringify(config)));
    requiredLoading=true; requiredError='';
    let timer=setTimeout(async()=>{
      try{
        let response=await fetch('/api/models/required',{method:'POST',
          headers:{'content-type':'application/json'},body:JSON.stringify(payload),signal:controller.signal});
        if(!response.ok) throw new Error('Could not list required models');
        let next=await response.json();
        if(stopped) return;
        requiredModels=next.files||[];
        comfyModels=next.comfy_models||[];
      }catch(error){if(!stopped&&error.name!=='AbortError') requiredError=error.message;}
      finally{if(!stopped) requiredLoading=false;}
    },250);
    return()=>{stopped=true;clearTimeout(timer);controller.abort();};
  });
  async function downloadRequired(){
    preparing=true; preparationError='';
    try{
      let response=await fetch('/api/models/prepare',{method:'POST',
        headers:{'content-type':'application/json'},body:JSON.stringify(config)});
      let next=await response.json();
      if(!response.ok) throw new Error(Array.isArray(next.detail)?next.detail.map(e=>e.message).join('; '):next.detail||'Could not download required models');
      preparation=next;
    }catch(error){preparationError=error.message;}
    finally{preparing=false;}
  }
  let preparationId=$derived(preparation?.id);
  $effect(()=>{
    let id=preparationId;
    if(!id) return;
    let controller=new AbortController(),timer,stopped=false;
    async function refresh(){
      try{
        let response=await fetch(`/api/models/prepare/${id}`,{signal:controller.signal});
        if(!response.ok) throw new Error('Could not read download progress');
        let next=await response.json();
        if(stopped) return;
        preparation=next;
        if(['queued','running'].includes(next.state)) timer=setTimeout(refresh,500);
        else requiredRevision+=1;
      }catch(error){if(!stopped&&error.name!=='AbortError') preparationError=error.message;}
    }
    refresh();
    return()=>{stopped=true;clearTimeout(timer);controller.abort();};
  });
  function fieldEnabledForConfig(entry){
    if(!conditionsMatch(entry.schema,config))return false;
    const reference=config?.disco?.sampler==='distilled'&&config.disco.distilled_guided===false;
    if(entry.section==='disco'&&entry.name==='quality_lora_strength'&&!config?.disco?.quality_lora_enabled) return false;
    if(entry.section==='disco'&&['distilled_guided','lora_strength'].includes(entry.name)&&config?.disco?.sampler!=='distilled') return false;
    if(activeModelFamily==='disco'&&!reference&&entry.section==='disco'
        &&['steps','style_strength'].includes(entry.name)) return false;
    if(reference&&activeModelFamily==='disco'){
      if(entry.section==='disco'&&!['sampler','distilled_guided','lora_strength','quality_lora_enabled','steps','checkpoint_path','few_step_lora_path','preview_interval','fixed_seed','use_secondary_model'].includes(entry.name)) return false;
      if(entry.section==='diffusion'&&!['seed','fixed_seed_randomness','fixed_seed_wiggle','fixed_seed_wiggle_direction'].includes(entry.name)) return false;
      if(entry.section==='turbo') return false;
      if(entry.section==='main'&&['text_prompts','negative_prompts'].includes(entry.name)) return false;
    }
    if(entry.section!=='diffusion') return true;
    if(entry.name==='unet_cache_interval')
      return config?.diffusion?.unet_cache==='deepcache';
    if(entry.name==='unet_cache_threshold')
      return config?.diffusion?.unet_cache==='first_block';
    return true;
  }
  function fieldValue(item){
    const section=item.section==='main'?config:config?.[item.section];
    if(item.section==='disco'&&item.name==='cut_icgray_schedule')
      return section?.cut_icgray_schedule ?? {0:section?.cut_icgray_p};
    return readFieldValue(config,item);
  }
  function comfyModelInfo(item){
    const row=comfyModels.find(row=>row.section===item.section&&row.field===item.name);
    return {...row,suggested:schema.comfy_model_suggestions?.[config.model_version]?.[item.name]
      ??row?.suggested??'',loading:requiredLoading};
  }
  let entries=$derived(allEntries.filter(
    e=>e.tier!=='hidden'&&!(e.section==='main'&&e.name==='model_version')
      &&fieldSupportsModel(e.schema,activeModelFamily)&&fieldEnabledForConfig(e)));
  // Groups for the selected tab, preserving field order within each.
  let groups=$derived.by(()=>{
    let out=[];
    for(let e of entries.filter(e=>inWorkflow(e,selected))){
      const name=fieldGroup(e,selected);
      let g=out.find(g=>g.name===name);
      if(!g) out.push(g={name,items:[],editor:fieldEditor(e,selected)});
      g.items.push(e);
    }
    // Order by what ui_layout DECLARED, not by the order fields sit on RunConfig -- the
    // latter buried "Input Video" behind "Output" and "Model" on the Project tab.
    let order=tabs.find(t=>t.id===selected)?.groups ?? [];
    for(const group of out)group.items.sort((a,b)=>(a.schema.workspace_order??999)-(b.schema.workspace_order??999));
    if(order.length) out.sort((a,b)=>{
      let ia=order.indexOf(a.name), ib=order.indexOf(b.name);
      return (ia<0?999:ia)-(ib<0?999:ib);
    });
    return out;
  });

  // Search spans EVERY tab. With ~190 fields across 4 tabs and Advanced collapsed by
  // default, finding `softcap_thresh` otherwise means guessing which of 13 groups it is in.
  let query=$state('');
  const pretty=(s)=>s.replaceAll('_',' ');
  let hits=$derived.by(()=>{
    let q=query.trim().toLowerCase();
    if(!q) return [];
    return searchFields(entries,q);
  });
  const tierLabel=(id)=>tabs.find(t=>t.id===id)?.label||id;

  // Validation errors arrive as {path:'video.video_init_path'} — the same shape as the
  // layout keys, so we can send you straight to the offending field instead of making you
  // guess which of 4 tabs and 13 collapsed groups it is hiding in.
  function locate(path){
    if(!path) return null;
    let dot=path.indexOf('.');
    let section=dot<0?'main':path.slice(0,dot), name=dot<0?path:path.slice(dot+1).split('.')[0];
    return entries.find(e=>e.section===section&&e.name===name)||null;
  }
  // 'section.field' -> message, so a field can show its OWN error instead of the sidebar
  // being the only place that knows.
  let errorFor=$derived.by(()=>{
    let map={};
    for(let e of errors){
      let hit=locate(e.path);
      if(hit) map[`${hit.section}.${hit.name}`]=e.message;
    }
    return map;
  });
  const fieldError=(item)=>errorFor[`${item.section}.${item.name}`]||'';
  // Bespoke editors (VideoEditor, ControlNetEditor) render their own inputs, so they need
  // the errors for their whole section rather than one field at a time.
  const sectionErrors=(section)=>Object.fromEntries(
    Object.entries(errorFor).filter(([k])=>k.startsWith(section+'.'))
                            .map(([k,v])=>[k.slice(section.length+1),v]));
  // How many errors each TAB holds, so a collapsed Advanced group can't hide one from you.
  let errorsPerTier=$derived.by(()=>{
    let count={};
    for(let e of errors){
      let hit=locate(e.path);
      if(hit){let location=fieldLocation(hit);count[location]=(count[location]||0)+1;}
    }
    return count;
  });

  async function goToError(path){
    let hit=locate(path);
    if(!hit) return;
    inspectorHidden=false;selected=fieldLocation(hit);
    mode='render';mobileView='settings';viewerMode='compare';
    query='';
    setGroupOpen(selected,fieldGroup(hit,selected),true);
    await tick();
    // Generic fields and custom editors expose anchors for validation links.
    let node=document.getElementById(`field-${hit.section}-${hit.name}`);
    if(!node) return;
    for(let parent=node.parentElement;parent;parent=parent.parentElement){
      if(parent.tagName==='DETAILS')parent.open=true;
    }
    node.scrollIntoView({behavior:'smooth',block:'center'});
    node.classList.remove('flash'); void node.offsetWidth; node.classList.add('flash');
    node.querySelector('input,textarea,select')?.focus({preventScroll:true});
  }
  // These sections have purpose-built editors; the generic Field grid can't express them.
  const custom={'Input Video':'video','ControlNet':'controlnet','IP-Adapter':'ipadapter',
                'Reference Images':'references'};
  let groupOpen=$state({});
  const groupIsOpen=(tier,name)=>groupOpen[`${tier}:${name}`] ?? !['advanced','all','system'].includes(tier);
  function setGroupOpen(tier,name,open){
    if(groupIsOpen(tier,name)!==open) groupOpen={...groupOpen,[`${tier}:${name}`]:open};
  }
  function setField(name,v){
    if(name==='model_version')modelSelections=rememberModelSelections(config,schema,modelSelections);
    config=name==='model_version'
      ? applyModelDefaults(config,v,schema,modelSelections)
      : {...config,[name]:v};
  }
  async function editProjectName(){
    if(!config)return;
    projectDraft=config.batch_name||'';editingProject=true;
    await tick();projectInput?.focus();projectInput?.select();
  }
  async function finishProjectName(save,blur=false){
    if(!editingProject)return;
    const name=projectDraft.trim();
    if(save&&!name&&!blur)return;
    if(save&&name)setField('batch_name',name);
    editingProject=false;
    if(!blur){await tick();projectTitle?.focus();}
  }
  function setIn(section,name,v){
    if(section==='main') return setField(name,v);
    if(section==='disco'&&name==='sampler'){
      config=switchDiscoSampler($state.snapshot(config),v).config;
      return;
    }
    config={...config,[section]:{...config[section],[name]:v}};
  }
  function fieldHint(item){
    if(item.section==='diffusion'&&item.name==='fixed_seed_wiggle')
      return '0 turns wiggle off. Otherwise sets random shift amplitude or directional roll speed in noise pixels. Disco uses image pixels; SD uses latent pixels (typically 8 image pixels). Set randomness to 0 for shifts only.';
    if(item.section==='diffusion'&&item.name==='fixed_seed_wiggle_direction')
      return '[0, 0] gives random wiggle. Other vectors are normalized and roll every frame: [1, 0] right, [0, 1] down. Fractional cumulative positions round to whole noise pixels, preserving noise values.';
    if(item.section==='diffusion'&&item.name==='fixed_seed_randomness')
      return 'With Fixed seed enabled: 0 keeps the same noise; 1 uses fresh noise after frame zero. Intermediate values preserve Gaussian variance. Seed and frame keep it reproducible.';
    if(item.section==='disco'&&item.name==='cut_max_size')
      return 'Limits the image used to construct cuts before padding and cropping. Final cuts still use each encoder’s required resolution. Try 1024 to reduce VRAM.';
    if(item.section==='main'&&item.name==='comfy_models_dir')
      return 'Local or mounted ComfyUI models directory. Downloads start only from Download required models or Start render.';
    if(item.section==='disco'){
      let hints={
        sampler:'Distilled runs the LoRA inside the guided DDIM/video pipeline. Guidance and frame schedules are preserved. Other step counts resample the original 1000-step noise schedule.',
        distilled_guided:'On: CLIP/SigLIP, frame guidance, masks, flow, turbo and scheduled steps. Off: exact unconditional trained-grid preview; import a reference profile for its compatible settings.',
        few_step_lora_path:'Custom supervised adapter .safetensors export, or a training run directory with best.json. This is not an SD/PEFT LoRA.',
        lora_strength:'Scales the LoRA contribution on top of its trained alpha/rank. 0 disables the adapter, 1 uses trained strength; values above 1 amplify it. Applies during guidance backward too.',
        quality_lora_enabled:'Optional aesthetic adapter. Works alone with DDIM or alongside the turbo adapter with distilled sampling. Keeps your secondary-model, guidance, flow and mask settings.',
        quality_lora_path:'Aesthetic real-image epsilon MSE .safetensors export for Crowson Disco 512. Metadata and complete tensor keys/shapes are validated; no automatic download.',
        quality_lora_strength:'Independent additive quality branch: 0 disables its contribution, 1 is 100%. The turbo strength remains separate.',
        attention:'SDPA is the default. Flash and Sage are optional; gradient paths under Sage use SDPA.',
        preview_interval:'Publish a clean-image estimate every N steps. 0 disables live denoising previews.',
        checkpoint_path:'Optional custom checkpoint. Leave blank to download the selected model on render or when you click Download required models.',
        secondary_path:'Optional custom secondary model. Leave blank to download it when needed, on render or from Download required models.',
        guidance_every:'Evaluate semantic guidance every N sampling steps. 1 guides every step.',
        guidance_noise_cap_percent:'0 disables this cap. 100 limits the combined guidance displacement RMS to one current-noise standard deviation per sampler update. Includes CLIP/SigLIP, frame and regularization losses. Gaussian noise has no fixed maximum. PLMS needs an extra unguided evaluation when restarting history.',
        lock_cutouts_across_frames:'Repeat crop positions across frames; moving subjects are not flow-tracked.',
        cut_overview:'Inference progress (%) → whole-frame cut count per batch. Values hold until the next key; no interpolation.',
        cut_innercut:'Inference progress (%) → random crop count per batch. Values hold until the next key; no interpolation.'
      };
      if(hints[item.name]) return hints[item.name];
    }
    let modelHint=modelFieldHint(config,item.section,item.name);
    if(modelHint) return modelHint;
    if(item.section==='diffusion'&&item.name==='sampler_scale_min_size')
      return 'Minimum longest-side resolution for multiscale passes, in pixels. 0 disables the floor; try 512 for SD1.5.';
    if(item.section==='diffusion'&&item.name==='unet_cache')
      return 'DeepCache refreshes the inner U-Net every N evaluations. First Block Cache reuses a result only when the first encoder activation changes less than its threshold. State resets for every frame.';
    if(item.section==='diffusion'&&item.name==='unet_cache_interval')
      return '2 is the conservative default: alternate a full inner U-Net pass with one cached pass. Larger values are faster but drift more.';
    if(item.section==='diffusion'&&item.name==='unet_cache_threshold')
      return 'Relative first-block change allowed for reuse. 0 only accepts an exact match; start near 0.05 for conservative caching.';
    if(item.section==='diffusion'&&item.name==='compile_unet')
      return 'Compiles the U-Net, active ControlNets, and VAE encode/decode. Artifacts persist under .vibewarp_cache/torchinductor. Each new multiscale, tile, or VAE shape may compile separately. Cannot currently be combined with a U-Net cache.';
    if(item.section!=='diffusion'||item.name!=='sampler_tile_size') return '';
    let vanillaSdxl=(config?.model_version||'').toLowerCase().includes('sdxl')&&!config?.animatediff?.enabled;
    return vanillaSdxl
      ? 'Recommended: 1024 px — vanilla SDXL was trained at 1024 px.'
      : 'Recommended: 512 px — SD1.5 and AnimateDiff models were trained at 512 px.';
  }
  const enabledDot=(section)=>config?.[section]?.enabled||config?.[section]?.flow_warp||config?.[section]?.do_freeunet||config?.[section]?.make_captions||config?.[section]?.use_background_mask;
  function mergeDefaults(defaults,saved){if(!saved||typeof saved!=='object'||Array.isArray(saved))return defaults;let result={...defaults};for(let key of Object.keys(saved)){if(defaults[key]&&typeof defaults[key]==='object'&&!Array.isArray(defaults[key]))result[key]=mergeDefaults(defaults[key],saved[key]);else result[key]=saved[key]}return result}
  async function init(){
    let [configResponse,jobsResponse]=await Promise.all([fetch('/api/config'),fetch('/api/jobs')]);
    let d=await configResponse.json();let saved=null;
    const existing=jobsResponse.ok?await jobsResponse.json():[];
    if(!d.prefilled){try{saved=JSON.parse(localStorage.getItem(storageKey))}catch{localStorage.removeItem(storageKey)}}
    if(saved?.disco&&saved.disco.cut_schedule_version!==2){
      // Migrate before merging defaults, which would otherwise stamp old axes as v2.
      const response=await fetch('/api/config/import',{method:'POST',headers:{'content-type':'application/json'},
        body:JSON.stringify({filename:'browser-settings.json',content:JSON.stringify(saved)})});
      const migrated=await response.json();
      if(!response.ok)throw new Error(migrated.detail?.[0]?.message||'Could not migrate saved cutout schedules');
      saved=migrated.config;
    }
    schema=d.schema;
    try{let savedSelections=JSON.parse(localStorage.getItem(modelSelectionsKey));if(savedSelections&&typeof savedSelections==='object'&&!Array.isArray(savedSelections))modelSelections=savedSelections;}catch{}
    config=normalizeSchedules(dropLegacyFrameHash(normalizeRawOpacities(migrateLegacyFrameLimit(mergeDefaults(d.defaults,saved)))),schema);selected=tabs[0]?.id||'source';initialized=true;
    try{inspectorWidth=Math.max(340,Math.min(680,Number(localStorage.getItem('vibewarp.inspector.width'))||420));}catch{}
    // The renderer lives on the server, so a browser reload must reconnect to
    // it rather than forgetting the only handle that can cancel the active job.
    if(jobsResponse.ok){
      let active=existing.find(item=>item.state==='running')
              || existing.find(item=>item.state==='queued');
      if(active) trackJob(active);
    }
  } init().catch(error=>{initError=error.message||'Could not load the workspace';});
  // Selecting a Comfy-backed model should explain a stopped/misconfigured server now,
  // not after the user submits a render and waits for backend initialization to fail.
  $effect(()=>{
    let target=comfyTarget;
    if(!initialized||!target?.url){comfyStatus=null;return}
    let controller=new AbortController();
    comfyStatus={checking:true,available:false,...target};
    let timer=setTimeout(async()=>{
      try{
        let r=await fetch(`/api/comfy/status?url=${encodeURIComponent(target.url)}`,
                          {signal:controller.signal});
        let result=await r.json();
        comfyStatus={checking:false,...target,...result};
      }catch(error){
        if(error.name!=='AbortError') comfyStatus={checking:false,available:false,...target,
          message:'Could not check the ComfyUI server'};
      }
    },250);
    return()=>{clearTimeout(timer);controller.abort()};
  });
  $effect(()=>{if(initialized&&config){
    localStorage.setItem(storageKey,JSON.stringify(config));
    localStorage.setItem(modelSelectionsKey,JSON.stringify(rememberModelSelections(config,schema,modelSelections)));
  }});
  function listen(id){
    source?.close();source=new EventSource(`/api/jobs/${id}/events`);
    source.onmessage=(event)=>{
      if(job?.id!==id)return;
      const next=JSON.parse(event.data);
      if(next.config){jobConfig=next.config;snapshotError='';snapshotLoading=false;snapshotRevision+=1;}
      job=next;
      if(['completed','failed','cancelled'].includes(job.state)){source.close();requiredRevision+=1;}
    };
  }
  async function loadJobSnapshot(id=job?.id){
    if(!id)return;
    const revision=++snapshotRevision;
    snapshotLoading=true;snapshotError='';
    try{
      const response=await fetch(`/api/jobs/${id}`);
      if(!response.ok)throw Error('Could not restore the submitted render settings.');
      const next=await response.json();
      if(revision!==snapshotRevision||job?.id!==id)return;
      if(!next.config)throw Error('The job snapshot has no submitted settings.');
      jobConfig=next.config;
      // Events may have advanced while the request was in flight.
      if((next.revision??-1)>=(job.revision??-1))job=next;
    }catch(error){if(revision===snapshotRevision&&job?.id===id)snapshotError=error.message;}
    finally{if(revision===snapshotRevision)snapshotLoading=false;}
  }
  function trackJob(next,submitted=null){
    const same=job?.id===next.id;
    jobConfig=next.config||submitted||(same?jobConfig:null);
    job=next;snapshotError='';snapshotRevision+=1;snapshotLoading=false;
    listen(next.id);
    if(!jobConfig)loadJobSnapshot(next.id);
  }
  async function chooseSource(){
    chooseWorkflow('source');setGroupOpen('source','Input Video',true);
    await tick();document.querySelector('#field-video-video_init_path input')?.focus();
  }
  async function toggleInspector(){
    if(inspectorVisible){inspectorHidden=true;mobileView='viewer';}
    else{inspectorHidden=false;mobileView='settings';if(viewerMode==='inspect')viewerMode='compare';await tick();document.querySelector('.settings-search input')?.focus();}
  }
  function viewDraft(){
    if(renderActive)return;
    source?.close();snapshotRevision+=1;snapshotLoading=false;snapshotError='';job=null;jobConfig=null;
  }
  async function run(oneFrame=false){
    if(busy||renderActive||!config||!supportedModel||conditioningConflict)return;
    busy=true;errors=[];
    try{
      const submitted=$state.snapshot(config);
      if(oneFrame)submitted.frame_range=[submitted.frame_range?.[0]??0];
      let r=await fetch('/api/jobs',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(submitted)});
      let d=await r.json();
      if(!r.ok){errors=Array.isArray(d.detail)?d.detail:[{message:d.detail||'Submission failed'}];mobileView='settings';return;}
      trackJob(d,submitted);mobileView='viewer';
    }catch(error){errors=[{message:error.message||'Could not start the render'}];}
    finally{busy=false;}
  }
  async function cancel(){
    if(!renderActive || job.cancel_requested || cancelling) return;
    // Apply the server snapshot immediately; the event stream will carry the
    // terminal state once the renderer and (if applicable) Comfy have stopped.
    cancelling=true;
    try{
      let r=await fetch(`/api/jobs/${job.id}/cancel`,{method:'POST'});
      if(!r.ok) throw new Error('Could not cancel render');
      job=await r.json();
    }catch(error){errors=[{message:error.message}];}
    finally{cancelling=false;}
  }
  // System-tier fields describe the MACHINE (model paths, VRAM, threads), so they are
  // stripped from an export and never travel to someone else's install.
  let systemFields=$derived(allEntries.filter(e=>e.tier==='system'));
  function withoutSystem(c){
    let out=structuredClone($state.snapshot(c));
    for(let e of systemFields){
      if(e.section==='main') delete out[e.name];
      else if(out[e.section]) delete out[e.section][e.name];
    }
    return out;
  }
  function exportConfig(){let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(withoutSystem(config),null,2)],{type:'application/json'}));a.download=`${config.batch_name||'vibewarp'}.json`;a.click();URL.revokeObjectURL(a.href)}
  // Persist the system tier server-side (system.json next to the install) whenever it
  // changes, so it survives a settings import, a reset, and a new browser.
  let systemSnapshot=$state(null);
  $effect(()=>{
    if(!initialized||!config||!systemFields.length) return;
    let current=JSON.stringify(systemFields.map(e=>e.section==='main'?config[e.name]:config[e.section]?.[e.name]));
    if(systemSnapshot===null){systemSnapshot=current;return}   // don't re-save what we just loaded
    if(current===systemSnapshot) return;
    systemSnapshot=current;
    fetch('/api/system',{method:'PUT',headers:{'content-type':'application/json'},body:JSON.stringify($state.snapshot(config))}).catch(()=>{});
  });
  async function importConfig(e){let f=e.target.files[0];if(!f)return;busy=true;errors=[];let content=await f.text();let r=await fetch('/api/config/import',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({filename:f.name,content})});let d=await r.json();busy=false;if(!r.ok){errors=Array.isArray(d.detail)?d.detail:[{message:d.detail||'Import failed'}];return}config=normalizeSchedules(d.config,schema);viewDraft();e.target.value=''}
  async function resetConfig(){localStorage.removeItem(storageKey);let r=await fetch('/api/config');let d=await r.json();config=normalizeSchedules(d.defaults,schema);viewDraft();errors=[];selected=tabs[0]?.id||'source'}
</script>
<svelte:head><title>VibeWarp</title><meta name="description" content="Local video diffusion workspace"/></svelte:head>
<svelte:window onpointermove={resizeInspector} onpointerup={()=>resizing=false}/>
{#snippet editField(item)}
  {#if !(item.section==='main'&&item.name==='model_version')}
    <Field name={item.name} schema={item.schema} path={item.section}
      modelInfo={comfyModelInfo(item)}
      hint={fieldHint(item)} error={fieldError(item)} value={fieldValue(item)}
      onchange={(value)=>setIn(item.section,item.name,value)}/>
  {/if}
{/snippet}
<div class="app-shell">
  <a class="skip-link" href="#workspace">Skip to workspace</a>
  <header class="topbar">
    <div class="topbar-left">
    <a class="brand" href="#workspace" aria-label="VibeWarp workspace"><span class="mark">V</span><strong>VibeWarp</strong></a>
    {#if config}
      <label class="model-picker"><span class="visually-hidden">Model Version</span><select value={config.model_version} onchange={(event)=>setField('model_version',event.currentTarget.value)}>
        {#if !supportedModel}<option value={config.model_version} disabled>{config.model_version} · legacy settings</option>{/if}
        {#each modelGroups as group}
          {#if group.id}<optgroup label={group.label}>{#each modelOptions.filter(option=>option.group===group.id) as option}<option value={option.value}>{option.label}</option>{/each}</optgroup>
          {:else}{#each modelOptions as option}<option value={option.value}>{option.label}</option>{/each}{/if}
        {/each}
      </select></label>
    {/if}
    </div>
    <div class="project-title">
      {#if editingProject}<input bind:this={projectInput} bind:value={projectDraft} aria-label="Edit project name"
        onblur={()=>finishProjectName(true,true)} onkeydown={(event)=>{
          if(event.key==='Enter'){event.preventDefault();finishProjectName(true);}
          if(event.key==='Escape'){event.preventDefault();finishProjectName(false);}
        }}/>
      {:else}<button bind:this={projectTitle} onclick={editProjectName} disabled={!config}
        aria-label="Edit project name" title={`${config?.batch_name||'Your project'} · Click to edit`}>{config?.batch_name||'Your project'}</button>{/if}
    </div>
    <div class="topbar-actions">
    <details class="project-menu" use:closeOnSelection>
      <summary aria-label="Project settings"><span>Project settings</span></summary>
      <div class="project-actions">
        <button onclick={()=>fileInput?.click()}>Import settings</button>
        <button onclick={exportConfig} disabled={!config}>Export settings</button>
        <button onclick={resetConfig} disabled={!initialized}>Reset settings</button>
      </div>
    </details>
    <input class="visually-hidden" tabindex="-1" type="file" accept=".json,.txt" bind:this={fileInput} onchange={importConfig} aria-label="Import settings file"/>
    <span class="topbar-state"><i class:live={renderActive}></i>{renderActive?'Rendering':missingModels?'Setup needed':'Local workspace'}</span>
    <button class="render-one" onclick={()=>run(true)} title="Test the style on the first frame of the selected range"
      disabled={renderActive||busy||!config||!supportedModel||!!conditioningConflict}>Render 1 frame</button>
    <button class="run" class:cancel={renderActive} onclick={()=>renderActive?cancel():run()}
      disabled={renderActive?(cancelling||job.cancel_requested):busy||!config||!supportedModel||!!conditioningConflict}>
      {renderActive?(cancelling||job.cancel_requested?'Cancelling…':'Cancel render'):busy?'Submitting…':'Start render'}
    </button>
    </div>
  </header>
  <div class="workspace-tools" role="navigation" aria-label="Workspace navigation">
    <button aria-pressed={mode==='render'} onclick={()=>{mode='render';mobileView='viewer';}}>Workspace</button>
    <button aria-pressed={mode==='preview'} onclick={()=>{mode='preview';mobileView='viewer';}}>Runs</button>
    {#if jobConfig}<span class="active-render-context">Viewing {jobConfig.batch_name} · {modelOptions.find(option=>option.value===jobConfig.model_version)?.label||jobConfig.model_version}</span>{/if}
    {#if job&&!renderActive}<button onclick={viewDraft}>View draft</button>{/if}
    <button class="inspector-toggle" disabled={mode==='preview'} aria-expanded={inspectorVisible} aria-controls="settings-inspector" onclick={toggleInspector}>{inspectorVisible?'Hide settings':'Show settings'}</button>
  </div>
  <main id="workspace" class="workspace" class:inspector-hidden={inspectorHidden} class:history-mode={mode==='preview'} class:inspecting={viewerMode==='inspect'} class:mobile-settings={mobileView==='settings'} style={`--inspector-width:${inspectorWidth}px`}>
    <nav class="workflow-nav" aria-label="Workflow">
      <small>Project workflow</small>
      {#each tabs.filter(tab=>tab.id!=='system') as tab,index}
        <button class:active={mode==='render'&&selected===tab.id} onclick={()=>chooseWorkflow(tab.id)} aria-current={mode==='render'&&selected===tab.id?'page':undefined}>
          <span class="nav-index" aria-hidden="true">{String(index+1).padStart(2,'0')}</span><span>{tab.label}</span>{#if errorsPerTier[tab.id]}<em>{errorsPerTier[tab.id]}</em>{/if}
        </button>
      {/each}
      <div class="nav-bottom">
        <button class:active={mode==='preview'} onclick={()=>{mode='preview';mobileView='viewer';}} aria-current={mode==='preview'?'page':undefined}>Runs &amp; settings</button>
        {#each tabs.filter(tab=>tab.id==='system') as tab}<button class:active={mode==='render'&&selected===tab.id} onclick={()=>chooseWorkflow(tab.id)}>{tab.label}</button>{/each}
      </div>
    </nav>
    <section class="viewer-pane" aria-label="Video workspace">
      {#if viewerConfig}<Preview config={viewerConfig} {job} history={mode==='preview'} bind:view={viewerMode}
        onload={(next)=>{config=normalizeSchedules(next,schema);errors=[];chooseWorkflow(selected);viewerMode='compare';if(!renderActive){job=null;jobConfig=null;}}}
        onjob={trackJob} onsource={chooseSource}/>
      {:else}<div class="workspace-empty" role="status">{#if snapshotError}<span>{snapshotError}</span><button onclick={()=>loadJobSnapshot()}>Retry render snapshot</button>{:else}{snapshotLoading?'Restoring submitted render…':initError||'Loading workspace…'}{/if}</div>{/if}
    </section>
    <div class="inspector-resizer" role="slider" tabindex="0" aria-label="Settings inspector width" aria-orientation="horizontal" aria-valuemin="340" aria-valuemax="680" aria-valuenow={inspectorWidth}
      onpointerdown={(event)=>{event.preventDefault();resizing=true;event.currentTarget.setPointerCapture?.(event.pointerId);}} onkeydown={resizeKey}></div>
    <aside id="settings-inspector" class="settings-inspector" aria-label="Settings inspector">
      <div class="inspector-head">
        <p>{selected==='system'?'System settings':config?.model_version?modelOptions.find(option=>option.value===config.model_version)?.label||config.model_version:'Configuration'}</p>
        <h2>{query.trim()?`${hits.length} matches`:tabs.find(tab=>tab.id===selected)?.label||'Settings'} {#if workflowHelp[selected]&&!query}<HelpTip label="Help: Settings section" text={workflowHelp[selected]}/>{/if}</h2>
        <label class="mobile-workflow"><span>Settings section</span><select value={selected} onchange={(event)=>chooseWorkflow(event.currentTarget.value)}>{#each tabs as tab}<option value={tab.id}>{tab.label}</option>{/each}</select></label>
        <label class="settings-search"><span class="visually-hidden">Search settings</span><input type="search" placeholder="Search settings…" bind:value={query} onkeydown={(event)=>{if(event.key==='Escape')query='';}}/></label>

      </div>
      <div class="inspector-content">
        {#if !supportedModel}<div class="notice warning" role="status">This model is retained in your imported settings but is no longer offered in the UI. Select a supported model above to continue.</div>{/if}
        {#if renderActive}<div class="notice">Editing the next render. The active job uses its submitted settings.</div>{/if}
        {#if errors.length}<div class="validation-errors" role="alert">{#each errors as error}
          <button onclick={()=>goToError(error.path)}><strong>{error.message}</strong>{#if error.path}<small>{error.path}</small>{/if}</button>
        {/each}</div>{/if}
        {#if comfyStatus&&(selected==='model'||selected==='system'||!schema?.workflows)}
          <section class="setup-state" aria-label="Backend connection">
            <h3>{comfyStatus.label} backend</h3>
            <dl><div><dt>ComfyUI server</dt><dd>{comfyStatus.checking?'Checking…':comfyStatus.available?'Reachable':'Unavailable'}</dd></div><div><dt>API compatibility</dt><dd>Not verified</dd></div></dl>
            <small>{comfyStatus.url}</small>
            {#if !comfyStatus.checking&&!comfyStatus.available}<p class="error">Start ComfyUI or check the configured server URL.</p>{/if}
            <button onclick={()=>goToError(comfyStatus.path)}>Server settings</button>
          </section>
        {/if}
        {#if selected==='model'||!schema?.workflows}
          <section class="model-readiness" aria-label="Model setup">
            <h3>Required model files</h3>
            {#if !requiredLoading&&!requiredError&&!missingModels}<p class="setup-success">Required files are present.</p>{/if}
            <RequiredModels files={requiredModels} loading={requiredLoading} error={requiredError}
              busy={preparing||['queued','running'].includes(preparation?.state)||renderActive} ondownload={downloadRequired}/>
            <DownloadProgress files={downloadFiles} onaction={controlDownload}/>
            {#if preparation?.error||preparationError}<p class="error" role="alert">{preparation?.error||preparationError}</p>{/if}
          </section>
        {/if}
        {#if conditioningConflict}<div class="notice warning" role="alert"><strong>Resolve this combination before rendering</strong><p>{conditioningConflict}</p><button onclick={()=>setIn('contact_sheet','mode','off')}>Turn off contact sheets</button></div>{/if}
        {#if query.trim()}
          {#if !hits.length}<div class="workspace-empty">No editable settings match “{query}”.</div>{/if}
          {#each hits as item (item.section+'.'+item.name)}
            {#if fieldEditor(item,fieldLocation(item))||custom[fieldGroup(item,fieldLocation(item))]}<button class="search-result" onclick={()=>{chooseWorkflow(fieldLocation(item));setGroupOpen(selected,fieldGroup(item,selected),true);}}><b>{item.schema.label||pretty(item.name)}</b><small>{tierLabel(fieldLocation(item))} / {fieldGroup(item,fieldLocation(item))}</small></button>
            {:else}<div class="search-result-field"><small>{tierLabel(fieldLocation(item))} / {fieldGroup(item,fieldLocation(item))}</small>{@render editField(item)}</div>{/if}
          {/each}
        {:else}
          <div class="groups">
            {#each groups as group (selected+':'+group.name)}
              {@const only=group.editor||custom[group.name]}
              <details class="panel group" data-tier={selected} open={groupIsOpen(selected,group.name)} ontoggle={(event)=>setGroupOpen(event.currentTarget.dataset.tier,group.name,event.currentTarget.open)}>
                <summary><span>{group.name}</span>{#if only&&enabledDot(only)}<i></i>{/if}</summary>
                <div class="group-fields" id={only&&only!=='video'&&group.items[0]?`field-${group.items[0].section}-${group.items[0].name}`:undefined}>
                  {#if only==='video'}
                    <VideoEditor value={config.video} errors={sectionErrors('video')} modelVersion={config.model_version}
                      dimensionMultiple={schema.comfy_image_dimension_multiples?.[config.model_version]}
                      discoReference={config.disco?.sampler==='distilled'&&config.disco?.distilled_guided===false}
                      onchange={(value)=>setField('video',value)} frameRange={config.frame_range} onFrameRange={(value)=>setField('frame_range',value)}/>
                    {#each group.items.filter(item=>item.name==='keyframes_relative_to_frame_range') as item}{@render editField(item)}{/each}
                  {:else if only==='controlnet'}
                    <ControlNetEditor value={config.controlnet} schema={schema.properties.controlnet} modelVersion={config.model_version} onchange={(value)=>setField('controlnet',value)}/>
                  {:else if only==='ipadapter'}
                    <IPAdapterEditor value={config.ipadapter} schema={schema.properties.ipadapter} onchange={(value)=>setField('ipadapter',value)} modelVersion={config.model_version}
                      modelDir={config.controlnet?.model_dir} videoPath={config.video?.video_init_path} frameRange={config.frame_range} extractNth={config.video?.extract_nth_frame}/>
                  {:else if only==='references'}
                    {#each group.items.filter(item=>item.schema.group_editor_extra) as item (item.section+'.'+item.name)}{@render editField(item)}{/each}
                    {@const refItem=group.items.find(item=>item.name==='references')}
                    {#if refItem}<ReferenceImagesEditor value={config[refItem.section]?.references} onchange={(value)=>setIn(refItem.section,'references',value)}
                      labelOpacity={config.reference_label_opacity} onLabelOpacityChange={(value)=>setField('reference_label_opacity',value)}
                      labelFontScale={config.reference_label_font_scale??1} onLabelFontScaleChange={value=>setField('reference_label_font_scale',value)}
                      labelBadgeHeight={config.reference_label_badge_height??0.12} onLabelBadgeHeightChange={value=>setField('reference_label_badge_height',value)}
                      videoPath={config.video?.video_init_path} frameRange={config.frame_range} extractNth={config.video?.extract_nth_frame}
                      maxReferences={schema.max_edit_references?.[activeModelFamily]??10} sourceLabels={schema.edit_reference_sources}/>{/if}
                  {:else}
                    {#each group.items.filter(item=>!item.schema.workspace_advanced) as item (item.section+'.'+item.name)}{@render editField(item)}{/each}
                    {#if group.items.some(item=>item.schema.workspace_advanced)}
                      <details class="expert-fields" open={group.items.every(item=>item.schema.workspace_advanced)}><summary>Advanced controls</summary><div>{#each group.items.filter(item=>item.schema.workspace_advanced) as item (item.section+'.'+item.name)}{@render editField(item)}{/each}</div></details>
                    {/if}
                    {#if group.items.some(item=>item.schema.preview==='raw_video')}<RawVideoPreview {config} {job} onjob={trackJob}/>{/if}
                  {/if}
                </div>
              </details>
            {/each}
          </div>
        {/if}
        {#if selected==='model'&&!seenFirstFrame}<details class="supporter-details" open><summary>Support VibeWarp</summary><Supporters/></details>{/if}
      </div>
    </aside>
  </main>
  <div class="mobile-tabs" role="tablist" aria-label="Workspace views">
    <button role="tab" aria-selected={mobileView==='viewer'&&mode==='render'} onclick={()=>{mobileView='viewer';mode='render';}}>Viewer</button>
    <button role="tab" aria-selected={mobileView==='settings'} onclick={()=>{mobileView='settings';mode='render';if(viewerMode==='inspect')viewerMode='compare';}}>Settings</button>
    <button role="tab" aria-selected={mode==='preview'&&mobileView!=='settings'} onclick={()=>{mode='preview';mobileView='viewer';}}>Runs</button>
  </div>
  <footer class="statusbar" aria-label="Render status">
    <span class="status-dot" class:live={renderActive} class:failed={job?.state==='failed'||errors.length}></span>
    {#if errors.length}<button class="status-link error" onclick={()=>goToError(errors[0].path)}>{errors.length} input error{errors.length===1?'':'s'}</button>
    {:else}<span class="status-message">{job?.message||'Ready for the next render'}</span>{/if}
    {#if job}<span class="status-numbers">{Math.round(job.progress||0)}%{#if job.total_frames} · {job.frame} / {job.total_frames}{/if}</span>{/if}
    {#if eta!==null}<span class="status-eta" title="Estimated time remaining from render progress logs">{eta===0&&job.eta_seconds>0?'ETA updating…':`ETA ${formatEta(eta)}`}</span>{/if}
    <button class="status-link" onclick={openModelSetup}>{pendingDownloads?`${pendingDownloads} model downloads`:missingModels?`${missingModels} missing model files`:'Model setup'}</button>
    <button class="log-toggle" onclick={()=>logCollapsed=!logCollapsed} aria-expanded={!logCollapsed} aria-controls="output-log">{logCollapsed?'Show log':'Hide log'}</button>
  </footer>
  {#if !logCollapsed}<OutputLog logs={job?.logs||[]} revision={job?.revision||0}/>{/if}
</div>
