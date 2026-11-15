import {beforeEach,afterEach,expect,it,vi} from 'vitest';
import {render,screen,fireEvent,waitFor,within} from '@testing-library/svelte';
import App from './App.svelte';

const field=(type,tier,group,extra={})=>({type,tier,group,...extra});
const schema={tiers:[{id:'project',label:'Project',groups:['Model']},{id:'advanced',label:'Advanced',groups:['Disco Diffusion']}],
  model_family_by_version:{disco_512:'disco',disco_256:'disco'},
  properties:{model_version:field('string','project','Model',{choices:['disco_512','disco_256']}),
    disco:{type:'dataclass',properties:{checkpoint_path:field('string','project','Model',{label:'Disco checkpoint'}),
      steps_schedule:field('any','advanced','Disco Diffusion')}}}};
const defaults={model_version:'disco_512',root_dir:'C:/project',model_path:'',sd_checkpoint_path:'',
  output_dir:'images_out',batch_name:'test',video:{},controlnet:{},ipadapter:{},flow:{flow_warp:false},
  disco:{sampler:'ddim',checkpoint_path:'',secondary_path:'',encoders:['ViT-B-32'],use_secondary_model:true,steps:50}};
const response=data=>({ok:true,json:async()=>data});
let fetchMock;
const requests=(path,method='POST')=>fetchMock.mock.calls.filter(([url,options])=>url===path&&options?.method===method);
beforeEach(()=>{
  localStorage.clear();
  vi.stubGlobal('EventSource',class{close(){}});
  fetchMock=vi.fn(async(url,options)=>{
    if(url==='/api/config')return response({schema,defaults:structuredClone(defaults),prefilled:true});
    if(url==='/api/jobs'&&options?.method==='POST')return response({id:'render1',state:'queued',logs:[]});
    if(url==='/api/jobs')return response([]);
    if(url==='/api/models/required')return response({files:[{name:'weights.pt',path:'models/weights.pt',status:'missing',downloadable:true}]});
    if(url==='/api/models/downloads')return response({files:[],tasks:[]});
    if(url==='/api/models/prepare')return response({id:'download1',state:'queued',files:[]});
    if(url==='/api/models/prepare/download1')return response({id:'download1',state:'completed',files:[]});
    if(String(url).startsWith('/api/fs/exists'))return response({exists:true});
    throw new Error(`Unexpected fetch ${url}`);
  });
  vi.stubGlobal('fetch',fetchMock);
});
afterEach(()=>vi.unstubAllGlobals());

it('lists required models without transferring, then downloads only on click',async()=>{
  render(App);
  await screen.findByText('weights.pt');
  expect(requests('/api/models/required')).toHaveLength(1);
  expect(requests('/api/models/prepare')).toHaveLength(0);
  await fireEvent.change(screen.getByRole('combobox',{name:'Model Version'}),{target:{value:'disco_256'}});
  await waitFor(()=>expect(requests('/api/models/required')).toHaveLength(2));
  expect(requests('/api/models/prepare')).toHaveLength(0);
  await fireEvent.click(screen.getByRole('button',{name:'Download required models'}));
  await waitFor(()=>expect(requests('/api/models/prepare')).toHaveLength(1));
  expect(JSON.parse(requests('/api/models/prepare')[0][1].body).model_version).toBe('disco_256');
});

it('keeps accordion choices through edits and tab changes',async()=>{
  const {container}=render(App);
  await screen.findByText('weights.pt');
  let project=container.querySelector('details[data-tier="project"]');
  project.open=false;
  await fireEvent(project,new Event('toggle'));
  await fireEvent.click(screen.getByRole('button',{name:'Advanced'}));
  const advanced=container.querySelector('details[data-tier="advanced"]');
  expect(advanced.open).toBe(false);
  advanced.open=true;
  await fireEvent(advanced,new Event('toggle'));
  expect(screen.getByRole('spinbutton',{name:'Value'})).toHaveValue(50);
  await fireEvent.change(screen.getByRole('spinbutton',{name:'Value'}),{target:{value:'40'}});
  expect(advanced.open).toBe(true);
  expect(screen.getByRole('spinbutton',{name:'Value'})).toHaveValue(40);
  await fireEvent.click(screen.getByRole('button',{name:'Project'}));
  expect(container.querySelector('details[data-tier="project"]').open).toBe(false);
  await fireEvent.click(screen.getByRole('button',{name:'Advanced'}));
  expect(container.querySelector('details[data-tier="advanced"]').open).toBe(true);
  expect(requests('/api/models/prepare')).toHaveLength(0);
});

it('starts render through the job endpoint without selection-triggered preparation',async()=>{
  render(App);
  await screen.findByText('weights.pt');
  await fireEvent.click(screen.getByRole('button',{name:'Start render'}));
  await waitFor(()=>expect(requests('/api/jobs')).toHaveLength(1));
  expect(requests('/api/models/prepare')).toHaveLength(0);
});

it.each(['queued','running'])('switches the main render button to cancel while %s',async(state)=>{
  const streams=[];
  vi.stubGlobal('EventSource',class{constructor(){streams.push(this);}close(){}});
  const original=fetchMock.getMockImplementation();
  fetchMock.mockImplementation(async(url,options)=>url==='/api/jobs/render1/cancel'
    ?response({id:'render1',state,cancel_requested:true,logs:[]})
    :original(url,options));
  render(App);
  await screen.findByText('weights.pt');
  await fireEvent.click(screen.getByRole('button',{name:'Start render'}));
  await waitFor(()=>expect(streams).toHaveLength(1));
  if(state==='running') streams[0].onmessage({data:JSON.stringify({id:'render1',state,logs:[]})});
  const button=await screen.findByRole('button',{name:'Cancel render'});
  expect(button).toBeEnabled();
  expect(screen.queryByRole('button',{name:'Start render'})).not.toBeInTheDocument();
  await fireEvent.click(button);
  await waitFor(()=>expect(requests('/api/jobs/render1/cancel')).toHaveLength(1));
  expect(await screen.findByRole('button',{name:'Cancelling…'})).toBeDisabled();
  streams[0].onmessage({data:JSON.stringify({id:'render1',state:'cancelled',logs:[]})});
  expect(await screen.findByRole('button',{name:'Start render'})).toBeEnabled();
  expect(requests('/api/jobs')).toHaveLength(1);
});

it('shows the global queue and sends controls only for the selected model',async()=>{
  const original=fetchMock.getMockImplementation();
  let files=[{id:'active',path:'models/a',name:'active.pt',status:'downloading',downloaded:5,total:10,can_pause:true,can_cancel:true},
    {id:'queued',path:'models/b',name:'queued.pt',status:'queued',downloaded:0,can_pause:true,can_cancel:true}];
  fetchMock.mockImplementation(async(url,options)=>{
    if(url==='/api/models/downloads')return response({files,tasks:[]});
    if(url==='/api/models/downloads/active/pause'){
      files=files.map(file=>file.id==='active'?{...file,status:'paused',can_pause:false,can_resume:true}:file);
      return response(files[0]);
    }
    return original(url,options);
  });
  render(App);
  await screen.findByRole('button',{name:'Pause active.pt'});
  expect(screen.getByRole('button',{name:'Cancel queued.pt'})).toBeInTheDocument();
  await fireEvent.click(screen.getByRole('button',{name:'Pause active.pt'}));
  await screen.findByRole('button',{name:'Resume active.pt'});
  expect(requests('/api/models/downloads/active/pause')).toHaveLength(1);
  expect(requests('/api/models/downloads/queued/pause')).toHaveLength(0);
  expect(requests('/api/models/prepare')).toHaveLength(0);
});

it('keeps distilled selection across tabs and submits the adapter configuration',async()=>{
  const original=fetchMock.getMockImplementation();
  const discoSchema=structuredClone(schema);
  discoSchema.tiers.push({id:'render',label:'Render',groups:['Disco Diffusion']});
  discoSchema.properties.disco.properties.sampler=field('string','render','Disco Diffusion',
    {choices:['ddim','plms','distilled']});
  discoSchema.properties.disco.properties.steps=field('integer','render','Disco Diffusion');
  discoSchema.properties.disco.properties.use_secondary_model=field('boolean','render','Disco Diffusion',{label:'Use secondary model'});
  discoSchema.properties.disco.properties.distilled_guided=field('boolean','render','Disco Diffusion',{label:'Use guidance'});
  discoSchema.properties.disco.properties.lora_strength=field('number','render','Disco Diffusion',{label:'LoRA strength'});
  discoSchema.disco_distilled_preset={model_version:'disco_512',
    disco:{sampler:'distilled',distilled_guided:true,use_secondary_model:false}};
  discoSchema.disco_guided_restore={model_version:'disco_512',
    disco:{sampler:'ddim',distilled_guided:true,use_secondary_model:true}};
  const guidedDefaults={...structuredClone(defaults),text_prompts:{0:'my painting'},
    video:{width:1024,height:768,max_size:1024},
    disco:{...structuredClone(defaults.disco),distilled_guided:true,steps_schedule:[6],clip_guidance_scale:5000,lora_strength:1,
      few_step_lora_path:'student-1680step.safetensors'}};
  fetchMock.mockImplementation(async(url,options)=>url==='/api/config'
    ?response({schema:discoSchema,defaults:guidedDefaults,prefilled:true})
    :original(url,options));
  render(App);
  await screen.findByText('weights.pt');
  await fireEvent.click(within(screen.getByRole('navigation',{name:'Workflow'})).getByRole('button',{name:'Render',exact:true}));
  await fireEvent.change(screen.getByRole('combobox',{name:/^Sampler/}),{target:{value:'distilled'}});
  await fireEvent.click(screen.getByRole('button',{name:'Project',exact:true}));
  await fireEvent.click(within(screen.getByRole('navigation',{name:'Workflow'})).getByRole('button',{name:'Render',exact:true}));
  expect(screen.getByRole('combobox',{name:/^Sampler/})).toHaveValue('distilled');
  expect(screen.getByRole('checkbox',{name:'Use secondary model'})).toBeChecked();
  expect(screen.getByRole('checkbox',{name:/^Use guidance/})).toBeChecked();
  expect(screen.queryByRole('spinbutton',{name:'Steps'})).not.toBeInTheDocument();
  await fireEvent.change(screen.getByRole('spinbutton',{name:/^LoRA strength/}),{target:{value:'0.5'}});
  await fireEvent.change(screen.getByRole('combobox',{name:/^Sampler/}),{target:{value:'plms'}});
  expect(JSON.parse(localStorage.getItem('vibewarp.config.v1')).text_prompts).toEqual({0:'my painting'});
  await fireEvent.change(screen.getByRole('combobox',{name:/^Sampler/}),{target:{value:'distilled'}});
  await fireEvent.click(screen.getByRole('button',{name:'Start render'}));
  await waitFor(()=>expect(requests('/api/jobs')).toHaveLength(1));
  const payload=JSON.parse(requests('/api/jobs')[0][1].body);
  expect(payload.disco.sampler).toBe('distilled');
  expect(payload.disco.few_step_lora_path).toBe('student-1680step.safetensors');
  expect(payload.video).toMatchObject({width:1024,height:768,max_size:1024});
  expect(payload.text_prompts).toEqual({0:'my painting'});
  expect(payload.disco).toMatchObject({distilled_guided:true,steps_schedule:[6],
    encoders:['ViT-B-32'],clip_guidance_scale:5000,use_secondary_model:true,lora_strength:.5});
});

it.each([true,false])('enables quality LoRA while retaining secondary/guidance choices (%s)',async(enabled)=>{
  const original=fetchMock.getMockImplementation();
  const qualitySchema=structuredClone(schema);
  qualitySchema.tiers.push({id:'render',label:'Render',groups:['Disco Diffusion']});
  Object.assign(qualitySchema.properties.disco.properties,{
    use_secondary_model:field('boolean','render','Disco Diffusion',{label:'Use secondary model'}),
    distilled_guided:field('boolean','render','Disco Diffusion',{label:'Use guidance'}),
    quality_lora_enabled:field('boolean','render','Disco Diffusion',{label:'Enable quality LoRA'}),
    quality_lora_strength:field('number','render','Disco Diffusion',{label:'Quality LoRA strength'}),
    quality_lora_path:field('string','project','Model',{label:'Quality LoRA file'})});
  const config={...structuredClone(defaults),model_version:'disco_512',text_prompts:{0:'painted cat'},
    video:{width:576,height:832,max_size:832},flow:{flow_warp:true},
    disco:{...structuredClone(defaults.disco),use_secondary_model:enabled,distilled_guided:enabled,sampler:'ddim',eta:.25,lora_strength:.7,
      quality_lora_enabled:false,quality_lora_strength:1,quality_lora_path:'quality.safetensors',
      clip_guidance_scale:70000,mask_frame_guidance:true,steps_schedule:[6]}};
  fetchMock.mockImplementation(async(url,options)=>url==='/api/config'
    ?response({schema:qualitySchema,defaults:config,prefilled:true}):original(url,options));
  render(App);
  await screen.findByText('weights.pt');
  await fireEvent.click(within(screen.getByRole('navigation',{name:'Workflow'})).getByRole('button',{name:'Render',exact:true}));
  expect(screen.queryByRole('spinbutton',{name:/Quality LoRA strength/})).not.toBeInTheDocument();
  await fireEvent.click(screen.getByRole('checkbox',{name:/Enable quality LoRA/}));
  expect(screen.getByRole('checkbox',{name:'Use secondary model'})).toHaveProperty('checked',enabled);
  await fireEvent.change(screen.getByRole('spinbutton',{name:/Quality LoRA strength/}),{target:{value:'0.75'}});
  await fireEvent.click(screen.getByRole('button',{name:'Start render'}));
  await waitFor(()=>expect(requests('/api/jobs')).toHaveLength(1));
  const payload=JSON.parse(requests('/api/jobs')[0][1].body);
  expect(payload.model_version).toBe('disco_512');
  expect(payload.disco).toMatchObject({quality_lora_enabled:true,quality_lora_strength:.75,
    quality_lora_path:'quality.safetensors',lora_strength:.7,sampler:'ddim',eta:.25,
    distilled_guided:enabled,use_secondary_model:enabled,mask_frame_guidance:true,clip_guidance_scale:70000,steps_schedule:[6]});
  expect(payload.text_prompts).toEqual({0:'painted cat'});
  expect(payload.flow.flow_warp).toBe(true);
  expect(payload.video).toEqual(config.video);
});
