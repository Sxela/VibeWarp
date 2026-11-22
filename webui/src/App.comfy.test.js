import {render,screen,fireEvent,waitFor,within} from '@testing-library/svelte';
import {beforeEach,afterEach,it,expect,vi} from 'vitest';
import App from './App.svelte';

const field=(type,group,extra={})=>({type,group,tier:'render',model_families:['comfy_image'],...extra});
const editOnly={visible_when:[{path:'comfy_image.mode',equals:'edit'}]};
const schema={workflows:[{id:'model',label:'Model'},{id:'prompts',label:'Prompts & sampling'},{id:'conditioning',label:'Conditioning'}],
  model_family_by_version:{moss_image:'comfy_image'},model_options:[{value:'moss_image',label:'MOSS-Image'}],
  properties:{model_version:{type:'string',tier:'project',group:'Model',choices:['moss_image']},comfy_image:{type:'dataclass',properties:{
    mode:field('string','Comfy Image',{label:'Generation mode',workflows:['prompts'],choices:['edit','text']}),
    references:field('array','Reference Images',{...editOnly,editor:'references',workflows:['conditioning']}),
    multi_reference_instruction:field('string','Reference Images',{...editOnly,editor:'references',group_editor_extra:true,
      label:'Reference-role instructions (2+ images)',workflows:['conditioning']}),
    comfy_edit_lora_name:field('string','Paths',{...editOnly,comfy_model:true,tier:'system',label:'MOSS edit LoRA filename',workflows:['model']}),
  }}}};
const defaults={model_version:'moss_image',batch_name:'test',video:{},frame_range:[0,10],comfy_image:{mode:'edit',comfy_edit_lora_name:'edit.safetensors',comfy_server_url:'http://comfy.example:8188'}};
const response=data=>({ok:true,json:async()=>data});
let fetchMock;
beforeEach(()=>{
  localStorage.clear();vi.stubGlobal('EventSource',class{close(){}});
  fetchMock=vi.fn(async(url,options)=>{
    if(url==='/api/config')return response({schema,defaults:structuredClone(defaults),prefilled:true});
    if(url==='/api/jobs')return response([]);
    if(url==='/api/models/required')return response({files:[],comfy_models:[{section:'comfy_image',field:'comfy_edit_lora_name',suggested:'edit.safetensors',available:['available.safetensors']}]});
    if(url==='/api/models/downloads')return response({files:[],tasks:[]});
    if(url.startsWith('/api/comfy/status'))return response({connected:true});
    if(url.startsWith('/api/preview/runs?'))return response({runs:[]});
    if(url.startsWith('/api/fs/exists'))return response({exists:true});
    throw Error(url);
  });vi.stubGlobal('fetch',fetchMock);
});
afterEach(()=>vi.unstubAllGlobals());

it('restores separate model dropdown selections across switches and reloads',async()=>{
  const modelSchema=structuredClone(schema);
  modelSchema.model_options=[{value:'moss_image',label:'MOSS'},{value:'minimax_h3_image',label:'MiniMax'}];
  modelSchema.model_family_by_version.minimax_h3_image='comfy_image';
  modelSchema.comfy_image_model_defaults={moss_image:{comfy_edit_lora_name:'edit.safetensors'},minimax_h3_image:{comfy_edit_lora_name:'mini.safetensors'}};
  const original=fetchMock.getMockImplementation();
  fetchMock.mockImplementation((url,options)=>url==='/api/config'
    ?response({schema:modelSchema,defaults:structuredClone(defaults),prefilled:true}):original(url,options));
  let app=render(App);
  let models=await screen.findByRole('combobox',{name:'Model Version'});
  let files=await screen.findByRole('combobox',{name:'MOSS edit LoRA filename'});
  await waitFor(()=>expect([...files.options].some(o=>o.value==='available.safetensors')).toBe(true));
  await fireEvent.change(files,{target:{value:'available.safetensors'}});
  await fireEvent.change(models,{target:{value:'minimax_h3_image'}});
  expect(screen.getByRole('combobox',{name:'MOSS edit LoRA filename'})).toHaveValue('mini.safetensors');
  files=screen.getByRole('combobox',{name:'MOSS edit LoRA filename'});
  await waitFor(()=>expect([...files.options].some(o=>o.value==='available.safetensors')).toBe(true));
  await fireEvent.change(files,{target:{value:'available.safetensors'}});
  await fireEvent.change(models,{target:{value:'moss_image'}});
  expect(screen.getByRole('combobox',{name:'MOSS edit LoRA filename'})).toHaveValue('available.safetensors');
  app.unmount();
  app=render(App);
  models=await screen.findByRole('combobox',{name:'Model Version'});
  await fireEvent.change(models,{target:{value:'minimax_h3_image'}});
  expect(screen.getByRole('combobox',{name:'MOSS edit LoRA filename'})).toHaveValue('available.safetensors');
});

it('exposes reference-role instructions above the custom reference editor and persists edits',async()=>{
  render(App);
  await screen.findByRole('combobox',{name:'Model Version'});
  await fireEvent.click(within(screen.getByRole('navigation',{name:'Workflow'})).getByRole('button',{name:/Conditioning/}));
  const instruction=screen.getByRole('textbox',{name:'Reference-role instructions (2+ images)'});
  expect(instruction.tagName).toBe('TEXTAREA');
  expect(screen.getByText('Image 1')).toBeInTheDocument();
  await fireEvent.input(screen.getByLabelText('Reference tag size'),{target:{value:'2'}});
  await fireEvent.input(screen.getByLabelText('Reference badge height'),{target:{value:'0.18'}});
  expect(JSON.parse(localStorage.getItem('vibewarp.config.v1')).reference_label_font_scale).toBe(2);
  expect(JSON.parse(localStorage.getItem('vibewarp.config.v1')).reference_label_badge_height).toBe(.18);
  await fireEvent.change(instruction,{target:{value:'Use image 1 for composition and image 2 for style.'}});
  expect(JSON.parse(localStorage.getItem('vibewarp.config.v1')).comfy_image.multi_reference_instruction)
    .toBe('Use image 1 for composition and image 2 for style.');
  await fireEvent.click(within(screen.getByRole('navigation',{name:'Workflow'})).getByRole('button',{name:/Prompts & sampling/}));
  await fireEvent.click(within(screen.getByRole('navigation',{name:'Workflow'})).getByRole('button',{name:/Conditioning/}));
  expect(screen.getByRole('textbox',{name:'Reference-role instructions (2+ images)'})).toHaveValue('Use image 1 for composition and image 2 for style.');
});

it('downloads the selected missing suggestion through the normal model queue',async()=>{
  schema.comfy_model_suggestions={moss_image:{comfy_edit_lora_name:'edit.safetensors'}};
  const original=fetchMock.getMockImplementation();
  fetchMock.mockImplementation((url,options)=>{
    if(url==='/api/models/required')return response({files:[{name:'edit.safetensors',path:'loras/edit.safetensors',status:'missing',downloadable:true}],
      comfy_models:[{section:'comfy_image',field:'comfy_edit_lora_name',suggested:'edit.safetensors',available:['other.safetensors']}]});
    if(url==='/api/models/prepare')return response({id:'download',state:'completed',files:[]});
    if(url==='/api/models/prepare/download')return response({id:'download',state:'completed',files:[]});
    return original(url,options);
  });
  render(App);
  const select=await screen.findByRole('combobox',{name:'MOSS edit LoRA filename'});
  await waitFor(()=>expect(select.options[0].textContent).toContain('Suggested'));
  expect(select.options[0].value).toBe('edit.safetensors');
  await fireEvent.click(await screen.findByRole('button',{name:'Download required models'}));
  await waitFor(()=>expect(fetchMock.mock.calls.some(([url,options])=>url==='/api/models/prepare'&&JSON.parse(options.body).comfy_image.comfy_edit_lora_name==='edit.safetensors')).toBe(true));
});

it('switches text/edit mode, refreshes required models, and keeps hidden edit settings',async()=>{
  render(App);
  expect(await screen.findByRole('combobox',{name:'Model Version'})).toHaveValue('moss_image');
  await fireEvent.click(screen.getByRole('button',{name:/Prompts & sampling/}));
  await fireEvent.change(screen.getByRole('combobox',{name:'Generation mode'}),{target:{value:'text'}});
  await waitFor(()=>expect(fetchMock.mock.calls.some(([url,options])=>url==='/api/models/required'&&JSON.parse(options.body).comfy_image.mode==='text')).toBe(true));
  await fireEvent.click(within(screen.getByRole('navigation',{name:'Workflow'})).getByRole('button',{name:/Model/}));
  expect(screen.queryByRole('combobox',{name:'MOSS edit LoRA filename'})).not.toBeInTheDocument();
  const saved=JSON.parse(localStorage.getItem('vibewarp.config.v1'));
  expect(saved.comfy_image.comfy_edit_lora_name).toBe('edit.safetensors');
  await fireEvent.click(screen.getByRole('button',{name:/Prompts & sampling/}));
  await fireEvent.change(screen.getByRole('combobox',{name:'Generation mode'}),{target:{value:'edit'}});
  await fireEvent.click(within(screen.getByRole('navigation',{name:'Workflow'})).getByRole('button',{name:/Model/}));
  const select=screen.getByRole('combobox',{name:'MOSS edit LoRA filename'});
  expect(select).toHaveValue('edit.safetensors');
  await waitFor(()=>expect([...select.options].some(option=>option.value==='available.safetensors')).toBe(true));
  await fireEvent.change(select,{target:{value:'available.safetensors'}});
  expect(JSON.parse(localStorage.getItem('vibewarp.config.v1')).comfy_image.comfy_edit_lora_name).toBe('available.safetensors');
});
