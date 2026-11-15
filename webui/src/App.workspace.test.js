import {render,screen,fireEvent,waitFor,within} from '@testing-library/svelte';
import {beforeEach,afterEach,expect,it,vi} from 'vitest';
import App from './App.svelte';

const prop=(type,tier,group,workflows,extra={})=>({type,tier,group,workflows,...extra});
const schema={
  workflows:[{id:'source',label:'Source & output',groups:['Output']},{id:'model',label:'Model',groups:[]},{id:'prompts',label:'Prompts & sampling',groups:['Prompts']},{id:'system',label:'System settings',groups:['Paths']}],
  model_options:[{value:'control_multi_v15',label:'Stable Diffusion 1.5'},{value:'qwen_image_edit_2511',label:'Qwen Image Edit 2511'}],
  model_family_by_version:{control_multi_v15:'sd',qwen_image_edit_2511:'qwen'},
  properties:{
    model_version:prop('string','project','Model',['model'],{choices:['control_multi_v15','qwen_image_edit_2511']}),
    batch_name:prop('string','project','Output',['source'],{label:'Project name'}),
    text_prompts:prop('any','render','Prompts',['prompts']),
    negative_prompts:prop('any','render','Prompts',['prompts']),
    root_dir:prop('string','system','Paths',['system'],{label:'Model storage folder'}),
    warp:{type:'dataclass',properties:{warp_mode:prop('string','hidden','Warp',[],{label:'Hidden legacy mode'})}},
  },
};
const defaults={model_version:'control_multi_v15',batch_name:'test',output_dir:'images_out',root_dir:'models',video:{},
  text_prompts:{0:'gold',10:'silver'},negative_prompts:{0:'blur'},warp:{warp_mode:'legacy'},
  qwen:{references:[{source:'warped'},{source:'upload',image_path:'style.png'}]},contact_sheet:{mode:'off'}};
const response=data=>({ok:true,json:async()=>data});
let fetchMock;
beforeEach(()=>{
  localStorage.clear();
  vi.stubGlobal('EventSource',class{close(){}});
  fetchMock=vi.fn(async(url)=>{
    if(url==='/api/config')return response({schema,defaults:structuredClone(defaults),prefilled:true});
    if(url==='/api/jobs')return response([]);
    if(url==='/api/models/downloads')return response({files:[],tasks:[]});
    if(url==='/api/models/required')return response({files:[]});
    if(url.startsWith('/api/preview/runs?'))return response({runs:[],root:'images_out/test'});
    if(url.startsWith('/api/fs/exists'))return response({exists:true});
    throw new Error(`Unexpected request ${url}`);
  });
  vi.stubGlobal('fetch',fetchMock);
});
afterEach(()=>vi.unstubAllGlobals());
const nav=()=>within(screen.getByRole('navigation',{name:'Workflow'}));

it('opens the Support VibeWarp category and its contents by default on Model',async()=>{
  render(App);
  await screen.findByRole('textbox',{name:'Project name'});
  await fireEvent.click(nav().getByRole('button',{name:'Model',exact:true}));
  const category=screen.getByText('Support VibeWarp',{selector:'summary'}).closest('details');
  expect(category.open).toBe(true);
  expect(screen.getByText('Supported by')).toBeVisible();
  expect(screen.getByRole('link',{name:'Support VibeWarp on Patreon'})).toBeVisible();
  await fireEvent.change(screen.getByRole('combobox',{name:'Model Version'}),{target:{value:'qwen_image_edit_2511'}});
  expect(category.open).toBe(true);
  await fireEvent.click(nav().getByRole('button',{name:'Source & output',exact:true}));
  await fireEvent.click(nav().getByRole('button',{name:'Model',exact:true}));
  expect(screen.getByText('Support VibeWarp',{selector:'summary'}).closest('details').open).toBe(true);
  expect(screen.getByRole('link',{name:'Support VibeWarp on Patreon'})).toBeVisible();
});

it('drops retired first-frame hashes from browser settings and never writes new ones',async()=>{
  const layout=structuredClone(schema);
  layout.properties.video={type:'dataclass',properties:{
    first_frame_path:prop('string','advanced','Video Input',['source'],{label:'First frame image'})}};
  localStorage.setItem('vibewarp.config.v1',JSON.stringify({...structuredClone(defaults),
    video:{first_frame_path:'old.png',first_frame_sha256:'old'}}));
  const original=fetchMock.getMockImplementation();
  fetchMock.mockImplementation(url=>url==='/api/config'?response({schema:layout,defaults:structuredClone(defaults),prefilled:false}):original(url));
  render(App);
  const input=await screen.findByRole('textbox',{name:'First frame image'});
  expect(JSON.parse(localStorage.getItem('vibewarp.config.v1')).video).not.toHaveProperty('first_frame_sha256');
  expect(screen.queryByRole('textbox',{name:'First frame SHA256'})).not.toBeInTheDocument();
  await fireEvent.change(input,{target:{value:'new.png'}});
  expect(JSON.parse(localStorage.getItem('vibewarp.config.v1')).video).toEqual({first_frame_path:'new.png'});
});

it('edits the navbar project title and updates the batch setting on Enter',async()=>{
  render(App);
  await screen.findByRole('textbox',{name:'Project name'});
  const title=screen.getByRole('button',{name:'Edit project name'});
  expect(title).toHaveTextContent('test');
  await fireEvent.click(title);
  const input=screen.getByRole('textbox',{name:'Edit project name'});
  expect(input).toHaveFocus();
  expect(input.selectionStart).toBe(0);expect(input.selectionEnd).toBe(4);
  await fireEvent.input(input,{target:{value:'golden-dance'}});
  await fireEvent.keyDown(input,{key:'Enter'});
  expect(screen.getByRole('button',{name:'Edit project name'})).toHaveTextContent('golden-dance');
  expect(screen.getByRole('button',{name:'Edit project name'})).toHaveFocus();
  expect(screen.getByRole('textbox',{name:'Project name'})).toHaveValue('golden-dance');
  expect(JSON.parse(localStorage.getItem('vibewarp.config.v1')).batch_name).toBe('golden-dance');
});

it('cancels navbar name editing on Escape and preserves the name for blank edits',async()=>{
  render(App);
  await screen.findByRole('textbox',{name:'Project name'});
  await fireEvent.click(screen.getByRole('button',{name:'Edit project name'}));
  const input=screen.getByRole('textbox',{name:'Edit project name'});
  await fireEvent.input(input,{target:{value:'discard'}});
  await fireEvent.keyDown(input,{key:'Escape'});
  expect(screen.getByRole('button',{name:'Edit project name'})).toHaveTextContent('test');
  await fireEvent.click(screen.getByRole('button',{name:'Edit project name'}));
  const blank=screen.getByRole('textbox',{name:'Edit project name'});
  await fireEvent.input(blank,{target:{value:'  '}});
  await fireEvent.keyDown(blank,{key:'Enter'});
  expect(blank).toBeInTheDocument();
  await fireEvent.blur(blank);
  expect(screen.getByRole('button',{name:'Edit project name'})).toHaveTextContent('test');
  expect(JSON.parse(localStorage.getItem('vibewarp.config.v1')).batch_name).toBe('test');
});

it('saves a navbar name on blur and keeps it in sync with the settings form',async()=>{
  render(App);
  await screen.findByRole('textbox',{name:'Project name'});
  await fireEvent.click(screen.getByRole('button',{name:'Edit project name'}));
  const input=screen.getByRole('textbox',{name:'Edit project name'});
  await fireEvent.input(input,{target:{value:'  next-project  '}});
  await fireEvent.blur(input);
  expect(screen.getByRole('button',{name:'Edit project name'})).toHaveTextContent('next-project');
  await fireEvent.change(screen.getByRole('textbox',{name:'Project name'}),{target:{value:'from-settings'}});
  expect(screen.getByRole('button',{name:'Edit project name'})).toHaveTextContent('from-settings');
});

it('shows before and after matching in a dedicated Color match tab',async()=>{
  const layout=structuredClone(schema);
  layout.workflows.push({id:'color',label:'Color match',groups:['Before diffusion','After diffusion']});
  layout.properties.color={type:'dataclass',properties:{
    before_enabled:prop('boolean','advanced','Before diffusion',['color'],{label:'Before enabled',workspace_advanced:true}),
    after_enabled:prop('boolean','advanced','After diffusion',['color'],{label:'After enabled',workspace_advanced:true})}};
  const original=fetchMock.getMockImplementation();
  fetchMock.mockImplementation(url=>url==='/api/config'?response({schema:layout,defaults:{...structuredClone(defaults),
    color:{before_enabled:false,after_enabled:true}},prefilled:true}):original(url));
  render(App);
  await screen.findByRole('textbox',{name:'Project name'});
  await fireEvent.click(nav().getByRole('button',{name:'Color match'}));
  expect(screen.getByRole('heading',{name:'Color match'})).toBeInTheDocument();
  expect(screen.getByRole('checkbox',{name:'After enabled'})).toBeVisible();
  await fireEvent.click(screen.getByRole('checkbox',{name:'Before enabled'}));
  expect(JSON.parse(localStorage.getItem('vibewarp.config.v1')).color).toEqual({before_enabled:true,after_enabled:true});
});

it('edits fixed-noise variation for unguided Disco and hides it for Comfy renderers',async()=>{
  const layout=structuredClone(schema);
  layout.model_options.push({value:'disco_512',label:'Disco 512'});
  layout.model_family_by_version.disco_512='disco';
  layout.properties.diffusion={type:'dataclass',properties:{
    seed:prop('integer','render','Seed',['prompts']),
    fixed_seed_randomness:prop('number','render','Seed',['prompts'],{label:'Fixed noise randomness',
      model_families:['sd','disco'],slider:{min:0,max:1,step:0.01,scale:1}}),
    fixed_seed_wiggle:prop('integer','render','Seed',['prompts'],{label:'Noise wiggle shift px',model_families:['sd','disco']}),
    fixed_seed_wiggle_direction:prop('array','render','Seed',['prompts'],{label:'Noise wiggle direction',
      vector_labels:['X','Y'],model_families:['sd','disco']})}};
  const original=fetchMock.getMockImplementation();
  fetchMock.mockImplementation(url=>url==='/api/config'?response({schema:layout,defaults:{...structuredClone(defaults),
    model_version:'disco_512',disco:{sampler:'distilled',distilled_guided:false,fixed_seed:true},
    diffusion:{seed:170,fixed_seed_randomness:0,fixed_seed_wiggle:0,fixed_seed_wiggle_direction:[0,0]}},prefilled:true}):original(url));
  render(App);
  await screen.findByRole('textbox',{name:'Project name'});
  await fireEvent.click(nav().getByRole('button',{name:'Prompts & sampling'}));
  await fireEvent.input(screen.getByRole('slider',{name:'Fixed noise randomness'}),{target:{value:'0.25'}});
  await fireEvent.change(screen.getByRole('spinbutton',{name:/^Noise wiggle shift/}),{target:{value:'3'}});
  await fireEvent.change(screen.getByRole('spinbutton',{name:'Noise wiggle direction X'}),{target:{value:'-1'}});
  await fireEvent.change(screen.getByRole('spinbutton',{name:'Noise wiggle direction Y'}),{target:{value:'0.5'}});
  const saved=JSON.parse(localStorage.getItem('vibewarp.config.v1'));
  expect(saved.diffusion).toEqual({seed:170,fixed_seed_randomness:0.25,fixed_seed_wiggle:3,
    fixed_seed_wiggle_direction:[-1,0.5]});
  expect(saved.disco.fixed_seed).toBe(true);
  await fireEvent.change(screen.getByRole('combobox',{name:'Model Version'}),{target:{value:'qwen_image_edit_2511'}});
  expect(screen.queryByRole('slider',{name:'Fixed noise randomness'})).not.toBeInTheDocument();
  expect(screen.queryByRole('spinbutton',{name:/^Noise wiggle shift/})).not.toBeInTheDocument();
  expect(screen.queryByRole('spinbutton',{name:'Noise wiggle direction X'})).not.toBeInTheDocument();
});

it('exposes a dedicated background tab and preview across model switches',async()=>{
  const layout=structuredClone(schema);
  layout.workflows.push({id:'background',label:'Mask & tint',groups:['Background masking']});
  layout.properties.mask={type:'dataclass',properties:{rvm_enabled:prop('boolean','render','Background masking',['background'],
    {label:'Apply masked tint',preview:'raw_video'})}};
  const original=fetchMock.getMockImplementation();
  fetchMock.mockImplementation(url=>url==='/api/config'?response({schema:layout,
    defaults:{...structuredClone(defaults),mask:{rvm_enabled:false}},prefilled:true}):original(url));
  render(App);
  await screen.findByRole('textbox',{name:'Project name'});
  await fireEvent.click(nav().getByRole('button',{name:'Mask & tint'}));
  expect(screen.getByRole('heading',{name:'Mask & tint'})).toBeInTheDocument();
  expect(screen.getByRole('button',{name:'Preview one frame'})).toBeDisabled();
  await fireEvent.click(screen.getByRole('checkbox',{name:'Apply masked tint'}));
  await fireEvent.change(screen.getByRole('combobox',{name:'Model Version'}),{target:{value:'qwen_image_edit_2511'}});
  expect(screen.getByRole('heading',{name:'Mask & tint'})).toBeInTheDocument();
  expect(screen.getByRole('checkbox',{name:'Apply masked tint'})).toBeChecked();
  expect(screen.getByRole('button',{name:'Preview one frame'})).toBeEnabled();
});

it('closes the project action popover when choosing import',async()=>{
  const {container}=render(App);
  await screen.findByRole('textbox',{name:'Project name'});
  const menu=container.querySelector('.project-menu');
  const trigger=menu.querySelector('summary');
  await fireEvent.click(trigger);
  expect(menu.open).toBe(true);
  const picker=screen.getByLabelText('Import settings file');
  const openPicker=vi.spyOn(picker,'click').mockImplementation(()=>{});
  await fireEvent.click(screen.getByRole('button',{name:'Import settings',exact:true}));
  expect(menu.open).toBe(false);
  expect(openPicker).toHaveBeenCalledOnce();
  expect(trigger).toHaveFocus();
});

it('keeps RVM file overrides editable in Mask & tint and System with the same stored value',async()=>{
  const layout=structuredClone(schema);
  layout.workflows.push({id:'background',label:'Mask & tint',groups:['RVM files']});
  layout.properties.mask={type:'dataclass',properties:{rvm_model_path:prop('string','system','Paths',['background','system'],{
    label:'RVM model file',workflow_groups:{background:'RVM files'},workflow_editors:{background:null}})}};
  const original=fetchMock.getMockImplementation();
  fetchMock.mockImplementation(url=>url==='/api/config'?response({schema:layout,defaults:{...structuredClone(defaults),
    mask:{rvm_model_path:'old-rvm.torchscript'}},prefilled:true}):original(url));
  render(App);await screen.findByRole('textbox',{name:'Project name'});
  await fireEvent.click(nav().getByRole('button',{name:'Mask & tint'}));
  expect(screen.getByText('RVM files',{selector:'summary span'})).toBeInTheDocument();
  await fireEvent.change(screen.getByRole('textbox',{name:'RVM model file'}),{target:{value:'new-rvm.torchscript'}});
  expect(JSON.parse(localStorage.getItem('vibewarp.config.v1')).mask.rvm_model_path).toBe('new-rvm.torchscript');
  await fireEvent.click(nav().getByRole('button',{name:'System settings'}));
  await fireEvent.click(screen.getByText('Paths',{selector:'summary span'}));
  expect(screen.getByRole('textbox',{name:'RVM model file'})).toHaveValue('new-rvm.torchscript');
});

it.each([false,true])('opens an advanced-only subgroup while respecting visible siblings (mixed=%s)',async(mixed)=>{
  const layout=structuredClone(schema);
  layout.workflows.push({id:'consistency',label:'Frame consistency',groups:['Warp']});
  layout.properties.warp.properties.padding_ratio=prop('number','advanced','Warp',['consistency'],{workspace_advanced:true});
  layout.properties.warp.properties.flow_blend=prop('number','render','Warp',['consistency'],{model_families:mixed?['sd']:['qwen']});
  const original=fetchMock.getMockImplementation();
  fetchMock.mockImplementation(url=>url==='/api/config'?response({schema:layout,defaults:{...structuredClone(defaults),
    warp:{padding_ratio:0.2,flow_blend:0.5}},prefilled:true}):original(url));
  const {container}=render(App);
  await screen.findByRole('textbox',{name:'Project name'});
  await fireEvent.click(nav().getByRole('button',{name:'Frame consistency'}));
  const advanced=container.querySelector('.expert-fields');
  expect(advanced.open).toBe(!mixed);
  if(!mixed)expect(screen.getByRole('spinbutton',{name:'Padding Ratio'})).toBeVisible();
  // The subgroup remains a normal disclosure that users can collapse themselves.
  await fireEvent.click(within(advanced).getByText('Advanced controls'));
  expect(advanced.open).toBe(mixed);
});

it('hides empty workflows and groups as models change and exposes Disco guidance and seed controls',async()=>{
  const layout=structuredClone(schema);
  layout.workflows.splice(3,0,
    {id:'clip',label:'Guidance',groups:['General','CLIP guidance','Cutouts','Advanced guidance']},
    {id:'conditioning',label:'Conditioning',groups:['SD inputs']});
  layout.model_options.push({value:'disco_512',label:'Disco 512'});
  layout.model_family_by_version.disco_512='disco';
  layout.properties.diffusion={type:'dataclass',properties:{
    seed:prop('integer','render','Seed',['prompts']),
    sd_input:prop('string','render','SD inputs',['conditioning'],{model_families:['sd']})}};
  const disco=(type,group,workflow,extra={})=>prop(type,'render',group,[workflow],{model_families:['disco'],...extra});
  layout.properties.disco={type:'dataclass',properties:{
    fixed_seed:disco('boolean','Seed','prompts'),
    use_secondary_model:disco('boolean','General','clip'),
    clip_guidance_scale:disco('number','CLIP guidance','clip'),
    cut_overview:disco('object','Cutouts','clip',{schedule_domain:'inference_progress',schedule_value:'cut_count'}),
    clip_grad_steps:disco('integer','Advanced guidance','clip')}};
  const original=fetchMock.getMockImplementation();
  fetchMock.mockImplementation(url=>url==='/api/config'?response({schema:layout,defaults:{...structuredClone(defaults),
    diffusion:{seed:123,sd_input:'style'},disco:{fixed_seed:false,use_secondary_model:true,clip_guidance_scale:5000,cut_overview:{0:4,60:0},clip_grad_steps:1}},prefilled:true}):original(url));
  const {container}=render(App);
  await screen.findByRole('textbox',{name:'Project name'});
  expect(nav().queryByRole('button',{name:'Guidance'})).not.toBeInTheDocument();
  await fireEvent.click(nav().getByRole('button',{name:'Conditioning'}));
  expect(screen.getByText('SD inputs',{selector:'summary span'})).toBeInTheDocument();
  await fireEvent.change(screen.getByRole('combobox',{name:'Model Version'}),{target:{value:'disco_512'}});
  expect(nav().queryByRole('button',{name:'Conditioning'})).not.toBeInTheDocument();
  expect(screen.getByRole('heading',{name:'Source & output'})).toBeInTheDocument();
  await fireEvent.click(nav().getByRole('button',{name:'Guidance'}));
  expect([...container.querySelectorAll('.panel.group>summary span')].map(el=>el.textContent)).toEqual(['General','CLIP guidance','Cutouts','Advanced guidance']);
  expect(screen.getByRole('checkbox',{name:'Use Secondary Model'})).toBeChecked();
  expect(container.querySelector('#field-disco-cut_overview').closest('.expert-fields')).toBeNull();
  await fireEvent.click(nav().getByRole('button',{name:'Prompts & sampling'}));
  const seed=screen.getByText('Seed',{selector:'summary span'}).closest('details');
  expect(within(seed).getByRole('spinbutton',{name:'Seed'})).toHaveValue(123);
  await fireEvent.click(within(seed).getByRole('checkbox',{name:'Fixed Seed'}));
  expect(JSON.parse(localStorage.getItem('vibewarp.config.v1')).disco.fixed_seed).toBe(true);
  await fireEvent.click(nav().getByRole('button',{name:'Model',exact:true}));
  expect(container.querySelector('.groups .panel.group')).toBeNull();
});

it('renders model sections from schema metadata and selects across them',async()=>{
  const grouped=structuredClone(schema);
  grouped.model_groups=[{id:'native',label:'Native models'},{id:'comfy',label:'ComfyUI-managed models'}];
  grouped.model_options[0].group='native';grouped.model_options[1].group='comfy';
  const original=fetchMock.getMockImplementation();
  fetchMock.mockImplementation(url=>url==='/api/config'?response({schema:grouped,defaults:structuredClone(defaults),prefilled:true}):original(url));
  render(App);
  const picker=await screen.findByRole('combobox',{name:'Model Version'});
  expect([...picker.querySelectorAll('optgroup')].map(group=>group.label)).toEqual(['Native models','ComfyUI-managed models']);
  expect(picker.querySelector('optgroup[label="Native models"] option')).toHaveValue('control_multi_v15');
  await fireEvent.change(picker,{target:{value:'qwen_image_edit_2511'}});
  expect(JSON.parse(localStorage.getItem('vibewarp.config.v1')).model_version).toBe('qwen_image_edit_2511');
});

it('uses schema workflows, searchable display labels, and excludes hidden settings',async()=>{
  render(App);
  await screen.findByRole('textbox',{name:'Project name'});
  const search=screen.getByRole('searchbox',{name:'Search settings'});
  await fireEvent.input(search,{target:{value:'Model storage folder'}});
  expect(screen.getByRole('textbox',{name:'Model storage folder'})).toHaveValue('models');
  await fireEvent.input(search,{target:{value:'Hidden legacy mode'}});
  expect(screen.getByText(/No editable settings match/)).toBeInTheDocument();
  expect(screen.queryByRole('textbox',{name:'Hidden legacy mode'})).not.toBeInTheDocument();
  await fireEvent.click(nav().getByRole('button',{name:'System settings'}));
  expect(screen.getByRole('heading',{name:'System settings'})).toBeInTheDocument();
  expect(screen.queryByRole('button',{name:'Import / export'})).not.toBeInTheDocument();
});

it('edits positive and negative prompts through the existing keyframe editor',async()=>{
  const {container}=render(App);
  await screen.findByRole('textbox',{name:'Project name'});
  await fireEvent.click(nav().getByRole('button',{name:'Prompts & sampling'}));
  const positive=within(container.querySelector('#field-main-text_prompts'));
  const negative=within(container.querySelector('#field-main-negative_prompts'));
  await fireEvent.change(positive.getAllByRole('textbox',{name:'Prompt'})[1],{target:{value:'bronze'}});
  await fireEvent.change(negative.getByRole('textbox',{name:'Prompt'}),{target:{value:'noise'}});
  await waitFor(()=>{
    const saved=JSON.parse(localStorage.getItem('vibewarp.config.v1'));
    expect(saved.text_prompts).toEqual({0:'gold',10:'bronze'});
    expect(saved.negative_prompts).toEqual({0:'noise'});
  });
});

it('retains optional references and requires an explicit choice to resolve contact sheets',async()=>{
  const original=fetchMock.getMockImplementation();
  fetchMock.mockImplementation(url=>url==='/api/config'?response({schema,defaults:{...structuredClone(defaults),model_version:'qwen_image_edit_2511',contact_sheet:{mode:'grid'}},prefilled:true}):original(url));
  render(App);
  expect(await screen.findByText(/Your references have been kept/)).toBeInTheDocument();
  expect(screen.getByRole('button',{name:'Start render'})).toBeDisabled();
  await fireEvent.click(screen.getByRole('button',{name:'Turn off contact sheets'}));
  expect(screen.getByRole('button',{name:'Start render'})).toBeEnabled();
  const saved=JSON.parse(localStorage.getItem('vibewarp.config.v1'));
  expect(saved.contact_sheet.mode).toBe('off');
  expect(saved.qwen.references).toEqual(defaults.qwen.references);
});

it('validation links leave Inspect and reveal and focus the offending setting',async()=>{
  const original=fetchMock.getMockImplementation();
  fetchMock.mockImplementation((url,options)=>url==='/api/jobs'&&options?.method==='POST'
    ?{ok:false,json:async()=>({detail:[{path:'batch_name',message:'Choose a project name'}]})}:original(url,options));
  const scroll=vi.fn();
  const {container}=render(App);
  await screen.findByRole('textbox',{name:'Project name'});
  container.querySelector('#field-main-batch_name').scrollIntoView=scroll;
  await fireEvent.click(screen.getByRole('tab',{name:'Inspect'}));
  expect(container.querySelector('.workspace')).toHaveClass('inspecting');
  await fireEvent.click(screen.getByRole('button',{name:'Start render'}));
  await fireEvent.click(await screen.findByRole('button',{name:/Choose a project name/}));
  expect(container.querySelector('.workspace')).not.toHaveClass('inspecting');
  expect(screen.getByRole('textbox',{name:/Project name/})).toHaveFocus();
  expect(scroll).toHaveBeenCalled();
});
