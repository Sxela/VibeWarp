import {render,screen,fireEvent,waitFor,within} from '@testing-library/svelte';
import {afterEach,beforeEach,expect,it,vi} from 'vitest';
import App from './App.svelte';

const schema={workflows:[{id:'source',label:'Source & output',groups:['Input Video','Output']}],
  model_options:[{value:'control_multi_v15',label:'Stable Diffusion 1.5'}],properties:{
    batch_name:{type:'string',tier:'project',group:'Output',workflows:['source'],label:'Project name'},
    video:{type:'dataclass',properties:{video_init_path:{type:'string',tier:'project',group:'Input Video',editor:'video',workflows:['source']}}},
  }};
const draft={model_version:'control_multi_v15',batch_name:'next-draft',output_dir:'next-out',frame_range:[0,0],video:{video_init_path:'',max_size:512,extract_nth_frame:1}};
const submitted={...draft,batch_name:'submitted-project',output_dir:'submitted-out',frame_range:[0,99],video:{...draft.video,video_init_path:'submitted.mp4'}};
const summary={id:'active',state:'running',frame:45,total_frames:100,progress:45,message:'Rendering frame 45',revision:1};
const response=data=>({ok:true,json:async()=>data});
let fetchMock,streams;
beforeEach(()=>{
  localStorage.clear();streams=[];
  vi.stubGlobal('EventSource',class{constructor(url){this.url=url;streams.push(this);}close(){}});
  fetchMock=vi.fn(async(url,options)=>{
    if(url==='/api/config')return response({schema,defaults:structuredClone(draft),prefilled:true});
    if(url==='/api/jobs')return response([summary]);
    if(url==='/api/jobs/active')return response({...summary,config:submitted});
    if(url==='/api/jobs/active/cancel')return response({...summary,cancel_requested:true});
    if(url==='/api/models/downloads')return response({files:[],tasks:[]});
    if(url==='/api/models/required')return response({files:[]});
    if(url.startsWith('/api/preview/runs?'))return response({runs:[],root:'submitted-out/submitted-project'});
    if(url.startsWith('/api/video/probe?'))return response({source:{width:1280,height:720,frames:100,fps:24}});
    throw Error(`Unexpected request ${url}`);
  });vi.stubGlobal('fetch',fetchMock);
});
afterEach(()=>vi.unstubAllGlobals());

it('reconnects using the full submitted snapshot while preserving the next-render draft',async()=>{
  const {container}=render(App);
  await screen.findByText(/Viewing submitted-project/);
  await screen.findByAltText('Source, frame 45');
  expect(screen.getByAltText('Source, frame 45').getAttribute('src')).toContain('path=submitted.mp4');
  expect(container.querySelector('.stepper')).toHaveTextContent('/ 99');
  expect(screen.getByRole('textbox',{name:'Project name'})).toHaveValue('next-draft');
  expect(screen.getByRole('textbox',{name:'Input video path'})).toHaveValue('');
  await fireEvent.change(screen.getByRole('textbox',{name:'Project name'}),{target:{value:'edited-next'}});
  expect(JSON.parse(localStorage.getItem('vibewarp.config.v1')).batch_name).toBe('edited-next');
  expect(fetchMock.mock.calls.filter(([url])=>url.startsWith('/api/preview/runs?')).every(([url])=>url.includes('batch_name=submitted-project'))).toBe(true);
  streams[0].onmessage({data:JSON.stringify({...summary,frame:46,revision:2})});
  await screen.findByAltText('Source, frame 46');
  expect(screen.getByText(/Viewing submitted-project/)).toBeInTheDocument();
  streams[0].onmessage({data:JSON.stringify({...summary,state:'completed',revision:3})});
  await screen.findByRole('button',{name:'View draft'});
  expect(screen.getByText(/Viewing submitted-project/)).toBeInTheDocument();
  await fireEvent.click(screen.getByRole('button',{name:'View draft'}));
  expect(screen.queryByText(/Viewing submitted-project/)).toBeNull();
  expect(await screen.findByRole('button',{name:'Choose input video'})).toBeInTheDocument();
});

it('shows render ETA in the status bar and keeps the resizable log reachable',async()=>{
  const original=fetchMock.getMockImplementation();
  fetchMock.mockImplementation((url,options)=>url==='/api/jobs/active'
    ?response({...summary,config:submitted,eta_seconds:5646,eta_updated_at:new Date().toISOString()}):original(url,options));
  render(App);
  expect(await screen.findByText('ETA 1h 34m')).toBeInTheDocument();
  await fireEvent.click(screen.getByRole('button',{name:'Show log'}));
  expect(screen.getByRole('separator',{name:'Resize system log'})).toBeInTheDocument();
  await fireEvent.click(screen.getByRole('button',{name:'Hide log'}));
  expect(screen.queryByRole('separator',{name:'Resize system log'})).toBeNull();
});

it('keeps cancellation available during recovery, ignores stale progress, and reports snapshot failure with retry',async()=>{
  let resolveSnapshot;
  const original=fetchMock.getMockImplementation();
  fetchMock.mockImplementation((url,options)=>url==='/api/jobs/active'
    ?new Promise(resolve=>{resolveSnapshot=resolve;}):original(url,options));
  render(App);
  await screen.findByText('Restoring submitted render…');
  expect(screen.queryByText('45 / 0')).toBeNull();
  await fireEvent.click(screen.getByRole('button',{name:'Cancel render',exact:true}));
  expect(await screen.findByRole('button',{name:'Cancelling…'})).toBeDisabled();
  resolveSnapshot({ok:false});
  await fireEvent.click(await screen.findByRole('button',{name:'Retry render snapshot'}));
  streams[0].onmessage({data:JSON.stringify({...summary,state:'cancelled',message:'Cancelled',revision:3})});
  resolveSnapshot(response({...summary,config:submitted}));
  await screen.findByText(/Viewing submitted-project/);
  expect(screen.getByRole('button',{name:'Start render'})).toBeInTheDocument();
  expect(screen.getByRole('textbox',{name:'Project name'})).toHaveValue('next-draft');
});

it('keeps Runs accessible and restores the inspector from Inspect without losing settings',async()=>{
  render(App);await screen.findByText(/Viewing submitted-project/);
  const navigation=within(screen.getByRole('navigation',{name:'Workspace navigation'}));
  await fireEvent.click(navigation.getByRole('button',{name:'Runs',exact:true}));
  expect(screen.getByRole('heading',{name:'Render history'})).toBeInTheDocument();
  await fireEvent.click(navigation.getByRole('button',{name:'Workspace'}));
  await fireEvent.click(screen.getByRole('tab',{name:'Inspect'}));
  expect(navigation.getByRole('button',{name:'Show settings'})).toHaveAttribute('aria-expanded','false');
  await fireEvent.click(navigation.getByRole('button',{name:'Show settings'}));
  expect(navigation.getByRole('button',{name:'Hide settings'})).toHaveAttribute('aria-expanded','true');
  expect(screen.getByRole('searchbox',{name:'Search settings'})).toHaveFocus();
  expect(screen.getByRole('textbox',{name:'Project name'})).toHaveValue('next-draft');
});

it('opens the source group and focuses its path from the initial source pane',async()=>{
  const original=fetchMock.getMockImplementation();fetchMock.mockImplementation((url,options)=>url==='/api/jobs'?response([]):original(url,options));
  const {container}=render(App);
  await fireEvent.click(await screen.findByRole('button',{name:'Choose input video'}));
  expect(screen.getByRole('textbox',{name:'Input video path'})).toHaveFocus();
  expect(container.querySelectorAll('.layer-stage figure')).toHaveLength(1);
  expect(screen.queryByText('Waiting for first output')).toBeNull();
});
