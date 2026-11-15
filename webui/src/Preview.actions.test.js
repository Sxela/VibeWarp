import {render,screen,fireEvent,waitFor} from '@testing-library/svelte';
import {afterEach,beforeEach,expect,it,vi} from 'vitest';
import Preview from './Preview.svelte';

const config={output_dir:'images_out',batch_name:'test',video:{}};
const response=data=>({ok:true,json:async()=>data});
let fetchMock;
beforeEach(()=>{
  localStorage.clear();
  fetchMock=vi.fn(async(url,options)=>{
    if(url.startsWith('/api/preview/runs?'))return response({runs:['1','2'].map(id=>({id,label:`Take ${id}`,frames:2,last_frame:1,modified:1,has_settings:true,video_available:true})),root:'images_out/test'});
    if(url.includes('/settings?')){
      const steps=url.includes('/1/')?20:30;
      return response({config:{diffusion:{steps_schedule:[steps]}},saved:{diffusion:{steps_schedule:[steps]}}});
    }
    if(url.includes('/video?')&&options?.method==='POST')return response({id:'assembly',state:'queued',operation:'video'});
    if(url.startsWith('/api/preview/runs/'))return response({run:url.includes('/1?')?'1':'2',frames:[0,1],layers:[{id:'init',label:'Init',frames:[0,1]},{id:'output',label:'Output',frames:[0,1]}]});
    if(url.startsWith('/api/video/probe?'))return response({source:{width:1280,height:720,frames:20,fps:24}});
    throw new Error(`Unexpected request ${url}`);
  });
  vi.stubGlobal('fetch',fetchMock);
});
afterEach(()=>vi.unstubAllGlobals());

it.each(['Current frame','Rendered video'])('closes the download popover after selecting %s',async(name)=>{
  render(Preview,{config,view:'output'});
  await screen.findByAltText('Output, frame 1');
  const trigger=screen.getByText('Download',{selector:'summary'});
  await fireEvent.click(trigger);
  const menu=trigger.closest('details');
  expect(menu.open).toBe(true);
  const link=screen.getByRole('link',{name});
  const href=link.getAttribute('href');
  // Prevent jsdom navigation; the real browser retains the native download.
  link.addEventListener('click',event=>event.preventDefault());
  await fireEvent.click(link);
  expect(menu.open).toBe(false);
  expect(link).toHaveAttribute('href',href);
  expect(trigger).toHaveFocus();
});

it('shows common run actions directly and loads settings without opening a menu',async()=>{
  const onload=vi.fn();
  render(Preview,{config,onload});
  const load=await screen.findByRole('button',{name:'Load settings',exact:true});
  await waitFor(()=>expect(load).toBeEnabled());
  expect(load.closest('details')).toBeNull();
  expect(screen.getByRole('button',{name:'Resume',exact:true})).toBeVisible();
  expect(screen.getByRole('button',{name:'Rename Take 1',exact:true})).toBeVisible();
  expect(screen.getByRole('radio',{name:'Play video',exact:true})).toBeVisible();
  await fireEvent.click(load);
  await waitFor(()=>expect(onload).toHaveBeenCalledWith({diffusion:{steps_schedule:[20]}}));
});

it('switches video and frames with radios beside the filmstrip',async()=>{
  const {container}=render(Preview,{config});
  await screen.findByAltText('Output, frame 1');
  const frames=screen.getByRole('radio',{name:'View frames'});
  const video=screen.getByRole('radio',{name:'Play video'});
  expect(frames).toBeChecked();
  expect(video.closest('.timeline')).toContainElement(screen.getByLabelText('Frame thumbnails'));
  await fireEvent.click(video);
  expect(video).toBeChecked();
  expect(container.querySelector('video')).toHaveAttribute('src',expect.stringContaining('/runs/1/video?'));
  expect(screen.queryByAltText('Output, frame 1')).not.toBeInTheDocument();
  await fireEvent.click(frames);
  expect(frames).toBeChecked();
  expect(container.querySelector('video')).toBeNull();
  expect(screen.getByAltText('Output, frame 1')).toBeInTheDocument();
});

it('renames a run by clicking its name while keeping resume and load directly available',async()=>{
  const original=fetchMock.getMockImplementation();
  fetchMock.mockImplementation((url,options)=>options?.method==='PUT'&&url.includes('/label?')
    ?response({label:JSON.parse(options.body).label}):original(url,options));
  render(Preview,{config});
  await fireEvent.click(await screen.findByRole('button',{name:'Rename Take 1'}));
  expect(screen.getByRole('textbox',{name:'Run display label'})).toHaveFocus();
  await fireEvent.input(screen.getByRole('textbox',{name:'Run display label'}),{target:{value:'Golden take'}});
  await fireEvent.click(screen.getByRole('button',{name:'Save',exact:true}));
  expect(await screen.findByRole('button',{name:'Rename Golden take'})).toHaveTextContent('Golden take');
  expect(screen.getByRole('button',{name:'Resume',exact:true}).closest('details')).toBeNull();
  expect(screen.queryByText('Run actions',{selector:'summary'})).not.toBeInTheDocument();
});

it('keeps the direct load action disabled for runs without saved settings',async()=>{
  const original=fetchMock.getMockImplementation();
  fetchMock.mockImplementation((url,options)=>url.startsWith('/api/preview/runs?')
    ?response({runs:[{id:'1',frames:2,last_frame:1,modified:1,has_settings:false}],root:'out/test'}):original(url,options));
  render(Preview,{config});
  await screen.findByAltText('Output, frame 1');
  expect(await screen.findByRole('button',{name:'Load settings',exact:true})).toBeDisabled();
});

it('compares explicitly selected A/B runs using their saved settings',async()=>{
  render(Preview,{config});
  await fireEvent.click(await screen.findByRole('button',{name:'Select Take 1 as A'}));
  await fireEvent.click(screen.getByRole('button',{name:'Select Take 2 as B'}));
  expect(await screen.findByRole('cell',{name:'[20]'})).toBeInTheDocument();
  expect(screen.getByRole('cell',{name:'[30]'})).toBeInTheDocument();
  expect(fetchMock.mock.calls.filter(([url])=>url.includes('/settings?')).every(([url])=>url.endsWith('overlay_system=false'))).toBe(true);
  expect(screen.getByRole('button',{name:'Close settings diff'})).toHaveFocus();
  await fireEvent.keyDown(window,{key:'Escape'});
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
});

it('offers actual image/video downloads and queues video assembly on click',async()=>{
  const onjob=vi.fn();
  render(Preview,{config,onjob,view:'output'});
  await screen.findByAltText('Output, frame 1');
  await fireEvent.click(screen.getByText('Download',{selector:'summary'}));
  expect(screen.getByRole('link',{name:'Current frame'})).toHaveAttribute('href',expect.stringContaining('layer=output&frame=1&download=true'));
  expect(screen.getByRole('link',{name:'Rendered video'})).toHaveAttribute('href',expect.stringContaining('/video?'));
  await fireEvent.click(screen.getByRole('button',{name:'Rebuild video'}));
  await waitFor(()=>expect(onjob).toHaveBeenCalledWith(expect.objectContaining({id:'assembly',operation:'video'})));
  expect(fetchMock.mock.calls.filter(([url,options])=>url.includes('/video?')&&options?.method==='POST')).toHaveLength(1);
});

it('previews and scrubs the selected source range before any render exists',async()=>{
  const original=fetchMock.getMockImplementation();
  fetchMock.mockImplementation((url,options)=>url.startsWith('/api/preview/runs?')?response({runs:[],root:'images_out/test'}):original(url,options));
  render(Preview,{config:{...config,frame_range:[3,8],video:{video_init_path:'input.mp4',extract_nth_frame:2,max_size:720}},history:false,view:'source'});
  expect(await screen.findByAltText('Source, frame 3')).toHaveAttribute('src',expect.stringContaining('frame=6'));
  await fireEvent.input(screen.getByRole('spinbutton',{name:'Frame',exact:true}),{target:{value:'4'}});
  expect(await screen.findByAltText('Source, frame 4')).toHaveAttribute('src',expect.stringContaining('frame=8'));
  expect(screen.queryByRole('button',{name:'Make video'})).not.toBeInTheDocument();
});

it('keeps the full source range accessible when the latest run rendered fewer frames',async()=>{
  render(Preview,{config:{...config,frame_range:[0,0],video:{video_init_path:'input.mp4',extract_nth_frame:1,max_size:720}},history:false,view:'source'});
  await screen.findByAltText('Init, frame 1');
  const frame=screen.getByRole('spinbutton',{name:'Frame',exact:true});
  expect(frame).toHaveAttribute('max','19');
  await fireEvent.input(frame,{target:{value:'12'}});
  expect(await screen.findByAltText('Init, frame 12')).toHaveAttribute('src',expect.stringContaining('frame=12'));
});
