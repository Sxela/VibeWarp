import {render,screen,fireEvent,waitFor} from '@testing-library/svelte';
import {afterEach,expect,it,vi} from 'vitest';
import RawVideoPreview from './RawVideoPreview.svelte';
import Field from './Field.svelte';
afterEach(()=>vi.unstubAllGlobals());
it.each(['foreground_opacity','background_opacity'])('edits %s on a 0–1 slider while storing fractional opacity',async(name)=>{
  const onchange=vi.fn();
  render(Field,{name,schema:{type:'number',slider:{min:0,max:1,step:0.01,scale:1}},value:0.5,onchange});
  const slider=screen.getByRole('slider');
  expect(slider).toHaveValue('0.5');
  expect(slider).toHaveAttribute('min','0');expect(slider).toHaveAttribute('max','1');
  await fireEvent.input(slider,{target:{value:'0.75'}});
  expect(onchange).toHaveBeenCalledWith(0.75);
});
it('writes RGB arrays from the native color picker',async()=>{
  const onchange=vi.fn();
  render(Field,{name:'foreground_color',schema:{type:'array',rgb_color:true},value:[10,20,30],onchange});
  const picker=screen.getByLabelText('Foreground Color');
  expect(picker).toHaveValue('#0a141e');
  await fireEvent.input(picker,{target:{value:'#ff0080'}});
  expect(onchange).toHaveBeenCalledWith([255,0,128]);
});
it('queues the chosen frame with its preview reference and shows the completed result',async()=>{
  const onjob=vi.fn(),config={video:{},mask:{rvm_enabled:true,raw_color_enabled:true}};
  const fetch=vi.fn(async()=>({ok:true,json:async()=>({id:'preview',state:'queued',operation:'raw_preview'})}));
  vi.stubGlobal('fetch',fetch);
  const {rerender}=render(RawVideoPreview,{config,onjob});
  await fireEvent.input(screen.getByLabelText('Raw preview frame'),{target:{value:'5'}});
  await fireEvent.input(screen.getByLabelText('Raw preview reference image'),{target:{value:'style.png'}});
  await fireEvent.click(screen.getByRole('button',{name:'Preview one frame'}));
  await waitFor(()=>expect(onjob).toHaveBeenCalled());
  expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({config,frame:5,reference_path:'style.png'});
  await rerender({config,onjob,job:{id:'preview',state:'completed',preview_available:true,revision:3}});
  expect(screen.getByAltText('Prepared raw video frame 5')).toHaveAttribute('src','/api/jobs/preview/preview?v=3');
});
it('shows preview errors without pretending a preview was produced',async()=>{
  vi.stubGlobal('fetch',vi.fn(async()=>({ok:false,json:async()=>({detail:'Choose a reference image'})})));
  render(RawVideoPreview,{config:{mask:{rvm_enabled:true}},onjob:vi.fn()});
  await fireEvent.click(screen.getByRole('button',{name:'Preview one frame'}));
  expect(await screen.findByRole('alert')).toHaveTextContent('Choose a reference image');
  expect(screen.queryByRole('img')).not.toBeInTheDocument();
});

it('allows an RVM preview when render masking alone is enabled',()=>{
  render(RawVideoPreview,{config:{mask:{use_background_mask:true,rvm_enabled:false,raw_color_enabled:false}},onjob:vi.fn()});
  expect(screen.getByRole('button',{name:'Preview one frame'})).toBeEnabled();
});

it('clears an old image and explains validation errors when the next preview fails',async()=>{
  const config={video:{},mask:{rvm_enabled:true}},onjob=vi.fn();
  const fetch=vi.fn()
    .mockResolvedValueOnce({ok:true,json:async()=>({id:'old',state:'queued'})})
    .mockResolvedValueOnce({ok:false,json:async()=>({detail:[{path:'mask.foreground_opacity',message:'Choose opacity between 0 and 1'}]})});
  vi.stubGlobal('fetch',fetch);
  const {rerender}=render(RawVideoPreview,{config,onjob});
  await fireEvent.click(screen.getByRole('button',{name:'Preview one frame'}));
  await waitFor(()=>expect(onjob).toHaveBeenCalled());
  await rerender({config,onjob,job:{id:'old',state:'completed',preview_available:true,revision:1}});
  expect(screen.getByRole('img')).toBeInTheDocument();
  await fireEvent.click(screen.getByRole('button',{name:'Preview one frame'}));
  expect(await screen.findByRole('alert')).toHaveTextContent('mask.foreground_opacity: Choose opacity between 0 and 1');
  expect(screen.queryByRole('img')).not.toBeInTheDocument();
});
