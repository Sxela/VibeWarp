import {render,screen,fireEvent} from '@testing-library/svelte';
import {expect,it,vi} from 'vitest';
import RequiredModels from './RequiredModels.svelte';

it('distinguishes an unavailable Comfy inventory from missing weights',()=>{
  render(RequiredModels,{files:[{name:'remote.safetensors',path:'diffusion_models/remote.safetensors',
    source:'comfy',server_url:'http://comfy.example:8188',status:'unknown',downloadable:false,error:'ComfyUI is offline'}]});
  expect(screen.getByText('Unavailable')).toBeInTheDocument();
  expect(screen.getByText('ComfyUI is offline')).toBeInTheDocument();
  expect(screen.getByText('http://comfy.example:8188')).toBeInTheDocument();
  expect(screen.getByRole('button',{name:'Download required models'})).toBeDisabled();
});

it('keeps the required list collapsed through updates with download still accessible',async()=>{
  const ondownload=vi.fn();
  const files=[{name:'weights.pt',path:'models/weights.pt',status:'missing',downloadable:true}];
  const {container,rerender}=render(RequiredModels,{files,ondownload});
  const details=container.querySelector('details');
  details.open=false;
  await fireEvent(details,new Event('toggle'));
  await rerender({files:files.map(file=>({...file,total:100})),ondownload});
  expect(details.open).toBe(false);
  await fireEvent.click(screen.getByRole('button',{name:'Download required models'}));
  expect(ondownload).toHaveBeenCalledTimes(1);
});

it('shows only missing models and hides the panel once every model is ready',async()=>{
  const files=[{name:'ready.pt',path:'ready',status:'present',downloadable:true},
    {name:'missing.pt',path:'missing',status:'missing',downloadable:true}];
  const {rerender}=render(RequiredModels,{files,ondownload:vi.fn()});
  expect(screen.queryByText('ready.pt')).not.toBeInTheDocument();
  expect(screen.getByText('missing.pt')).toBeInTheDocument();
  await rerender({files:files.map(file=>({...file,status:'present'})),ondownload:vi.fn()});
  expect(screen.queryByRole('region',{name:'Required models'})).not.toBeInTheDocument();
  expect(screen.queryByRole('button',{name:'Download required models'})).not.toBeInTheDocument();
});
