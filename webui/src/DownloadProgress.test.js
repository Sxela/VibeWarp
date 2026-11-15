import {render, screen, fireEvent} from '@testing-library/svelte';
import {describe,it,expect,vi} from 'vitest';
import DownloadProgress from './DownloadProgress.svelte';

describe('model download progress',()=>{
  it('hides ready models and removes the list when all transfers finish',async()=>{
    const files=[{id:'a',path:'a',name:'ready.pt',status:'ready'},
      {id:'b',path:'b',name:'active.pt',status:'downloading'}];
    const {container,rerender}=render(DownloadProgress,{files});
    expect(screen.queryByText('ready.pt')).not.toBeInTheDocument();
    expect(screen.getByText('active.pt')).toBeInTheDocument();
    await rerender({files:files.map(file=>({...file,status:'ready'}))});
    expect(container.querySelector('details')).toBeNull();
  });
  it('controls individual rows and stays collapsed on progress updates',async()=>{
    const onaction=vi.fn();
    const files=[{id:'a',path:'a',name:'active',status:'downloading',can_pause:true,can_cancel:true},
      {id:'b',path:'b',name:'paused',status:'paused',can_resume:true,can_cancel:true},
      {id:'c',path:'c',name:'queued',status:'queued',can_pause:true,can_cancel:true}];
    const {container,rerender}=render(DownloadProgress,{files,onaction});
    await fireEvent.click(screen.getByRole('button',{name:'Pause active'}));
    expect(onaction).toHaveBeenLastCalledWith(files[0],'pause');
    await fireEvent.click(screen.getByRole('button',{name:'Resume paused'}));
    expect(onaction).toHaveBeenLastCalledWith(files[1],'resume');
    await fireEvent.click(screen.getByRole('button',{name:'Cancel queued'}));
    expect(onaction).toHaveBeenLastCalledWith(files[2],'cancel');
    const details=container.querySelector('details');
    details.open=false;
    await fireEvent(details,new Event('toggle'));
    await rerender({files:files.map(file=>({...file,downloaded:50})),onaction});
    expect(details.open).toBe(false);
    expect(screen.getByText(/1 active · 1 queued/)).toBeInTheDocument();
  });
  it('shows byte progress with an accessible bar',()=>{
    render(DownloadProgress,{files:[{path:'/models/a',name:'checkpoint.pt',downloaded:5,total:10,status:'downloading'}]});
    let bar=screen.getByRole('progressbar',{name:'Download checkpoint.pt'});
    expect(bar).toHaveAttribute('value','5');
    expect(bar).toHaveAttribute('max','10');
    expect(screen.getByText('5 B / 10 B')).toBeInTheDocument();
  });
  it('uses indeterminate progress when the server gives no size',()=>{
    render(DownloadProgress,{files:[{path:'hub',name:'encoder',downloaded:0,total:null,status:'downloading'}]});
    expect(screen.getByRole('progressbar')).not.toHaveAttribute('value');
  });
  it('keeps download failures visible',()=>{
    render(DownloadProgress,{files:[{path:'hub',name:'encoder',downloaded:0,status:'failed',error:'Connection lost'}]});
    expect(screen.getByText('Connection lost')).toBeInTheDocument();
  });
});
