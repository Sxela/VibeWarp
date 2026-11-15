import {render,screen,fireEvent} from '@testing-library/svelte';
import {expect,it,vi} from 'vitest';
import LayerViewer from './LayerViewer.svelte';

const layers=[
  {id:'output',label:'Output',group:'Render',frames:[0,1]},
  {id:'debug:consistency_mask',label:'CC mask',group:'Optical flow',frames:[1]},
  {id:'debug:depth_detected',label:'Depth map',group:'ControlNet',frames:[]},
];

it('selects real diagnostic layers and never fabricates missing maps',async()=>{
  const {container}=render(LayerViewer,{layers,frame:1,selected:['output'],imageUrl:id=>`/image/${id}`});
  await fireEvent.click(screen.getByRole('button',{name:'CC mask'}));
  expect(screen.getByAltText('CC mask, frame 1')).toHaveAttribute('src','/image/debug:consistency_mask');
  await fireEvent.click(screen.getByRole('button',{name:/Depth map/}));
  expect(screen.getByText('Not saved for frame 1')).toBeInTheDocument();
  expect(container.querySelectorAll('img')).toHaveLength(2);
  expect(screen.getByText(/white keeps the warp/)).toBeInTheDocument();
});

it('enlarges the selected saved layer and keeps phone focus independent of selection',async()=>{
  const onzoom=vi.fn();
  const {container}=render(LayerViewer,{layers,frame:1,selected:['output','debug:consistency_mask'],imageUrl:id=>`/image/${id}`,onzoom});
  await fireEvent.change(screen.getByRole('combobox',{name:'Displayed diagnostic layer'}),{target:{value:'debug:consistency_mask'}});
  expect(container.querySelector('figure.focused figcaption')).toHaveTextContent('CC mask');
  expect(container.querySelectorAll('figure')).toHaveLength(2);
  await fireEvent.click(screen.getByRole('button',{name:'Enlarge CC mask'}));
  expect(onzoom).toHaveBeenCalledWith('debug:consistency_mask');
});

it('provides a source preview before a run exists',()=>{
  render(LayerViewer,{view:'source',sourceUrl:'/source.jpg',imageUrl:vi.fn()});
  expect(screen.getByAltText('Source, frame 0')).toHaveAttribute('src','/source.jpg');
});

it('shows the current source instead of a previous run input in the render workspace',()=>{
  render(LayerViewer,{view:'source',layers:[{id:'init',label:'Init',frames:[0]}],sourceUrl:'/new-source.jpg',imageUrl:()=>'/old-source.jpg'});
  expect(screen.getByAltText('Init, frame 0')).toHaveAttribute('src','/new-source.jpg');
});

it('starts with one actionable source pane and introduces comparison when output exists',async()=>{
  const onchoose=vi.fn(),imageUrl=id=>`/image/${id}`;
  const props={view:'compare',layers:[],imageUrl,onchoose};
  const {container,rerender}=render(LayerViewer,props);
  expect(container.querySelectorAll('figure')).toHaveLength(1);
  await fireEvent.click(screen.getByRole('button',{name:'Choose input video'}));
  expect(onchoose).toHaveBeenCalledOnce();
  await rerender({...props,sourceUrl:'/source.jpg',liveUrl:'/live.jpg'});
  expect(container.querySelectorAll('figure')).toHaveLength(2);
  expect(screen.getByRole('tab',{name:'Source / output'})).toHaveAttribute('aria-selected','true');
});

it('respects an explicit source view when output arrives',async()=>{
  const {container,rerender}=render(LayerViewer,{view:'source',sourceUrl:'/source.jpg',imageUrl:vi.fn()});
  await rerender({view:'source',sourceUrl:'/source.jpg',liveUrl:'/live.jpg',imageUrl:vi.fn()});
  expect(container.querySelectorAll('figure')).toHaveLength(1);
  expect(screen.queryByAltText('Output, frame 0')).toBeNull();
});

it('distinguishes loading, input failure and pending output from a missing saved image',async()=>{
  const props={view:'source',layers:[{id:'init',label:'Old input',frames:[0]}],imageUrl:vi.fn()};
  const {container,rerender}=render(LayerViewer,{...props,sourcePending:true});
  expect(screen.getByText('Loading source preview…')).toBeInTheDocument();
  expect(container.querySelector('img')).toBeNull();
  expect(screen.queryByRole('button',{name:'Choose input video'})).toBeNull();
  await rerender({...props,sourcePending:false,sourceError:'Input missing'});
  expect(screen.getByText('Input missing')).toBeInTheDocument();
  expect(screen.getByRole('button',{name:'Check input video'})).toBeInTheDocument();
  expect(container.querySelector('img')).toBeNull();
  await rerender({...props,sourcePending:false,sourceError:'',sourceEmpty:true});
  expect(screen.getByRole('button',{name:'Choose input video'})).toBeInTheDocument();
  await rerender({...props,view:'output'});
  expect(screen.getByText('Waiting for first output')).toBeInTheDocument();
  expect(screen.queryByText(/Not saved/)).toBeNull();
  await rerender({...props,view:'output',frame:1,layers:[{id:'output',label:'Output',frames:[0]}]});
  expect(screen.getByText('Not saved for frame 1')).toBeInTheDocument();
});
