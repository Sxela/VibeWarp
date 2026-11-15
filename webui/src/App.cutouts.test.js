import {render,screen,fireEvent,waitFor,within} from '@testing-library/svelte';
import {afterEach,expect,it,vi} from 'vitest';
import App from './App.svelte';

afterEach(()=>{vi.unstubAllGlobals();localStorage.clear();});

it('migrates stored cutout axes before defaults and edits imported/new schedules in the same percentage domain',async()=>{
  const old={model_version:'disco_512',output_dir:'out',batch_name:'test',video:{},disco:{steps:6,style_strength:4/6,cut_schedule_original_time:true,cut_overview:{0:4,400:2,600:0}}};
  const migrated={...old,disco:{steps:6,style_strength:4/6,cut_schedule_version:2,cut_overview:{0:4,'33.33333333333333':2,'66.66666666666666':0}}};
  const schema={workflows:[{id:'prompts',label:'Prompts & sampling',groups:['Disco guidance']}],
    model_options:[{value:'disco_512',label:'Disco 512'}],model_family_by_version:{disco_512:'disco'},
    properties:{disco:{type:'dataclass',properties:{cut_overview:{type:'object',tier:'advanced',group:'Disco guidance',workflows:['prompts'],schedule_domain:'inference_progress',schedule_value:'cut_count'}}}}};
  localStorage.setItem('vibewarp.config.v1',JSON.stringify(old));
  const response=data=>({ok:true,json:async()=>data});
  const fetchMock=vi.fn(async(url,options)=>{
    if(url==='/api/config')return response({schema,defaults:migrated});
    if(url==='/api/config/import')return response({config:migrated});
    if(url==='/api/jobs')return response([]);
    if(url==='/api/models/downloads')return response({files:[],tasks:[]});
    if(url==='/api/models/required')return response({files:[]});
    if(url.startsWith('/api/preview/runs?'))return response({runs:[],root:'out/test'});
    throw new Error(`Unexpected fetch ${url}`);
  });
  vi.stubGlobal('fetch',fetchMock);vi.stubGlobal('EventSource',class{close(){}});
  const {container}=render(App);
  await screen.findByRole('button',{name:'Paste Cut Overview JSON'});
  const editor=within(container.querySelector('#field-disco-cut_overview'));
  const importCall=fetchMock.mock.calls.find(([url])=>url==='/api/config/import');
  expect(JSON.parse(JSON.parse(importCall[1].body).content).disco.cut_schedule_original_time).toBe(true);
  expect(editor.getAllByRole('spinbutton',{name:'Inference progress (%)'})[1]).toHaveValue(33.33333333333333);
  await fireEvent.change(editor.getAllByRole('spinbutton',{name:'Inference progress (%)'})[1],{target:{value:'33.25'}});
  await waitFor(()=>expect(JSON.parse(localStorage.getItem('vibewarp.config.v1')).disco.cut_overview['33.25']).toBe(2));
  await fireEvent.click(editor.getByRole('button',{name:'Paste Cut Overview JSON'}));
  await fireEvent.input(await editor.findByRole('textbox',{name:'Schedule JSON'}),{target:{value:'{"0":4,"40":2,"60":0}'}});
  await fireEvent.click(editor.getByRole('button',{name:'Paste keyframes'}));
  expect(editor.getAllByRole('spinbutton',{name:'Inference progress (%)'})[1]).toHaveValue(40);
  expect(editor.getAllByRole('spinbutton',{name:'Cut count per batch'})[1]).toHaveValue(2);
  await fireEvent.focus(editor.getByRole('button',{name:'Help: Cut Overview'}));
  expect(await screen.findByRole('tooltip')).toHaveTextContent('Values hold until the next key; no interpolation');
  expect(JSON.parse(localStorage.getItem('vibewarp.config.v1')).disco.cut_schedule_version).toBe(2);
});
