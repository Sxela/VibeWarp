import {render,screen,fireEvent,waitFor} from '@testing-library/svelte';
import {afterEach,describe,expect,it,vi} from 'vitest';
import ScheduleEditor from './ScheduleEditor.svelte';
afterEach(()=>vi.unstubAllGlobals());
function editor(props){
  const onchange=vi.fn();const view=render(ScheduleEditor,{...props,onchange});
  const commit=async()=>{const value=onchange.mock.calls.at(-1)?.[0]??props.value;await view.rerender({...props,value,onchange});return value;};
  return {...view,onchange,commit};
}
const click=label=>fireEvent.click(screen.getByRole('button',{name:label}));
async function paste(text){await click(/^Paste .* JSON$/);await fireEvent.input(await screen.findByRole('textbox',{name:'Schedule JSON'}),{target:{value:text}});await click('Paste keyframes');}

describe('always-visible keyframes',()=>{
  it.each([[7,[7]],[{0:7,24:9},[7,9]],[[7,9],[7,9]]])('shows every value without mode switches: %j',async(value,expected)=>{
    editor({name:'cfg_scale_schedule',value});
    for(const name of ['Constant','Per-frame','Keyframes','JSON','Off'])expect(screen.queryByRole('button',{name,exact:true})).toBeNull();
    expect(screen.getAllByLabelText('Value').map(input=>Number(input.value))).toEqual(expected);
    expect(screen.getAllByLabelText('Frame').map(input=>Number(input.value))).toEqual(Array.isArray(value)?[0,1]:typeof value==='number'?[0]:[0,24]);
    expect(screen.getByRole('button',{name:/\+ keyframe/})).toBeInTheDocument();
  });
  it('edits flat lists as lossless frame-keyed values',async()=>{
    const {onchange,commit}=editor({name:'steps_schedule',value:[10,11,12]});
    await fireEvent.change(screen.getAllByLabelText('Value')[1],{target:{value:'20'}});
    expect(onchange).toHaveBeenLastCalledWith({0:10,1:20,2:12});await commit();
    expect(screen.getAllByLabelText('Value')).toHaveLength(3);
  });
  it('adds/removes keys while retaining all other values and rejects collisions',async()=>{
    const {onchange,commit}=editor({name:'steps_schedule',value:{0:10,20:30}});
    await fireEvent.change(screen.getAllByLabelText('Frame')[1],{target:{value:'0'}});
    expect(onchange).not.toHaveBeenCalled();await click(/\+ keyframe/);await commit();
    expect(onchange).toHaveBeenLastCalledWith({0:10,20:30,30:30});
    await fireEvent.click(screen.getAllByRole('button',{name:'Remove'})[1]);await commit();
    expect(onchange).toHaveBeenLastCalledWith({0:10,30:30});
  });
  it('keeps step help in a focusable tooltip and uses step keys',async()=>{
    editor({name:'sampler_scale_schedule',value:{0:50,15:100},axis:'step'});
    expect(screen.getAllByLabelText('Step')).toHaveLength(2);
    expect(screen.queryByRole('tooltip')).toBeNull();
    await fireEvent.focus(screen.getByRole('button',{name:'Help: Sampler Scale Schedule'}));
    expect(await screen.findByRole('tooltip')).toHaveTextContent(/only.*100%.*remains active.*off/i);
    await fireEvent.keyDown(window,{key:'Escape'});expect(screen.queryByRole('tooltip')).toBeNull();
  });
  it('uses backend labels and hold semantics for frame guidance',async()=>{
    const {container}=editor({name:'init_scale_schedule',label:'LPIPS frame guidance scale',value:{0:0,10:50000},interpolation:'hold'});
    await fireEvent.focus(screen.getByRole('button',{name:'Help: LPIPS frame guidance scale'}));
    expect(await screen.findByRole('tooltip')).toHaveTextContent('Values hold until the next key; no interpolation.');
    expect(container.querySelector('.spark path').getAttribute('d')).toContain('H100.0V0.0');
  });
  it('renders imported prompt lists, emits strings, and retains all keys',async()=>{
    const {onchange,commit}=editor({name:'text_prompts',kind:'text',value:{0:['cat','dog'],24:'scene'}});
    expect(screen.getAllByLabelText('Prompt')[0]).toHaveValue('cat | dog');
    await fireEvent.change(screen.getAllByLabelText('Prompt')[0],{target:{value:'new scene'}});await commit();
    expect(onchange).toHaveBeenLastCalledWith({0:'new scene',24:'scene'});
    await paste('{"0":"first","24":"second"}');await commit();
    expect(screen.getAllByLabelText('Prompt').map(input=>input.value)).toEqual(['first','second']);
  });
  it('allows an empty negative prompt schedule without inventing a value',async()=>{
    const {onchange}=editor({name:'negative_prompts',kind:'text',value:{}});
    expect(screen.queryByLabelText('Prompt')).toBeNull();await click(/\+ keyframe/);
    expect(onchange).toHaveBeenLastCalledWith({0:''});
  });
});
describe('JSON transfer',()=>{
  it('clears stale transfer errors when settings are replaced externally',async()=>{
    const {rerender,onchange}=editor({name:'steps_schedule',value:{0:20}});
    await paste('{bad');expect(screen.getByRole('alert')).toBeInTheDocument();
    await rerender({name:'steps_schedule',value:{0:10,24:30},onchange});
    expect(screen.queryByRole('alert')).toBeNull();expect(screen.queryByRole('textbox',{name:'Schedule JSON'})).toBeNull();
    expect(screen.getAllByLabelText('Value').map(input=>Number(input.value))).toEqual([10,30]);
  });
  it('copies the visible frame-keyed JSON',async()=>{
    const writeText=vi.fn(async()=>{});vi.stubGlobal('navigator',{clipboard:{writeText}});
    editor({name:'steps_schedule',value:[20]});await click('Copy Steps Schedule JSON');
    expect(JSON.parse(writeText.mock.calls[0][0])).toEqual({0:20});
  });
  it('reads clipboard JSON directly and supports legacy scalars/lists',async()=>{
    vi.stubGlobal('navigator',{clipboard:{readText:async()=>'[7,9]'}});
    const {onchange,commit}=editor({name:'cfg_scale_schedule',value:7});
    await click('Paste Cfg Scale Schedule JSON');await waitFor(()=>expect(onchange).toHaveBeenCalledWith({0:7,1:9}));await commit();
    expect(screen.getAllByLabelText('Value')).toHaveLength(2);
  });
  it.each(['{bad','{"-1":7}','{"0":"wrong"}','{"0":true}','{"0":1e999}','{"01":7,"1":9}'])('rejects invalid paste without changing values: %s',async(text)=>{
    const {onchange}=editor({name:'steps_schedule',value:{0:20}});await paste(text);
    expect(screen.getByRole('alert')).toBeInTheDocument();expect(onchange).not.toHaveBeenCalled();
    expect(screen.getByLabelText('Value')).toHaveValue(20);
  });
  it('provides selectable JSON when clipboard copy is denied',async()=>{
    const {onchange}=editor({name:'steps_schedule',value:{0:20}});await click('Copy Steps Schedule JSON');
    const textarea=await screen.findByRole('textbox',{name:'Schedule JSON'});
    expect(JSON.parse(textarea.value)).toEqual({0:20});expect(textarea).toHaveAttribute('readonly');expect(onchange).not.toHaveBeenCalled();
  });
  it('preserves nested per-step JSON and returns to chips after a simple paste',async()=>{
    const {onchange,commit}=editor({name:'cfg_scale_schedule',value:[[3,7,3]]});
    expect(screen.getByRole('textbox',{name:'Advanced schedule JSON'})).toHaveValue('[\n  [\n    3,\n    7,\n    3\n  ]\n]');
    await fireEvent.change(screen.getByRole('textbox',{name:'Advanced schedule JSON'}),{target:{value:'[[4,8,4]]'}});expect(onchange).toHaveBeenLastCalledWith([[4,8,4]]);await commit();
    await paste('{"0":7,"24":9}');await commit();expect(screen.getAllByLabelText('Frame')).toHaveLength(2);
  });
});
describe('sparkline', () => {
  const spark = () => document.querySelector('.spark');

  it('plots progress keys at their absolute percentages and holds through 100%',()=>{
    const {onchange}=editor({name:'cut_overview',axis:'progress',value:{0:4,40:2,60:0}});
    expect([...spark().querySelectorAll('circle')].map(dot=>Number(dot.getAttribute('cx')))).toEqual([0,40,60]);
    expect(spark().querySelector('path').getAttribute('d')).toMatch(/H100\.0V22\.0$/);
    expect(spark()).toHaveAccessibleName('Inference progress 0–100%; Schedule from 0 to 4');
    expect(onchange).not.toHaveBeenCalled();
  });

  it('shows a constant progress schedule across the entire axis',()=>{
    editor({name:'cut_overview',axis:'progress',value:{0:4}});
    expect(spark().querySelector('path').getAttribute('d')).toBe('M0.0,11.0 H100.0V11.0');
    expect(spark().querySelectorAll('circle')).toHaveLength(1);
  });

  it('shows zero before the first progress key without inventing editable keys',()=>{
    editor({name:'cut_overview',axis:'progress',value:{40:4}});
    expect(spark().querySelector('path').getAttribute('d')).toBe('M0.0,22.0 H40.0V0.0 H100.0V0.0');
    expect(screen.getAllByRole('spinbutton',{name:'Inference progress (%)'})).toHaveLength(1);
  });

  it('draws the shape of a multi-point schedule', () => {
    editor({ name: 'cfg_scale_schedule', value: { 0: 7, 20: 9, 40: 5 } });
    expect(spark()).toBeInTheDocument();
    expect(screen.getByText('5 – 9')).toBeInTheDocument();
  });

  it('is skipped where it would say nothing', () => {
    editor({ name: 'steps_schedule', value: [20] });        // a constant is a flat line
    expect(spark()).not.toBeInTheDocument();
  });

  it('is skipped for prompts — text has no magnitude', () => {
    editor({ name: 'text_prompts', value: { 0: 'a', 10: 'b' }, kind: 'text' });
    expect(spark()).not.toBeInTheDocument();
  });

  it('survives a flat schedule instead of dividing by zero', () => {
    editor({ name: 'steps_schedule', value: { 0: 5, 10: 5 } });
    expect(spark()).toBeInTheDocument();
    expect(spark().querySelector('path').getAttribute('d')).not.toContain('NaN');
  });
});
