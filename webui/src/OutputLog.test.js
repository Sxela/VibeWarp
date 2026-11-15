import {render,screen,fireEvent} from '@testing-library/svelte';
import {beforeEach,afterEach,expect,it,vi} from 'vitest';
import OutputLog from './OutputLog.svelte';

beforeEach(()=>localStorage.clear());
afterEach(()=>vi.unstubAllGlobals());
const handle=()=>screen.getByRole('separator',{name:'Resize system log'});

it('resizes by dragging the top handle and remembers the height when reopened',async()=>{
  vi.stubGlobal('PointerEvent',MouseEvent);
  const {unmount}=render(OutputLog,{logs:['first line','next line']});
  expect(screen.getByText(/first line/)).toHaveTextContent('next line');
  expect(handle()).toHaveAttribute('aria-valuenow','240');
  await fireEvent.pointerDown(handle(),{clientY:500,button:0});
  await fireEvent.pointerMove(window,{clientY:400});
  await fireEvent.pointerUp(window,{clientY:400});
  expect(handle()).toHaveAttribute('aria-valuenow','340');
  expect(localStorage.getItem('vibewarp.log.height')).toBe('340');
  unmount();render(OutputLog);
  expect(handle()).toHaveAttribute('aria-valuenow','340');
});

it('supports keyboard resizing with bounded limits and retains logs',async()=>{
  render(OutputLog,{logs:['a warning']});
  await fireEvent.keyDown(handle(),{key:'ArrowUp'});
  expect(handle()).toHaveAttribute('aria-valuenow','272');
  await fireEvent.keyDown(handle(),{key:'ArrowDown'});
  expect(handle()).toHaveAttribute('aria-valuenow','240');
  await fireEvent.keyDown(handle(),{key:'Home'});
  await fireEvent.keyDown(handle(),{key:'ArrowDown'});
  expect(handle()).toHaveAttribute('aria-valuenow','120');
  await fireEvent.keyDown(handle(),{key:'End'});
  expect(handle().getAttribute('aria-valuenow')).toBe(handle().getAttribute('aria-valuemax'));
  expect(screen.getByText('a warning')).toBeInTheDocument();
});

it('rejects invalid saved heights and clamps to a shorter window without losing the preference',async()=>{
  localStorage.setItem('vibewarp.log.height','Infinity');
  const view=render(OutputLog);
  expect(handle()).toHaveAttribute('aria-valuenow','240');view.unmount();
  localStorage.setItem('vibewarp.log.height','600');render(OutputLog);
  vi.stubGlobal('innerHeight',500);await fireEvent.resize(window);
  expect(handle()).toHaveAttribute('aria-valuenow','350');
  expect(localStorage.getItem('vibewarp.log.height')).toBe('600');
});
