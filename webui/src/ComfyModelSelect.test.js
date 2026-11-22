import {render,screen,fireEvent} from '@testing-library/svelte';
import {it,expect,vi} from 'vitest';
import ComfyModelSelect from './ComfyModelSelect.svelte';

it('shows the suggested missing model first and submits exact server filenames',async()=>{
  const onchange=vi.fn();
  render(ComfyModelSelect,{props:{title:'Text encoder',anchor:'encoder',value:'qwen-stock',onchange,
    info:{suggested:'qwen-stock',available:['other','QWEN-similar','qwen-closer']}}});
  const select=screen.getByRole('combobox',{name:'Text encoder'});
  expect(select.options[0].value).toBe('qwen-stock');
  expect(select.options[0].textContent).toBe('qwen-stock (Suggested · Missing)');
  await fireEvent.change(select,{target:{value:'QWEN-similar'}});
  expect(onchange).toHaveBeenCalledWith('QWEN-similar');
});

it('preserves the selection during refresh and displays an inventory failure',async()=>{
  const onchange=vi.fn();
  const {rerender}=render(ComfyModelSelect,{props:{title:'VAE',anchor:'vae',value:'custom',onchange,
    info:{suggested:'recommended',available:['custom']}}});
  await rerender({props:{title:'VAE',anchor:'vae',value:'custom',onchange,
    info:{suggested:'recommended',available:[],error:'ComfyUI is offline'}}});
  expect(screen.getByRole('combobox',{name:'VAE'})).toHaveValue('custom');
  expect(screen.getByText('ComfyUI is offline')).toBeInTheDocument();
  expect(onchange).not.toHaveBeenCalled();
});
