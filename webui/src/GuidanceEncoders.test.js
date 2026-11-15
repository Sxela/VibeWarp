import {render,screen,fireEvent} from '@testing-library/svelte';
import {it,expect,vi} from 'vitest';
import GuidanceEncoders from './GuidanceEncoders.svelte';

it('selects modern guidance without discarding the selected classic encoder',async()=>{
  let onchange=vi.fn();
  render(GuidanceEncoders,{value:['ViT-B-32'],choices:['ViT-B-32','siglip2-base'],onchange});
  expect(screen.getByRole('checkbox',{name:'ViT-B-32'})).toBeChecked();
  await fireEvent.click(screen.getByRole('checkbox',{name:'siglip2-base'}));
  expect(onchange).toHaveBeenCalledWith(['ViT-B-32','siglip2-base']);
});
