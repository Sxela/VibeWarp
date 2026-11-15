import {render,screen,fireEvent} from '@testing-library/svelte';
import {expect,it,vi} from 'vitest';
import Field from './Field.svelte';

it('keeps hints out of the form until keyboard focus, without stealing the field label or toggling it',async()=>{
  const onchange=vi.fn();
  render(Field,{name:'enabled',schema:{type:'boolean',label:'Enable guidance'},value:false,onchange,hint:'Guidance explanation',error:'Fix this combination'});
  const checkbox=screen.getByRole('checkbox',{name:/Enable guidance/});
  expect(screen.getByText('Fix this combination')).toBeInTheDocument();
  expect(screen.queryByText('Guidance explanation')).toBeNull();
  const help=screen.getByRole('button',{name:'Help: Enable guidance'});
  await fireEvent.focus(help);
  expect(await screen.findByRole('tooltip')).toHaveTextContent('Guidance explanation');
  expect(help).toHaveAttribute('aria-describedby',screen.getByRole('tooltip').id);
  await fireEvent.keyDown(help,{key:' '});
  expect(checkbox).not.toBeChecked();expect(onchange).not.toHaveBeenCalled();
  await fireEvent.focus(help);await fireEvent.keyDown(window,{key:'Escape'});
  expect(screen.queryByRole('tooltip')).toBeNull();
});
