// Action menus keep native links/buttons, including keyboard activation.
export function closeOnSelection(menu) {
  function select(event) {
    const action=event.target.closest?.('a[href],button');
    if(!action||!menu.contains(action)||action.closest('summary')||action.disabled||action.getAttribute('aria-disabled')==='true')return;
    menu.open=false;
    menu.querySelector('summary')?.focus({preventScroll:true});
  }
  menu.addEventListener('click',select,true);
  return {destroy(){menu.removeEventListener('click',select,true);}};
}
