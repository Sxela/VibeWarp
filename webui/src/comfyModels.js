export function prefixLength(left,right) {
  const a=left.toLowerCase(),b=right.toLowerCase();
  let length=0;
  while(length<a.length&&length<b.length&&a[length]===b[length])length++;
  return length;
}

export function sortedComfyModels(models,suggested,current='') {
  const names=[...new Set(models)];
  const others=names.filter(name=>name!==suggested).sort((a,b)=>
    prefixLength(b,suggested)-prefixLength(a,suggested)
    ||a.toLowerCase().localeCompare(b.toLowerCase(),'en')||a.localeCompare(b,'en'));
  const result=suggested?[suggested,...others]:others;
  // Preserve imported/custom selections instead of silently selecting another file.
  if(current&&!result.includes(current))result.push(current);
  return result;
}
