<script>
  let {
    value = [], onchange, videoPath = '', frameRange = [0, 0],
    extractNth = 1, maxReferences = 10, sourceLabels = {},
    labelOpacity = 0.7, onLabelOpacityChange = () => {},
  } = $props();

  const choices = [
    ['raw', 'Raw frame'],
    ['previous', 'Previous stylized frame'],
    ['warped', 'Previous stylized + warp + consistency'],
    ['none', 'None'],
    ['upload', 'Upload image'],
  ];
  let uploading = $state(-1);
  let uploadError = $state({});
  let dragOver = $state(-1);
  let inputs = $state([]);

  let refs = $derived(value?.length ? value : [
    {source:'raw', label:false, image_path:''},
    {source:'none', label:false, image_path:''},
  ]);
  let visible = $derived.by(() => {
    let end = refs.findIndex((ref, index) => index > 0 && ref.source === 'none');
    return refs.slice(0, end < 0 ? Math.min(refs.length, maxReferences) : end + 1);
  });
  let sourceFrame = $derived(
    Math.max(0, Number(frameRange?.[0] ?? 0))
      * Math.max(1, Number(extractNth || 1)));
  let videoThumb = $derived(videoPath
    ? `/api/video/thumbnail?path=${encodeURIComponent(videoPath)}&frame=${sourceFrame}`
    : '');

  function normalized(list) {
    let result = list.slice(0, maxReferences).map(ref => ({
      source: ref.source || 'none',
      label: !!ref.label,
      image_path: ref.image_path || '',
    }));
    if (!result.length) result.push({source:'raw', label:false, image_path:''});
    let none = result.findIndex((ref, index) => index > 0 && ref.source === 'none');
    if (none >= 0) result = result.slice(0, none + 1);
    if (none < 0 && result.length < maxReferences)
      result.push({source:'none', label:false, image_path:''});
    return result;
  }
  function patch(index, changes) {
    let next = refs.map(ref => ({...ref}));
    next[index] = {...next[index], ...changes};
    if (changes.source && changes.source !== 'upload')
      next[index].image_path = '';
    onchange(normalized(next));
  }
  async function upload(index, file) {
    if (!file) return;
    uploading = index;
    uploadError = {...uploadError, [index]: ''};
    try {
      let response = await fetch(
        `/api/references/upload?filename=${encodeURIComponent(file.name)}`,
        {method:'POST', headers:{'content-type':file.type || 'application/octet-stream'},
         body:file});
      let data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Could not upload image');
      patch(index, {source:'upload', image_path:data.path});
    } catch (error) {
      uploadError = {...uploadError, [index]:error.message || 'Could not upload image'};
    } finally {
      uploading = -1;
    }
  }
  function drop(index, event) {
    event.preventDefault();
    dragOver = -1;
    upload(index, event.dataTransfer?.files?.[0]);
  }
  const preview = (ref) => ref.source === 'upload' && ref.image_path
    ? `/api/references/thumbnail?path=${encodeURIComponent(ref.image_path)}`
    : videoThumb;
  const sourceName = (source) => sourceLabels?.[source]
    || choices.find(([key]) => key === source)?.[1] || source;
</script>

<div class="reference-list">
  <div class="intro">
    <div><b>Ordered model references</b>
      <span>Image 1 is always sent. Choose another source to reveal the next slot.</span>
    </div>
    <em>{visible.filter(ref=>ref.source!=='none').length}/{maxReferences} active</em>
  </div>
  <div class="opacity-control">
    <span>Label opacity</span>
    <input type="range" min="0" max="1" step="0.05"
           aria-label="Reference label opacity"
           value={labelOpacity}
           oninput={(event)=>onLabelOpacityChange(Number(event.currentTarget.value))}/>
    <output>{Math.round(labelOpacity * 100)}%</output>
  </div>

  <div class="cards">
    {#each visible as ref, index (`${index}-${ref.source}-${ref.image_path}`)}
      <article class:required={index===0} class:inactive={ref.source==='none'}>
        <header><b>Image {index+1}</b>
          {#if index===0}<span>Required</span>{:else if ref.source==='none'}<span>Not sent</span>{/if}
        </header>

        {#if ref.source==='upload'}
          <button class="drop" class:over={dragOver===index}
                  ondragover={(e)=>{e.preventDefault();dragOver=index}}
                  ondragleave={()=>dragOver=-1} ondrop={(e)=>drop(index,e)}
                  onclick={()=>inputs[index]?.click()}
                  aria-label={`Upload reference image ${index+1}`}>
            {#if ref.image_path}
              <img src={preview(ref)} alt={`Uploaded reference image ${index+1}`}/>
              <span class="replace">Replace image</span>
            {:else}
              <i>+</i><span>{uploading===index?'Uploading…':'Drop image or click'}</span>
            {/if}
          </button>
          <input class="picker" type="file" accept="image/*"
                 bind:this={inputs[index]}
                 onchange={(e)=>upload(index,e.target.files?.[0])}/>
          {#if uploadError[index]}<small class="err">{uploadError[index]}</small>{/if}
        {:else}
          <div class="thumb" class:empty={!videoThumb || ref.source==='none'}>
            {#if ref.source==='none'}
              <i>+</i><span>Select a source to add image {index+1}</span>
            {:else if videoThumb}
              <img src={preview(ref)} alt={`First render-range frame preview for ${sourceName(ref.source)}`}/>
              {#if ref.source!=='raw'}
                <span>{index===0 ? 'Raw fallback on first frame' : 'Not sent on first frame'}</span>
              {/if}
            {:else}
              <span>Choose an input video to preview this reference</span>
            {/if}
          </div>
        {/if}

        <label><span>Source</span>
          <select value={ref.source}
                  onchange={(e)=>patch(index,{source:e.target.value})}>
            {#each choices.filter(([key])=>index>0 || !['none','upload'].includes(key)) as choice}
              <option value={choice[0]}>{choice[1]}</option>
            {/each}
          </select>
        </label>
        {#if ref.source!=='none'}
          <label class="label-toggle">
            <input type="checkbox" checked={ref.label}
                   onchange={(e)=>patch(index,{label:e.target.checked})}/>
            <span>Bake @Image{index+1} label</span>
          </label>
        {/if}
      </article>
    {/each}
  </div>
</div>

<style>
  .reference-list{display:flex;flex-direction:column;gap:14px}
  .intro{display:flex;align-items:flex-start;justify-content:space-between;gap:16px}
  .intro b,.intro span{display:block}.intro b{color:#e7eaee;font-size:13px}
  .intro span{margin-top:4px;color:#747d88;font-size:11px}
  .intro em{white-space:nowrap;color:#a9c66b;font:11px Consolas,monospace;font-style:normal}
  .opacity-control{display:flex;align-items:center;gap:10px;color:#8f98a3;font-size:11px}
  .opacity-control input{width:180px;padding:0;border:0;background:transparent;accent-color:#b8dc50}
  .opacity-control output{min-width:34px;color:#dbe4cb;font:11px Consolas,monospace}
  .cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px}
  article{min-width:0;border:1px solid #303640;background:#0d1014;border-radius:11px;padding:12px}
  article.required{border-color:#4b5836}article.inactive{border-style:dashed;background:#0b0d10}
  article header{display:flex;justify-content:space-between;align-items:center;margin-bottom:9px}
  article header b{font-size:12px;color:#e3e6ea}article header span{font-size:9px;color:#87936f;text-transform:uppercase;letter-spacing:.1em}
  .thumb,.drop{position:relative;width:100%;aspect-ratio:1;border:1px solid #292f37;border-radius:8px;background:#080a0d;overflow:hidden;margin:0 0 11px}
  .thumb img,.drop img{width:100%;height:100%;object-fit:cover;display:block}
  .thumb>span,.drop>span{color:#68717d;font-size:10px;text-align:center;padding:12px}
  .thumb.empty,.drop{display:grid;place-items:center;align-content:center;gap:5px}
  .thumb i,.drop i{color:#9daa83;font:300 32px/1 system-ui;font-style:normal}
  .thumb img+span{position:absolute;left:7px;bottom:7px;padding:4px 6px;border-radius:5px;background:#090b0dcc;color:#b9c2ad;text-transform:uppercase;letter-spacing:.08em;font-size:8px}
  .drop{cursor:pointer;color:#7d8793}.drop:hover,.drop.over{border-color:#9fbe4d;background:#11170d}
  .drop .replace{position:absolute;left:50%;bottom:8px;transform:translateX(-50%);padding:5px 8px;border-radius:6px;background:#090b0ddd;color:#dce4d0;white-space:nowrap}
  .picker{display:none}
  label{display:flex;flex-direction:column;gap:6px;color:#8f98a3;font-size:10px}
  select{width:100%;border:1px solid #303640;background:#11151a;color:#e5e8ec;border-radius:7px;padding:8px;outline:none;font-size:11px}
  select:focus{border-color:#8ea834}
  .label-toggle{flex-direction:row;align-items:center;margin-top:9px;color:#929aa4}
  .label-toggle input{appearance:none;width:30px;height:17px;padding:2px;border:1px solid #37404a;background:#11151a;border-radius:12px}
  .label-toggle input:checked{background:#b8dc50}.label-toggle input:after{content:'';display:block;width:11px;height:11px;border-radius:50%;background:#69727d}.label-toggle input:checked:after{transform:translateX(13px);background:#111}
  .err{display:block;margin:-5px 0 8px;color:#f18b91;font-size:10px}
  @media(max-width:700px){.cards{grid-template-columns:1fr 1fr}}
</style>
