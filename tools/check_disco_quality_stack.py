"""Real-weight guided quality/turbo smoke checks; no quality-optimum claim."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vibewarp'/'vendor'))

import numpy as np
import torch
from PIL import Image

from vibewarp.config import RunConfig
from vibewarp.core.disco_lora import AdapterLayer,DistilledDiscoBackend,sha256
from vibewarp.render_events import render_event_scope


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality',type=Path,required=True)
    parser.add_argument('--turbo',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--size',type=int,default=512)
    parser.add_argument('--output',type=Path,default=Path('.cache/disco-quality-stack-smoke'))
    parser.add_argument('--deterministic',action='store_true',help='Disable random augmentations and require deterministic CUDA algorithms')
    args=parser.parse_args()
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(args.deterministic)
    args.output.mkdir(parents=True,exist_ok=True)
    config=RunConfig(model_version='disco_512')
    d=config.disco
    d.quality_lora_enabled=True;d.quality_lora_path=str(args.quality.resolve())
    d.use_secondary_model=False;d.encoders=['ViT-B-32']
    d.steps_schedule=[6];d.style_strength_schedule=[.5]
    d.clip_guidance_scale=70000;d.guidance_noise_cap_percent=15
    d.init_scale=d.image_scale=d.range_scale=d.tv_scale=d.sat_scale=0
    d.cut_overview={0:4};d.cut_innercut={0:0};d.cutn_batches=1
    d.skip_augs=args.deterministic
    d.preview_interval=0;d.inpaint=False;d.eta=0
    config.flow.flow_warp=False
    config.video.width=config.video.height=args.size
    init=Image.open(args.input).convert('RGB').resize((args.size,args.size),Image.Resampling.LANCZOS)
    prompt=('A beautiful highly detailed matte painting of a cyberpunk cat, greg rutkowski, Trending on artstation:1\n'
            'black gold color scheme:1\ndof blur:-1')
    results=[]
    report={'scope':'Functional first-frame checks; single ViT-B/32, four overview cutouts, not the full handoff candidate or a video-coherence comparison.',
        'size':args.size,'deterministic':args.deterministic,'augmentations':not d.skip_augs,
        'quality':str(args.quality),'quality_sha256':sha256(args.quality),
        'turbo':str(args.turbo),'turbo_sha256':sha256(args.turbo),'runs':results}
    for stacked in (False,True):
        d.sampler='distilled' if stacked else 'ddim'
        d.few_step_lora_path=str(args.turbo.resolve()) if stacked else ''
        backend=DistilledDiscoBackend(config)
        assert backend.has_turbo==stacked
        if not stacked: assert backend.adapter_path is None
        report['quality_iteration']=backend.quality_metadata.get('iteration')
        modes=[('vanilla',0.,0.),('quality-only',1.,0.),('turbo-only',0.,1.),('stacked',1.,1.)] if stacked else [('quality-without-turbo-file',1.,0.)]
        for label,q,t in modes:
            d.quality_lora_strength,d.lora_strength=q,t
            events=[];started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            with render_event_scope(events.append):
                image=backend.render(config,init,prompt,'',100500,guidance_image=init)
            torch.cuda.synchronize()
            assert len(events)==3
            assert [int(event['original_timestep']) for event in events]==[499,332,165]
            assert all(event['guidance_update_rms']<=event['guidance_noise_budget_rms']+1e-5 for event in events)
            assert all(event['nonfinite_guidance_values']==0 for event in events)
            assert all(layer.active is None and layer.quality_strength==0 for layer in backend.model.modules() if isinstance(layer,AdapterLayer))
            assert all(parameter.grad is None for parameter in backend.model.parameters())
            image.save(args.output/f'{label}.png')
            row={'mode':label,'quality_strength':q,'turbo_strength':t,'seconds':time.perf_counter()-started,
                 'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'updates':events}
            results.append(row)
            print(json.dumps({key:value for key,value in row.items() if key!='updates'}),flush=True)
            (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        del backend
        gc.collect();torch.cuda.empty_cache()
    a=np.asarray(Image.open(args.output/'quality-without-turbo-file.png')).astype('float32')
    b=np.asarray(Image.open(args.output/'quality-only.png')).astype('float32')
    report['quality_only_file_independence']={'exact_pixels':bool(np.array_equal(a,b)),
                                             'mean_abs_rgb':float(np.abs(a-b).mean())}
    (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['quality_only_file_independence']),flush=True)


if __name__=='__main__': main()
