"""Experimental six-step resampling of a four-step-trained Disco adapter.

Does not modify adapter metadata or the application's trained-grid guard.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image,ImageDraw,ImageFont

from benchmark_disco_attention import image_metrics
from vibewarp.config import DISCO_DISTILLED_PRESET
from vibewarp.config_io import config_from_dict
from vibewarp.core.disco import tensor_image
from vibewarp.core.disco_lora import DistilledDiscoBackend,GRIDS,rollout,sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher',type=Path,required=True)
    parser.add_argument('--adapter',type=Path,required=True)
    parser.add_argument('--reference',type=Path,default=Path('.cache/disco-lora-teacher-comparison'))
    parser.add_argument('--output',type=Path,default=Path('.cache/disco-lora-six-step-experiment'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    prior=json.loads((args.reference/'results.json').read_text())
    if sha256(args.adapter)!=prior['adapter_sha256']:
        raise ValueError('Adapter does not match the cached teacher comparison')
    cached=torch.load(args.reference/'raw_tensors.pt',weights_only=True,map_location='cpu')
    config=config_from_dict(DISCO_DISTILLED_PRESET)
    config.disco.checkpoint_path=str(args.teacher.resolve())
    config.disco.few_step_lora_path=str(args.adapter.resolve())
    backend=DistilledDiscoBackend(config)
    if backend.grid!=GRIDS[4]:
        raise ValueError('This experiment specifically compares four-step weights on four/six-step grids')
    seeds=prior['seeds']
    images={}
    report={'adapter':str(args.adapter),'adapter_sha256':sha256(args.adapter),
        'trained_grid':backend.grid,'experimental_grid':GRIDS[6],'seeds':seeds,
        'scope':'Same four-step-trained weights, original 1000-step linear VP schedule, uniform six-step grid. Not a six-step-trained adapter. No prompts/guidance. Same full-noise seeds, FP32 parameters/head/VP arithmetic and FP16 autocast.',
        'metrics':{}}
    for seed in seeds:
        images[('teacher48',seed)]=cached[f'teacher48_{seed}']
        for label,grid,active in [('base4',GRIDS[4],None),('lora4',GRIDS[4],'student'),
                                 ('base6',GRIDS[6],None),('lora6',GRIDS[6],'student')]:
            noise=torch.randn((1,3,512,512),device=backend.device,dtype=torch.float32,
                generator=torch.Generator(device=backend.device).manual_seed(seed))
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                result=list(rollout(backend.model,backend.schedule,noise,grid,adapter=active))[-1].cpu()
            images[(label,seed)]=result
            tensor_image(result).save(args.output/f'{label}_{seed}.png')
            if label in ('base4','lora4'):
                key=f'{"base4" if label=="base4" else "student1680_4"}_{seed}'
                if not torch.equal(result,cached[key]):
                    raise RuntimeError(f'{label} control differs from cached reference session')
            print(f'{label}: seed {seed}, finite={bool(torch.isfinite(result).all())}',flush=True)
    # Same metric as the prior study; import only the scalar metric from source.
    import torch.nn.functional as F
    def score(result,target):
        clipped=result.clamp(-1,1)
        terms=[F.l1_loss(clipped,target)]
        terms += [F.l1_loss(F.avg_pool2d(clipped,s),F.avg_pool2d(target,s)) for s in (2,4,8)]
        return float(torch.stack(terms).mean()+.1*(result-clipped).abs().mean())
    for seed in seeds:
        reference=images[('teacher48',seed)]
        report['metrics'][str(seed)]={}
        for label in ('base4','lora4','base6','lora6'):
            result=images[(label,seed)]
            metrics=image_metrics(tensor_image(reference),tensor_image(result))
            metrics['paired_multiscale_l1']=score(result,reference)
            report['metrics'][str(seed)][label]=metrics
    report['mean_teacher48_score']={label:sum(report['metrics'][str(seed)][label]['paired_multiscale_l1']
        for seed in seeds)/len(seeds) for label in ('base4','lora4','base6','lora6')}
    torch.save({f'{label}_{seed}':value for (label,seed),value in images.items()},args.output/'raw_tensors.pt')
    (args.output/'results.json').write_text(json.dumps(report,indent=2))
    columns=[('Teacher: 48 steps','teacher48'),('Base: 6 steps','base6'),
             ('1680 LoRA: trained 4-step grid','lora4'),('1680 LoRA: rescaled 6-step grid','lora6')]
    sheet=Image.new('RGB',(2048,560*len(seeds)),'#171b22'); draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    for row,seed in enumerate(seeds):
        for col,(title,label) in enumerate(columns):
            draw.text((col*512+10,row*560+10),f'{title} | seed {seed}',font=font,fill='white')
            sheet.paste(tensor_image(images[(label,seed)]),(col*512,row*560+40))
    sheet.save(args.output/'comparison.png')
    print(json.dumps(report['mean_teacher48_score'],indent=2),flush=True)


if __name__=='__main__':
    main()
