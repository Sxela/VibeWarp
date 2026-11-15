"""Compare few-step adapters to their fine-step teacher, at identical noise."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from safetensors.torch import load_file

from benchmark_disco_attention import image_metrics
from vibewarp.core.disco import tensor_image
from vibewarp.core.disco_lora import adapter_config, resolve_adapter, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--teacher',type=Path,required=True)
    parser.add_argument('--earlier-adapter',type=Path,required=True)
    parser.add_argument('--adapter',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[12345,12346])
    parser.add_argument('--output',type=Path,default=Path('.cache/disco-lora-teacher-comparison'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    current,earlier=resolve_adapter(args.adapter),resolve_adapter(args.earlier_adapter)
    cfg,old=adapter_config(current),adapter_config(earlier)
    assert cfg['grid']==old['grid'] and cfg['rank']==old['rank'] and cfg['alpha']==old['alpha']
    used=set(range(cfg['train_seed'],cfg['train_seed']+cfg['train_count']))|set(range(cfg['holdout_seed'],cfg['holdout_seed']+cfg['holdout_count']))
    if used.intersection(args.seeds):
        raise ValueError('Comparison seeds must be outside training and validation seeds')
    sys.path.insert(0,str(args.source_root.resolve()))
    from tdm.core import load_model,inject_adapters,load_adapter_state,rollout,timestep_grid
    from tdm.evaluation import autocast_context,paired_score
    model,schedule=load_model(args.teacher,'cuda')
    fine=[]
    for start,end in zip(cfg['grid'][:-1],cfg['grid'][1:]):
        fine.extend(int(t) for t in np.linspace(start,end,cfg['substeps']+1)[:-1])
    fine.append(-1)
    report={'seeds':args.seeds,'teacher_grid':fine,'base_50_grid':timestep_grid(50),
        'adapter_grid':cfg['grid'],'adapter_sha256':sha256(current),'earlier_adapter_sha256':sha256(earlier),
        'teacher_substeps_per_interval':cfg['substeps'],'torch':torch.__version__,
        'device':torch.cuda.get_device_name(),'metrics':{},
        'scope':'Unconditional 512px full-noise inference, identical seed/noise, original linear VP schedule, FP32 parameters/head/arithmetic, FP16 autocast; no intermediate clipping. Two fresh seeds are illustrative, not a distributional quality study.'}
    images={}
    def run(label,grid,active):
        for seed in args.seeds:
            noise=torch.randn((1,3,512,512),device='cuda',dtype=torch.float32,
                generator=torch.Generator(device='cuda').manual_seed(seed))
            with torch.inference_mode(),autocast_context('fp16','cuda'):
                result,_=rollout(model,schedule,noise,grid,adapter=active)
            if not torch.isfinite(result).all():
                raise FloatingPointError(label)
            images[(label,seed)]=result.cpu()
            tensor_image(result).save(args.output/f'{label}_{seed}.png')
            print(f'{label}: seed {seed}, {len(grid)-1} model evaluations',flush=True)
    run('teacher48',fine,None)
    run('base50',timestep_grid(50),None)
    run('base4',cfg['grid'],None)
    inject_adapters(model,cfg['rank'],cfg['alpha'],names=('student',))
    model.to('cuda').eval().requires_grad_(False)
    for label,path in [('earlier_ema4',earlier),('student1680_4',current)]:
        load_adapter_state(model,load_file(str(path)),'student')
        run(label,cfg['grid'],'student')
    for seed in args.seeds:
        report['metrics'][str(seed)]={}
        for target in ('teacher48','base50'):
            reference=images[(target,seed)]
            report['metrics'][str(seed)][target]={}
            for label in ('base4','earlier_ema4','student1680_4'):
                result=images[(label,seed)]
                metrics=image_metrics(tensor_image(reference),tensor_image(result))
                # Supervised evaluation compares against the unclipped teacher
                # endpoint, while paired_score clips/penalizes the student.
                metrics['paired_multiscale_l1']=float(paired_score(result,reference))
                metrics['raw_mae']=float((result-reference).abs().mean())
                report['metrics'][str(seed)][target][label]=metrics
    report['mean_teacher48_score']={label:float(np.mean([
        report['metrics'][str(seed)]['teacher48'][label]['paired_multiscale_l1'] for seed in args.seeds]))
        for label in ('base4','earlier_ema4','student1680_4')}
    torch.save({f'{label}_{seed}':value for (label,seed),value in images.items()},args.output/'raw_tensors.pt')
    (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    columns=[('Teacher, training grid: 48 steps','teacher48'),('Base: 50 steps','base50'),
             ('Earlier EMA: 4 steps','earlier_ema4'),('Raw update 1680: 4 steps','student1680_4')]
    sheet=Image.new('RGB',(512*len(columns),560*len(args.seeds)),'#171b22')
    draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    for row,seed in enumerate(args.seeds):
        for col,(title,label) in enumerate(columns):
            draw.text((col*512+10,row*560+10),f'{title} | seed {seed}',font=font,fill='white')
            sheet.paste(tensor_image(images[(label,seed)]),(col*512,row*560+40))
    sheet.save(args.output/'comparison.png')
    print(json.dumps(report['mean_teacher48_score'],indent=2),flush=True)


if __name__=='__main__':
    main()
