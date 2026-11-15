"""Audit the supplied non-LoRA Disco notebook without executing its setup cells.

Compares actual notebook definitions and its local sampler checkout. CPU checks
use deterministic small tensors; an optional CUDA check uses real CLIP. Neither
is a full video render or a quality benchmark.
"""
import argparse
import ast
from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from vibewarp.config import RunConfig
from vibewarp.core.disco import make_cutouts
from vibewarp.core import disco_secondary, losses
from vibewarp.vendor.guided_diffusion.script_util import create_gaussian_diffusion


def schedule_literal(expression):
    """Read only bounded lists/numbers and list concatenation/repetition."""
    def read(node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):
            return node.value
        if isinstance(node,ast.List): return [read(item) for item in node.elts]
        if isinstance(node,ast.BinOp):
            left,right=read(node.left),read(node.right)
            if isinstance(node.op,ast.Add) and isinstance(left,list) and isinstance(right,list):
                value=left+right
            elif isinstance(node.op,ast.Mult) and isinstance(left,list) and type(right) is int and 0<=right<=10000:
                if len(left)*right>10000: raise ValueError('Schedule too long')
                value=left*right
            else: raise ValueError('Unsupported schedule operation')
            if len(value)>10000: raise ValueError('Schedule too long')
            return value
        raise ValueError('Unsupported schedule expression')
    value=read(ast.parse(expression,mode='eval').body)
    if not isinstance(value,list): raise ValueError('Schedule must be a list')
    return value


def notebook_sources(path):
    notebook=json.loads(path.read_text(encoding='utf-8'))
    return [(index,''.join(cell.get('source',[]))) for index,cell in enumerate(notebook['cells'])
            if cell['cell_type']=='code']


def selected_definitions(sources,names,namespace):
    nodes=[]
    for index,source in sources:
        try: tree=ast.parse(source)
        except SyntaxError: continue  # IPython setup/magic cells are never executed.
        nodes.extend(node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names)
    missing=set(names)-{node.name for node in nodes}
    if missing: raise ValueError(f'Missing notebook definitions: {missing}')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<selected-notebook-definitions>','exec'),namespace)


def saved_literals(sources):
    values={}
    for index,source in sources:
        try: tree=ast.parse(source)
        except SyntaxError: continue
        for node in tree.body:
            if not isinstance(node,ast.Assign) or len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name): continue
            try: value=ast.literal_eval(node.value)
            except (ValueError,TypeError): continue
            values[node.targets[0].id]={'value':value,'cell':index,'line':node.lineno}
    return values


def package(path,name):
    spec=importlib.util.spec_from_file_location(name,path/'__init__.py',submodule_search_locations=[str(path)])
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def metrics(actual,expected):
    difference=(actual.detach().float()-expected.detach().float())
    return {'exact':torch.equal(actual,expected),'max_abs':float(difference.abs().max()),
            'mean_abs':float(difference.abs().mean())}


def compare_samplers(reference):
    class Model(nn.Module):
        def forward(self,x,t,**kwargs):
            epsilon=.1*x.tanh()+t[:,None,None,None]*.00001
            return torch.cat((epsilon,torch.zeros_like(epsilon)),dim=1)
    model=Model()
    rows=[]
    for steps in (30,50):
        options=dict(steps=(1000//steps)*steps,learn_sigma=True,rescale_timesteps=True,timestep_respacing=f'ddim{steps}')
        ours,old=create_gaussian_diffusion(**options),reference.create_gaussian_diffusion(**options)
        for sampler,eta in (('ddim',0.),('ddim',.25),('plms',0.)):
            for skipped in (10,math.floor(steps*.7)):
                for masked in (False,True):
                    inputs=torch.randn(2,1,3,8,8,generator=torch.Generator().manual_seed(72))
                    mask=np.ones((8,8,3),dtype=np.float32);mask[:4]=0
                    kwargs=dict(noise=inputs[0],device='cpu',progress=False,clip_denoised=False,
                        cond_fn=lambda x,t,**kw:x*.001,model_kwargs={},skip_timesteps=skipped,
                        init_image=inputs[1],mask=mask if masked else None,inpainting_stop=6)
                    if sampler=='ddim': kwargs['eta']=eta
                    outputs=[]
                    for diffusion in (old,ours):
                        torch.manual_seed(100500)
                        outputs.append(list(getattr(diffusion,f'{sampler}_sample_loop_progressive')(model,(1,3,8,8),**kwargs)))
                    checks=[metrics(a[key],b[key]) for a,b in zip(outputs[1],outputs[0]) for key in ('sample','pred_xstart')]
                    rows.append(dict(steps=steps,skipped=skipped,executed=len(outputs[0]),sampler=sampler,eta=eta,masked=masked,
                        same_timesteps=ours.timestep_map==old.timestep_map,
                        same_betas=np.array_equal(ours.betas,old.betas),same_count=len(outputs[0])==len(outputs[1]),
                        exact=all(check['exact'] for check in checks),max_abs=max(check['max_abs'] for check in checks)))
    return rows


def compare_losses(namespace):
    rows={}
    image=torch.randn(1,3,12,16,generator=torch.Generator().manual_seed(72),requires_grad=True)
    for name in ('spherical_dist_loss','tv_loss','range_loss'):
        arguments=(image,image.detach().flip(-1)) if name=='spherical_dist_loss' else (image,)
        expected=namespace[name](*arguments)
        actual=getattr(losses,name)(*arguments)
        gradient_a=torch.autograd.grad(actual.sum(),image)[0]
        gradient_b=torch.autograd.grad(expected.sum(),image)[0]
        rows[name]={'output':metrics(actual,expected),'gradient':metrics(gradient_a,gradient_b)}
    return rows


def compare_cutouts(namespace):
    rows=[]
    cases=((64,96,4,0,False,1.,.2),(64,96,16,0,False,1.,.2),
           (64,96,4,4,False,1.,.2),(64,96,4,4,True,1.,.2),
           (65,97,1,3,False,.5,0.),(97,65,2,5,True,1.3,1.),
           (64,64,3,8,True,1.,.2),(64,96,0,4,True,.5,0.))
    for height,width,overview,inner,augment,power,gray in cases:
        image=torch.rand(1,3,height,width,generator=torch.Generator().manual_seed(72),requires_grad=True)
        namespace['skip_augs']=not augment
        torch.manual_seed(100500)
        expected=namespace['MakeCutoutsDango'](32,Overview=overview,InnerCrop=inner,IC_Size_Pow=power,IC_Grey_P=gray)(image)
        reference_rng=torch.get_rng_state().clone()
        torch.manual_seed(100500)
        actual=make_cutouts(image,32,overview,inner,power,gray,augment=augment)
        rng_equal=torch.equal(reference_rng,torch.get_rng_state())
        grad_a=torch.autograd.grad(actual.square().mean(),image)[0]
        grad_b=torch.autograd.grad(expected.square().mean(),image)[0]
        rows.append(dict(height=height,width=width,overview=overview,inner=inner,augment=augment,
            power=power,gray_probability=gray,cpu_rng_equal=rng_equal,
            output=metrics(actual,expected),gradient=metrics(grad_a,grad_b),
            gradient_cosine=float(F.cosine_similarity(grad_a.flatten(),grad_b.flatten(),dim=0))))
    return rows


def compare_clip_cutouts(namespace,root):
    """CUDA comparison with one shared real encoder, isolating crop behavior."""
    from vibewarp.core.disco import GuidanceEncoder
    encoder=GuidanceEncoder('ViT-B-32',root,'cuda')
    target=encoder.text(['a detailed oil painting'])
    rows=[]
    for augment in (False,True):
        image=torch.rand(1,3,128,192,device='cuda',
            generator=torch.Generator(device='cuda').manual_seed(72),requires_grad=True)
        namespace['skip_augs']=not augment
        outputs=[]
        for reference in (True,False):
            torch.manual_seed(100500)
            cuts=(namespace['MakeCutoutsDango'](224,Overview=4,InnerCrop=4,IC_Size_Pow=1.,IC_Grey_P=.2)(image)
                  if reference else make_cutouts(image,224,4,4,1.,.2,augment=augment))
            embedding=encoder.image(cuts)
            loss=losses.spherical_dist_loss(embedding,target).mean()
            gradient=torch.autograd.grad(loss,image)[0]
            if not torch.isfinite(gradient).all(): raise FloatingPointError('Nonfinite CLIP gradient')
            outputs.append((cuts.detach(),embedding.detach(),gradient.detach()))
        old,new=outputs
        rows.append(dict(augment=augment,cutouts=metrics(new[0],old[0]),embedding=metrics(new[1],old[1]),
            clip_gradient=metrics(new[2],old[2]),
            gradient_cosine=float(F.cosine_similarity(new[2].flatten(),old[2].flatten(),dim=0))))
    return rows


def compare_secondary(sources,checkpoint=None):
    name='_disco_notebook_secondary_audit'
    module=ModuleType(name);sys.modules[name]=module
    module.__dict__.update(torch=torch,nn=nn,F=F,math=math,partial=partial,dataclass=dataclass)
    selected_definitions(sources,('append_dims','expand_to_planes','alpha_sigma_to_t','t_to_alpha_sigma',
        'DiffusionOutput','ConvBlock','SkipBlock','FourierFeatures','SecondaryDiffusionImageNet2'),module.__dict__)
    torch.manual_seed(72)
    expected=module.SecondaryDiffusionImageNet2().eval().requires_grad_(False)
    actual=disco_secondary.SecondaryDiffusionImageNet2().eval().requires_grad_(False)
    if checkpoint is not None:
        expected.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    actual.load_state_dict(expected.state_dict(),strict=True)
    x=torch.randn(1,3,64,64,requires_grad=True)
    t=torch.tensor([.4])
    a,b=actual(x,t).pred,expected(x,t).pred
    return {'strict_state_dict':True,
            'weights':'identical initialized weights' if checkpoint is None else str(checkpoint.resolve()),
            'weights_sha256':None if checkpoint is None else hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'output':metrics(a,b),
            'gradient':metrics(torch.autograd.grad(a.mean(),x)[0],torch.autograd.grad(b.mean(),x)[0])}


def compare_range_wiring():
    """The notebook differentiates its combined losses through x_in first."""
    rows=[]
    for scale in (0.,150.):
        x=torch.linspace(-5,5,48).reshape(1,3,4,4).requires_grad_()
        out=x*.5
        x_in=out*.7+x*.3
        # Zero TV preserves the notebook's x_in branch even at disabled scales.
        combined=losses.tv_loss(x_in).sum()*0+losses.range_loss(out).sum()*scale
        image_grad=torch.autograd.grad(combined,x_in,retain_graph=True)[0]
        old=-torch.autograd.grad(x_in,x,image_grad,retain_graph=True)[0]
        actual=old-torch.autograd.grad(losses.range_loss(out).sum()*scale,x)[0]
        rows.append(dict(range_scale=scale,gradient=metrics(actual,old),
                         notebook_rms=float(old.square().mean().sqrt()),
                         application_rms=float(actual.square().mean().sqrt())))
    return rows


def audit_import(saved,output):
    from vibewarp.settings import load_warpfusion_settings
    names=('diffusion_model','use_secondary_model','diffusion_sampling_mode','steps','clip_guidance_scale',
           'cutn_batches','frames_scale','frames_skip_steps','skip_steps','init_scale','eta','clamp_max',
           'flow_blend','turbo_mode','turbo_steps','turbo_preroll','ViTB32','ViTB16','RN50')
    raw={name:saved[name]['value'] for name in names if name in saved}
    raw['width'],raw['height']=saved['width_height']['value']
    for name in ('cut_overview','cut_innercut','cut_icgray_p'):
        raw[name]=str(schedule_literal(saved[name]['value']))
    path=output/'notebook-settings-subset.json'
    path.write_text(json.dumps(raw,indent=2),encoding='utf-8')
    imported=load_warpfusion_settings(str(path))
    return {'scope':'Representative saved notebook settings subset, not a captured notebook settings export',
            'model_version':imported['model_version'],'disco_section_present':'disco' in imported,
            'diffusion_steps':imported['diffusion']['steps'],'path':str(path)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--secondary-checkpoint',type=Path)
    parser.add_argument('--clip-root',type=Path,help='Optional CUDA CLIP gradient check; directory containing clip/ViT-B-32.pt')
    parser.add_argument('--output',type=Path,default=Path('.cache/disco-notebook-parity-072'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    sources=notebook_sources(args.notebook)
    saved=saved_literals(sources)
    fork=args.source_root/'guided-diffusion-sxela'
    revision=subprocess.check_output(['git','-C',str(fork),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(fork),'status','--porcelain'],text=True).strip()
    if dirty: raise ValueError('Reference sampler checkout has local modifications; audit a clean reference')
    # Import only isolated, local definition modules. Never run notebook setup,
    # downloads, shell cells, training code or its global rendering loop.
    sys.dont_write_bytecode=True
    package(fork/'guided_diffusion','_disco_notebook_gd_audit')
    reference=importlib.import_module('_disco_notebook_gd_audit.script_util')
    sys.path.insert(0,str(args.source_root/'ResizeRight'))
    resize=importlib.import_module('resize_right').resize
    namespace=dict(torch=torch,nn=nn,F=F,T=T,TF=TF,resize=resize,
        args=SimpleNamespace(animation_mode='Video Input'),skip_augs=True,cutout_debug=False,padargs={})
    selected_definitions(sources,('MakeCutoutsDango','spherical_dist_loss','tv_loss','range_loss'),namespace)
    report={'notebook':str(args.notebook.resolve()),'notebook_sha256':hashlib.sha256(args.notebook.read_bytes()).hexdigest(),
        'reference_revision':revision,'scope':'CPU component/settings audit, with optional CUDA CLIP cutout checks. No LoRA, full-video or real-primary-checkpoint pixel parity claim.',
        'application_source_sha256':{name:hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
            for name in ('vibewarp/core/disco.py','vibewarp/core/disco_secondary.py','vibewarp/config.py',
                         'vibewarp/settings.py','vibewarp/vendor/guided_diffusion/gaussian_diffusion.py')},
        'saved_settings':{key:saved[key] for key in ('batch_name','steps','width_height','clip_guidance_scale','cutn_batches',
            'use_secondary_model','diffusion_sampling_mode','init_scale','frames_scale','frames_skip_steps','skip_steps',
            'eta','clamp_max','range_scale','flow_blend','cut_overview','cut_innercut','cut_icgray_p','turbo_mode','turbo_steps',
            'turbo_preroll','match_color_strength','inpainting_early_stop') if key in saved},
        'application_defaults':{'disco':asdict(RunConfig().disco),'warp':asdict(RunConfig().warp)},
        'losses':compare_losses(namespace),'samplers':compare_samplers(reference),
        'range_wiring':compare_range_wiring(),'settings_import':audit_import(saved,args.output),
        'secondary':compare_secondary(sources,args.secondary_checkpoint),'cutouts':compare_cutouts(namespace),
        'cuda_clip_cutouts':compare_clip_cutouts(namespace,args.clip_root) if args.clip_root else None}
    (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    passed=sum(row['exact'] and row['same_betas'] and row['same_timesteps'] and row['same_count'] for row in report['samplers'])
    print(json.dumps({'sampler_cases':len(report['samplers']),'sampler_exact':passed,
        'losses':report['losses'],'secondary':report['secondary'],'cutouts':report['cutouts'],
        'range_wiring':report['range_wiring'],'settings_import':report['settings_import'],
        'cuda_clip_cutouts':report['cuda_clip_cutouts'],
        'report':str(args.output/'results.json')},indent=2))


if __name__=='__main__':
    main()
