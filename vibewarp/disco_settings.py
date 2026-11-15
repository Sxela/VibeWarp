"""Import pixel-Disco notebook exports without importing Stable Diffusion defaults."""
import ast
import os
import warnings


def cut_schedule(value, default):
    if value is None:
        return default
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {0: value}
    if len(value) > 16000:
        raise ValueError('Cutout schedule expression is too large')
    def interpret(node):
        if isinstance(node, ast.Expression): return interpret(node.body)
        if isinstance(node, (ast.List, ast.Tuple)): return [interpret(x) for x in node.elts]
        if isinstance(node, ast.Constant) and isinstance(node.value, (int,float)): return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add,ast.Mult)):
            a,b=interpret(node.left),interpret(node.right)
            if isinstance(node.op,ast.Add) and isinstance(a,list) and isinstance(b,list) and len(a)+len(b)<=1000: return a+b
            if isinstance(node.op,ast.Mult) and isinstance(a,list) and type(b) is int and 0<=b<=1000 and len(a)*b<=1000: return a*b
        raise ValueError('Cutout schedule supports only numeric lists, repetition and concatenation')
    values=interpret(ast.parse(value,mode='eval'))
    if not isinstance(values,list) or len(values)!=1000:
        raise ValueError('Notebook cutout schedule must have 1000 entries')
    return {i: v for i,v in enumerate(values) if i==0 or v!=values[i-1]}


def import_disco_notebook(raw, models_root=None):
    from vibewarp.frame_range import imported_frame_range
    root=models_root or 'models'
    stack=raw.get('lora_config') or {}
    def setting(name, default=None, *aliases):
        # Native/lowercase fields take precedence, including explicit false/zero.
        for key in (name, *aliases):
            if key in raw and raw[key] is not None:
                return raw[key]
        return default
    steps=int(raw.get('steps',50))
    turbo=setting('lora_path',stack.get('turbo_path',''),'LORA_PATH')
    quality=setting('quality_lora_path',stack.get('quality_path',''),'QUALITY_LORA_PATH')
    if not setting('use_turbo_lora',bool(turbo),'USE_TURBO_LORA'): turbo=''
    if not setting('use_quality_lora',bool(quality),'USE_QUALITY_LORA'): quality=''
    name=raw.get('diffusion_model','512x512_diffusion_uncond_finetune_008100')
    base=setting('disco_checkpoint_path',None,'BASE_MODEL_PATH') or os.path.join(root,'disco',name+'.pt')
    padded=bool(setting('pad_to_multiple_64',False,'PAD_TO_MULTIPLE_64'))
    visible=raw.get('_visible_size') if padded else None
    if not visible: visible=[0,0]
    skip=int(raw.get('skip_steps',0))
    frame_skip=raw.get('frames_skip_steps',skip)
    if isinstance(frame_skip,str) and frame_skip.endswith('%'):
        frame_skip=int(steps*float(frame_skip[:-1])/100)
    frame_skip=int(frame_skip)
    turbo_skip=raw.get('turbo_frame_skips_steps')
    if isinstance(turbo_skip,str):
        turbo_skip=float(turbo_skip.split('%')[0])/100 if '%' in turbo_skip else float(turbo_skip)
    turbo_strength=0.0 if turbo_skip is None else 1-min(1,max(0,float(turbo_skip)))
    stop=raw.get('inpainting_stop',1)
    if not isinstance(stop,int):
        warnings.warn('Notebook exported an unresolved inpainting_stop; using 1. Save effective settings on the next export.',RuntimeWarning)
        stop=1
    def prompts(value):
        return {int(k): '\n'.join(v) if isinstance(v,list) else str(v) for k,v in value.items()}
    encoders=[name for flag,name in [('ViTB32','ViT-B-32'),('ViTB16','ViT-B-16'),('RN50','RN50'),('RN101','RN101'),('ViTL14','ViT-L-14')] if raw.get(flag,False)]
    d=dict(checkpoint_path=base,sampler='distilled' if turbo else raw.get('diffusion_sampling_mode','ddim'),
        distilled_guided=True,few_step_lora_path=turbo,lora_strength=float(setting('lora_strength',stack.get('turbo_strength',1),'LORA_STRENGTH')),
        quality_lora_enabled=bool(quality),quality_lora_path=quality,quality_lora_strength=float(setting('quality_lora_strength',stack.get('quality_strength',1),'QUALITY_LORA_STRENGTH')),
        steps=steps,timestep_grid=raw.get('timestep_grid') or raw.get('_effective_grid') or stack.get('grid',[]),
        secondary_precision='auto',
        clip_precision='fp16',
        cut_batch_group=int(setting('cut_batch_group',1,'SPEED_CUT_BATCH_GROUP')),
        pad_to_multiple_64=padded,
        style_strength=1-skip/steps,style_strength_schedule={0:1-skip/steps,1:1-frame_skip/steps},
        init_scale=float(raw.get('init_scale',0)),init_scale_schedule={0:float(raw.get('init_scale',0)),1:float(raw.get('frames_scale',0))},
        eta=float(raw.get('eta',0)),use_secondary_model=bool(raw.get('use_secondary_model',True)),
        secondary_path=os.path.join(root,'disco','secondary_model_imagenet_2.pth'),encoders=encoders,
        clip_guidance_scale=float(raw.get('clip_guidance_scale',35000)),cutn_batches=int(raw.get('cutn_batches',4)),
        clip_grad_steps=int(setting('clip_grad_steps',1,'CLIP_GRAD_STEPS')),clip_divide_cuts=bool(setting('clip_divide_cuts',False,'CLIP_DIVIDE_CUTS')),
        notebook_rng=True,cut_schedule_version=1,cut_schedule_original_time=True,cut_overview=cut_schedule(raw.get('cut_overview'),{0:4}),
        cut_innercut=cut_schedule(raw.get('cut_innercut'),{0:12}),cut_icgray_schedule=cut_schedule(raw.get('cut_icgray_p'),{0:0}),
        cut_ic_pow=float(raw.get('cut_ic_pow',1)),skip_augs=bool(raw.get('skip_augs',False)),
        tv_scale=float(raw.get('tv_scale',0)),range_scale=float(raw.get('range_scale',0)),sat_scale=float(raw.get('sat_scale',0)),
        mask_frame_guidance=bool(setting('frame_loss_cc_trusted_only',False,'FRAME_LOSS_CC_TRUSTED_ONLY')),clamp_max=float(raw.get('clamp_max',.05)),
        inpaint=bool(raw.get('do_diffuse_inpaint',False)),inpainting_stop=stop,
        inpaint_mask_blur=int(raw.get('diffuse_inpaint_mask_blur',15)),inpaint_mask_sigma=float(raw.get('inpaint_mask_sigma',4)),
        inpaint_mask_threshold=float(raw.get('diffuse_inpaint_mask_thresh',220))/255,
        guidance_noise_cap_percent=float(setting('lora_guidance_cap_percent',10,'GUIDANCE_CAP_PERCENT')) if setting('lora_noise_guidance_cap',False,'GUIDANCE_NOISE_CAP') else 0,
        merge_loras=bool(setting('merge_loras',False,'SPEED_MERGE_LORAS')),image_prompts=prompts(raw.get('image_prompts',{})))
    # Never interpret training config fields (steps, init_scale) as render values.
    return dict(model_version='disco_512' if '512' in name else 'disco_256', model_path=root,sd_checkpoint_path=base,
        batch_name=raw.get('batch_name','disco-notebook'),disco=d,
        text_prompts=prompts(raw.get('text_prompts',{})),negative_prompts=prompts(raw.get('negative_prompts',{})),
        frame_range=imported_frame_range(raw.get('frame_range'), int(raw.get('max_frames',0))),
        diffusion=dict(seed=int(raw.get('seed',-1)),guidance_mode='prev warped + cc',
            temporal_guidance_cc_trusted_only=bool(raw.get('temporal_guidance_cc_trusted_only',False))),
        video=dict(video_init_path=raw.get('video_init_path',''),width=int(raw.get('width',512)),height=int(raw.get('height',512)),
            visible_width=int(visible[0]),visible_height=int(visible[1]),
            extract_nth_frame=int(raw.get('extract_nth_frame',1)),first_frame_path=setting('first_frame_path','','FIRST_FRAME_PATH')),
        flow=dict(flow_warp=bool(raw.get('flow_warp',True)),check_consistency=bool(raw.get('check_consistency',True)),
            consistency_blur=int(raw.get('consistency_blur',2)),soften_consistency_mask=float(raw.get('forward_weights_clip',0))),
        warp=dict(flow_blend=float(raw.get('flow_blend',.999)),padding_ratio=float(raw.get('padding_ratio',.2)),padding_mode=raw.get('padding_mode','reflect')),
        turbo=dict(enabled=bool(raw.get('turbo_mode',False)),interval=int(raw.get('turbo_steps',3)),preroll=int(raw.get('turbo_preroll',10)),
            intermediate_strength=turbo_strength,
            consistency_soften=float(setting('forward_weights_clip_turbo_step',0,'soften_consistency_mask_for_turbo_frames')),
            disable_cc=bool(raw.get('disable_cc_for_turbo_frames',False)),scene_cut_threshold=0),
        brightness=dict(enable=bool(raw.get('enable_adjust_brightness',False)),**{k:raw[k] for k in ['high_brightness_threshold','high_brightness_adjust_ratio','low_brightness_threshold','low_brightness_adjust_ratio','high_brightness_adjust_fix_amount','low_brightness_adjust_fix_amount','max_brightness_threshold','min_brightness_threshold'] if k in raw}),
        color=dict(before_enabled=float(raw.get('match_color_strength',0))>0,before_strength=float(raw.get('match_color_strength',0)),after_enabled=False),
        video_assembly=dict(fps=float(raw.get('video_output_fps',0)),
            crop_to_visible=bool(setting('crop_to_visible',False,'EXPORT_CROP_TO_VISIBLE')),
            keep_audio=bool(setting('keep_audio',True,'EXPORT_KEEP_AUDIO')),blend_mode='None'))
