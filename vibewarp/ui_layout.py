"""Which UI tab each config field belongs to.

The nav used to be generated straight from the config dataclasses — one tab per section,
17 tabs, ~180 fields, ordered by how the code happens to be structured rather than by how
the tool is used. The grouping we actually want cuts ACROSS the dataclasses: "model paths"
and "VRAM tuning" live in six different sections but belong on one screen.

So the tab layout is declared here, in the backend, and `config_io._dataclass_schema`
stamps `tier`/`group` onto every property in the schema. The UI renders whatever the
schema says and holds no field list of its own — a mapping table in JS would drift
silently the moment config.py changed.

TIERS
  project   per-project input/output. Set for each new video.
  render    the 80%. Touched every run.
  advanced  set once, or never.
  system    machine-level. Model paths, VRAM, threads. These do NOT travel with a settings
            file — importing someone else's settings would otherwise clobber your local
            model paths. They persist to `system.json` next to the install and are
            re-applied after every import; see vibewarp/system_settings.py.
  hidden    exists in the config (settings import, back-compat) but is not editable.

Every field must be classified. `test_ui_layout.py` fails if one is not, so a new config
field is invisible until someone decides where it goes — which is the failure mode we
want, rather than it silently landing in a default bucket.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Dict, Tuple

# "section.field" -> (tier, group). Section is the RunConfig attribute name; top-level
# RunConfig fields use the section "main".
LAYOUT: Dict[str, Tuple[str, str]] = {}

# Model-family compatibility is part of the schema contract, alongside layout. Fields
# omitted here work for every renderer; incompatible fields remain in settings files but
# disappear from the form when another model family is selected.
MODEL_FAMILY_BY_VERSION = {
    'control_multi_v15': 'sd',
    'control_multi_sdxl': 'sd',
    'flux2_klein_edit': 'flux',
    'flux2_klein_9b_edit': 'flux',
    'hidream_o1_edit': 'hidream',
    'qwen_image_edit_2511': 'qwen',
    'qwen_image_edit_2511_gguf': 'qwen',
    'mage_flow_edit': 'mage',
    'mage_flow_edit_turbo': 'mage',
    'disco_512': 'disco',
    'disco_256': 'disco',
}
ALL_MODEL_FAMILIES = ('sd', 'flux', 'hidream', 'qwen', 'mage', 'disco')

# Workspace navigation is independent of storage tiers. In particular, a model's
# server/files can appear during setup while remaining machine-local on export.
WORKFLOW_GROUPS = {
    'source': ('Input Video', 'Output', 'Video Input'),
    'model': ('Model',),
    'prompts': ('Prompts', 'Diffusion', 'Seed', 'Flux Edit', 'HiDream-O1 Edit',
                'Qwen Image Edit', 'Mage-Flow Edit', 'Disco Diffusion', 'Disco inpainting masks', 'Turbo'),
    'clip': ('General', 'CLIP guidance', 'Cutouts', 'Advanced guidance'),
    'conditioning': ('ControlNet', 'IP-Adapter', 'Reference Images', 'Temporal Contact Sheet'),
    'background': ('Background masking', 'RVM files'),
    'color': ('Raw input color match', 'Color Match — Before Diffusion', 'Color Match — After Diffusion'),
    'consistency': ('Motion', 'Flow & Consistency', 'Warp', 'AnimateDiff',
                    'Reconstruction Noise', 'Scene Scheduling'),
    'export': ('Video Output',),
    'all': (),
    'system': (),
}
WORKFLOW_LABELS = {
    'source': 'Source & output', 'model': 'Model', 'prompts': 'Prompts & sampling',
    'clip': 'Guidance',
    'background': 'Mask & tint',
    'color': 'Color match',
    'conditioning': 'Conditioning', 'consistency': 'Frame consistency',
    'export': 'Video export', 'all': 'All settings', 'system': 'System settings',
}
MODEL_LABELS = {
    'control_multi_v15': 'Stable Diffusion 1.5',
    'control_multi_sdxl': 'Stable Diffusion XL',
    'flux2_klein_edit': 'FLUX.2 Klein · 4B',
    'flux2_klein_9b_edit': 'FLUX.2 Klein · 9B',
    'hidream_o1_edit': 'HiDream-O1',
    'qwen_image_edit_2511': 'Qwen Image Edit · INT8',
    'qwen_image_edit_2511_gguf': 'Qwen Image Edit · GGUF',
    'mage_flow_edit': 'Mage-Flow Edit',
    'mage_flow_edit_turbo': 'Mage-Flow Edit · Turbo',
    'disco_512': 'Disco Diffusion · 512',
}
MODEL_GROUP_LABELS = {'native': 'Native models', 'comfy': 'ComfyUI-managed models'}


def model_runtime_group(version: str) -> str:
    # Group the edit models by their default backend; FLUX retains its optional
    # Diffusers backend in model settings.
    return 'comfy' if MODEL_FAMILY_BY_VERSION[version] in ('flux', 'hidream', 'qwen', 'mage') else 'native'


GROUP_EDITORS = {
    'Input Video': 'video', 'ControlNet': 'controlnet',
    'IP-Adapter': 'ipadapter', 'Reference Images': 'references',
}

# Keep the shared mask next to its two losses without tying frontend ordering
# to config dataclass declaration order.
GROUP_FIELD_ORDER = {
    'Background masking': ('mask.rvm_enabled', 'mask.foreground_color', 'mask.foreground_opacity',
                          'mask.background_color', 'mask.background_opacity',
                          'mask.use_background_mask', 'mask.mask_source', 'mask.mask_path',
                          'mask.render_mask_stage',
                          'mask.rvm_downsample_ratio'),
    'General': ('disco.init_scale_schedule', 'disco.image_scale_schedule',
                'disco.mask_frame_guidance', 'disco.use_secondary_model',
                'diffusion.temporal_guidance_cc_trusted_only'),
    'CLIP guidance': ('disco.encoders', 'disco.clip_guidance_scale', 'disco.image_prompts'),
}

FIELD_HELP = {
    'mask.rvm_enabled': 'Tint foreground and background in the raw render input using RVM alpha. Either Apply masked tint or Use Background Mask generates the shared masks on the GPU.',
    'mask.use_background_mask': 'Composite the render image onto the configured background using the mask from Mask source. This does not mask the warp step.',
    'mask.mask_source': 'Where background masks come from, for render and video-assembly masking. RVM matting extracts them from the input video on the GPU. Mask video and Image folder use your own mattes (light = foreground), one per input video frame; Extract nth frame applies to them too. Masked tint always uses RVM.',
    'mask.mask_path': 'Mask video file, or folder of mask images sorted by name. Used when Mask source is not RVM matting.',
    'mask.render_mask_stage': 'Before diffusion replaces the background of the model input, allowing diffusion to style the fresh background. After diffusion composites the final output. The mask is applied only at the selected stage.',
}

RVM_FILE_FIELDS = ('mask.rvm_model_path', 'mask.rvm_cache_dir')
SCHEDULE_FALLBACKS = {
    'flow.missed_consistency_schedule': 'missed_consistency_weight',
    'flow.overshoot_consistency_schedule': 'overshoot_consistency_weight',
    'flow.edges_consistency_schedule': 'edges_consistency_weight',
}
FIELD_HELP.update({
    'disco.mask_frame_guidance': 'Restrict LPIPS/image-space frame guidance to the optical-flow consistency mask. This uses consistency, not the RVM foreground alpha.',
    'diffusion.temporal_guidance_cc_trusted_only': 'Restrict temporal gradient guidance to consistent warped areas. Independent of RVM foreground/background masks.',
    'ipadapter.flip_uc': 'Flip the unconditional attention mask used by IP-Adapter. This is an adapter attention option, not an RVM image mask.',
    'video_assembly.use_background_mask_video': 'Composite during video assembly using the run masks from Mask source (Background masking). Independent of the render-mask switch and its before/after-diffusion stage; masks are generated when either is on.',
})


def workspace_metadata(section: str, name: str) -> dict:
    placed = classify(section, name)
    if not placed or placed[0] == 'hidden':
        return {}
    tier, group = placed
    if tier == 'system':
        workflows = ['system']
        if group == 'Paths' and (section in ('flux', 'hidream', 'qwen', 'mage')
                                 or (section == 'main' and name == 'comfy_models_dir')):
            workflows.insert(0, 'model')
    else:
        workflows = [next((key for key, groups in WORKFLOW_GROUPS.items()
                           if group in groups), 'all')]
    result = {'workflows': workflows,
              'workspace_advanced': tier == 'advanced' and group not in ('Motion', 'Advanced guidance')}
    if group in GROUP_EDITORS:
        result['editor'] = GROUP_EDITORS[group]
    order = GROUP_FIELD_ORDER.get(group, ())
    key = f'{section}.{name}'
    if key in order:
        result['workspace_order'] = order.index(key)
    if key in FIELD_HELP:
        result['help'] = FIELD_HELP[key]
    if key in RVM_FILE_FIELDS:
        result['workflows'] = ['background', 'system']
        result['workflow_groups'] = {'background': 'RVM files'}
        result['workflow_editors'] = {'background': None}
    if key in SCHEDULE_FALLBACKS:
        result['schedule_fallback'] = SCHEDULE_FALLBACKS[key]
    if group in ('Background masking', 'Raw input color match'):
        result['preview'] = 'raw_video'
    return result


SECTION_MODEL_FAMILIES = {
    'controlnet': ('sd',),
    'freeu': ('sd',),
    'animatediff': ('sd',),
    'reconstruction_noise': ('sd',),
    'ipadapter': ('sd',),
    'vae': ('sd',),
    'diffusion': ('sd',),
    'flux': ('flux',),
    'hidream': ('hidream',),
    'qwen': ('qwen',),
    'mage': ('mage',),
    'disco': ('disco',),
    'contact_sheet': ('flux', 'hidream', 'qwen', 'mage'),
}
FIELD_MODEL_FAMILIES = {
    # External edit renderers still use this seed as their base starting noise.
    'diffusion.seed': ALL_MODEL_FAMILIES,
    'diffusion.fixed_seed_randomness': ('sd', 'disco'),
    'diffusion.fixed_seed_wiggle': ('sd', 'disco'),
    'diffusion.fixed_seed_wiggle_direction': ('sd', 'disco'),
    'diffusion.guidance_mode': ('sd', 'disco'),
    **{f'diffusion.{name}': ('sd', 'disco') for name in (
        'tiled_sampler', 'sampler_tile_size', 'sampler_tile_overlap',
        'sampler_tile_split_threshold', 'sampler_scale_schedule', 'sampler_scale_min_size')},
    'main.reference_label_opacity': ('flux', 'hidream', 'qwen', 'mage'),
    # FLUX.2 uses positive instructions with zeroed negative conditioning.
    'main.negative_prompts': ('sd', 'hidream', 'qwen', 'mage', 'disco'),
    # These paths and precision settings belong to the local SD stack.
    'main.sd_checkpoint_path': ('sd',),
    'main.model_path': ('sd', 'disco'),
    'main.comfy_models_dir': ('flux', 'hidream', 'qwen', 'mage'),
    'main.lora_dir': ('sd',),
    'main.lora_merge_precision': ('sd',),
    'video.visible_width': ('disco',),
    'video.visible_height': ('disco',),
    'video_assembly.crop_to_visible': ('disco',),
    # Scene analysis and flow/consistency scheduling are shared. These templates
    # specifically mutate SD conditioning/sampling fields and do nothing for edit models.
    'scene.steps_template': ('sd',),
    'scene.style_strength_template': ('sd',),
    'scene.cfg_scale_template': ('sd',),
    'scene.latent_scale_template': ('sd',),
    'scene.init_scale_template': ('sd',),
    'scene.image_scale_template': ('sd',),
}

# Display name overrides. The UI titles a field from its name ("enabled" -> "Enabled"),
# which reads fine INSIDE its own section but is meaningless once a field is hoisted next
# to strangers: `animatediff.enabled` on the Model card would just say "Enabled".
LABELS: Dict[str, str] = {
    'disco.guidance_noise_cap_percent':'Guidance update cap (% of step noise RMS, 0 off)',
    'disco.few_step_lora_path':'Distilled LoRA file or run directory',
    'disco.distilled_guided':'Guided distilled sampling',
    'disco.lora_strength':'LoRA strength (0 off, 1 trained strength)',
    'disco.quality_lora_enabled':'Enable quality LoRA',
    'disco.quality_lora_path':'Quality LoRA file',
    'disco.quality_lora_strength':'Quality LoRA strength (1 = 100%)',
    'disco.init_scale': 'LPIPS frame guidance scale (0 off)',
    'disco.image_scale': 'Image-space frame guidance scale (0 off)',
    'disco.mask_frame_guidance': 'Mask frame guidance to consistent areas',
    'diffusion.guidance_mode': 'Gradient guidance source',
    'diffusion.init_scale': 'Pixel guidance scale (0 off; try 1000)',
    'diffusion.init_latent_scale': 'Latent guidance scale (0 off; try 100)',
    'diffusion.clamp_grad': 'Clamp guidance gradient',
    'diffusion.clamp_max': 'Maximum gradient RMS',
    'diffusion.guidance_add_noise': 'Noise latent guidance target',
    'diffusion.sampler_scale_min_size':
        'Minimum multiscale size (longest side, px)',
    'diffusion.unet_cache': 'U-Net cache',
    'diffusion.unet_cache_interval': 'DeepCache interval',
    'diffusion.unet_cache_threshold': 'First Block similarity threshold',
    'diffusion.compile_unet': 'Compile U-Net, ControlNets, and VAE',
    'ipadapter.legacy_layer_indexing':
        'Legacy layer addressing (pre-0.7.1 renders)',
    'controlnet.pose_include_body': 'Detect body',
    'controlnet.pose_include_hand': 'Detect hands',
    'controlnet.pose_include_face': 'Detect face',
    'animatediff.enabled': 'AnimateDiff (motion module)',
    'main.keyframes_relative_to_frame_range':
        'Keyframes relative to frame range',
    'color.before_enabled': 'Enabled',
    'color.before_strength': 'Strength',
    'color.before_method': 'Method',
    'color.before_regrain': 'Regrain',
    'color.after_enabled': 'Enabled',
    'color.after_strength': 'Strength',
    'color.after_method': 'Method',
    'color.after_regrain': 'Regrain',
}


# Group order per tier, in the order the _assign calls below declare them. Without this the
# UI falls back to schema property order -- i.e. the order fields happen to be declared on
# RunConfig -- which put "Output" and "Model" ahead of "Input Video" on the Project tab.
GROUP_ORDER: Dict[str, list] = {}


def _assign(tier: str, group: str, *keys: str) -> None:
    groups = GROUP_ORDER.setdefault(tier, [])
    if group not in groups:
        groups.append(group)
    for key in keys:
        LAYOUT[key] = (tier, group)


# --- PROJECT ---------------------------------------------------------------------------
# frame_range belongs with the video: its ceiling comes from the clip's frame count, and
# the UI clamps it against what the probe reports.
_assign('project', 'Input Video',
        'video.video_init_path', 'video.extract_nth_frame',
        'video.width', 'video.height', 'video.max_size', 'main.frame_range',
        'main.keyframes_relative_to_frame_range')
_assign('project', 'Output',
        'main.batch_name', 'main.output_dir')
# The model is picked once when a project starts and then left alone, so it belongs with
# the other project setup, not on the tab you touch every run.
# AnimateDiff's on/off switch rides with it: in the notebook the motion module IS the model
# version (`control_multi_animatediff`). Its tuning stays in Advanced.
_assign('project', 'Model',
        'main.model_version', 'main.sd_checkpoint_path', 'animatediff.enabled')

# --- RENDER (the 80%) ------------------------------------------------------------------
_assign('render', 'Prompts',
        'main.text_prompts', 'main.negative_prompts',
        'flux.multi_reference_instruction',
        'hidream.multi_reference_instruction',
        'qwen.multi_reference_instruction',
        'mage.multi_reference_instruction')
_assign('render', 'Diffusion',
        'diffusion.steps_schedule', 'diffusion.cfg_scale_schedule',
        'diffusion.style_strength_schedule', 'diffusion.sampler',
        'diffusion.clip_skip', 'diffusion.guidance_mode',
        'diffusion.init_scale', 'diffusion.init_latent_scale',
        'diffusion.temporal_guidance_cc_trusted_only')
_assign('render', 'Seed', 'diffusion.seed')
_assign('render', 'ControlNet',
        'controlnet.enabled', 'controlnet.models', 'controlnet.mode',
        'controlnet.cond_image_src', 'controlnet.normalize_weights')
_assign('render', 'IP-Adapter',
        'ipadapter.enabled', 'ipadapter.models', 'ipadapter.flip_uc',
        'ipadapter.legacy_layer_indexing')
# Flux 2 Klein Edit (diffusers). Only active when model_version='flux2_klein_edit';
# the fields live on the Render tab so the edit knobs sit next to the prompts they
# drive. model_repo is a machine path and goes to System > Paths (below).
_assign('render', 'Reference Images',
        'main.reference_label_opacity', 'flux.references')
_assign('render', 'Flux Edit',
        'flux.guidance_scale', 'flux.steps', 'flux.fixed_seed')
_assign('render', 'Reference Images', 'hidream.references')
_assign('render', 'HiDream-O1 Edit',
        'hidream.guidance_scale', 'hidream.steps', 'hidream.fixed_seed',
        'hidream.use_trained_resolution')
_assign('render', 'Reference Images', 'qwen.references')
_assign('render', 'Qwen Image Edit',
        'qwen.use_lightning_lora', 'qwen.lora_strength',
        'qwen.guidance_scale', 'qwen.steps', 'qwen.fixed_seed')
_assign('render', 'Temporal Contact Sheet',
        'contact_sheet.mode', 'contact_sheet.layout',
        'contact_sheet.gutter', 'contact_sheet.instruction',
        'contact_sheet.save_debug')
_assign('render', 'Reference Images', 'mage.references')
_assign('render', 'Mage-Flow Edit',
        'mage.guidance_scale', 'mage.steps', 'mage.fixed_seed')
# The per-net settings (detectors, thresholds, mask options) live INSIDE the card of the
# net they affect -- ControlNetEditor renders them from the catalog's `detectors` tuple.
# Do not hoist them into an "advanced" tab: you would have to leave the ControlNet screen
# and come back just to switch the depth model. They are tiered here only so the schema
# classifies them; the generic Field grid never draws them, because the ControlNet group
# is claimed by ControlNetEditor.
_assign('render', 'ControlNet',
        'controlnet.depth_detector', 'controlnet.seg_detector',
        'controlnet.scribble_detector', 'controlnet.softedge_detector',
        'controlnet.pose_include_body', 'controlnet.pose_include_hand',
        'controlnet.pose_include_face',
        'controlnet.canny_low_threshold', 'controlnet.canny_high_threshold',
        'controlnet.mlsd_value_threshold', 'controlnet.mlsd_distance_threshold',
        'controlnet.qr_cn_mask_grayscale', 'controlnet.qr_cn_mask_invert',
        'controlnet.qr_cn_mask_thresh', 'controlnet.qr_cn_mask_clip_low',
        'controlnet.qr_cn_mask_clip_high', 'controlnet.downscale_tile',
        'controlnet.temporalnet_source', 'controlnet.temporalnet_skip_1st_frame',
        'controlnet.inpaint_mask_source')

# --- SYSTEM (machine-level; never travels with a settings file) -------------------------
_assign('system', 'Paths',
        'main.root_dir', 'main.model_path', 'main.lora_dir',
        'controlnet.model_dir', 'ipadapter.clip_vision_model_path',
        'animatediff.motion_module_path', 'video_assembly.upscale_model_path',
        'reconstruction_noise.cache_dir', 'flux.backend',
        'flux.comfy_server_url', 'flux.comfy_unet_name',
        'flux.comfy_clip_name', 'flux.comfy_vae_name',
        'flux.comfy_timeout', 'flux.model_repo',
        'hidream.comfy_server_url', 'hidream.comfy_checkpoint_name',
        'hidream.comfy_timeout',
        'qwen.comfy_server_url', 'qwen.comfy_unet_name',
        'qwen.comfy_clip_name', 'qwen.comfy_vae_name',
        'qwen.comfy_lora_name',
        'qwen.comfy_timeout',
        'mage.comfy_server_url', 'mage.comfy_unet_name',
        'mage.comfy_clip_name', 'mage.comfy_vae_name',
        'mage.comfy_timeout')
# "Performance" was a junk drawer: tiled VAE, tiled sampler and the optical-flow workers
# are three unrelated subsystems with their own knobs. Give each its own group so you can
# see what a setting actually belongs to.
_assign('system', 'Performance',
        'diffusion.sd_batch_size', 'main.lora_merge_precision',
        'diffusion.unet_cache', 'diffusion.unet_cache_interval',
        'diffusion.unet_cache_threshold', 'diffusion.compile_unet',
        'ipadapter.cache_embeddings', 'ipadapter.cache_size')
_assign('system', 'Multiscale / Tiled Sampler',
        'diffusion.tiled_sampler', 'diffusion.sampler_scale_schedule',
        'diffusion.sampler_scale_min_size',
        'diffusion.sampler_tile_size',
        'diffusion.sampler_tile_overlap', 'diffusion.sampler_tile_split_threshold')
_assign('system', 'Tiled VAE',
        'vae.use_tiled_vae', 'vae.num_tiles', 'vae.tile_size',
        'vae.vae_stride', 'vae.vae_padding')
_assign('system', 'Optical Flow (technical)',
        'flow.num_flow_workers', 'flow.flow_threads', 'flow.flow_maxsize',
        'flow.num_flow_updates', 'flow.flow_lq', 'flow.force_flow_generation',
        'flow.lazy_warp', 'flow.flow_save_img_preview')
_assign('system', 'Output Format',
        'video.save_img_format')

# --- HIDDEN ----------------------------------------------------------------------------
# Scalars superseded by their schedule. The notebook has no separate scalar at all --
# `steps = get_scheduled_arg(frame_num, steps_schedule)` -- and a one-element schedule IS
# the constant. We keep the fields so imported settings and old configs still resolve, but
# the schedule is the only thing you edit.
_assign('hidden', 'Superseded by schedule',
        'diffusion.steps', 'diffusion.cfg_scale', 'diffusion.style_strength',
        'warp.flow_blend',
        'flow.missed_consistency_weight', 'flow.overshoot_consistency_weight',
        'flow.edges_consistency_weight', 'flow.consistency_blur',
        'flow.consistency_dilate', 'flow.soften_consistency_mask')
_assign('hidden', 'Legacy Flux fields',
        'flux.denoise', 'flux.max_sequence_length')
# Not editable, but still in the config so settings files keep resolving:
#   animation_mode  - 'Video Input' is the only value the engine supports.
#   brightness.*    - the brightness adjuster is a legacy knob nobody should reach for.
_assign('hidden', 'Not applicable',
        'main.animation_mode',
        # Retained for WarpFusion/settings compatibility, but not useful in the
        # image-space flow path exposed by VibeWarp. warp_num_k belongs to the
        # unused k-means warp implementation; the other legacy warp controls
        # stay at the established defaults because their alternatives are
        # incomplete or tend to degrade the current image-space flow result.
        'warp.warp_mode', 'warp.warp_strength', 'warp.warp_num_k',
        'warp.warp_forward', 'warp.warp_towards_init',
        'brightness.enable', 'brightness.high_brightness_threshold',
        'brightness.high_brightness_adjust_ratio',
        'brightness.high_brightness_adjust_fix_amount',
        'brightness.max_brightness_threshold', 'brightness.low_brightness_threshold',
        'brightness.low_brightness_adjust_ratio',
        'brightness.low_brightness_adjust_fix_amount',
        'brightness.min_brightness_threshold')

# --- ADVANCED --------------------------------------------------------------------------
# Motion: set once for a project, then rarely touched again.
_assign('advanced', 'Motion',
        'warp.flow_blend_schedule', 'flow.flow_warp', 'flow.check_consistency')
_assign('advanced', 'Flow & Consistency',
        'flow.missed_consistency_dilation',
        'flow.edge_consistency_width',
        'flow.missed_consistency_schedule', 'flow.overshoot_consistency_schedule',
        'flow.edges_consistency_schedule', 'flow.consistency_blur_schedule',
        'flow.consistency_dilate_schedule', 'flow.soften_consistency_schedule')
_assign('advanced', 'Warp',
        'warp.padding_ratio', 'warp.padding_mode')
_assign('advanced', 'Diffusion (advanced)',
        'diffusion.fixed_seed', 'diffusion.use_karras_noise',
        'diffusion.clip_final_layer_norm', 'diffusion.dynamic_thresh',
        'diffusion.code_randomness', 'diffusion.do_softcap',
        'diffusion.softcap_thresh', 'diffusion.softcap_q', 'diffusion.noise_mode',
        'diffusion.clamp_grad', 'diffusion.clamp_max',
        'diffusion.guidance_add_noise', 'diffusion.guidance_use_start_code')
_assign('advanced', 'HiDream-O1',
        'hidream.sampler', 'hidream.scheduler', 'hidream.noise_scale',
        'hidream.patch_seam_smoothing', 'hidream.patch_seam_start',
        'hidream.patch_seam_passes', 'hidream.patch_seam_blend')
_assign('advanced', 'Qwen Image Edit',
        'qwen.sampler', 'qwen.scheduler', 'qwen.sampling_shift')
_assign('advanced', 'Mage-Flow Edit',
        'mage.sampler', 'mage.scheduler')
_assign('advanced', 'Reconstruction Noise',
        'reconstruction_noise.enabled', 'reconstruction_noise.randomness',
        'reconstruction_noise.cfg_scale', 'reconstruction_noise.steps_pct',
        'reconstruction_noise.source', 'reconstruction_noise.steps_cutoff',
        'reconstruction_noise.noise_scale', 'reconstruction_noise.separate_uncond',
        'reconstruction_noise.disable_controlnets',
        'reconstruction_noise.sliding_context', 'reconstruction_noise.batched',
        'reconstruction_noise.prompts', 'reconstruction_noise.neg_prompts')
# NOTE: animatediff.enabled is deliberately NOT here — it lives on the Render/Model card
# (see above). _assign is last-write-wins, so listing it again would silently pull it back
# into Advanced.
_assign('advanced', 'AnimateDiff',
        'animatediff.batch_length', 'animatediff.batch_overlap',
        'animatediff.context_length', 'animatediff.context_overlap',
        'animatediff.stop_early', 'animatediff.blend_batch_outputs',
        'animatediff.overlap_stylized', 'animatediff.reinject_stylized',
        'animatediff.lerp_reinject_stylized_frames', 'animatediff.frame_keyed_noise',
        'animatediff.pingpong_noise', 'animatediff.batched_adiff_rec_noise',
        'animatediff.rec_sliding_ctx', 'animatediff.looped_noise')
_assign('advanced', 'FreeU',
        'freeu.do_freeunet', 'freeu.apply_freeu_after_control',
        'freeu.b1', 'freeu.b2', 'freeu.s1', 'freeu.s2')
_assign('advanced', 'Scene Scheduling',
        'scene.analyze_video', 'scene.diff_function', 'scene.make_schedules',
        'scene.respect_sched', 'scene.diff_override', 'scene.flow_blend_template',
        'scene.cc_masked_template', 'scene.steps_template',
        'scene.style_strength_template', 'scene.cfg_scale_template',
        'scene.latent_scale_template', 'scene.init_scale_template',
        'scene.image_scale_template')
_assign('advanced', 'Captions',
        'captions.make_captions', 'captions.keyframe_source', 'captions.nth_frame',
        'captions.user_defined_keyframes', 'captions.diff_thresh',
        'captions.offset_mode', 'captions.fixed_offset', 'captions.blip_question')
_assign('advanced', 'Render Mask',
        'mask.use_background_mask', 'mask.invert_mask', 'mask.apply_mask_after_warp',
        'mask.background', 'mask.background_source',
        'mask.mask_clip_low', 'mask.mask_clip_high')
_assign('advanced', 'Color Match — Before Diffusion',
        'color.before_enabled', 'color.before_strength',
        'color.before_method', 'color.before_regrain')
_assign('advanced', 'Color Match — After Diffusion',
        'color.after_enabled', 'color.after_strength',
        'color.after_method', 'color.after_regrain')
_assign('hidden', 'Legacy Color fields',
        'color.match_color_strength', 'color.colormatch_method',
        'color.colormatch_regrain', 'color.colormatch_mode',
        'color.colormatch_after', 'color.colormatch_turbo')
# NOTE: brightness.* is deliberately HIDDEN (see the hidden tier above). _assign is
# last-write-wins and this block runs later, so re-listing it here would silently drag the
# whole group back into Advanced.
_assign('advanced', 'Video Output',
        'video_assembly.keep_audio', 'video_assembly.use_deflicker',
        'video_assembly.blend_mode', 'video_assembly.blend',
        'video_assembly.missed_consistency_weight',
        'video_assembly.overshoot_consistency_weight',
        'video_assembly.edges_consistency_weight',
        'video_assembly.upscale_ratio', 'video_assembly.upscale_model',
        'video_assembly.use_background_mask_video', 'video_assembly.invert_mask_video',
        'video_assembly.background_video', 'video_assembly.background_source_video',
        'video_assembly.mask_clip_low', 'video_assembly.mask_clip_high')

TIERS = ('project', 'render', 'advanced', 'system')
TIER_LABELS = {
    'project': 'Project',
    'render': 'Render',
    'advanced': 'Advanced',
    'system': 'System',
}


# Pixel model controls share the existing schema-driven form.
from vibewarp.config import DiscoConfig, TurboConfig
_assign('system', 'Paths', 'main.comfy_models_dir')
for _field in fields(DiscoConfig):
    _tier = 'system' if _field.name in ('checkpoint_path', 'secondary_path', 'few_step_lora_path') else 'advanced'
    if _field.name in ('sampler', 'distilled_guided', 'lora_strength', 'steps', 'steps_schedule', 'style_strength', 'style_strength_schedule', 'clip_guidance_scale', 'encoders', 'use_secondary_model', 'init_scale', 'image_scale', 'mask_frame_guidance'):
        _tier = 'render'
    _assign(_tier, 'Disco Diffusion', f'disco.{_field.name}')
_assign('project','Model','disco.checkpoint_path','disco.secondary_path','disco.few_step_lora_path')
_assign('project','Model','disco.quality_lora_path')
_assign('render','Disco Diffusion','disco.quality_lora_enabled','disco.quality_lora_strength')
for _field in fields(TurboConfig):
    _assign('render' if _field.name == 'enabled' else 'advanced', 'Turbo', f'turbo.{_field.name}')


def classify(section: str, field: str) -> Tuple[str, str] | None:
    """(tier, group) for a field, or None if it has not been classified."""
    return LAYOUT.get(f'{section}.{field}')


def model_families(section: str, field: str) -> Tuple[str, ...]:
    """Renderer families for which a field has an effect."""
    key = f'{section}.{field}'
    return FIELD_MODEL_FAMILIES.get(
        key, SECTION_MODEL_FAMILIES.get(section, ALL_MODEL_FAMILIES))


def model_family_for_version(model_version: str) -> str:
    """Resolve a configured model version without importing the heavy model loader."""
    declared = MODEL_FAMILY_BY_VERSION.get(model_version)
    if declared:
        return declared
    lowered = (model_version or '').lower()
    if 'hidream' in lowered:
        return 'hidream'
    if 'qwen' in lowered and 'edit' in lowered:
        return 'qwen'
    if 'flux' in lowered:
        return 'flux'
    return 'sd'


def label(section: str, field: str) -> str | None:
    """Display name override, or None to let the UI title it from the field name."""
    return LABELS.get(f'{section}.{field}')


def all_config_keys(config_cls: type) -> list[str]:
    """Every 'section.field' key a RunConfig exposes, in declaration order."""
    keys = []
    for item in fields(config_cls):
        annotation = item.type
        if is_dataclass(annotation):
            keys.extend(f'{item.name}.{sub.name}' for sub in fields(annotation))
        else:
            keys.append(f'main.{item.name}')
    return keys


def unclassified(config_cls: type) -> list[str]:
    """Config fields with no tab. Non-empty means the UI would silently drop them."""
    return [key for key in all_config_keys(config_cls) if key not in LAYOUT]


def system_keys() -> list[str]:
    """Fields that are machine-level and must not travel with a settings file."""
    return [key for key, (tier, _) in LAYOUT.items() if tier == 'system']

_assign('advanced', 'Disco guidance', 'disco.clip_grad_steps', 'disco.clip_divide_cuts',
    'disco.timestep_grid', 'disco.notebook_rng', 'disco.cut_icgray_schedule',
    'disco.init_scale_schedule', 'disco.image_scale_schedule', 'disco.inpaint_mask_sigma', 'disco.merge_loras')
_assign('advanced', 'Video Input', 'video.first_frame_path')
_assign('advanced', 'Video Input', 'video.visible_width', 'video.visible_height')
_assign('advanced', 'Video Output', 'video_assembly.crop_to_visible')
LABELS['disco.pad_to_multiple_64'] = 'Pad visible content to a 64-pixel canvas'
LABELS['disco.secondary_precision'] = 'Secondary model precision'
_assign('hidden', 'Unused compatibility fields', 'disco.secondary_precision')
_assign('hidden', 'Unused compatibility fields', 'disco.clip_precision')
LABELS['disco.cut_batch_group'] = 'CLIP cut batches per encoder call'
LABELS['video.visible_width'] = 'Visible content width (0 = automatic)'
LABELS['video.visible_height'] = 'Visible content height (0 = automatic)'
_assign('advanced', 'Video Output', 'video_assembly.fps')
_assign('hidden', 'Unused compatibility fields', 'flow.use_legacy_cc')
_assign('hidden', 'Unused compatibility fields', 'disco.cut_schedule_version')
LABELS['diffusion.temporal_guidance_cc_trusted_only'] = 'Temporal guidance only in CC-trusted regions'

# Keep each renderer's seed policy next to its shared base seed.
_assign('render', 'Seed', 'diffusion.fixed_seed', 'disco.fixed_seed',
        'flux.fixed_seed', 'hidream.fixed_seed', 'qwen.fixed_seed', 'mage.fixed_seed',
        'diffusion.fixed_seed_randomness', 'diffusion.fixed_seed_wiggle',
        'diffusion.fixed_seed_wiggle_direction')
LABELS['diffusion.fixed_seed_randomness'] = 'Fixed noise randomness'
LABELS['diffusion.fixed_seed_wiggle'] = 'Noise wiggle shift px (0 = off)'
LABELS['diffusion.fixed_seed_wiggle_direction'] = 'Noise wiggle direction [x, y]'

# Guidance has its own workspace; cutout schedules are directly editable before
# the less common gradient controls. Sampling and temporal controls stay put.
_assign('render', 'CLIP guidance', 'disco.encoders', 'disco.clip_guidance_scale',
        'disco.image_prompts', 'disco.use_secondary_model',
        'disco.init_scale_schedule', 'disco.image_scale_schedule', 'disco.mask_frame_guidance')
_assign('hidden', 'Superseded by schedule', 'disco.init_scale', 'disco.image_scale')
LABELS.update({
    'disco.init_scale_schedule': 'LPIPS frame guidance scale (0 off)',
    'disco.image_scale_schedule': 'Image-space frame guidance scale (0 off)',
})
_assign('render', 'Cutouts', 'disco.cutn_batches', 'disco.cut_overview',
        'disco.cut_max_size',
        'disco.cut_innercut', 'disco.cut_icgray_schedule', 'disco.cut_ic_pow',
        'disco.skip_augs', 'disco.cut_batch_group', 'disco.clip_divide_cuts',
        'disco.lock_cutouts_across_frames')
_assign('hidden', 'Unused compatibility fields', 'disco.cut_icgray_p')
_assign('advanced', 'Advanced guidance', 'disco.clip_grad_steps',
        'disco.guidance_every', 'disco.balance_guidance', 'disco.clamp_max',
        'disco.guidance_noise_cap_percent', 'disco.tv_scale', 'disco.range_scale',
        'disco.sat_scale')

_assign('render', 'Background masking',
        'mask.rvm_enabled', 'mask.rvm_downsample_ratio',
        'mask.foreground_color', 'mask.foreground_opacity',
        'mask.background_color', 'mask.background_opacity',
        'mask.raw_color_enabled', 'mask.raw_color_method',
        'mask.raw_color_strength', 'mask.raw_color_reference')
_assign('system', 'Paths', 'mask.rvm_model_path', 'mask.rvm_cache_dir')
_assign('advanced', 'Background masking', 'mask.use_background_mask',
        'mask.invert_mask', 'mask.apply_mask_after_warp', 'mask.background',
        'mask.background_source', 'mask.mask_clip_low', 'mask.mask_clip_high')
_assign('render', 'Background masking', 'mask.use_background_mask', 'mask.mask_source',
        'mask.mask_path', 'mask.render_mask_stage')
_assign('hidden', 'Legacy mask placement', 'mask.apply_mask_after_warp')
LABELS.update({
    'disco.cut_max_size': 'Cutout working size cap (px, 0 = native)',
    'mask.rvm_enabled': 'Apply masked tint',
    'mask.use_background_mask': 'Use Background Mask',
    'mask.render_mask_stage': 'Render mask stage',
    'mask.mask_source': 'Mask source',
    'mask.mask_path': 'Mask video or image folder',
    'mask.rvm_downsample_ratio': 'Matting resolution ratio',
    'mask.raw_color_enabled': 'Color-match raw video frames',
    'mask.raw_color_method': 'Raw frame color method',
    'mask.raw_color_strength': 'Raw frame color strength',
    'mask.raw_color_reference': 'Raw frame color reference',
    'mask.rvm_model_path': 'RVM FP16 TorchScript model (blank downloads official model)',
    'mask.rvm_cache_dir': 'Video input cache folder (blank uses workspace cache)',
})

# Each functional mask setting has one workflow, apart from machine-local RVM
# file overrides which are intentionally reachable through System too.
_assign('render', 'General', 'disco.init_scale_schedule', 'disco.image_scale_schedule',
        'disco.use_secondary_model', 'disco.mask_frame_guidance',
        'diffusion.temporal_guidance_cc_trusted_only')
_assign('render', 'Flow & Consistency', 'flow.check_consistency')
_assign('advanced', 'Flow & Consistency', 'turbo.consistency_soften', 'turbo.disable_cc')
for _path in ('disco.inpaint', 'disco.inpainting_stop', 'disco.inpaint_mask_blur',
              'disco.inpaint_mask_sigma', 'disco.inpaint_mask_threshold'):
    _assign(LAYOUT[_path][0], 'Disco inpainting masks', _path)
_assign('render', 'Raw input color match', 'mask.raw_color_enabled', 'mask.raw_color_method',
        'mask.raw_color_strength', 'mask.raw_color_reference')
