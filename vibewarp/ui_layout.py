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
}
ALL_MODEL_FAMILIES = ('sd', 'flux', 'hidream', 'qwen', 'mage')
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
}
FIELD_MODEL_FAMILIES = {
    # External edit renderers still use this seed as their base starting noise.
    'diffusion.seed': ALL_MODEL_FAMILIES,
    'main.reference_label_opacity': ('flux', 'hidream', 'qwen', 'mage'),
    # FLUX.2 uses positive instructions with zeroed negative conditioning.
    'main.negative_prompts': ('sd', 'hidream', 'qwen', 'mage'),
    # These paths and precision settings belong to the local SD stack.
    'main.sd_checkpoint_path': ('sd',),
    'main.model_path': ('sd',),
    'main.lora_dir': ('sd',),
    'main.lora_merge_precision': ('sd',),
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
    'diffusion.guidance_mode': 'Temporal guidance source',
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
        'diffusion.clip_skip', 'diffusion.guidance_mode')
_assign('render', 'Seed', 'diffusion.seed')
_assign('render', 'ControlNet',
        'controlnet.enabled', 'controlnet.models', 'controlnet.mode',
        'controlnet.cond_image_src', 'controlnet.normalize_weights')
_assign('render', 'IP-Adapter',
        'ipadapter.enabled', 'ipadapter.models', 'ipadapter.flip_uc')
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
        'ipadapter.cache_embeddings', 'ipadapter.cache_size')
_assign('system', 'Tiled Sampler',
        'diffusion.tiled_sampler', 'diffusion.sampler_tile_size',
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
        'flow.use_legacy_cc', 'flow.missed_consistency_dilation',
        'flow.edge_consistency_width',
        'flow.missed_consistency_schedule', 'flow.overshoot_consistency_schedule',
        'flow.edges_consistency_schedule', 'flow.consistency_blur_schedule',
        'flow.consistency_dilate_schedule', 'flow.soften_consistency_schedule')
_assign('advanced', 'Warp',
        'warp.padding_ratio', 'warp.padding_mode')
_assign('advanced', 'Diffusion (advanced)',
        'diffusion.fixed_seed', 'diffusion.use_karras_noise',
        'diffusion.clip_final_layer_norm', 'diffusion.init_scale',
        'diffusion.init_latent_scale', 'diffusion.dynamic_thresh',
        'diffusion.code_randomness', 'diffusion.do_softcap',
        'diffusion.softcap_thresh', 'diffusion.softcap_q', 'diffusion.noise_mode',
        'diffusion.guidance_use_start_code')
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
