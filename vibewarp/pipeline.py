"""Top-level pipeline entry points for VibeWarp.

Usage:
    from vibewarp.pipeline import run
    from vibewarp.config import RunConfig

    config = RunConfig(
        video=VideoConfig(video_init_path='input.mp4'),
        text_prompts={0: 'a watercolor painting'},
        diffusion=DiffusionConfig(steps=30),
    )
    output_frames = run(config)
"""

import os
import warnings
from typing import Callable, Dict, List, Optional

warnings.filterwarnings("ignore")

import torch

from vibewarp.config import RunConfig
from vibewarp.controlnet_catalog import resolve_mode
from vibewarp.core.diffusion import (
    RenderContext,
    get_sampler_fn,
    run_frames,
)
from vibewarp.core.model_loader import (
    CFGDenoiser,
    configure_sdxl_clip_layer_norm,
    is_sdxl_model,
    load_controlnet,
    load_sd_model,
    setup_sdxl_model,
    wrap_model_for_kdiffusion,
)
from vibewarp.settings import save_settings, settings_to_dict
from vibewarp.utils.misc import create_path, seed_everything
from vibewarp.video.input import extract_frames, resolve_video_dimensions
from vibewarp.video.output import create_video


def _next_run_number(batch_dir: str) -> int:
    """Find the next available run number in a batch directory."""
    if not os.path.isdir(batch_dir):
        return 0
    existing = []
    for name in os.listdir(batch_dir):
        if os.path.isdir(os.path.join(batch_dir, name)):
            try:
                existing.append(int(name))
            except ValueError:
                pass
    return max(existing, default=-1) + 1


def setup_directories(config: RunConfig, existing_run_dir: Optional[str] = None) -> dict:
    """Create all required output directories for a run.

    Each run gets its own numbered subdirectory under the batch folder
    so previous runs are never overwritten.

    Args:
        config: run configuration

    Returns:
        Dict mapping directory names to their paths.
    """
    root = config.root_dir
    if existing_run_dir is not None:
        run_dir = os.path.abspath(existing_run_dir)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
    else:
        batch_base = os.path.join(root, config.output_dir, config.batch_name)
        run_num = _next_run_number(batch_base)
        run_dir = os.path.join(batch_base, str(run_num))

    dirs = {
        'output': os.path.join(root, config.output_dir),
        'batch': run_dir,
        'settings': os.path.join(run_dir, 'settings'),
        'temp': os.path.join(run_dir, 'temp'),
        'models': os.path.dirname(config.model_path) if config.model_path and os.path.isfile(config.model_path) else (config.model_path or os.path.join(root, 'models')),
        'init_images': os.path.join(root, 'init_images'),
        'embeddings': os.path.join(root, 'embeddings'),
        'video_frames': os.path.join(run_dir, 'video_frames'),
    }
    for d in dirs.values():
        create_path(d)
    print(f"  Run directory: {run_dir}")
    return dirs


def load_models(config: RunConfig, device: str = 'cuda') -> dict:
    """Load all required models for a run.

    Args:
        config: run configuration
        device: target device

    Returns:
        Dict with 'sd_model', 'model_wrap', 'model_wrap_cfg', and optionally
        'controlnets' and 'motion_module' keys.
    """
    from vibewarp.core.model_loader import MODEL_CONFIGS, is_edit_model

    # Flux edit models use the diffusers pipeline, not the CompVis/k-diffusion
    # stack — load it and return before any SD-specific setup runs.
    if is_edit_model(config.model_version):
        from vibewarp.core.edit import load_edit_backend
        return {'edit_backend': load_edit_backend(config, device=device)}

    # Resolve config path
    config_name = MODEL_CONFIGS.get(config.model_version, 'cldm_v15.yaml')
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'configs')
    config_path = os.path.join(config_dir, config_name)

    # Load SD model
    sd_model = load_sd_model(
        ckpt_path=config.sd_checkpoint_path,
        config_path=config_path,
        half_precision=True,
        device=device,
    )
    if 'sdxl' not in config.model_version.lower():
        from vibewarp.core.model_loader import configure_clip_skip
        configure_clip_skip(sd_model, config.diffusion.clip_skip)

    # Load AnimateDiff motion module BEFORE the k-diffusion wrap: the notebook
    # re-registers the beta schedule for AnimateDiff, and the wrap bakes
    # alphas_cumprod into its sigmas at construction.
    motion_module = None
    if config.animatediff.enabled:
        mm_path = config.animatediff.motion_module_path
        # The render still switches to AnimateDiff batch mode on `enabled` alone,
        # so skipping the module here would inject no temporal attention and look
        # like AnimateDiff silently did nothing.
        if not mm_path:
            raise ValueError(
                "AnimateDiff is enabled but animatediff.motion_module_path is not set")
        if not os.path.exists(mm_path):
            raise FileNotFoundError(f"AnimateDiff motion module not found: {mm_path}")
        from vibewarp.core.animatediff import load_motion_module
        from vibewarp.vendor.animatediff_mm import register_animatediff_schedule
        motion_module = load_motion_module(
            mm_path,
            model_version=config.model_version,
            device=device,
        )
        register_animatediff_schedule(sd_model, motion_module, config.model_version)
        print(f"Loaded AnimateDiff motion module: {os.path.basename(mm_path)}")

    # SDXL needs its SD1.5-style API patched on (apply_model vector conditioning,
    # cond_stage_model alias, alphas_cumprod). Must run AFTER the AnimateDiff beta
    # re-registration (which it prefers) and BEFORE the k-diffusion wrap, which
    # bakes alphas_cumprod into its sigmas at construction.
    if is_sdxl_model(config.model_version):
        setup_sdxl_model(sd_model, width=config.video.width, height=config.video.height)
        configure_sdxl_clip_layer_norm(sd_model, config.diffusion.clip_final_layer_norm)
        print(f"Patched SDXL model API ({config.video.width}x{config.video.height})")

    # Wrap for k-diffusion
    model_wrap, model_wrap_cfg = wrap_model_for_kdiffusion(
        sd_model,
        model_version=config.model_version,
    )

    result = {
        'sd_model': sd_model,
        'model_wrap': model_wrap,
        'model_wrap_cfg': model_wrap_cfg,
    }
    if motion_module is not None:
        result['motion_module'] = motion_module

    # Load only ControlNet models requested in settings (skip weight=0)
    if config.controlnet.enabled and config.controlnet.models:
        loaded_cns = {}
        for cn_key, cn_entry in config.controlnet.models.items():
            if cn_entry.weight == 0:
                print(f"Skipping ControlNet {cn_key}: weight=0")
                continue
            cn_path = _resolve_controlnet_path(cn_key, cn_entry.path, config.controlnet.model_dir)
            if not cn_path:
                raise ValueError(f"ControlNet {cn_key} requested but no path configured")
            if not os.path.exists(cn_path):
                raise FileNotFoundError(f"ControlNet {cn_key} checkpoint not found: {cn_path}")
            cn_model = load_controlnet(cn_path, model_version=config.model_version, device=device)
            # `mode` is the user-facing knob; layer_weights/zero_uncond are what
            # the engine reads. Deriving here (not just in the WarpFusion importer)
            # means UI/JSON/Python configs honour the mode too. 'custom' keeps the
            # entry's own hand-set weights.
            preset = resolve_mode(cn_entry.mode)
            layer_weights, zero_uncond = preset if preset is not None else (
                cn_entry.layer_weights, cn_entry.zero_uncond)
            loaded_cns[cn_key] = {
                'model': cn_model,
                'weight': cn_entry.weight,
                'start': cn_entry.start,
                'end': cn_entry.end,
                'annotator': cn_entry.annotator,
                'source': cn_entry.source,
                'detect_resolution': cn_entry.detect_resolution,
                'mode': cn_entry.mode,
                'layer_weights': layer_weights,
                'zero_uncond': zero_uncond,
            }
            print(f"Loaded ControlNet: {cn_key} (weight={cn_entry.weight})")
        result['controlnets'] = loaded_cns

    # Load IP-Adapter models if configured
    if config.ipadapter.enabled and not config.ipadapter.models:
        raise ValueError(
            "IP-Adapter is enabled but no adapter models are configured")
    if config.ipadapter.enabled and config.ipadapter.models:
        from vibewarp.core.ipadapter import load_ipadapter_model, detect_ipadapter_type
        from vibewarp.ipadapter_catalog import resolve_ipadapter_path
        loaded_ipa = {}
        weights_by_path = {}
        for ipa_key, ipa_entry in config.ipadapter.models.items():
            if ipa_entry.weight == 0:
                print(f"Skipping IP-Adapter {ipa_key}: weight=0")
                continue
            model_key = getattr(ipa_entry, 'model_key', '') or ipa_key
            ipa_path = resolve_ipadapter_path(
                model_key, ipa_entry.path, config.controlnet.model_dir)
            if not ipa_path:
                raise ValueError(
                    f"IP-Adapter {ipa_key} is enabled but no checkpoint path "
                    "is configured")
            if not os.path.isfile(ipa_path):
                raise FileNotFoundError(
                    f"IP-Adapter {ipa_key} checkpoint not found: {ipa_path}")

            # Repeated instances share immutable checkpoint tensors but receive
            # separate hooks, sources, weights, and schedules.
            try:
                ipa_weights = weights_by_path.get(ipa_path)
                if ipa_weights is None:
                    ipa_weights = load_ipadapter_model(ipa_path)
                    weights_by_path[ipa_path] = ipa_weights
                ipa_type = detect_ipadapter_type(ipa_weights)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load IP-Adapter {ipa_key} from {ipa_path}: "
                    f"{exc}") from exc
            loaded_ipa[ipa_key] = {
                'model_key': model_key,
                'weights': ipa_weights,
                'type_info': ipa_type,
                'path': ipa_path,
                'weight': ipa_entry.weight,
                'start': ipa_entry.start,
                'end': ipa_entry.end,
                'source_image': ipa_entry.source_image,
                'source_images': list(
                    getattr(ipa_entry, 'source_images', None) or []),
                'weight_type': ipa_entry.weight_type,
                'combine_embeds': getattr(ipa_entry, 'combine_embeds', 'concat'),
                'embeds_scaling': ipa_entry.embeds_scaling,
            }
            print(
                f"Loaded IP-Adapter {ipa_key}: {ipa_path} "
                f"(plus={ipa_type['is_plus']}, sdxl={ipa_type['is_sdxl']})")
        result['ipadapters'] = loaded_ipa

    return result


def _resolve_controlnet_path(model_key: str, configured_path: str, model_dir: str) -> str:
    """Resolve a ControlNet checkpoint against its optional shared directory."""
    if configured_path and os.path.isabs(configured_path):
        return configured_path
    if model_dir:
        if configured_path:
            return os.path.join(model_dir, configured_path)
        from vibewarp.core.download import CONTROL_MODEL_FILES
        filename = CONTROL_MODEL_FILES.get(model_key)
        if filename:
            return os.path.join(model_dir, filename)
    return configured_path


def _apply_ipadapter_hooks(
    sd_model: any,
    ipadapters: Dict,
    config: RunConfig,
) -> Dict[str, any]:
    """Load encoders now; defer hooks until immediately before sampling."""
    from vibewarp.core.ipadapter import (
        detect_clip_variant,
        load_clip_vision_model,
    )

    variants = set()
    for ipa_key, ipa_data in ipadapters.items():
        variant = detect_clip_variant(
            ipa_data.get('model_key', '') or ipa_key,
            ipa_data.get('weights'))
        ipa_data['clip_variant'] = variant
        variants.add(variant)

    configured_path = config.ipadapter.clip_vision_model_path
    if not configured_path:
        raise ValueError(
            "IP-Adapter is enabled but clip_vision_model_path is empty")

    filenames = {
        'vit_h': 'clip_vision_vit_h.safetensors',
        'vit_g': 'clip_vision_vit_bigg.safetensors',
    }
    models = {}
    for variant in variants:
        model_path = configured_path
        if os.path.isdir(configured_path):
            candidate = os.path.join(configured_path, filenames[variant])
            if os.path.isfile(candidate):
                model_path = candidate
            elif len(variants) > 1:
                raise FileNotFoundError(
                    f"IP-Adapter needs {filenames[variant]} in {configured_path}")
        models[variant] = load_clip_vision_model(model_path, variant=variant)
        print(f"Loaded IP-Adapter CLIP vision encoder: {variant}")
    return models


def _configure_unified_forward(
    sd_model,
    loaded_controlnets: Dict,
    freeu_config,
    diffusion_config=None,
    compile_cache_dir: str = '',
    normalize_cn_weights: bool = False,
    validate_stock_forward: bool = False,
) -> None:
    """Install the single production UNet forward and optional FreeU state.

    ``validate_stock_forward`` exists only for the GPU regression harness; it
    preserves the removed production baseline for same-seed A/B comparisons.
    """
    if validate_stock_forward:
        if freeu_config.do_freeunet:
            raise RuntimeError(
                "FreeU cannot be used with VIBEWARP_VALIDATE_STOCK_FORWARD=1")
        print("  Validation: using legacy stock UNet forward")
        return

    from vibewarp.core.controlnet import patch_model_for_controlnets
    patch_model_for_controlnets(
        sd_model, loaded_controlnets, normalize_weights=normalize_cn_weights)

    if freeu_config.do_freeunet:
        from vibewarp.core.freeu import set_freeu
        set_freeu(sd_model.model.diffusion_model, freeu_config)
        print(f"  FreeU enabled (b1={freeu_config.b1}, b2={freeu_config.b2}, "
              f"s1={freeu_config.s1}, s2={freeu_config.s2}, "
              f"after_control={freeu_config.apply_freeu_after_control})")

    if diffusion_config is not None:
        from vibewarp.core.unet_acceleration import configure_unet_acceleration
        configure_unet_acceleration(
            sd_model,
            cache_mode=diffusion_config.unet_cache,
            cache_interval=diffusion_config.unet_cache_interval,
            cache_threshold=diffusion_config.unet_cache_threshold,
            compile_unet=diffusion_config.compile_unet,
            compile_cache_dir=compile_cache_dir,
        )
        if diffusion_config.compile_unet:
            from vibewarp.core.unet_acceleration import compile_auxiliary_models
            compile_auxiliary_models(sd_model, loaded_controlnets)


def run(
    config: RunConfig,
    resume_from: int = 0,
    progress_fn: Optional[Callable] = None,
    existing_run_dir: Optional[str] = None,
) -> List[str]:
    """Run the full vid2vid style transfer pipeline.

    Pipeline stages:
    1. Setup directories
    2. Save settings
    3. Extract video frames (if video input)
    4. Load models
    5. Run frame rendering loop
    6. Return output frame paths

    Args:
        config: complete run configuration
        resume_from: frame number to resume from (0 = start fresh)
        progress_fn: optional callback(frame_num, total_frames)

    Returns:
        List of paths to rendered frame images.
    """
    from vibewarp.config_io import apply_path_defaults, validate_config
    apply_path_defaults(config)
    path_errors = validate_config(config, require_inputs=True, check_paths=True)
    if path_errors:
        details = "; ".join(f"{item['path']}: {item['message']}" for item in path_errors)
        raise ValueError(f"Invalid render configuration: {details}")
    if config.video.max_size > 0:
        config.video.width, config.video.height = resolve_video_dimensions(
            config.video.video_init_path, config.video.max_size)
        print(f"  Source aspect resolution: {config.video.width}x{config.video.height} "
              f"(max size {config.video.max_size})")
    # 1. Setup directories
    dirs = setup_directories(config, existing_run_dir=existing_run_dir)

    # 2. Save settings
    settings = settings_to_dict(config)
    save_settings(settings, dirs['batch'], config.batch_name, 0)

    # 3. Seed
    if config.diffusion.seed >= 0:
        seed_everything(config.diffusion.seed)

    # 4. Extract video frames (only for the requested frame range)
    video_frames_folder = dirs['video_frames']
    if config.video.video_init_path:
        if not os.path.isfile(config.video.video_init_path):
            raise FileNotFoundError(
                f"Input video not found: {config.video.video_init_path}")
        start_f = config.frame_range[0] if config.frame_range and config.frame_range[0] > 0 else 0
        end_f = config.frame_range[1] if config.frame_range and len(config.frame_range) > 1 and config.frame_range[1] > 0 else 999999999
        existing_frames = any(
            name.lower().endswith(('.jpg', '.jpeg', '.png'))
            for name in os.listdir(video_frames_folder))
        if existing_run_dir and existing_frames:
            print(f"  Reusing extracted frames: {video_frames_folder}")
        else:
            extract_frames(
                video_path=config.video.video_init_path,
                output_dir=video_frames_folder,
                nth_frame=config.video.extract_nth_frame,
                start_frame=start_f,
                end_frame=end_f,
            )

    # 4a-0. Content-aware scene analysis (notebook cells 26–28). Needed when
    # schedules are template-driven or captioning uses content-aware keyframes.
    scene_diff = None
    sc = config.scene
    _need_diff = sc.analyze_video or sc.make_schedules or (
        config.captions.make_captions
        and config.captions.keyframe_source == 'Content-aware scheduling keyframes')
    if _need_diff:
        if sc.diff_override:
            scene_diff = list(sc.diff_override)
            print(f"  Scene analysis: using diff_override ({len(scene_diff)} values)")
        else:
            from vibewarp.core.scene_detect import analyze_video_frames
            scene_diff = analyze_video_frames(
                video_frames_folder, diff_function=sc.diff_function)
    if sc.make_schedules:
        if scene_diff is None:
            raise RuntimeError(
                "make_schedules=True but frames were not analyzed — enable "
                "analyze_video or provide diff_override (notebook behavior)")
        from vibewarp.core.scene_detect import apply_content_schedules
        apply_content_schedules(config, scene_diff)

    # 4a. Keyframe captioning pre-pass (notebook cell 30) — enables the
    # {caption} prompt placeholder, substituted per frame in render_frame.
    if config.captions.make_captions:
        from glob import glob as _glob
        from vibewarp.core.captioning import (
            select_keyframes, remap_keyframes, generate_captions,
        )
        cap = config.captions
        n_frames = len(_glob(os.path.join(video_frames_folder, '*.jpg')))
        keyframes = select_keyframes(
            n_frames, source=cap.keyframe_source, nth_frame=cap.nth_frame,
            user_keyframes=cap.user_defined_keyframes,
            diffs=scene_diff,
            diff_thresh=cap.diff_thresh,
        )
        if keyframes:
            keyframes = remap_keyframes(keyframes, cap.offset_mode, cap.fixed_offset)
            folder = generate_captions(
                video_frames_folder, keyframes, blip_question=cap.blip_question)
            print(f"  Captions written: {folder} ({len(keyframes)} keyframes)")

    # 4b. CUDA best practices for inference
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    # 5. Load models
    from vibewarp.core.timing import log_time as _log_time
    import time as _time
    from datetime import datetime as _dt
    try:
        from vibewarp.core.timing import _LOG_PATH as _profile_log
        with open(_profile_log, 'a') as _f:
            _f.write(f"\n=== Run {config.batch_name} {_dt.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    except Exception:
        pass
    _t_models = _time.time()
    models = load_models(config)
    _log_time("load_models total", _time.time() - _t_models)

    # 5b. Load RAFT model if flow warping is enabled
    raft_model = None
    if config.flow.flow_warp:
        from vibewarp.flow.flow_utils import load_raft_model
        raft_model = load_raft_model(half_precision=config.flow.flow_lq)

    # Flux uses the diffusers pipeline directly; the AnimateDiff / IP-Adapter /
    # ControlNet / tiled-VAE / k-diffusion setup below is all SD-specific and
    # would dereference an sd_model that Flux never loads. Skip it wholesale.
    from vibewarp.core.model_loader import is_edit_model
    is_edit = is_edit_model(config.model_version)

    motion_module = None
    ipadapters = {}
    clip_vision_models = {}
    loaded_controlnets = {}
    if not is_edit:
        # 5c. Inject AnimateDiff motion modules if loaded
        motion_module = models.get('motion_module')
        if motion_module is not None:
            from vibewarp.core.animatediff import inject_motion_modules
            inject_motion_modules(models['sd_model'], motion_module)

        # 5d. Hook IP-Adapters if loaded
        ipadapters = models.get('ipadapters', {})
        if ipadapters:
            clip_vision_models = _apply_ipadapter_hooks(
                models['sd_model'], ipadapters, config)

        # 5e. Always use the unified ControlNet-capable forward. With empty model
        # and hint dictionaries it is identical to the stock UNet forward.
        loaded_controlnets = models.get('controlnets', {})
        validate_stock_forward = os.environ.get('VIBEWARP_VALIDATE_STOCK_FORWARD') == '1'
        _configure_unified_forward(
            models['sd_model'], loaded_controlnets, config.freeu,
            diffusion_config=config.diffusion,
            compile_cache_dir=os.path.join(
                config.root_dir, '.vibewarp_cache', 'torchinductor'),
            normalize_cn_weights=config.controlnet.normalize_weights,
            validate_stock_forward=validate_stock_forward)

        # 5f. Offload VAE and CLIP text encoder to CPU (moved to GPU on demand)
        sd_model = models['sd_model']
        if hasattr(sd_model, 'cond_stage_model'):
            sd_model.cond_stage_model.cpu()
        if hasattr(sd_model, 'first_stage_model'):
            sd_model.first_stage_model.cpu()
        torch.cuda.empty_cache()

        # 5g. Patch tiled VAE if enabled
        if config.vae.use_tiled_vae:
            from vibewarp.core.vae import patch_model_for_tiled_vae
            patch_model_for_tiled_vae(models['sd_model'], config.vae)

    # 6. Build render context
    flow_folder = os.path.join(dirs['batch'], 'flow')
    ctx = RenderContext(
        config=config,
        sd_model=models.get('sd_model'),
        model_wrap=models.get('model_wrap'),
        model_wrap_cfg=models.get('model_wrap_cfg'),
        edit_backend=models.get('edit_backend'),
        loaded_controlnets=loaded_controlnets,
        loaded_ipadapters=ipadapters,
        clip_vision_model=clip_vision_models,
        raft_model=raft_model,
        motion_module=motion_module,
        sampler_fn=None if is_edit else get_sampler_fn(config.diffusion.sampler),
        batch_folder=dirs['batch'],
        video_frames_folder=video_frames_folder,
        flow_folder=flow_folder,
        progress_fn=progress_fn,
    )

    # 7. Run frame loop
    frame_range = config.frame_range if config.frame_range != [0, 0] else None
    output_paths = run_frames(ctx, frame_range=frame_range, resume_from=resume_from)

    # 7b. Eject AnimateDiff motion modules after rendering
    if motion_module is not None:
        from vibewarp.core.animatediff import eject_motion_modules
        eject_motion_modules(models['sd_model'], motion_module)

    # 7b2. Unpatch tiled VAE if it was applied (never patched on the Flux path)
    if config.vae.use_tiled_vae and not is_edit:
        from vibewarp.core.vae import unpatch_model_tiled_vae
        unpatch_model_tiled_vae(models['sd_model'])

    # 7c. Unhook IP-Adapters after rendering
    if ipadapters:
        from vibewarp.core.ipadapter import unhook_ipadapter
        unhook_ipadapter(models['sd_model'])

    # 8. Assemble video
    if output_paths:
        try:
            assemble_video(config, dirs['batch'])
        except Exception as e:
            print(f"Video assembly failed: {e}")

    # 9. Hand the VRAM back. The server renders jobs back to back in one process, so
    # without this the next job starts against a ceiling set by this one -- and the OOM
    # reads as "these settings are too big" rather than "we never let go of the last run".
    from vibewarp.utils.misc import release_vram
    release_vram()

    return output_paths


def assemble_video(config: RunConfig, run_dir: str) -> str:
    """Build (or rebuild) a video from every completed frame in an existing run."""
    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    va = config.video_assembly
    flow_folder = os.path.join(run_dir, 'flow')
    print(f"  Assembling video from completed frames in: {run_dir}")
    return create_video(
        frames_dir=run_dir,
        output_path=os.path.join(run_dir, f"{config.batch_name}.mp4"),
        fps=12.0,
        batch_name=config.batch_name,
        img_format=config.video.save_img_format,
        blend_mode=va.blend_mode if config.flow.flow_warp else 'None',
        blend=va.blend,
        flow_dir=flow_folder if config.flow.flow_warp else None,
        check_consistency=config.flow.check_consistency,
        missed_consistency_weight=va.missed_consistency_weight,
        overshoot_consistency_weight=va.overshoot_consistency_weight,
        edges_consistency_weight=va.edges_consistency_weight,
        keep_audio=va.keep_audio,
        source_video=config.video.video_init_path,
        use_deflicker=va.use_deflicker,
        upscale_ratio=va.upscale_ratio,
        upscale_model=va.upscale_model,
        upscale_model_path=va.upscale_model_path,
        use_background_mask_video=va.use_background_mask_video,
        invert_mask_video=va.invert_mask_video,
        background_video=va.background_video,
        background_source_video=va.background_source_video,
        video_frames_folder=os.path.join(run_dir, 'video_frames'),
        mask_clip_low=va.mask_clip_low,
        mask_clip_high=va.mask_clip_high,
    )
