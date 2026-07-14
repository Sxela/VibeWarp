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


def setup_directories(config: RunConfig) -> dict:
    """Create all required output directories for a run.

    Each run gets its own numbered subdirectory under the batch folder
    so previous runs are never overwritten.

    Args:
        config: run configuration

    Returns:
        Dict mapping directory names to their paths.
    """
    root = config.root_dir
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
    from vibewarp.core.model_loader import MODEL_CONFIGS

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
    if config.ipadapter.enabled and config.ipadapter.models:
        from vibewarp.core.ipadapter import load_ipadapter_model, detect_ipadapter_type
        loaded_ipa = {}
        for ipa_key, ipa_entry in config.ipadapter.models.items():
            if ipa_entry.path and os.path.exists(ipa_entry.path):
                try:
                    ipa_weights = load_ipadapter_model(ipa_entry.path)
                    ipa_type = detect_ipadapter_type(ipa_weights)
                    loaded_ipa[ipa_key] = {
                        'weights': ipa_weights,
                        'type_info': ipa_type,
                        'path': ipa_entry.path,
                        'weight': ipa_entry.weight,
                        'start': ipa_entry.start,
                        'end': ipa_entry.end,
                        'source_image': ipa_entry.source_image,
                        'weight_type': ipa_entry.weight_type,
                        'combine_embeds': getattr(ipa_entry, 'combine_embeds', 'concat'),
                        'embeds_scaling': ipa_entry.embeds_scaling,
                    }
                    print(f"Loaded IP-Adapter: {ipa_key} "
                          f"(plus={ipa_type['is_plus']}, sdxl={ipa_type['is_sdxl']})")
                except Exception as e:
                    print(f"Failed to load IP-Adapter {ipa_key}: {e}")
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
) -> None:
    """Apply IP-Adapter hooks to the SD model.

    For each loaded IP-Adapter, generates CLIP embeddings from the source image
    and hooks the adapter into the UNet's cross-attention blocks.

    Args:
        sd_model: loaded SD model
        ipadapters: dict of loaded IP-Adapter data from load_models()
        config: run configuration (for CLIP model path)
    """
    from vibewarp.core.ipadapter import (
        hook_ipadapter,
        load_image_for_clip,
        encode_clip_vision,
        IPAdapterEmbeddingCache,
    )

    # Load CLIP vision model if needed
    clip_model = None
    if config.ipadapter.clip_vision_model_path:
        try:
            from transformers import CLIPVisionModelWithProjection
            clip_model = CLIPVisionModelWithProjection.from_pretrained(
                config.ipadapter.clip_vision_model_path
            ).half().cuda().eval()
        except ImportError:
            print("transformers not installed — CLIP vision encoding unavailable")
        except Exception as e:
            print(f"Failed to load CLIP vision model: {e}")

    embed_cache = IPAdapterEmbeddingCache() if config.ipadapter.cache_embeddings else None

    for ipa_key, ipa_data in ipadapters.items():
        source_image = ipa_data.get('source_image', '')
        clip_output = None

        if source_image and os.path.exists(source_image) and clip_model is not None:
            # Check cache
            cache_key = f"{source_image}_{ipa_key}"
            if embed_cache and cache_key in embed_cache:
                clip_output = embed_cache.get(cache_key)
            else:
                image_tensor = load_image_for_clip(source_image, device='cuda')
                clip_output = encode_clip_vision(clip_model, image_tensor)
                if embed_cache:
                    embed_cache.put(cache_key, clip_output)

        if clip_output is None:
            # Create dummy embeddings for hookup (will produce zero contribution)
            clip_output = {'image_embeds': torch.zeros(1, 768)}

        # Notebook: is_v2 = 'v2' in the adapter checkpoint filename
        adapter_path = ipa_data.get('path', '') or ''
        hook_ipadapter(
            sd_model=sd_model,
            adapter_weights=ipa_data['weights'],
            clip_vision_output=clip_output,
            weight=ipa_data['weight'],
            start=ipa_data['start'],
            end=ipa_data['end'],
            weight_type=ipa_data['weight_type'],
            embeds_scaling=ipa_data['embeds_scaling'],
            is_v2='v2' in os.path.basename(adapter_path).lower(),
            combine_embeds=ipa_data.get('combine_embeds', 'concat'),
            flip_uc=config.ipadapter.flip_uc,
        )
        print(f"Hooked IP-Adapter: {ipa_key} (weight={ipa_data['weight']})")


def _configure_unified_forward(
    sd_model,
    loaded_controlnets: Dict,
    freeu_config,
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


def run(
    config: RunConfig,
    resume_from: int = 0,
    progress_fn: Optional[Callable] = None,
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
    dirs = setup_directories(config)

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

    # 5c. Inject AnimateDiff motion modules if loaded
    motion_module = models.get('motion_module')
    if motion_module is not None:
        from vibewarp.core.animatediff import inject_motion_modules
        inject_motion_modules(models['sd_model'], motion_module)

    # 5d. Hook IP-Adapters if loaded
    ipadapters = models.get('ipadapters', {})
    if ipadapters:
        _apply_ipadapter_hooks(models['sd_model'], ipadapters, config)

    # 5e. Always use the unified ControlNet-capable forward. With empty model
    # and hint dictionaries it is identical to the stock UNet forward.
    loaded_controlnets = models.get('controlnets', {})
    validate_stock_forward = os.environ.get('VIBEWARP_VALIDATE_STOCK_FORWARD') == '1'
    _configure_unified_forward(
        models['sd_model'], loaded_controlnets, config.freeu,
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
    loaded_controlnets = models.get('controlnets', {})
    ctx = RenderContext(
        config=config,
        sd_model=models['sd_model'],
        model_wrap=models['model_wrap'],
        model_wrap_cfg=models['model_wrap_cfg'],
        loaded_controlnets=loaded_controlnets,
        loaded_ipadapters=ipadapters,
        raft_model=raft_model,
        motion_module=models.get('motion_module'),
        sampler_fn=get_sampler_fn(config.diffusion.sampler),
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

    # 7b2. Unpatch tiled VAE if it was applied
    if config.vae.use_tiled_vae:
        from vibewarp.core.vae import unpatch_model_tiled_vae
        unpatch_model_tiled_vae(models['sd_model'])

    # 7c. Unhook IP-Adapters after rendering
    if ipadapters:
        from vibewarp.core.ipadapter import unhook_ipadapter
        unhook_ipadapter(models['sd_model'])

    # 8. Assemble video
    if output_paths:
        video_output_path = os.path.join(
            dirs['batch'],
            f"{config.batch_name}.mp4"
        )
        va = config.video_assembly
        try:
            create_video(
                frames_dir=dirs['batch'],
                output_path=video_output_path,
                fps=12.0,
                batch_name=config.batch_name,
                img_format=config.video.save_img_format,
                blend_mode=va.blend_mode if config.flow.flow_warp else 'None',
                blend=va.blend,
                flow_dir=flow_folder if config.flow.flow_warp else None,
                check_consistency=config.flow.check_consistency,
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
                video_frames_folder=video_frames_folder,
                mask_clip_low=va.mask_clip_low,
                mask_clip_high=va.mask_clip_high,
            )
        except Exception as e:
            print(f"Video assembly failed: {e}")

    # 9. Hand the VRAM back. The server renders jobs back to back in one process, so
    # without this the next job starts against a ceiling set by this one -- and the OOM
    # reads as "these settings are too big" rather than "we never let go of the last run".
    from vibewarp.utils.misc import release_vram
    release_vram()

    return output_paths
