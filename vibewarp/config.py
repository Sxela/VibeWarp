"""Default configuration, model URLs, and settings dataclass."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from vibewarp.ipadapter_catalog import ipadapter_urls


# ---- Model download URLs ----

MODEL_URLS = {
    "sd_v1_5": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
    "dpt_hybrid_midas": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt",
    "clip_vision_vit_h": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors",
    "clip_vision_vit_bigg": "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors",
}

CONTROLNET_MODEL_URLS = {
    "control_sd15_canny": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth",
    "control_sd15_depth": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth",
    "control_sd15_openpose": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth",
    "control_sd15_softedge": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_softedge.pth",
    "control_sd15_scribble": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble.pth",
    "control_sd15_lineart": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth",
    "control_sd15_seg": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_seg.pth",
    "control_sd15_shuffle": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11e_sd15_shuffle.pth",
    "control_sd15_inpaint": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_inpaint.pth",
    "control_sd15_ip2p": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11e_sd15_ip2p.pth",
    "control_sd15_tile": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth",
    "control_sd15_normalbae": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_normalbae.pth",
}

CONTROLNET_ANNOTATOR_URLS = {
    "control_sd15_openpose": [
        "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx",
        "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx",
    ],
}

IPADAPTER_URLS = ipadapter_urls()


# ---- Run configuration ----

@dataclass
class FlowConfig:
    """Optical flow generation settings."""
    flow_warp: bool = True
    check_consistency: bool = True
    use_legacy_cc: bool = False
    flow_lq: bool = True
    flow_save_img_preview: bool = False
    num_flow_updates: int = 20  # notebook default is 20, torchvision RAFT default is 12
    num_flow_workers: int = 0
    flow_threads: int = 4
    flow_maxsize: int = 0  # 0 = full resolution (matches notebook default)
    missed_consistency_dilation: int = 2
    edge_consistency_width: int = 11
    # Per-component CC mask weights (0 = ignore that layer, 1 = full effect)
    missed_consistency_weight: float = 1.0
    overshoot_consistency_weight: float = 1.0
    edges_consistency_weight: float = 1.0
    # Post-processing for CC mask
    consistency_blur: int = 1    # gaussian blur sigma applied after threshold
    consistency_dilate: int = 3  # morphological dilation radius applied after threshold
    # soften_consistency_mask: clips the minimum value of the CC weights after load_cc.
    # 0 = no softening (hard mask). 1 = fully ignore consistency (weight stays 1 everywhere).
    # Matches notebook's forward_weights_clip / soften_consistency_mask.
    soften_consistency_mask: float = 0.0
    # Per-frame schedules (dict/list/scalar — same formats as diffusion schedules).
    # When None, the corresponding static field above is used for every frame.
    missed_consistency_schedule: Any = None
    overshoot_consistency_schedule: Any = None
    edges_consistency_schedule: Any = None
    consistency_blur_schedule: Any = None
    consistency_dilate_schedule: Any = None
    soften_consistency_schedule: Any = None
    lazy_warp: bool = True
    force_flow_generation: bool = False


@dataclass
class WarpConfig:
    """Frame warping settings."""
    # DEPRECATED: 'use_latent' was hidden as deprecated in the notebook GUI and
    # is not implemented here; the field only exists so old settings files
    # parse. Only 'use_image' is supported — nothing consumes this value.
    warp_mode: str = 'use_image'
    warp_strength: float = 1.0
    warp_num_k: int = 8
    warp_forward: bool = True
    warp_towards_init: str = 'off'
    padding_ratio: float = 0.1
    padding_mode: str = 'reflect'
    flow_blend: float = 0.5
    flow_blend_schedule: Any = None


@dataclass
class DiffusionConfig:
    """Stable Diffusion inference settings."""
    steps: int = 25
    cfg_scale: float = 7.5
    sampler: str = 'sample_euler_ancestral'
    seed: int = -1
    fixed_seed: bool = False
    use_karras_noise: bool = True
    clip_skip: int = 1
    # SDXL only. True (default) = notebook/A1111 behaviour: CLIP's final layer norm is
    # applied to the clip-skipped hidden state. False leaves the raw hidden state,
    # whose magnitude is ~10x larger — the prompt is followed much more strongly, but
    # it does NOT match the notebook. Kept as an escape hatch because the unnormalised
    # look is often preferred; leave it on for parity work.
    clip_final_layer_norm: bool = True
    sd_batch_size: int = 1
    # Optional local SD/SDXL U-Net acceleration. DeepCache reuses the expensive
    # inner U-Net branch on intermediate denoiser evaluations; First Block Cache
    # reuses a previous output when the first encoder block changes less than the
    # configured relative threshold. Cache state is reset for every rendered frame.
    unet_cache: str = 'off'
    unet_cache_interval: int = 2
    unet_cache_threshold: float = 0.05
    # Compile the unified U-Net, active ControlNets, and VAE encode/decode paths.
    # Compilation is deliberately opt-in: every new multiscale/tile/VAE shape may
    # need another graph. Inductor artifacts persist under .vibewarp_cache.
    compile_unet: bool = False
    # MultiDiffusion-style spatial tiling: every denoiser evaluation runs on
    # overlapping latent tiles and blends a full-size prediction for the sampler.
    tiled_sampler: bool = False
    # Sampling-step -> percent of final latent resolution. Below 100%, the UNet
    # evaluates the whole downscaled frame and tiling is deliberately disabled;
    # at 100%, the normal tiled path takes over. The sampler latent itself always
    # remains full resolution.
    sampler_scale_schedule: Dict[int, float] = field(
        default_factory=lambda: {0: 100.0},
        metadata={'schedule_domain': 'sampling_step'})
    # Pixel-space lower bound for the longest side of a temporary multiscale
    # evaluation. 0 disables the floor. For SD1.5, 512 keeps coarse passes at
    # least at the model's nominal training resolution.
    sampler_scale_min_size: int = 0
    # Pixel-space target tile edge. Tile rows/columns are derived from the
    # actual latent dimensions; overlap is shared by both spatial axes.
    sampler_tile_size: int = 512
    # Overlap between neighbouring tiles, as a percentage of the tile edge. This is the
    # cross-fade that hides the seams. 0 = hard cuts (fastest, seams likely).
    sampler_tile_overlap: float = 35.0
    # How far a dimension may exceed sampler_tile_size before it is worth splitting into
    # another tile, as a percentage of the tile edge.
    #
    # The tile size is DERIVED from the tile count, so a plain ceil(total/tile) is a trap:
    # 1080px against a 1024px target asks for 2 tiles, and those come out ~640px each --
    # you request 1024 and silently get 640, at twice the model evals. This slack lets a
    # single tile stretch a little instead. At 20%, a 1024px target covers up to ~1229px
    # with one tile.
    #
    # 0 = never stretch: split as soon as the dimension exceeds the tile.
    sampler_tile_split_threshold: float = 20.0
    style_strength: float = 0.65
    # Target used by pixel/latent gradient guidance after the first frame.
    # This is independent of the normal img2img init, which remains the
    # warp+consistency composite.
    guidance_mode: str = 'prev warped + cc'
    init_scale: float = 0.0
    init_latent_scale: float = 0.0
    clamp_grad: bool = True
    clamp_max: float = 2.0
    guidance_add_noise: bool = True
    dynamic_thresh: float = 30.0  # static thresholding clamp value (notebook: dynamic_thresh)
    code_randomness: float = 0.0  # blend ratio for fixed_code noise path (notebook: code_randomness)
    # Softcap: soft-compress decoded-image values beyond ±softcap_thresh
    # (damps cross-frame feedback loop; notebook cell 20/settings)
    do_softcap: bool = False
    softcap_thresh: float = 0.9
    softcap_q: float = 1.0
    # noise_mode: 'default' (randn each frame), 'fixed' (persistent start_code), 'reconstructed' (DDIM inversion)
    noise_mode: str = 'default'
    # Reuse one noise tensor when guidance_add_noise perturbs the latent target
    # to the current sampler sigma. The draw also preserves notebook RNG order.
    guidance_use_start_code: bool = True
    # Schedules: dict {frame: value} or list [value_per_frame] or scalar (constant)
    steps_schedule: Any = None
    cfg_scale_schedule: Any = None
    style_strength_schedule: Any = None


@dataclass
class VideoConfig:
    """Video I/O settings."""
    video_init_path: str = ''
    extract_nth_frame: int = 1
    width: int = 512
    height: int = 512
    max_size: int = 0  # when >0, derive width/height from source aspect ratio
    save_img_format: str = 'png'


@dataclass
class BrightnessConfig:
    """Automatic brightness adjustment settings."""
    enable: bool = False
    high_brightness_threshold: float = 180
    high_brightness_adjust_ratio: float = 0.97
    high_brightness_adjust_fix_amount: float = 2
    max_brightness_threshold: float = 254
    low_brightness_threshold: float = 40
    low_brightness_adjust_ratio: float = 1.03
    low_brightness_adjust_fix_amount: float = 2
    min_brightness_threshold: float = 1


@dataclass
class ColorConfig:
    """Independent color matching before and after diffusion."""
    # Before diffusion: colorize raw/current pixels toward the warped previous
    # render before consistency-mask blending exposes them to the model.
    before_enabled: bool = False
    before_strength: float = 0.5
    before_method: str = 'PDF'
    before_regrain: bool = False
    # After diffusion: pull the new model output back toward the previous
    # frame's warped palette before saving it and feeding it forward.
    after_enabled: bool = False
    after_strength: float = 0.5
    after_method: str = 'PDF'
    after_regrain: bool = False
    # Legacy shared/dropdown surface retained for settings compatibility only.
    match_color_strength: float = 0.0
    colormatch_method: str = 'PDF'
    colormatch_regrain: bool = False
    colormatch_mode: str = 'before'
    colormatch_after: bool = True
    colormatch_turbo: bool = False


@dataclass
class ControlNetEntry:
    """Per-ControlNet model configuration."""
    path: str = ''
    weight: float = 1.0
    start: float = 0.0
    end: float = 1.0
    annotator: str = ''  # override annotator (auto-detected from model key if empty)
    source: str = ''  # 'init' or 'video' or path — defaults to init image
    detect_resolution: int = -1  # annotator input resolution (−1 = match output)
    layer_weights: Optional[List[float]] = None  # per-layer weights (13 values for SD1.5)
    mode: str = 'balanced'  # notebook: balanced / controlnet / prompt
    zero_uncond: bool = False


@dataclass
class ControlNetConfig:
    """ControlNet multi-model settings."""
    enabled: bool = False
    model_dir: str = ''  # base directory for inferred or relative checkpoint paths
    models: Dict[str, ControlNetEntry] = field(default_factory=dict)
    mode: str = 'internal'  # 'internal' or 'external'
    cond_image_src: str = 'init'  # global CN source: 'init' (raw video), 'stylized', 'cond_video'
    normalize_weights: bool = False  # normalize CN weights so they sum to 1
    depth_detector: str = 'Midas'  # 'Midas' / 'Zoe' / 'depth_anything' (control_sd15_depth_detector)
    seg_detector: str = 'Seg_UFADE20K'  # 'Seg_OFADE20K' / 'Seg_OFCOCO' / 'Seg_UFADE20K' / 'Seg_SAM' (control_sd15_seg_detector)
    scribble_detector: str = 'PIDI'  # 'PIDI' / 'HED' (control_sd15_scribble_detector)
    softedge_detector: str = 'PIDI'  # 'PIDI' / 'HED' (control_sd15_softedge_detector)
    pose_include_body: bool = True   # draw body keypoints for OpenPose / DWPose
    pose_include_hand: bool = False  # detect and draw hand keypoints
    pose_include_face: bool = False  # detect and draw facial keypoints
    canny_low_threshold: int = 100   # notebook low_threshold
    canny_high_threshold: int = 200  # notebook high_threshold
    mlsd_value_threshold: float = 0.1     # notebook value_threshold
    mlsd_distance_threshold: float = 0.1  # notebook distance_threshold
    qr_cn_mask_grayscale: bool = False
    qr_cn_mask_invert: bool = False
    qr_cn_mask_thresh: int = 0
    qr_cn_mask_clip_low: int = 0
    qr_cn_mask_clip_high: int = 255
    downscale_tile: int = 4  # tile CNs: blur-via-downscale factor (notebook default)
    temporalnet_source: str = 'init'  # 'init' (prev raw frame) / 'stylized' (prev render)
    temporalnet_skip_1st_frame: bool = True  # skip temporalnet CN when no previous frame exists
    inpaint_mask_source: str = 'consistency_mask'  # 'consistency_mask' / 'none' (control_sd15_inpaint_mask_source)


@dataclass
class VideoAssemblyConfig:
    """Video post-processing and assembly settings (cell 57)."""
    keep_audio: bool = True                         # reattach audio from source video
    use_deflicker: bool = False                     # ffmpeg deflicker=mode=pm:size=10 filter
    blend_mode: str = 'optical flow'                # 'None', 'linear', or 'optical flow'
    blend: float = 0.5                              # blend factor for linear/optical-flow modes
    # WarpFusion's export-time consistency mask channel weights. These are
    # intentionally separate from the render-loop FlowConfig mask controls.
    missed_consistency_weight: float = 1.0
    overshoot_consistency_weight: float = 1.0
    edges_consistency_weight: float = 1.0
    upscale_ratio: int = 1                          # 1 = disabled; 2 or 4 via RealESRGAN
    upscale_model: str = 'realesr-animevideov3'     # RealESRGAN model name
    upscale_model_path: str = ''                    # path to .pth weights file (required when upscale_ratio>1)
    use_background_mask_video: bool = False         # composite frames onto background using alpha masks
    invert_mask_video: bool = False                 # invert the alpha mask
    background_video: str = 'init_video'            # background type: 'color', 'image', 'init_video'
    background_source_video: str = ''               # color string or image path
    mask_clip_low: int = 0                          # clip alpha mask: values below this → 0
    mask_clip_high: int = 255                       # clip alpha mask: values above this → 255


@dataclass
class FreeUConfig:
    """FreeU (notebook: do_freeunet in the control_multi UNet forward;
    VibeWarp uses that unified forward with or without selected ControlNets).
    Backbone scaling (b1/b2) + skip Fourier damping (s1/s2)."""
    do_freeunet: bool = False
    apply_freeu_after_control: bool = False  # apply after CN residual is added
    b1: float = 1.2
    b2: float = 1.4
    s1: float = 0.9
    s2: float = 0.2


@dataclass
class SceneConfig:
    """Content-aware scheduling (notebook cells 26–28).

    Templates are ``[normal_val, new_scene_val, diff_thresh, falloff_frames]``;
    empty/None templates leave the schedule untouched.
    """
    analyze_video: bool = False
    diff_function: str = 'rmse+lpips'  # 'rmse' / 'lpips' / 'rmse+lpips'
    make_schedules: bool = False
    respect_sched: bool = True  # keep existing schedules, only alter peak frames
    diff_override: List[float] = field(default_factory=list)
    # Notebook template defaults (inert while make_schedules=False)
    flow_blend_template: Any = field(default_factory=lambda: [0.8, 0.0, 0.51, 2])
    cc_masked_template: Any = field(default_factory=lambda: [0.7, 0, 0.51, 2])
    steps_template: Any = None
    style_strength_template: Any = None
    cfg_scale_template: Any = None
    latent_scale_template: Any = None
    init_scale_template: Any = None
    image_scale_template: Any = None


@dataclass
class CaptionConfig:
    """Keyframe captioning (notebook cell 30). Enables the {caption}
    placeholder in text/rec prompts."""
    make_captions: bool = False
    keyframe_source: str = 'Every n-th frame'  # / 'User-defined keyframe list' / 'Content-aware scheduling keyframes'
    nth_frame: int = 10
    user_defined_keyframes: List[int] = field(default_factory=list)
    diff_thresh: float = 0.33  # for content-aware keyframes
    offset_mode: str = 'Fixed'  # 'Fixed' / 'Between Keyframes' / 'None'
    fixed_offset: int = 0
    blip_question: str = ''  # non-empty → BLIP-VQA instead of captioning


@dataclass
class BackgroundMaskConfig:
    """Render-time background masking (notebook cell 42 — 'Video mask settings').

    Composites the frame onto a background using pre-extracted alpha masks in
    ``{video_frames_folder}_alpha/{frame_num+1:06d}.jpg`` (same convention as
    the video-assembly masking). The notebook's deprecated warp_mode='use_latent'
    latent-space compositing branch is intentionally not implemented.
    """
    use_background_mask: bool = False
    invert_mask: bool = False
    # True: mask the warped image right before it feeds the model;
    # False: mask the raw current video frame before warping.
    apply_mask_after_warp: bool = True
    background: str = 'init_video'   # 'color', 'image', 'init_video'
    background_source: str = 'red'   # color string or image path
    mask_clip_low: int = 0
    mask_clip_high: int = 255


@dataclass
class VaeConfig:
    """Tiled VAE settings for high-resolution renders."""
    use_tiled_vae: bool = False
    # num_tiles: [rows, cols] grid — tile size derived as H/rows, W/cols (in pixel space)
    # When set, takes priority over tile_size.
    num_tiles: Optional[List[int]] = None
    # tile_size: tile edge length in *latent* space (multiplied by 8 to get pixel size).
    # Ignored when num_tiles is set. Default 128 → 1024px tiles.
    tile_size: int = 128
    # vae_stride: stride between tiles in latent space. Ignored when vae_padding is set.
    vae_stride: int = 96
    # vae_padding: fractional overlap as [h_frac, w_frac] of tile_size.
    # e.g. [0.5, 0.5] → stride = tile_size * 0.5. Overrides vae_stride.
    vae_padding: Optional[List[float]] = None


@dataclass
class AnimateDiffConfig:
    """AnimateDiff motion module settings."""
    enabled: bool = False
    motion_module_path: str = ''
    batch_length: int = 32
    batch_overlap: int = 8
    context_length: int = 16
    context_overlap: int = 10
    stop_early: int = 0  # stop processing before all context windows if > 0
    blend_batch_outputs: bool = True  # blend overlapping batch outputs (notebook: blend_batch_outputs)
    overlap_stylized: bool = True  # use stylized frames at overlap boundaries (notebook: overlap_stylized)
    # Re-inject the PREVIOUS batch's rendered latents into the overlap frames on
    # every denoiser call (notebook default: True). This is what actually carries
    # identity across batch seams: with style_strength=1 the init latent is thrown
    # away (skip_steps=0), so `overlap_stylized` alone — which only picks the init
    # image — has no effect at all, and consecutive batches invent new characters.
    reinject_stylized: bool = True
    # 0 = pin the overlap frames hard. >0 ramps the last N of them back toward the
    # batch's own latent, so the anchor fades out instead of ending abruptly.
    lerp_reinject_stylized_frames: int = 0
    # Noise keyed to the ABSOLUTE frame number, so a frame gets the same noise vector
    # in every batch it appears in. This MATCHES the notebook structurally: it draws
    # big_noise once for the whole video and slices `big_noise[frame_idxs]`.
    #
    # It was previously defaulted OFF, on the belief that the notebook keys noise by
    # position-in-batch (`args.seed += 1`). That was WRONG — the seed increment drives
    # the SAMPLER's RNG in prepare_latents, not the initial noise. Verified 2026-07-13.
    #
    # Keeping it on also stops reinject_stylized re-noising the overlap frames against
    # a conflicting sample (visible as grain) and keeps un-pinned frames correlated
    # across batches (identity). Set False only to reproduce pre-2026-07-13 runs.
    #
    # NOTE: we do not reproduce the notebook's noise VALUES, because it has no RNG
    # stream to reproduce — nothing seeds the CPU generator before big_noise, so its
    # AnimateDiff output is not seed-reproducible (two same-seed notebook runs differ
    # at MAE 0.30). Ours is deterministic. For parity, inject the notebook's saved
    # noise via VIBEWARP_ADIFF_NOISE (see _adiff_injected_noise).
    frame_keyed_noise: bool = True
    # Mirror the first half of the batch's noise onto the second half, so the clip
    # loops back on itself (notebook: pingpong_noise).
    pingpong_noise: bool = False
    # Rec noise inside a batch. False (notebook default) ejects the motion module
    # while inverting, so the noise is found per-frame without temporal attention.
    batched_adiff_rec_noise: bool = False
    rec_sliding_ctx: bool = False  # use context windows during rec-noise inversion
    # Present in WarpFusion settings but commented out in the notebook itself
    # (dump line 6717) — accepted for settings compatibility, not implemented.
    looped_noise: bool = False


@dataclass
class ReconstructionNoiseConfig:
    """Reconstruction noise (predicted noise / DDIM inversion) settings.

    When enabled, reconstructs the noise that would produce each frame's init
    image via DDIM inversion. This noise is blended with random noise to improve
    temporal consistency at AnimateDiff batch boundaries.
    """
    enabled: bool = False
    randomness: float = 0.0  # blend ratio: 0 = pure reconstructed, 1 = pure random
    cfg_scale: float = 1.0  # CFG scale for noise reconstruction pass
    steps_pct: float = 1.0  # fraction of diffusion steps for reconstruction
    source: str = 'init'  # 'init' or 'stylized' — which frame to invert
    steps_cutoff: int = 0  # stop inversion early (0 = run all steps)
    noise_scale: float = 1.0  # scale factor for reconstructed noise
    separate_uncond: bool = True  # process uncond separately (faster for batched)
    disable_controlnets: bool = False  # disable controlnets during reconstruction
    cache_dir: str = ''  # directory to cache reconstructed noise tensors
    sliding_context: bool = False  # use sliding context windows during inversion (rec_sliding_ctx)
    batched: bool = False  # encode all frames at once instead of per-frame loop
    # Separate prompt schedules for reconstruction (notebook: rec_prompts, rec_neg_prompts).
    # When None, falls back to main text_prompts / negative_prompts.
    prompts: Optional[Dict[int, str]] = None
    neg_prompts: Optional[Dict[int, str]] = None


@dataclass
class IPAdapterEntry:
    """Per-IP-Adapter model configuration."""
    # Catalog model identity. The containing dict key is an instance ID, allowing
    # the same checkpoint to be applied repeatedly with independent references.
    # Empty keeps legacy configs compatible (their dict key was the model key).
    model_key: str = ''
    path: str = ''
    weight: float = 1.0
    start: float = 0.0
    end: float = 1.0
    # Unified reference selector: none, previous, warped, or upload/fixed.
    # Legacy WarpFusion strings/directories/frame schedules remain loadable.
    source_image: Any = field(
        default_factory=lambda: {'source': 'none', 'image_path': ''})
    # Ordered sources combined inside this adapter instance. Empty means use
    # source_image for backward compatibility.
    source_images: List[Any] = field(default_factory=list)
    weight_type: str = 'linear'  # linear, ease in, ease out, etc. (ip_weight_type)
    combine_embeds: str = 'concat'  # concat, add, ... (ip_combine_embeds)
    embeds_scaling: str = 'V only'  # V only, K+V, K+V w/ C penalty (ip_embeds_scaling)


@dataclass
class IPAdapterConfig:
    """IP-Adapter settings."""
    enabled: bool = False
    clip_vision_model_path: str = ''  # path to CLIP ViT-H or ViT-G
    models: Dict[str, IPAdapterEntry] = field(default_factory=dict)
    cache_embeddings: bool = True  # notebook: cache_ipadapter
    cache_size: int = 10  # notebook: ipadapter_embeds_cache_size
    flip_uc: bool = False  # notebook: ip_flip_uc — flip the uncond mask in hooks
    # Address the layer-weight presets the way VibeWarp did before 0.7.1: by a
    # block's index within its own list rather than by execution order. That
    # numbering is wrong (it collides input against output blocks and never
    # reaches the last few entries) and matches no other implementation, so it
    # exists only to reproduce renders made with an older version. It affects
    # every weight_type that consults the index — the ease/reverse family and
    # all the style/composition presets — but not 'linear'.
    legacy_layer_indexing: bool = False


@dataclass
class EditReferenceConfig:
    """One ordered visual reference supplied to an image-edit model."""
    source: str = 'none'
    label: bool = False
    image_path: str = ''


@dataclass
class ContactSheetConfig:
    """Experimental joint-frame rendering for ComfyUI image-edit models."""
    mode: str = 'off'
    layout: str = 'adaptive'
    gutter: int = 4
    instruction: str = ''
    save_debug: bool = True


def _default_edit_references() -> List[EditReferenceConfig]:
    # Slot 1 is mandatory. Slot 2 is the visible no-op that lets the UI grow
    # the ordered list one card at a time.
    return [
        EditReferenceConfig(source='raw'),
        EditReferenceConfig(source='none'),
    ]


FLUX_MODEL_DEFAULTS = {
    'flux2_klein_edit': {
        'comfy_unet_name': 'flux-2-klein-4b-fp8.safetensors',
        'comfy_clip_name': 'qwen_3_4b.safetensors',
        'comfy_vae_name': 'flux2-vae.safetensors',
        'model_repo': 'black-forest-labs/FLUX.2-klein-4B',
    },
    'flux2_klein_9b_edit': {
        'comfy_unet_name': 'flux-2-klein-9b-fp8.safetensors',
        'comfy_clip_name': 'qwen_3_8b_fp8mixed.safetensors',
        'comfy_vae_name': 'flux2-vae.safetensors',
        'model_repo': 'black-forest-labs/FLUX.2-klein-9B',
    },
}


def flux_model_files(config: 'RunConfig') -> Dict[str, str]:
    """Resolve model-specific defaults while preserving explicit custom files."""
    baseline = FLUX_MODEL_DEFAULTS['flux2_klein_edit']
    selected = FLUX_MODEL_DEFAULTS.get(config.model_version, baseline)
    resolved = {}
    for key in baseline:
        current = getattr(config.flux, key)
        known_defaults = {
            defaults[key] for defaults in FLUX_MODEL_DEFAULTS.values()}
        resolved[key] = selected[key] if current in known_defaults else current
    return resolved


@dataclass
class FluxConfig:
    """FLUX.2 Klein Edit through ComfyUI or the optional Diffusers backend.

    References use the same ordered source list as HiDream. Dynamic temporal
    sources fall back to the raw current frame when no previous render exists.
    """
    # Comfy is the exploratory/default backend because it directly supports the
    # split FP8/GGUF ecosystem checkpoints. Diffusers remains available for a
    # complete Hugging Face diffusers-format repository.
    backend: str = 'comfy'
    comfy_server_url: str = 'http://127.0.0.1:8188'
    comfy_unet_name: str = 'flux-2-klein-4b-fp8.safetensors'
    comfy_clip_name: str = 'qwen_3_4b.safetensors'
    comfy_vae_name: str = 'flux2-vae.safetensors'
    comfy_timeout: float = 600.0
    model_repo: str = 'black-forest-labs/FLUX.2-klein-4B'
    # Legacy field retained while the experimental Diffusers backend remains.
    # Klein reference editing through Comfy does not expose img2img strength.
    denoise: float = 1.0
    guidance_scale: float = 1.0
    steps: int = 4
    # Reference editing starts every frame from noise. Reusing that noise is
    # important for temporal stability; opt out only for deliberate variation.
    fixed_seed: bool = True
    references: List[EditReferenceConfig] = field(
        default_factory=_default_edit_references)
    multi_reference_instruction: str = (
        'Use reference image 1 for content, composition, geometry, objects, and '
        'motion. Use the remaining reference images for visual style and '
        'temporal continuity.'
    )
    # Prompt token budget used only by the Diffusers backend.
    max_sequence_length: int = 512


@dataclass
class HiDreamConfig:
    """HiDream-O1 pixel-space instruction editing through ComfyUI.

    Ordered references share the same source controls as Flux. The model starts
    from fixed pixel-space noise; ``fixed_seed`` controls whether that noise is
    reused across the video.
    """
    comfy_server_url: str = 'http://127.0.0.1:8188'
    comfy_checkpoint_name: str = 'hidream_o1_image_fp8_scaled.safetensors'
    comfy_timeout: float = 1200.0
    # Defaults mirror ComfyUI's native Full workflow. ModelNoiseScale is not a
    # cosmetic tuning knob: Full was trained at absolute noise scale 8.
    steps: int = 40
    guidance_scale: float = 5.0
    sampler: str = 'dpmpp_2m_sde_gpu'
    scheduler: str = 'normal'
    noise_scale: float = 8.0
    # O1 is sharply resolution-dependent: 1024x1024 produces noise-like output.
    # Render at the nearest ~4MP training preset, then downsample to video size.
    use_trained_resolution: bool = True
    fixed_seed: bool = True
    references: List[EditReferenceConfig] = field(
        default_factory=_default_edit_references)
    multi_reference_instruction: str = (
        'Use reference image 1 for content, composition, geometry, objects, and '
        'motion. Use the remaining reference images for visual style and '
        'temporal continuity.'
    )
    # Full's official workflow smooths 32-pixel patch-grid seams during the
    # final sampling phase. This costs extra model evaluations only near the end.
    patch_seam_smoothing: bool = True
    patch_seam_start: float = 0.8
    patch_seam_passes: str = 'ramp_2_4'
    patch_seam_blend: str = 'median'


QWEN_MODEL_DEFAULTS = {
    'qwen_image_edit_2511': {
        'comfy_unet_name': 'qwen_image_edit_2511_int8_convrot.safetensors',
    },
    'qwen_image_edit_2511_gguf': {
        'comfy_unet_name': 'Qwen-Image-Edit-2511-Q5_K_M.gguf',
    },
}


def qwen_model_files(config: 'RunConfig') -> Dict[str, str]:
    """Resolve the Qwen transformer preset without replacing a custom file."""
    baseline = QWEN_MODEL_DEFAULTS['qwen_image_edit_2511']
    selected = QWEN_MODEL_DEFAULTS.get(config.model_version, baseline)
    resolved = {}
    for key in baseline:
        current = getattr(config.qwen, key)
        known_defaults = {
            defaults[key] for defaults in QWEN_MODEL_DEFAULTS.values()}
        resolved[key] = selected[key] if current in known_defaults else current
    return resolved


@dataclass
class QwenEditConfig:
    """Qwen-Image-Edit-2511 through ComfyUI's native edit nodes."""
    comfy_server_url: str = 'http://127.0.0.1:8188'
    comfy_unet_name: str = 'qwen_image_edit_2511_int8_convrot.safetensors'
    comfy_clip_name: str = 'qwen_2.5_vl_7b_fp8_scaled.safetensors'
    comfy_vae_name: str = 'qwen_image_vae.safetensors'
    comfy_lora_name: str = (
        'Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors')
    comfy_timeout: float = 1200.0
    use_lightning_lora: bool = True
    lora_strength: float = 1.0
    steps: int = 8
    guidance_scale: float = 1.0
    sampler: str = 'euler'
    scheduler: str = 'simple'
    sampling_shift: float = 3.1
    fixed_seed: bool = True
    references: List[EditReferenceConfig] = field(
        default_factory=_default_edit_references)
    multi_reference_instruction: str = (
        'Preserve the content, composition, geometry, identity, and pose from '
        '@Image1. Apply only the visual style from the remaining reference images.'
    )


MAGE_MODEL_DEFAULTS = {
    'mage_flow_edit': {
        'comfy_unet_name': 'mage_flow_edit_int8_convrot.safetensors',
        'steps': 30,
        'guidance_scale': 5.0,
    },
    'mage_flow_edit_turbo': {
        'comfy_unet_name': 'mage_flow_edit_turbo_int8_convrot.safetensors',
        'steps': 4,
        'guidance_scale': 1.0,
    },
}


def mage_model_defaults(config: 'RunConfig') -> Dict[str, object]:
    """Resolve the Mage checkpoint/sampling preset without replacing custom values."""
    baseline = MAGE_MODEL_DEFAULTS['mage_flow_edit']
    selected = MAGE_MODEL_DEFAULTS.get(config.model_version, baseline)
    resolved = {}
    for key in baseline:
        current = getattr(config.mage, key)
        known_defaults = {
            defaults[key] for defaults in MAGE_MODEL_DEFAULTS.values()}
        resolved[key] = selected[key] if current in known_defaults else current
    return resolved


@dataclass
class MageFlowEditConfig:
    """Mage-Flow-Edit through ComfyUI's native multi-image edit node."""
    comfy_server_url: str = 'http://127.0.0.1:8188'
    comfy_unet_name: str = 'mage_flow_edit_int8_convrot.safetensors'
    comfy_clip_name: str = 'qwen3vl_4b_bf16.safetensors'
    comfy_vae_name: str = 'mage_flow_vae_bf16.safetensors'
    comfy_timeout: float = 1200.0
    steps: int = 30
    guidance_scale: float = 5.0
    sampler: str = 'euler'
    scheduler: str = 'simple'
    fixed_seed: bool = True
    references: List[EditReferenceConfig] = field(
        default_factory=_default_edit_references)
    multi_reference_instruction: str = (
        'Use @Image1 as the primary source for content, composition, geometry, '
        'identity, and pose. Use the remaining reference images only for the '
        'requested appearance or visual style.'
    )


@dataclass
class RunConfig:
    """Top-level run configuration combining all sub-configs."""
    # Paths
    root_dir: str = field(default_factory=os.getcwd)
    model_path: str = ''
    output_dir: str = 'images_out'
    batch_name: str = 'warpfusion'

    # Model
    model_version: str = 'control_multi_v15'
    sd_checkpoint_path: str = ''

    # Prompts
    text_prompts: Dict[int, str] = field(default_factory=lambda: {0: "a beautiful painting"})
    negative_prompts: Dict[int, str] = field(default_factory=lambda: {0: ""})
    # Shared alpha for every optional @ImageN reference label.
    reference_label_opacity: float = 0.7

    # LORA directory (for per-frame LORA scheduling via <lora:name:weight> in prompts)
    lora_dir: str = ''
    # LORA merge precision: 'fp16' replicates the notebook's A1111 merge
    # arithmetic exactly (required for output parity); 'fp32' computes more
    # accurate deltas (rounded once at apply) at the cost of tiny deviations
    # from notebook renders.
    lora_merge_precision: str = 'fp16'

    # Frames
    frame_range: List[int] = field(default_factory=lambda: [0, 0])
    # Interpret frame-keyed schedules relative to frame_range[0]. For example,
    # key 0 targets source frame 60 when the selected range starts at 60.
    keyframes_relative_to_frame_range: bool = False
    animation_mode: str = 'Video Input'

    # Sub-configs
    flow: FlowConfig = field(default_factory=FlowConfig)
    warp: WarpConfig = field(default_factory=WarpConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    brightness: BrightnessConfig = field(default_factory=BrightnessConfig)
    color: ColorConfig = field(default_factory=ColorConfig)
    controlnet: ControlNetConfig = field(default_factory=ControlNetConfig)
    vae: VaeConfig = field(default_factory=VaeConfig)
    mask: BackgroundMaskConfig = field(default_factory=BackgroundMaskConfig)
    captions: CaptionConfig = field(default_factory=CaptionConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    freeu: FreeUConfig = field(default_factory=FreeUConfig)
    video_assembly: VideoAssemblyConfig = field(default_factory=VideoAssemblyConfig)
    animatediff: AnimateDiffConfig = field(default_factory=AnimateDiffConfig)
    ipadapter: IPAdapterConfig = field(default_factory=IPAdapterConfig)
    reconstruction_noise: ReconstructionNoiseConfig = field(default_factory=ReconstructionNoiseConfig)
    contact_sheet: ContactSheetConfig = field(default_factory=ContactSheetConfig)
    flux: FluxConfig = field(default_factory=FluxConfig)
    hidream: HiDreamConfig = field(default_factory=HiDreamConfig)
    qwen: QwenEditConfig = field(default_factory=QwenEditConfig)
    mage: MageFlowEditConfig = field(default_factory=MageFlowEditConfig)
