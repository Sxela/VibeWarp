"""Canonical ControlNet catalog: the single source of truth for which nets exist.

Before this module the same facts were spread across three partial tables that
drifted apart: ``ANNOTATOR_MAP`` in ``core/controlnet.py`` (29 keys), a local
``cn_file_map`` in ``settings.py`` (14 filenames, one of them wrong), and
``CONTROLNET_MODEL_URLS`` in ``config.py`` (12 URLs). The Svelte UI then
hardcoded a fourth list of 15. Those consumers now derive from the catalog here.

Keys, URLs and filenames mirror the notebook's ``control_model_urls`` /
``control_model_filenames`` (refs/notebook_dump.txt). Many checkpoints download
as ``diffusion_pytorch_model.safetensors``, which is why an explicit rename map
exists: ``filename`` is what the file is called *on disk*, not in the URL.

Only nets the render engine can actually drive are listed — the notebook also
ships SD2.1 and control-LoRA URLs that VibeWarp has no loader for, and listing
them would offer the UI a net that cannot render.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Detector/annotator settings on ControlNetConfig that apply to a given net.
# These are GLOBAL (one value shared by every net using that annotator), not
# per-entry — the UI must say so, because editing them from inside one net's
# card silently changes the others.
_DEPTH = ('depth_detector',)
_CANNY = ('canny_low_threshold', 'canny_high_threshold')
_SEG = ('seg_detector',)
_SCRIBBLE = ('scribble_detector',)
_SOFTEDGE = ('softedge_detector',)
_POSE = ('pose_include_body', 'pose_include_hand', 'pose_include_face')
_MLSD = ('mlsd_value_threshold', 'mlsd_distance_threshold')
_TILE = ('downscale_tile',)
_QR = ('qr_cn_mask_grayscale', 'qr_cn_mask_invert', 'qr_cn_mask_thresh',
       'qr_cn_mask_clip_low', 'qr_cn_mask_clip_high')
_TEMPORAL = ('temporalnet_source', 'temporalnet_skip_1st_frame')
_INPAINT = ('inpaint_mask_source',)

_HF = 'https://huggingface.co'
_LLYAS = f'{_HF}/lllyasviel/ControlNet-v1-1/resolve/main'


@dataclass(frozen=True)
class ControlNetSpec:
    """One ControlNet the engine can drive."""
    key: str
    label: str
    model_version: str          # 'sd15' | 'sd21' | 'sdxl'
    annotator: str              # dispatch key into the annotator registry
    filename: str = ''          # expected name on disk ('' = user must pick a file)
    url: str = ''               # download source ('' = not auto-downloadable)
    detectors: Tuple[str, ...] = ()  # global ControlNetConfig fields that affect it


def _spec(key, label, version, annotator, filename='', url='', detectors=()):
    return ControlNetSpec(key, label, version, annotator, filename, url, detectors)


CONTROLNET_CATALOG: Dict[str, ControlNetSpec] = {
    spec.key: spec for spec in [
        # ---- SD 1.5 (lllyasviel ControlNet v1.1) ----
        _spec('control_sd15_canny', 'Canny', 'sd15', 'canny',
              'control_v11p_sd15_canny.pth', f'{_LLYAS}/control_v11p_sd15_canny.pth', _CANNY),
        _spec('control_sd15_depth', 'Depth', 'sd15', 'depth',
              'control_v11f1p_sd15_depth.pth', f'{_LLYAS}/control_v11f1p_sd15_depth.pth', _DEPTH),
        _spec('control_sd15_softedge', 'Soft Edge', 'sd15', 'softedge',
              'control_v11p_sd15_softedge.pth', f'{_LLYAS}/control_v11p_sd15_softedge.pth', _SOFTEDGE),
        _spec('control_sd15_scribble', 'Scribble', 'sd15', 'scribble',
              'control_v11p_sd15_scribble.pth', f'{_LLYAS}/control_v11p_sd15_scribble.pth', _SCRIBBLE),
        _spec('control_sd15_openpose', 'OpenPose', 'sd15', 'openpose',
              'control_v11p_sd15_openpose.pth', f'{_LLYAS}/control_v11p_sd15_openpose.pth',
              _POSE),
        _spec('control_sd15_lineart', 'Line Art', 'sd15', 'lineart',
              'control_v11p_sd15_lineart.pth', f'{_LLYAS}/control_v11p_sd15_lineart.pth'),
        # NOTE: the real file is "sd15s2", not "sd15" — settings.py had this wrong,
        # so the net could never be resolved from model_dir.
        _spec('control_sd15_lineart_anime', 'Anime Line Art', 'sd15', 'lineart_anime',
              'control_v11p_sd15s2_lineart_anime.pth', f'{_LLYAS}/control_v11p_sd15s2_lineart_anime.pth'),
        _spec('control_sd15_normalbae', 'Normal BAE', 'sd15', 'normalbae',
              'control_v11p_sd15_normalbae.pth', f'{_LLYAS}/control_v11p_sd15_normalbae.pth'),
        _spec('control_sd15_mlsd', 'MLSD', 'sd15', 'mlsd',
              'control_v11p_sd15_mlsd.pth', f'{_LLYAS}/control_v11p_sd15_mlsd.pth', _MLSD),
        _spec('control_sd15_seg', 'Segmentation', 'sd15', 'seg',
              'control_v11p_sd15_seg.pth', f'{_LLYAS}/control_v11p_sd15_seg.pth', _SEG),
        _spec('control_sd15_shuffle', 'Shuffle', 'sd15', 'shuffle',
              'control_v11e_sd15_shuffle.pth', f'{_LLYAS}/control_v11e_sd15_shuffle.pth'),
        _spec('control_sd15_ip2p', 'Instruct Pix2Pix', 'sd15', 'ip2p',
              'control_v11e_sd15_ip2p.pth', f'{_LLYAS}/control_v11e_sd15_ip2p.pth'),
        _spec('control_sd15_tile', 'Tile', 'sd15', 'tile',
              'control_v11f1e_sd15_tile.pth', f'{_LLYAS}/control_v11f1e_sd15_tile.pth', _TILE),
        _spec('control_sd15_inpaint', 'Inpaint', 'sd15', 'inpaint',
              'control_v11p_sd15_inpaint.pth', f'{_LLYAS}/control_v11p_sd15_inpaint.pth', _INPAINT),
        # ---- SD 1.5 (third-party) ----
        _spec('control_sd15_temporalnet', 'TemporalNet', 'sd15', 'temporalnet',
              'diff_control_sd15_temporalnet_fp16.safetensors',
              f'{_HF}/CiaraRowles/TemporalNet/resolve/main/diff_control_sd15_temporalnet_fp16.safetensors',
              _TEMPORAL),
        _spec('control_sd15_depth_anything', 'Depth Anything', 'sd15', 'depth',
              'control_sd15_depth_anything.safetensors',
              f'{_HF}/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_controlnet/diffusion_pytorch_model.safetensors',
              _DEPTH),
        _spec('control_sd15_qr', 'QR Code', 'sd15', 'tile',
              'control_v1p_sd15_qrcode.safetensors',
              f'{_HF}/DionTimmer/controlnet_qrcode/resolve/main/control_v1p_sd15_qrcode.safetensors', _QR),
        _spec('control_sd15_monster_qr', 'QR Monster', 'sd15', 'tile',
              'control_v1p_sd15_qrcode_monster_v2.safetensors',
              f'{_HF}/monster-labs/control_v1p_sd15_qrcode_monster/resolve/main/v2/control_v1p_sd15_qrcode_monster_v2.safetensors',
              _QR),
        _spec('control_sd15_gif', 'AnimateDiff GIF', 'sd15', 'tile',
              'control_v11p_sd15_gif.ckpt',
              f'{_HF}/crishhh/animatediff_controlnet/resolve/main/controlnet_checkpoint.ckpt', _TILE),
        # ---- SD 2.1 ----
        _spec('control_sd21_qr', 'QR Code', 'sd21', 'tile',
              'control_v11p_sd21_qrcode.safetensors',
              f'{_HF}/DionTimmer/controlnet_qrcode/resolve/main/control_v11p_sd21_qrcode.safetensors', _QR),
        # ---- SDXL ----
        _spec('control_sdxl_canny', 'Canny', 'sdxl', 'canny',
              'diffusers-controlnet-canny-sdxl-1.0.fp16.safetensors',
              f'{_HF}/diffusers/controlnet-canny-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors',
              _CANNY),
        _spec('control_sdxl_depth', 'Depth', 'sdxl', 'depth',
              'diffusers-controlnet-depth-sdxl-1.0.fp16.safetensors',
              f'{_HF}/diffusers/controlnet-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors',
              _DEPTH),
        _spec('control_sdxl_softedge', 'Soft Edge', 'sdxl', 'softedge',
              'SargeZT-controlnet-sd-xl-1.0-softedge-dexined.safetensors',
              f'{_HF}/SargeZT/controlnet-sd-xl-1.0-softedge-dexined/resolve/main/controlnet-sd-xl-1.0-softedge-dexined.safetensors',
              _SOFTEDGE),
        _spec('control_sdxl_openpose', 'OpenPose', 'sdxl', 'openpose',
              'thibaud-OpenPoseXL2.safetensors',
              f'{_HF}/thibaud/controlnet-openpose-sdxl-1.0/resolve/main/OpenPoseXL2.safetensors',
              _POSE),
        _spec('control_sdxl_inpaint', 'Inpaint', 'sdxl', 'inpaint',
              'control_v01e_sdxl_inpaint_diffusers_15k.safetensors',
              f'{_HF}/sxela/out/resolve/main/checkpoint-15000/controlnet/diffusion_pytorch_model.safetensors',
              _INPAINT),
        _spec('control_sdxl_temporalnet_v1', 'TemporalNet v1', 'sdxl', 'temporalnet',
              'CiaraRowles-temporalnet-sdxl-v1.safetensors',
              f'{_HF}/CiaraRowles/controlnet-temporalnet-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors',
              _TEMPORAL),
        _spec('control_sdxl_xinsir_tile', 'Tile (xinsir)', 'sdxl', 'tile',
              'control_sdxl_xinsir_tile.safetensors',
              f'{_HF}/xinsir/controlnet-tile-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors',
              _TILE),
        # No published checkpoint the notebook agrees on — user supplies the file.
        _spec('control_sdxl_lineart', 'Line Art', 'sdxl', 'lineart'),
        _spec('control_sdxl_tile', 'Tile', 'sdxl', 'tile', detectors=_TILE),
    ]
}

# ---- Derived views (consumers import these instead of re-declaring) ----

def annotator_map() -> Dict[str, str]:
    """ControlNet key -> annotator dispatch string."""
    return {key: spec.annotator for key, spec in CONTROLNET_CATALOG.items()}


def controlnet_filenames() -> Dict[str, str]:
    """ControlNet key -> expected checkpoint filename on disk (known ones only)."""
    return {key: spec.filename for key, spec in CONTROLNET_CATALOG.items() if spec.filename}


def controlnet_urls() -> Dict[str, str]:
    """ControlNet key -> download URL (downloadable ones only)."""
    return {key: spec.url for key, spec in CONTROLNET_CATALOG.items() if spec.url}


def specs_for_version(model_version: str = '') -> List[ControlNetSpec]:
    """Nets usable with a given base model. Empty version returns everything."""
    version = _normalize_version(model_version)
    specs = list(CONTROLNET_CATALOG.values())
    if version:
        specs = [spec for spec in specs if spec.model_version == version]
    return sorted(specs, key=lambda spec: spec.label)


def normalize_model_version(model_version: str) -> str:
    """Public alias — callers outside this module need the family, too."""
    return _normalize_version(model_version)


def _normalize_version(model_version: str) -> str:
    """Map a RunConfig model_version to a family.

    Real values carry separators ('control_multi_v15', 'v1_5', 'v2_1'), so strip
    them before matching — 'v1_5' contains neither '15' nor '1.5' literally.
    """
    value = (model_version or '').lower()
    for separator in ('_', '.', '-', ' '):
        value = value.replace(separator, '')
    if 'xl' in value:
        return 'sdxl'
    if '21' in value:
        return 'sd21'
    if '15' in value:
        return 'sd15'
    return ''


# ---- Mode presets ----
#
# The notebook derives per-layer weights and zero_uncond from `mode` — they are
# OUTPUTS of the mode, not independent settings (refs/notebook_dump.txt, the
# controlnet_multimodel_inferred block). This is the A1111 "ControlNet is more
# important" / "My prompt is more important" soft-weighting curve.

CUSTOM_MODE = 'custom'
NUM_CN_LAYERS = 13  # SD1.5: 12 encoder + 1 middle

_SOFT_CURVE = [0.825 ** float(12 - i) for i in range(NUM_CN_LAYERS)]

MODE_PRESETS: Dict[str, Tuple[List[float], bool]] = {
    'balanced': ([1.0] * NUM_CN_LAYERS, False),
    'controlnet': (list(_SOFT_CURVE), True),
    'prompt': (list(_SOFT_CURVE), False),
}

CONTROLNET_MODES = (*MODE_PRESETS, CUSTOM_MODE)


def resolve_mode(mode: str) -> Optional[Tuple[List[float], bool]]:
    """Layer weights + zero_uncond implied by `mode`, or None to keep the entry's own.

    Returns None for 'custom' (and anything unrecognised) so hand-authored
    layer_weights/zero_uncond survive.
    """
    preset = MODE_PRESETS.get(mode)
    if preset is None:
        return None
    weights, zero_uncond = preset
    return list(weights), zero_uncond
