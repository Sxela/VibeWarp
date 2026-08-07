"""ControlNet annotation and conditioning pipeline.

Extracted from notebook's get_controlnet_annotations() (cell 20 ~line 3463).
Handles: source image → annotator preprocessing → conditioning tensor.

All annotation goes through the controlnet-aux PyPI package.
Detectors are cached after first load to avoid re-downloading per frame.
"""

import os
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from vibewarp.controlnet_catalog import annotator_map


# ---- Annotator dispatch ----

# Cached annotator instances (avoid reloading models per frame)
_annotator_cache: Dict[str, Any] = {}

# Mapping from ControlNet model key to annotator type string.
# Derived from the catalog so the key list has exactly one definition.
ANNOTATOR_MAP = annotator_map()

# Tile CN keys that get the downscale_tile blur (qr/gif keys do NOT — the
# notebook applies downscale only to actual tile models)
TILE_CN_KEYS = {'control_sd15_tile', 'control_sdxl_tile', 'control_sdxl_xinsir_tile'}

# Annotator types that use the source image as-is (no model inference).
# temporalnet's conditioning IS the previous frame — source selection happens
# in the frame loop (temporalnet_source setting), annotation is passthrough.
# ('tile' is handled by _annotate_tile: passthrough + qr/tile mask options.)
_PASSTHROUGH_ANNOTATORS = {'ip2p', 'temporalnet'}

# CN model keys that consume the inpaint-marker conditioning
INPAINT_CN_KEYS = {'control_sd15_inpaint', 'control_sdxl_inpaint'}


def annotate_image(
    source_image: np.ndarray,
    annotator_type: str,
    resolution: int = 512,
    depth_detector: str = 'Midas',
    colored_depth: bool = False,
    inpaint_mask: Optional[np.ndarray] = None,
    seg_detector: str = 'Seg_UFADE20K',
    opts: Optional[dict] = None,
) -> np.ndarray:
    """Run a ControlNet annotator on a source image.

    Args:
        source_image: HWC uint8 numpy array (RGB)
        annotator_type: annotator name ('canny', 'depth', 'openpose', etc.)
        resolution: processing resolution
        depth_detector: for 'depth' — 'Midas', 'Zoe', or 'depth_anything'
            (notebook's control_sd15_depth_detector)
        seg_detector: for 'seg' — 'Seg_OFADE20K', 'Seg_OFCOCO',
            'Seg_UFADE20K' (notebook's control_sd15_seg_detector; UniFormer
            isn't on PyPI, so UFADE20K routes to OneFormer-ADE20K — same 150
            classes and palette), or 'Seg_SAM' (non-notebook: class-agnostic
            random-colored instance masks — avoid for control_sd15_seg)
        colored_depth: apply INFERNO heatmap to the depth map — the notebook
            does this whenever the CN model itself is a depth_anything model
            ('depth_anything' in control_key), regardless of detector
        inpaint_mask: for 'inpaint' — consistency/keep mask, HW or HWC,
            float 0-1 or uint8 (1/255 = keep, 0 = regenerate). Required;
            callers skip the CN entirely when no mask is available (notebook
            drops the model for the frame).
        opts: per-run annotator options dict (built by
            prepare_cn_hints_for_frame from ControlNetConfig):
            canny_low/high thresholds, mlsd value/distance thresholds,
            scribble_detector / softedge_detector ('PIDI'/'HED'),
            pose_include_body / pose_include_hand / pose_include_face,
            qr_cn_mask_* + downscale_tile (+ 'is_tile_model' per CN key),
            'target_hw' for post-passes that run at render resolution.

    Returns:
        Annotated map as HWC numpy array — uint8, except 'inpaint' which
        returns float32 with -255.0 markers (→ -1.0 after /255 in
        map_to_conditioning, the notebook's inpaint marker value).
    """
    opts = opts or {}
    if annotator_type == 'canny':
        return _annotate_canny(source_image, resolution,
                               low=opts.get('canny_low_threshold', 100),
                               high=opts.get('canny_high_threshold', 200))
    elif annotator_type == 'tile':
        return _annotate_tile(source_image, opts)
    elif annotator_type in _PASSTHROUGH_ANNOTATORS:
        return source_image
    elif annotator_type == 'inpaint':
        if inpaint_mask is None:
            raise ValueError(
                "annotator 'inpaint' requires inpaint_mask — callers must skip "
                "the inpaint CN for frames without a mask (notebook behavior)")
        return _annotate_inpaint(source_image, inpaint_mask)
    elif annotator_type == 'depth':
        depth_map = _annotate_depth(source_image, resolution, depth_detector)
        if colored_depth:
            # Notebook: heatmap any depth map for depth_anything CN models
            gray = depth_map[..., 0] if depth_map.ndim == 3 else depth_map
            depth_map = cv2.applyColorMap(gray.astype(np.uint8),
                                          cv2.COLORMAP_INFERNO)[:, :, ::-1]
        return depth_map
    elif annotator_type == 'softedge':
        if opts.get('softedge_detector', 'PIDI') == 'HED':
            return _run_detector('softedge_hed', _load_hed, source_image, resolution)
        return _run_detector('softedge', _load_pidinet, source_image, resolution)
    elif annotator_type == 'scribble':
        return _annotate_scribble(source_image, resolution, opts)
    elif annotator_type == 'lineart':
        return _run_detector('lineart', _load_lineart, source_image, resolution)
    elif annotator_type == 'lineart_anime':
        return _run_detector('lineart_anime', _load_lineart_anime, source_image, resolution)
    elif annotator_type == 'mlsd':
        return _run_detector(
            'mlsd', _load_mlsd, source_image, resolution,
            thr_v=opts.get('mlsd_value_threshold', 0.1),
            thr_d=opts.get('mlsd_distance_threshold', 0.1))
    elif annotator_type == 'openpose':
        return _run_detector(
            'openpose', _load_openpose, source_image, resolution,
            include_body=opts.get('pose_include_body', True),
            include_hand=opts.get('pose_include_hand', False),
            include_face=opts.get('pose_include_face', False))
    elif annotator_type == 'dwpose':
        return _run_dwpose(
            source_image, resolution,
            include_body=opts.get('pose_include_body', True),
            include_hand=opts.get('pose_include_hand', False),
            include_face=opts.get('pose_include_face', False))
    elif annotator_type == 'shuffle':
        return _run_detector('shuffle', _load_shuffle, source_image, resolution)
    elif annotator_type == 'normalbae':
        # Notebook BGR-flips the normalbae map before feeding the CN;
        # controlnet_aux emits the same array as the reference detector
        # (verified identical code), so the flip applies here too.
        detected = _run_detector('normalbae', _load_normalbae, source_image, resolution)
        return detected[:, :, ::-1]
    elif annotator_type == 'seg':
        if seg_detector == 'Seg_SAM':
            return _run_detector('seg_sam', _load_sam, source_image, resolution)
        dataset = 'coco' if seg_detector == 'Seg_OFCOCO' else 'ade20k'
        return _annotate_oneformer(source_image, resolution, dataset)
    else:
        raise ValueError(f"Unknown annotator type: '{annotator_type}'")


def clear_annotator_cache() -> None:
    """Release all cached annotator models from memory."""
    _annotator_cache.clear()


# ---- Cached detector helpers ----

def _to_cuda(detector) -> None:
    """Move a controlnet_aux detector's inner model(s) to CUDA.

    controlnet_aux detectors are wrapper objects, not nn.Modules.  Each has
    one or more nn.Module attributes under different names depending on the
    detector type.  We probe the common attribute names and move whatever we
    find — this covers MiDaS, PidiNet, HED, Lineart, Openpose, etc.
    """
    import torch.nn as nn
    for attr in ('model', 'netNetwork', 'net', 'body_estimation',
                 'hand_estimation', 'face_estimation'):
        m = getattr(detector, attr, None)
        if isinstance(m, nn.Module):
            m.cuda()
    if isinstance(detector, nn.Module):
        detector.cuda()
    elif torch.cuda.is_available() and callable(getattr(detector, 'to', None)):
        # DWPose owns its modules behind a Wholebody wrapper rather than one of
        # the conventional attributes above.
        detector.to('cuda')


def _run_detector(
    cache_key: str,
    loader_fn,
    image: np.ndarray,
    resolution: int,
    **det_kwargs,
) -> np.ndarray:
    """Load detector (cached on GPU), run on image, return HWC uint8 array."""
    if cache_key not in _annotator_cache:
        import time
        from vibewarp.core.timing import log_time
        t0 = time.time()
        detector = loader_fn()
        _to_cuda(detector)
        _annotator_cache[cache_key] = detector
        log_time(f"annotator {cache_key}", time.time() - t0)

    import time
    from vibewarp.core.timing import log_time
    detector = _annotator_cache[cache_key]
    pil_img = Image.fromarray(image)
    t1 = time.time()
    result = detector(pil_img, detect_resolution=resolution,
                      image_resolution=resolution, **det_kwargs)
    log_time(f"annotator {cache_key} inference", time.time() - t1)
    return np.array(result)


def _run_dwpose(
    image: np.ndarray,
    resolution: int,
    *,
    include_body: bool = True,
    include_hand: bool = False,
    include_face: bool = False,
) -> np.ndarray:
    """Run DWPose and selectively draw body, hand, and face landmarks.

    controlnet_aux 0.0.10 accepts arbitrary kwargs in ``DWposeDetector`` but
    ignores them and always draws every landmark group. Rendering the detected
    groups here gives DWPose the same meaningful switches as OpenPose.
    """
    cache_key = 'dwpose'
    if cache_key not in _annotator_cache:
        import time
        from vibewarp.core.timing import log_time
        t0 = time.time()
        detector = _load_dwpose()
        _to_cuda(detector)
        _annotator_cache[cache_key] = detector
        log_time(f"annotator {cache_key}", time.time() - t0)

    import time
    from controlnet_aux.dwpose import util as dwpose_util
    from controlnet_aux.util import HWC3, resize_image
    from vibewarp.core.timing import log_time

    detector = _annotator_cache[cache_key]
    detector_input = cv2.cvtColor(
        np.asarray(Image.fromarray(image), dtype=np.uint8), cv2.COLOR_RGB2BGR)
    detector_input = resize_image(HWC3(detector_input), resolution)
    height, width = detector_input.shape[:2]

    t1 = time.time()
    with torch.no_grad():
        candidate, subset = detector.pose_estimation(detector_input)
    log_time(f"annotator {cache_key} inference", time.time() - t1)

    candidate = candidate.copy()
    subset = subset.copy()
    candidate[..., 0] /= float(width)
    candidate[..., 1] /= float(height)
    candidate[subset < 0.3] = -1

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if include_body:
        people, _, values = candidate.shape
        body = candidate[:, :18].copy().reshape(people * 18, values)
        body_subset = subset[:, :18].copy()
        for person in range(len(body_subset)):
            for point in range(len(body_subset[person])):
                body_subset[person, point] = (
                    18 * person + point
                    if body_subset[person, point] > 0.3 else -1)
        canvas = dwpose_util.draw_bodypose(canvas, body, body_subset)
    if include_hand:
        hands = np.vstack((candidate[:, 92:113], candidate[:, 113:]))
        canvas = dwpose_util.draw_handpose(canvas, hands)
    if include_face:
        canvas = dwpose_util.draw_facepose(canvas, candidate[:, 24:92])

    target = resize_image(detector_input, resolution)
    return cv2.resize(
        HWC3(canvas), (target.shape[1], target.shape[0]),
        interpolation=cv2.INTER_LINEAR)


def _load_midas():
    from controlnet_aux import MidasDetector
    return MidasDetector.from_pretrained("lllyasviel/Annotators")


def _load_pidinet():
    from controlnet_aux import PidiNetDetector
    return PidiNetDetector.from_pretrained("lllyasviel/Annotators")


def _load_hed():
    from controlnet_aux import HEDdetector
    return HEDdetector.from_pretrained("lllyasviel/Annotators")


def _load_lineart():
    from controlnet_aux import LineartDetector
    return LineartDetector.from_pretrained("lllyasviel/Annotators")


def _load_lineart_anime():
    from controlnet_aux import LineartAnimeDetector
    return LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")


def _load_mlsd():
    from controlnet_aux import MLSDdetector
    return MLSDdetector.from_pretrained("lllyasviel/Annotators")


def _load_openpose():
    from controlnet_aux import OpenposeDetector
    return OpenposeDetector.from_pretrained("lllyasviel/Annotators")


def _load_dwpose():
    from controlnet_aux import DWposeDetector
    return DWposeDetector.from_pretrained("lllyasviel/Annotators")


def _load_shuffle():
    from controlnet_aux import ContentShuffleDetector
    return ContentShuffleDetector()


def _load_normalbae():
    from controlnet_aux import NormalBaeDetector
    return NormalBaeDetector.from_pretrained("lllyasviel/Annotators")


def _load_sam():
    from controlnet_aux import SamDetector
    return SamDetector.from_pretrained("ybelkada/segment-anything",
                                       subfolder="checkpoints")


def _annotate_canny(image: np.ndarray, resolution: int,
                    low: int = 100, high: int = 200) -> np.ndarray:
    """Canny edge detection via cv2 (no model weights needed).
    low/high = notebook's low_threshold/high_threshold."""
    h, w = image.shape[:2]
    scale = resolution / min(h, w)
    new_h, new_w = int(h * scale) // 64 * 64, int(w * scale) // 64 * 64
    new_h, new_w = max(new_h, 64), max(new_w, 64)
    resized = cv2.resize(image, (new_w, new_h))
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    return np.stack([edges, edges, edges], axis=-1)


def _nms(x: np.ndarray, t: float, s: float) -> np.ndarray:
    """Directional NMS from the reference ControlNet annotator/util.py
    (Apache-2.0) — used by the notebook's scribble post-pass."""
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z


def _annotate_scribble(image: np.ndarray, resolution: int, opts: dict) -> np.ndarray:
    """Notebook scribble: HED/PIDI detector (control_sd15_scribble_detector,
    default PIDI) then, at render resolution, the nms → blur → binarize
    post-pass:

        detected_map = nms(detected_map, 127, 3.0)
        detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
        detected_map[detected_map > 4] = 255
        detected_map[detected_map < 255] = 0
    """
    if opts.get('scribble_detector', 'PIDI') == 'HED':
        detected = _run_detector('scribble_hed', _load_hed, image, resolution)
    else:
        detected = _run_detector('scribble', _load_pidinet, image, resolution)

    # Notebook applies the post-pass after resizing to (W, H) — do the same
    # so the binarized edges aren't smeared by a later resize.
    target_hw = opts.get('target_hw')
    if target_hw is not None:
        detected = cv2.resize(detected, (target_hw[1], target_hw[0]),
                              interpolation=cv2.INTER_LINEAR)

    detected = _nms(detected, 127, 3.0)
    detected = cv2.GaussianBlur(detected, (0, 0), 3.0)
    detected[detected > 4] = 255
    detected[detected < 255] = 0
    return detected


def _annotate_tile(image: np.ndarray, opts: dict) -> np.ndarray:
    """Notebook tile/qr/gif conditioning: passthrough + mask options.

        if qr_cn_mask_grayscale: gray → 3ch
        if qr_cn_mask_invert: 255 - map
        if qr_cn_mask_thresh > 0: binarize at thresh
        clip(qr_cn_mask_clip_low, qr_cn_mask_clip_high)
        tile models with downscale_tile > 1: REPLACE the map with the raw
        input downscaled (blur-via-downscale; discards the mask ops — notebook
        quirk, replicated)
    """
    detected = image.copy()
    if opts.get('qr_cn_mask_grayscale', False):
        detected = cv2.cvtColor(detected, cv2.COLOR_BGR2GRAY)[..., None].repeat(3, 2)
    if opts.get('qr_cn_mask_invert', False):
        detected = 255 - detected
    thresh = opts.get('qr_cn_mask_thresh', 0)
    if thresh > 0:
        detected = np.where(detected > thresh, 255, 0).astype(image.dtype)
    detected = detected.clip(opts.get('qr_cn_mask_clip_low', 0),
                             opts.get('qr_cn_mask_clip_high', 255))

    downscale = opts.get('downscale_tile', 1)
    if opts.get('is_tile_model', False) and downscale > 1:
        h, w = image.shape[:2]
        target = max(int(max(h, w) / downscale), 8)
        scale = target / min(h, w)
        new_w = max(int(w * scale) // 64 * 64, 64)
        new_h = max(int(h * scale) // 64 * 64, 64)
        detected = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return detected


def _annotate_depth(image: np.ndarray, resolution: int, detector: str) -> np.ndarray:
    """Depth map with detector selection (notebook: control_sd15_depth_detector)."""
    if detector == 'Zoe':
        return _run_detector('depth_zoe', _load_zoe, image, resolution)
    elif detector == 'depth_anything':
        return _annotate_depth_anything(image, resolution)
    # default: Midas
    return _run_detector('depth', _load_midas, image, resolution)


def _annotate_depth_anything(image: np.ndarray, resolution: int) -> np.ndarray:
    """Depth Anything via transformers (the notebook used the
    Sxela/Depth-Anything-light clone with depth_anything_vitl14 — the
    LiheYoung/depth-anything-large-hf checkpoint is the same ViT-L/14 model).
    Returns a grayscale HWC3 uint8 map, matching the notebook's
    apply_depth(input, colored=False)."""
    cache_key = 'depth_anything'
    if cache_key not in _annotator_cache:
        import time
        from vibewarp.core.timing import log_time
        t0 = time.time()
        from transformers import pipeline as hf_pipeline
        device = 0 if torch.cuda.is_available() else -1
        _annotator_cache[cache_key] = hf_pipeline(
            'depth-estimation', model='LiheYoung/depth-anything-large-hf',
            device=device)
        log_time('annotator depth_anything load', time.time() - t0)
    pipe = _annotator_cache[cache_key]

    h, w = image.shape[:2]
    scale = resolution / min(h, w)
    new_w, new_h = max(int(w * scale) // 14 * 14, 14), max(int(h * scale) // 14 * 14, 14)
    pil_img = Image.fromarray(image).resize((new_w, new_h))
    depth = np.array(pipe(pil_img)['depth'])  # PIL 'L' image, uint8
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.stack([depth, depth, depth], axis=-1)


_ONEFORMER_CKPTS = {
    'ade20k': 'shi-labs/oneformer_ade20k_swin_large',
    'coco': 'shi-labs/oneformer_coco_swin_large',
}


def _annotate_oneformer(image: np.ndarray, resolution: int, dataset: str) -> np.ndarray:
    """Semantic segmentation via transformers OneFormer, rendered with the
    dataset palette — matching the reference ControlNet annotator
    (OneformerADE20kDetector / OneformerCOCODetector), which draws
    detectron2 `stuff_colors` per class. Known minor delta: the reference
    visualizer also strokes a thin off-white edge along segment boundaries;
    we render flat class colors only.

    The shi-labs checkpoints are the HF ports of the same swin-large models
    (label id order == the palette list order in vendor/seg_palettes.py).
    """
    cache_key = f'seg_oneformer_{dataset}'
    if cache_key not in _annotator_cache:
        import time
        from vibewarp.core.timing import log_time
        t0 = time.time()
        from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
        ckpt = _ONEFORMER_CKPTS[dataset]
        processor = OneFormerProcessor.from_pretrained(ckpt)
        model = OneFormerForUniversalSegmentation.from_pretrained(ckpt)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        _annotator_cache[cache_key] = (processor, model)
        log_time(f'annotator {cache_key} load', time.time() - t0)
    processor, model = _annotator_cache[cache_key]

    h, w = image.shape[:2]
    scale = resolution / min(h, w)
    new_w, new_h = max(int(w * scale), 64), max(int(h * scale), 64)
    pil_img = Image.fromarray(image).resize((new_w, new_h))

    inputs = processor(images=pil_img, task_inputs=['semantic'], return_tensors='pt')
    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    seg = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[(new_h, new_w)])[0].cpu().numpy()

    from vibewarp.vendor.seg_palettes import ADE20K_PALETTE, COCO_PALETTE
    palette = np.array(ADE20K_PALETTE if dataset == 'ade20k' else COCO_PALETTE,
                       dtype=np.uint8)
    colored = palette[seg.clip(0, len(palette) - 1)]
    # NEAREST — palette colors are class codes; never blend them
    return cv2.resize(colored, (w, h), interpolation=cv2.INTER_NEAREST)


def _annotate_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaint CN conditioning — notebook cell 20 (control_sd15_inpaint):

        control_inpainting_mask *= 255
        control_inpainting_mask = 255 - control_inpainting_mask
        detected_mask = resize(mask[:, :, 0], (w, h))
        detected_map = img.float().copy()
        detected_map[detected_mask > 127] = -255.0   # -1.0 after /255

    Args:
        image: HWC uint8 source image (the CN's source — defaults to the
            warped previous render / init image)
        mask: keep mask, HW or HWC, float 0-1 or uint8 (1/255 = keep,
            0 = regenerate — e.g. a consistency mask)

    Returns:
        HWC float32 map with -255.0 markers in the regenerate region.
    """
    mask = np.asarray(mask)
    if mask.dtype != np.uint8:
        mask = (mask.astype(np.float32) * 255).clip(0, 255).astype(np.uint8)
    mask = 255 - mask  # invert: 255 = regenerate
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    h, w = image.shape[:2]
    detected_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    detected_map = image.astype(np.float32).copy()
    detected_map[detected_mask > 127] = -255.0
    return detected_map


def _load_zoe():
    from controlnet_aux import ZoeDetector
    return ZoeDetector.from_pretrained("lllyasviel/Annotators")


# ---- Conditioning tensor creation ----

def map_to_conditioning(
    detected_map: np.ndarray,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Convert an annotated map to a conditioning tensor for the model.

    Args:
        detected_map: HWC uint8 numpy array from annotator
        target_h: target height (model input size)
        target_w: target width (model input size)

    Returns:
        Tensor of shape (1, 3, H, W) normalized to [0, 1].
    """
    resized = cv2.resize(detected_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    if resized.ndim == 2:
        resized = np.stack([resized, resized, resized], axis=-1)
    elif resized.shape[2] == 1:
        resized = np.repeat(resized, 3, axis=2)
    tensor = torch.from_numpy(resized.copy()).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor


def get_controlnet_conditioning(
    source_images: Dict[str, Any],
    controlnet_config: Dict[str, dict],
    target_h: int,
    target_w: int,
    resolution: int = 512,
) -> Optional[torch.Tensor]:
    """Generate combined ControlNet conditioning from source images.

    Args:
        source_images: dict mapping model_key → PIL Image, numpy array, or path
        controlnet_config: dict mapping model_key → config dict with:
            - 'weight': float, conditioning strength
            - 'annotator': str, annotator type override (optional)
        target_h: model input height
        target_w: model input width
        resolution: annotator processing resolution

    Returns:
        Combined conditioning tensor (1, 3, H, W) or None if no sources.
    """
    if not source_images:
        return None

    combined = None

    for model_key, source in source_images.items():
        cfg = controlnet_config.get(model_key, {})
        weight = cfg.get('weight', 1.0)
        annotator_type = cfg.get('annotator', '') or ANNOTATOR_MAP.get(model_key, 'canny')

        if isinstance(source, str):
            if not os.path.exists(source):
                continue
            source_img = np.array(Image.open(source).convert('RGB'))
        elif isinstance(source, Image.Image):
            source_img = np.array(source.convert('RGB'))
        elif isinstance(source, np.ndarray):
            source_img = source
        else:
            continue

        detected = annotate_image(source_img, annotator_type, resolution)
        cond = map_to_conditioning(detected, target_h, target_w)

        if combined is None:
            combined = cond * weight
        else:
            combined = combined + cond * weight

    if combined is not None:
        combined = combined.clamp(0, 1)

    return combined


# ---- Layer-wise ControlNet weights ----

DEFAULT_NUM_CN_LAYERS = 13  # SD1.5: 12 encoder + 1 middle


def apply_layer_weights(
    control_outputs: list,
    layer_weights: Optional[list] = None,
    global_weight: float = 1.0,
) -> list:
    """Apply per-layer weights to ControlNet output tensors."""
    if not control_outputs:
        return control_outputs
    if layer_weights is not None:
        weights = layer_weights[:len(control_outputs)]
        return [output * (weights[i] if i < len(weights) else global_weight)
                for i, output in enumerate(control_outputs)]
    return [output * global_weight for output in control_outputs]


def normalize_controlnet_weights(
    controlnet_configs: Dict[str, dict],
    num_layers: int = DEFAULT_NUM_CN_LAYERS,
) -> Dict[str, list]:
    """Normalize layer weights across multiple ControlNets so they sum to 1 per layer."""
    if not controlnet_configs:
        return {}

    model_weights = {}
    for key, cfg in controlnet_configs.items():
        lw = cfg.get('layer_weights', None)
        gw = cfg.get('weight', 1.0)
        if lw is not None:
            weights = list(lw[:num_layers])
            while len(weights) < num_layers:
                weights.append(gw)
        else:
            weights = [gw] * num_layers
        model_weights[key] = weights

    normalized = {k: list(v) for k, v in model_weights.items()}
    for layer_idx in range(num_layers):
        total = sum(normalized[k][layer_idx] for k in normalized)
        if total > 0:
            for k in normalized:
                normalized[k][layer_idx] /= total

    return normalized


# ---- Model patching for multi-ControlNet ----

def _split_text_and_vector_conditioning(sd_model, cond):
    """Return cross-attention context and optional SDXL pooled conditioning."""
    vector = None
    if isinstance(cond, dict):
        crossattn = cond.get('c_crossattn', cond.get('crossattn'))
        if isinstance(crossattn, list):
            crossattn = torch.cat(crossattn, 1)
        vector = cond.get('y', cond.get('vector'))
        cond_txt = crossattn
    else:
        cond_txt = cond

    if vector is None and hasattr(sd_model, 'conditioner'):
        vector = getattr(sd_model.conditioner, 'vector_in', None)
    return cond_txt, vector

def patch_model_for_controlnets(sd_model, loaded_controlnets: Dict[str, dict],
                                normalize_weights: bool = False):
    """Patch sd_model.apply_model to support multi-ControlNet inference.

    After patching, per-frame hint tensors must be set on
    `sd_model._cn_hints` (dict of cn_key → (1,3,H,W) tensor) before sampling.
    """
    from ldm.modules.diffusionmodules.util import timestep_embedding

    diffusion_model = sd_model.model.diffusion_model
    sd_model._cn_models = loaded_controlnets
    sd_model._cn_hints = {}
    sd_model._cn_t_ratio = 0.0
    sd_model._cn_normalize_weights = normalize_weights

    def controlled_unet_forward(x, timesteps=None, context=None, control=None,
                                only_mid_control=False, **kwargs):
        from vibewarp.core.unet_acceleration import next_cache_slot

        hs = []
        cache_mode = getattr(diffusion_model, '_vw_cache_mode', 'off')
        cache_slot = None
        refresh_cache = True
        if cache_mode != 'off':
            cache_slot, refresh_cache = next_cache_slot(diffusion_model)

        input_blocks = diffusion_model.input_blocks
        output_blocks = diffusion_model.output_blocks
        shallow_blocks = max(1, min(
            len(input_blocks), len(output_blocks), len(input_blocks) // 4))
        deep_output_count = len(output_blocks) - shallow_blocks
        cache_signature = (
            tuple(x.shape),
            tuple(context.shape) if torch.is_tensor(context) else None,
            len(input_blocks),
            len(output_blocks),
        )
        reused_deep = False

        with torch.inference_mode():
            t_emb = timestep_embedding(timesteps, diffusion_model.model_channels, repeat_only=False)
            emb = diffusion_model.time_embed(t_emb)
            if getattr(diffusion_model, 'num_classes', None) is not None:
                y = kwargs.get('y')
                assert y is not None, "SDXL/class-conditional UNet requires vector conditioning"
                emb = emb + diffusion_model.label_emb(y)
            h = x.type(diffusion_model.dtype)

            # First Block Cache follows the common FBCache rule: always evaluate
            # the first encoder block, compare its relative change, and reuse the
            # last full U-Net residual when it is sufficiently similar.
            first_block = input_blocks[0]
            h = first_block(h, emb, context)
            hs.append(h)
            if cache_mode == 'first_block':
                previous = cache_slot.get('first')
                residual = cache_slot.get('output_residual')
                compatible = (
                    cache_slot.get('signature') == cache_signature
                    and torch.is_tensor(previous)
                    and previous.shape == h.shape
                    and torch.is_tensor(residual)
                    and residual.shape == x.shape
                )
                metric = None
                if compatible:
                    denominator = previous.float().abs().mean().clamp_min(1e-6)
                    metric = (
                        (h.float() - previous.float()).abs().mean() / denominator)
                cache_slot['first'] = h.detach()
                threshold = float(
                    getattr(diffusion_model, '_vw_cache_threshold', 0.05))
                if metric is not None and float(metric) <= threshold:
                    diffusion_model._vw_cache_runtime['hits'] += 1
                    cache_slot['last_metric'] = float(metric)
                    return (x + residual.to(device=x.device, dtype=x.dtype))

            # DeepCache refreshes only every Nth logical denoiser evaluation.
            # A reuse pass recomputes the shallow encoder blocks, injects the
            # cached inner decoder feature, and executes the matching shallow
            # decoder blocks with fresh skip connections.
            can_reuse_deep = (
                cache_mode == 'deepcache'
                and not refresh_cache
                and cache_slot.get('signature') == cache_signature
                and torch.is_tensor(cache_slot.get('deep_feature'))
            )
            encoder_stop = shallow_blocks if can_reuse_deep else len(input_blocks)
            for module in input_blocks[1:encoder_stop]:
                h = module(h, emb, context)
                hs.append(h)
            if can_reuse_deep:
                cached = cache_slot['deep_feature']
                expected_spatial = hs[-1].shape[-2:]
                if cached.shape[0] == x.shape[0] and cached.shape[-2:] == expected_spatial:
                    h = cached.to(device=x.device, dtype=diffusion_model.dtype)
                    reused_deep = True
                else:
                    # A new tile/multiscale shape gets a full pass and replaces
                    # this stream's cache instead of risking a malformed skip.
                    for module in input_blocks[encoder_stop:]:
                        h = module(h, emb, context)
                        hs.append(h)
                    h = diffusion_model.middle_block(h, emb, context)
            else:
                h = diffusion_model.middle_block(h, emb, context)

        # Active latent/pixel guidance deliberately leaves inference mode for
        # the sampler. Materialize the middle activation as a normal tensor
        # before ControlNet or FreeU can update it in place. Ordinary renders
        # remain inside the outer inference context and pay no clone cost.
        if not torch.is_inference_mode_enabled():
            h = h.clone()

        if reused_deep and control is not None:
            # The cached inner decoder feature already contains the previous
            # evaluation's middle/deep ControlNet residuals. Consume the current
            # counterparts so the remaining pops line up with shallow blocks.
            for _ in range(1 + deep_output_count):
                if control:
                    control.pop()
        elif control is not None:
            # `h` was created in the inference_mode block above. Gradient
            # guidance intentionally runs the outer sampler under no_grad so
            # it can locally enable autograd for the predicted latent; an
            # in-place write to that inference tensor is illegal once the
            # nested inference context exits. Out-of-place addition preserves
            # the value and produces a normal tensor for the remaining UNet.
            h = h + control.pop()

        # FreeU lives in the notebook's control_multi forward. VibeWarp uses
        # this forward universally, including with an empty CN model dict.
        # Enabled via freeu.set_freeu → diffusion_model._freeu.
        freeu_cfg = getattr(diffusion_model, '_freeu', None)
        if freeu_cfg is not None:
            from vibewarp.core.freeu import apply_freeu

        output_start = deep_output_count if reused_deep else 0
        for i, module in enumerate(output_blocks[output_start:], start=output_start):
            _hs = hs.pop()
            if freeu_cfg is not None and not freeu_cfg['after_control']:
                h, _hs = apply_freeu(h, _hs, freeu_cfg['b1'], freeu_cfg['b2'],
                                     freeu_cfg['s1'], freeu_cfg['s2'])
            if not only_mid_control and control is not None:
                _hs = _hs + control.pop()
            if freeu_cfg is not None and freeu_cfg['after_control']:
                h, _hs = apply_freeu(h, _hs, freeu_cfg['b1'], freeu_cfg['b2'],
                                     freeu_cfg['s1'], freeu_cfg['s2'])
            # notebook cat8: guard odd-resolution spatial mismatch
            if h.shape[-2:] != _hs.shape[-2:]:
                h = torch.nn.functional.interpolate(h, _hs.shape[-2:], mode='nearest')
            h = torch.cat([h, _hs], dim=1)
            h = module(h, emb, context)
            if (cache_mode == 'deepcache' and not reused_deep
                    and i + 1 == deep_output_count):
                cache_slot['deep_feature'] = h.detach()

        h = h.type(x.dtype)
        output = diffusion_model.out(h)
        if cache_mode != 'off':
            cache_slot['signature'] = cache_signature
            if reused_deep:
                diffusion_model._vw_cache_runtime['hits'] += 1
            else:
                diffusion_model._vw_cache_runtime['misses'] += 1
            if cache_mode == 'first_block':
                cache_slot['output_residual'] = (output - x).detach()
        return output

    diffusion_model.forward = controlled_unet_forward

    _cn_logged_step = [None]

    def apply_model_with_controlnets(x_noisy, t, cond, *args, **kwargs):
        cond_txt, vector_cond = _split_text_and_vector_conditioning(sd_model, cond)
        context_override = os.environ.get('VIBEWARP_DEBUG_CN_CONTEXT_PATH')
        if context_override and not getattr(sd_model, '_cn_context_override_used', False):
            cond_txt = torch.load(context_override, map_location=x_noisy.device,
                                  weights_only=True).to(x_noisy.device)
            sd_model._cn_context_override_used = True
            print(f"[diag] using ControlNet context override: {context_override}")
        unet_kwargs = {'y': vector_cond} if vector_cond is not None else {}

        cn_hints = sd_model._cn_hints
        cn_models = sd_model._cn_models
        # WarpFusion gates ControlNets on the model timestep, not directly on
        # sigma: t_ratio = 1 - t / num_timesteps. Sigma is nonlinear in t, so
        # sigma/sigma_max activates start/end ranges on different Euler steps.
        num_timesteps = float(getattr(sd_model, 'num_timesteps', 1000))
        timestep = float(t.flatten()[0].item()) if torch.is_tensor(t) else float(t)
        t_ratio = 1.0 - timestep / num_timesteps
        sd_model._cn_t_ratio = t_ratio
        normalize = sd_model._cn_normalize_weights

        if not cn_hints or not cn_models:
            return diffusion_model(
                x=x_noisy, timesteps=t, context=cond_txt, control=None,
                **unet_kwargs)

        active = {}
        for key, cn_data in cn_models.items():
            if key not in cn_hints:
                continue
            cn_start = cn_data.get('start', 0.0)
            cn_end = cn_data.get('end', 1.0)
            weight = cn_data.get('weight', 1.0)
            if weight != 0 and cn_start <= t_ratio <= cn_end:
                active[key] = cn_data

        if not active:
            return diffusion_model(
                x=x_noisy, timesteps=t, context=cond_txt, control=None,
                **unet_kwargs)

        t_ratio_key = round(t_ratio, 3)
        _cn_logged_step[0] = t_ratio_key

        import numpy as np_
        weights = np_.array([active[k]['weight'] for k in active])
        if normalize:
            weights = weights / weights.sum()

        batch_size = x_noisy.shape[0]
        control_wsum = None
        for i, key in enumerate(active):
            cn_data = active[key]
            cn_model = cn_data['model']
            hint = cn_hints[key].to(x_noisy.device)

            if hint.shape[0] < batch_size:
                hint = hint.repeat(batch_size, 1, 1, 1)

            if cn_data.get('zero_uncond', False):
                uc_mask = getattr(diffusion_model, 'uc_mask_shape', None)
                if uc_mask is not None:
                    hint = hint * uc_mask.to(hint)[:, None, None, None]

            # Validation-only exact notebook hint injection. This isolates CN
            # model/UNet execution from annotator and resize differences.
            hint_override = os.environ.get('VIBEWARP_DEBUG_CN_HINT_PATH')
            if hint_override and not getattr(sd_model, '_cn_hint_override_used', False):
                hint = torch.load(hint_override, map_location=x_noisy.device,
                                  weights_only=True).to(x_noisy.device)
                sd_model._cn_hint_override_used = True
                print(f"[diag] using ControlNet hint override: {hint_override}")

            # SDXL ControlNets fold the pooled vector into their timestep embedding
            # exactly like the UNet does, so `y` has to reach them too. SD1.5 nets
            # ignore it (num_classes is None).
            control = cn_model(x=x_noisy, hint=hint, timesteps=t, context=cond_txt,
                               **unet_kwargs)

            if (os.environ.get('VIBEWARP_DEBUG_DIFFUSION')
                    and not getattr(sd_model, '_cn_input_probed', False)):
                def diag_tensor(value):
                    cpu = value.detach().float().cpu().contiguous()
                    return {
                        'shape': list(value.shape), 'dtype': str(value.dtype),
                        'sha256_f32': hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
                        'first8': [round(v, 6) for v in cpu.flatten()[:8].tolist()],
                    }
                print('[diag] controlnet_first_call=' + json.dumps({
                    'key': key, 'x': diag_tensor(x_noisy),
                    'hint': diag_tensor(hint), 'timestep': diag_tensor(t),
                    'context': diag_tensor(cond_txt),
                    'residuals': [diag_tensor(value) for value in control],
                }, sort_keys=True))
                sd_model._cn_input_probed = True

            layer_weights = cn_data.get('layer_weights', None)
            if layer_weights is not None:
                control = [c * scale for c, scale in zip(control, layer_weights)]

            if key == 'control_sd15_shuffle':
                control = [torch.mean(c, dim=(2, 3), keepdim=True) for c in control]

            if control_wsum is None:
                control_wsum = [weights[i] * c for c in control]
            else:
                control_wsum = [weights[i] * c + cs for c, cs in zip(control, control_wsum)]

        return diffusion_model(
            x=x_noisy, timesteps=t, context=cond_txt, control=control_wsum,
            **unet_kwargs)

    sd_model.apply_model = apply_model_with_controlnets
    print(f"  Patched apply_model for {len(loaded_controlnets)} ControlNets")


def prepare_cn_hints_for_frame(
    sd_model,
    loaded_controlnets: Dict[str, dict],
    source_paths: Dict[str, str],
    target_h: int,
    target_w: int,
    t_ratio: float = 0.0,
    debug_dir: Optional[str] = None,
    frame_num: int = 0,
    depth_detector: str = 'Midas',
    inpaint_mask: Optional[np.ndarray] = None,
    seg_detector: str = 'Seg_UFADE20K',
    annotator_opts: Optional[dict] = None,
):
    """Prepare per-ControlNet hint tensors for a frame and store on sd_model.

    Called before each frame's sampling to set up `sd_model._cn_hints`.

    Args:
        depth_detector: 'Midas' / 'Zoe' / 'depth_anything'
            (notebook control_sd15_depth_detector, applies to all depth CNs)
        inpaint_mask: keep-mask for inpaint CNs (e.g. consistency mask,
            0-1 float). When None, inpaint CNs are skipped for this frame —
            same as the notebook removing them from the model list.
        seg_detector: 'Seg_OFADE20K' / 'Seg_OFCOCO' / 'Seg_UFADE20K' / 'Seg_SAM'
            (notebook control_sd15_seg_detector)
        annotator_opts: extra annotator options (canny/mlsd thresholds,
            scribble/softedge detector selects, qr/tile mask options) — see
            annotate_image. 'is_tile_model' and 'target_hw' are filled in here.
    """
    hints = {}
    for cn_key, cn_data in loaded_controlnets.items():
        source_path = source_paths.get(cn_key)
        if not source_path or not os.path.exists(source_path):
            print(f"  CN hint skip {cn_key}: source missing ({source_path})")
            continue

        annotator_type = cn_data.get('annotator', '') or ANNOTATOR_MAP.get(cn_key, 'canny')

        if annotator_type == 'inpaint' and inpaint_mask is None:
            print(f"  CN hint skip {cn_key}: no inpaint mask for this frame")
            continue

        detect_res = cn_data.get('detect_resolution', -1)
        resolution = max(target_h, target_w) if detect_res <= 0 else detect_res

        source_img = np.array(
            Image.open(source_path).convert('RGB').resize(
                (target_w, target_h), Image.Resampling.BICUBIC))
        if annotator_type == 'inpaint':
            # Notebook builds the inpaint marker map at render resolution —
            # resize first so the -255 markers aren't smeared by the later
            # conditioning resize.
            source_img = source_img.copy()

        opts = dict(annotator_opts or {})
        opts['is_tile_model'] = cn_key in TILE_CN_KEYS
        opts['target_hw'] = (target_h, target_w)

        detected = annotate_image(
            source_img, annotator_type, resolution=resolution,
            depth_detector=depth_detector,
            colored_depth=('depth_anything' in cn_key),
            inpaint_mask=inpaint_mask,
            seg_detector=seg_detector,
            opts=opts,
        )

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            Image.fromarray(source_img).save(
                os.path.join(debug_dir, f"{cn_key}_source_{frame_num:06d}.jpg"), quality=95)
            det_img = detected if detected.ndim == 3 else np.stack([detected] * 3, axis=-1)
            Image.fromarray(det_img.clip(0, 255).astype(np.uint8)).save(
                os.path.join(debug_dir, f"{cn_key}_detected_{frame_num:06d}.jpg"), quality=95)

        hint = map_to_conditioning(detected, target_h, target_w)
        # Notebook keeps conditioning maps in float32 and relies on autocast
        # inside the ControlNet call. Premature fp16 quantization changes the
        # first hint convolution and amplifies through all 13 residuals.
        hints[cn_key] = hint.cuda()

    if not hints:
        print(f"  WARNING: No CN hints prepared for frame {frame_num}")
    else:
        print(f"  CN hints ready: {list(hints.keys())}")

    sd_model._cn_hints = hints
