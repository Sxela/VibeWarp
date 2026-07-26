"""Frame warping using optical flow fields."""

import math

import cv2
import numpy as np
from PIL import Image

from vibewarp.flow.flow_utils import warp_flow


def warp_frame(frame1, frame2, flow, blend=0.5, weights=None, forward_clip=0.,
               pad_pct=0.1, padding_mode='reflect', warp_mul=1.,
               match_color_fn=None, adjust_brightness_fn=None,
               video_mode=False, return_warped=False):
    """Warp frame1 toward frame2 using optical flow, then blend.

    Args:
        frame1: PIL Image (previous frame / stylized)
        frame2: PIL Image (current frame / init)
        flow: numpy array (H, W, 2) optical flow field
        blend: blend ratio between warped frame1 and frame2
        weights: optional consistency weights (H, W, 3) numpy array, values in [0, 1]
        forward_clip: clip consistency weights to this minimum value
        pad_pct: padding percentage to avoid edge artifacts
        padding_mode: numpy pad mode ('reflect', 'edge', etc.)
        warp_mul: flow magnitude multiplier
        match_color_fn: optional callable(stylized, raw, opacity) for color matching
        adjust_brightness_fn: optional callable(image) for brightness correction
        video_mode: if True, skip color matching and brightness adjustment
        return_warped: also return the motion-warped frame before raw-frame
            blending and consistency compositing

    Returns:
        Blended PIL image, or ``(blended, warped_only)`` when requested.
    """
    flow21 = flow
    pad = int(max(flow21.shape) * pad_pct)
    flow21 = np.pad(flow21, pad_width=((pad, pad), (pad, pad), (0, 0)), mode='constant')

    frame1pil = np.array(frame1.convert('RGB'))
    frame1pil = np.pad(frame1pil, pad_width=((pad, pad), (pad, pad), (0, 0)), mode=padding_mode)

    if video_mode:
        warp_mul = 1.

    frame1_warped = warp_flow(frame1pil, flow21, warp_mul)
    frame1_warped = frame1_warped[pad:frame1_warped.shape[0] - pad, pad:frame1_warped.shape[1] - pad, :]

    target_w = flow21.shape[1] - pad * 2
    target_h = flow21.shape[0] - pad * 2
    frame2pil = np.array(frame2.convert('RGB').resize((target_w, target_h), Image.LANCZOS))

    if weights is not None:
        if not video_mode and match_color_fn is not None:
            frame2pil = match_color_fn(frame1_warped, frame2pil)
        clipped_weights = weights.clip(forward_clip, 1.)
        blended = frame2pil * (1 - blend) + blend * (frame1_warped * clipped_weights + frame2pil * (1 - clipped_weights))
    else:
        if not video_mode and match_color_fn is not None:
            frame2pil = match_color_fn(frame1_warped, frame2pil)
        blended = frame2pil * (1 - blend) + frame1_warped * blend

    result = Image.fromarray(blended.round().astype('uint8'))
    warped_result = Image.fromarray(
        np.clip(frame1_warped, 0, 255).round().astype('uint8'))

    if not video_mode and adjust_brightness_fn is not None:
        result = adjust_brightness_fn(result)
        warped_result = adjust_brightness_fn(warped_result)

    return (result, warped_result) if return_warped else result


def blended_roll(img, shift, axis):
    """Roll an image by a fractional amount, blending between integer shifts.

    (c) Alex Spirin 2022
    """
    if int(shift) == shift:
        return np.roll(img, int(shift), axis=axis)

    max_shift = math.ceil(shift)
    min_shift = math.floor(shift)
    if min_shift != 0:
        img_min = np.roll(img, min_shift, axis=axis)
    else:
        img_min = img
    img_max = np.roll(img, max_shift, axis=axis)
    blend = max_shift - shift
    return img_min * blend + img_max * (1 - blend)


def get_k_means_clusters(flow, K):
    """Cluster flow vectors using k-means.

    Returns:
        (clustered_flow, centers) tuple.
    """
    Z = flow.reshape((-1, 2)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    res = center[label.flatten()]
    res2 = res.reshape(flow.shape)
    return res2, center


def k_means_warp(flow, img, num_k=8):
    """Warp an image using k-means clustered flow vectors.

    (c) Alex Spirin 2022

    Args:
        flow: numpy array (H, W, 2)
        img: PIL Image
        num_k: number of clusters

    Returns:
        Warped PIL Image.
    """
    img_arr = np.array(img.convert('RGB'))
    res2, center = get_k_means_clusters(flow, num_k)
    center = sorted(list(center), key=lambda x: abs(x).mean())

    img_arr = cv2.resize(img_arr, (res2.shape[:-1][::-1]))
    img_out = np.ones_like(img_arr) * 255.

    for i in range(num_k):
        motion = center[i]
        mask = np.where(res2 == motion, 1, 0)[..., 0][..., None]
        img_copy = img_arr.copy()
        y, x = motion
        img_copy = blended_roll(img_copy, x, 0)
        img_copy = blended_roll(img_copy, y, 1)
        img_out = img_out * (1 - mask) + img_copy * mask

    return Image.fromarray(img_out.astype('uint8'))
