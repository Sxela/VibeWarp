"""Consistency map generation for optical flow warping."""

import cv2
import numpy as np
import scipy.ndimage
from scipy.ndimage import binary_fill_holes
from skimage.morphology import disk, binary_erosion, binary_dilation


def extract_occlusion_mask(flow_tensor, threshold=10):
    """Extract occlusion mask from flow tensor.

    Args:
        flow_tensor: torch tensor [1, 2, H, W]
        threshold: magnitude threshold for occlusion detection

    Returns:
        (occlusion_mask, magnitude) tuple as numpy arrays.
    """
    flow = flow_tensor.clone()[0].permute(1, 2, 0).detach().cpu().numpy()
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    occlusion_mask = (mag > threshold).astype(np.uint8)
    return occlusion_mask, mag


def edge_detector(image, threshold=0.5, edge_width=1):
    """Detect edges using Sobel operator.

    Args:
        image: numpy BGR image (uint8)
        threshold: edge magnitude threshold (0-1)
        edge_width: Sobel kernel size

    Returns:
        Binary edge image (uint8, 0 or 255).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=edge_width)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=edge_width)
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    mag = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
    edge_image = (mag > threshold).astype(np.uint8) * 255
    return edge_image


def get_unreliable(flow):
    """Find pixels that have no valid source in the warped frame.

    Args:
        flow: numpy array (H, W, 2)

    Returns:
        (unreliable_mask, valid_mask) - both (H, W, 3) numpy arrays.
    """
    h, w = flow.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    new_x = x + flow[..., 0]
    new_y = y + flow[..., 1]

    mask = (new_x >= 0) & (new_x < w) & (new_y >= 0) & (new_y < h)

    new_frame = np.zeros((h, w, 3)) - 1.
    new_frame[new_y[mask].astype(np.int32), new_x[mask].astype(np.int32)] = 255

    unreliable = (new_frame == -1)
    return unreliable, mask


def remove_small_holes(mask, min_size=50):
    """Remove small holes from a binary mask.

    Args:
        mask: binary mask (uint8, 0 or 255)
        min_size: minimum contour area to keep

    Returns:
        Cleaned binary mask (uint8).
    """
    result = mask.copy()
    contours, hierarchy = cv2.findContours(result, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < min_size:
            cv2.drawContours(result, [contours[i]], 0, 255, -1, cv2.LINE_AA, hierarchy, 0)
    return result


def filter_unreliable(mask, dilation=1):
    """Filter and clean up unreliable pixel mask.

    Args:
        mask: (H, W, 3) numpy array from get_unreliable
        dilation: morphological dilation radius

    Returns:
        Binary mask (H, W) as numpy array.
    """
    img = 255 - remove_small_holes((1 - mask[..., 0].astype('uint8')) * 255, 200)
    img = binary_erosion(img, disk(1))
    img = binary_dilation(img, disk(dilation))
    return img


def make_cc_map(flow_fwd, flow_bwd, dilation=1, edge_width=11):
    """Create a 3-channel consistency map from forward and backward flow.

    Channels:
        R: missed consistency (unreliable areas from forward flow)
        G: overshoot consistency (valid areas from backward flow)
        B: edge consistency (edges detected in backward flow visualization)

    Args:
        flow_fwd: numpy (H, W, 2) forward flow
        flow_bwd: numpy (H, W, 2) backward flow
        dilation: morphological dilation for unreliable mask
        edge_width: Sobel kernel size for edge detection

    Returns:
        (H, W, 3) numpy uint8 array.
    """
    from vibewarp.flow.flow_utils import flow_to_image

    flow_imgs = flow_to_image(flow_bwd)
    edge = edge_detector(flow_imgs.astype('uint8'), threshold=0.1, edge_width=edge_width)
    res, _ = get_unreliable(flow_fwd)
    _, overshoot = get_unreliable(flow_bwd)

    joint_mask = np.ones_like(res) * 255
    joint_mask[..., 0] = 255 - (filter_unreliable(res, dilation) * 255)
    joint_mask[..., 1] = overshoot * 255
    joint_mask[..., 2] = 255 - edge

    return joint_mask.astype(np.uint8)


def load_cc(
    cc_map: np.ndarray,
    missed_consistency_weight: float = 1.0,
    overshoot_consistency_weight: float = 1.0,
    edges_consistency_weight: float = 1.0,
    blur: int = 1,
    dilate: int = 3,
) -> np.ndarray:
    """Process a 3-channel consistency map into B&W blending weights.

    Matches the notebook's load_cc() function exactly:
    1. Normalize each channel to [0, 1]
    2. Multiply channels together with per-channel clipping
    3. Binary threshold at 0.5
    4. Morphological dilation to expand unreliable regions
    5. Gaussian blur to soften mask edges
    6. Expand to 3-channel for blending

    Args:
        cc_map: (H, W, 3) uint8 consistency map from make_cc_map()
        missed_consistency_weight: weight for R channel (0-1, default 1.0)
        overshoot_consistency_weight: weight for G channel (0-1, default 1.0)
        edges_consistency_weight: weight for B channel (0-1, default 1.0)
        blur: Gaussian blur sigma (0 = no blur, default 1)
        dilate: morphological dilation radius (0 = no dilation, default 3)

    Returns:
        (H, W, 3) float32 array with values in [0, 1].
    """
    multilayer_weights = cc_map.astype(np.float32) / 255.0
    weights = np.ones_like(multilayer_weights[..., 0])
    weights *= multilayer_weights[..., 0].clip(1 - missed_consistency_weight, 1)
    weights *= multilayer_weights[..., 1].clip(1 - overshoot_consistency_weight, 1)
    weights *= multilayer_weights[..., 2].clip(1 - edges_consistency_weight, 1)
    weights = np.where(weights < 0.5, 0, 1).astype(np.float64)
    if dilate > 0:
        weights = (1 - binary_dilation(1 - weights, disk(dilate))).astype(np.float64)
    if blur > 0:
        weights = scipy.ndimage.gaussian_filter(weights, [blur, blur])
    return np.repeat(weights[..., None], 3, axis=2).astype(np.float32)
