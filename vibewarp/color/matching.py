"""Color matching and transfer between frames."""

import time

import cv2
import numpy as np


def _default_pdf_transfer(img_arr_in, img_arr_ref):
    """Default PDF transfer — GPU when torch+CUDA available, CPU numpy otherwise."""
    try:
        import torch
        if torch.cuda.is_available():
            from vibewarp.color.pdf_gpu import pdf_transfer_gpu
            return pdf_transfer_gpu(img_arr_in, img_arr_ref)
    except Exception:
        pass
    from vibewarp.vendor.python_color_transfer.color_transfer import ColorTransfer
    return ColorTransfer().pdf_transfer(img_arr_in=img_arr_in, img_arr_ref=img_arr_ref)


def match_color_var(stylized_img, raw_img, opacity=1., transfer_fn=None, regrain_fn=None, regrain=False):
    """Match colors of stylized_img to raw_img using a PDF transfer function.

    Args:
        stylized_img: reference image (numpy RGB, uint8 or float)
        raw_img: input image to color-match (numpy RGB, uint8 or float)
        opacity: blend factor (1.0 = full color match)
        transfer_fn: callable(img_arr_in, img_arr_ref) -> color-transferred image.
            When None, uses the GPU PDF transfer (or CPU fallback) — matches the
            notebook's `match_color_var(..., f=PT.pdf_transfer)` default.
        regrain_fn: optional callable for re-graining
        regrain: whether to apply re-graining

    Returns:
        Color-matched image as numpy RGB uint8 array.
    """
    img_arr_ref = cv2.cvtColor(np.array(stylized_img).round().astype('uint8'), cv2.COLOR_RGB2BGR)
    img_arr_in = cv2.cvtColor(np.array(raw_img).round().astype('uint8'), cv2.COLOR_RGB2BGR)
    img_arr_ref = cv2.resize(img_arr_ref, (img_arr_in.shape[1], img_arr_in.shape[0]),
                             interpolation=cv2.INTER_CUBIC)

    if transfer_fn is None:
        transfer_fn = _default_pdf_transfer
    img_arr_col = transfer_fn(img_arr_in=img_arr_in, img_arr_ref=img_arr_ref)

    if regrain and regrain_fn is not None:
        img_arr_col = regrain_fn(img_arr_in=img_arr_col, img_arr_col=img_arr_ref)

    img_arr_col = img_arr_col * opacity + img_arr_in * (1 - opacity)
    img_arr_reg = cv2.cvtColor(img_arr_col.round().astype('uint8'), cv2.COLOR_BGR2RGB)
    return img_arr_reg


def match_color_simple(stylized_img, raw_img, opacity=1.):
    """Simple mean/std color matching between two images.

    Args:
        stylized_img: numpy RGB array
        raw_img: numpy RGB array
        opacity: blend factor

    Returns:
        Color-matched image as numpy RGB uint8 array.
    """
    stylized = stylized_img.astype(np.float32)
    raw = raw_img.astype(np.float32)

    # Match per-channel mean and std
    for c in range(3):
        s_mean, s_std = stylized[..., c].mean(), stylized[..., c].std()
        r_mean, r_std = raw[..., c].mean(), raw[..., c].std()
        if s_std > 0:
            raw[..., c] = (raw[..., c] - r_mean) * (s_std / (r_std + 1e-8)) + s_mean

    result = raw * opacity + raw_img.astype(np.float32) * (1 - opacity)
    return np.clip(result, 0, 255).astype(np.uint8)


def get_stats(image):
    """Get mean and std per channel of an image array."""
    result = {}
    for i, ch_name in enumerate(['r', 'g', 'b']):
        ch = image[..., i].astype(np.float32)
        result[ch_name] = {'mean': ch.mean(), 'std': ch.std()}
    return result
