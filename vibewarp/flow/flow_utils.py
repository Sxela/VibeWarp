"""Optical flow I/O, visualization, and helper utilities."""

import numpy as np
import cv2
import torch

# Magic number for .flo file format
TAG_FLOAT = 202021.25
TAG_CHAR = np.array([TAG_FLOAT], dtype=np.float32).tobytes()


def write_flow(filename, uv, v=None):
    """Write optical flow to a .flo file.

    Based on NVIDIA flownet2-pytorch (Apache 2.0).
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2
    if v is None:
        assert uv.ndim == 3 and uv.shape[2] == 2
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert u.shape == v.shape
    height, width = u.shape
    with open(filename, 'wb') as f:
        f.write(TAG_CHAR)
        np.array(width).astype(np.int32).tofile(f)
        np.array(height).astype(np.int32).tofile(f)
        tmp = np.zeros((height, width * nBands))
        tmp[:, np.arange(width) * 2] = u
        tmp[:, np.arange(width) * 2 + 1] = v
        tmp.astype(np.float32).tofile(f)


def read_flow(filename):
    """Read a .flo optical flow file.

    Returns:
        numpy array of shape (H, W, 2)
    """
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        assert magic == TAG_FLOAT, f"Invalid .flo file: bad magic number {magic}"
        width = np.fromfile(f, np.int32, count=1)[0]
        height = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * width * height)
    data = data.reshape(height, width, 2)
    return data


def warp_flow(img, flow, mul=1.):
    """Warp an image using an optical flow field.

    Args:
        img: numpy array (H, W, C) or (H, W)
        flow: numpy array (H, W, 2) of displacement vectors
        mul: flow multiplier

    Returns:
        Warped image as numpy array.
    """
    h, w = flow.shape[:2]
    flow = flow.copy()
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    flow *= mul
    res = cv2.remap(img, flow.astype(np.float32), None, cv2.INTER_LANCZOS4)
    return res


def make_colorwheel():
    """Create the Middlebury color wheel for flow visualization.

    Returns:
        numpy array of shape (N, 3) with RGB values.
    """
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col += RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col += GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(0, CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col += BM
    # MR
    colorwheel[col:col + MR, 0] = 255
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(0, MR) / MR)

    return colorwheel


def flow_to_image(flow_uv, clip_flow=None):
    """Convert optical flow to an RGB visualization image.

    Args:
        flow_uv: numpy array (H, W, 2)
        clip_flow: optional max magnitude to clip to

    Returns:
        numpy array (H, W, 3) with uint8 RGB values.
    """
    assert flow_uv.ndim == 3 and flow_uv.shape[2] == 2
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, -clip_flow, clip_flow)

    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(u**2 + v**2)
    maxrad = max(np.max(rad), 1e-5)
    u = u / maxrad
    v = v / maxrad

    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]
    angle = np.arctan2(-v, -u) / np.pi
    fk = (angle + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    h, w = flow_uv.shape[:2]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(3):
        col0 = colorwheel[k0, i] / 255.0
        col1 = colorwheel[k1, i] / 255.0
        col = (1 - f) * col0 + f * col1
        # Increase saturation with magnitude
        col = 1 - rad[..., np.newaxis].squeeze() / maxrad * (1 - col) if False else col
        img[:, :, i] = (col * 255).astype(np.uint8)

    return img


def load_raft_model(half_precision: bool = False, device: str = 'cuda'):
    """Load the RAFT optical flow model from torchvision.

    Uses torchvision's built-in RAFT-Large implementation.

    Args:
        half_precision: use float16 for faster inference
        device: target device

    Returns:
        RAFT model in eval mode.
    """
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    import time
    from vibewarp.core.timing import log_time
    t0 = time.time()
    weights = Raft_Large_Weights.C_T_SKHT_V1
    model = raft_large(weights=weights, progress=False).to(device)
    model.eval()
    log_time("RAFT-Large load+to(device)", time.time() - t0)
    if half_precision:
        model = model.half()
    return model


def _load_frame_for_flow(path_or_img, size=None, device='cuda'):
    """Load and prepare a frame for RAFT inference.

    Args:
        path_or_img: file path (str) or PIL Image
        size: optional (width, height) to resize to
        device: target device

    Returns:
        Tensor of shape (1, 3, H, W) on device, float32.
    """
    from PIL import Image as PILImage
    if isinstance(path_or_img, str):
        img = PILImage.open(path_or_img).convert('RGB')
    else:
        img = path_or_img.convert('RGB')
    if size is not None:
        img = img.resize(size, PILImage.LANCZOS)
    arr = np.array(img).astype(np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    # Normalize [0, 255] → [-1, 1] as expected by RAFT.
    # Torchvision RAFT does NOT normalize internally — the OpticalFlow transform
    # does it externally. Notebook: batch = 2 * (batch / 255.0) - 1.0
    tensor = 2 * (tensor / 255.0) - 1.0
    return tensor.to(device)


def compute_flow(frame1, frame2, model, half: bool = False, num_flow_updates: int = 20):
    """Compute optical flow between two frames using RAFT.

    Args:
        frame1: (1, 3, H, W) tensor or path string
        frame2: (1, 3, H, W) tensor or path string
        model: loaded RAFT model
        half: use half precision
        num_flow_updates: number of RAFT refinement iterations (notebook default: 20)

    Returns:
        numpy array (H, W, 2) of flow vectors.
    """
    if isinstance(frame1, str):
        frame1 = _load_frame_for_flow(frame1)
    if isinstance(frame2, str):
        frame2 = _load_frame_for_flow(frame2)

    padder = InputPadder(frame1.shape)
    frame1, frame2 = padder.pad(frame1, frame2)

    # Notebook always uses autocast for RAFT (torch.cuda.amp.autocast()).
    # This affects flow precision: fp16 quantizes flow values, producing fewer
    # unique sub-pixel positions, which reduces noise in get_unreliable() CC maps.
    with torch.inference_mode(), torch.amp.autocast('cuda'):
        flow_predictions = model(frame1, frame2, num_flow_updates=num_flow_updates)
        # RAFT returns a list of flow predictions (one per iteration); take the last
        if isinstance(flow_predictions, (list, tuple)):
            flow12 = flow_predictions[-1]
        else:
            flow12 = flow_predictions

    flow12 = padder.unpad(flow12)
    return flow12[0].permute(1, 2, 0).detach().cpu().numpy()


def _fit_size(size, maxsize=512):
    """Scale (W, H) so max dimension = maxsize. Matches notebook's fit_size()."""
    maxdim = max(size)
    ratio = maxsize / maxdim
    return int(size[0] * ratio), int(size[1] * ratio)


def get_flow_and_cc(
    frame1_path,
    frame2_path,
    flow_path,
    cc_path='',
    model=None,
    flow_maxsize=None,
    flow_size=None,
    half=False,
    num_flow_updates=20,
    dilation=1,
    edge_width=11,
    force=False,
):
    """Load or compute optical flow and consistency map between two frames.

    Matches notebook behavior:
    - Flow is computed and SAVED at computation resolution
    - CC map is computed and SAVED at computation resolution
    - Both are returned at computation resolution
    - The CALLER is responsible for resizing to target resolution at warp time

    Args:
        frame1_path: path to first frame image
        frame2_path: path to second frame image
        flow_path: path for caching flow .npy file
        cc_path: path for caching consistency map ('' to skip)
        model: RAFT model (required if flow needs computation)
        flow_maxsize: max dimension for flow computation (0/None = full size).
            Ignored if flow_size is provided.
        flow_size: exact (W, H) resolution for flow computation. When provided,
            overrides flow_maxsize. Matches notebook's flow_width_height = width_height.
        half: use half precision for RAFT
        dilation: dilation for consistency map
        edge_width: edge detection width for consistency map
        force: force recomputation even if cached

    Returns:
        (flow, cc) tuple — flow is (H, W, 2) numpy at computation resolution,
        cc is (H, W, 3) numpy at computation resolution or None.
    """
    import os
    from PIL import Image as PILImage

    cc = None

    # Try loading cached results (stored at computation resolution)
    if not force and os.path.exists(flow_path):
        flow = np.load(flow_path)
        if cc_path and os.path.exists(cc_path):
            # Load with PIL (RGB) to match notebook — cv2.imread returns BGR
            from PIL import Image as _PILImage
            cc = np.array(_PILImage.open(cc_path).convert('RGB'))
        return flow, cc

    # Need to compute — model is required
    if model is None:
        raise ValueError("RAFT model required for flow computation but not provided")

    # Determine flow computation size
    # If flow_size is provided, use it directly (notebook: flow_width_height = width_height)
    # Otherwise, derive from flow_maxsize and source image dimensions
    if flow_size is None:
        img1 = PILImage.open(frame1_path).convert('RGB')
        orig_size = img1.size  # (W, H)
        if flow_maxsize and flow_maxsize not in (0, -1) and max(orig_size) > flow_maxsize:
            flow_size = _fit_size(orig_size, flow_maxsize)
        # else flow_size stays None → full video resolution

    # Load frames at flow resolution
    f1 = _load_frame_for_flow(frame1_path, size=flow_size)
    f2 = _load_frame_for_flow(frame2_path, size=flow_size)

    # Compute backward flow (2→1) — this is the main flow used for warping
    # Notebook: flow21 = raft_model(frame_2, frame_1)
    flow_bwd = compute_flow(f2, f1, model, half=half, num_flow_updates=num_flow_updates)

    # Crop to flow_size if needed (notebook: flow21 = flow21[:size[1], :size[0], ...])
    if flow_size is not None:
        flow_bwd = flow_bwd[:flow_size[1], :flow_size[0], ...]

    # Magnitude threshold: zero out tiny flow on nearly-static frames
    # to avoid noisy CC maps (matches notebook's mag_thresh=0.5 clamping)
    mag = np.sqrt(flow_bwd[..., 0] ** 2 + flow_bwd[..., 1] ** 2)
    mag_thresh = 0.5
    if mag.max() < mag_thresh:
        flow_bwd_clamped = np.where(mag[..., None] < mag_thresh, 0, flow_bwd)
    else:
        flow_bwd_clamped = flow_bwd

    # Save backward flow at computation resolution (notebook does the same)
    os.makedirs(os.path.dirname(flow_path) or '.', exist_ok=True)
    np.save(flow_path, flow_bwd_clamped)

    # Compute consistency map if cc_path requested
    if cc_path:
        # Forward flow (1→2) needed for CC map
        flow_fwd = compute_flow(f1, f2, model, half=half, num_flow_updates=num_flow_updates)
        if flow_size is not None:
            flow_fwd = flow_fwd[:flow_size[1], :flow_size[0], ...]
        from vibewarp.flow.consistency import make_cc_map
        # Notebook: make_cc_map(flow12, flow21_clamped)
        cc = make_cc_map(flow_fwd, flow_bwd_clamped, dilation=dilation, edge_width=edge_width)
        # Save with PIL (RGB) to match notebook — cv2.imwrite would swap R/B channels
        from PIL import Image as _PILImage
        _PILImage.fromarray(cc.astype(np.uint8)).save(cc_path)

    return flow_bwd_clamped.astype(np.float32), cc


class InputPadder:
    """Pads images so dimensions are divisible by a given factor."""

    def __init__(self, dims, mode='sintel', divisor=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (divisor - (self.ht % divisor)) % divisor
        pad_wd = (divisor - (self.wd % divisor)) % divisor
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [torch.nn.functional.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
