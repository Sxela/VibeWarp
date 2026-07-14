"""VAE encode/decode with tiled VAE support.

These functions are designed to be monkey-patched onto the SD model instance,
or called standalone with the model passed explicitly. They support tiled
encoding/decoding for high-resolution images that don't fit in VRAM.
"""

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def get_weighting(h, w, Ly, Lx, device, clip_min=0.0001, clip_max=1.0):
    """Generate border-distance weighting for tiled VAE overlap blending."""
    weighting = _delta_border(h, w)
    weighting = torch.clip(weighting, clip_min, clip_max)
    weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)
    return weighting


def _meshgrid(h, w):
    """Create coordinate grid of shape (h, w, 2)."""
    y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
    x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)
    return torch.cat([y, x], dim=-1)


def _delta_border(h, w):
    """Compute normalized distance to image border.

    Min distance = 0 at border, max dist = 0.5 at center.

    Returns:
        Tensor of shape (h, w).
    """
    lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2).float()
    arr = _meshgrid(h, w).float() / lower_right_corner.clamp(min=1)
    dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
    dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
    edge_dist = torch.min(
        torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1
    )[0]
    return edge_dist


def get_fold_unfold(x, kernel_size, stride, df=1, uf=1):
    """Create fold/unfold operations for tiled processing.

    Args:
        x: input tensor (bs, c, h, w)
        kernel_size: tile size (h, w)
        stride: stride between tiles (h, w)
        df: downscale factor
        uf: upscale factor

    Returns:
        (fold, unfold, normalization, weighting) tuple
    """
    bs, nc, h, w = x.shape
    Ly = (h - kernel_size[0]) // stride[0] + 1
    Lx = (w - kernel_size[1]) // stride[1] + 1

    # Determine effective kernel and stride sizes based on scale factors
    if uf == 1 and df == 1:
        eff_ks = kernel_size
        eff_stride = stride
        out_size = x.shape[2:]
    elif uf > 1 and df == 1:
        eff_ks = (kernel_size[0] * uf, kernel_size[1] * uf)
        eff_stride = (stride[0] * uf, stride[1] * uf)
        out_size = (h * uf, w * uf)
    elif df > 1 and uf == 1:
        eff_ks = (kernel_size[0] // df, kernel_size[1] // df)
        eff_stride = (stride[0] // df, stride[1] // df)
        out_size = (h // df, w // df)
    else:
        raise NotImplementedError(f"uf={uf}, df={df} not supported")

    # Input unfold always uses original kernel/stride
    fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
    unfold = torch.nn.Unfold(**fold_params)

    # Output fold uses effective sizes
    if uf == 1 and df == 1:
        fold = torch.nn.Fold(output_size=out_size, **fold_params)
    else:
        fold_params2 = dict(kernel_size=eff_ks, dilation=1, padding=0, stride=eff_stride)
        fold = torch.nn.Fold(output_size=out_size, **fold_params2)

    # Weighting: (1, eff_ks_h * eff_ks_w, Ly*Lx)
    weighting = get_weighting(eff_ks[0], eff_ks[1], Ly, Lx, x.device).to(x.dtype)
    normalization = fold(weighting).view(1, 1, *out_size)
    weighting = weighting.view((1, 1, eff_ks[0], eff_ks[1], Ly * Lx))

    normalization = torch.where(normalization == 0., 1e-6, normalization)
    return fold, unfold, normalization, weighting


@torch.inference_mode()
def tiled_encode(model, x, num_tiles=None, ks=None, stride=None, padding=None, df=8):
    """Encode an image using tiled VAE for large resolutions.

    Args:
        model: SD model with first_stage_model
        x: input image tensor (bs, c, h, w)
        num_tiles: (rows, cols) tile grid, overrides ks
        ks: kernel size per tile
        stride: stride between tiles
        padding: overlap padding factor
        df: VAE downscale factor (usually 8)

    Returns:
        Encoded latent tensor.
    """
    bs, nc, h, w = x.shape

    if num_tiles is not None:
        # Tile size derived from image dims — already in pixel space; no df scaling
        ks = [h // num_tiles[0], w // num_tiles[1]]
    elif ks is None:
        ks = [h, w]  # single tile, pixel space
    else:
        # Caller-supplied ks may be in latent space — scale to pixel space if needed
        ks = [o * df for o in ks] if max(ks) < h else list(ks)

    if padding is not None:
        stride = [int(ks[0] * padding[0]), int(ks[1] * padding[1])]
    elif stride is None:
        stride = [ks[0] // 2, ks[1] // 2]
    elif num_tiles is not None:
        # stride passed but num_tiles also set — treat stride as pixel-space
        stride = list(stride)
    else:
        stride = [o * df for o in stride] if max(stride) < h else list(stride)

    # Pad to be divisible by kernel size
    target_h = math.ceil(h / ks[0]) * ks[0]
    target_w = math.ceil(w / ks[1]) * ks[1]
    if target_h != h or target_w != w:
        x = F.pad(x, (0, target_w - w, 0, target_h - h), mode='reflect')

    ks = (min(ks[0], x.shape[2]), min(ks[1], x.shape[3]))
    stride = (min(stride[0], x.shape[2]), min(stride[1], x.shape[3]))

    fold, unfold, normalization, weighting = get_fold_unfold(x, ks, stride, df=df)

    z = unfold(x)
    z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))

    # Encode each tile through the SAME pair the untiled path uses, because where
    # scale_factor gets applied differs by architecture:
    #   ldm  (SD1.5): encode_first_stage -> posterior; get_first_stage_encoding
    #                 samples it AND multiplies by scale_factor.
    #   sgm  (SDXL):  encode_first_stage ALREADY returns scale_factor * z, and
    #                 get_first_stage_encoding is an identity (setup_sdxl_model).
    # Calling first_stage_model.encode directly bypasses sgm's scaling, which left
    # SDXL latents 1/scale_factor (~7.7x) too large — invisible for txt2img (which
    # never encodes) and catastrophic for img2img. Mirror of the tiled_decode bug.
    encode = getattr(model, '_orig_encode_first_stage', None) or model.encode_first_stage
    scale = getattr(model, '_orig_get_first_stage_encoding', None) or model.get_first_stage_encoding

    with torch.amp.autocast('cuda'):
        output_list = [
            scale(encode(z[:, :, :, :, i]))
            for i in range(z.shape[-1])
        ]

    o = torch.stack(output_list, axis=-1)
    o = o * weighting
    o = o.view((o.shape[0], -1, o.shape[-1]))
    decoded = fold(o)
    decoded = decoded / normalization

    return decoded[..., :h // df, :w // df]


@torch.inference_mode()
def tiled_decode(model, z, num_tiles=None, ks=None, stride=None, padding=None, df=8):
    """Decode a latent using tiled VAE for large resolutions.

    Args:
        model: SD model with first_stage_model
        z: latent tensor (bs, c, h, w)
        num_tiles: (rows, cols) tile grid
        ks: kernel size per tile (in latent space)
        stride: stride between tiles (in latent space)
        padding: overlap padding factor
        df: VAE downscale factor (usually 8)

    Returns:
        Decoded image tensor.
    """
    # decode_first_stage rescales BEFORE tiling (ldm applies 1/scale_factor,
    # then decodes); first_stage_model.decode below is the raw decoder, so the
    # rescale must happen here. Missing this washed out every tiled render
    # (contrast compressed by exactly scale_factor ≈ 0.18215).
    z = 1.0 / model.scale_factor * z
    bs, nc, h, w = z.shape

    if num_tiles is not None:
        ks_h = h // num_tiles[0]
        ks_w = w // num_tiles[1]
        ks = (ks_h, ks_w)
    elif ks is None:
        ks = (h, w)

    if padding is not None:
        stride = [int(ks[0] * padding[0]), int(ks[1] * padding[1])]
    elif stride is None:
        stride = [ks[0] // 2, ks[1] // 2]

    # Pad to be divisible by kernel size
    target_h = math.ceil(h / ks[0]) * ks[0]
    target_w = math.ceil(w / ks[1]) * ks[1]
    if target_h != h or target_w != w:
        z = F.pad(z, (0, target_w - w, 0, target_h - h), mode='reflect')

    ks = (min(ks[0], z.shape[2]), min(ks[1], z.shape[3]))
    stride = (min(stride[0], z.shape[2]), min(stride[1], z.shape[3]))

    fold, unfold, normalization, weighting = get_fold_unfold(z, ks, stride, uf=df)

    patches = unfold(z)
    patches = patches.view((patches.shape[0], -1, ks[0], ks[1], patches.shape[-1]))

    with torch.amp.autocast('cuda'):
        output_list = [
            model.first_stage_model.decode(patches[:, :, :, :, i])
            for i in range(patches.shape[-1])
        ]

    o = torch.stack(output_list, axis=-1)
    o = o * weighting
    o = o.view((o.shape[0], -1, o.shape[-1]))
    decoded = fold(o)
    decoded = decoded / normalization

    return decoded[..., :h * df, :w * df]


def patch_model_for_tiled_vae(sd_model, vae_config) -> None:
    """Monkey-patch sd_model to use tiled VAE encode/decode.

    Patches three methods:
      - encode_first_stage → tiled_encode
      - decode_first_stage → tiled_decode
      - get_first_stage_encoding → identity when input is already a scaled tensor
        (avoids double-scaling since tiled_encode already scales internally)

    Args:
        sd_model: loaded SD model instance
        vae_config: VaeConfig dataclass
    """
    import types

    cfg = vae_config
    num_tiles = list(cfg.num_tiles) if cfg.num_tiles else None
    # tile_size / vae_stride are in latent space; tiled_encode scales them by df=8
    ks = cfg.tile_size
    stride = cfg.vae_stride
    padding = list(cfg.vae_padding) if cfg.vae_padding else None

    # Build fixed kwargs for encode/decode calls
    _encode_kw = dict(num_tiles=num_tiles, df=8)
    _decode_kw = dict(num_tiles=num_tiles, df=8)
    if num_tiles is None:
        _encode_kw['ks'] = (ks, ks)
        _decode_kw['ks'] = (ks, ks)
        if padding is not None:
            _encode_kw['padding'] = padding
            _decode_kw['padding'] = padding
        else:
            _encode_kw['stride'] = (stride, stride)
            _decode_kw['stride'] = (stride, stride)

    # Save originals as instance attributes so unpatch can restore them
    sd_model._orig_encode_first_stage = sd_model.encode_first_stage
    sd_model._orig_decode_first_stage = sd_model.decode_first_stage
    sd_model._orig_get_first_stage_encoding = sd_model.get_first_stage_encoding

    _ek = _encode_kw
    _dk = _decode_kw

    def _tiled_encode_first_stage(self, x):
        return tiled_encode(self, x, **_ek)

    def _tiled_decode_first_stage(self, z):
        return tiled_decode(self, z, **_dk)

    def _patched_get_first_stage_encoding(self, encoder_posterior):
        # tiled_encode returns a finished, correctly scaled latent (it runs each tile
        # through the model's own encode + get_first_stage_encoding), so scaling it
        # again here would double-apply scale_factor.
        if isinstance(encoder_posterior, torch.Tensor):
            return encoder_posterior
        return self._orig_get_first_stage_encoding(encoder_posterior)

    sd_model.encode_first_stage = types.MethodType(_tiled_encode_first_stage, sd_model)
    sd_model.decode_first_stage = types.MethodType(_tiled_decode_first_stage, sd_model)
    sd_model.get_first_stage_encoding = types.MethodType(_patched_get_first_stage_encoding, sd_model)

    tile_px = ks * 8
    print(f"Tiled VAE: tile={tile_px}px (latent {ks}), "
          f"stride={stride * 8}px, num_tiles={num_tiles}, padding={padding}")


def unpatch_model_tiled_vae(sd_model) -> None:
    """Restore original VAE methods saved by patch_model_for_tiled_vae."""
    for attr in ('_orig_encode_first_stage', '_orig_decode_first_stage',
                 '_orig_get_first_stage_encoding'):
        public = attr.replace('_orig_', '')
        if hasattr(sd_model, attr):
            setattr(sd_model, public, getattr(sd_model, attr))
            delattr(sd_model, attr)


def pil_to_latent(model, img, size=None, device='cuda'):
    """Convert a PIL image to a latent representation.

    Args:
        model: SD model with encode_first_stage
        img: PIL Image
        size: optional (W, H) to resize to
        device: torch device

    Returns:
        Latent tensor.
    """
    if size:
        img = img.resize(size, Image.LANCZOS)
    arr = np.array(img.convert('RGB')).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    tensor = (2.0 * tensor - 1.0).half().to(device)
    with torch.inference_mode(), torch.amp.autocast('cuda'):
        return model.get_first_stage_encoding(model.encode_first_stage(tensor))


def latent_to_pil(model, lat):
    """Convert a latent tensor to a PIL image.

    Args:
        model: SD model with decode_first_stage
        lat: latent tensor

    Returns:
        PIL Image.
    """
    import torchvision.transforms.functional as TF
    with torch.inference_mode(), torch.amp.autocast('cuda'):
        img = model.decode_first_stage(lat.half().cuda())[0]
    return TF.to_pil_image(img.float().add(1).div(2).clamp(0, 1))
