"""Spatial trust weighting for temporal losses, independent of image encoders."""
import torch
import torch.nn.functional as F


def resize_trust(mask, reference):
    mask = torch.as_tensor(mask, device=reference.device, dtype=torch.float32).detach()
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[..., :1].permute(2, 0, 1)[None]
    if mask.ndim != 4 or mask.shape[1] != 1 or not torch.isfinite(mask).all():
        raise ValueError('Temporal trust mask must be a finite single-channel spatial map')
    if mask.shape[0] not in (1, reference.shape[0]):
        raise ValueError('Temporal trust mask batch does not match guidance batch')
    return F.interpolate(mask.clamp(0, 1), reference.shape[-2:], mode='area')


def masked_spherical_loss(image, target, mask):
    from vibewarp.core.losses import spherical_dist_loss
    if mask is None or bool((mask == 1).all()):
        return spherical_dist_loss(image.float(), target.float()).sum()
    mask = resize_trust(mask, image)
    # Keep the historical width-vector definition, with trust-weighted vectors
    # and row coverage. Do not amplify tiny trusted regions by their area.
    root = mask.sqrt()
    x = F.normalize(image.float()*root, dim=-1)
    y = F.normalize(target.float()*root, dim=-1)
    distance = (x-y).norm(dim=-1).div(2).clamp(max=1-1e-7)
    return (distance.arcsin().square()*2*mask.mean(-1)).sum()


def masked_lpips_loss(image, target, mask, spatial_model):
    mask = resize_trust(mask, image)
    # Retain actual perceptual context outside trust, without input gradients.
    context = image*mask + image.detach()*(1-mask)
    loss_map = spatial_model(context.float(), target.float())
    if loss_map.ndim != 4:
        raise ValueError('Trusted pixel guidance requires spatial LPIPS output')
    weights = resize_trust(mask, loss_map)
    return (loss_map*weights).flatten(1).mean(1).sum()
