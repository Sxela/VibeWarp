"""Optional notebook execution modes for secondary guidance and CLIP batches."""
import torch

from vibewarp.core.losses import spherical_dist_loss


class SecondaryPrecision(torch.nn.Module):
    """Use device-selected half-precision weights and secondary inference.

    Returned predictions remain FP32 for the existing guidance/loss calculations.
    Input gradients remain enabled, including guidance-backward recomputation.
    """
    def __init__(self, inner, precision='auto', device='cuda'):
        super().__init__()
        device = torch.device(device)
        if precision not in ('fp32', 'fp16', 'bf16', 'auto'):
            raise ValueError(f'Unknown secondary precision: {precision}')
        # Legacy requests are accepted, but never override the enforced policy.
        if device.type == 'cuda':
            with torch.cuda.device(device):
                precision = 'bf16' if torch.cuda.is_bf16_supported(including_emulation=False) else 'fp16'
        else:
            precision = 'bf16' if device.type == 'cpu' else 'fp16'
        self.precision = precision
        self.dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16}[precision]
        self.inner = inner.to(dtype=self.dtype)

    def forward(self, x, t):
        with torch.autocast(x.device.type, dtype=self.dtype):
            value = self.inner(x, t)
        return type(value)(value.v.float(), value.pred.float(), value.eps.float())


def grouped_clip_gradient(image, encoder, targets, weights, make_cuts, *, batches,
                          group=1, scale=1., balance=False):
    """Generate each cut batch in order, then encode groups with equal total weight."""
    if type(group) is not int or group < 1 or type(batches) is not int or batches < 1:
        raise ValueError('CLIP cut batches and batch group must be positive integers')
    result = torch.zeros_like(image)
    for offset in range(0, batches, group):
        from vibewarp.cancellation import raise_if_cancelled
        raise_if_cancelled()
        count = min(group, batches-offset)
        cuts = torch.cat([make_cuts(offset+i) for i in range(count)])
        raise_if_cancelled()
        encoded = encoder.image(cuts)
        raise_if_cancelled()
        loss = (spherical_dist_loss(encoded[:, None], targets[None]) * weights).sum(-1).mean()
        grad = torch.autograd.grad(loss, image, retain_graph=True)[0]
        if balance:
            grad = grad / grad.square().mean().sqrt().clamp_min(1e-8)
        result += grad * scale * (count/batches)
    return result
