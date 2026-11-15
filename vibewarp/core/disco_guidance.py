"""Noise-relative caps on the actual DDIM/PLMS guidance displacement."""
from contextlib import contextmanager

import torch


def cap_guidance_update(guided, baseline, noise_std, percent, stats=None):
    """Bound each image's RMS displacement, retaining the update direction.

    100% means one current marginal VP noise standard deviation in pixel
    units. It is not a per-pixel maximum or a gradient-norm cap.
    """
    if percent <= 0:
        return guided
    delta = guided.float()-baseline.float()
    dims = tuple(range(1,delta.ndim))
    rms = delta.square().mean(dims,keepdim=True).sqrt()
    if not torch.isfinite(rms).all():
        raise FloatingPointError('Nonfinite sampler guidance displacement')
    sigma = torch.as_tensor(noise_std,device=delta.device,dtype=torch.float32)
    if sigma.numel() == 1:
        sigma = sigma.reshape((1,)*delta.ndim)
    else:
        sigma = sigma.reshape(delta.shape[0],-1)[:,0].reshape((-1,)+(1,)*(delta.ndim-1))
    budget = sigma*(percent/100.)
    factor = (budget/rms.clamp_min(1e-20)).clamp(max=1)
    # Preserve uncapped values exactly, including when cancellation in the
    # baseline subtraction would otherwise introduce a rounding difference.
    capped = baseline.float()+delta*factor
    result = torch.where(factor < 1,capped.to(guided.dtype),guided)
    if stats is not None:
        actual = (result.float()-baseline.float()).square().mean(dims).sqrt()
        stats.update(guidance_cap_percent=float(percent), guidance_update_rms=float(actual.max()),
            guidance_raw_update_rms=float(rms.max()),guidance_cap_engaged=bool((factor<1).any()),
            guidance_noise_budget_rms=float(budget.max()), guidance_cap_scale=float(factor.min()))
    return result


@contextmanager
def guidance_noise_cap(diffusion, percent, stats=None):
    """Install a render-local sampler hook, restoring it even on cancellation."""
    missing = object()
    previous = getattr(diffusion,'guidance_update_cap',missing)
    diffusion.guidance_update_cap = (
        lambda guided,baseline,noise_std: cap_guidance_update(guided,baseline,noise_std,percent,stats)
    ) if percent > 0 else None
    try:
        yield
    finally:
        if previous is missing:
            del diffusion.guidance_update_cap
        else:
            diffusion.guidance_update_cap = previous
