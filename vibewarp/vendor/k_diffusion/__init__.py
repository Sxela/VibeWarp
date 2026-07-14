"""Vendored k-diffusion — https://github.com/crowsonkb/k-diffusion

Only the modules needed by WarpFusion are included:
- sampling: K-diffusion samplers and sigma schedules
- external: CompVis model wrappers (CompVisDenoiser, CompVisVDenoiser)
- utils: append_dims and other utilities
"""
from . import external, sampling, utils
