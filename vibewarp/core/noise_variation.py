"""Deterministic, variance-preserving variation of fixed sampler noise."""
import hashlib
import math

import torch


def wiggle_offsets(seed, frame, pixels, direction=None):
    """Absolute-frame offsets, independent of noise/augmentation RNG streams."""
    if type(pixels) is not int or pixels < 0:
        raise ValueError('Noise wiggle must be a nonnegative integer')
    direction=(0.,0.) if direction is None else direction
    if (not isinstance(direction,(list,tuple)) or len(direction)!=2
            or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in direction)):
        raise ValueError('Noise direction must contain two finite numbers [x, y]')
    if pixels == 0 or frame == 0:
        return 0,0
    x,y=direction
    scale=max(abs(x),abs(y))
    if scale:
        x,y=x/scale,y/scale
        length=math.hypot(x,y)
        # Derive the cumulative position directly for exact absolute-frame resume.
        return round(frame*pixels*y/length),round(frame*pixels*x/length)
    key=f'vibewarp-noise-wiggle-v1:{int(seed)}:{int(frame)}'.encode()
    digest=hashlib.blake2b(key,digest_size=16).digest()
    span=2*pixels+1
    return (int.from_bytes(digest[:8],'little')%span-pixels,
            int.from_bytes(digest[8:],'little')%span-pixels)


def vary_fixed_noise(base, strength, seed, frame, *, wiggle=0, direction=None):
    """Mix independent unit Gaussian noise without consuming the sampler RNG.

    Wiggle translates the fixed component with integer wraparound first.
    Strength is the fresh-noise share of variance. Keep frame zero and both
    controls disabled byte-identical to the original draw. Finite-sample mean/std
    fluctuate naturally; no image-wise normalization changes the Gaussian law.
    """
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Fixed seed randomness must be between 0 and 1')
    dy,dx=wiggle_offsets(seed,frame,wiggle,direction)
    if frame == 0:
        return base
    if dy or dx:
        base=torch.roll(base,shifts=(dy%base.shape[-2],dx%base.shape[-1]),dims=(-2,-1))
    if strength == 0:
        return base
    key = f'vibewarp-fixed-noise-v1:{int(seed)}:{int(frame)}'.encode()
    fresh_seed = int.from_bytes(hashlib.blake2b(key,digest_size=8).digest(),'little')
    generator = torch.Generator(device=base.device).manual_seed(fresh_seed)
    fresh = torch.randn(base.shape,device=base.device,dtype=base.dtype,generator=generator)
    if strength == 1:
        return fresh
    return base*math.sqrt(1-strength)+fresh*math.sqrt(strength)
