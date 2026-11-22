"""Spatial clean-prediction guidance. Also loadable as a standalone ComfyUI node."""

import torch
import torch.nn.functional as F


def motion_grid(flow, consistency, feature_size, prediction_size):
    """Return grid_sample coordinates and a conservative mask for one frame pair.

    Flow vectors are in pixels of the flow canvas. FLUX.2 warps its unpacked
    VAE features, but applies the correction on its coarser packed prediction.
    """
    flow = torch.as_tensor(flow, dtype=torch.float32)
    trust = torch.as_tensor(consistency, dtype=torch.float32)
    if flow.ndim != 3 or flow.shape[-1] != 2 or trust.shape != flow.shape[:2]:
        raise ValueError("Backward flow and consistency must share a pixel canvas")
    if not torch.isfinite(flow).all() or not torch.isfinite(trust).all():
        raise ValueError("Non-finite temporal guidance flow or consistency")
    if ((trust < 0) | (trust > 1)).any():
        raise ValueError("Temporal guidance consistency values must be in [0, 1]")
    h, w = trust.shape
    fh, fw = feature_size

    def source_coordinates(vectors):
        height, width = vectors.shape[:2]
        y, x = torch.meshgrid(torch.arange(height, device=flow.device),
                              torch.arange(width, device=flow.device), indexing="ij")
        sx, sy = x + vectors[..., 0], y + vectors[..., 1]
        return sx, sy, (sx >= 0) & (sx <= width - 1) & (sy >= 0) & (sy <= height - 1)

    def reduce_mask(mask):
        return F.interpolate(mask[None, None].float(), size=prediction_size, mode="area") >= .999

    _, _, valid_pixels = source_coordinates(flow)
    scaled = F.interpolate(flow.permute(2, 0, 1)[None], size=feature_size,
                           mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    sx, sy, valid_features = source_coordinates(scaled * flow.new_tensor([fw / w, fh / h]))
    grid = torch.stack((2 * sx / max(1, fw - 1) - 1,
                        2 * sy / max(1, fh - 1) - 1), dim=-1)[None]
    return grid, reduce_mask((trust >= .999) & valid_pixels) & reduce_mask(valid_features)


def warp_target(samples, flow, trust, autoencoder=None):
    """Warp one image or singleton-time latent, in its codec's feature space."""
    singleton_time = samples.ndim == 5 and samples.shape[2] == 1
    previous = samples.squeeze(2) if singleton_time else samples
    if previous.ndim != 4 or previous.shape[0] != 1:
        raise ValueError("Temporal guidance requires one sampled image per frame")
    features = previous.float()
    bn = getattr(autoencoder, "bn", None)
    if bn is not None and getattr(autoencoder, "ps", None) == [2, 2]:
        mean = bn.running_mean.view(1, -1, 1, 1).to(previous).float()
        std = (bn.running_var + autoencoder.bn_eps).sqrt().view(1, -1, 1, 1).to(previous).float()
        features = F.pixel_shuffle(features * std + mean, 2)
    else:
        bn = None
    grid, mask = motion_grid(flow, trust, features.shape[-2:], previous.shape[-2:])
    warped = F.grid_sample(features, grid.to(features), mode="bilinear",
                           padding_mode="border", align_corners=True)
    if bn is not None:
        warped = (F.pixel_unshuffle(warped, 2) - mean) / std
    target = warped.to(previous.dtype)
    return (target.unsqueeze(2), mask.unsqueeze(2)) if singleton_time else (target, mask)


def sigma_window(sigmas, start, end):
    """Use indices of the actual scheduled steps. The terminal zero is excluded."""
    steps = len(sigmas) - 1
    indices = [i for i in range(steps) if start <= i / max(1, steps - 1) <= end]
    if indices:
        return float(sigmas[indices[-1]]), float(sigmas[indices[0]])
    return None


def project(clean, target, mask, strength):
    if clean.shape != target.shape:
        raise ValueError(f"Temporal target shape {target.shape} differs from prediction {clean.shape}")
    corrected = clean.float() + strength * (target.float() - clean.float())
    return torch.where(mask, corrected.to(clean.dtype), clean)


class VibeWarpTemporalGuidance:
    CATEGORY = "VibeWarp"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "vae": ("VAE",), "canvas": ("LATENT",), "motion": ("LATENT",),
            "strength": ("FLOAT", {"default": 0.2, "min": 0, "max": 1}),
            "start": ("FLOAT", {"default": 0.1, "min": 0, "max": 1}),
            "end": ("FLOAT", {"default": 0.85, "min": 0, "max": 1}),
        }, "optional": {"previous": ("LATENT",), "initial_image": ("IMAGE",)}}

    def patch(self, model, vae, canvas, motion, strength, start, end, previous=None, initial_image=None):
        if not (0 <= start < end <= 1 and 0 <= strength <= 1):
            raise ValueError("Invalid temporal guidance strength or step window")
        if strength == 0:
            return (model,)
        h, w = canvas["samples"].shape[-2:]
        if initial_image is not None:
            factor = vae.spacial_compression_encode()
            pixels = F.interpolate(initial_image.movedim(-1, 1), size=(h*factor, w*factor),
                                   mode="bilinear", align_corners=False).movedim(1, -1)
            history = vae.encode(pixels[..., :3])
        else:
            history = previous["samples"]
        if history.shape[-2:] != (h, w):
            raise ValueError("Sampling dimensions changed between temporal guidance frames")
        vectors = motion["samples"][0]
        target, mask = warp_target(history, vectors[:2].movedim(0, -1), vectors[2],
                                   vae.first_stage_model)
        if not mask.any():
            return (model,)
        target = model.model.process_latent_in(target).to(model.load_device)
        mask = mask.to(model.load_device)
        # KSampler and custom samplers both pass their executed schedule here.
        # Cache bounds on the first evaluation, preserving other model patches.
        bounds = None
        last_schedule = None

        def correct(args):
            nonlocal bounds, last_schedule
            schedule = args['model_options']['transformer_options']['sample_sigmas']
            if schedule is not last_schedule:
                bounds = sigma_window(schedule, start, end)
                last_schedule = schedule
            clean = args["denoised"]
            if bounds and bounds[0] <= float(args["sigma"].flatten()[0]) <= bounds[1]:
                if target.ndim == 4 and clean.ndim == 5 and clean.shape[2] == 1:
                    return project(clean, target.unsqueeze(2), mask.unsqueeze(2), strength)
                if target.ndim == 5 and target.shape[2] == 1 and clean.ndim == 4:
                    return project(clean, target.squeeze(2), mask.squeeze(2), strength)
                return project(clean, target, mask, strength)
            return clean

        patched = model.clone()
        patched.set_model_sampler_post_cfg_function(correct)
        return (patched,)


NODE_CLASS_MAPPINGS = {"VibeWarpTemporalGuidance": VibeWarpTemporalGuidance}
