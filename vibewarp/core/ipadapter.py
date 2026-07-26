"""IP-Adapter loading, CLIP vision encoding, and UNet hook management.

Extracted from notebook's IP-Adapter integration:
- load_ipadapter_model() — load adapter weights from checkpoint
- clip_preprocess() / encode_clip_vision() — generate CLIP embeddings from images
- hook_ipadapter() / unhook_ipadapter() — inject/remove attention patches in UNet

IP-Adapter adds image-conditioned attention to the UNet's cross-attention blocks.
For each block, the standard text-conditioned attention output is augmented with
an additional attention pass using CLIP vision embeddings projected through
learned K/V layers: out = out + weight * Attention(Q, IP_K, IP_V)
"""

import gc
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from vibewarp.config import IPAdapterConfig, IPAdapterEntry


# ---- CLIP Vision preprocessing ----

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


def clip_preprocess(
    image: torch.Tensor,
    size: int = 224,
) -> torch.Tensor:
    """Preprocess an image tensor for CLIP vision encoding.

    Matches the notebook's clip_preprocess() function.

    Args:
        image: (B, H, W, 3) float tensor in [0, 1] (HWC format)
        size: CLIP input resolution (224 for ViT-H/G)

    Returns:
        (B, 3, size, size) normalized tensor ready for CLIP.
    """
    device = image.device
    dtype = image.dtype
    mean = CLIP_MEAN.to(device=device, dtype=dtype)
    std = CLIP_STD.to(device=device, dtype=dtype)

    # Resize to at least `size` on shortest side
    scale = size / min(image.shape[1], image.shape[2])
    new_h = round(scale * image.shape[1])
    new_w = round(scale * image.shape[2])

    # HWC → CHW for interpolation
    image = image.movedim(-1, 1)
    image = nn.functional.interpolate(
        image, size=(new_h, new_w), mode='bicubic', antialias=True,
    )

    # Center crop
    h = (image.shape[2] - size) // 2
    w = (image.shape[3] - size) // 2
    image = image[:, :, h:h + size, w:w + size]

    # Quantize and re-normalize (matches notebook behavior)
    image = torch.clip((255.0 * image), 0, 255).round() / 255.0

    # Normalize with ImageNet mean/std
    return (image - mean.view(3, 1, 1)) / std.view(3, 1, 1)


def load_image_for_clip(
    image_path: str,
    device: str = 'cuda',
) -> torch.Tensor:
    """Load an image file and prepare it for CLIP encoding.

    Args:
        image_path: path to image file
        device: target device

    Returns:
        (1, H, W, 3) float tensor in [0, 1] on device.
    """
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device)
    return tensor


def encode_clip_vision(
    clip_model: Any,
    image: torch.Tensor,
) -> Dict[str, Any]:
    """Encode an image through CLIP vision model.

    Args:
        clip_model: CLIP vision model (e.g., from transformers or comfy)
        image: (B, H, W, 3) float tensor in [0, 1]

    Returns:
        Dict with the ComfyUI/WarpFusion ``image_embeds`` and complete
        ``hidden_states`` outputs.
    """
    pixel_values = clip_preprocess(image)

    with torch.inference_mode():
        if hasattr(clip_model, '__call__'):
            outputs = clip_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
            )
        else:
            # Fallback for models with a forward() method
            outputs = clip_model.forward(
                pixel_values=pixel_values,
                output_hidden_states=True,
            )

    result = {}
    if hasattr(outputs, 'image_embeds'):
        result['image_embeds'] = outputs.image_embeds.cpu()
    elif isinstance(outputs, dict) and 'image_embeds' in outputs:
        result['image_embeds'] = outputs['image_embeds'].cpu()

    if hasattr(outputs, 'hidden_states'):
        result['hidden_states'] = tuple(x.cpu() for x in outputs.hidden_states)
        result['penultimate_hidden_states'] = result['hidden_states'][-2]
    elif isinstance(outputs, dict) and 'hidden_states' in outputs:
        result['hidden_states'] = tuple(x.cpu() for x in outputs['hidden_states'])
        result['penultimate_hidden_states'] = result['hidden_states'][-2]

    return result


def combine_clip_vision_outputs(
    outputs: List[Dict[str, torch.Tensor]],
    method: str = 'concat',
) -> Dict[str, Any]:
    """Combine ordered CLIP outputs before IP-Adapter image projection.

    The formulas match ComfyUI_IPAdapter_plus. ``concat`` preserves every
    image as a batch item; the hook later flattens their projected tokens into
    one attention context. Other methods reduce the image batch to one item.
    """
    if not outputs:
        raise ValueError("At least one CLIP vision output is required")
    allowed = {'concat', 'add', 'subtract', 'average', 'norm average'}
    if method not in allowed:
        raise ValueError(f"Unknown IP-Adapter embed combination: {method}")

    def combine(tensors):
        batch = torch.cat(tensors, dim=0)
        if method == 'concat' or batch.shape[0] == 1:
            return batch
        if method == 'add':
            return batch.sum(dim=0, keepdim=True)
        if method == 'subtract':
            return (batch[0] - batch[1:].mean(dim=0)).unsqueeze(0)
        if method == 'average':
            return batch.mean(dim=0, keepdim=True)
        epsilon = (torch.finfo(batch.dtype).eps
                   if batch.dtype.is_floating_point else 1e-12)
        norms = torch.linalg.vector_norm(
            batch, dim=0, keepdim=True).clamp_min(epsilon)
        return (batch / norms).mean(dim=0, keepdim=True)

    result = {}
    if all('image_embeds' in output for output in outputs):
        result['image_embeds'] = combine(
            [output['image_embeds'] for output in outputs])
    if all('hidden_states' in output for output in outputs):
        count = min(len(output['hidden_states']) for output in outputs)
        result['hidden_states'] = tuple(
            combine([output['hidden_states'][index] for output in outputs])
            for index in range(count))
        result['penultimate_hidden_states'] = result['hidden_states'][-2]
    if not result:
        raise ValueError("CLIP outputs contain no compatible embeddings")
    return result


def _actionable_clip_key_mismatches(missing, unexpected):
    """Ignore buffers saved by older CLIP exporters but not current Transformers."""
    harmless_unexpected = {
        'vision_model.embeddings.position_ids',
    }
    return (
        list(missing),
        [key for key in unexpected if key not in harmless_unexpected],
    )


def load_clip_vision_model(
    model_path: str,
    variant: str = 'vit_h',
    device: str = 'cpu',
) -> Any:
    """Load an h94 IP-Adapter encoder directory or standalone weight file."""
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

    if os.path.isdir(model_path):
        model = CLIPVisionModelWithProjection.from_pretrained(
            model_path, local_files_only=True)
    elif os.path.isfile(model_path):
        if variant == 'vit_g':
            config = CLIPVisionConfig(
                hidden_size=1664, intermediate_size=8192,
                num_hidden_layers=48, num_attention_heads=16,
                image_size=224, patch_size=14, projection_dim=1280)
        else:
            config = CLIPVisionConfig(
                hidden_size=1280, intermediate_size=5120,
                num_hidden_layers=32, num_attention_heads=16,
                image_size=224, patch_size=14, projection_dim=1024)
        model = CLIPVisionModelWithProjection(config)
        if model_path.lower().endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(model_path, device='cpu')
        else:
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        missing, unexpected = _actionable_clip_key_mismatches(
            missing, unexpected)
        if missing or unexpected:
            details = ', '.join((missing + unexpected)[:4])
            raise ValueError(
                f"CLIP vision checkpoint does not match {variant}: "
                f"{len(missing)} missing, {len(unexpected)} unexpected keys"
                + (f" ({details})" if details else ""))
    else:
        raise FileNotFoundError(f"CLIP vision model not found: {model_path}")

    return model.half().to(device=device).eval()


# ---- IP-Adapter model loading ----

def load_ipadapter_model(ckpt_path: str) -> dict:
    """Load IP-Adapter weights from a checkpoint file.

    Parses the checkpoint into 'image_proj' and 'ip_adapter' sections.

    Args:
        ckpt_path: path to .safetensors or .bin IP-Adapter checkpoint

    Returns:
        Dict with 'image_proj' and 'ip_adapter' keys.
    """
    from vibewarp.vendor.cldm.model import load_state_dict
    model = load_state_dict(ckpt_path, location='cpu')

    if ckpt_path.lower().endswith('.safetensors'):
        st_model = {'image_proj': {}, 'ip_adapter': {}}
        for key in model.keys():
            if key.startswith('image_proj.'):
                st_model['image_proj'][key.replace('image_proj.', '')] = model[key]
            elif key.startswith('ip_adapter.'):
                st_model['ip_adapter'][key.replace('ip_adapter.', '')] = model[key]

        # Sort ip_adapter keys by block index
        sorted_ip = {}
        sorted_keys = sorted(
            st_model['ip_adapter'].keys(),
            key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 0,
        )
        for key in sorted_keys:
            sorted_ip[key] = st_model['ip_adapter'][key]

        model = {'image_proj': st_model['image_proj'], 'ip_adapter': sorted_ip}

    if 'ip_adapter' not in model or not model['ip_adapter']:
        raise ValueError(f"Invalid IP-Adapter model: {ckpt_path}")

    return model


def detect_ipadapter_type(state_dict: dict) -> Dict[str, bool]:
    """Detect the IP-Adapter variant from its state dict.

    Args:
        state_dict: loaded IP-Adapter weights dict

    Returns:
        Dict with boolean flags: is_plus, is_full, is_faceid, is_sdxl.
    """
    ip_sd = state_dict.get('ip_adapter', {})
    proj_sd = state_dict.get('image_proj', {})

    is_full = 'proj.3.weight' in proj_sd
    is_faceid = '0.to_q_lora.down.weight' in ip_sd
    is_plus = (
        is_full
        or 'latents' in proj_sd
        or 'perceiver_resampler.proj_in.weight' in proj_sd
    )

    # Detect SDXL from cross_attention_dim
    is_sdxl = False
    for key in ip_sd:
        if 'to_k_ip.weight' in key:
            if ip_sd[key].shape[1] == 2048:
                is_sdxl = True
            break

    return {
        'is_plus': is_plus,
        'is_full': is_full,
        'is_faceid': is_faceid,
        'is_sdxl': is_sdxl,
    }


# ---- CLIP model auto-detection ----

# IP-Adapter variants that use ViT-H (1024-dim) CLIP model
VIT_H_VARIANTS = {
    'ipadapter_sd15', 'ipadapter_sd15_light', 'ipadapter_sd15_plus',
    'ipadapter_sd15_plus_face', 'ipadapter_sd15_full_face',
    'ipadapter_sdxl_vit_h', 'ipadapter_sdxl_plus_vit_h',
    'ipadapter_sdxl_plus_face_vit_h',
    'ipadapter_sd15_faceid_plus', 'ipadapter_sd15_faceid_plus_v2',
}

# IP-Adapter variants that use ViT-bigG (1280-dim) CLIP model
VIT_G_VARIANTS = {
    'ipadapter_sd15_vit_G', 'ipadapter_sdxl',
}


def detect_clip_variant(
    adapter_name: str = '',
    state_dict: Optional[dict] = None,
) -> str:
    """Auto-detect which CLIP vision model an IP-Adapter needs.

    Detection strategy (in priority order):
    1. Match adapter_name against known variant lists
    2. Check adapter_name for 'vit_h' or 'vit_g'/'vit_G' substrings
    3. Examine image_proj weight dimensions in state_dict:
       - Input dim 1024 → ViT-H
       - Input dim 1280 → ViT-bigG
    4. Default to ViT-H (most common)

    Args:
        adapter_name: IP-Adapter model key/name (e.g., 'ipadapter_sd15_plus')
        state_dict: loaded IP-Adapter state dict (optional, for dimension detection)

    Returns:
        'vit_h' or 'vit_g' string.
    """
    # 1. Check known variant lists
    name_lower = adapter_name.lower().replace('-', '_')
    if adapter_name in VIT_H_VARIANTS or name_lower in {v.lower() for v in VIT_H_VARIANTS}:
        return 'vit_h'
    if adapter_name in VIT_G_VARIANTS or name_lower in {v.lower() for v in VIT_G_VARIANTS}:
        return 'vit_g'

    # 2. Check name substrings
    if 'vit_h' in name_lower or 'vit-h' in name_lower:
        return 'vit_h'
    if 'vit_g' in name_lower or 'vit-g' in name_lower or 'vit_bigg' in name_lower:
        return 'vit_g'

    # 3. Examine state dict dimensions
    if state_dict is not None:
        proj_sd = state_dict.get('image_proj', {})
        for key in proj_sd:
            if 'weight' in key and isinstance(proj_sd[key], torch.Tensor):
                shape = proj_sd[key].shape
                if len(shape) >= 2:
                    # Input dimension of projection layer
                    in_dim = shape[-1]
                    if in_dim == 1024:
                        return 'vit_h'
                    elif in_dim == 1280:
                        return 'vit_g'
                    break

    # 4. Default
    return 'vit_h'


def get_clip_model_url(variant: str) -> str:
    """Get the download URL for a CLIP vision model variant.

    Args:
        variant: 'vit_h' or 'vit_g'

    Returns:
        HuggingFace URL string.
    """
    from vibewarp.config import MODEL_URLS
    if variant == 'vit_g':
        return MODEL_URLS.get('clip_vision_vit_bigg', '')
    return MODEL_URLS.get('clip_vision_vit_h', '')


# ---- Hook / Unhook ----

def hook_ipadapter(
    sd_model: Any,
    adapter_weights: dict,
    clip_vision_output: Dict[str, torch.Tensor],
    weight: float = 1.0,
    start: float = 0.0,
    end: float = 1.0,
    weight_type: str = 'linear',
    embeds_scaling: str = 'V only',
    is_v2: bool = False,
    combine_embeds: str = 'concat',
    flip_uc: bool = False,
) -> None:
    """Hook an IP-Adapter into the SD model's UNet cross-attention blocks.

    This patches the UNet's attention layers to add IP-Adapter conditioning,
    using the vendored PlugableIPAdapter (vibewarp/vendor/controlmodel_ipadapter.py,
    from the Sxela/sxela-stablediffusion fork the reference notebook uses).

    Args:
        sd_model: loaded SD model
        adapter_weights: loaded IP-Adapter state dict (from load_ipadapter_model)
        clip_vision_output: CLIP vision embeddings (from encode_clip_vision)
        weight: conditioning strength
        start: start percentage for time-based gating (0.0 = beginning)
        end: end percentage for time-based gating (1.0 = end)
        weight_type: layer-wise weight strategy
        embeds_scaling: embedding scaling strategy
        is_v2: FaceID-v2 flag — notebook derives it as 'v2' in the adapter filename
    """
    from vibewarp.vendor.controlmodel_ipadapter import PlugableIPAdapter

    adapter = PlugableIPAdapter(adapter_weights, is_v2=is_v2)
    adapter.half().cuda()

    adapter.hook(
        model=sd_model.model.diffusion_model,
        clip_vision_output=clip_vision_output,
        weight=weight,
        start=start,
        end=end,
        weight_type=weight_type,
        combine_embeds=combine_embeds,
        embeds_scaling=embeds_scaling,
        flip_uc=flip_uc,
    )

    # Store reference for cleanup
    if not hasattr(sd_model, '_ipadapter_hooks'):
        sd_model._ipadapter_hooks = []
    sd_model._ipadapter_hooks.append(adapter)


def unhook_ipadapter(sd_model: Any) -> None:
    """Remove all IP-Adapter hooks from the SD model.

    Args:
        sd_model: loaded SD model with IP-Adapter hooks.
    """
    from vibewarp.vendor.controlmodel_ipadapter import clear_all_ip_adapter
    clear_all_ip_adapter()

    if hasattr(sd_model, '_ipadapter_hooks'):
        sd_model._ipadapter_hooks.clear()


# ---- Embedding cache ----

class IPAdapterEmbeddingCache:
    """Cache for CLIP vision embeddings to avoid recomputation across frames.

    The notebook caches embeddings by (image_hash + adapter_type) to avoid
    running CLIP for every frame when the source image hasn't changed.
    """

    def __init__(self, max_size: int = 16):
        self.cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self.max_size = max_size

    def get(self, key: str) -> Optional[Dict[str, torch.Tensor]]:
        return self.cache.get(key)

    def put(self, key: str, embeddings: Dict[str, torch.Tensor]) -> None:
        if len(self.cache) >= self.max_size:
            # Evict oldest
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        self.cache[key] = embeddings

    def clear(self) -> None:
        self.cache.clear()

    def __len__(self) -> int:
        return len(self.cache)

    def __contains__(self, key: str) -> bool:
        return key in self.cache
