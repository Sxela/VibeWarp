"""LORA loading and merging for SD1.5 models.

Supports A1111-format and Diffusers-format LORA files (.safetensors or .pt).
Keys follow the pattern:
  lora_unet_<layer_name>.lora_down.weight
  lora_unet_<layer_name>.lora_up.weight
  lora_unet_<layer_name>.alpha            (optional)
  lora_te_<layer_name>.lora_down.weight   (text encoder)
  lora_te_<layer_name>.lora_up.weight

Layer names may use either:
  - ldm/A1111 style: input_blocks_1_1_transformer_blocks_0_attn1_to_q
  - Diffusers style: down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q

The loader:
  1. Builds a reverse map from normalized param name -> actual param name
  2. Matches LORA keys to model params, trying direct match then Diffusers→ldm conversion
  3. Computes delta = (alpha/rank) * up @ down and adds to the weight
  4. Saves originals so we can restore exactly
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch


# ---- Diffusers → ldm block prefix mapping (SD 1.5 UNet) ----
# Maps (block_type, block_id, sub_type, sub_id) → ldm normalized prefix
_BLOCK_PREFIX_MAP: Dict[tuple, str] = {
    # down_blocks → input_blocks
    ('down', 0, 'resnets',      0): 'input_blocks_1_0',
    ('down', 0, 'attentions',   0): 'input_blocks_1_1',
    ('down', 0, 'resnets',      1): 'input_blocks_2_0',
    ('down', 0, 'attentions',   1): 'input_blocks_2_1',
    ('down', 0, 'downsamplers', 0): 'input_blocks_3_0',
    ('down', 1, 'resnets',      0): 'input_blocks_4_0',
    ('down', 1, 'attentions',   0): 'input_blocks_4_1',
    ('down', 1, 'resnets',      1): 'input_blocks_5_0',
    ('down', 1, 'attentions',   1): 'input_blocks_5_1',
    ('down', 1, 'downsamplers', 0): 'input_blocks_6_0',
    ('down', 2, 'resnets',      0): 'input_blocks_7_0',
    ('down', 2, 'attentions',   0): 'input_blocks_7_1',
    ('down', 2, 'resnets',      1): 'input_blocks_8_0',
    ('down', 2, 'attentions',   1): 'input_blocks_8_1',
    ('down', 2, 'downsamplers', 0): 'input_blocks_9_0',
    ('down', 3, 'resnets',      0): 'input_blocks_10_0',
    ('down', 3, 'resnets',      1): 'input_blocks_11_0',
    # mid_block → middle_block
    ('mid',  0, 'resnets',      0): 'middle_block_0',
    ('mid',  0, 'attentions',   0): 'middle_block_1',
    ('mid',  0, 'resnets',      1): 'middle_block_2',
    # up_blocks → output_blocks
    ('up',   0, 'resnets',      0): 'output_blocks_0_0',
    ('up',   0, 'resnets',      1): 'output_blocks_1_0',
    ('up',   0, 'resnets',      2): 'output_blocks_2_0',
    ('up',   0, 'upsamplers',   0): 'output_blocks_2_1',
    ('up',   1, 'resnets',      0): 'output_blocks_3_0',
    ('up',   1, 'attentions',   0): 'output_blocks_3_1',
    ('up',   1, 'resnets',      1): 'output_blocks_4_0',
    ('up',   1, 'attentions',   1): 'output_blocks_4_1',
    ('up',   1, 'resnets',      2): 'output_blocks_5_0',
    ('up',   1, 'attentions',   2): 'output_blocks_5_1',
    ('up',   1, 'upsamplers',   0): 'output_blocks_5_2',
    ('up',   2, 'resnets',      0): 'output_blocks_6_0',
    ('up',   2, 'attentions',   0): 'output_blocks_6_1',
    ('up',   2, 'resnets',      1): 'output_blocks_7_0',
    ('up',   2, 'attentions',   1): 'output_blocks_7_1',
    ('up',   2, 'resnets',      2): 'output_blocks_8_0',
    ('up',   2, 'attentions',   2): 'output_blocks_8_1',
    ('up',   2, 'upsamplers',   0): 'output_blocks_8_2',
    ('up',   3, 'resnets',      0): 'output_blocks_9_0',
    ('up',   3, 'attentions',   0): 'output_blocks_9_1',
    ('up',   3, 'resnets',      1): 'output_blocks_10_0',
    ('up',   3, 'attentions',   1): 'output_blocks_10_1',
    ('up',   3, 'resnets',      2): 'output_blocks_11_0',
    ('up',   3, 'attentions',   2): 'output_blocks_11_1',
}

# Diffusers resnet sub-module names → ldm names
_RESNET_SUBMODULE_MAP: Dict[str, str] = {
    'conv1':         'in_layers_2',
    'conv2':         'out_layers_3',
    'time_emb_proj': 'emb_layers_1',
    'conv_shortcut': 'skip_connection',
    'norm1':         'in_layers_0',
    'norm2':         'out_layers_0',
}


# ---- Key mapping ----

def _normalize(name: str) -> str:
    """Normalize a parameter name for fuzzy matching.

    Replaces '.' and non-alphanumeric chars with '_', lowercases.
    """
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def _diffusers_to_ldm_unet_key(lora_short: str) -> Optional[str]:
    """Convert Diffusers-format UNet LoRA base key to ldm normalized key.

    lora_short: key with 'lora_unet_' prefix already stripped.
    Returns normalized ldm key matching _build_param_index() entries, or None.
    """
    # Try down_blocks / up_blocks
    m = re.match(
        r'^(down_blocks|up_blocks)_(\d+)_(resnets|attentions|downsamplers|upsamplers)_(\d+)(?:_(.+))?$',
        lora_short,
    )
    if m:
        block_str, block_id_str, sub_type, sub_id_str, sub_rest = m.groups()
        block_type = 'down' if block_str == 'down_blocks' else 'up'
        block_id = int(block_id_str)
        sub_id = int(sub_id_str)
    else:
        # Try mid_block (no block index)
        m = re.match(
            r'^mid_block_(resnets|attentions)_(\d+)(?:_(.+))?$',
            lora_short,
        )
        if not m:
            return None
        sub_type, sub_id_str, sub_rest = m.groups()
        block_type, block_id, sub_id = 'mid', 0, int(sub_id_str)

    ldm_prefix = _BLOCK_PREFIX_MAP.get((block_type, block_id, sub_type, sub_id))
    if ldm_prefix is None:
        return None

    if sub_rest is None:
        return ldm_prefix

    if sub_type == 'resnets':
        mapped = _RESNET_SUBMODULE_MAP.get(sub_rest)
        return f'{ldm_prefix}_{mapped}' if mapped is not None else None
    elif sub_type == 'attentions':
        # Attention sub-module names match in both formats (already normalized)
        return f'{ldm_prefix}_{sub_rest}'
    elif sub_type == 'downsamplers':
        # conv → op
        return f'{ldm_prefix}_op' if sub_rest == 'conv' else None
    elif sub_type == 'upsamplers':
        # conv → conv (same)
        return f'{ldm_prefix}_{sub_rest}'

    return None


def _build_param_index(model: Any) -> Dict[str, str]:
    """Build {normalized_name: full_param_name} for all model parameters.

    We strip the leading 'model_diffusion_model_' and 'cond_stage_model_'
    prefixes (and trailing '_weight'/'_bias') before normalizing, to match
    LORA key patterns that also strip those prefixes.
    """
    index: Dict[str, str] = {}

    unet_prefix_norm = _normalize('model.diffusion_model.')
    te_prefix_norm_v1 = _normalize('cond_stage_model.transformer.')
    te_prefix_norm_v2 = _normalize('cond_stage_model.')

    def _strip_affixes(norm: str) -> str:
        for prefix in (unet_prefix_norm, te_prefix_norm_v1, te_prefix_norm_v2):
            if norm.startswith(prefix):
                norm = norm[len(prefix):].lstrip('_')
                break
        for suffix in ('_weight', '_bias'):
            if norm.endswith(suffix):
                norm = norm[: -len(suffix)]
                break
        return norm

    for full_name, _ in model.named_parameters():
        if not full_name.endswith('.weight'):
            continue  # LoRA applies to weights only; skip biases to avoid key collisions
        norm = _normalize(full_name)
        short = _strip_affixes(norm)
        index[short] = full_name

    return index


def _match_lora_key(lora_base: str, param_index: Dict[str, str]) -> Optional[str]:
    """Find the model parameter name that matches a LORA base key.

    lora_base examples:
      'lora_unet_input_blocks_1_1_to_q'          -> ldm-style direct match
      'lora_unet_down_blocks_0_attentions_0_...'  -> Diffusers-style, converted to ldm
      'lora_te_text_model_encoder_layers_0_...'   -> text encoder param
    """
    if lora_base.startswith('lora_unet_'):
        short = lora_base[len('lora_unet_'):]
    elif lora_base.startswith('lora_te_'):
        short = lora_base[len('lora_te_'):]
    else:
        return None

    # Direct match (ldm/A1111 style keys)
    norm = _normalize(short)
    result = param_index.get(norm)
    if result is not None:
        return result

    # Diffusers-style keys (down_blocks, up_blocks, mid_block)
    if lora_base.startswith('lora_unet_'):
        ldm_norm = _diffusers_to_ldm_unet_key(short)
        if ldm_norm is not None:
            return param_index.get(ldm_norm)

    return None


# ---- LORA delta cache and apply/unapply ----

# {(lora_path, mtime): {param_name: delta_tensor (cpu, float32)}}
# Keyed by path+mtime so file changes invalidate automatically.
_LORA_DELTA_CACHE: Dict[tuple, Dict[str, torch.Tensor]] = {}

# {lora_path: {param_name: original_tensor}} — saved per active LoRA
_LOADED_LORA_ORIGINALS: Dict[str, Dict[str, torch.Tensor]] = {}


def _compute_lora_deltas(
    lora_path: str,
    sd_model: Any,
    te_multiplier: float,
    unet_multiplier: float,
    precision: str = 'fp16',
) -> Dict[str, torch.Tensor]:
    """Parse a LoRA file and compute weight deltas, with in-memory caching.

    precision:
      'fp16' (default) — WarpFusion/A1111-exact arithmetic: up/down cast to
        the target param's dtype BEFORE the matmul, scale and multiplier
        applied as separate ops. Required for notebook output parity.
      'fp32' — matmul in float32 (more accurate deltas, rounded once at
        apply time; diverges from notebook renders by ~1 fp16 ulp/weight).

    Returns {param_name: delta_tensor (cpu)}. Skips (and does not cache)
    keys that don't match any model parameter. The cache is keyed by
    (path, mtime, multipliers, precision) so on-disk edits invalidate it.
    """
    if precision not in ('fp16', 'fp32'):
        raise ValueError(f"lora merge precision must be 'fp16' or 'fp32', got {precision!r}")
    mtime = os.path.getmtime(lora_path)
    cache_key = (lora_path, mtime, te_multiplier, unet_multiplier, precision)

    if cache_key in _LORA_DELTA_CACHE:
        return _LORA_DELTA_CACHE[cache_key]

    from vibewarp.core.model_loader import load_state_dict
    sd = load_state_dict(lora_path, location='cpu')

    layers: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in sd.items():
        if '.lora_down' in key:
            base = key[:key.index('.lora_down')]
            layers.setdefault(base, {})['down'] = tensor
        elif '.lora_up' in key:
            base = key[:key.index('.lora_up')]
            layers.setdefault(base, {})['up'] = tensor
        elif key.endswith('.alpha'):
            base = key[:-len('.alpha')]
            layers.setdefault(base, {})['alpha'] = tensor
    del sd

    param_index = _build_param_index(sd_model)
    deltas: Dict[str, torch.Tensor] = {}
    applied = 0
    skipped = 0

    for base, data in layers.items():
        if 'down' not in data or 'up' not in data:
            skipped += 1
            continue

        param_name = _match_lora_key(base, param_index)
        if param_name is None:
            skipped += 1
            continue

        param = sd_model
        for part in param_name.split('.'):
            param = getattr(param, part, None)
            if param is None:
                break
        if param is None or not isinstance(param, torch.nn.Parameter):
            skipped += 1
            continue

        rank = data['down'].shape[0]
        alpha_val = data.get('alpha')
        alpha = float(alpha_val.item()) if alpha_val is not None else float(rank)
        scale = alpha / rank

        multiplier = te_multiplier if base.startswith('lora_te_') else unet_multiplier

        # 'fp16': notebook computes calc_updown entirely in the TARGET dtype
        # — up/down cast to the weight's dtype BEFORE the matmul, scale and
        # multiplier applied as separate ops; every rounding happens where
        # the notebook rounds (required for same-seed parity).
        # 'fp32': more accurate deltas, rounded once at apply time — diverges
        # from notebook renders by ~1 fp16 ulp per weight on every layer.
        merge_dtype = param.dtype if precision == 'fp16' else torch.float32
        down = data['down'].to(merge_dtype)
        up = data['up'].to(merge_dtype)

        if down.dim() == 2:
            delta = (up @ down) * scale * multiplier
        elif down.dim() == 4:
            delta = ((up.flatten(1) @ down.flatten(1)).reshape(param.shape)
                     * scale * multiplier)
        else:
            skipped += 1
            continue

        if delta.shape != param.data.shape:
            skipped += 1
            continue

        deltas[param_name] = delta.cpu()
        applied += 1

    _LORA_DELTA_CACHE[cache_key] = deltas
    print(f"LORA parsed: {os.path.basename(lora_path)} ({applied} layers, {skipped} skipped)")
    return deltas


def load_lora(
    sd_model: Any,
    lora_path: str,
    te_multiplier: float = 1.0,
    unet_multiplier: float = 1.0,
    precision: str = 'fp16',
) -> None:
    """Merge a LoRA file into the SD model weights (delta cached after first parse).

    Saves original weights so they can be restored later by unload_loras().
    precision: 'fp16' (notebook/A1111-exact, default) or 'fp32' (more
    accurate deltas) — see _compute_lora_deltas.
    """
    deltas = _compute_lora_deltas(lora_path, sd_model, te_multiplier,
                                  unet_multiplier, precision)
    if not deltas:
        print(f"LORA: no layers matched for {os.path.basename(lora_path)}")
        return

    originals: Dict[str, torch.Tensor] = {}

    for param_name, delta in deltas.items():
        param = sd_model
        for part in param_name.split('.'):
            param = getattr(param, part, None)
            if param is None:
                break
        if param is None or not isinstance(param, torch.nn.Parameter):
            continue

        if param_name not in originals:
            originals[param_name] = param.data.clone()

        with torch.no_grad():
            param.data.add_(delta.to(param.data.dtype).to(param.data.device))

    _LOADED_LORA_ORIGINALS[lora_path] = originals
    print(f"LORA applied: {os.path.basename(lora_path)} ({len(deltas)} layers)")


def unload_loras(sd_model: Any) -> None:
    """Restore all model parameters modified by load_lora().

    Should be called before loading new LORAs each frame, or at the end of a run.
    """
    if not _LOADED_LORA_ORIGINALS:
        return

    for lora_path, originals in _LOADED_LORA_ORIGINALS.items():
        for param_name, original in originals.items():
            param = sd_model
            for part in param_name.split('.'):
                param = getattr(param, part, None)
                if param is None:
                    break
            if param is not None and isinstance(param, torch.nn.Parameter):
                with torch.no_grad():
                    param.data.copy_(original.to(param.data.device))

    _LOADED_LORA_ORIGINALS.clear()


def load_loras_for_frame(
    sd_model: Any,
    names: List[str],
    weights: List[float],
    lora_dir: str,
    precision: str = 'fp16',
) -> None:
    """Unload existing LORAs and load the given list for a frame.

    Args:
        sd_model: the loaded SD model
        names: list of LORA names (filename stems, e.g. 'my_style')
        weights: corresponding weights
        lora_dir: directory to search for LORA files
        precision: 'fp16' (notebook-exact merge, default) or 'fp32'
    """
    unload_loras(sd_model)

    if not names or not lora_dir:
        return

    for name, weight in zip(names, weights):
        if weight == 0.0:
            continue
        path = _find_lora_file(name, lora_dir)
        if path is None:
            print(f"LORA not found: {name} in {lora_dir}")
            continue
        load_lora(sd_model, path, te_multiplier=weight, unet_multiplier=weight,
                  precision=precision)


def _find_lora_file(name: str, lora_dir: str) -> Optional[str]:
    """Search lora_dir for a file matching the given name stem."""
    for ext in ('.safetensors', '.pt', '.ckpt'):
        path = os.path.join(lora_dir, name + ext)
        if os.path.exists(path):
            return path
    # Also try name as a full path
    if os.path.exists(name):
        return name
    return None
