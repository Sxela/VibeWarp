"""Convert a diffusers-format ControlNet checkpoint to the vendored cldm layout.

Every published SDXL ControlNet ships in diffusers format (`down_blocks.*`,
`controlnet_down_blocks.*`, `add_embedding.*`), while the vendored ControlNet in
`vendor/cldm/cldm.py` is the original cldm layout (`input_blocks.*`,
`zero_convs.*`, `label_emb.*`). Nothing in the notebook helps here — it shells out
to a full ComfyUI clone (`from comfy.sd import load_controlnet`), which is exactly
the external dependency this package exists to remove. So the mapping lives here.

The two layouts describe the same graph; only the naming differs:

    diffusers                                cldm
    conv_in                                  input_blocks.0.0
    down_blocks[l].resnets[i]                input_blocks[n].0
    down_blocks[l].attentions[i]             input_blocks[n].1
    down_blocks[l].downsamplers[0].conv      input_blocks[n].0.op
    mid_block.resnets[0/1]                   middle_block.0 / .2
    mid_block.attentions[0]                  middle_block.1
    controlnet_down_blocks[n]                zero_convs[n].0
    controlnet_mid_block                     middle_block_out.0
    controlnet_cond_embedding.*              input_hint_block.*
    time_embedding.linear_1/2                time_embed.0 / .2
    add_embedding.linear_1/2                 label_emb.0.0 / label_emb.0.2

Conversion is verified by loading with strict=True: any key we fail to map, or map
wrongly, surfaces immediately as a missing/unexpected key rather than as a silently
wrong render.
"""

from typing import Dict

import torch

# Within a ResBlock / SpatialTransformer the leaf names differ; the transformer's
# inner blocks (attn1/attn2/ff/norm1-3) happen to match already.
_RESNET_MAP = {
    'norm1': 'in_layers.0',
    'conv1': 'in_layers.2',
    'time_emb_proj': 'emb_layers.1',
    'norm2': 'out_layers.0',
    'conv2': 'out_layers.3',
    'conv_shortcut': 'skip_connection',
}

# controlnet_cond_embedding -> input_hint_block (SiLUs sit on the odd indices).
_HINT_MAP = {'conv_in': '0', 'conv_out': '14'}
for _i in range(6):
    _HINT_MAP[f'blocks.{_i}'] = str(2 + _i * 2)


def is_diffusers_controlnet(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(key.startswith('controlnet_down_blocks.') for key in state_dict)


def _block_layout(channel_mult, num_res_blocks: int):
    """cldm input_block index for each (level, resnet) and each downsampler.

    input_blocks[0] is conv_in; then per level: num_res_blocks blocks, plus a
    downsampler on every level except the last.
    """
    resnets, downsamplers = {}, {}
    index = 1
    for level in range(len(channel_mult)):
        for i in range(num_res_blocks):
            resnets[(level, i)] = index
            index += 1
        if level != len(channel_mult) - 1:
            downsamplers[level] = index
            index += 1
    return resnets, downsamplers


def diffusers_to_cldm(
    state_dict: Dict[str, torch.Tensor],
    channel_mult=(1, 2, 4),
    num_res_blocks: int = 2,
) -> Dict[str, torch.Tensor]:
    """Rename a diffusers ControlNet state dict into the cldm layout."""
    resnets, downsamplers = _block_layout(channel_mult, num_res_blocks)
    out: Dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        new_key = _convert_key(key, resnets, downsamplers)
        if new_key is None:
            raise KeyError(f"Unmapped diffusers ControlNet key: {key}")
        out[new_key] = value
    return out


def _convert_key(key: str, resnets: dict, downsamplers: dict):
    parts = key.split('.')
    head = parts[0]

    if head == 'conv_in':
        return f"input_blocks.0.0.{'.'.join(parts[1:])}"

    if head == 'time_embedding':
        # linear_1 -> time_embed.0, linear_2 -> time_embed.2 (SiLU is index 1).
        index = {'linear_1': '0', 'linear_2': '2'}[parts[1]]
        return f"time_embed.{index}.{'.'.join(parts[2:])}"

    if head == 'add_embedding':
        index = {'linear_1': '0', 'linear_2': '2'}[parts[1]]
        return f"label_emb.0.{index}.{'.'.join(parts[2:])}"

    if head == 'controlnet_cond_embedding':
        if parts[1] in ('conv_in', 'conv_out'):
            return f"input_hint_block.{_HINT_MAP[parts[1]]}.{'.'.join(parts[2:])}"
        # blocks.N.<leaf>
        return f"input_hint_block.{_HINT_MAP[f'blocks.{parts[2]}']}.{'.'.join(parts[3:])}"

    if head == 'controlnet_down_blocks':
        # Each is a 1x1 zero conv wrapped in a TimestepEmbedSequential.
        return f"zero_convs.{parts[1]}.0.{'.'.join(parts[2:])}"

    if head == 'controlnet_mid_block':
        return f"middle_block_out.0.{'.'.join(parts[1:])}"

    if head == 'down_blocks':
        level = int(parts[1])
        kind = parts[2]
        if kind == 'resnets':
            index = resnets[(level, int(parts[3]))]
            return f"input_blocks.{index}.0.{_resnet_leaf(parts[4:])}"
        if kind == 'attentions':
            index = resnets[(level, int(parts[3]))]
            return f"input_blocks.{index}.1.{'.'.join(parts[4:])}"
        if kind == 'downsamplers':
            # downsamplers.0.conv -> input_blocks.N.0.op
            index = downsamplers[level]
            return f"input_blocks.{index}.0.op.{'.'.join(parts[5:])}"
        return None

    if head == 'mid_block':
        kind = parts[1]
        if kind == 'resnets':
            # resnets 0 and 1 straddle the middle attention (0, 1, 2).
            index = 0 if parts[2] == '0' else 2
            return f"middle_block.{index}.{_resnet_leaf(parts[3:])}"
        if kind == 'attentions':
            return f"middle_block.1.{'.'.join(parts[3:])}"
        return None

    return None


def _resnet_leaf(rest) -> str:
    """Map the leaf of a diffusers resnet key onto the cldm ResBlock."""
    name = rest[0]
    if name not in _RESNET_MAP:
        raise KeyError(f"Unmapped ControlNet resnet leaf: {'.'.join(rest)}")
    return f"{_RESNET_MAP[name]}.{'.'.join(rest[1:])}"
