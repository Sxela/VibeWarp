"""AnimateDiff motion modules — vendored, self-contained (SD1.5 + SDXL/HotshotXL).

Sources, in layers. Every layer is either permissively licensed, GPL-3.0 (this project's
own licence), or original to VibeWarp — see THIRD_PARTY_LICENSES.md:

1. Temporal-attention stack (VanillaTemporalModule, TemporalTransformer3DModel,
   TemporalTransformerBlock, PositionalEncoding, VersatileAttention) — the AnimateDiff
   architecture itself, from guoyww/AnimateDiff `animatediff/models/motion_module.py`
   (Apache-2.0). Its diffusers CrossAttention base is replaced here by the self-contained
   SDPA CrossAttention + GEGLU FeedForward below, which keep the same parameter names
   (to_q/to_k/to_v, to_out.0, ff.net.0.proj, ff.net.2) so checkpoints load unchanged.
2. Attention/FeedForward primitives (CrossAttention, GEGLU, FeedForward) — original to
   VibeWarp, written against the parameter layout above.
3. Block grouping (BlockType, MotionModule, get_motion_module) — original to VibeWarp.
   guoyww attaches motion modules while building its own UNet; we load them as a standalone
   checkpoint and inject them into an existing one, so we need a container keyed the way
   those checkpoints are. Its shape follows the file format, not a design choice.
4. MotionWrapper + state-dict detection/normalisation (is_hotshotxl, is_animatelcm, is_pia,
   normalize_ad_state_dict, ...) and inject/eject: the NOTEBOOK's inline code (cell 23,
   MIT WarpFusion), which extends it with is_sdxl / is_v3 / is_hotshot / hack_gn and is the
   source of truth. That code is itself merged from Kosinkadink/ComfyUI-AnimateDiff-Evolved
   (GPL-3.0).
5. `get_position_encoding_max_len` is called by the notebook's normalize_ad_state_dict but
   never defined there (it would NameError for SD1.5 v1/v3 modules) — from
   ComfyUI-AnimateDiff-Evolved (GPL-3.0).

Patches vs sources (behaviour-preserving):
- comfy.model_management / cli_args attention selection dropped: the reference env has no
  xformers, so plain attention math was the active path; SDPA computes the same attention.
- `mm_injected` read with getattr(default False) instead of requiring the attribute to be
  pre-set.
- SDXL beta tensors computed on the model's device instead of hardcoded 'cuda'.
"""

import math
import re

import torch
from torch import Tensor, nn
from einops import rearrange, repeat


# ---------------------------------------------------------------------------
# Attention / FeedForward primitives (replacing comfy.ldm.modules.attention)
# Parameter layout matches comfy/ldm CrossAttention and diffusers FeedForward
# so AnimateDiff checkpoints load without key remapping.
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None, value=None, mask=None):
        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(value if value is not None else context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.view(b, -1, self.heads, self.dim_head).transpose(1, 2),
            (q, k, v),
        )
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, -1, self.heads * self.dim_head)
        return self.to_out(out)


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        project_in = (
            GEGLU(dim, inner_dim) if glu
            else nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
        )
        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


# ---------------------------------------------------------------------------
# State-dict detection & normalization (notebook inline / AnimateDiff-Evolved)
# ---------------------------------------------------------------------------

class ModelTypeSD:
    SD1_5 = "SD1.5"
    SD2_1 = "SD2.1"
    SDXL = "SDXL"
    SDXL_REFINER = "SDXL_Refiner"
    SVD = "SVD"

    _LIST = [SD1_5, SD2_1, SDXL, SDXL_REFINER, SVD]


class AnimateDiffFormat:
    ANIMATEDIFF = "AnimateDiff"
    HOTSHOTXL = "HotshotXL"
    ANIMATELCM = "AnimateLCM"
    PIA = "PIA"

    _LIST = [ANIMATEDIFF, HOTSHOTXL, ANIMATELCM, PIA]


class AnimateDiffVersion:
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"

    _LIST = [V1, V2, V3]


class AnimateDiffInfo:
    def __init__(self, sd_type: str, mm_format: str, mm_version: str, mm_name: str):
        self.sd_type = sd_type
        self.mm_format = mm_format
        self.mm_version = mm_version
        self.mm_name = mm_name

    def get_string(self):
        return f"{self.mm_name}:{self.mm_version}:{self.mm_format}:{self.sd_type}"


_regex_hotshotxl_module_num = re.compile(r'temporal_attentions\.(\d+)\.')


def find_hotshot_module_num(key: str):
    found = _regex_hotshotxl_module_num.search(key)
    if found:
        return int(found.group(1))
    return None


def is_hotshotxl(mm_state_dict) -> bool:
    # use pos_encoder naming to determine if hotshotxl model
    for key in mm_state_dict.keys():
        if key.endswith("pos_encoder.positional_encoding"):
            return True
    return False


def is_animatelcm(mm_state_dict) -> bool:
    # use lack of ANY pos_encoder keys to determine if animatelcm model
    for key in mm_state_dict.keys():
        if "pos_encoder" in key:
            return False
    return True


def is_pia(mm_state_dict) -> bool:
    # check if conv_in.weight and .bias are present
    if "conv_in.weight" in mm_state_dict and "conv_in.bias" in mm_state_dict:
        return True
    return False


def has_img_encoder(mm_state_dict):
    for key in mm_state_dict.keys():
        if key.startswith("img_encoder."):
            return True
    return False


def get_down_block_max(mm_state_dict):
    # keep track of biggest down_block count in module
    biggest_block = 0
    for key in mm_state_dict.keys():
        if "down_blocks" in key:
            try:
                block_int = key.split(".")[1]
                block_num = int(block_int)
                if block_num > biggest_block:
                    biggest_block = block_num
            except ValueError:
                pass
    return biggest_block


def get_encoding_max_len(mm_state_dict) -> int:
    # use pos_encoder.pe entries to determine max length - [1, {max_length}, {320|640|1280}]
    for key in mm_state_dict.keys():
        if key.endswith("pos_encoder.pe"):
            return mm_state_dict[key].size(1)  # get middle dim
    raise ValueError("No pos_encoder.pe found in mm_state_dict")


def get_position_encoding_max_len(mm_state_dict, mm_name: str, mm_format: str):
    # Ported from ComfyUI-AnimateDiff-Evolved (the notebook calls this but never
    # defines it). Same as get_encoding_max_len, with an AnimateLCM default.
    for key in mm_state_dict.keys():
        if key.endswith("pos_encoder.pe"):
            return mm_state_dict[key].size(1)
    if mm_format == AnimateDiffFormat.ANIMATELCM:
        return 64
    raise ValueError(f"No pos_encoder.pe found in mm_state_dict of {mm_name}")


def has_mid_block(mm_state_dict):
    # check if keys contain mid_block
    for key in mm_state_dict.keys():
        if key.startswith("mid_block."):
            return True
    return False


def normalize_ad_state_dict(mm_state_dict, mm_name: str):
    # determine what SD model the motion module is intended for
    sd_type = None
    down_block_max = get_down_block_max(mm_state_dict)
    if down_block_max == 3:
        sd_type = ModelTypeSD.SD1_5
    elif down_block_max == 2:
        sd_type = ModelTypeSD.SDXL
    else:
        raise ValueError(
            f"'{mm_name}' is not a valid SD1.5 nor SDXL motion module - "
            f"contained {down_block_max} downblocks.")
    # determine the model's format
    mm_format = AnimateDiffFormat.ANIMATEDIFF
    if is_hotshotxl(mm_state_dict):
        mm_format = AnimateDiffFormat.HOTSHOTXL
    if is_animatelcm(mm_state_dict):
        mm_format = AnimateDiffFormat.ANIMATELCM
    if is_pia(mm_state_dict):
        mm_format = AnimateDiffFormat.PIA
    # for AnimateLCM-I2V purposes, check for img_encoder keys
    contains_img_encoder = has_img_encoder(mm_state_dict)
    # remove all non-temporal keys (in case model has extra stuff in it)
    for key in list(mm_state_dict.keys()):
        if "temporal" not in key:
            if mm_format == AnimateDiffFormat.ANIMATELCM and contains_img_encoder and key.startswith("img_encoder."):
                continue
            if mm_format == AnimateDiffFormat.PIA and key.startswith("conv_in."):
                continue
            del mm_state_dict[key]
    # determine the model's version
    mm_version = AnimateDiffVersion.V1
    if has_mid_block(mm_state_dict):
        mm_version = AnimateDiffVersion.V2
    elif sd_type == ModelTypeSD.SD1_5 and get_position_encoding_max_len(mm_state_dict, mm_name, mm_format) == 32:
        mm_version = AnimateDiffVersion.V3
    info = AnimateDiffInfo(sd_type=sd_type, mm_format=mm_format, mm_version=mm_version, mm_name=mm_name)
    # convert to AnimateDiff format, if needed
    if mm_format == AnimateDiffFormat.HOTSHOTXL:
        # HotshotXL is AD-based architecture applied to SDXL instead of SD1.5
        # By renaming the keys, no code needs to be adapted at all
        #
        # reformat temporal_attentions:
        # HSXL: temporal_attentions.#.
        #   AD: motion_modules.#.temporal_transformer.
        # HSXL: pos_encoder.positional_encoding
        #   AD: pos_encoder.pe
        for key in list(mm_state_dict.keys()):
            module_num = find_hotshot_module_num(key)
            if module_num is not None:
                new_key = key.replace(f"temporal_attentions.{module_num}",
                                      f"motion_modules.{module_num}.temporal_transformer", 1)
                new_key = new_key.replace("pos_encoder.positional_encoding", "pos_encoder.pe")
                mm_state_dict[new_key] = mm_state_dict[key]
                del mm_state_dict[key]
    # return adjusted mm_state_dict and info
    return mm_state_dict, info


# ---------------------------------------------------------------------------
# Temporal-attention stack — the AnimateDiff architecture itself.
# From guoyww/AnimateDiff `animatediff/models/motion_module.py` (Apache-2.0), with its
# diffusers CrossAttention base swapped for the self-contained SDPA CrossAttention above
# (same parameter names, so checkpoints load unchanged).
# ---------------------------------------------------------------------------

class BlockType:
    """Where in the UNet a group of motion modules is attached."""
    UP = "up"
    DOWN = "down"
    MID = "mid"


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=24):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class VersatileAttention(CrossAttention):
    def __init__(
        self,
        attention_mode=None,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert attention_mode == "Temporal"

        self.attention_mode = attention_mode
        self.is_cross_attention = kwargs["context_dim"] is not None

        self.pos_encoder = (
            PositionalEncoding(
                kwargs["query_dim"],
                dropout=0.0,
                max_len=temporal_position_encoding_max_len,
            )
            if (temporal_position_encoding and attention_mode == "Temporal")
            else None
        )

    def extra_repr(self):
        return f"(Module Info) Attention_Mode: {self.attention_mode}, Is_Cross_Attention: {self.is_cross_attention}"

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
        **cross_attention_kwargs,
    ):
        if self.attention_mode != "Temporal":
            raise NotImplementedError

        d = hidden_states.shape[1]
        hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)

        if self.pos_encoder is not None:
            hidden_states = self.pos_encoder(hidden_states)

        encoder_hidden_states = (
            repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d)
            if encoder_hidden_states is not None
            else encoder_hidden_states
        )

        hidden_states = super().forward(
            hidden_states,
            encoder_hidden_states,
            value=None,
            mask=attention_mask,
        )

        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states


class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=768,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
    ):
        super().__init__()

        attention_blocks = []
        norms = []

        for block_name in attention_block_types:
            attention_blocks.append(
                VersatileAttention(
                    attention_mode=block_name.split("_")[0],
                    context_dim=cross_attention_dim if block_name.endswith("_Cross") else None,
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
            )
            norms.append(nn.LayerNorm(dim))

        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, glu=(activation_fn == "geglu"))
        self.ff_norm = nn.LayerNorm(dim)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
    ):
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states)
            hidden_states = (
                attention_block(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if attention_block.is_cross_attention else None,
                    video_length=video_length,
                )
                + hidden_states
            )

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

        output = hidden_states
        return output


class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,
        num_layers,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=768,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_block_types=attention_block_types,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)
        self.video_length = 16

    def set_video_length(self, video_length: int):
        self.video_length = video_length

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        # Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                video_length=self.video_length,
            )

        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        return output


class VanillaTemporalModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads=8,
        num_transformer_block=1,
        attention_block_types=("Temporal_Self", "Temporal_Self"),
        cross_frame_attention_mode=None,
        temporal_position_encoding=True,
        temporal_position_encoding_max_len=24,
        temporal_attention_dim_div=1,
        zero_initialize=True,
    ):
        super().__init__()

        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels // num_attention_heads // temporal_attention_dim_div,
            num_layers=num_transformer_block,
            attention_block_types=attention_block_types,
            cross_frame_attention_mode=cross_frame_attention_mode,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
        )

        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(self.temporal_transformer.proj_out)

    def set_video_length(self, video_length: int):
        self.temporal_transformer.set_video_length(video_length)

    def forward(self, input_tensor, encoder_hidden_states=None, attention_mask=None):
        return self.temporal_transformer(input_tensor, encoder_hidden_states, attention_mask)


# ---------------------------------------------------------------------------
# Block grouping — original to VibeWarp.
#
# guoyww's reference implementation attaches motion modules inline while building its own
# UNet. We load motion modules as a STANDALONE checkpoint and inject them into an existing
# UNet, so we need a container keyed the way those checkpoints are:
#
#   down_blocks.<i>.motion_modules.<j>.temporal_transformer...
#   mid_block.motion_modules.0.temporal_transformer...
#   up_blocks.<i>.motion_modules.<j>.temporal_transformer...
#
# The module count per block is a property of that file format, not a design choice: a
# down block carries 2 (one per resnet), a mid block 1, and an up block 3 (the decoder has
# an extra resnet for the skip connection).
# ---------------------------------------------------------------------------

MOTION_MODULES_PER_BLOCK = {
    BlockType.DOWN: 2,
    BlockType.MID: 1,
    BlockType.UP: 3,
}


def get_motion_module(in_channels, max_len):
    """One temporal-attention module sized for a UNet block of `in_channels`."""
    return VanillaTemporalModule(in_channels=in_channels,
                                 temporal_position_encoding_max_len=max_len)


class MotionModule(nn.Module):
    """The motion modules attached to a single UNet block."""

    def __init__(self, in_channels, block_type: str = BlockType.DOWN, encoding_max_len=24):
        super().__init__()
        self.block_type = block_type
        count = MOTION_MODULES_PER_BLOCK[block_type]
        self.motion_modules = nn.ModuleList(
            get_motion_module(in_channels, encoding_max_len) for _ in range(count))

    def set_video_length(self, video_length: int):
        for motion_module in self.motion_modules:
            motion_module.set_video_length(video_length)


# ---------------------------------------------------------------------------
# MotionWrapper (notebook version — sdxl / v3 / hotshot / hack_gn support)
# ---------------------------------------------------------------------------

class MotionWrapper(nn.Module):
    def __init__(self, mm_type: str, encoding_max_len: int = 24, is_v2=False,
                 is_sdxl=False, is_v3=False, is_hotshot=False):
        super().__init__()
        self.mm_type = mm_type
        self.is_v2 = is_v2
        self.is_sdxl = is_sdxl
        self.is_v3 = is_v3
        self.hack_gn = not (is_sdxl or is_v3)
        self.is_hotshot = is_hotshot

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        self.mid_block = None
        self.encoding_max_len = encoding_max_len

        channels = [320, 640, 1280, 1280] if not is_sdxl else [320, 640, 1280]
        for c in channels:
            self.down_blocks.append(MotionModule(c, BlockType.DOWN, encoding_max_len=encoding_max_len))
            self.up_blocks.insert(0, MotionModule(c, BlockType.UP, encoding_max_len=encoding_max_len))
        if is_v2:
            self.mid_block = MotionModule(1280, BlockType.MID, encoding_max_len=encoding_max_len)

    @classmethod
    def from_state_dict(cls, mm_state_dict, mm_type, is_sdxl=False, is_v3=False):
        is_hotshot = is_hotshotxl(mm_state_dict)
        mm_state_dict, _ = normalize_ad_state_dict(mm_state_dict, mm_type)
        encoding_max_len = get_encoding_max_len(mm_state_dict)
        print('Encoding max len ', encoding_max_len)
        is_v2 = has_mid_block(mm_state_dict)

        mm = cls(mm_type, encoding_max_len=encoding_max_len, is_v2=is_v2,
                 is_sdxl=is_sdxl, is_v3=is_v3, is_hotshot=is_hotshot)
        mm.load_state_dict(mm_state_dict, strict=False)
        return mm

    def set_video_length(self, video_length: int):
        for block in self.down_blocks:
            block.set_video_length(video_length)
        for block in self.up_blocks:
            block.set_video_length(video_length)
        if self.mid_block is not None:
            self.mid_block.set_video_length(video_length)


# ---------------------------------------------------------------------------
# UNet injection / ejection (notebook version, verbatim logic)
# ---------------------------------------------------------------------------

def inject_motion_module_to_unet(diffusion_model, mm):
    if getattr(diffusion_model, 'mm_injected', False):
        print('mm already injected. exiting')
        return

    # insert motion modules depending on surrounding layers
    for i in range(12 if not mm.is_sdxl else 9):
        a, b = divmod(i, 3)
        if type(diffusion_model.input_blocks[i][-1]).__name__ not in ["Downsample", "Conv2d"]:
            diffusion_model.input_blocks[i].append(mm.down_blocks[a].motion_modules[b - 1])

        if type(diffusion_model.output_blocks[i][-1]).__name__ == "Upsample":
            diffusion_model.output_blocks[i].insert(-1, mm.up_blocks[a].motion_modules[b])
        else:
            diffusion_model.output_blocks[i].append(mm.up_blocks[a].motion_modules[b])
    if mm.is_v2:
        diffusion_model.middle_block.insert(-1, mm.mid_block.motion_modules[0])
    elif mm.hack_gn:
        # AnimateDiff v1 GroupNorm hack: normalize across the full frame batch
        # (b=2 → the uncond+cond halves), matching the notebook.
        if mm.is_hotshot:
            from sgm.modules.diffusionmodules.util import GroupNorm32
        else:
            from ldm.modules.diffusionmodules.util import GroupNorm32
        diffusion_model.gn32_original_forward = GroupNorm32.forward
        gn32_original_forward = diffusion_model.gn32_original_forward

        def groupnorm32_mm_forward(self, x):
            x = rearrange(x, "(b f) c h w -> b c f h w", b=2)
            x = gn32_original_forward(self, x)
            x = rearrange(x, "b c f h w -> (b f) c h w", b=2)
            return x

        GroupNorm32.forward = groupnorm32_mm_forward

    diffusion_model.mm_injected = True


def eject_motion_module_from_unet(diffusion_model, mm):
    if not getattr(diffusion_model, 'mm_injected', False):
        print('mm not injected. exiting')
        return
    # remove motion modules depending on surrounding layers
    for i in range(12 if not mm.is_sdxl else 9):
        a, b = divmod(i, 3)
        if type(diffusion_model.input_blocks[i][-1]).__name__ == 'VanillaTemporalModule':
            diffusion_model.input_blocks[i].pop(-1)

        if type(diffusion_model.output_blocks[i][-2]).__name__ == 'VanillaTemporalModule':
            diffusion_model.output_blocks[i].pop(-2)
        elif type(diffusion_model.output_blocks[i][-1]).__name__ == 'VanillaTemporalModule':
            diffusion_model.output_blocks[i].pop(-1)
    if mm.is_v2:
        if type(diffusion_model.middle_block[-2]).__name__ == 'VanillaTemporalModule':
            diffusion_model.middle_block.pop(-2)
    elif mm.hack_gn:
        if mm.is_hotshot:
            from sgm.modules.diffusionmodules.util import GroupNorm32
        else:
            from ldm.modules.diffusionmodules.util import GroupNorm32
        GroupNorm32.forward = diffusion_model.gn32_original_forward
        diffusion_model.gn32_original_forward = None

    diffusion_model.mm_injected = False


# ---------------------------------------------------------------------------
# Beta-schedule adjustment after injection (notebook, post-inject block)
# ---------------------------------------------------------------------------

def register_animatediff_schedule(sd_model, mm, model_version: str):
    """Re-register the diffusion beta schedule for AnimateDiff, as the notebook
    does right after injecting the motion module (before the k-diffusion wrap
    is created — the wrap bakes alphas_cumprod into its sigmas).

    SD1.5: register_schedule with sqrt_linear betas (0.00085 → 0.012).
    SDXL:  sqrt-linspace betas with beta_end=0.020 → ad_alphas_cumprod; for
           non-hotshot modules this also replaces alphas_cumprod.
    """
    if 'sdxl' not in model_version:
        sd_model.register_schedule(
            given_betas=None,
            beta_schedule="sqrt_linear",
            timesteps=1000,
            linear_start=0.00085,
            linear_end=0.012,
            cosine_s=8e-3,
        )
        # ldm's register_schedule builds its buffers with torch.tensor(...), i.e.
        # on the CPU — so alphas_cumprod comes back on the CPU even though the model
        # is on the GPU. CompVisDenoiser then bakes CPU sigmas at construction and
        # sampling dies with "two devices, cuda:0 and cpu". (The notebook hides this
        # by calling model_wrap.cuda() afterwards.)
        try:
            device = next(sd_model.parameters()).device
        except (StopIteration, AttributeError):
            device = None
        if device is not None:
            for name, buffer in list(sd_model.named_buffers(recurse=False)):
                if buffer is not None and buffer.device != device:
                    sd_model.register_buffer(name, buffer.to(device))
    else:
        # Notebook computes on 'cuda'; device only affects placement, not values.
        try:
            device = next(sd_model.parameters()).device
        except (StopIteration, AttributeError):
            device = 'cpu'
        beta_start = 0.00085
        beta_end = 0.020
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, 1000,
                               dtype=torch.float32, device=device) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sd_model.ad_alphas_cumprod = alphas_cumprod
        if not mm.is_hotshot:
            sd_model.register_buffer('alphas_cumprod', alphas_cumprod)
