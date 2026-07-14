"""ControlNet and ControlledUnetModel — adapted from ControlNet/cldm/cldm.py.

Contains the ControlNet encoder module (mirrors UNet encoder + zero convs)
and the ControlledUnetModel that accepts control residuals.
"""

import torch
import torch.nn as nn
import numpy as np

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import (
    UNetModel,
    TimestepEmbedSequential,
    ResBlock,
    Downsample,
    AttentionBlock,
)
from ldm.util import exists


class ControlledUnetModel(UNetModel):
    """UNetModel with ControlNet residual injection into skip connections."""

    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


def _context_dims(context_dim, depth):
    """ldm's SpatialTransformer indexes context_dim per transformer block, but only
    wraps a scalar into a 1-element list — fine at SD1.5's depth 1, IndexError at
    SDXL's depth 10. Give it one entry per block."""
    if context_dim is None or isinstance(context_dim, list):
        return context_dim
    return [context_dim] * depth


class ControlNet(nn.Module):
    """ControlNet encoder — mirrors UNet encoder with zero convolution outputs."""

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,
        transformer_depth=1,
        context_dim=None,
        n_embed=None,
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        # SDXL: the pooled text/size vector ("y") is projected and added to the
        # timestep embedding, and each level has its own transformer depth
        # ([0, 2, 10] — level 0 has no attention at all).
        num_classes=None,
        adm_in_channels=None,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None

        if context_dim is not None:
            assert use_spatial_transformer
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1

        if num_head_channels == -1:
            assert num_heads != -1

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int or list matching channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.num_classes = num_classes

        # SDXL gives each level its own transformer depth; SD1.x/2.x use one int.
        if isinstance(transformer_depth, int):
            transformer_depth = [transformer_depth] * len(channel_mult)
        self.transformer_depth = transformer_depth

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # SDXL's pooled vector (text pooled + original/crop/target sizes = 2816)
        # projected into the timestep embedding. Same layout as ldm's UNetModel.
        if num_classes is not None:
            if num_classes == 'sequential':
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            elif isinstance(num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            else:
                raise ValueError(f"Unsupported num_classes: {num_classes!r}")

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=self.transformer_depth[level],
                                context_dim=_context_dims(context_dim, self.transformer_depth[level]),
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(
                # The middle block sits at the deepest level.
                ch, num_heads, dim_head, depth=self.transformer_depth[-1],
                context_dim=_context_dims(context_dim, self.transformer_depth[-1]),
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, y=None, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        # SDXL: fold the pooled text/size vector into the timestep embedding.
        if self.num_classes is not None:
            if y is None:
                raise ValueError(
                    "This ControlNet expects the SDXL conditioning vector 'y'")
            emb = emb + self.label_emb(y.type(emb.dtype))

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


# SD 1.5 ControlNet architecture parameters (matches cldm_v15.yaml)
SD15_CONTROLNET_PARAMS = dict(
    image_size=32,
    in_channels=4,
    model_channels=320,
    hint_channels=3,
    num_res_blocks=2,
    attention_resolutions=[4, 2, 1],
    channel_mult=[1, 2, 4, 4],
    use_checkpoint=True,
    use_fp16=True,
    num_heads=8,
    num_head_channels=-1,
    use_spatial_transformer=True,
    transformer_depth=1,
    context_dim=768,
    legacy=False,
)

# SDXL ControlNet architecture. Three levels (not four), no attention at level 0,
# per-level transformer depth, 64-channel heads, and the 2816-wide pooled vector
# (1280 text-pooled + 6*256 size embeddings) folded into the timestep embedding.
SDXL_CONTROLNET_PARAMS = dict(
    image_size=32,
    in_channels=4,
    model_channels=320,
    hint_channels=3,
    num_res_blocks=2,
    attention_resolutions=[4, 2],
    channel_mult=[1, 2, 4],
    use_checkpoint=True,
    use_fp16=True,
    num_heads=-1,
    num_head_channels=64,
    use_spatial_transformer=True,
    transformer_depth=[0, 2, 10],
    context_dim=2048,
    use_linear_in_transformer=True,
    num_classes='sequential',
    adm_in_channels=2816,
    legacy=False,
)

# SD 2.1 ControlNet architecture parameters
SD21_CONTROLNET_PARAMS = dict(
    image_size=32,
    in_channels=4,
    model_channels=320,
    hint_channels=3,
    num_res_blocks=2,
    attention_resolutions=[4, 2, 1],
    channel_mult=[1, 2, 4, 4],
    use_checkpoint=True,
    use_fp16=True,
    num_heads=-1,
    num_head_channels=64,
    use_spatial_transformer=True,
    transformer_depth=1,
    context_dim=1024,
    legacy=False,
)


def instantiate_controlnet(state_dict, model_version='control_multi_v15', device='cuda'):
    """Create a ControlNet nn.Module from a state dict.

    Detects architecture from state dict keys and instantiates the right model.

    Args:
        state_dict: raw state dict from .pth/.safetensors checkpoint
        model_version: model version string for architecture selection
        device: target device

    Returns:
        Instantiated ControlNet module with weights loaded.
    """
    # Every published SDXL ControlNet ships in diffusers format, which shares no
    # key names with cldm — detect it from the checkpoint rather than trusting
    # model_version, and translate before doing anything else.
    from vibewarp.core.controlnet_diffusers import (
        diffusers_to_cldm,
        is_diffusers_controlnet,
    )

    if is_diffusers_controlnet(state_dict):
        params = dict(SDXL_CONTROLNET_PARAMS)
        state_dict = diffusers_to_cldm(
            state_dict,
            channel_mult=params['channel_mult'],
            num_res_blocks=params['num_res_blocks'],
        )
    elif 'sdxl' in model_version.lower():
        params = dict(SDXL_CONTROLNET_PARAMS)
    elif 'v21' in model_version or 'v2' in model_version:
        params = dict(SD21_CONTROLNET_PARAMS)
    else:
        params = dict(SD15_CONTROLNET_PARAMS)

    # Auto-detect context_dim from state dict if possible. SDXL has no attention at
    # level 0, so input_blocks.1 carries no transformer — look at 4 as well.
    for probe in ('input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight',
                  'input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight'):
        if probe in state_dict:
            params['context_dim'] = state_dict[probe].shape[1]
            break

    # Strip 'control_model.' prefix if present (from ControlLDM checkpoints)
    cleaned_sd = {}
    for k, v in state_dict.items():
        if k.startswith('control_model.'):
            cleaned_sd[k[len('control_model.'):]] = v
        else:
            cleaned_sd[k] = v

    # Instantiate and load
    model = ControlNet(**params)
    m, u = model.load_state_dict(cleaned_sd, strict=False)
    if m:
        print(f"  ControlNet missing keys: {len(m)}")
    if u:
        print(f"  ControlNet unexpected keys: {len(u)}")

    model.half().to(device).eval()
    return model
