"""SDXL ControlNet support: diffusers->cldm conversion and architecture.

Every published SDXL ControlNet ships in diffusers format, which shares no key
names with the vendored cldm ControlNet. The notebook doesn't help here — it
shells out to a full ComfyUI clone (`from comfy.sd import load_controlnet`), so
the conversion is ours and needs its own tests.
"""

import pytest
import torch

from vibewarp.core.controlnet_diffusers import (
    diffusers_to_cldm,
    is_diffusers_controlnet,
)
from vibewarp.vendor.cldm.cldm import (
    SD15_CONTROLNET_PARAMS,
    SDXL_CONTROLNET_PARAMS,
    ControlNet,
)


class TestFormatDetection:
    def test_detects_diffusers(self):
        assert is_diffusers_controlnet({'controlnet_down_blocks.0.weight': None})

    def test_cldm_is_not_diffusers(self):
        assert not is_diffusers_controlnet({'zero_convs.0.0.weight': None,
                                            'input_blocks.0.0.weight': None})


class TestKeyConversion:
    """Renames only — the two layouts describe the same graph."""

    @pytest.mark.parametrize("diffusers, cldm", [
        ('conv_in.weight', 'input_blocks.0.0.weight'),
        ('time_embedding.linear_1.weight', 'time_embed.0.weight'),
        ('time_embedding.linear_2.bias', 'time_embed.2.bias'),
        # SDXL's pooled vector projection.
        ('add_embedding.linear_1.weight', 'label_emb.0.0.weight'),
        ('add_embedding.linear_2.bias', 'label_emb.0.2.bias'),
        # Zero convs.
        ('controlnet_down_blocks.3.weight', 'zero_convs.3.0.weight'),
        ('controlnet_mid_block.bias', 'middle_block_out.0.bias'),
        # Hint encoder: SiLUs occupy the odd indices in cldm.
        ('controlnet_cond_embedding.conv_in.weight', 'input_hint_block.0.weight'),
        ('controlnet_cond_embedding.blocks.0.weight', 'input_hint_block.2.weight'),
        ('controlnet_cond_embedding.blocks.5.bias', 'input_hint_block.12.bias'),
        ('controlnet_cond_embedding.conv_out.weight', 'input_hint_block.14.weight'),
        # Level 0 has two resnets and no attention -> input_blocks 1 and 2.
        ('down_blocks.0.resnets.0.norm1.weight', 'input_blocks.1.0.in_layers.0.weight'),
        ('down_blocks.0.resnets.1.conv2.bias', 'input_blocks.2.0.out_layers.3.bias'),
        ('down_blocks.0.resnets.0.time_emb_proj.weight', 'input_blocks.1.0.emb_layers.1.weight'),
        # Downsampler -> the Downsample module's .op
        ('down_blocks.0.downsamplers.0.conv.weight', 'input_blocks.3.0.op.weight'),
        # Level 1 starts at input_blocks.4; attention is the .1 slot.
        ('down_blocks.1.resnets.0.conv1.weight', 'input_blocks.4.0.in_layers.2.weight'),
        ('down_blocks.1.attentions.0.proj_in.weight', 'input_blocks.4.1.proj_in.weight'),
        ('down_blocks.1.attentions.1.norm.bias', 'input_blocks.5.1.norm.bias'),
        ('down_blocks.1.resnets.0.conv_shortcut.weight', 'input_blocks.4.0.skip_connection.weight'),
        # Level 2 (last) has no downsampler.
        ('down_blocks.2.attentions.1.proj_out.weight', 'input_blocks.8.1.proj_out.weight'),
        # Middle: resnets straddle the attention (0, 1, 2).
        ('mid_block.resnets.0.norm1.weight', 'middle_block.0.in_layers.0.weight'),
        ('mid_block.attentions.0.proj_in.weight', 'middle_block.1.proj_in.weight'),
        ('mid_block.resnets.1.conv2.bias', 'middle_block.2.out_layers.3.bias'),
    ])
    def test_key_maps(self, diffusers, cldm):
        out = diffusers_to_cldm({diffusers: torch.zeros(1)})
        assert list(out) == [cldm]

    def test_transformer_block_leaves_pass_through(self):
        """attn1/attn2/ff/norm names already agree between the two layouts."""
        key = 'down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight'
        out = diffusers_to_cldm({key: torch.zeros(1)})
        assert list(out) == [
            'input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight']

    def test_unknown_key_raises_rather_than_being_dropped(self):
        """A silently dropped key would load as a random-init layer."""
        with pytest.raises(KeyError):
            diffusers_to_cldm({'some_new_block.weight': torch.zeros(1)})


class TestSDXLArchitecture:
    def test_params_describe_sdxl(self):
        assert SDXL_CONTROLNET_PARAMS['context_dim'] == 2048
        assert SDXL_CONTROLNET_PARAMS['adm_in_channels'] == 2816
        assert SDXL_CONTROLNET_PARAMS['num_classes'] == 'sequential'
        # Three levels, no attention on level 0.
        assert SDXL_CONTROLNET_PARAMS['channel_mult'] == [1, 2, 4]
        assert SDXL_CONTROLNET_PARAMS['transformer_depth'] == [0, 2, 10]

    def test_per_level_transformer_depth(self):
        """SD1.5 passes an int; SDXL passes a list. Both must work."""
        net = ControlNet(**SDXL_CONTROLNET_PARAMS)
        assert net.transformer_depth == [0, 2, 10]
        # An int is broadcast across the levels.
        sd15 = ControlNet(**SD15_CONTROLNET_PARAMS)
        assert sd15.transformer_depth == [1, 1, 1, 1]

    def test_sdxl_builds_the_label_embedding(self):
        net = ControlNet(**SDXL_CONTROLNET_PARAMS)
        assert net.num_classes == 'sequential'
        assert net.label_emb[0][0].in_features == 2816

    def test_sd15_has_no_label_embedding(self):
        net = ControlNet(**SD15_CONTROLNET_PARAMS)
        assert net.num_classes is None
        assert not hasattr(net, 'label_emb')

    def test_sdxl_forward_requires_the_vector(self):
        """Without y the ControlNet would silently drop SDXL's size/pooled
        conditioning instead of failing."""
        net = ControlNet(**SDXL_CONTROLNET_PARAMS).float()
        net.dtype = torch.float32
        with pytest.raises(ValueError, match="'y'"):
            net(x=torch.zeros(1, 4, 8, 8), hint=torch.zeros(1, 3, 64, 64),
                timesteps=torch.tensor([1]), context=torch.zeros(1, 77, 2048))


class TestConvertedCheckpointFitsTheModel:
    """The whole conversion is only correct if a real checkpoint loads with no
    missing and no unexpected keys. Synthesised here from the model's own
    parameter names so it runs without the 2.4 GB download."""

    def test_round_trip_has_no_missing_or_unexpected_keys(self):
        net = ControlNet(**SDXL_CONTROLNET_PARAMS)
        cldm_keys = set(net.state_dict())

        # Build the diffusers-side names, then convert them back.
        diffusers = _synthetic_diffusers_keys()
        converted = diffusers_to_cldm(
            {k: torch.zeros(1) for k in diffusers},
            channel_mult=SDXL_CONTROLNET_PARAMS['channel_mult'],
            num_res_blocks=SDXL_CONTROLNET_PARAMS['num_res_blocks'],
        )
        assert set(converted) <= cldm_keys, sorted(set(converted) - cldm_keys)


def _synthetic_diffusers_keys():
    """A representative diffusers key from every structural family."""
    return [
        'conv_in.weight',
        'time_embedding.linear_1.weight',
        'add_embedding.linear_2.weight',
        'controlnet_cond_embedding.conv_in.weight',
        'controlnet_cond_embedding.blocks.4.weight',
        'controlnet_cond_embedding.conv_out.weight',
        'controlnet_down_blocks.0.weight',
        'controlnet_down_blocks.8.weight',
        'controlnet_mid_block.weight',
        'down_blocks.0.resnets.0.norm1.weight',
        'down_blocks.0.downsamplers.0.conv.weight',
        'down_blocks.1.resnets.1.conv1.weight',
        'down_blocks.1.attentions.1.proj_in.weight',
        'down_blocks.1.downsamplers.0.conv.weight',
        'down_blocks.2.resnets.0.conv2.weight',
        'down_blocks.2.attentions.0.transformer_blocks.9.attn1.to_q.weight',
        'mid_block.resnets.0.norm1.weight',
        'mid_block.attentions.0.proj_out.weight',
        'mid_block.resnets.1.conv2.weight',
    ]


class TestSD15Unaffected:
    """The vendored SpatialTransformer/ControlNet changes are shared with SD1.5."""

    def test_sd15_controlnet_still_builds(self):
        net = ControlNet(**SD15_CONTROLNET_PARAMS)
        # 4 levels x 2 res blocks + 3 downsamples + conv_in = 12 input blocks.
        assert len(net.input_blocks) == 12
        assert len(net.zero_convs) == 12

    def test_sd15_forward_needs_no_vector(self):
        net = ControlNet(**SD15_CONTROLNET_PARAMS).float()
        net.dtype = torch.float32
        out = net(x=torch.zeros(1, 4, 8, 8), hint=torch.zeros(1, 3, 64, 64),
                  timesteps=torch.tensor([1]), context=torch.zeros(1, 77, 768))
        assert len(out) == 13  # 12 zero convs + middle
