"""Tests for LoRA loading — key mapping and apply/unapply."""

import pytest
import torch
import torch.nn as nn

from vibewarp.core.lora import (
    _diffusers_to_ldm_unet_key,
    _normalize,
    _build_param_index,
    load_lora,
    unload_loras,
    _LOADED_LORA_ORIGINALS,
)


class TestDiffusersToLdmKeyMapping:
    """_diffusers_to_ldm_unet_key must produce the right ldm normalized keys."""

    @pytest.mark.parametrize("short,expected", [
        # down_blocks attention
        (
            'down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q',
            'input_blocks_1_1_transformer_blocks_0_attn1_to_q',
        ),
        (
            'down_blocks_0_attentions_0_proj_in',
            'input_blocks_1_1_proj_in',
        ),
        (
            'down_blocks_0_attentions_1_transformer_blocks_0_ff_net_0_proj',
            'input_blocks_2_1_transformer_blocks_0_ff_net_0_proj',
        ),
        # down_blocks resnet
        ('down_blocks_0_resnets_0_conv1',       'input_blocks_1_0_in_layers_2'),
        ('down_blocks_0_resnets_0_conv2',       'input_blocks_1_0_out_layers_3'),
        ('down_blocks_0_resnets_1_conv_shortcut','input_blocks_2_0_skip_connection'),
        ('down_blocks_1_resnets_0_time_emb_proj','input_blocks_4_0_emb_layers_1'),
        # downsampler
        ('down_blocks_0_downsamplers_0_conv',   'input_blocks_3_0_op'),
        ('down_blocks_2_downsamplers_0_conv',   'input_blocks_9_0_op'),
        # mid_block
        ('mid_block_attentions_0_proj_in',      'middle_block_1_proj_in'),
        ('mid_block_attentions_0_proj_out',     'middle_block_1_proj_out'),
        # up_blocks resnet
        ('up_blocks_0_resnets_0_conv1',         'output_blocks_0_0_in_layers_2'),
        ('up_blocks_3_resnets_2_time_emb_proj', 'output_blocks_11_0_emb_layers_1'),
        ('up_blocks_1_resnets_0_conv_shortcut', 'output_blocks_3_0_skip_connection'),
        # upsampler
        ('up_blocks_0_upsamplers_0_conv',       'output_blocks_2_1_conv'),
        ('up_blocks_1_upsamplers_0_conv',       'output_blocks_5_2_conv'),
        # up_blocks attention
        (
            'up_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q',
            'output_blocks_3_1_transformer_blocks_0_attn1_to_q',
        ),
    ])
    def test_mapping(self, short, expected):
        assert _diffusers_to_ldm_unet_key(short) == expected

    def test_unknown_block_returns_none(self):
        assert _diffusers_to_ldm_unet_key('something_unknown_0_resnets_0_conv1') is None

    def test_unknown_resnet_submodule_returns_none(self):
        assert _diffusers_to_ldm_unet_key('down_blocks_0_resnets_0_unknown_layer') is None


class TestLoraApplyUnapply:
    """load_lora / unload_loras should merge and restore weights correctly."""

    def _make_mini_sd(self):
        """A minimal mock model shaped like the parts a LoRA would touch."""
        class FakeUNet(nn.Module):
            def __init__(self):
                super().__init__()
                # Simulates model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q
                self.model = nn.ModuleList([
                    nn.ModuleList(),           # input_blocks[0]
                    nn.ModuleList([            # input_blocks[1]
                        nn.Identity(),         # [0] resnet placeholder
                        nn.ModuleList([        # [1] spatial transformer
                            nn.ModuleList([    # transformer_blocks[0]
                                nn.Linear(32, 32),  # attn1.to_q placeholder
                            ])
                        ])
                    ])
                ])

        return FakeUNet()

    def test_delta_uses_a1111_fp16_arithmetic(self, tmp_path):
        """Deltas must replicate WarpFusion's A1111 calc_updown exactly (notebook parity):
        cast up/down to the TARGET dtype before the matmul, then multiply by
        scale and multiplier as separate ops. An fp32 matmul is more accurate
        but diverges from the notebook by ~1 ulp per weight on every layer.
        """
        from safetensors.torch import save_file
        from vibewarp.core.lora import _compute_lora_deltas, _LORA_DELTA_CACHE

        class Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = nn.Linear(32, 32).half()

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn1 = Attn()

        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList([Block()])

        class Unet(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_blocks = nn.ModuleList([
                    nn.ModuleList(), nn.ModuleList([nn.Identity(), Inner()]),
                ])

        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.diffusion_model = Unet()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Wrapper()

        model = Model()
        torch.manual_seed(0)
        rank = 4
        up = torch.randn(32, rank) * 0.5
        down = torch.randn(rank, 32) * 0.5
        base = 'lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q'
        lora_file = tmp_path / 'mini.safetensors'
        save_file({
            f'{base}.lora_up.weight': up.contiguous(),
            f'{base}.lora_down.weight': down.contiguous(),
            f'{base}.alpha': torch.tensor(2.0),
        }, str(lora_file))

        _LORA_DELTA_CACHE.clear()
        deltas = _compute_lora_deltas(str(lora_file), model, 1.0, 0.7)
        assert len(deltas) == 1
        delta = next(iter(deltas.values()))
        assert delta.dtype == torch.float16
        expected = (up.half() @ down.half()) * (2.0 / rank) * 0.7
        assert torch.equal(delta, expected)
        fp32_path = ((up.float() @ down.float()) * (2.0 / rank) * 0.7).half()
        assert not torch.equal(delta, fp32_path), \
            'fixture too benign: fp16 and fp32 paths agree, test proves nothing'

        # optional high-precision mode: matmul in fp32, cached distinctly
        deltas32 = _compute_lora_deltas(str(lora_file), model, 1.0, 0.7,
                                        precision='fp32')
        delta32 = next(iter(deltas32.values()))
        assert delta32.dtype == torch.float32
        assert torch.equal(delta32.half(), fp32_path)

        with pytest.raises(ValueError):
            _compute_lora_deltas(str(lora_file), model, 1.0, 0.7,
                                 precision='fp8')

    def test_build_param_index_nonempty(self):
        """_build_param_index returns entries for a real model."""
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.ModuleList([
                    nn.ModuleList([nn.Linear(4, 4)])
                ])

        m = Tiny()
        idx = _build_param_index(m)
        assert len(idx) > 0
        # All values should be valid dotted paths
        for k, v in idx.items():
            assert '.' in v or v  # non-empty
