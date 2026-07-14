"""Tests for the vendored AnimateDiff motion modules (vendor/animatediff_mm.py).

Covers: state-dict detection/normalization (v1/v2/v3, HotshotXL key remap,
AnimateLCM/PIA detection), the zero-init identity property of a fresh
VanillaTemporalModule, checkpoint-compatible parameter naming, notebook-
faithful UNet injection/ejection positions, and the AnimateDiff beta-schedule
registration.
"""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from vibewarp.core.model_loader import _ensure_vendor_on_path

_ensure_vendor_on_path()

from vibewarp.vendor.animatediff_mm import (  # noqa: E402
    AnimateDiffFormat,
    AnimateDiffVersion,
    BlockType,
    MotionModule,
    MotionWrapper,
    VanillaTemporalModule,
    eject_motion_module_from_unet,
    get_encoding_max_len,
    get_motion_module,
    has_mid_block,
    inject_motion_module_to_unet,
    is_animatelcm,
    is_hotshotxl,
    is_pia,
    normalize_ad_state_dict,
    register_animatediff_schedule,
)


# ---- Synthetic state dicts (small, key-shape only where possible) ----

def make_min_ad_state(version='v2', sd='sd15'):
    """Minimal key set that exercises detection + normalization."""
    pe_len = 32 if version == 'v3' else 24
    state = {
        'down_blocks.0.motion_modules.0.temporal_transformer.pos_encoder.pe':
            torch.zeros(1, pe_len, 320),
        'down_blocks.3.motion_modules.0.temporal_transformer.proj_in.weight'
        if sd == 'sd15' else
        'down_blocks.2.motion_modules.0.temporal_transformer.proj_in.weight':
            torch.zeros(4, 4),
    }
    if version == 'v2':
        state['mid_block.motion_modules.0.temporal_transformer.proj_in.weight'] = torch.zeros(4, 4)
    return state


class TestDetection:
    def test_v2_detection(self):
        state = make_min_ad_state('v2')
        assert has_mid_block(state)
        assert get_encoding_max_len(state) == 24

    def test_v3_pe_len(self):
        state = make_min_ad_state('v3')
        assert not has_mid_block(state)
        assert get_encoding_max_len(state) == 32

    def test_hotshot_detection_and_remap(self):
        state = {
            'temporal_attentions.0.pos_encoder.positional_encoding':
                torch.zeros(1, 8, 320),
            'temporal_attentions.0.proj_in.weight': torch.zeros(4, 4),
            'down_blocks.2.dummy_temporal_marker': torch.zeros(1),
        }
        assert is_hotshotxl(state)
        normalized, info = normalize_ad_state_dict(dict(state), 'hsxl_test.ckpt')
        assert info.mm_format == AnimateDiffFormat.HOTSHOTXL
        assert info.sd_type == 'SDXL'
        assert 'motion_modules.0.temporal_transformer.pos_encoder.pe' in '\n'.join(normalized.keys())
        assert not any('temporal_attentions' in k for k in normalized.keys())

    def test_animatelcm_detection(self):
        # no pos_encoder keys at all → AnimateLCM
        state = {'down_blocks.0.motion_modules.0.temporal_transformer.proj_in.weight': torch.zeros(4, 4)}
        assert is_animatelcm(state)

    def test_pia_detection(self):
        state = {'conv_in.weight': torch.zeros(1), 'conv_in.bias': torch.zeros(1)}
        assert is_pia(state)

    def test_normalize_strips_non_temporal(self):
        state = make_min_ad_state('v2')
        state['some_random_extra.weight'] = torch.zeros(1)
        normalized, info = normalize_ad_state_dict(state, 'mm_sd_v15_v2.ckpt')
        assert 'some_random_extra.weight' not in normalized
        assert info.mm_version == AnimateDiffVersion.V2
        assert info.sd_type == 'SD1.5'

    def test_v3_version_detected(self):
        state = make_min_ad_state('v3')
        _, info = normalize_ad_state_dict(state, 'mm_sd15_v3.safetensors')
        assert info.mm_version == AnimateDiffVersion.V3


class TestModules:
    def test_fresh_module_is_identity(self):
        """zero_initialize zeroes proj_out → fresh module output == input."""
        m = get_motion_module(64, 24)
        m.set_video_length(2)
        x = torch.randn(4, 64, 8, 8)  # (b*f, c, h, w) with b=2, f=2
        with torch.no_grad():
            out = m(x, None)
        assert torch.allclose(out, x)

    def test_nonzero_proj_out_changes_output(self):
        m = VanillaTemporalModule(in_channels=64, zero_initialize=False,
                                  temporal_position_encoding_max_len=24)
        m.set_video_length(2)
        x = torch.randn(4, 64, 8, 8)
        with torch.no_grad():
            out = m(x, None)
        assert out.shape == x.shape
        assert not torch.allclose(out, x)

    def test_checkpoint_compatible_param_names(self):
        """Real AnimateDiff checkpoints use these key patterns; our SDPA
        attention / GEGLU ff replacements must not change the layout."""
        m = get_motion_module(64, 24)
        keys = set(m.state_dict().keys())
        prefix = 'temporal_transformer.'
        expected = [
            'norm.weight', 'proj_in.weight', 'proj_out.weight',
            'transformer_blocks.0.attention_blocks.0.to_q.weight',
            'transformer_blocks.0.attention_blocks.0.to_out.0.weight',
            'transformer_blocks.0.attention_blocks.0.pos_encoder.pe',
            'transformer_blocks.0.norms.0.weight',
            'transformer_blocks.0.ff.net.0.proj.weight',
            'transformer_blocks.0.ff.net.2.weight',
            'transformer_blocks.0.ff_norm.weight',
        ]
        for k in expected:
            assert prefix + k in keys, f"missing key: {prefix + k}"

    def test_motion_module_block_counts(self):
        down = MotionModule(64, BlockType.DOWN, encoding_max_len=24)
        up = MotionModule(64, BlockType.UP, encoding_max_len=24)
        mid = MotionModule(64, BlockType.MID, encoding_max_len=24)
        assert len(down.motion_modules) == 2
        assert len(up.motion_modules) == 3
        assert len(mid.motion_modules) == 1


# ---- Fake UNet for injection tests ----

class Downsample(nn.Module):
    def forward(self, x):
        return x


class Upsample(nn.Module):
    def forward(self, x):
        return x


class Spatial(nn.Module):
    """Stands in for a spatial transformer block."""
    def forward(self, x):
        return x


def make_fake_unet(is_sdxl=False):
    """Mimics ldm UNet block layout for the notebook's injection rules:
    SD1.5: 12 input / 12 output blocks; downsample at input 3, 6, 9;
    upsample at the last module of output 2, 5, 8.
    Blocks are nn.Sequential — like ldm's TimestepEmbedSequential — because
    the notebook relies on Sequential's negative-index insert/pop."""
    n = 12 if not is_sdxl else 9
    unet = nn.Module()
    unet.input_blocks = nn.ModuleList()
    unet.output_blocks = nn.ModuleList()
    for i in range(n):
        if i > 0 and i % 3 == 0:
            unet.input_blocks.append(nn.Sequential(Downsample()))
        else:
            unet.input_blocks.append(nn.Sequential(Spatial()))
        if i > 0 and (i + 1) % 3 == 0 and i != n - 1:
            unet.output_blocks.append(nn.Sequential(Spatial(), Upsample()))
        else:
            unet.output_blocks.append(nn.Sequential(Spatial()))
    unet.middle_block = nn.Sequential(Spatial(), Spatial(), Spatial())
    return unet


class _TinyWrapper:
    pass


def make_tiny_wrapper(is_v2=True, is_sdxl=False, is_v3=False, is_hotshot=False):
    """MotionWrapper-shaped object with tiny (64-ch) motion modules so tests
    stay fast — injection only indexes the structure, channels don't matter."""
    mm = _TinyWrapper()
    mm.is_v2 = is_v2
    mm.is_sdxl = is_sdxl
    mm.is_v3 = is_v3
    mm.is_hotshot = is_hotshot
    mm.hack_gn = not (is_sdxl or is_v3)
    n_levels = 4 if not is_sdxl else 3
    mm.down_blocks = [MotionModule(64, BlockType.DOWN, 24) for _ in range(n_levels)]
    mm.up_blocks = [MotionModule(64, BlockType.UP, 24) for _ in range(n_levels)]
    mm.mid_block = MotionModule(64, BlockType.MID, 24) if is_v2 else None
    return mm


class TestInjectEject:
    def _counts(self, unet):
        n_in = sum(1 for b in unet.input_blocks for m in b
                   if type(m).__name__ == 'VanillaTemporalModule')
        n_out = sum(1 for b in unet.output_blocks for m in b
                    if type(m).__name__ == 'VanillaTemporalModule')
        n_mid = sum(1 for m in unet.middle_block
                    if type(m).__name__ == 'VanillaTemporalModule')
        return n_in, n_out, n_mid

    def test_inject_v2_sd15_positions(self):
        unet = make_fake_unet()
        mm = make_tiny_wrapper(is_v2=True)
        inject_motion_module_to_unet(unet, mm)
        n_in, n_out, n_mid = self._counts(unet)
        # inputs: skipped where last module is Downsample (i=3,6,9) and i=0? —
        # i=0 has Spatial → injected. 12 - 3 downsample = 9 injections.
        assert n_in == 9
        assert n_out == 12
        assert n_mid == 1  # v2 middle block
        assert unet.mm_injected is True
        # Upsample blocks: temporal module inserted BEFORE the Upsample
        up_block = unet.output_blocks[2]
        assert type(up_block[-1]).__name__ == 'Upsample'
        assert type(up_block[-2]).__name__ == 'VanillaTemporalModule'

    def test_eject_restores_structure(self):
        unet = make_fake_unet()
        before = [len(b) for b in unet.input_blocks] + [len(b) for b in unet.output_blocks]
        mm = make_tiny_wrapper(is_v2=True)
        inject_motion_module_to_unet(unet, mm)
        eject_motion_module_from_unet(unet, mm)
        after = [len(b) for b in unet.input_blocks] + [len(b) for b in unet.output_blocks]
        assert before == after
        assert self._counts(unet) == (0, 0, 0)
        assert unet.mm_injected is False

    def test_double_inject_is_noop(self):
        unet = make_fake_unet()
        mm = make_tiny_wrapper(is_v2=True)
        inject_motion_module_to_unet(unet, mm)
        counts = self._counts(unet)
        inject_motion_module_to_unet(unet, mm)  # prints 'already injected'
        assert self._counts(unet) == counts

    def test_inject_sdxl_9_blocks(self):
        unet = make_fake_unet(is_sdxl=True)
        mm = make_tiny_wrapper(is_v2=False, is_sdxl=True)
        inject_motion_module_to_unet(unet, mm)
        n_in, n_out, n_mid = self._counts(unet)
        assert n_in == 7   # 9 blocks - downsample at 3, 6
        assert n_out == 9
        assert n_mid == 0  # not v2
        eject_motion_module_from_unet(unet, mm)
        assert self._counts(unet) == (0, 0, 0)

    def test_v1_gn_hack_applied_and_restored(self):
        from ldm.modules.diffusionmodules.util import GroupNorm32
        original = GroupNorm32.forward
        unet = make_fake_unet()
        mm = make_tiny_wrapper(is_v2=False, is_sdxl=False, is_v3=False)
        assert mm.hack_gn
        inject_motion_module_to_unet(unet, mm)
        assert GroupNorm32.forward is not original
        eject_motion_module_from_unet(unet, mm)
        assert GroupNorm32.forward is original


class TestMotionWrapper:
    def test_roundtrip_from_state_dict(self):
        """A wrapper's own state dict must round-trip through from_state_dict
        with identical detection (v2, encoding_max_len)."""
        src = MotionWrapper('src', encoding_max_len=24, is_v2=True)
        state = {k: v.clone() for k, v in src.state_dict().items()}
        mm = MotionWrapper.from_state_dict(state, 'roundtrip.ckpt')
        assert mm.is_v2 is True
        assert mm.encoding_max_len == 24
        assert mm.hack_gn is True  # not sdxl, not v3

    def test_sdxl_channel_layout(self):
        mm = MotionWrapper('x', encoding_max_len=24, is_sdxl=True)
        assert len(mm.down_blocks) == 3
        assert len(mm.up_blocks) == 3
        assert mm.hack_gn is False

    def test_v3_disables_gn_hack(self):
        mm = MotionWrapper('x', encoding_max_len=32, is_v3=True)
        assert mm.hack_gn is False


class TestRegisterSchedule:
    def test_sd15_reregisters_sqrt_linear(self):
        sd_model = MagicMock()
        mm = make_tiny_wrapper()
        register_animatediff_schedule(sd_model, mm, 'control_multi_animatediff_v15')
        sd_model.register_schedule.assert_called_once_with(
            given_betas=None, beta_schedule="sqrt_linear", timesteps=1000,
            linear_start=0.00085, linear_end=0.012, cosine_s=8e-3,
        )

    def test_sdxl_sets_ad_alphas_cumprod(self):
        class FakeModel(nn.Module):
            pass
        sd_model = FakeModel()
        mm = make_tiny_wrapper(is_sdxl=True, is_v2=False)
        register_animatediff_schedule(sd_model, mm, 'control_multi_animatediff_sdxl')
        ac = sd_model.ad_alphas_cumprod
        assert ac.shape == (1000,)
        # betas: linspace(sqrt(0.00085), sqrt(0.020))**2
        assert torch.isclose(ac[0], torch.tensor(1.0 - 0.00085), atol=1e-6)
        assert ac[-1] < 0.01  # heavy decay by t=1000
        # non-hotshot → alphas_cumprod buffer replaced too
        assert torch.equal(sd_model.alphas_cumprod, ac)

    def test_sdxl_hotshot_keeps_alphas_buffer(self):
        class FakeModel(nn.Module):
            pass
        sd_model = FakeModel()
        mm = make_tiny_wrapper(is_sdxl=True, is_v2=False, is_hotshot=True)
        register_animatediff_schedule(sd_model, mm, 'control_multi_animatediff_sdxl')
        assert hasattr(sd_model, 'ad_alphas_cumprod')
        assert not hasattr(sd_model, 'alphas_cumprod') or sd_model.alphas_cumprod is not sd_model.ad_alphas_cumprod


class TestBlockGrouping:
    """The block container is original to VibeWarp, but its SHAPE is not ours to choose.

    Motion modules ship as a standalone checkpoint keyed
    `down_blocks.<i>.motion_modules.<j>....`, so the module count per block is a property
    of that file format: 2 per down block (one per resnet), 1 for the mid block, 3 per up
    block (the decoder has an extra resnet for the skip connection). Get this wrong and the
    checkpoint loads with missing/unexpected keys and the motion module silently does
    nothing.
    """

    def test_modules_per_block_follows_the_checkpoint_layout(self):
        from vibewarp.vendor.animatediff_mm import (MOTION_MODULES_PER_BLOCK, BlockType,
                                                    MotionModule)

        assert MOTION_MODULES_PER_BLOCK == {
            BlockType.DOWN: 2, BlockType.MID: 1, BlockType.UP: 3}
        for block_type, expected in MOTION_MODULES_PER_BLOCK.items():
            block = MotionModule(320, block_type, encoding_max_len=24)
            assert len(block.motion_modules) == expected
            assert block.block_type == block_type

    def test_wrapper_exposes_the_keys_a_checkpoint_expects(self):
        from vibewarp.vendor.animatediff_mm import MotionWrapper

        wrapper = MotionWrapper('mm_sd_v15_v2.ckpt', encoding_max_len=24, is_v2=True)
        keys = wrapper.state_dict().keys()
        assert any(k.startswith('down_blocks.0.motion_modules.0.temporal_transformer.')
                   for k in keys)
        assert any(k.startswith('up_blocks.0.motion_modules.2.temporal_transformer.')
                   for k in keys)
        assert any(k.startswith('mid_block.motion_modules.0.temporal_transformer.')
                   for k in keys)
        # SD1.5 has four resolution levels; SDXL/Hotshot has three and no mid block.
        assert len(wrapper.down_blocks) == 4 and len(wrapper.up_blocks) == 4

    def test_sdxl_wrapper_has_three_levels_and_no_mid_block(self):
        from vibewarp.vendor.animatediff_mm import MotionWrapper

        wrapper = MotionWrapper('hsxl', encoding_max_len=32, is_sdxl=True, is_hotshot=True)
        assert len(wrapper.down_blocks) == 3 and len(wrapper.up_blocks) == 3
        assert wrapper.mid_block is None

    def test_set_video_length_reaches_every_module(self):
        from vibewarp.vendor.animatediff_mm import BlockType, MotionModule

        block = MotionModule(320, BlockType.UP, encoding_max_len=24)
        block.set_video_length(7)
        for module in block.motion_modules:
            assert module.temporal_transformer.video_length == 7
