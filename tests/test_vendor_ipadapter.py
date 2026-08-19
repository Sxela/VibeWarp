"""Tests for the vendored controlmodel_ipadapter (PlugableIPAdapter).

Exercises the real vendored module on CPU with a synthetic SD1.5-style
adapter state dict and a minimal fake UNet built from real (vendored) ldm
CrossAttention blocks — verifying hook installation, the hacked attention
forward with IP contribution, and clean unhooking.
"""

import pytest
import torch
import torch.nn as nn

from vibewarp.core.model_loader import _ensure_vendor_on_path

_ensure_vendor_on_path()

from vibewarp.vendor.controlmodel_ipadapter import (  # noqa: E402
    PlugableIPAdapter,
    clear_all_ip_adapter,
    clip_vision_h_uc,
    clip_vision_vith_uc,
    composition_precise_weights,
)

CROSS_DIM = 768   # SD1.5 cross-attention dim (read from '1.to_k_ip.weight')
CLIP_DIM = 1024   # ViT-H image_embeds dim
TOKENS = 4        # non-plus adapter context tokens
INNER = 64        # attention inner dim in the fake UNet
HEADS = 4


def make_sd15_adapter_state():
    """Synthetic non-plus SD1.5 IP-Adapter state dict (16 attn layers)."""
    torch.manual_seed(0)
    image_proj = {
        'proj.weight': torch.randn(TOKENS * CROSS_DIM, CLIP_DIM) * 0.02,
        'proj.bias': torch.zeros(TOKENS * CROSS_DIM),
        'norm.weight': torch.ones(CROSS_DIM),
        'norm.bias': torch.zeros(CROSS_DIM),
    }
    ip_adapter = {}
    for n in range(16):
        key = n * 2 + 1
        ip_adapter[f'{key}.to_k_ip.weight'] = torch.randn(INNER, CROSS_DIM) * 0.02
        ip_adapter[f'{key}.to_v_ip.weight'] = torch.randn(INNER, CROSS_DIM) * 0.02
    return {'image_proj': image_proj, 'ip_adapter': ip_adapter}


SDXL_CROSS_DIM = 2048   # what PlugableIPAdapter sniffs to set .sdxl
SDXL_LAYERS = 70        # attention layers in an SDXL IP-Adapter checkpoint


def make_sdxl_adapter_state():
    """Synthetic non-plus SDXL IP-Adapter state dict (70 attn layers)."""
    torch.manual_seed(0)
    image_proj = {
        'proj.weight': torch.randn(TOKENS * SDXL_CROSS_DIM, CLIP_DIM) * 0.02,
        'proj.bias': torch.zeros(TOKENS * SDXL_CROSS_DIM),
        'norm.weight': torch.ones(SDXL_CROSS_DIM),
        'norm.bias': torch.zeros(SDXL_CROSS_DIM),
    }
    ip_adapter = {}
    for n in range(SDXL_LAYERS):
        key = n * 2 + 1
        ip_adapter[f'{key}.to_k_ip.weight'] = (
            torch.randn(INNER, SDXL_CROSS_DIM) * 0.02)
        ip_adapter[f'{key}.to_v_ip.weight'] = (
            torch.randn(INNER, SDXL_CROSS_DIM) * 0.02)
    return {'image_proj': image_proj, 'ip_adapter': ip_adapter}


def make_fake_sdxl_unet():
    """SDXL block skeleton with the real transformer depths.

    Only the structure the hook walks matters: which blocks carry cross
    attention and how many inner transformer_blocks each holds.
    """
    from sgm.modules.attention import CrossAttention

    def spatial(depth):
        holder = nn.Module()
        holder.transformer_blocks = nn.ModuleList()
        for _ in range(depth):
            blk = nn.Module()
            blk.attn2 = CrossAttention(
                query_dim=INNER, context_dim=SDXL_CROSS_DIM,
                heads=HEADS, dim_head=INNER // HEADS,
            )
            holder.transformer_blocks.append(blk)
        return holder

    def stack(depths):
        return nn.ModuleList(
            [nn.ModuleList([nn.Identity(), spatial(d)]) for d in depths])

    unet = nn.Module()
    # input ids 4,5 have depth 2; ids 7,8 have depth 10; the rest are unused
    unet.input_blocks = stack([1, 1, 1, 1, 2, 2, 1, 10, 10])
    # output ids 0,1,2 have depth 10; ids 3,4,5 have depth 2
    unet.output_blocks = stack([10, 10, 10, 2, 2, 2])
    unet.middle_block = nn.ModuleList([nn.Identity(), spatial(10)])
    return unet


def make_fake_unet():
    """Minimal UNet skeleton matching the hook's block indexing.

    get_block(model, flag)[id][1].transformer_blocks[0].attn2 must be a
    (vendored) ldm CrossAttention — the same class the hook targets.
    """
    from ldm.modules.attention import CrossAttention

    def spatial():
        holder = nn.Module()
        blk = nn.Module()
        blk.attn2 = CrossAttention(
            query_dim=INNER, context_dim=CROSS_DIM,
            heads=HEADS, dim_head=INNER // HEADS,
        )
        holder.transformer_blocks = nn.ModuleList([blk])
        return holder

    unet = nn.Module()
    # input ids used: 1,2,4,5,7,8 → need 9 entries; output ids: 3..11 → 12
    unet.input_blocks = nn.ModuleList(
        [nn.ModuleList([nn.Identity(), spatial()]) for _ in range(9)])
    unet.output_blocks = nn.ModuleList(
        [nn.ModuleList([nn.Identity(), spatial()]) for _ in range(12)])
    unet.middle_block = nn.ModuleList([nn.Identity(), spatial()])
    return unet


class TestVendoredModule:
    def test_uncond_embeds_loaded(self):
        """The .data files must ship with the vendor dir and load on CPU."""
        assert isinstance(clip_vision_h_uc, torch.Tensor)
        assert isinstance(clip_vision_vith_uc, torch.Tensor)
        assert clip_vision_h_uc.device.type == 'cpu'

    def test_adapter_flags_sd15_nonplus(self):
        adapter = PlugableIPAdapter(make_sd15_adapter_state())
        assert adapter.is_plus is False
        assert adapter.is_full is False
        assert adapter.is_faceid is False
        assert adapter.sdxl is False
        assert adapter.ipadapter.clip_extra_context_tokens == TOKENS

    def test_image_embeds_shapes(self):
        adapter = PlugableIPAdapter(make_sd15_adapter_state())
        clip_out = {'image_embeds': torch.randn(1, CLIP_DIM)}
        cond, uncond = adapter.ipadapter.get_image_embeds(clip_out)
        assert cond.shape == (1, TOKENS, CROSS_DIM)
        assert uncond.shape == (1, TOKENS, CROSS_DIM)
        assert not torch.allclose(cond, uncond)


def sdpa_reference(attn, x, ctx):
    """Independent plain-attention reference (matches attn_forward_hacked
    minus the IP contribution). The vendored ldm CrossAttention.forward
    itself is CUDA-only on this box, so we can't call it on CPU."""
    b, s, d = x.shape
    h = attn.heads
    hd = d // h
    q, k, v = attn.to_q(x), attn.to_k(ctx), attn.to_v(ctx)
    q, k, v = [t.view(b, -1, h, hd).transpose(1, 2) for t in (q, k, v)]
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    out = out.transpose(1, 2).reshape(b, -1, h * hd)
    return attn.to_out(out)


class TestHookForwardUnhook:
    def teardown_method(self):
        clear_all_ip_adapter()

    def _hook(self, unet, adapter, weight=1.0, start=0.0, end=1.0,
              weight_type='linear'):
        adapter.hook(
            model=unet,
            clip_vision_output={'image_embeds': torch.randn(1, CLIP_DIM)},
            weight=weight, start=start, end=end,
            weight_type=weight_type, embeds_scaling='V only',
        )

    def test_hook_replaces_forwards(self):
        adapter = PlugableIPAdapter(make_sd15_adapter_state())
        unet = make_fake_unet()
        self._hook(unet, adapter)
        attn = unet.input_blocks[1][1].transformer_blocks[0].attn2
        assert hasattr(attn, 'ipadapter_hacks')
        assert len(attn.ipadapter_hacks) == 1
        # 6 input + 9 output + 1 middle = 16 hooked blocks
        hooked = sum(
            1 for m in unet.modules() if getattr(m, 'ipadapter_hacks', None))
        assert hooked == 16

    def test_composition_precise_sd15_layer_map(self):
        adapter = PlugableIPAdapter(make_sd15_adapter_state())
        unet = make_fake_unet()
        self._hook(
            unet, adapter, weight=2.0,
            weight_type='composition precise')

        assert adapter.weight == {
            0: 2.0, 1: 2.0, 2: 2.0, 3: 2.0,
            4: 0.5, 5: 2.0,
            6: 0.2, 7: 0.2, 8: 0.2,
            9: 2.0, 10: 2.0, 11: 2.0, 12: 2.0,
            13: 2.0, 14: 2.0, 15: 2.0,
        }

    def test_composition_precise_sdxl_layer_map(self):
        assert composition_precise_weights(2.0, is_sdxl=True) == {
            0: 0.2, 1: 0.2, 2: 0.2,
            3: 2.0,
            4: 0.2, 5: 0.2,
            6: 2.0,
            7: 0.2, 8: 0.2, 9: 0.2, 10: 0.2,
        }

    def test_forward_adds_ip_contribution_and_unhook_restores(self):
        torch.manual_seed(1)
        adapter = PlugableIPAdapter(make_sd15_adapter_state())
        unet = make_fake_unet()
        attn = unet.input_blocks[1][1].transformer_blocks[0].attn2
        original_forward = attn.forward

        x = torch.randn(2, 8, INNER)
        ctx = torch.randn(2, 77, CROSS_DIM)
        with torch.no_grad():
            plain = sdpa_reference(attn, x, ctx)

        self._hook(unet, adapter, weight=1.0)
        # CFGDenoiser sets this per batch: 0 = uncond row, 1 = cond row
        unet.uc_mask_shape = torch.tensor([0.0, 1.0])
        with torch.no_grad():
            after = attn(x, context=ctx)

        assert after.shape == plain.shape
        assert not torch.allclose(plain, after, atol=1e-4)

        clear_all_ip_adapter()
        assert attn.forward == original_forward
        assert attn.ipadapter_hacks == []

    def test_zero_weight_matches_plain_attention(self):
        """weight=0 → IP contribution is zero; hacked forward must equal
        plain attention."""
        torch.manual_seed(2)
        adapter = PlugableIPAdapter(make_sd15_adapter_state())
        unet = make_fake_unet()
        attn = unet.input_blocks[1][1].transformer_blocks[0].attn2

        x = torch.randn(2, 8, INNER)
        ctx = torch.randn(2, 77, CROSS_DIM)

        self._hook(unet, adapter, weight=0.0)
        unet.uc_mask_shape = torch.tensor([0.0, 1.0])
        with torch.no_grad():
            hacked = attn(x, context=ctx)
            plain = sdpa_reference(attn, x, ctx)
        assert torch.allclose(hacked, plain, atol=1e-5)

    def _active_blocks(self, unet, x, ctx):
        """Names of hooked blocks whose IP contribution is non-zero.

        A layer-weight preset returns 0 for any block whose t_idx is absent
        from the weight dict, so this reports which physical UNet blocks a
        preset actually targets.
        """
        active = []

        def contributes(attn):
            with torch.no_grad():
                return not torch.allclose(
                    attn(x, context=ctx), sdpa_reference(attn, x, ctx),
                    atol=1e-6)

        for label, blocks in (('input', unet.input_blocks),
                              ('output', unet.output_blocks)):
            for i, b in enumerate(blocks):
                attn = b[1].transformer_blocks[0].attn2
                if getattr(attn, 'ipadapter_hacks', None) and contributes(attn):
                    active.append(f'{label}[{i}]')
        middle = unet.middle_block[1].transformer_blocks[0].attn2
        if contributes(middle):
            active.append('middle')
        return active

    def test_composition_preset_targets_the_deepest_input_blocks(self):
        """The weight presets are keyed by ComfyUI's `transformer_index`:
        SpatialTransformer modules in EXECUTION order (input -> middle ->
        output), which is NOT the order the checkpoint's to_kvs layers are
        stored in (input -> output -> middle).

        SD1.5 `composition` is {4: w*0.25, 5: w}, so it must land on the two
        deepest input blocks — the ldm equivalent of InstantStyle's
        down_blocks.2.attentions.*, which is where spatial layout lives.
        """
        torch.manual_seed(6)
        adapter = PlugableIPAdapter(make_sd15_adapter_state())
        unet = make_fake_unet()
        x = torch.randn(2, 8, INNER)
        ctx = torch.randn(2, 77, CROSS_DIM)

        self._hook(unet, adapter, weight=1.0, weight_type='composition')
        unet.uc_mask_shape = torch.tensor([0.0, 1.0])
        assert self._active_blocks(unet, x, ctx) == ['input[7]', 'input[8]']

    def test_style_transfer_preset_skips_the_middle_block(self):
        """SD1.5 `style transfer` is {0,1,2,3, 9..15}: the shallow input blocks
        and the late (high-resolution) output blocks. The middle block is the
        most semantic layer and must NOT be included — under the old block-id
        indexing it wrongly was, while keys 12-15 were unreachable."""
        torch.manual_seed(7)
        adapter = PlugableIPAdapter(make_sd15_adapter_state())
        unet = make_fake_unet()
        x = torch.randn(2, 8, INNER)
        ctx = torch.randn(2, 77, CROSS_DIM)

        self._hook(unet, adapter, weight=1.0, weight_type='style transfer')
        unet.uc_mask_shape = torch.tensor([0.0, 1.0])
        active = self._active_blocks(unet, x, ctx)

        assert 'middle' not in active
        assert active == [
            'input[1]', 'input[2]', 'input[4]', 'input[5]',
            'output[5]', 'output[6]', 'output[7]', 'output[8]',
            'output[9]', 'output[10]', 'output[11]',
        ]

    def test_every_weight_map_key_is_reachable_exactly_once(self):
        """t_idx must be a bijection onto 0..15 for SD1.5. The old indexing
        used the raw block id: it collided (input/output both had 4,5,7,8) and
        never produced 12-15, silently dropping those preset entries."""
        torch.manual_seed(8)
        adapter = PlugableIPAdapter(make_sd15_adapter_state())
        unet = make_fake_unet()
        x = torch.randn(2, 8, INNER)
        ctx = torch.randn(2, 77, CROSS_DIM)

        seen = []
        for t_idx in range(16):
            clear_all_ip_adapter()
            unet = make_fake_unet()
            adapter = PlugableIPAdapter(make_sd15_adapter_state())
            adapter.hook(
                model=unet,
                clip_vision_output={'image_embeds': torch.randn(1, CLIP_DIM)},
                weight=1.0, start=0.0, end=1.0, weight_type='linear',
                embeds_scaling='V only', layer_weights=f'{t_idx}:1.0')
            unet.uc_mask_shape = torch.tensor([0.0, 1.0])
            active = self._active_blocks(unet, x, ctx)
            assert len(active) == 1, f't_idx {t_idx} hit {active}'
            seen.append(active[0])

        assert len(set(seen)) == 16, f'collisions: {seen}'

    def test_sdxl_transformer_index_matches_instantstyle_blocks(self):
        """SDXL has 70 hooked attention layers but only 11 t_idx values: every
        inner block of one SpatialTransformer shares its module's index, in
        execution order (input 0-3, middle 4, output 5-10).

        The presets rely on this: `composition` == {3} must be input_blocks[8]
        (down_blocks.2.attentions.1) and `style transfer` == {6} must be
        output_blocks[1] (up_blocks.0.attentions.1) — the two layers
        InstantStyle identifies as layout and style.
        """
        adapter = PlugableIPAdapter(make_sdxl_adapter_state())
        assert adapter.sdxl is True
        unet = make_fake_sdxl_unet()
        adapter.hook(
            model=unet,
            clip_vision_output={'image_embeds': torch.randn(1, CLIP_DIM)},
            weight=1.0, start=0.0, end=1.0, weight_type='linear',
            embeds_scaling='V only')

        def t_indices(spatial):
            """t_idx values carried by one SpatialTransformer's inner blocks."""
            found = set()
            for blk in spatial.transformer_blocks:
                for f in getattr(blk.attn2, 'ipadapter_hacks', []) or []:
                    inner = f.__wrapped__  # torch.no_grad() wraps the closure
                    idx = inner.__code__.co_freevars.index('t_idx')
                    found.add(inner.__closure__[idx].cell_contents)
            return found

        assert t_indices(unet.input_blocks[8][1]) == {3}, 'composition layer'
        assert t_indices(unet.output_blocks[1][1]) == {6}, 'style layer'
        assert t_indices(unet.middle_block[1]) == {4}

        seen = {}
        for label, blocks in (('input', unet.input_blocks),
                              ('output', unet.output_blocks)):
            for i, b in enumerate(blocks):
                got = t_indices(b[1])
                if got:
                    assert len(got) == 1, f'{label}[{i}] spans {got}'
                    seen[f'{label}[{i}]'] = got.pop()
        seen['middle'] = 4
        # 11 modules, one distinct index each, covering 0..10
        assert sorted(seen.values()) == list(range(11)), seen

    def _capture_ip_feat(self, adapter, attn, x, ctx):
        """Run the hacked forward, returning the embedding handed to to_k_ip.

        That tensor is ``cond * cond_mark + uncond * (1 - cond_mark)``, so it
        shows directly whether the layer's conditioning was inverted.
        """
        seen = {}
        original_call_ip = adapter.call_ip

        def recording_call_ip(key, feat, device):
            seen.setdefault('feat', feat)
            return original_call_ip(key, feat, device)

        adapter.call_ip = recording_call_ip
        with torch.no_grad():
            attn(x, context=ctx)
        return seen['feat']

    def test_composition_precise_inverts_non_composition_layers(self):
        """ComfyUI's "precise" presets do not merely attenuate the unwanted
        layers — they move their conditioning into the uncond slot
        (``uncond = cond; cond = cond * 0``). For "composition precise" that
        applies to every layer except the composition pair (SD1.5: 4 and 5)."""
        torch.manual_seed(4)
        adapter = PlugableIPAdapter(make_sd15_adapter_state())
        unet = make_fake_unet()

        x = torch.randn(2, 8, INNER)
        ctx = torch.randn(2, 77, CROSS_DIM)

        self._hook(unet, adapter, weight=1.0,
                   weight_type='composition precise')
        unet.uc_mask_shape = torch.tensor([0.0, 1.0])
        cond = adapter.image_emb
        uncond = adapter.uncond_image_emb

        # input_blocks[7] is t_idx 4 -> a composition layer: untouched, so the
        # uncond row keeps the real uncond embedding.
        kept = self._capture_ip_feat(
            adapter, unet.input_blocks[7][1].transformer_blocks[0].attn2,
            x, ctx)
        assert torch.allclose(kept[0], uncond[0], atol=1e-6)
        assert torch.allclose(kept[1], cond[0], atol=1e-6)

        # input_blocks[1] is t_idx 0 -> not a composition layer, so it is
        # inverted: the cond row is zeroed and the uncond row gets cond.
        inverted = self._capture_ip_feat(
            adapter, unet.input_blocks[1][1].transformer_blocks[0].attn2,
            x, ctx)
        assert torch.allclose(inverted[0], cond[0], atol=1e-6)
        assert torch.allclose(inverted[1], torch.zeros_like(inverted[1]))

    def test_style_transfer_precise_inverts_the_composition_layer(self):
        """The mirror image of the above: only layers 4/5 are inverted."""
        torch.manual_seed(5)
        adapter = PlugableIPAdapter(make_sd15_adapter_state())
        unet = make_fake_unet()

        x = torch.randn(2, 8, INNER)
        ctx = torch.randn(2, 77, CROSS_DIM)

        self._hook(unet, adapter, weight=1.0,
                   weight_type='style transfer precise')
        unet.uc_mask_shape = torch.tensor([0.0, 1.0])
        cond = adapter.image_emb
        uncond = adapter.uncond_image_emb

        inverted = self._capture_ip_feat(
            adapter, unet.input_blocks[7][1].transformer_blocks[0].attn2,
            x, ctx)
        assert torch.allclose(inverted[0], cond[0], atol=1e-6)
        assert torch.allclose(inverted[1], torch.zeros_like(inverted[1]))

        kept = self._capture_ip_feat(
            adapter, unet.input_blocks[1][1].transformer_blocks[0].attn2,
            x, ctx)
        assert torch.allclose(kept[0], uncond[0], atol=1e-6)
        assert torch.allclose(kept[1], cond[0], atol=1e-6)

    def test_start_end_gating_defaults_to_active(self):
        """Notebook never sets current_sampling_percent → getattr default 0.5.
        start=0.6 gates the hook OFF (0.5 < 0.6): output equals plain attention."""
        torch.manual_seed(3)
        adapter = PlugableIPAdapter(make_sd15_adapter_state())
        unet = make_fake_unet()
        attn = unet.input_blocks[1][1].transformer_blocks[0].attn2

        x = torch.randn(2, 8, INNER)
        ctx = torch.randn(2, 77, CROSS_DIM)

        self._hook(unet, adapter, weight=1.0, start=0.6, end=1.0)
        unet.uc_mask_shape = torch.tensor([0.0, 1.0])
        with torch.no_grad():
            gated = attn(x, context=ctx)
            plain = sdpa_reference(attn, x, ctx)
        assert torch.allclose(gated, plain, atol=1e-5)


class TestLegacyLayerAddressing(TestHookForwardUnhook):
    """Every release before 0.7.1 addressed the layer-weight presets by a
    block's index within its own list. That numbering collides input against
    output blocks and never reaches the last few entries, so it matches no
    other implementation — but renders were made with it, and reproducing them
    needs it back."""

    def _hook_legacy(self, unet, adapter, weight_type):
        adapter.hook(
            model=unet,
            clip_vision_output={'image_embeds': torch.randn(1, CLIP_DIM)},
            weight=1.0, start=0.0, end=1.0, weight_type=weight_type,
            embeds_scaling='V only', legacy_layer_indexing=True)

    def test_legacy_targets_the_old_blocks(self):
        """`composition` is {4: w*0.25, 5: w}. Under execution order that is
        the two deepest input blocks; under block ids it hit input 4/5 AND
        output 4/5, because those ids collide."""
        torch.manual_seed(20)
        adapter = PlugableIPAdapter(make_sd15_adapter_state())
        unet = make_fake_unet()
        x = torch.randn(2, 8, INNER)
        ctx = torch.randn(2, 77, CROSS_DIM)

        self._hook_legacy(unet, adapter, 'composition')
        unet.uc_mask_shape = torch.tensor([0.0, 1.0])
        assert self._active_blocks(unet, x, ctx) == [
            'input[4]', 'input[5]', 'output[4]', 'output[5]']

    def test_default_stays_on_the_corrected_addressing(self):
        torch.manual_seed(21)
        adapter = PlugableIPAdapter(make_sd15_adapter_state())
        unet = make_fake_unet()
        x = torch.randn(2, 8, INNER)
        ctx = torch.randn(2, 77, CROSS_DIM)

        self._hook(unet, adapter, weight=1.0, weight_type='composition')
        unet.uc_mask_shape = torch.tensor([0.0, 1.0])
        assert self._active_blocks(unet, x, ctx) == ['input[7]', 'input[8]']

    def test_linear_is_identical_either_way(self):
        """'linear' resolves to a scalar and never consults the index, so the
        legacy switch must be a no-op for it — the common case is unaffected."""
        torch.manual_seed(22)
        x = torch.randn(2, 8, INNER)
        ctx = torch.randn(2, 77, CROSS_DIM)
        outputs = []
        for legacy in (False, True):
            clear_all_ip_adapter()
            torch.manual_seed(99)
            adapter = PlugableIPAdapter(make_sd15_adapter_state())
            unet = make_fake_unet()
            adapter.hook(
                model=unet,
                clip_vision_output={'image_embeds': torch.zeros(1, CLIP_DIM)},
                weight=1.0, start=0.0, end=1.0, weight_type='linear',
                embeds_scaling='V only', legacy_layer_indexing=legacy)
            unet.uc_mask_shape = torch.tensor([0.0, 1.0])
            attn = unet.input_blocks[1][1].transformer_blocks[0].attn2
            with torch.no_grad():
                outputs.append(attn(x, context=ctx))
        assert torch.allclose(outputs[0], outputs[1], atol=1e-6)


class TestBlockTypePresets:
    """weak input/middle/output and strong middle scale one class of block by
    0.2. They were dead: the forward compared `attn_blk.type` -- which held the
    attention CLASS, because setting it also shadowed nn.Module.type() --
    against 'input'/'middle'/'output', so no branch ever matched and all four
    presets behaved exactly like 'linear'."""

    def teardown_method(self):
        clear_all_ip_adapter()

    def contributions(self, weight_type, legacy=False):
        """IP contribution per block class, relative to the plain attention."""
        clear_all_ip_adapter()
        torch.manual_seed(5)
        adapter = PlugableIPAdapter(make_sd15_adapter_state())
        unet = make_fake_unet()
        adapter.hook(
            model=unet,
            clip_vision_output={'image_embeds': torch.randn(1, CLIP_DIM)},
            weight=1.0, start=0.0, end=1.0, weight_type=weight_type,
            embeds_scaling='V only', legacy_layer_indexing=legacy)
        unet.uc_mask_shape = torch.tensor([0.0, 1.0])

        x = torch.randn(2, 8, INNER)
        ctx = torch.randn(2, 77, CROSS_DIM)
        out = {}
        for label, attn in (
                ('input', unet.input_blocks[1][1].transformer_blocks[0].attn2),
                ('middle', unet.middle_block[1].transformer_blocks[0].attn2),
                ('output', unet.output_blocks[3][1].transformer_blocks[0].attn2)):
            with torch.no_grad():
                delta = attn(x, context=ctx) - sdpa_reference(attn, x, ctx)
            out[label] = delta.abs().sum().item()
        return out

    def ratios(self, weight_type, legacy=False):
        base = self.contributions('linear', legacy=legacy)
        got = self.contributions(weight_type, legacy=legacy)
        return {k: round(got[k] / base[k], 2) for k in got}

    @pytest.mark.parametrize('weight_type,expected', [
        ('weak input', {'input': 0.2, 'middle': 1.0, 'output': 1.0}),
        ('weak middle', {'input': 1.0, 'middle': 0.2, 'output': 1.0}),
        ('weak output', {'input': 1.0, 'middle': 1.0, 'output': 0.2}),
        # Everything BUT the middle is weakened, leaving it dominant.
        ('strong middle', {'input': 0.2, 'middle': 1.0, 'output': 0.2}),
    ])
    def test_weakens_only_its_own_block_class(self, weight_type, expected):
        assert self.ratios(weight_type) == expected

    @pytest.mark.parametrize('weight_type', [
        'weak input', 'weak middle', 'weak output', 'strong middle'])
    def test_legacy_keeps_them_inert(self, weight_type):
        """Old renders were made with these doing nothing, so reproducing one
        has to keep them doing nothing."""
        assert self.ratios(weight_type, legacy=True) == {
            'input': 1.0, 'middle': 1.0, 'output': 1.0}

    def test_block_type_no_longer_shadows_module_type(self):
        """`block.type = CrossAttention` clobbered nn.Module.type(), so casting
        a hooked block raised TypeError."""
        adapter = PlugableIPAdapter(make_sd15_adapter_state())
        unet = make_fake_unet()
        adapter.hook(
            model=unet,
            clip_vision_output={'image_embeds': torch.randn(1, CLIP_DIM)},
            weight=1.0, start=0.0, end=1.0, weight_type='linear',
            embeds_scaling='V only')
        attn = unet.input_blocks[1][1].transformer_blocks[0].attn2
        assert callable(attn.type)
        assert attn.type(torch.float32) is attn      # nn.Module.type()
        assert attn.ip_block_type == 'input'
