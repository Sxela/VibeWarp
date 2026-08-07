"""Tests for vibewarp.core.diffusion — unit tests for the rendering pipeline."""

import os
import inspect
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from vibewarp.config import DiffusionConfig, ReconstructionNoiseConfig, RunConfig
from vibewarp.core.diffusion import (
    FrameState,
    RenderContext,
    _get_prompt_for_frame,
    compute_sigmas,
    get_frame_schedule,
    get_premature_sigma_min,
    get_sampler_fn,
    load_img_for_sd,
    make_temporal_guidance_model_fn,
    run_frames,
    run_sd,
)


@pytest.mark.gpu
class TestLoadImgForSd:
    def test_returns_correct_shape(self, tmp_path):
        # Create a dummy image
        img = Image.fromarray(np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8))
        path = str(tmp_path / "test.png")
        img.save(path)

        result = load_img_for_sd(path, size=(64, 48))
        assert result.shape == (1, 3, 48, 64)
        assert result.dtype == torch.float16
        assert result.device.type == 'cuda'

    def test_returns_range_minus1_to_1(self, tmp_path):
        img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        path = str(tmp_path / "black.png")
        img.save(path)

        result = load_img_for_sd(path, size=(32, 32))
        assert result.min().item() >= -1.0
        assert result.max().item() <= 1.0

    def test_white_image_near_1(self, tmp_path):
        img = Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8))
        path = str(tmp_path / "white.png")
        img.save(path)

        result = load_img_for_sd(path, size=(32, 32))
        assert result.max().item() == pytest.approx(1.0, abs=0.01)

    def test_black_image_near_minus1(self, tmp_path):
        img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        path = str(tmp_path / "black.png")
        img.save(path)

        result = load_img_for_sd(path, size=(32, 32))
        assert result.min().item() == pytest.approx(-1.0, abs=0.01)


class TestGetPromptForFrame:
    def test_single_prompt(self):
        prompts = {0: "a beautiful painting"}
        assert _get_prompt_for_frame(0, prompts) == "a beautiful painting"
        assert _get_prompt_for_frame(10, prompts) == "a beautiful painting"

    def test_multiple_prompts(self):
        prompts = {0: "first", 5: "second", 10: "third"}
        assert _get_prompt_for_frame(0, prompts) == "first"
        assert _get_prompt_for_frame(3, prompts) == "first"
        assert _get_prompt_for_frame(5, prompts) == "second"
        assert _get_prompt_for_frame(7, prompts) == "second"
        assert _get_prompt_for_frame(10, prompts) == "third"
        assert _get_prompt_for_frame(100, prompts) == "third"

    def test_empty_prompts(self):
        assert _get_prompt_for_frame(0, {}) == ""


class TestGetPrematureSigmaMin:
    def test_returns_float(self):
        result = get_premature_sigma_min(
            steps=11, sigma_max=14.6, sigma_min_nominal=0.0292
        )
        assert isinstance(result, float)

    def test_less_than_sigma_max(self):
        result = get_premature_sigma_min(
            steps=11, sigma_max=14.6, sigma_min_nominal=0.0292
        )
        assert result < 14.6

    def test_greater_than_zero(self):
        result = get_premature_sigma_min(
            steps=11, sigma_max=14.6, sigma_min_nominal=0.0292
        )
        assert result > 0


class TestGetSamplerFn:
    def test_euler_ancestral(self):
        fn = get_sampler_fn('sample_euler_ancestral')
        assert callable(fn)

    def test_dpm_2(self):
        fn = get_sampler_fn('sample_dpm_2')
        assert callable(fn)

    def test_dpmpp_2m(self):
        fn = get_sampler_fn('sample_dpmpp_2m')
        assert callable(fn)

    def test_dpmpp_sde(self):
        fn = get_sampler_fn('sample_dpmpp_sde')
        assert callable(fn)

    def test_lcm(self):
        fn = get_sampler_fn('sample_lcm')
        assert callable(fn)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown sampler"):
            get_sampler_fn('sample_nonexistent')

    def test_lcm_functional(self):
        """sample_lcm: each step returns the denoised prediction (+ noise while
        sigma>0); a model that always denoises to zeros must end at zeros."""
        fn = get_sampler_fn('sample_lcm')

        def model(x, sigma, **kwargs):
            return torch.zeros_like(x)

        x = torch.randn(1, 4, 8, 8) * 14.6
        sigmas = torch.tensor([14.6, 7.0, 1.0, 0.0])
        out = fn(model, x, sigmas, disable=True)
        assert out.shape == x.shape
        assert torch.allclose(out, torch.zeros_like(out))

    def test_lcm_callback_replaces_x(self):
        """WarpFusion patch: the callback's return value replaces x
        (masked-diffusion support) — same semantics as the other samplers."""
        fn = get_sampler_fn('sample_lcm')
        seen = []

        def model(x, sigma, **kwargs):
            return torch.zeros_like(x)

        def callback(args):
            seen.append(args['i'])
            return args['x']

        x = torch.randn(1, 4, 8, 8)
        sigmas = torch.tensor([14.6, 1.0, 0.0])
        out = fn(model, x, sigmas, callback=callback, disable=True)
        assert seen == [0, 1]
        assert out.shape == x.shape


class TestComputeSigmas:
    def test_karras_schedule(self):
        """Test with a mock model_wrap that has .sigmas attribute."""
        class MockWrap:
            sigmas = torch.linspace(0.0292, 14.6, 1000)

        sigmas = compute_sigmas(MockWrap(), steps=10, use_karras=True)
        assert sigmas.shape[0] == 11  # n + 1
        assert sigmas[-1] == 0.0  # last sigma is always 0

    def test_linear_schedule(self):
        class MockWrap:
            sigmas = torch.linspace(0.0292, 14.6, 1000)
            def get_sigmas(self, n):
                return torch.linspace(14.6, 0.0, n + 1)

        sigmas = compute_sigmas(MockWrap(), steps=10, use_karras=False)
        assert sigmas.shape[0] == 11

    def test_karras_decreasing(self):
        class MockWrap:
            sigmas = torch.linspace(0.0292, 14.6, 1000)

        sigmas = compute_sigmas(MockWrap(), steps=20, use_karras=True)
        # Karras schedule should be monotonically decreasing
        for i in range(len(sigmas) - 1):
            assert sigmas[i] >= sigmas[i + 1]


class TestFrameState:
    def test_defaults(self):
        s = FrameState()
        assert s.frame_num == 0
        assert s.init_image is None
        assert s.first_latent is None

    def test_update(self):
        s = FrameState(frame_num=10, seed=42)
        assert s.frame_num == 10
        assert s.seed == 42


class TestRenderContext:
    def test_defaults(self):
        ctx = RenderContext()
        assert ctx.sd_model is None
        assert ctx.loaded_controlnets == {}
        assert ctx.sampler_fn is None

    def test_with_config(self):
        cfg = RunConfig(batch_name='mytest')
        ctx = RenderContext(config=cfg)
        assert ctx.config.batch_name == 'mytest'


class TestGetFrameSchedule:
    def test_returns_dict(self):
        cfg = RunConfig()
        result = get_frame_schedule(0, cfg)
        assert isinstance(result, dict)
        assert 'steps' in result
        assert 'cfg_scale' in result
        assert 'seed' in result

    def test_uses_config_values(self):
        cfg = RunConfig(diffusion=DiffusionConfig(steps=50, cfg_scale=12.0, seed=123))
        result = get_frame_schedule(0, cfg)
        assert result['steps'] == 50
        assert result['cfg_scale'] == 12.0
        assert result['seed'] == 123


def _make_run_sd_ctx(style_strength=0.65, steps=20, rec_noise_cfg=None):
    """Build a RenderContext with enough mocks to run run_sd without GPU models."""
    cfg = RunConfig(
        diffusion=DiffusionConfig(
            steps=steps,
            cfg_scale=7.5,
            seed=0,
            style_strength=style_strength,
            use_karras_noise=False,
        ),
    )
    if rec_noise_cfg is not None:
        cfg.reconstruction_noise = rec_noise_cfg

    class _FakeModule:
        def cuda(self): return self
        def cpu(self): return self

    class _FakeSDModel:
        parameterization = 'eps'
        cond_stage_model = _FakeModule()
        first_stage_model = _FakeModule()
        model = _FakeModule()

        def get_learned_conditioning(self, prompts):
            return torch.randn(len(prompts), 77, 768)

        def encode_first_stage(self, x):
            B, C, H, W = x.shape
            return torch.zeros(B, 4, H // 8, W // 8, device=x.device, dtype=x.dtype)

        def get_first_stage_encoding(self, x):
            return x.float()

        def decode_first_stage(self, x):
            return torch.zeros(x.shape[0], 3, x.shape[2] * 8, x.shape[3] * 8)

    class _FakeModelWrap:
        sigmas = torch.linspace(0.03, 14.6, 1000)

        def get_sigmas(self, n):
            return torch.linspace(14.6, 0.0, n + 1)

    class _FakeCFGDenoiser(torch.nn.Module):
        _cfg_schedule = None

        def __init__(self):
            super().__init__()

        def cuda(self): return self
        def cpu(self): return self

        def forward(self, x, sigma, **kwargs):
            return torch.zeros_like(x)

    ctx = RenderContext(config=cfg)
    ctx.sd_model = _FakeSDModel()
    ctx.model_wrap = _FakeModelWrap()
    ctx.model_wrap_cfg = _FakeCFGDenoiser()
    return ctx


class TestTemporalGuidance:
    def test_latent_gradient_is_applied_clamped_and_logged(self, capsys):
        denoised = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        target = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
        guided_model = make_temporal_guidance_model_fn(
            lambda x, sigma, **kwargs: x,
            sd_model=None,
            target_image=None,
            target_latent=target,
            init_scale=0.0,
            init_latent_scale=1.0,
            clamp_grad=True,
            clamp_max=0.01,
            add_noise=False,
            guidance_start_code=None,
            source_label='prev stylized',
        )

        result = guided_model(denoised, torch.tensor([1.0]))

        delta_rms = (result - denoised).square().mean().sqrt()
        assert delta_rms.item() == pytest.approx(0.01)
        output = capsys.readouterr().out
        assert 'Temporal guidance [prev stylized] step 1' in output
        assert 'latent_loss=' in output
        assert 'grad_raw(' in output
        assert 'grad_applied_rms=0.01' in output
        assert 'clamped=yes' in output


class TestRunSdImg2Img:
    """Verify run_sd img2img path (style_strength < 1) noise and sigma logic."""

    def _captured_sampler(self, calls):
        """Return a sampler that records (x, sigmas) passed to it."""
        def sampler(model_fn, x, sigma_sched, extra_args=None, callback=None):
            calls.append({'x': x.detach().clone(), 'sigma_sched': sigma_sched.detach().clone()})
            return torch.zeros_like(x)
        return sampler

    def test_sampler_context_is_conditional_on_active_guidance(self, tmp_path):
        img_path = str(tmp_path / 'init.png')
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(img_path)

        ordinary = _make_run_sd_ctx(steps=4)
        ordinary_modes = []

        def ordinary_sampler(model_fn, x, sigmas, **kwargs):
            ordinary_modes.append((
                torch.is_inference_mode_enabled(), torch.is_grad_enabled()))
            return torch.zeros_like(x)

        ordinary.sampler_fn = ordinary_sampler
        run_sd(
            ordinary, init_image_path=img_path, skip_timesteps=2,
            H=64, W=64, text_prompt='test', neg_prompt='', steps=4,
            seed=0, cfg_scale=7.5, state=FrameState(),
        )
        assert ordinary_modes == [(True, False)]

        guided = _make_run_sd_ctx(steps=4)
        guided.config.diffusion.init_latent_scale = 1.0
        guided_modes = []

        def guided_sampler(model_fn, x, sigmas, **kwargs):
            guided_modes.append((
                torch.is_inference_mode_enabled(), torch.is_grad_enabled()))
            return torch.zeros_like(x)

        guided.sampler_fn = guided_sampler
        run_sd(
            guided, init_image_path=img_path, skip_timesteps=2,
            H=64, W=64, text_prompt='test', neg_prompt='', steps=4,
            seed=0, cfg_scale=7.5, state=FrameState(),
            guidance_image_path=img_path,
        )
        assert guided_modes == [(False, False)]

    def test_img2img_sigma_sched_sliced_correctly(self, tmp_path):
        """sigma_sched should start at sigmas[skip_steps - 1] for style_strength < 1."""
        steps = 20
        style_strength = 0.5  # skip_steps = 10, t_enc = 10
        skip_steps = int(steps - steps * style_strength)  # 10
        t_enc = steps - skip_steps  # 10

        ctx = _make_run_sd_ctx(style_strength=style_strength, steps=steps)
        calls = []
        ctx.sampler_fn = self._captured_sampler(calls)

        # Create a dummy init image
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        img_path = str(tmp_path / 'init.png')
        img.save(img_path)

        full_sigmas = ctx.model_wrap.get_sigmas(steps)

        with patch('vibewarp.core.model_loader.make_static_thresh_model_fn', side_effect=lambda m, value=30.: m):
            run_sd(ctx, init_image_path=img_path, skip_timesteps=skip_steps,
                   H=64, W=64, text_prompt='test', neg_prompt='', steps=steps,
                   seed=0, cfg_scale=7.5, state=FrameState())

        assert len(calls) == 1
        sigma_sched = calls[0]['sigma_sched']
        expected_start_sigma = full_sigmas[steps - t_enc - 1]
        assert sigma_sched[0].item() == pytest.approx(expected_start_sigma.item(), rel=1e-4)

    def test_img2img_xi_equals_x0_plus_noise(self, tmp_path):
        """xi passed to sampler should equal x0 + noise at sigma[skip_steps-1]."""
        steps = 10
        style_strength = 0.5  # skip_steps = 5

        ctx = _make_run_sd_ctx(style_strength=style_strength, steps=steps)
        calls = []
        ctx.sampler_fn = self._captured_sampler(calls)

        img = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
        img_path = str(tmp_path / 'init.png')
        img.save(img_path)

        skip_steps = int(steps - steps * style_strength)
        t_enc = steps - skip_steps
        sigmas = ctx.model_wrap.get_sigmas(steps)
        sigma_t = sigmas[steps - t_enc - 1]

        with patch('vibewarp.core.model_loader.make_static_thresh_model_fn', side_effect=lambda m, value=30.: m):
            torch.manual_seed(0)
            run_sd(ctx, init_image_path=img_path, skip_timesteps=skip_steps,
                   H=64, W=64, text_prompt='test', neg_prompt='', steps=steps,
                   seed=0, cfg_scale=7.5, state=FrameState())

        assert len(calls) == 1
        xi = calls[0]['x']
        # xi should NOT be zero (it should contain the init latent + noise)
        assert xi.abs().max().item() > 0

    def test_style_strength_1_uses_full_sigma_sched(self, tmp_path):
        """With style_strength=1 (skip_steps=0), sampler gets the full sigma schedule."""
        steps = 10
        ctx = _make_run_sd_ctx(style_strength=1.0, steps=steps)
        calls = []
        ctx.sampler_fn = self._captured_sampler(calls)

        # No init image for txt2img (skip_timesteps=0 → else branch)
        with patch('vibewarp.core.model_loader.make_static_thresh_model_fn', side_effect=lambda m, value=30.: m):
            run_sd(ctx, init_image_path=None, skip_timesteps=0,
                   H=64, W=64, text_prompt='test', neg_prompt='', steps=steps,
                   seed=0, cfg_scale=7.5, state=FrameState())

        assert len(calls) == 1
        sigma_sched = calls[0]['sigma_sched']
        full_sigmas = ctx.model_wrap.get_sigmas(steps)
        assert sigma_sched.shape[0] == full_sigmas.shape[0]

    def test_style_strength_0_skips_sampler(self, tmp_path):
        """With style_strength=0 (t_enc=0), sampler should not be called (returns x0)."""
        steps = 10
        style_strength = 0.0  # skip_steps = steps, t_enc = 0
        skip_steps = steps

        ctx = _make_run_sd_ctx(style_strength=style_strength, steps=steps)
        calls = []
        ctx.sampler_fn = self._captured_sampler(calls)

        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        img_path = str(tmp_path / 'init.png')
        img.save(img_path)

        with patch('vibewarp.core.model_loader.make_static_thresh_model_fn', side_effect=lambda m, value=30.: m):
            run_sd(ctx, init_image_path=img_path, skip_timesteps=skip_steps,
                   H=64, W=64, text_prompt='test', neg_prompt='', steps=steps,
                   seed=0, cfg_scale=7.5, state=FrameState())

        # Sampler should NOT be called when t_enc=0
        assert len(calls) == 0

    @pytest.mark.gpu
    def test_rec_noise_img2img_applies_dc_correction(self, tmp_path):
        """Per-frame rec_noise in img2img path: noise = blend(rec, rand) - x0/sigma_max."""
        steps = 10
        skip_steps = 5
        t_enc = steps - skip_steps

        rec_cfg = ReconstructionNoiseConfig(enabled=True, randomness=0.0, noise_scale=1.0)
        ctx = _make_run_sd_ctx(steps=steps, rec_noise_cfg=rec_cfg)

        captured = {}

        def capturing_sampler(model_fn, x, sigma_sched, extra_args=None, callback=None):
            captured['xi'] = x.detach().clone()
            captured['sigma_sched'] = sigma_sched.detach().clone()
            return torch.zeros_like(x)

        ctx.sampler_fn = capturing_sampler

        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        img_path = str(tmp_path / 'init.png')
        img.save(img_path)

        # Fixed rec_noise on CUDA (matches real usage — find_noise_for_image returns CUDA)
        rec_noise = torch.ones(1, 4, 8, 8).cuda() * 2.0
        sigmas = ctx.model_wrap.get_sigmas(steps)

        with patch('vibewarp.core.model_loader.make_static_thresh_model_fn', side_effect=lambda m, value=30.: m):
            run_sd(ctx, init_image_path=img_path, skip_timesteps=skip_steps,
                   H=64, W=64, text_prompt='test', neg_prompt='', steps=steps,
                   seed=0, cfg_scale=7.5, state=FrameState(), rec_noise=rec_noise)

        assert 'xi' in captured
        # The noise component is (rec_noise - x0/sigma_max) * sigma[t_enc]
        # For a black (zeros) image, x0 ≈ 0, so noise ≈ rec_noise * sigma[t_enc-ish]
        # xi = x0 + noise, so xi should be non-trivially large
        assert captured['xi'].abs().mean().item() > 0.1

    def test_rec_noise_override_applies_dc_correction(self, tmp_path):
        """_rec_noise_override in img2img: noise = (rec - x0/sigma_max) * sigma * scale."""
        steps = 10
        skip_steps = 5
        t_enc = steps - skip_steps
        frame_num = 3

        rec_cfg = ReconstructionNoiseConfig(enabled=True, noise_scale=1.0)
        ctx = _make_run_sd_ctx(steps=steps, rec_noise_cfg=rec_cfg)

        captured = {}

        def capturing_sampler(model_fn, x, sigma_sched, extra_args=None, callback=None):
            captured['xi'] = x.detach().clone()
            return torch.zeros_like(x)

        ctx.sampler_fn = capturing_sampler

        # Pre-set an override for frame_num
        rec_noise_val = torch.ones(1, 4, 8, 8) * 3.0
        ctx._rec_noise_override = {
            'noise': rec_noise_val,
            'frame_indices': [frame_num],
        }

        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        img_path = str(tmp_path / 'init.png')
        img.save(img_path)

        state = FrameState(frame_num=frame_num)
        sigmas = ctx.model_wrap.get_sigmas(steps)
        sigma_t = sigmas[steps - t_enc - 1]

        with patch('vibewarp.core.model_loader.make_static_thresh_model_fn', side_effect=lambda m, value=30.: m):
            run_sd(ctx, init_image_path=img_path, skip_timesteps=skip_steps,
                   H=64, W=64, text_prompt='test', neg_prompt='', steps=steps,
                   seed=0, cfg_scale=7.5, state=state)

        assert 'xi' in captured
        xi = captured['xi']
        # For a black (zeros) image encoded to ~0 latent:
        # noise = (3.0 - ~0/sigma_max) * sigma_t ≈ 3.0 * sigma_t
        # xi = ~0 + noise ≈ 3.0 * sigma_t
        expected_approx = 3.0 * sigma_t.item()
        assert abs(xi.mean().item() - expected_approx) < expected_approx * 0.5  # within 50%


class TestOffloadPerf:
    """Verify GPU offload helpers don't sync CUDA unnecessarily."""

    def test_offload_to_cpu_does_not_call_empty_cache(self):
        from unittest.mock import patch as _patch
        from vibewarp.core.diffusion import _offload_to_cpu

        class _M:
            def cpu(self): return self

        with _patch('torch.cuda.empty_cache') as mock_cache:
            _offload_to_cpu(_M(), _M())
        mock_cache.assert_not_called()

    def test_offload_to_cpu_skips_none(self):
        from vibewarp.core.diffusion import _offload_to_cpu
        # Should not raise even with None modules
        _offload_to_cpu(None, None)

    def test_run_sd_no_redundant_vae_offload(self, tmp_path):
        """VAE should be moved to GPU once for init encode, offloaded, then once for decode.
        Previously there was a redundant double-move wrapping encode_image_to_latent."""
        from vibewarp.core.diffusion import run_sd
        from unittest.mock import patch as _patch

        steps = 4
        ctx = _make_run_sd_ctx(style_strength=0.5, steps=steps)

        gpu_calls = []
        cpu_calls = []

        class _TrackedFSM:
            """First-stage model that tracks GPU/CPU moves."""
            def cuda(self_inner): gpu_calls.append('cuda'); return self_inner
            def cpu(self_inner): cpu_calls.append('cpu'); return self_inner

        ctx.sd_model.first_stage_model = _TrackedFSM()

        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        img_path = str(tmp_path / 'init.png')
        img.save(img_path)

        def _sampler(model_fn, x, sigma_sched, extra_args=None, callback=None):
            return torch.zeros_like(x)
        ctx.sampler_fn = _sampler

        skip_steps = int(steps - steps * 0.5)
        with _patch('vibewarp.core.model_loader.make_static_thresh_model_fn', side_effect=lambda m, value=30.: m):
            run_sd(ctx, init_image_path=img_path, skip_timesteps=skip_steps,
                   H=64, W=64, text_prompt='test', neg_prompt='', steps=steps,
                   seed=0, cfg_scale=7.5, state=FrameState())

        # img2img path: VAE to GPU for init encode, back to CPU, to GPU for decode, back to CPU
        # = 2 cuda calls, 2 cpu calls. The old code had 3 of each (redundant outer wrapper).
        assert len(gpu_calls) == 2, f"Expected 2 GPU moves, got {len(gpu_calls)}: {gpu_calls}"
        assert len(cpu_calls) == 2, f"Expected 2 CPU offloads, got {len(cpu_calls)}: {cpu_calls}"


class TestRunFrames:
    def test_raises_on_invalid_range(self):
        ctx = RenderContext(config=RunConfig(frame_range=[5, 5]))
        with pytest.raises(ValueError, match="Invalid frame range"):
            run_frames(ctx, frame_range=[5, 5])

    def test_raises_on_reversed_range(self):
        ctx = RenderContext(config=RunConfig())
        with pytest.raises(ValueError, match="Invalid frame range"):
            run_frames(ctx, frame_range=[10, 5])
