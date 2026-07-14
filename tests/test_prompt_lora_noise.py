"""Tests for prompt parsing, LORA schedule extraction, and fixed_code noise path."""

import pytest
from vibewarp.core.prompt import (
    parse_prompt,
    split_weighted_prompts,
    split_lora_from_prompts,
    get_prompt_and_loras_for_frame,
)


# ---- parse_prompt ----

class TestParsePrompt:
    def test_no_lora(self):
        clean, loras = parse_prompt("a painting in watercolor style")
        assert clean == "a painting in watercolor style"
        assert loras == {}

    def test_single_lora(self):
        clean, loras = parse_prompt("a painting <lora:my_style:0.8>")
        assert clean == "a painting"
        assert loras == {"my_style": 0.8}

    def test_lora_no_weight(self):
        clean, loras = parse_prompt("portrait <lora:face_detail>")
        assert clean == "portrait"
        assert loras == {"face_detail": 1.0}

    def test_multiple_loras(self):
        clean, loras = parse_prompt("<lora:style_a:0.5> sunset <lora:detail:0.3>")
        assert clean == "sunset"
        assert set(loras.keys()) == {"style_a", "detail"}
        assert abs(loras["style_a"] - 0.5) < 1e-6
        assert abs(loras["detail"] - 0.3) < 1e-6

    def test_lora_case_insensitive(self):
        clean, loras = parse_prompt("<LORA:MyStyle:0.7>")
        assert "MyStyle" in loras
        assert abs(loras["MyStyle"] - 0.7) < 1e-6

    def test_empty_prompt(self):
        clean, loras = parse_prompt("")
        assert clean == ""
        assert loras == {}


# ---- split_weighted_prompts ----

class TestSplitWeightedPrompts:
    def test_single_no_weight(self):
        prompts, weights = split_weighted_prompts("a watercolor painting")
        assert prompts == ["a watercolor painting"]
        assert weights == [1.0]

    def test_single_with_weight(self):
        prompts, weights = split_weighted_prompts("a painting:0.7")
        assert prompts == ["a painting"]
        assert abs(weights[0] - 0.7) < 1e-6

    def test_multi_prompt(self):
        prompts, weights = split_weighted_prompts("a painting:0.7 | anime style:0.3")
        assert len(prompts) == 2
        assert "a painting" in prompts
        assert "anime style" in prompts
        idx = prompts.index("a painting")
        assert abs(weights[idx] - 0.7) < 1e-6

    def test_multi_prompt_no_weights(self):
        prompts, weights = split_weighted_prompts("prompt A | prompt B")
        assert len(prompts) == 2
        assert all(w == 1.0 for w in weights)

    def test_empty_parts_ignored(self):
        prompts, weights = split_weighted_prompts("a | | b")
        assert len(prompts) == 2
        assert "a" in prompts and "b" in prompts

    def test_empty_string(self):
        prompts, weights = split_weighted_prompts("")
        assert prompts == ['']
        assert weights == [1.0]


# ---- split_lora_from_prompts ----

class TestSplitLoraFromPrompts:
    def test_no_loras(self):
        prompts = {0: "a painting"}
        clean, schedule = split_lora_from_prompts(prompts)
        assert clean == {0: "a painting"}
        assert schedule == {}

    def test_single_lora_single_keyframe(self):
        prompts = {0: "a painting <lora:my_style:0.8>"}
        clean, schedule = split_lora_from_prompts(prompts)
        assert clean == {0: "a painting"}
        assert "my_style" in schedule
        assert schedule["my_style"][0] == pytest.approx(0.8)

    def test_lora_across_keyframes(self):
        prompts = {
            0: "painting <lora:style_a:0.5>",
            30: "portrait <lora:style_a:0.8>",
        }
        clean, schedule = split_lora_from_prompts(prompts)
        assert clean[0] == "painting"
        assert clean[30] == "portrait"
        assert schedule["style_a"][0] == pytest.approx(0.5)
        assert schedule["style_a"][30] == pytest.approx(0.8)

    def test_different_loras_per_keyframe(self):
        prompts = {
            0: "painting <lora:style_a:0.5>",
            30: "portrait <lora:style_b:0.9>",
        }
        _, schedule = split_lora_from_prompts(prompts)
        assert "style_a" in schedule and "style_b" in schedule
        assert 0 in schedule["style_a"]
        assert 30 in schedule["style_b"]


# ---- get_prompt_and_loras_for_frame ----

class TestGetPromptAndLorasForFrame:
    def test_before_first_keyframe(self):
        clean_prompts = {0: "painting"}
        lora_sched = {"style": {0: 0.8}}
        prompt, names, weights = get_prompt_and_loras_for_frame(0, clean_prompts, lora_sched)
        assert prompt == "painting"
        assert "style" in names
        assert abs(weights[names.index("style")] - 0.8) < 1e-6

    def test_keyframe_lookup(self):
        clean_prompts = {0: "painting", 20: "portrait"}
        lora_sched = {}
        prompt, _, _ = get_prompt_and_loras_for_frame(25, clean_prompts, lora_sched)
        assert prompt == "portrait"

    def test_zero_weight_excluded(self):
        clean_prompts = {0: "painting"}
        lora_sched = {"style": {0: 0.0}}
        _, names, weights = get_prompt_and_loras_for_frame(0, clean_prompts, lora_sched)
        assert "style" not in names

    def test_no_loras(self):
        clean_prompts = {0: "painting"}
        _, names, weights = get_prompt_and_loras_for_frame(0, clean_prompts, {})
        assert names == []
        assert weights == []


# ---- noise_mode='fixed' in config ----

class TestNoiseModeConfig:
    def test_default_noise_mode(self):
        from vibewarp.config import DiffusionConfig
        d = DiffusionConfig()
        assert d.noise_mode == 'default'

    def test_fixed_noise_mode(self):
        from vibewarp.config import DiffusionConfig
        d = DiffusionConfig(noise_mode='fixed')
        assert d.noise_mode == 'fixed'

    def test_noise_mode_from_settings(self, tmp_path):
        """noise_mode='fixed' in settings file maps to diffusion.noise_mode."""
        import json
        from vibewarp.settings import load_warpfusion_settings

        path = str(tmp_path / "s.txt")
        with open(path, 'w') as f:
            json.dump({
                "model_path": "model.safetensors",
                "noise_mode": "fixed",
                "code_randomness": 0.3,
                "text_prompts": {"0": ["a painting"]},
                "negative_prompts": {"0": [""]},
            }, f)
        cfg = load_warpfusion_settings(path)
        assert cfg['diffusion']['noise_mode'] == 'fixed'
        assert abs(cfg['diffusion']['code_randomness'] - 0.3) < 1e-6

    def test_reconstructed_noise_mode_maps_to_reconstruction_noise(self, tmp_path):
        """noise_mode='reconstructed' should still enable reconstruction_noise."""
        import json
        from vibewarp.settings import load_warpfusion_settings

        path = str(tmp_path / "s.txt")
        with open(path, 'w') as f:
            json.dump({
                "model_path": "model.safetensors",
                "noise_mode": "reconstructed",
                "text_prompts": {"0": ["a painting"]},
                "negative_prompts": {"0": [""]},
            }, f)
        cfg = load_warpfusion_settings(path)
        assert cfg['reconstruction_noise']['enabled'] is True
