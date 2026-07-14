"""Pasted paths carry quotes — Windows "Copy as path" gives `"C:\\models\\x"`."""

from dataclasses import asdict

import pytest
from fastapi.testclient import TestClient

from vibewarp.config import ControlNetEntry, IPAdapterEntry, RunConfig
from vibewarp.config_io import apply_path_defaults, sanitize_paths, strip_path_quotes
from vibewarp.web import create_app
from vibewarp.web_jobs import JobManager


class TestStripPathQuotes:
    @pytest.mark.parametrize("raw, expected", [
        ('"C:\\models\\ControlNet"', 'C:\\models\\ControlNet'),
        ("'C:\\models\\ControlNet'", 'C:\\models\\ControlNet'),
        ('  "C:\\models\\x"  ', 'C:\\models\\x'),
        ('C:\\models\\x', 'C:\\models\\x'),
        ('/home/me/models', '/home/me/models'),
        ('', ''),
        ('""', ''),
        ('"  "', ''),
    ])
    def test_strips(self, raw, expected):
        assert strip_path_quotes(raw) == expected

    def test_leaves_interior_and_unbalanced_quotes_alone(self):
        # A lone leading quote is not a wrapper; don't eat half the path.
        assert strip_path_quotes('"C:\\models') == '"C:\\models'
        assert strip_path_quotes("C:\\my's folder") == "C:\\my's folder"


class TestSanitizeConfig:
    def test_strips_across_the_whole_tree(self):
        config = RunConfig(sd_checkpoint_path='"C:\\models\\sd.safetensors"')
        config.lora_dir = "'C:\\models\\Lora'"
        config.video.video_init_path = '"C:\\videos\\in.mp4"'
        config.controlnet.model_dir = '"C:\\models\\ControlNet"'
        config.controlnet.models = {
            'control_sd15_depth': ControlNetEntry(path='"C:\\models\\d.pth"',
                                                  source='"C:\\frames\\depth"'),
        }
        config.ipadapter.models = {
            'ipadapter_sd15': IPAdapterEntry(path='"C:\\models\\ip.bin"',
                                             source_image='"C:\\img\\style.png"'),
        }
        config.animatediff.motion_module_path = '"C:\\models\\mm.ckpt"'
        config.video_assembly.upscale_model_path = '"C:\\models\\esrgan.pth"'

        sanitize_paths(config)

        assert config.sd_checkpoint_path == 'C:\\models\\sd.safetensors'
        assert config.lora_dir == 'C:\\models\\Lora'
        assert config.video.video_init_path == 'C:\\videos\\in.mp4'
        assert config.controlnet.model_dir == 'C:\\models\\ControlNet'
        # Nested dataclasses inside dicts are reached too.
        entry = config.controlnet.models['control_sd15_depth']
        assert entry.path == 'C:\\models\\d.pth'
        assert entry.source == 'C:\\frames\\depth'
        ipa = config.ipadapter.models['ipadapter_sd15']
        assert ipa.path == 'C:\\models\\ip.bin'
        assert ipa.source_image == 'C:\\img\\style.png'
        assert config.animatediff.motion_module_path == 'C:\\models\\mm.ckpt'
        assert config.video_assembly.upscale_model_path == 'C:\\models\\esrgan.pth'

    def test_leaves_non_path_strings_untouched(self):
        """Prompts and enums may legitimately contain quotes."""
        config = RunConfig()
        config.text_prompts = {0: '"a quoted phrase" in a painting'}
        config.batch_name = '"keep"'
        config.diffusion.sampler = 'sample_euler'
        sanitize_paths(config)
        assert config.text_prompts[0] == '"a quoted phrase" in a painting'
        assert config.batch_name == '"keep"'  # not a path field
        assert config.diffusion.sampler == 'sample_euler'

    def test_quote_only_value_counts_as_blank_for_defaults(self):
        config = RunConfig()
        config.root_dir = '""'
        config.output_dir = '" "'
        apply_path_defaults(config)
        assert config.root_dir  # replaced by cwd, not left as '""'
        assert config.output_dir == 'images_out'


class TestApiStripsQuotes:
    def test_submitted_config_is_sanitized(self, tmp_path):
        checkpoint = tmp_path / "model.safetensors"
        video = tmp_path / "input.mp4"
        checkpoint.write_bytes(b"m")
        video.write_bytes(b"v")

        config = RunConfig(sd_checkpoint_path=f'"{checkpoint}"')
        config.video.video_init_path = f'"{video}"'
        payload = asdict(config)

        client = TestClient(create_app(JobManager(runner=lambda *_a, **_k: [])))
        response = client.post("/api/config/validate", json=payload)
        # Without stripping, check_paths validation would fail: the quoted path
        # does not exist on disk.
        assert response.status_code == 200, response.json()
        body = response.json()["config"]
        assert body["sd_checkpoint_path"] == str(checkpoint)
        assert body["video"]["video_init_path"] == str(video)
