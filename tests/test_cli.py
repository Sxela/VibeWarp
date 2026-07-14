"""Tests for VibeWarp CLI entry point."""

import json
import os
import pytest

from vibewarp.__main__ import parse_args, config_from_args, config_from_json


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.steps == 25
        assert args.cfg_scale == 7.5
        assert args.seed == -1
        assert args.width == 512
        assert args.height == 512
        assert args.sampler == 'sample_euler_ancestral'
        assert args.no_flow is False

    def test_video_and_prompt(self):
        args = parse_args(['--video', 'test.mp4', '--prompt', 'a painting'])
        assert args.video == 'test.mp4'
        assert args.prompt == 'a painting'

    def test_diffusion_args(self):
        args = parse_args(['--steps', '50', '--cfg-scale', '12.0', '--seed', '42'])
        assert args.steps == 50
        assert args.cfg_scale == 12.0
        assert args.seed == 42

    def test_output_args(self):
        args = parse_args(['--output-dir', 'out', '--batch-name', 'test_run'])
        assert args.output_dir == 'out'
        assert args.batch_name == 'test_run'

    def test_frame_range(self):
        args = parse_args(['--start-frame', '10', '--end-frame', '50'])
        assert args.start_frame == 10
        assert args.end_frame == 50

    def test_no_flow_flag(self):
        args = parse_args(['--no-flow'])
        assert args.no_flow is True

    def test_config_path(self):
        args = parse_args(['--config', 'settings.json'])
        assert args.config == 'settings.json'

    def test_print_config_flag(self):
        args = parse_args(['--print-config'])
        assert args.print_config is True


class TestConfigFromArgs:
    def test_basic_config(self):
        args = parse_args(['--video', 'test.mp4', '--prompt', 'painting',
                           '--checkpoint', 'model.safetensors'])
        config = config_from_args(args)
        assert config.video.video_init_path == 'test.mp4'
        assert config.text_prompts == {0: 'painting'}
        assert config.sd_checkpoint_path == 'model.safetensors'

    def test_diffusion_settings(self):
        args = parse_args(['--steps', '40', '--cfg-scale', '10.0', '--seed', '123',
                           '--style-strength', '0.8'])
        config = config_from_args(args)
        assert config.diffusion.steps == 40
        assert config.diffusion.cfg_scale == 10.0
        assert config.diffusion.seed == 123
        assert config.diffusion.style_strength == 0.8

    def test_video_settings(self):
        args = parse_args(['--width', '768', '--height', '512', '--nth-frame', '2'])
        config = config_from_args(args)
        assert config.video.width == 768
        assert config.video.height == 512
        assert config.video.extract_nth_frame == 2

    def test_max_size_is_deferred_to_source_aspect_resolution(self):
        config = config_from_args(parse_args(['--max-size', '768']))
        assert config.video.max_size == 768
        assert config.video.width == 512
        assert config.video.height == 512

    def test_flow_disabled(self):
        args = parse_args(['--no-flow'])
        config = config_from_args(args)
        assert config.flow.flow_warp is False

    def test_flow_blend(self):
        args = parse_args(['--flow-blend', '0.7'])
        config = config_from_args(args)
        assert config.warp.flow_blend == 0.7


class TestConfigFromJson:
    def test_basic_json(self, tmp_path):
        data = {
            'sd_checkpoint_path': '/models/sd.safetensors',
            'batch_name': 'test',
            'text_prompts': {0: 'a painting'},
            'diffusion': {'steps': 30, 'cfg_scale': 8.0},
            'video': {'video_init_path': 'input.mp4', 'width': 768},
        }
        path = str(tmp_path / 'config.json')
        with open(path, 'w') as f:
            json.dump(data, f)

        config = config_from_json(path)
        assert config.sd_checkpoint_path == '/models/sd.safetensors'
        assert config.batch_name == 'test'
        assert config.diffusion.steps == 30
        assert config.diffusion.cfg_scale == 8.0
        assert config.video.width == 768

    def test_with_flow_config(self, tmp_path):
        data = {
            'flow': {'flow_warp': False, 'check_consistency': False},
        }
        path = str(tmp_path / 'config.json')
        with open(path, 'w') as f:
            json.dump(data, f)

        config = config_from_json(path)
        assert config.flow.flow_warp is False
        assert config.flow.check_consistency is False

    def test_unknown_keys_passed_through(self, tmp_path):
        data = {
            'batch_name': 'custom',
            'output_dir': 'my_output',
        }
        path = str(tmp_path / 'config.json')
        with open(path, 'w') as f:
            json.dump(data, f)

        config = config_from_json(path)
        assert config.batch_name == 'custom'
        assert config.output_dir == 'my_output'
