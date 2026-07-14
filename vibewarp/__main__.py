"""CLI entry point for VibeWarp.

Usage:
    python -m vibewarp --config settings.json
    python -m vibewarp --video input.mp4 --prompt "a watercolor painting" --steps 30
    python -m vibewarp --help
"""

import argparse
import json
import os
import sys
from dataclasses import asdict

from vibewarp.config import (
    RunConfig,
    DiffusionConfig,
    VideoConfig,
    FlowConfig,
    WarpConfig,
    BrightnessConfig,
    ColorConfig,
    ControlNetConfig,
    ControlNetEntry,
    AnimateDiffConfig,
    IPAdapterConfig,
    IPAdapterEntry,
    ReconstructionNoiseConfig,
    VaeConfig,
    BackgroundMaskConfig,
    CaptionConfig,
    SceneConfig,
    FreeUConfig,
    VideoAssemblyConfig,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog='vibewarp',
        description='VibeWarp — video-to-video style transfer pipeline',
    )

    # Config file (overrides all other args)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON config file (structured RunConfig format)')
    parser.add_argument('--settings', type=str, default=None,
                        help='Path to VibeWarp saved settings .txt file')
    parser.add_argument('--warpfusion-settings', type=str, default=None,
                        help='Path to WarpFusion notebook settings .txt (auto-detects format)')
    parser.add_argument('--models-root', type=str, default=None,
                        help='Root directory for model files (default: ./models under the current directory)')

    # Basic settings
    parser.add_argument('--video', type=str, default='',
                        help='Path to input video')
    parser.add_argument('--prompt', type=str, default='a beautiful painting',
                        help='Text prompt for style transfer')
    parser.add_argument('--negative-prompt', type=str, default='',
                        help='Negative prompt')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to SD checkpoint (.safetensors or .ckpt)')
    parser.add_argument('--model-version', type=str, default='control_multi_v15',
                        help='Model version (control_multi_v15, v1_5, v2_1)')

    # Output
    parser.add_argument('--output-dir', type=str, default='images_out',
                        help='Output directory')
    parser.add_argument('--batch-name', type=str, default='warpfusion',
                        help='Batch name for output files')

    # Frame range
    parser.add_argument('--start-frame', type=int, default=0,
                        help='Start frame number')
    parser.add_argument('--end-frame', type=int, default=0,
                        help='End frame number (0 = auto-detect)')
    parser.add_argument('--resume-from', type=int, default=0,
                        help='Resume from frame number')

    # Diffusion
    parser.add_argument('--steps', type=int, default=25,
                        help='Number of diffusion steps')
    parser.add_argument('--cfg-scale', type=float, default=7.5,
                        help='CFG scale')
    parser.add_argument('--seed', type=int, default=-1,
                        help='Random seed (-1 = random)')
    parser.add_argument('--style-strength', type=float, default=0.65,
                        help='Style strength (0=keep init, 1=full regeneration)')
    parser.add_argument('--sampler', type=str, default='sample_euler_ancestral',
                        help='Sampler function name')

    # Video
    parser.add_argument('--width', type=int, default=512, help='Output width')
    parser.add_argument('--height', type=int, default=512, help='Output height')
    parser.add_argument('--max-size', type=int, default=0,
                        help='Max dimension (scales proportionally, rounded to 64)')
    parser.add_argument('--nth-frame', type=int, default=1,
                        help='Extract every Nth frame from video')

    # VAE
    parser.add_argument('--no-tiled-vae', action='store_true',
                        help='Disable tiled VAE even if enabled in settings')

    # Flow
    parser.add_argument('--no-flow', action='store_true',
                        help='Disable optical flow warping')
    parser.add_argument('--flow-blend', type=float, default=0.5,
                        help='Flow blend ratio')
    parser.add_argument('--flow-maxsize', type=int, default=-1,
                        help='Max flow computation resolution (0=full, default=from config)')

    # Model download
    parser.add_argument('--download-models', action='store_true',
                        help='Download checkpoint and ControlNets from HuggingFace, then exit')
    parser.add_argument('--download-cn-only', action='store_true',
                        help='Download ControlNets only (use with --download-models)')
    parser.add_argument('--download-ckpt-only', action='store_true',
                        help='Download checkpoint only (use with --download-models)')

    # Print config
    parser.add_argument('--print-config', action='store_true',
                        help='Print the resolved config as JSON and exit')

    return parser.parse_args(argv)


from vibewarp.config_io import config_from_json


def config_from_args(args) -> RunConfig:
    """Build a RunConfig from CLI arguments."""
    return RunConfig(
        sd_checkpoint_path=args.checkpoint,
        model_version=args.model_version,
        output_dir=args.output_dir,
        batch_name=args.batch_name,
        text_prompts={0: args.prompt},
        negative_prompts={0: args.negative_prompt},
        frame_range=[args.start_frame, args.end_frame],
        diffusion=DiffusionConfig(
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
            style_strength=args.style_strength,
            sampler=args.sampler,
        ),
        video=VideoConfig(
            video_init_path=args.video,
            width=args.width,
            height=args.height,
            max_size=args.max_size,
            extract_nth_frame=args.nth_frame,
        ),
        flow=FlowConfig(
            flow_warp=not args.no_flow,
        ),
        warp=WarpConfig(
            flow_blend=args.flow_blend,
        ),
    )


def main(argv=None):
    # Force unbuffered output so progress is visible
    import functools
    print = functools.partial(__builtins__.__dict__['print'] if isinstance(__builtins__, dict) else getattr(__builtins__, 'print'), flush=True)

    args = parse_args(argv)

    # Model download mode — resolve models_root, then download and exit
    if args.download_models or args.download_cn_only or args.download_ckpt_only:
        from vibewarp.core.download import download_models, DEFAULT_CKPT_SAVE_AS
        # Default to a ./models folder under the current working directory —
        # no hardcoded absolute path.
        models_root = args.models_root or os.path.join(os.getcwd(), 'models')
        download_models(
            models_root=models_root,
            settings_path=args.warpfusion_settings,
            ckpt_only=args.download_ckpt_only,
            cn_only=args.download_cn_only,
        )
        return

    # Build config
    if args.settings:
        from vibewarp.settings import load_vibewarp_settings
        wf_data = load_vibewarp_settings(args.settings)
        if args.video:
            wf_data.setdefault('video', {})['video_init_path'] = args.video
        if args.checkpoint:
            wf_data['sd_checkpoint_path'] = args.checkpoint
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(wf_data, tmp, indent=2)
        tmp.close()
        config = config_from_json(tmp.name)
        os.unlink(tmp.name)
    elif args.warpfusion_settings:
        from vibewarp.settings import load_warpfusion_settings, load_vibewarp_settings, load_settings, is_vibewarp_settings
        raw = load_settings(args.warpfusion_settings)
        if is_vibewarp_settings(raw):
            wf_data = load_vibewarp_settings(args.warpfusion_settings)
        else:
            wf_data = load_warpfusion_settings(args.warpfusion_settings,
                                                models_root=args.models_root)
        # Override video/checkpoint from CLI if provided
        if args.video:
            wf_data.setdefault('video', {})['video_init_path'] = args.video
        if args.checkpoint:
            wf_data['sd_checkpoint_path'] = args.checkpoint
        # Write to temp JSON and load via config_from_json
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(wf_data, tmp, indent=2)
        tmp.close()
        config = config_from_json(tmp.name)
        os.unlink(tmp.name)
    elif args.config:
        config = config_from_json(args.config)
    else:
        config = config_from_args(args)

    # Apply CLI overrides for frame range
    if args.end_frame > 0:
        config.frame_range = [args.start_frame, args.end_frame]
    elif args.start_frame > 0:
        config.frame_range[0] = args.start_frame

    # Apply CLI overrides for resolution
    if args.width != 512:
        config.video.width = args.width
    if args.height != 512:
        config.video.height = args.height

    if args.max_size > 0:
        config.video.max_size = args.max_size

    # --flow-maxsize: override flow computation resolution
    if args.flow_maxsize >= 0:
        config.flow.flow_maxsize = args.flow_maxsize

    # --no-tiled-vae: force disable regardless of settings
    if args.no_tiled_vae:
        config.vae.use_tiled_vae = False

    # Print config mode
    if args.print_config:
        print(json.dumps(asdict(config), indent=2, default=str))
        return

    # Validate
    if not config.sd_checkpoint_path:
        print("Error: --checkpoint is required (path to SD model)")
        sys.exit(1)

    if not config.video.video_init_path:
        print("Error: --video is required (path to input video)")
        sys.exit(1)

    # Run
    from vibewarp.pipeline import run
    print(f"VibeWarp — starting run '{config.batch_name}'")
    print(f"  Video: {config.video.video_init_path}")
    print(f"  Prompt: {config.text_prompts.get(0, '')}")
    print(f"  Steps: {config.diffusion.steps}, CFG: {config.diffusion.cfg_scale}")
    print(f"  Resolution: {config.video.width}x{config.video.height}")

    output_paths = run(config, resume_from=args.resume_from)

    print(f"\nDone! {len(output_paths)} frames rendered.")
    if output_paths:
        print(f"Output: {os.path.dirname(output_paths[0])}")


if __name__ == '__main__':
    main()
