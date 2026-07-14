"""Video output - assemble rendered frames into video files.

Extracted from notebook cell 57 (Create Video).
Supports: optical flow / linear blending, RealESRGAN upscaling,
background mask compositing, ffmpeg deflicker filter, audio preservation.
"""

import gc
import os
import subprocess
from glob import glob
from multiprocessing.pool import ThreadPool as Pool
from typing import Optional

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from vibewarp.flow.consistency import load_cc
from vibewarp.flow.warp import warp_frame


# ---- RealESRGAN model configurations (matching notebook cell 57) ----

_REALESRGAN_CONFIGS = {
    'RealESRGAN_x4plus': {
        'arch': 'rrdbnet', 'scale': 4, 'num_block': 23, 'num_grow_ch': 32,
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    },
    'RealESRNet_x4plus': {
        'arch': 'rrdbnet', 'scale': 4, 'num_block': 23, 'num_grow_ch': 32,
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth',
    },
    'RealESRGAN_x4plus_anime_6B': {
        'arch': 'rrdbnet', 'scale': 4, 'num_block': 6, 'num_grow_ch': 32,
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
    },
    'RealESRGAN_x2plus': {
        'arch': 'rrdbnet', 'scale': 2, 'num_block': 23, 'num_grow_ch': 32,
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
    },
    'realesr-animevideov3': {
        'arch': 'srvgg', 'scale': 4, 'num_conv': 16,
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth',
    },
    'realesr-general-x4v3': {
        'arch': 'srvgg', 'scale': 4, 'num_conv': 32,
        'url': [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
        ],
    },
}


def load_realesrgan(model_name: str, model_path: str):
    """Load a RealESRGAN upsampler.

    Args:
        model_name: one of the keys in _REALESRGAN_CONFIGS
        model_path: path to the .pth weights file

    Returns:
        RealESRGANer instance ready for inference.

    Raises:
        ImportError: if realesrgan / basicsr are not installed
        FileNotFoundError: if model_path does not exist
        ValueError: if model_name is unrecognised
    """
    try:
        from realesrgan import RealESRGANer
    except ImportError as e:
        raise ImportError(
            "realesrgan package is required for upscaling. "
            "Install it with: pip install realesrgan"
        ) from e

    if model_name not in _REALESRGAN_CONFIGS:
        raise ValueError(
            f"Unknown upscale_model '{model_name}'. "
            f"Valid options: {list(_REALESRGAN_CONFIGS)}"
        )

    if not os.path.isfile(model_path):
        cfg = _REALESRGAN_CONFIGS[model_name]
        urls = cfg['url'] if isinstance(cfg['url'], list) else [cfg['url']]
        raise FileNotFoundError(
            f"RealESRGAN weights not found: {model_path}\n"
            f"Download from: {urls[-1]}"
        )

    cfg = _REALESRGAN_CONFIGS[model_name]
    scale = cfg['scale']

    if cfg['arch'] == 'rrdbnet':
        from basicsr.archs.rrdbnet_arch import RRDBNet
        up_model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=cfg['num_block'], num_grow_ch=cfg['num_grow_ch'],
            scale=scale,
        )
    else:
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        up_model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_conv=cfg['num_conv'], upscale=4, act_type='prelu',
        )

    return RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=up_model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        device='cuda',
    )


def upscale_frame(upsampler, frame: Image.Image, upscale_ratio: int) -> Image.Image:
    """Upscale a PIL image using a RealESRGANer instance.

    The upsampler uses BGR internally (OpenCV convention); this handles the
    RGB↔BGR conversion so callers always work with RGB PIL images.

    Args:
        upsampler: RealESRGANer instance from load_realesrgan()
        frame: RGB PIL Image
        upscale_ratio: output scale factor passed to upsampler.enhance()

    Returns:
        Upscaled RGB PIL Image.
    """
    arr = np.array(frame.convert('RGB'))[..., ::-1]  # RGB → BGR
    output, _ = upsampler.enhance(arr.clip(0, 255), outscale=upscale_ratio)
    return Image.fromarray(output[..., ::-1].clip(0, 255).astype('uint8'))  # BGR → RGB


def apply_video_mask(
    frame: Image.Image,
    frame_num: int,
    background: str,
    background_source: str,
    video_frames_folder: str,
    invert: bool = False,
    clip_low: int = 0,
    clip_high: int = 255,
) -> Image.Image:
    """Composite a stylized frame onto a background using a pre-computed alpha mask.

    The alpha masks must be in ``{video_frames_folder}_alpha/`` named as
    ``{frame_num+1:06d}.jpg`` — the same numbering used by the notebook.

    Args:
        frame: rendered stylized frame (RGB PIL Image)
        frame_num: 0-based frame index
        background: 'color', 'image', or 'init_video'
        background_source: CSS color string or image path (for 'color'/'image')
        video_frames_folder: path to raw video frame images
        invert: invert the alpha mask before compositing
        clip_low: pixels below this in the mask are set to 0
        clip_high: pixels above this are set to 255

    Returns:
        Composited RGB PIL Image, or the original frame if the mask is missing.
    """
    alpha_dir = video_frames_folder + '_alpha'
    alpha_path = os.path.join(alpha_dir, f'{frame_num + 1:06d}.jpg')

    if not os.path.exists(alpha_path):
        return frame

    size = frame.size
    alpha = Image.open(alpha_path).resize(size).convert('L')

    if invert:
        alpha = ImageOps.invert(alpha)

    if clip_low > 0 or clip_high < 255:
        arr = np.array(alpha)
        if clip_high < 255:
            arr = np.where(arr < clip_high, arr, 255)
        if clip_low > 0:
            arr = np.where(arr > clip_low, arr, 0)
        alpha = Image.fromarray(arr)

    if background == 'color':
        bg = Image.new('RGB', size, background_source or 'black')
    elif background == 'image':
        bg = Image.open(background_source).convert('RGB').resize(size)
    else:  # 'init_video'
        init_path = os.path.join(video_frames_folder, f'{frame_num + 1:06d}.jpg')
        if os.path.exists(init_path):
            bg = Image.open(init_path).convert('RGB').resize(size)
        else:
            return frame

    bg.paste(frame.convert('RGB'), (0, 0), alpha)
    return bg


def attach_audio(
    video_path: str,
    source_video: str,
    ffmpeg_path: str,
    output_path: Optional[str] = None,
) -> str:
    """Attach the audio track from source_video to video_path.

    Args:
        video_path: path to the silent rendered video
        source_video: path to the original video with audio
        ffmpeg_path: ffmpeg executable path
        output_path: destination path (replaces video_path if None)

    Returns:
        Path to the video with audio attached, or video_path if failed.
    """
    if not os.path.exists(video_path) or not os.path.exists(source_video):
        print("Audio attachment skipped: input or source video not found")
        return video_path

    dest = output_path or (video_path[:-4] + '_audio' + video_path[-4:])
    cmd = [
        ffmpeg_path, '-y',
        '-i', video_path,
        '-i', source_video,
        '-map', '0:v',
        '-map', '1:a',
        '-c:v', 'copy',
        '-shortest',
        dest,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Audio attachment failed (source may have no audio): "
              + result.stderr.split('\n')[-2])
        return video_path

    if output_path is None:
        # Replace silent video with audio version
        os.replace(dest, video_path)
        return video_path

    return dest


def create_video(
    frames_dir: str,
    output_path: str,
    fps: float = 12.0,
    batch_name: str = 'warpfusion',
    batch_num: int = 0,
    img_format: str = 'png',
    blend_mode: str = 'None',
    blend: float = 0.5,
    check_consistency: bool = False,
    threads: int = 4,
    ffmpeg_path: Optional[str] = None,
    flow_dir: Optional[str] = None,
    # Audio
    keep_audio: bool = False,
    source_video: str = '',
    # Deflicker
    use_deflicker: bool = False,
    # RealESRGAN upscaling
    upscale_ratio: int = 1,
    upscale_model: str = 'realesr-animevideov3',
    upscale_model_path: str = '',
    # Background mask compositing
    use_background_mask_video: bool = False,
    invert_mask_video: bool = False,
    background_video: str = 'init_video',
    background_source_video: str = '',
    video_frames_folder: str = '',
    mask_clip_low: int = 0,
    mask_clip_high: int = 255,
) -> str:
    """Assemble rendered frames into a video file.

    Per-frame post-processing order (when enabled):
      1. Blend with previous frame (optical flow or linear)
      2. Apply background mask composite
      3. RealESRGAN upscale
    Then encode with ffmpeg, optionally adding a deflicker filter and audio.

    Args:
        frames_dir: directory containing rendered frame images
        output_path: path for the output video file
        fps: frames per second
        batch_name: batch name prefix for frame filename pattern
        batch_num: batch number
        img_format: image file extension ('png' or 'jpg')
        blend_mode: 'None', 'linear', or 'optical flow'
        blend: blend factor for inter-frame blending
        check_consistency: use consistency maps for flow blending
        threads: number of processing threads
        ffmpeg_path: path to ffmpeg executable
        flow_dir: directory with optical flow files
        keep_audio: attach audio from source_video to output
        source_video: path to source video for audio extraction
        use_deflicker: add ffmpeg deflicker filter (deflicker=mode=pm:size=10)
        upscale_ratio: RealESRGAN upscale factor (1 = disabled)
        upscale_model: RealESRGAN model name
        upscale_model_path: path to RealESRGAN .pth weights
        use_background_mask_video: composite onto background using alpha masks
        invert_mask_video: invert alpha mask before compositing
        background_video: background type ('color', 'image', 'init_video')
        background_source_video: color string or image path for background
        video_frames_folder: raw video frames dir (for bg + alpha masks)
        mask_clip_low: threshold below which alpha mask values → 0
        mask_clip_high: threshold above which alpha mask values → 255

    Returns:
        Path to the created video file.
    """
    if ffmpeg_path is None:
        from vibewarp.video.input import _find_ffmpeg
        ffmpeg_path = _find_ffmpeg()

    frame_pattern = os.path.join(frames_dir,
                                 f'{batch_name}({batch_num})_*.{img_format}')
    frames = sorted(glob(frame_pattern))

    if not frames:
        raise FileNotFoundError(f"No frames found matching {frame_pattern}")

    if blend_mode == 'optical flow' and flow_dir is None:
        print("No flow directory — falling back to no blending")
        blend_mode = 'None'

    # Load RealESRGAN upsampler once if needed
    upsampler = None
    if upscale_ratio > 1:
        try:
            upsampler = load_realesrgan(upscale_model, upscale_model_path)
            print(f"RealESRGAN loaded: {upscale_model} x{upscale_ratio}")
        except (ImportError, FileNotFoundError, ValueError) as e:
            print(f"Upscaling disabled: {e}")
            upscale_ratio = 1

    # Build a per-frame post-processor that applies mask + upscale
    def _postprocess(img: Image.Image, frame_idx: int) -> Image.Image:
        if use_background_mask_video and video_frames_folder:
            img = apply_video_mask(
                img, frame_idx, background_video, background_source_video,
                video_frames_folder, invert_mask_video, mask_clip_low, mask_clip_high,
            )
        if upscale_ratio > 1 and upsampler is not None:
            img = upscale_frame(upsampler, img, upscale_ratio)
        return img

    # Build processed frames in a temp dir
    blend_dir = os.path.join(frames_dir, 'blend_temp')
    os.makedirs(blend_dir, exist_ok=True)

    if blend_mode != 'None' and len(frames) > 1:
        _blend_frames(frames, blend_dir, blend_mode, blend, flow_dir,
                      check_consistency, threads, _postprocess)
    else:
        for i, f in enumerate(tqdm(frames, desc="Copying frames")):
            img = _postprocess(Image.open(f), i)
            img.save(os.path.join(blend_dir, f'{i:06d}.png'))

    frame_input = os.path.join(blend_dir, '%06d.png')

    # Build ffmpeg encode command
    vf_filters = [f'fps={fps}']
    if use_deflicker:
        vf_filters.append('deflicker=mode=pm:size=10')

    cmd = [
        ffmpeg_path, '-y',
        '-r', str(fps),
        '-i', frame_input,
        '-c:v', 'libx264',
        '-vf', ','.join(vf_filters),
        '-pix_fmt', 'yuv420p',
        '-crf', '17',
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}")

    # Clean up upsampler before audio pass
    if upsampler is not None:
        del upsampler
        gc.collect()

    # Attach audio from source video
    if keep_audio and source_video:
        output_path = attach_audio(output_path, source_video, ffmpeg_path)

    return output_path


def _blend_frames(frames, output_dir, blend_mode, blend, flow_dir,
                  check_consistency, threads, postprocess_fn=None):
    """Blend consecutive frames for smoother video output.

    Args:
        frames: sorted list of frame paths
        output_dir: directory to save processed frames
        blend_mode: 'linear' or 'optical flow'
        blend: blend factor
        flow_dir: directory with .npy flow files
        check_consistency: load CC maps for flow blending
        threads: thread count (currently unused — sequential for simplicity)
        postprocess_fn: optional callable(img, frame_idx) → img applied after blend
    """
    if postprocess_fn is None:
        postprocess_fn = lambda img, _: img

    for i, frame_path in enumerate(tqdm(frames, desc="Processing frames")):
        if i == 0 or blend_mode == 'None':
            img = postprocess_fn(Image.open(frame_path).convert('RGB'), i)
        else:
            curr = np.array(Image.open(frame_path).convert('RGB'))
            prev_path = os.path.join(output_dir, f'{i - 1:06d}.png')
            if os.path.exists(prev_path):
                prev = np.array(Image.open(prev_path).convert('RGB'))
            else:
                prev = np.array(Image.open(frames[i - 1]).convert('RGB'))

            if blend_mode == 'linear':
                blended = prev * (1 - blend) + curr * blend
                img = Image.fromarray(blended.round().astype('uint8'))
            elif blend_mode == 'optical flow' and flow_dir:
                flow_path = _find_flow_file(flow_dir, frames[i - 1])
                if flow_path and os.path.exists(flow_path):
                    flow = np.load(flow_path)
                    weights = None
                    if check_consistency:
                        cc_path = _find_cc_file(flow_dir, flow_path)
                        if cc_path and os.path.exists(cc_path):
                            cc_map = np.array(Image.open(cc_path).convert('RGB'))
                            weights = load_cc(cc_map)
                    img = warp_frame(
                        Image.fromarray(prev), Image.fromarray(curr),
                        flow, blend=blend, weights=weights, video_mode=True,
                    )
                else:
                    img = Image.fromarray(curr)
            else:
                img = Image.fromarray(curr)

            img = postprocess_fn(img, i)

        img.save(os.path.join(output_dir, f'{i:06d}.png'))


def _find_flow_file(flow_dir, frame_path):
    """Find the flow .npy file corresponding to a frame."""
    basename = os.path.splitext(os.path.basename(frame_path))[0]
    candidates = [
        os.path.join(flow_dir, f'{basename}.npy'),
        os.path.join(flow_dir, f'{basename}.jpg.npy'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _find_cc_file(flow_dir, flow_path):
    """Find the CC map file corresponding to a flow file."""
    stem = os.path.basename(flow_path)
    if stem.endswith('.npy'):
        stem = stem[:-4]
    candidates = [
        os.path.join(flow_dir, f'{stem}_12-21_cc.jpg'),
    ]
    if not stem.endswith('.jpg'):
        candidates.append(os.path.join(flow_dir, f'{stem}.jpg_12-21_cc.jpg'))
    for c in candidates:
        if os.path.exists(c):
            return c
    return None
