"""CUDA raw-input preparation; cached originals and RVM alpha are immutable inputs."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from vibewarp.cancellation import raise_if_cancelled

RVM_URL = 'https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp16.torchscript'
_lock = threading.RLock()


def enabled(config):
    return needs_alpha(config) or config.mask.raw_color_enabled


def needs_alpha(config):
    """RVM matting runs for masked tint, or when RVM is the background-mask source.
    User mask files never need RVM, CUDA, or the RVM model."""
    from vibewarp.video import masks
    return config.mask.rvm_enabled or (masks.uses_rvm(config) and masks.wanted(config))


def transforms_raw(config):
    return config.mask.rvm_enabled or config.mask.raw_color_enabled


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError('Raw video matting and color preparation require CUDA')


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('wb') as stream:
            write(stream)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def missing_spans(indices, exists):
    """Contiguous absent indices, without regenerating cached overlap."""
    spans = []
    for index in indices:
        if exists(index):
            continue
        if spans and spans[-1][1] + 1 == index:
            spans[-1][1] = index
        else:
            spans.append([index, index])
    return spans


class VideoCache:
    def __init__(self, config):
        self.config = config
        self.video = Path(config.video.video_init_path)
        self.stride = config.video.extract_nth_frame
        identity = {'version': 1, 'video': file_hash(self.video), 'stride': self.stride}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        base = Path(config.mask.rvm_cache_dir or Path(config.root_dir) / '.vibewarp_cache' / 'video_inputs')
        self.root = base / key
        self.frames = self.root / 'frames'
        self.frames.mkdir(parents=True, exist_ok=True)
        self.alpha = None

    def frame_path(self, index):
        return self.frames / f'{index+1:06d}.jpg'

    def ensure_frames(self, start, end):
        from vibewarp.video.input import _find_ffmpeg
        for first, last in missing_spans(range(start, end+1), lambda i: self.frame_path(i).is_file()):
            raise_if_cancelled()
            temporary = self.root / ('extract-' + uuid.uuid4().hex)
            temporary.mkdir()
            try:
                # CUDA decode. No software decode fallback: this feature explicitly
                # requests GPU processing. Frames are not kept as CUDA surfaces, so
                # ffmpeg downloads them in their native layout (nv12, p010, yuv444p...)
                # rather than a hard-coded `hwdownload,format=nv12`, which only fits
                # 8-bit 4:2:0 sources.
                selection = (f'select=between(n\\,{first*self.stride}\\,{last*self.stride})'
                             f'*not(mod(n\\,{self.stride}))')
                result = subprocess.run([_find_ffmpeg(), '-v', 'error', '-hwaccel', 'cuda',
                    '-i', str(self.video), '-vf', selection,
                    '-vsync', 'vfr', '-frames:v', str(last-first+1), '-q:v', '2', '-start_number', str(first+1),
                    str(temporary / '%06d.jpg')], capture_output=True)
                if result.returncode:
                    raise RuntimeError('CUDA video extraction failed: ' + result.stderr.decode(errors='replace')[-1500:])
                for index in range(first, last+1):
                    source = temporary / f'{index+1:06d}.jpg'
                    if not source.is_file():
                        raise ValueError(f'Source video has no extracted frame {index}')
                for source in temporary.glob('*.jpg'):
                    os.replace(source, self.frames / source.name)
            finally:
                shutil.rmtree(temporary)

    def ensure_alpha(self, start, end):
        from vibewarp.acquisition import acquire
        m = self.config.mask
        model_path = Path(m.rvm_model_path or Path(self.config.model_path) / 'rvm' / 'rvm_mobilenetv3_fp16.torchscript')
        if not model_path.is_file():
            if m.rvm_model_path:
                raise FileNotFoundError(model_path)
            acquire(RVM_URL, model_path)
        identity = f'{file_hash(model_path)}:{m.rvm_downsample_ratio}:rvm-recurrent-v1'
        self.alpha = self.root / hashlib.sha256(identity.encode()).hexdigest()
        self.alpha.mkdir(exist_ok=True)
        absent = [i for i in range(start, end+1) if not (self.alpha / f'{i:06d}.npy').is_file()]
        if not absent:
            return
        # Restore a preceding recurrent checkpoint; reset only at video start.
        checkpoints = sorted(int(p.stem) for p in self.alpha.glob('*.pt') if int(p.stem) < absent[0])
        previous = checkpoints[-1] if checkpoints else -1
        self.ensure_frames(previous+1, absent[-1])
        rec = ([v.cuda() for v in torch.load(self.alpha / f'{previous:06d}.pt', map_location='cpu', weights_only=True)]
               if previous >= 0 else [None]*4)
        model = torch.jit.load(str(model_path), map_location='cuda').eval()
        with torch.inference_mode():
            for index in range(previous+1, absent[-1]+1):
                raise_if_cancelled()
                with Image.open(self.frame_path(index)) as image:
                    source = image_tensor(image).half()
                _, alpha, *rec = model(source, *rec, m.rvm_downsample_ratio)
                target = self.alpha / f'{index:06d}.npy'
                if not target.exists():
                    values = alpha[0,0].float().clamp(0,1).cpu().numpy().astype(np.float16)
                    _atomic_bytes(target, lambda stream: np.save(stream, values, allow_pickle=False))
                if index % 16 == 15 or index == absent[-1]:
                    states = [v.cpu() for v in rec]
                    _atomic_bytes(self.alpha / f'{index:06d}.pt', lambda stream: torch.save(states, stream))
        del model, rec

    def prepare(self, start, end):
        require_cuda()
        with _lock:
            self.ensure_frames(start, end)
            if needs_alpha(self.config):
                self.ensure_alpha(start, end)
        return self


def image_tensor(image, device='cuda'):
    return torch.from_numpy(np.array(image.convert('RGB'), copy=True)).to(device).permute(2,0,1).unsqueeze(0).float()/255


def tint_tensor(raw, alpha, foreground, background, fg_opacity, bg_opacity):
    fg = raw.new_tensor(foreground).view(1,3,1,1)/255
    bg = raw.new_tensor(background).view(1,3,1,1)/255
    return torch.lerp(raw, fg, alpha*fg_opacity) + (bg-raw)*(1-alpha)*bg_opacity


def load_alpha(config, cache, frame, size, device=None):
    """Read immutable RVM alpha and match the render's resize/edge padding."""
    alpha = torch.from_numpy(np.load(cache.alpha / f'{frame:06d}.npy', allow_pickle=False).astype(np.float32))
    alpha = (alpha.cuda() if device is None else alpha.to(device))[None,None]
    return fit_alpha(alpha, config, size)


def fit_alpha(alpha, config, size):
    """Resize a (1,1,H,W) alpha to the render canvas, edge-padding Disco's padded canvas."""
    v = config.video
    visible = ((v.visible_height, v.visible_width)
               if config.disco.pad_to_multiple_64 and config.model_version.startswith('disco_') else size)
    alpha = F.interpolate(alpha, size=visible, mode='bilinear', align_corners=False)
    if alpha.shape[-2:] != size:
        dh, dw = size[0]-visible[0], size[1]-visible[1]
        alpha = F.pad(alpha, (dw//2,dw-dw//2,dh//2,dh-dh//2), mode='replicate')
    return alpha.clamp(0,1)


def apply_cached_mask(image, config, cache, frame, video_frames_folder=None):
    from vibewarp.video.output import apply_video_mask
    alpha = load_alpha(config, cache, frame, (image.height,image.width), device='cpu')
    mask = Image.fromarray(alpha[0,0].mul(255).round().byte().numpy())
    m = config.mask
    return apply_video_mask(image, frame, m.background, m.background_source,
        str(video_frames_folder or cache.frames), invert=m.invert_mask,
        clip_low=m.mask_clip_low, clip_high=m.mask_clip_high, alpha_mask=mask)


def color_tensor(raw, reference, method, strength):
    reference = F.interpolate(reference, size=raw.shape[-2:], mode='bilinear', align_corners=False)
    if method == 'PDF':
        from vibewarp.color.pdf_gpu import _get_rotations_torch, _pdf_transfer_nd_torch
        source = raw[0].flip(0).reshape(3,-1)
        target = reference[0].flip(0).reshape(3,-1)
        matched = _pdf_transfer_nd_torch(source, target, _get_rotations_torch(raw.device)).reshape_as(raw).flip(1)
    else:
        if method == 'LAB':
            source, target = rgb_to_lab(raw), rgb_to_lab(reference)
        else:
            source, target = raw, reference
        dims = (-2,-1)
        sm, tm = source.mean(dims,keepdim=True), target.mean(dims,keepdim=True)
        ss, ts = source.std(dims,keepdim=True,unbiased=False), target.std(dims,keepdim=True,unbiased=False)
        matched = (source-sm)*ts/ss.clamp_min(1e-6)+tm
        if method == 'LAB':
            matched = lab_to_rgb(matched)
    return torch.lerp(raw, matched.clamp(0,1), strength).clamp(0,1)


def rgb_to_lab(rgb):
    linear = torch.where(rgb > 0.04045, ((rgb+0.055)/1.055).pow(2.4), rgb/12.92)
    matrix = rgb.new_tensor([[0.4124564,0.3575761,0.1804375],
                             [0.2126729,0.7151522,0.0721750],
                             [0.0193339,0.1191920,0.9503041]])
    xyz = torch.einsum('ij,bjhw->bihw',matrix,linear)/rgb.new_tensor([0.95047,1,1.08883]).view(1,3,1,1)
    f = torch.where(xyz > (6/29)**3, xyz.clamp_min(0).pow(1/3), xyz/(3*(6/29)**2)+4/29)
    return torch.stack([116*f[:,1]-16,500*(f[:,0]-f[:,1]),200*(f[:,1]-f[:,2])],dim=1)


def lab_to_rgb(lab):
    fy = (lab[:,0]+16)/116
    f = torch.stack([fy+lab[:,1]/500,fy,fy-lab[:,2]/200],dim=1)
    xyz = torch.where(f>6/29,f.pow(3),3*(6/29)**2*(f-4/29))
    xyz = xyz*lab.new_tensor([0.95047,1,1.08883]).view(1,3,1,1)
    matrix = lab.new_tensor([[3.2404542,-1.5371385,-0.4985314],
                            [-0.9692660,1.8760108,0.0415560],
                            [0.0556434,-0.2040259,1.0572252]])
    rgb = torch.einsum('ij,bjhw->bihw',matrix,xyz)
    return torch.where(rgb>0.0031308,1.055*rgb.clamp_min(0).pow(1/2.4)-0.055,12.92*rgb).clamp(0,1)


def preview_frame(config, frame, reference_path=''):
    """Single-frame preparation, using the same cache and kernels as rendering."""
    from dataclasses import replace
    from vibewarp.video.input import probe_video
    require_cuda()
    if frame < 0:
        raise ValueError('Preview frame must be nonnegative')
    info = probe_video(config.video.video_init_path)
    if frame*config.video.extract_nth_frame >= info['frames']:
        raise ValueError('Preview frame exceeds the source video')
    cache = VideoCache(config).prepare(frame,frame)
    with Image.open(cache.frame_path(frame)) as opened:
        raw = opened.convert('RGB')
    reference = None
    if reference_path:
        with Image.open(reference_path) as opened:
            reference = opened.convert('RGB').copy()
    if config.mask.raw_color_enabled and config.mask.raw_color_strength and reference is None:
        raise ValueError('Choose a stylized reference image to preview raw color matching')
    # Cache frames and alpha are at source resolution. Preview them without
    # render-canvas padding; the render applies its own matching geometry.
    preview_config = replace(config,model_version='control_multi_v15')
    result = process_image(raw,preview_config,cache,frame,reference)
    if config.mask.use_background_mask:
        from vibewarp.video import masks
        if masks.uses_rvm(config):
            result = apply_cached_mask(result,preview_config,cache,frame)
        else:
            from vibewarp.video.output import apply_video_mask
            m = config.mask
            mask = masks.preview_mask(config,frame,(result.height,result.width),cache.root/'previews')
            result = apply_video_mask(result,frame,m.background,m.background_source,str(cache.frames),
                invert=m.invert_mask,clip_low=m.mask_clip_low,clip_high=m.mask_clip_high,alpha_mask=mask)
    target = cache.root / 'previews' / f'{uuid.uuid4().hex}.png'
    _atomic_bytes(target,lambda stream: result.save(stream,format='PNG'))
    return str(target)


def process_image(image, config, cache, frame, reference=None, size=None):
    require_cuda()
    m = config.mask
    with torch.inference_mode():
        raw = image_tensor(image)
        if size is not None and raw.shape[-2:] != size:
            raw = F.interpolate(raw,size=size,mode='bicubic',align_corners=False).clamp(0,1)
        if m.raw_color_enabled and m.raw_color_strength and reference is not None:
            raw = color_tensor(raw, image_tensor(reference), m.raw_color_method, m.raw_color_strength)
        if m.rvm_enabled and (m.foreground_opacity or m.background_opacity):
            alpha = load_alpha(config,cache,frame,raw.shape[-2:])
            raw = tint_tensor(raw,alpha,m.foreground_color,m.background_color,m.foreground_opacity,m.background_opacity)
        pixels = raw[0].permute(1,2,0).clamp(0,1).mul(255).round().byte().cpu().numpy()
    return Image.fromarray(pixels)


def prepare_run(config, extraction_folder):
    from vibewarp.video.input import probe_video
    total = (probe_video(config.video.video_init_path)['frames'] + config.video.extract_nth_frame-1)//config.video.extract_nth_frame
    selected = config.frame_range
    start = selected[0] if selected else 0
    end = selected[0] if len(selected)==1 else selected[1] if selected and selected[1]>0 else total-1
    if start >= total or end >= total:
        raise ValueError('Selected raw preparation range exceeds the source video')
    cache = VideoCache(config).prepare(start,end)
    folder = Path(extraction_folder); folder.mkdir(parents=True,exist_ok=True)
    for index in range(start,end+1):
        shutil.copyfile(cache.frame_path(index),folder/f'{index+1:06d}.jpg')
    return cache


def render_raw(image, ctx, frame, warped_reference=None):
    if not transforms_raw(ctx.config):
        return image
    reference = warped_reference
    first = getattr(ctx,'_sequence_start_frame',ctx.config.frame_range[0] if ctx.config.frame_range else 0)
    if ctx.config.mask.raw_color_reference == 'init stylized' or reference is None:
        index = first if ctx.config.mask.raw_color_reference == 'init stylized' else frame-1
        path = Path(ctx.batch_folder)/f'{ctx.config.batch_name}(0)_{index:06d}.{ctx.config.video.save_img_format}'
        # Batch renderers can load future raw cells before their immediately
        # preceding render exists. Use the available first stylized anchor.
        if not path.is_file():
            path = Path(ctx.batch_folder)/f'{ctx.config.batch_name}(0)_{first:06d}.{ctx.config.video.save_img_format}'
        if path.is_file():
            with Image.open(path) as opened:
                reference = opened.convert('RGB').copy()
        elif ctx.config.video.first_frame_path:
            with Image.open(ctx.config.video.first_frame_path) as opened:
                reference = opened.convert('RGB').copy()
        else:
            reference = None
    if ctx.config.mask.raw_color_enabled and reference is None:
        print(f'Raw color frame {frame}: no stylized reference yet; tint only')
    return process_image(image,ctx.config,ctx.raw_video_cache,frame,reference,
                         size=(ctx.config.video.height,ctx.config.video.width))


def prepared_raw_path(ctx, frame, warped_reference=None, *, purpose='init'):
    """Prepared raw frame on disk. `purpose` keeps the init and raw-frame guidance
    copies apart: they can be prepared against different colour references."""
    original = Path(ctx.video_frames_folder)/f'{frame+1:06d}.jpg'
    if not transforms_raw(ctx.config):
        return str(original)
    with Image.open(original) as opened:
        image = opened.convert('RGB')
    prepared = render_raw(image,ctx,frame,warped_reference)
    target = Path(ctx.batch_folder)/'temp'/f'raw_prepared_{purpose}_{frame:06d}.png'
    _atomic_bytes(target,lambda stream: prepared.save(stream,format='PNG'))
    return str(target)
