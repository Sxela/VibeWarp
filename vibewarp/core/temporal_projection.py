"""Frame history and graph integration for temporal guidance."""

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image
from safetensors.numpy import save


def history_path(ctx, frame, suffix):
    return Path(ctx.batch_folder) / 'temp' / 'temporal-guidance' / f'{frame:06d}.{suffix}'


def save_native_history(ctx, state, samples):
    from safetensors.torch import save_file
    path = history_path(ctx, state.frame_num, 'safetensors')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    save_file({'samples': samples.detach().cpu().contiguous()}, str(tmp))
    tmp.replace(path)


def native_target(ctx, state, encode_initial, device=None):
    """Recover the exact preceding sample, including after a process restart."""
    from safetensors.torch import load_file
    from vibewarp.core.comfy_temporal_guidance import warp_target
    start = getattr(ctx, '_sequence_start_frame', 0)
    if state.frame_num == start:
        return None
    if ctx.temporal_motion is None:
        raise ValueError(f'Temporal guidance requires flow and consistency for frame {state.frame_num}')
    if state.frame_num == start + 1 and ctx.config.video.first_frame_path:
        history = encode_initial(state.prev_frame)
    elif state.prev_latent is not None:
        history = state.prev_latent
    else:
        path = history_path(ctx, state.frame_num - 1, 'safetensors')
        if not path.is_file():
            raise ValueError(f'Missing temporal latent {path}. Resume from the sequence start.')
        history = load_file(str(path))['samples']
    # Use the same interpolation kernel on a resumed frame as on a live one.
    return warp_target(history.to(device=device), *ctx.temporal_motion)


def wrap_sd_projection(model, target_and_mask, sigmas, options):
    from vibewarp.core.comfy_temporal_guidance import project, sigma_window
    bounds = sigma_window(sigmas, options.temporal_guidance_start, options.temporal_guidance_end)
    if bounds is None:
        return model
    target, mask = target_and_mask

    def denoise(x, sigma, **kwargs):
        nonlocal target, mask
        clean = model(x, sigma, **kwargs)
        if bounds[0] <= float(sigma.flatten()[0]) <= bounds[1]:
            target = target.to(clean.device)
            mask = mask.to(clean.device)
            return project(clean, target, mask, options.temporal_guidance_strength)
        return clean

    return denoise


@dataclass
class ComfyFrameGuidance:
    checkpoint: Path
    motion: tuple | None
    previous_checkpoint: Path | None
    initial_image: Image.Image | None

    @classmethod
    def for_frame(cls, ctx, state):
        frame = state.frame_num
        start = getattr(ctx, '_sequence_start_frame', 0)
        checkpoint = history_path(ctx, frame, 'json')
        if frame == start:
            return cls(checkpoint, None, None, None)
        if ctx.temporal_motion is None:
            raise ValueError(f'Temporal guidance needs backward flow and consistency for frame {frame}')
        initial = frame == start + 1 and bool(ctx.config.video.first_frame_path)
        return cls(checkpoint, ctx.temporal_motion,
                   None if initial else history_path(ctx, frame - 1, 'json'),
                   state.prev_frame if initial else None)

    def _upload(self, client, data, extension, mime):
        name = hashlib.sha256(data).hexdigest() + extension
        ref = client._request('POST', '/upload/image', files={'image': (name, data, mime)},
                              data={'type': 'input', 'subfolder': client.upload_subfolder}).json()
        return str(PurePosixPath(ref.get('subfolder', '')) / ref['name'])

    def prepare(self, client, graph, config):
        """Attach after the backend's final model patch and capture the decoded sample."""
        if not getattr(client, '_temporal_node_checked', False):
            node = 'VibeWarpTemporalGuidance'
            if not client._request('GET', '/object_info/' + node).json().get(node):
                raise ValueError('Copy vibewarp/core/comfy_temporal_guidance.py into ComfyUI/custom_nodes/ and restart ComfyUI to enable temporal guidance')
            client._temporal_node_checked = True
        decode = next(n['inputs'] for n in graph.values() if n['class_type'] == 'VAEDecode')
        sampled = decode['samples']
        graph['vw_history'] = {'class_type': 'SaveLatent', 'inputs': {
            'samples': sampled, 'filename_prefix': f'vibewarp/{client.client_id}/{self.checkpoint.stem}'}}
        if self.motion is None:
            return
        sampler = graph[sampled[0]]['inputs']
        owner = graph[sampler['guider'][0]]['inputs'] if 'guider' in sampler else sampler
        options = config.flow
        args = {'vae': decode['vae'], 'canvas': sampler['latent_image'], 'model': owner['model'],
                'strength': options.temporal_guidance_strength, 'start': options.temporal_guidance_start,
                'end': options.temporal_guidance_end}
        if self.initial_image is not None:
            data = io.BytesIO()
            self.initial_image.save(data, format='PNG')
            name = self._upload(client, data.getvalue(), '.png', 'image/png')
            graph['vw_initial'] = {'class_type': 'LoadImage', 'inputs': {'image': name}}
            args['initial_image'] = ['vw_initial', 0]
        else:
            if not self.previous_checkpoint.is_file():
                raise ValueError(f'Missing temporal latent checkpoint {self.previous_checkpoint}. Resume from the sequence start.')
            ref = json.loads(self.previous_checkpoint.read_text())
            if ref['server'] != client.base_url or ref['model'] != config.model_version:
                raise ValueError('Temporal history belongs to a different ComfyUI server or model')
            graph['vw_previous'] = {'class_type': 'LoadLatent', 'inputs': {'latent': ref['latent']}}
            args['previous'] = ['vw_previous', 0]
        flow, mask = self.motion
        packed = np.stack((flow[..., 0], flow[..., 1], mask))[None].astype(np.float32)
        data = save({'latent_tensor': packed, 'latent_format_version_0': np.empty(0, np.float32)})
        name = self._upload(client, data, '.latent', 'application/octet-stream')
        graph['vw_motion'] = {'class_type': 'LoadLatent', 'inputs': {'latent': name}}
        args['motion'] = ['vw_motion', 0]
        graph['vw_guidance'] = {'class_type': 'VibeWarpTemporalGuidance', 'inputs': args}
        owner['model'] = ['vw_guidance', 0]

    def record(self, client, prompt_id, config):
        history = client._request('GET', f'/history/{prompt_id}').json()[prompt_id]
        ref, = history['outputs']['vw_history']['latents']
        name = PurePosixPath(ref['subfolder']) / ref['filename']
        metadata = {'server': client.base_url, 'model': config.model_version, 'latent': f'{name} [output]'}
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.checkpoint.with_suffix('.tmp-' + client.client_id)
        tmp.write_text(json.dumps(metadata))
        tmp.replace(self.checkpoint)
