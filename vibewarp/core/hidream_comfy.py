"""Native ComfyUI HTTP backend for HiDream-O1 pixel-space editing."""

from __future__ import annotations

import io
import math
import time
import uuid
from typing import Any

import requests
from PIL import Image

from vibewarp.cancellation import is_cancel_requested
from vibewarp.core.comfy_progress import ComfyProgressMonitor
from vibewarp.core.edit_references import save_debug_reference


class ComfyHiDreamError(RuntimeError):
    """A ComfyUI request or HiDream graph execution failed."""


HIDREAM_TRAINED_RESOLUTIONS = (
    (2048, 2048),
    (2304, 1728), (2304, 1792), (2496, 1664), (2560, 1440), (3104, 1312),
    (1728, 2304), (1792, 2304), (1664, 2496), (1440, 2560), (1312, 3104),
)


def nearest_trained_resolution(width: int, height: int) -> tuple[int, int]:
    """Return O1's training preset with the closest aspect ratio."""
    if width <= 0 or height <= 0:
        raise ValueError("HiDream render dimensions must be positive")
    target = width / height
    return min(
        HIDREAM_TRAINED_RESOLUTIONS,
        key=lambda size: abs(math.log((size[0] / size[1]) / target)),
    )


def _upload_filename(stem: str, frame_num: int | None) -> str:
    suffix = f'_{frame_num:06d}' if frame_num is not None else ''
    return f'{stem}{suffix}.png'


class ComfyHiDreamClient:
    """Synchronous client for ComfyUI's native HiDream-O1 nodes."""

    output_node_id = '9'

    def __init__(self, config: Any):
        hd = config.hidream
        self.base_url = hd.comfy_server_url.rstrip('/')
        self.timeout = float(hd.comfy_timeout)
        self.checkpoint_name = hd.comfy_checkpoint_name
        self.session = requests.Session()
        self.client_id = f"vibewarp-hidream-{uuid.uuid4()}"
        self.upload_subfolder = f"vibewarp/{self.client_id}"
        self._validate_server()

    def _request(self, method: str, path: str, **kwargs):
        timeout = kwargs.pop('timeout', min(self.timeout, 30.0))
        try:
            response = self.session.request(
                method, f"{self.base_url}{path}", timeout=timeout, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            detail = ''
            response = getattr(exc, 'response', None)
            if response is not None:
                detail = f": {response.text[:2000]}"
            raise ComfyHiDreamError(
                f"ComfyUI request {method} {path} failed at "
                f"{self.base_url}{detail}") from exc

    def _validate_server(self) -> None:
        self._request('GET', '/system_stats')
        required = (
            'CheckpointLoaderSimple', 'CLIPTextEncode', 'LoadImage',
            'HiDreamO1ReferenceImages', 'EmptyHiDreamO1LatentImage',
            'ModelNoiseScale', 'HiDreamO1PatchSeamSmoothing',
            'BasicScheduler', 'KSamplerSelect', 'SamplerCustom',
            'VAEDecode', 'PreviewImage',
        )
        infos = {}
        for node in required:
            info = self._request('GET', f'/object_info/{node}').json().get(node)
            if not info:
                raise ComfyHiDreamError(
                    f"ComfyUI at {self.base_url} does not provide required node {node}; "
                    "update ComfyUI to a build with native HiDream-O1 support")
            infos[node] = info

        choices = (infos['CheckpointLoaderSimple'].get('input', {})
                   .get('required', {}).get('ckpt_name', [[]])[0])
        if self.checkpoint_name not in choices:
            raise ComfyHiDreamError(
                f"ComfyUI cannot find {self.checkpoint_name!r} in models/checkpoints; "
                f"available: {choices}")
        print(f"Using ComfyUI HiDream-O1 backend: {self.base_url}")
        print(f"  checkpoint: {self.checkpoint_name}")

    def _upload_image(
        self, image: Image.Image, filename: str = 'edit_target.png'
    ) -> str:
        data = io.BytesIO()
        image.convert('RGB').save(data, format='PNG')
        data.seek(0)
        response = self._request(
            'POST', '/upload/image',
            files={'image': (filename, data, 'image/png')},
            data={
                'type': 'input',
                'subfolder': self.upload_subfolder,
                'overwrite': 'true',
            },
        ).json()
        name = response.get('name', filename)
        subfolder = response.get('subfolder', self.upload_subfolder)
        return f"{subfolder}/{name}" if subfolder else name

    def _build_prompt(
        self,
        image_names: list[str],
        prompt: str,
        negative_prompt: str,
        seed: int,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        sampler: str,
        scheduler: str,
        noise_scale: float = 8.0,
        patch_seam_smoothing: bool = True,
        patch_seam_start: float = 0.8,
        patch_seam_passes: str = 'ramp_2_4',
        patch_seam_blend: str = 'median',
    ) -> dict[str, dict[str, Any]]:
        """Build an API-format graph using only native ComfyUI nodes."""
        graph = {
            '1': {'class_type': 'CheckpointLoaderSimple', 'inputs': {
                'ckpt_name': self.checkpoint_name}},
            '2': {'class_type': 'CLIPTextEncode', 'inputs': {
                'text': prompt, 'clip': ['1', 1]}},
            '3': {'class_type': 'CLIPTextEncode', 'inputs': {
                'text': negative_prompt, 'clip': ['1', 1]}},
            '5': {'class_type': 'HiDreamO1ReferenceImages', 'inputs': {
                'positive': ['2', 0],
                'negative': ['3', 0],
            }},
            '6': {'class_type': 'EmptyHiDreamO1LatentImage', 'inputs': {
                'width': int(width), 'height': int(height), 'batch_size': 1}},
            # Mirror the native Full workflow and explicitly lock its absolute
            # training noise scale instead of relying on checkpoint metadata.
            '7': {'class_type': 'ModelNoiseScale', 'inputs': {
                'model': ['1', 0], 'noise_scale': float(noise_scale)}},
            '11': {'class_type': 'BasicScheduler', 'inputs': {
                'model': ['7', 0],
                'scheduler': scheduler,
                'steps': int(steps),
                'denoise': 1.0,
            }},
            '12': {'class_type': 'KSamplerSelect', 'inputs': {
                'sampler_name': sampler}},
            '13': {'class_type': 'SamplerCustom', 'inputs': {
                'model': ['10', 0] if patch_seam_smoothing else ['7', 0],
                'add_noise': True,
                'noise_seed': int(seed) & ((1 << 64) - 1),
                'cfg': float(guidance_scale),
                'positive': ['5', 0],
                'negative': ['5', 1],
                'sampler': ['12', 0],
                'sigmas': ['11', 0],
                'latent_image': ['6', 0],
            }},
            # HiDream's "VAE" output is a raw-pixel codec; this node does not
            # introduce an autoencoder round-trip.
            '8': {'class_type': 'VAEDecode', 'inputs': {
                'samples': ['13', 0], 'vae': ['1', 2]}},
            '9': {'class_type': 'PreviewImage', 'inputs': {'images': ['8', 0]}},
        }
        for index, image_name in enumerate(image_names, start=1):
            load_id = '4' if index == 1 else str(12 + index)
            graph[load_id] = {
                'class_type': 'LoadImage', 'inputs': {'image': image_name}}
            # Comfy V3 dynamic inputs are flat, dot-qualified API keys.
            graph['5']['inputs'][f'images.image_{index}'] = [load_id, 0]
        if patch_seam_smoothing:
            graph['10'] = {'class_type': 'HiDreamO1PatchSeamSmoothing', 'inputs': {
                'model': ['7', 0],
                'start_percent': float(patch_seam_start),
                'end_percent': 1.0,
                'pattern': 'single_shift',
                'passes': patch_seam_passes,
                'blend': patch_seam_blend,
                'strength': 1.0,
            }}
        return graph

    def _wait_for_output(
        self, prompt_id: str, progress: ComfyProgressMonitor | None = None
    ) -> Image.Image:
        if progress is None:
            progress = getattr(self, '_active_progress', None)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if is_cancel_requested():
                if progress is not None:
                    progress.cancel(self._request)
                raise KeyboardInterrupt('Cancelled by user')
            if progress is not None:
                progress.poll(self._request)
            history = self._request(
                'GET', f'/history/{prompt_id}',
                timeout=min(30.0, self.timeout)).json()
            item = history.get(prompt_id)
            if item:
                status = item.get('status', {})
                if status.get('status_str') == 'error':
                    raise ComfyHiDreamError(
                        f"ComfyUI execution failed: {status.get('messages', status)}")
                images = (item.get('outputs', {}).get(self.output_node_id, {})
                          .get('images', []))
                if images:
                    ref = images[0]
                    response = self._request('GET', '/view', params={
                        'filename': ref['filename'],
                        'subfolder': ref.get('subfolder', ''),
                        'type': ref.get('type', 'temp'),
                    })
                    image = Image.open(io.BytesIO(response.content))
                    image.load()
                    return image.convert('RGB')
                if status.get('completed'):
                    raise ComfyHiDreamError(
                        "ComfyUI completed the HiDream graph without an image output")
            time.sleep(0.25)
        raise ComfyHiDreamError(
            f"Timed out after {self.timeout:g}s waiting for ComfyUI prompt {prompt_id}")

    def render(
        self,
        config: Any,
        edit_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        seed: int,
        width: int,
        height: int,
        reference_images: list[Image.Image] | None = None,
        debug_dir: str | None = None,
        frame_num: int | None = None,
    ) -> Image.Image:
        hd = config.hidream
        if hd.use_trained_resolution:
            render_width, render_height = nearest_trained_resolution(width, height)
            if (render_width, render_height) != (width, height):
                print(
                    f"  HiDream: sampling at trained resolution "
                    f"{render_width}x{render_height}; output will be resized to "
                    f"{width}x{height}")
            edit_image = edit_image.resize(
                (render_width, render_height), Image.Resampling.LANCZOS)
            width, height = render_width, render_height

        references = ([image.convert('RGB') for image in reference_images]
                      if reference_images else [edit_image.convert('RGB')])
        references = [
            image.resize((width, height), Image.Resampling.LANCZOS)
            if image.size != (width, height) else image
            for image in references
        ]
        from vibewarp.core.edit_references import reference_prompt
        prompt = reference_prompt(
            prompt, hd.multi_reference_instruction, len(references))
        uploaded = []
        for index, image in enumerate(references, start=1):
            save_debug_reference(image, debug_dir, 'hidream', index, frame_num)
            uploaded.append(self._upload_image(
                image, _upload_filename(f'reference_{index}', frame_num)))
        graph = self._build_prompt(
            uploaded, prompt, negative_prompt, seed, width, height,
            hd.steps, hd.guidance_scale, hd.sampler, hd.scheduler,
            hd.noise_scale, hd.patch_seam_smoothing, hd.patch_seam_start,
            hd.patch_seam_passes, hd.patch_seam_blend)
        progress = ComfyProgressMonitor(
            getattr(self, 'base_url', ''), self.client_id, graph)
        progress.start()
        try:
            result = self._request('POST', '/prompt', json={
                'prompt': graph, 'client_id': self.client_id}).json()
            prompt_id = result.get('prompt_id')
            if not prompt_id:
                raise ComfyHiDreamError(f"ComfyUI returned no prompt_id: {result}")
            progress.set_prompt_id(prompt_id)
            self._active_progress = progress
            return self._wait_for_output(prompt_id)
        finally:
            progress.close()
            self._active_progress = None
