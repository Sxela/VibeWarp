"""ComfyUI HTTP backend for FLUX.2 Klein Edit.

The Comfy process owns model loading and GPU memory. VibeWarp uploads reference
images, submits a small API-format graph, and returns its preview as a PIL image.
"""

from __future__ import annotations

import io
import time
import uuid
from typing import Any, Iterable

import requests
from PIL import Image

from vibewarp.cancellation import is_cancel_requested
from vibewarp.core.comfy_progress import ComfyProgressMonitor
from vibewarp.core.edit_references import save_debug_reference
from vibewarp.core.flux import prepare_flux_references


class ComfyFluxError(RuntimeError):
    """A Comfy server request or execution failed."""


class ComfyFluxClient:
    """Synchronous client for the distilled FLUX.2 Klein Comfy workflow."""

    def __init__(self, config: Any):
        fx = config.flux
        self.base_url = fx.comfy_server_url.rstrip('/')
        self.timeout = float(fx.comfy_timeout)
        self.unet_name = fx.comfy_unet_name
        self.clip_name = fx.comfy_clip_name
        self.vae_name = fx.comfy_vae_name
        self.session = requests.Session()
        self.client_id = f"vibewarp-{uuid.uuid4()}"
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
            raise ComfyFluxError(
                f"ComfyUI request {method} {path} failed at "
                f"{self.base_url}{detail}") from exc

    def _validate_server(self) -> None:
        self._request('GET', '/system_stats')
        expected = {
            'UNETLoader': ('unet_name', self.unet_name),
            'CLIPLoader': ('clip_name', self.clip_name),
            'VAELoader': ('vae_name', self.vae_name),
            'Flux2Scheduler': None,
            'ReferenceLatent': None,
        }
        for node, model in expected.items():
            info = self._request('GET', f'/object_info/{node}').json().get(node)
            if not info:
                raise ComfyFluxError(
                    f"ComfyUI at {self.base_url} does not provide required node {node}")
            if model is not None:
                field, filename = model
                choices = info.get('input', {}).get('required', {}).get(field, [[]])[0]
                if filename not in choices:
                    raise ComfyFluxError(
                        f"ComfyUI cannot find {filename!r} for {node}.{field}; "
                        f"available: {choices}")

        print(f"Using ComfyUI FLUX backend: {self.base_url}")
        print(f"  UNet: {self.unet_name}")
        print(f"  text encoder: {self.clip_name}")
        print(f"  VAE: {self.vae_name}")

    def _upload_image(self, image: Image.Image, slot: int) -> str:
        data = io.BytesIO()
        image.convert('RGB').save(data, format='PNG')
        data.seek(0)
        filename = f"reference_{slot}.png"
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
        references: Iterable[str],
        prompt: str,
        seed: int,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
    ) -> dict[str, dict[str, Any]]:
        graph: dict[str, dict[str, Any]] = {
            '1': {'class_type': 'UNETLoader', 'inputs': {
                'unet_name': self.unet_name, 'weight_dtype': 'default'}},
            '2': {'class_type': 'CLIPLoader', 'inputs': {
                'clip_name': self.clip_name, 'type': 'flux2', 'device': 'default'}},
            '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': self.vae_name}},
            '4': {'class_type': 'CLIPTextEncode', 'inputs': {
                'text': prompt, 'clip': ['2', 0]}},
            '5': {'class_type': 'ConditioningZeroOut', 'inputs': {
                'conditioning': ['4', 0]}},
        }

        positive: list[Any] = ['4', 0]
        negative: list[Any] = ['5', 0]
        node_id = 10
        for image_name in references:
            load_id, encode_id = str(node_id), str(node_id + 1)
            positive_id, negative_id = str(node_id + 2), str(node_id + 3)
            graph[load_id] = {
                'class_type': 'LoadImage', 'inputs': {'image': image_name}}
            graph[encode_id] = {
                'class_type': 'VAEEncode',
                'inputs': {'pixels': [load_id, 0], 'vae': ['3', 0]},
            }
            graph[positive_id] = {
                'class_type': 'ReferenceLatent',
                'inputs': {'conditioning': positive, 'latent': [encode_id, 0]},
            }
            graph[negative_id] = {
                'class_type': 'ReferenceLatent',
                'inputs': {'conditioning': negative, 'latent': [encode_id, 0]},
            }
            positive, negative = [positive_id, 0], [negative_id, 0]
            node_id += 4

        graph.update({
            '30': {'class_type': 'EmptyFlux2LatentImage', 'inputs': {
                'width': int(width), 'height': int(height), 'batch_size': 1}},
            '31': {'class_type': 'RandomNoise', 'inputs': {
                'noise_seed': int(seed) & ((1 << 64) - 1)}},
            '32': {'class_type': 'KSamplerSelect', 'inputs': {'sampler_name': 'euler'}},
            '33': {'class_type': 'Flux2Scheduler', 'inputs': {
                'steps': int(steps), 'width': int(width), 'height': int(height)}},
            '34': {'class_type': 'CFGGuider', 'inputs': {
                'model': ['1', 0], 'positive': positive,
                'negative': negative, 'cfg': float(guidance_scale)}},
            '35': {'class_type': 'SamplerCustomAdvanced', 'inputs': {
                'noise': ['31', 0], 'guider': ['34', 0], 'sampler': ['32', 0],
                'sigmas': ['33', 0], 'latent_image': ['30', 0]}},
            '36': {'class_type': 'VAEDecode', 'inputs': {
                'samples': ['35', 0], 'vae': ['3', 0]}},
            '37': {'class_type': 'PreviewImage', 'inputs': {'images': ['36', 0]}},
        })
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
                'GET', f'/history/{prompt_id}', timeout=min(30.0, self.timeout)).json()
            item = history.get(prompt_id)
            if item:
                status = item.get('status', {})
                if status.get('status_str') == 'error':
                    raise ComfyFluxError(
                        f"ComfyUI execution failed: {status.get('messages', status)}")
                images = item.get('outputs', {}).get('37', {}).get('images', [])
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
                    raise ComfyFluxError(
                        "ComfyUI completed the Flux graph without an image output")
            time.sleep(0.25)
        raise ComfyFluxError(
            f"Timed out after {self.timeout:g}s waiting for ComfyUI prompt {prompt_id}")

    def render(
        self,
        config: Any,
        edit_image: Image.Image,
        prompt: str,
        seed: int,
        width: int,
        height: int,
        context_image: Image.Image | None = None,
        prev_context_image: Image.Image | None = None,
        mask_context_image: Image.Image | None = None,
        debug_dir: str | None = None,
        frame_num: int | None = None,
    ) -> Image.Image:
        references, prompt = prepare_flux_references(
            config, edit_image, context_image, prompt,
            raw_edit=frame_num == 0)
        if (mask_context_image is not None
                and config.flux.feed_consistency_mask_as_context):
            references.append(mask_context_image)
        if (prev_context_image is not None
                and config.flux.feed_prev_frame_as_context):
            references.append(prev_context_image)
        for index, reference in enumerate(references, start=1):
            save_debug_reference(reference, debug_dir, 'flux', index, frame_num)
        uploaded = [self._upload_image(image, i) for i, image in enumerate(references)]
        graph = self._build_prompt(
            uploaded, prompt, seed, width, height,
            config.flux.steps, config.flux.guidance_scale)
        progress = ComfyProgressMonitor(
            getattr(self, 'base_url', ''), self.client_id, graph)
        progress.start()
        try:
            result = self._request('POST', '/prompt', json={
                'prompt': graph, 'client_id': self.client_id}).json()
            prompt_id = result.get('prompt_id')
            if not prompt_id:
                raise ComfyFluxError(f"ComfyUI returned no prompt_id: {result}")
            progress.set_prompt_id(prompt_id)
            self._active_progress = progress
            return self._wait_for_output(prompt_id)
        finally:
            progress.close()
            self._active_progress = None
