"""ComfyUI HTTP backend for Microsoft Mage-Flow-Edit."""

from __future__ import annotations

import uuid
from typing import Any, Iterable

import requests
from PIL import Image

from vibewarp.config import mage_model_defaults
from vibewarp.core.comfy_progress import ComfyProgressMonitor
from vibewarp.core.edit_references import reference_prompt, save_debug_reference
from vibewarp.core.flux_comfy import ComfyFluxClient


class ComfyMageFlowEditClient(ComfyFluxClient):
    """Synchronous client for ComfyUI's native Mage-Flow-Edit workflow."""

    def __init__(self, config: Any):
        mage = config.mage
        preset = mage_model_defaults(config)
        self.base_url = mage.comfy_server_url.rstrip('/')
        self.timeout = float(mage.comfy_timeout)
        self.unet_name = str(preset['comfy_unet_name'])
        self.clip_name = mage.comfy_clip_name
        self.vae_name = mage.comfy_vae_name
        self.session = requests.Session()
        self.client_id = f"vibewarp-mage-{uuid.uuid4()}"
        self.upload_subfolder = f"vibewarp/{self.client_id}"
        self._validate_server()

    def _validate_server(self) -> None:
        self._request('GET', '/system_stats')
        expected = {
            'UNETLoader': ('unet_name', self.unet_name),
            'CLIPLoader': ('clip_name', self.clip_name),
            'VAELoader': ('vae_name', self.vae_name),
            'LoadImage': None,
            'TextEncodeMageFlowEdit': None,
            'KSampler': None,
            'VAEDecode': None,
            'PreviewImage': None,
        }
        for node, model in expected.items():
            info = self._request('GET', f'/object_info/{node}').json().get(node)
            if not info:
                raise RuntimeError(
                    f"ComfyUI at {self.base_url} does not provide required node "
                    f"{node}; update ComfyUI before using Mage-Flow-Edit")
            if model is None:
                continue
            field, filename = model
            choices = info.get('input', {}).get('required', {}).get(
                field, [[]])[0]
            if filename not in choices:
                raise RuntimeError(
                    f"ComfyUI cannot find {filename!r} for {node}.{field}; "
                    f"available: {choices}")

        print(f"Using ComfyUI Mage-Flow-Edit backend: {self.base_url}")
        print(f"  diffusion model: {self.unet_name}")
        print(f"  text encoder: {self.clip_name}")
        print(f"  VAE: {self.vae_name}")

    def _build_prompt(
        self,
        references: Iterable[str],
        prompt: str,
        negative_prompt: str,
        seed: int,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        sampler: str,
        scheduler: str,
    ) -> dict[str, dict[str, Any]]:
        image_names = list(references)[:16]
        if not image_names:
            raise ValueError('Mage-Flow-Edit requires at least one reference image')

        graph: dict[str, dict[str, Any]] = {
            '1': {'class_type': 'UNETLoader', 'inputs': {
                'unet_name': self.unet_name, 'weight_dtype': 'default'}},
            '2': {'class_type': 'CLIPLoader', 'inputs': {
                'clip_name': self.clip_name,
                'type': 'mage',
                'device': 'default',
            }},
            '3': {'class_type': 'VAELoader', 'inputs': {
                'vae_name': self.vae_name}},
        }
        image_inputs: dict[str, list[Any]] = {}
        for index, image_name in enumerate(image_names, start=1):
            node_id = str(9 + index)
            graph[node_id] = {
                'class_type': 'LoadImage', 'inputs': {'image': image_name}}
            image_inputs[f'images.image_{index}'] = [node_id, 0]

        graph.update({
            '30': {'class_type': 'TextEncodeMageFlowEdit', 'inputs': {
                'clip': ['2', 0],
                'vae': ['3', 0],
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'width': int(width),
                'height': int(height),
                'batch_size': 1,
                **image_inputs,
            }},
            '35': {'class_type': 'KSampler', 'inputs': {
                'model': ['1', 0],
                'positive': ['30', 0],
                'negative': ['30', 1],
                'latent_image': ['30', 2],
                'seed': int(seed) & ((1 << 64) - 1),
                'steps': int(steps),
                'cfg': float(guidance_scale),
                'sampler_name': sampler,
                'scheduler': scheduler,
                'denoise': 1.0,
            }},
            '36': {'class_type': 'VAEDecode', 'inputs': {
                'samples': ['35', 0], 'vae': ['3', 0]}},
            '37': {'class_type': 'PreviewImage', 'inputs': {
                'images': ['36', 0]}},
        })
        return graph

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
        mage = config.mage
        preset = mage_model_defaults(config)
        references = (
            [image.convert('RGB') for image in reference_images[:16]]
            if reference_images else [edit_image.convert('RGB')])
        prompt = reference_prompt(
            prompt, mage.multi_reference_instruction, len(references))
        uploaded = []
        for index, reference in enumerate(references, start=1):
            save_debug_reference(reference, debug_dir, 'mage', index, frame_num)
            uploaded.append(self._upload_image(reference, index))
        graph = self._build_prompt(
            uploaded, prompt, negative_prompt, seed, width, height,
            int(preset['steps']), float(preset['guidance_scale']),
            mage.sampler, mage.scheduler)
        progress = ComfyProgressMonitor(
            getattr(self, 'base_url', ''), self.client_id, graph)
        progress.start()
        try:
            result = self._request('POST', '/prompt', json={
                'prompt': graph, 'client_id': self.client_id}).json()
            prompt_id = result.get('prompt_id')
            if not prompt_id:
                raise RuntimeError(f"ComfyUI returned no prompt_id: {result}")
            progress.set_prompt_id(prompt_id)
            self._active_progress = progress
            return self._wait_for_output(prompt_id)
        finally:
            progress.close()
            self._active_progress = None
