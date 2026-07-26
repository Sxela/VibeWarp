"""ComfyUI HTTP backend for Qwen-Image-Edit-2511."""

from __future__ import annotations

import uuid
from typing import Any, Iterable

import requests
from PIL import Image

from vibewarp.core.comfy_progress import ComfyProgressMonitor
from vibewarp.core.edit_references import reference_prompt, save_debug_reference
from vibewarp.core.flux_comfy import ComfyFluxClient


class ComfyQwenEditClient(ComfyFluxClient):
    """Synchronous client for ComfyUI's native Qwen Image Edit workflow."""

    def __init__(self, config: Any):
        qwen = config.qwen
        self.base_url = qwen.comfy_server_url.rstrip('/')
        self.timeout = float(qwen.comfy_timeout)
        self.unet_name = qwen.comfy_unet_name
        self.clip_name = qwen.comfy_clip_name
        self.vae_name = qwen.comfy_vae_name
        self.lora_name = qwen.comfy_lora_name
        self.use_lightning_lora = bool(qwen.use_lightning_lora)
        self.session = requests.Session()
        self.client_id = f"vibewarp-qwen-{uuid.uuid4()}"
        self.upload_subfolder = f"vibewarp/{self.client_id}"
        self._validate_server()

    def _unet_loader(self) -> tuple[str, dict[str, Any]]:
        if self.unet_name.lower().endswith('.gguf'):
            return 'UnetLoaderGGUF', {'unet_name': self.unet_name}
        return 'UNETLoader', {
            'unet_name': self.unet_name, 'weight_dtype': 'default'}

    def _validate_server(self) -> None:
        self._request('GET', '/system_stats')
        unet_loader, _ = self._unet_loader()
        expected = {
            unet_loader: ('unet_name', self.unet_name),
            'CLIPLoader': ('clip_name', self.clip_name),
            'VAELoader': ('vae_name', self.vae_name),
            'LoadImage': None,
            'TextEncodeQwenImageEditPlus': None,
            'FluxKontextImageScale': None,
            'FluxKontextMultiReferenceLatentMethod': None,
            'ModelSamplingAuraFlow': None,
            'CFGNorm': None,
            'VAEEncode': None,
            'KSampler': None,
            'VAEDecode': None,
            'PreviewImage': None,
        }
        if self.use_lightning_lora:
            expected['LoraLoaderModelOnly'] = ('lora_name', self.lora_name)
        for node, model in expected.items():
            info = self._request('GET', f'/object_info/{node}').json().get(node)
            if not info:
                if node == 'UnetLoaderGGUF':
                    raise RuntimeError(
                        f"ComfyUI at {self.base_url} does not provide required "
                        "node UnetLoaderGGUF; install ComfyUI-GGUF and restart "
                        "ComfyUI before using the Qwen GGUF mode")
                raise RuntimeError(
                    f"ComfyUI at {self.base_url} does not provide required node "
                    f"{node}; update ComfyUI before using Qwen Image Edit 2511")
            if model is None:
                continue
            field, filename = model
            choices = info.get('input', {}).get('required', {}).get(field, [[]])[0]
            if filename not in choices:
                raise RuntimeError(
                    f"ComfyUI cannot find {filename!r} for {node}.{field}; "
                    f"available: {choices}")

        print(f"Using ComfyUI Qwen Image Edit backend: {self.base_url}")
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
        sampling_shift: float,
        use_lightning_lora: bool,
        lora_strength: float,
    ) -> dict[str, dict[str, Any]]:
        del width, height  # Qwen derives the latent size from reference image 1.
        image_names = list(references)[:3]
        if not image_names:
            raise ValueError('Qwen Image Edit requires at least one reference image')

        unet_loader, unet_inputs = self._unet_loader()
        graph: dict[str, dict[str, Any]] = {
            '1': {'class_type': unet_loader, 'inputs': unet_inputs},
            '2': {'class_type': 'CLIPLoader', 'inputs': {
                'clip_name': self.clip_name,
                'type': 'qwen_image',
                'device': 'default',
            }},
            '3': {'class_type': 'VAELoader', 'inputs': {
                'vae_name': self.vae_name}},
        }
        image_links: list[list[Any]] = []
        for index, image_name in enumerate(image_names, start=1):
            node_id = str(9 + index)
            graph[node_id] = {
                'class_type': 'LoadImage', 'inputs': {'image': image_name}}
            image_links.append([node_id, 0])

        conditioning_images = {
            f'image{index}': link
            for index, link in enumerate(image_links, start=1)
        }
        # The official workflow feeds the preferred-resolution image into both
        # Qwen encoders as image1 as well as into the starting latent.
        conditioning_images['image1'] = ['20', 0]
        graph.update({
            '20': {'class_type': 'FluxKontextImageScale', 'inputs': {
                'image': image_links[0]}},
            '21': {'class_type': 'VAEEncode', 'inputs': {
                'pixels': ['20', 0], 'vae': ['3', 0]}},
            '30': {'class_type': 'TextEncodeQwenImageEditPlus', 'inputs': {
                'clip': ['2', 0], 'vae': ['3', 0], 'prompt': prompt,
                **conditioning_images,
            }},
            '31': {'class_type': 'TextEncodeQwenImageEditPlus', 'inputs': {
                'clip': ['2', 0], 'vae': ['3', 0],
                'prompt': negative_prompt,
                **conditioning_images,
            }},
            '32': {'class_type': 'FluxKontextMultiReferenceLatentMethod',
                   'inputs': {
                       'conditioning': ['30', 0],
                       'reference_latents_method': 'index_timestep_zero',
                   }},
            '33': {'class_type': 'FluxKontextMultiReferenceLatentMethod',
                   'inputs': {
                       'conditioning': ['31', 0],
                       'reference_latents_method': 'index_timestep_zero',
                   }},
            '34': {'class_type': 'ModelSamplingAuraFlow', 'inputs': {
                'model': ['1', 0], 'shift': float(sampling_shift)}},
            '38': {'class_type': 'CFGNorm', 'inputs': {
                'model': ['34', 0], 'strength': 1.0, 'pre_cfg': False}},
            '35': {'class_type': 'KSampler', 'inputs': {
                'model': ['39', 0] if use_lightning_lora else ['38', 0],
                'positive': ['32', 0],
                'negative': ['33', 0],
                'latent_image': ['21', 0],
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
        if use_lightning_lora:
            graph['39'] = {'class_type': 'LoraLoaderModelOnly', 'inputs': {
                'model': ['38', 0],
                'lora_name': self.lora_name,
                'strength_model': float(lora_strength),
            }}
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
        qwen = config.qwen
        references = ([image.convert('RGB') for image in reference_images[:3]]
                      if reference_images else [edit_image.convert('RGB')])
        prompt = reference_prompt(
            prompt, qwen.multi_reference_instruction, len(references))
        uploaded = []
        for index, reference in enumerate(references, start=1):
            save_debug_reference(reference, debug_dir, 'qwen', index, frame_num)
            uploaded.append(self._upload_image(reference, index))
        graph = self._build_prompt(
            uploaded, prompt, negative_prompt, seed, width, height,
            qwen.steps, qwen.guidance_scale, qwen.sampler, qwen.scheduler,
            qwen.sampling_shift, qwen.use_lightning_lora,
            qwen.lora_strength)
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
