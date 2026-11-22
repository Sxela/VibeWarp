"""Native ComfyUI still-image graphs: FLUX.3 Base, MOSS and MiniMax H3 Image."""
import uuid

import requests

from vibewarp.config import comfy_image_defaults
from vibewarp.comfy_inventory import diffusion_loader, diffusion_model_category, model_specs
from vibewarp.core.flux_comfy import ComfyFluxClient, ComfyFluxError
from vibewarp.core.comfy_progress import ComfyProgressMonitor
from vibewarp.core.edit_references import reference_prompt, save_debug_reference


class ComfyImageClient(ComfyFluxClient):
    output_node_id = '32'

    def __init__(self, config):
        self.config = config
        section = config.comfy_image
        self.base_url = section.comfy_server_url.rstrip('/')
        self.timeout = float(section.comfy_timeout)
        self.session = requests.Session()
        self.client_id = f'vibewarp-image-{uuid.uuid4()}'
        self.upload_subfolder = f'vibewarp/{self.client_id}'
        self._validate_server()

    def _validate_server(self):
        self._request('GET', '/system_stats')
        graph = self._build_prompt(['reference.png'] if self.config.comfy_image.mode == 'edit' else [],
                                   'preflight', '', 0, 1024, 1024)
        for node in sorted({entry['class_type'] for entry in graph.values()}):
            info = self._request('GET', f'/object_info/{node}').json().get(node)
            if not info:
                if node == 'UnetLoaderGGUF':
                    raise ComfyFluxError('ComfyUI lacks UnetLoaderGGUF. Install ComfyUI-GGUF and restart ComfyUI.')
                raise ComfyFluxError(f'ComfyUI lacks {node}; update and restart ComfyUI for {self.config.model_version}')
            if node == 'CLIPLoader':
                types = info.get('input', {}).get('required', {}).get('type', [[]])[0]
                clip_type = graph['2']['inputs']['type']
                if clip_type not in types:
                    raise ComfyFluxError(f'ComfyUI lacks CLIPLoader type {clip_type}; update and restart ComfyUI')
        _, specs = model_specs(self.config)
        inventories = {}
        self._resolved_names = {}
        for category, filename in specs:
            if category not in inventories:
                data = self._request('GET', f'/models/{category}').json()
                if not isinstance(data, list) or any(not isinstance(name, str) for name in data):
                    raise ComfyFluxError(f'Invalid ComfyUI model listing for {category}')
                inventories[category] = {name.replace('\\', '/'): name for name in data}
            if filename.replace('\\', '/') not in inventories[category]:
                raise ComfyFluxError(f'ComfyUI cannot find {category}/{filename}; choose an available model or download it in Model setup')
            self._resolved_names[category, filename.replace('\\', '/')] = inventories[category][filename.replace('\\', '/')]

    def _build_prompt(self, references, prompt, negative_prompt, seed, width, height):
        config = self.config
        ci = config.comfy_image
        preset = comfy_image_defaults(config)
        unet_category = diffusion_model_category(preset['comfy_unet_name'])
        for category, key in [(unet_category, 'comfy_unet_name'), ('text_encoders', 'comfy_clip_name'),
                              ('vae', 'comfy_vae_name'), ('loras', 'comfy_edit_lora_name')]:
            preset[key] = getattr(self, '_resolved_names', {}).get((category, str(preset[key]).replace('\\', '/')), preset[key])
        version = config.model_version
        editing = ci.mode == 'edit'
        references = list(references) if editing else []
        if editing and not references:
            raise ValueError('Image edit mode requires a reference image')
        if len(references) > 5:
            raise ValueError('At most five references are supported')
        clip_type = {'flux3_klein': 'flux3', 'moss_image': 'moss_image', 'minimax_h3_image': 'minimax'}[version]
        graph = {
            '1': diffusion_loader(preset['comfy_unet_name']),
            '2': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': preset['comfy_clip_name'], 'type': clip_type, 'device': 'default'}},
            '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': preset['comfy_vae_name']}},
        }
        images = {}
        for index, name in enumerate(references):
            node = str(10+index)
            graph[node] = {'class_type': 'LoadImage', 'inputs': {'image': name}}
            images[f'images.image_{index+1}'] = [node, 0]
        model = ['1', 0]
        latent = ['6', 0]
        positive, negative = ['4', 0], ['5', 0]
        if version == 'minimax_h3_image':
            graph['4'] = {'class_type': 'MiniMaxH3ImageConditioning', 'inputs': {
                'clip': ['2', 0], 'vae': ['3', 0], 'prompt': prompt,
                'width': width, 'height': height, **images}}
            graph['5'] = {'class_type': 'ConditioningZeroOut', 'inputs': {'conditioning': positive}}
            latent = ['4', 1]
        else:
            graph['6'] = {'class_type': 'EmptyFlux2LatentImage', 'inputs': {'width': width, 'height': height, 'batch_size': 1}}
            for node, text in [('4', prompt), ('5', negative_prompt)]:
                graph[node] = {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['2', 0], 'text': text}}
            if version == 'flux3_klein':
                graph['20'] = {'class_type': 'ModelSamplingFlux', 'inputs': {
                    'model': model, 'max_shift': ci.sampling_shift, 'base_shift': ci.sampling_shift,
                    'width': width, 'height': height}}
                model = ['20', 0]
                if editing:
                    graph['21'] = {'class_type': 'ReferenceMultiLatent', 'inputs': {
                        'positive': positive, 'negative': negative, 'vae': ['3', 0],
                        'resolution': 'resize', 'resolution.megapixels': 1.0,
                        **{f'images.image{i}': [str(10+i), 0] for i in range(len(references))}}}
                    positive, negative = ['21', 0], ['21', 1]
            elif version == 'moss_image':
                if editing:
                    graph['20'] = {'class_type': 'LoraLoaderModelOnly', 'inputs': {
                        'model': model, 'lora_name': preset['comfy_edit_lora_name'], 'strength_model': ci.edit_lora_strength}}
                    model = ['20', 0]
                    for node, text in [('4', prompt), ('5', negative_prompt)]:
                        graph[node] = {'class_type': 'TextEncodeMOSSImageEdit', 'inputs': {
                            'clip': ['2', 0], 'vae': ['3', 0], 'prompt': text, 'edit_format': 'Moss-Image-Edit',
                            'image_strength': 1.0,
                            **{f'images.image_{i}': [str(10+i), 0] for i in range(len(references))}}}
                graph['22'] = {'class_type': 'SkipLayerGuidanceDiTSimple', 'inputs': {
                    'model': model, 'double_layers': '', 'single_layers': '22',
                    'start_percent': 0.0, 'end_percent': 1.0 if editing else 0.5}}
                model = ['22', 0]
        graph['30'] = {'class_type': 'KSampler', 'inputs': {
            'model': model, 'positive': positive, 'negative': negative, 'latent_image': latent,
            'seed': int(seed) & ((1 << 64)-1), 'steps': int(preset['steps']), 'cfg': float(preset['guidance_scale']),
            'sampler_name': ci.sampler, 'scheduler': preset['scheduler'], 'denoise': 1.0}}
        graph['31'] = {'class_type': 'VAEDecode', 'inputs': {'samples': ['30', 0], 'vae': ['3', 0]}}
        graph[self.output_node_id] = {'class_type': 'PreviewImage', 'inputs': {'images': ['31', 0]}}
        return graph

    def render(self, config, edit_image, prompt, negative_prompt, seed, width, height,
               reference_images=None, debug_dir=None, frame_num=None, temporal_frame=None):
        ci = config.comfy_image
        references = (reference_images or [edit_image]) if ci.mode == 'edit' else []
        uploaded = []
        for index, image in enumerate(references, 1):
            save_debug_reference(image, debug_dir, 'comfy_image', index, frame_num)
            uploaded.append(self._upload_image(image, index))
        if references:
            prompt = reference_prompt(prompt, ci.multi_reference_instruction, len(references))
        graph = self._build_prompt(uploaded, prompt, negative_prompt, seed, width, height)
        if temporal_frame is not None:
            temporal_frame.prepare(self, graph, config)
        monitor = ComfyProgressMonitor(self.base_url, self.client_id, graph)
        monitor.start()
        try:
            response = self._request('POST', '/prompt', json={'prompt': graph, 'client_id': self.client_id}).json()
            prompt_id = response.get('prompt_id')
            if not prompt_id: raise ComfyFluxError(f'ComfyUI returned no prompt_id: {response}')
            monitor.set_prompt_id(prompt_id)
            self._active_progress = monitor
            image = self._wait_for_output(prompt_id)
            if temporal_frame is not None:
                temporal_frame.record(self, prompt_id, config)
            return image
        finally:
            monitor.close()
            self._active_progress = None
