"""FLUX.2 Klein Edit render path with ComfyUI and Diffusers backends.

This is deliberately separate from the CompVis / k-diffusion stack in
`core/model_loader.py` + `core/diffusion.py`. Flux2 is a flow-matching DiT with a
128-channel VAE latent and Qwen text encoder; none of the SD k-diffusion sigma
machinery, `CFGDenoiser`, or `wrap_model_for_kdiffusion` applies. Rather than
shoehorn it in, VibeWarp keeps the model-agnostic frame loop (warp the previous
render, save, colour-match) and branches here only for the actual denoise.

Warp stylization, unchanged from the SD path:

  * First frame  — edits the raw video frame (VibeWarp seeds `state.init_image`
    with the raw frame on frame 0).
  * Later frames — edit the flow-warped, consistency-weighted previous render
    (VibeWarp fills `state.init_image` with exactly that before calling us).
  * Optional      — the raw current frame is passed as a second reference.

The default backend submits the native reference-latent workflow to a running
ComfyUI server. The optional Diffusers backend resolves its pipeline class and
call signature at runtime. Its exact pipeline class and call signature
are confirmed against the installed library at runtime, not hardcoded: we load
via `DiffusionPipeline.from_pretrained` (which resolves the class the repo itself
declares in model_index.json) and pass only the keyword arguments the resolved
pipeline's `__call__` actually accepts (see `_supported_call_kwargs`). This keeps
the module working across diffusers versions and is the seam to adjust as the
model's real API is explored on a GPU box with the weights.

`diffusers` is an optional dependency (lazy-imported here) so the CPU test suite,
which never touches this path, does not require it.
"""

import inspect
from typing import Any, Optional

from PIL import Image

from vibewarp.config import RunConfig, flux_model_files
from vibewarp.core.edit_references import save_debug_reference


def _require_diffusers():
    """Import diffusers, or raise a message that says how to get it."""
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
        return diffusers
    except ImportError as e:  # pragma: no cover - exercised only without diffusers
        raise ImportError(
            "The Flux 2 Klein Edit model (model_version='flux2_klein_edit') needs "
            "the `diffusers` package, which is an optional dependency. Install it "
            "with `pip install vibewarp[flux]` (or `pip install diffusers`)."
        ) from e


def load_flux_pipeline(config: RunConfig, device: str = 'cuda') -> Any:
    """Load the Flux edit pipeline declared by the configured repo.

    `DiffusionPipeline.from_pretrained` reads the repo's model_index.json and
    instantiates whatever pipeline class it names, so we do not hardcode a class
    name that a post-release rename could invalidate.

    Args:
        config: run configuration (reads config.flux.model_repo).
        device: target device ('cuda' or 'cpu').

    Returns:
        A ready-to-call diffusers pipeline moved to `device`.
    """
    if config.flux.backend == 'comfy':
        from vibewarp.core.flux_comfy import ComfyFluxClient
        return ComfyFluxClient(config)

    diffusers = _require_diffusers()
    import torch

    from diffusers import DiffusionPipeline

    repo = flux_model_files(config)['model_repo']
    if not repo:
        raise ValueError(
            "flux.model_repo is empty — set it to the Flux 2 Klein Edit repo id "
            "or a local diffusers-format directory.")

    print(f"Loading Flux edit pipeline: {repo}")
    pipe = DiffusionPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
    pipe = pipe.to(device)
    # Trim VRAM: attention slicing is safe and universally supported; the fancier
    # offload hooks are left to the caller/exploration since they interact with
    # VibeWarp's own per-frame cleanup.
    if hasattr(pipe, 'set_progress_bar_config'):
        pipe.set_progress_bar_config(disable=True)
    print(f"  Flux pipeline ready: {type(pipe).__name__}")
    return pipe


def _supported_call_kwargs(pipe: Any, candidate: dict) -> dict:
    """Keep only the kwargs the pipeline's __call__ actually declares.

    Flux edit pipelines differ across diffusers versions in whether they expose
    `strength`, `max_sequence_length`, `image`/`image_2`, etc. Passing an unknown
    keyword raises; silently dropping a *needed* one would render wrongly. So we
    intersect with the real signature and log what we dropped, loudly enough to
    notice during exploration.
    """
    try:
        sig = inspect.signature(pipe.__call__)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return candidate
    params = sig.parameters
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return candidate
    supported = {k: v for k, v in candidate.items() if k in params}
    dropped = [k for k in candidate if k not in supported]
    if dropped:
        print(f"  Flux: pipeline {type(pipe).__name__} does not accept "
              f"{dropped}; not passing them. Adjust core/flux.py if one is needed.")
    return supported


def prepare_flux_references(
    config: RunConfig,
    edit_image: Image.Image,
    prompt: str,
    reference_images: Optional[list[Image.Image]] = None,
) -> tuple[list[Image.Image], str]:
    """Normalize references and add the multi-reference role instruction."""
    fx = config.flux
    references = ([image.convert('RGB') for image in reference_images]
                  if reference_images else [edit_image.convert('RGB')])
    from vibewarp.core.edit_references import reference_prompt
    return references, reference_prompt(
        prompt, fx.multi_reference_instruction, len(references))


def render_flux_frame(
    pipe: Any,
    config: RunConfig,
    edit_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    reference_images: Optional[list[Image.Image]] = None,
    debug_dir: Optional[str] = None,
    frame_num: Optional[int] = None,
) -> Image.Image:
    """Run one Flux edit step and return the rendered PIL image.

    Args:
        pipe: pipeline from load_flux_pipeline.
        config: run configuration (reads config.flux.*).
        edit_image: first ordered reference image.
        reference_images: all ordered references, including ``edit_image``.
        prompt: positive edit instruction / prompt for this frame.
        negative_prompt: negative prompt.
        seed: RNG seed for this frame.
        width, height: target render size.

    Returns:
        Rendered PIL.Image (RGB).
    """
    if hasattr(pipe, 'render'):
        return pipe.render(
            config=config,
            edit_image=edit_image,
            prompt=prompt,
            seed=seed,
            width=width,
            height=height,
            reference_images=reference_images,
            debug_dir=debug_dir,
            frame_num=frame_num,
        )

    import torch

    fx = config.flux
    device = getattr(pipe, 'device', None)
    generator = torch.Generator(device='cpu').manual_seed(int(seed))

    references, prompt = prepare_flux_references(
        config, edit_image, prompt, reference_images=reference_images)
    edit_image = references[0]

    # Candidate arguments; _supported_call_kwargs prunes to what the pipeline
    # declares. `image` is the edit target. When we have a structural reference
    # (raw frame) we offer it under the common Flux-edit spellings; the pipeline
    # keeps whichever it recognises.
    candidate = {
        'image': edit_image,
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'num_inference_steps': int(fx.steps),
        'guidance_scale': float(fx.guidance_scale),
        'strength': float(fx.denoise),
        'max_sequence_length': int(fx.max_sequence_length),
        'height': int(height),
        'width': int(width),
        'generator': generator,
    }

    extra_references = list(references[1:])

    all_references = [edit_image, *extra_references]
    for index, reference in enumerate(all_references, start=1):
        save_debug_reference(reference, debug_dir, 'flux', index, frame_num)

    for index, reference in enumerate(extra_references, start=2):
        candidate[f'image_{index}'] = reference
    if extra_references:
        # Some evolving Diffusers edit pipelines use a named single-reference
        # argument instead of image_2. The signature filter keeps the spelling
        # supported by the installed pipeline.
        candidate['reference_image'] = extra_references[0]
        candidate['control_image'] = extra_references[0]

    call_kwargs = _supported_call_kwargs(pipe, candidate)

    with torch.inference_mode():
        result = pipe(**call_kwargs)

    images = getattr(result, 'images', None)
    if images is None and isinstance(result, (list, tuple)):
        images = result
    if not images:
        raise RuntimeError(
            "Flux pipeline returned no images — inspect the pipeline's output "
            "type in core/flux.py.render_flux_frame.")
    image = images[0]
    if not isinstance(image, Image.Image):  # pragma: no cover - defensive
        raise TypeError(
            f"Flux pipeline returned {type(image)!r}, expected a PIL image.")
    return image.convert('RGB')
