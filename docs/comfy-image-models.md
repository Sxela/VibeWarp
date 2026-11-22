# FLUX.3 Image Klein, MOSS-Image and MiniMax H3 Image

These three model choices use native ComfyUI nodes through the HTTP backend.
MOSS edit graphs use zero-based reference input names on both positive and negative
conditioning nodes; MiniMax uses one-based names. These conventions are independent
of the visible reference labels.
GGUF diffusion files use the GGUF loader's `unet_gguf` API listing, which includes
`.gguf` files filtered out of the standard `diffusion_models` listing. Its reported
folders also supply the local download destination. A compatible GGUF diffusion
model requires the `UnetLoaderGGUF` node from ComfyUI-GGUF.
ComfyUI owns the model loading and execution; VibeWarp retains frame ranges,
warping, consistency, masking, color matching, progress, cancellation and history.

Choose the model in the navbar. **Prompts & sampling → Comfy Image → Generation
mode** switches between `edit` and `text`. Edit uses the ordered reference
images; text submits no image conditioning and does not upload references.
Text mode still runs within the video/frame-range workflow. It does not add a
separate standalone image-generation page. Hidden edit settings are retained
when switching modes.

The main style/edit prompt is under **Prompts & sampling → Prompts**. In edit
mode, **Conditioning → Reference-role instructions (2+ images)** appears above
the image cards and can explain each reference's role. It is prepended to the
main prompt only when at least two images are sent. For a single reference,
put the editing instruction directly in the main prompt.

Model-file dropdown choices are remembered per model in this browser. Switching to
a model for the first time uses its suggested files; returning restores its last
selections, including after a page reload. Loaded project settings remain authoritative
for the active model. Custom FLUX encoders therefore do not carry into MiniMax.

Starting the UI with a preselected FLUX.2 9B, Qwen GGUF or Mage Turbo configuration
loads that variant's suggested files before applying any saved machine settings.

**Conditioning → Tag size** scales the baked `@ImageN` text from 0.5x to 4x.
**Badge height** controls its footer band from 4% to 30% of image height
(default 12%); label opacity remains independent. These controls change the
actual reference pixels uploaded to ComfyUI, not just the browser preview.
Text scales with reference resolution and fits inside the badge. The font
loader uses a scalable embedded fallback if installed bold fonts are unavailable,
so missing system fonts cannot reduce every tag to fixed-size tiny text.

Switching from Disco retains its hidden canvas/visible-size settings for when
you switch back. Those settings are ignored for validation and video cropping
while a different model is selected.

| Model | Diffusion filename | Text encoder | VAE | Default sampling |
|---|---|---|---|---|
| FLUX.3 Image Klein Base | `flux3-image-klein-base-int8-convrot.safetensors` | `qwen3vl_4b_int8_convrot.safetensors` | `flux2-vae.safetensors` | 50 steps, CFG 4, Euler/simple; shift log(16) |
| MOSS-Image SFT | `moss_image_sft_int8_convrot.safetensors` | `qwen3.5_9b_w4a8_gptq.safetensors` | `flux2-vae_small.safetensors` | 30 steps, CFG 4, Euler/normal |
| MiniMax H3 Image | `minimax_h3_image_pruned_w4a8.safetensors` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `minimax_h3_image_vae_fp16.safetensors` | 6 steps, CFG 1, Euler/simple |

Sampling controls remain editable. Switching models in the UI applies the
selected preset to recognized preset values. Loading a saved run preserves
explicit values, including values shared with another model's preset.
MOSS edit mode additionally loads
`moss_image_edit_lora_rank_64_bf16.safetensors` at strength 1. Text mode omits the
LoRA. The native MOSS workflow uses skip-layer guidance on single layer 22.
FLUX.3 uses `ReferenceMultiLatent`; MiniMax uses
`MiniMaxH3ImageConditioning`, with zeroed negative conditioning as in its native
template. MiniMax therefore hides the negative-prompt editor. Automatic prompt
expansion is not included; your prompt goes directly to conditioning.

Up to five reference images are supported. FLUX.3/MOSS dimensions must be
divisible by 16; MiniMax dimensions by 32. With **Max size** enabled, the
source aspect ratio determines the canvas and each dimension is rounded to
the model's required multiple. The existing shared contact-sheet
mode is not enabled for these new backends.

## Server setup and availability

Use a ComfyUI version that supplies `CLIPLoader` types `flux3`, `moss_image` and
`minimax`, plus the conditioning nodes above and `TextEncodeMOSSImageEdit`.
An older running server can list the weight files while still lacking those
nodes. Backend preflight fails with an update/restart message before submitting
a prompt. VibeWarp does not automatically update or restart ComfyUI.

Model fields are filenames relative to ComfyUI's model categories, including
subfolders if needed. Do not enter a VibeWarp-local absolute path. Availability
for **all ComfyUI-managed families** comes from the server's
`GET /models/<category>` API, including checkpoints, diffusion models, text
encoders, VAEs and enabled LoRAs. Existing remote files are recognized without
mounting ComfyUI's model folder. An unreachable/invalid listing is shown as
unavailable rather than declaring the weights missing.

Each filename field is a dropdown populated from its ComfyUI category. The
suggested model is always first, including when missing. Other reported models
sort by descending shared-prefix length with the suggestion (both lowercased),
then alphabetically. Selecting an alternative preserves that exact filename;
it is not replaced by the suggestion on submission. Imported missing selections
also remain visible rather than silently changing to another file.

For a local loopback ComfyUI, VibeWarp discovers download folders from its
`GET /experiment/models` API. **System settings → Comfy models directory**
overrides this discovery or supplies a shared mount for a remote ComfyUI.
It is not needed for presence checks. **Download required models** downloads missing catalogued
weights into their category subfolders. Files already reported by ComfyUI are
not downloaded again. Gated model sources may require accepting the publisher's
terms and installing the files manually. Custom files reported by the server
appear in the dropdown. Missing names imported from existing settings remain
visible, but their download sources are not guessed.

Sources and graph references:

- [BFL FLUX.3 Comfy weights](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy)
- [ComfyUI FLUX.3 edit template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_flux3_klein_distilled_image_edit.json)
- [MOSS edit template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_moss_image_edit.json)
- [MiniMax H3 Image edit template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_edit.json)

The graph and HTTP behavior have CPU/unit coverage. Output retrieval uses each
backend's declared preview node, and reports the prompt ID and returned nodes
if a completed job has no expected image. FLUX.3 output retrieval has also been
checked against an existing successful 720x1280 ComfyUI render. MOSS and MiniMax
have not been inference-verified against the newly added weights in this checkout.

The workflow audit covers all 13 Comfy variants/modes: the six FLUX.3/MOSS/
MiniMax text/edit graphs plus FLUX.2 4B/9B, Qwen INT8/GGUF, Mage standard/Turbo,
and HiDream. Each has a submit/history/image-fetch regression using its generated
preview node. Their required inputs and connected output types were also checked
against the running server's node schemas without submitting live render jobs.
