# Settings reachable only through All settings

Schema audit dated 2026-11-15, after the mask workflow routing update: **38 editable fields** across all model families. Model and sampler-mode filters still apply; these do not all appear for every selected model.

This means no dedicated workflow in the backend schema (`workflows == ["all"]`). Search can still find them. Hidden compatibility fields are excluded. System settings and fields shared between multiple named workflows are excluded.

Before this change there were 39 such fields. `disco.inpaint_mask_sigma` now belongs to Prompts & sampling / Disco inpainting masks. The remaining fields below have been inventoried; their proposed destinations have not been applied.

| Category | Fields | Suggested destination |
| --- | ---: | --- |
| Advanced SD sampling | 12 | Prompts & sampling, in an Advanced sampling group. |
| Captions | 8 | A Captions workflow, or a Captions group within Prompts & sampling. |
| FreeU | 6 | Conditioning or Prompts & sampling, with its own FreeU group. |
| HiDream-O1 advanced sampling and seams | 7 | Prompts & sampling. The existing HiDream-O1 group currently lacks a workflow mapping. |
| Disco execution and model identity | 5 | Sampling/seed options in Prompts & sampling; LoRA merge and checksums in Model. Consider making derived checksums read-only. |

## Advanced SD sampling (12)

Compatibility: sd.

| Config field | Display label |
| --- | --- |
| `diffusion.use_karras_noise` | Use Karras Noise |
| `diffusion.clip_final_layer_norm` | Clip Final Layer Norm |
| `diffusion.clamp_grad` | Clamp guidance gradient |
| `diffusion.clamp_max` | Maximum gradient RMS |
| `diffusion.guidance_add_noise` | Noise latent guidance target |
| `diffusion.dynamic_thresh` | Dynamic Thresh |
| `diffusion.code_randomness` | Code Randomness |
| `diffusion.do_softcap` | Do Softcap |
| `diffusion.softcap_thresh` | Softcap Thresh |
| `diffusion.softcap_q` | Softcap Q |
| `diffusion.noise_mode` | Noise Mode |
| `diffusion.guidance_use_start_code` | Guidance Use Start Code |

## Captions (8)

Compatibility: all.

| Config field | Display label |
| --- | --- |
| `captions.make_captions` | Make Captions |
| `captions.keyframe_source` | Keyframe Source |
| `captions.nth_frame` | Nth Frame |
| `captions.user_defined_keyframes` | User Defined Keyframes |
| `captions.diff_thresh` | Diff Thresh |
| `captions.offset_mode` | Offset Mode |
| `captions.fixed_offset` | Fixed Offset |
| `captions.blip_question` | Blip Question |

## FreeU (6)

Compatibility: sd.

| Config field | Display label |
| --- | --- |
| `freeu.do_freeunet` | Do Freeunet |
| `freeu.apply_freeu_after_control` | Apply Freeu After Control |
| `freeu.b1` | B1 |
| `freeu.b2` | B2 |
| `freeu.s1` | S1 |
| `freeu.s2` | S2 |

## HiDream-O1 advanced sampling and seams (7)

Compatibility: hidream.

| Config field | Display label |
| --- | --- |
| `hidream.sampler` | Sampler |
| `hidream.scheduler` | Scheduler |
| `hidream.noise_scale` | Noise Scale |
| `hidream.patch_seam_smoothing` | Patch Seam Smoothing |
| `hidream.patch_seam_start` | Patch Seam Start |
| `hidream.patch_seam_passes` | Patch Seam Passes |
| `hidream.patch_seam_blend` | Patch Seam Blend |

## Disco execution and model identity (5)

Compatibility: disco.

| Config field | Display label |
| --- | --- |
| `disco.timestep_grid` | Timestep Grid |
| `disco.notebook_rng` | Notebook Rng |
| `disco.merge_loras` | Merge Loras |

## Mask workflow destinations

Mask & tint contains only RVM/background masking, tint and RVM file overrides.

| Controls | Workflow / group |
| --- | --- |
| Video output masks | Video export / Video Output |
| Flow consistency and Turbo consistency masks | Frame consistency / Flow & Consistency |
| LPIPS/image-space frame guidance, secondary model and guidance masks | Guidance / General |
| Disco inpainting masks, including sigma | Prompts & sampling / Disco inpainting masks |
| ControlNet/IP-Adapter masks | Conditioning / their original custom editors |
| Scene-mask scheduling | Frame consistency / Scene Scheduling |
| Raw-input color matching | Color match / Raw input color match |

RVM file paths retain their machine-local storage tier and appear in Mask & tint
and System settings. Other functional mask controls have a single workflow home.
See the [duplicate-settings audit](settings-duplicates.md) for the remaining
editable overlaps outside All settings.
