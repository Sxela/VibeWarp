# Settings with multiple editable locations

Audit dated 2026-11-15. All settings and search results are excluded, as are
read-only summaries and actions. Counts span every model family; model filters
mean these fields are not all visible simultaneously. Each overlap edits the same
stored setting, rather than creating an independent copy.

There are **24 schema fields shared between named workflows**, plus ControlNet's
custom model-directory editor (**25 cross-workflow fields**). Including the
editable project title gives **26 settings with multiple editable entry points**.

## Model and System settings (22)

These machine-local connection/model settings remain available during model setup.
Their storage tier is still System, so imported project settings cannot overwrite them.

| Config field | Locations |
| --- | --- |
| `comfy_models_dir` | Model; System settings |
| `flux.backend` | Model; System settings |
| `flux.comfy_server_url` | Model; System settings |
| `flux.comfy_unet_name` | Model; System settings |
| `flux.comfy_clip_name` | Model; System settings |
| `flux.comfy_vae_name` | Model; System settings |
| `flux.comfy_timeout` | Model; System settings |
| `flux.model_repo` | Model; System settings |
| `hidream.comfy_server_url` | Model; System settings |
| `hidream.comfy_checkpoint_name` | Model; System settings |
| `hidream.comfy_timeout` | Model; System settings |
| `qwen.comfy_server_url` | Model; System settings |
| `qwen.comfy_unet_name` | Model; System settings |
| `qwen.comfy_clip_name` | Model; System settings |
| `qwen.comfy_vae_name` | Model; System settings |
| `qwen.comfy_lora_name` | Model; System settings |
| `qwen.comfy_timeout` | Model; System settings |
| `mage.comfy_server_url` | Model; System settings |
| `mage.comfy_unet_name` | Model; System settings |
| `mage.comfy_clip_name` | Model; System settings |
| `mage.comfy_vae_name` | Model; System settings |
| `mage.comfy_timeout` | Model; System settings |

## Mask & tint and System settings (2)

| Config field | Locations |
| --- | --- |
| `mask.rvm_model_path` | Mask & tint / RVM files; System settings / Paths |
| `mask.rvm_cache_dir` | Mask & tint / RVM files; System settings / Paths |

## Additional editors outside the schema workflow list (2)

| Config field | Locations |
| --- | --- |
| `controlnet.model_dir` | Conditioning / ControlNet custom editor; System settings / Paths |
| `batch_name` | Editable header project title; Source & output / Output |

`model_version` uses the header picker; the generic field is suppressed.
Reference-image settings use their custom editor instead of a second generic
editor. Hidden scalar fallbacks do not create another visible schedule editor.

No functional mask controls remain duplicated between Mask & tint and other
render workflows. Only its two machine-local RVM file overrides remain shared.
The [All-settings-only audit](settings-workflow-audit.md) separately lists the 38
fields without a named workflow.
