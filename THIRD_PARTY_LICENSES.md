# Third-party code and licenses

VibeWarp is licensed **GPL-3.0** (see LICENSE). It vendors or ports code from
the projects below; each retains its original license and copyright.

## Why GPL-3.0 (and not AGPL-3.0 or a permissive license)

VibeWarp was initially published under AGPL-3.0 out of caution and relicensed
to GPL-3.0 on 2026-07-10 after auditing the license of every vendored source:

- **GPL-3.0 is the floor.** Several vendored components are GPL-3.0
  (`sample_lcm` ported from ComfyUI; IP-Adapter hook logic derived from
  cubiq/ComfyUI_IPAdapter_plus; AnimateDiff detection/normalization helpers
  merged from Kosinkadink/ComfyUI-AnimateDiff-Evolved). Any license more
  permissive than GPL-3.0 was therefore never an option.
- **No AGPL-3.0 code is included.** AGPL would only be forced by code taken from
  AUTOMATIC1111 stable-diffusion-webui itself, and VibeWarp contains none of it —
  no `modules.*` / `shared.*` / `devices.*` imports, no hijack, sampler or
  processing code from webui core.

  The one file with any webui lineage is `vendor/controlmodel_ipadapter.py`. Its
  upstream is [Mikubill/sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet)
  (`scripts/controlmodel_ipadapter.py`), which is a **separate repository licensed
  GPL-3.0** — a webui *extension*, not webui itself. Its model classes in turn come
  from tencent-ailab/IP-Adapter (Apache-2.0) and cubiq/ComfyUI_IPAdapter_plus
  (GPL-3.0). It reached VibeWarp via the author's own fork
  (Sxela/sxela-stablediffusion, AGPL as a whole because of its webui lineage), but
  transiting an AGPL repository does not relicense a GPL-3.0 file, and the author
  holds the copyright on his own changes and licenses them here under GPL-3.0.

  Elsewhere in the codebase, "A1111" appears only in comments describing *behaviour*
  we match (clip-skip applying CLIP's final layer norm; the "ControlNet is more
  important" soft-weighting curve; fp16 LORA merge arithmetic) or a *file format*
  we read (A1111-style LORA key naming). No code was copied for any of them.
- Everything else vendored is MIT (the WarpFusion notebook itself — © Alex
  Spirin — ldm, sgm, k-diffusion, resize_right, python-color-transfer) or
  Apache-2.0 (ControlNet/cldm, guoyww AnimateDiff, detectron2 palette data),
  all GPL-3.0-compatible.

| Vendored code | Source | License |
|---|---|---|
| Core pipeline logic (ported) | [Sxela/WarpFusion](https://github.com/Sxela/WarpFusion) notebook v0.37 | MIT © Alex Spirin |
| `vendor/ldm/` | [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion) (via Sxela fork) | MIT |
| `vendor/sgm/` | [Stability-AI/generative-models](https://github.com/Stability-AI/generative-models) (via Sxela fork) | MIT |
| `vendor/k_diffusion/` | [crowsonkb/k-diffusion](https://github.com/crowsonkb/k-diffusion) (via Sxela fork) | MIT |
| `vendor/k_diffusion/sampling.py::sample_lcm` | [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI) (via Sxela fork) | GPL-3.0 |
| `vendor/cldm/` | [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet) | Apache-2.0 |
| `vendor/resize_right/` | [assafshocher/ResizeRight](https://github.com/assafshocher/ResizeRight) | MIT |
| `vendor/python_color_transfer/` | [pengbo-learn/python-color-transfer](https://github.com/pengbo-learn/python-color-transfer) | MIT |
| `vendor/controlmodel_ipadapter.py` | [Mikubill/sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) (GPL-3.0), whose model classes come from [tencent-ailab/IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) (Apache-2.0) and [cubiq/ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus) (GPL-3.0); reached VibeWarp via [Sxela/sxela-stablediffusion](https://github.com/Sxela/sxela-stablediffusion) | Apache-2.0 + GPL-3.0 components |
| `vendor/animatediff_mm.py` | reimplemented from [guoyww/AnimateDiff](https://github.com/guoyww/AnimateDiff) (Apache-2.0) and [Kosinkadink/ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved) (GPL-3.0), with original attention and block-container code | Apache-2.0 + GPL-3.0 components |
| `vendor/seg_palettes.py` | palette data from [SHI-Labs/OneFormer](https://github.com/SHI-Labs/OneFormer) (MIT) dataset registrations / [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2) (Apache-2.0) | MIT / Apache-2.0 |

Runtime PyPI dependencies (torch, transformers, controlnet-aux, open-clip-torch,
etc.) are not vendored and carry their own licenses.
