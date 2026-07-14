"""Keyframe captioning (notebook cell 30 — 'Generate captions for keyframes').

Generates BLIP captions for selected keyframes and substitutes them into
prompts via the ``{caption}`` placeholder. The notebook used the BLIP git
clone; this implementation uses the transformers ports of the same
checkpoints:

- captioning: Salesforce/blip-image-captioning-base
  (model_base_caption_capfilt_large — ViT-B, capfilt-large)
- VQA (when ``blip_question`` is set): Salesforce/blip-vqa-base
  (model_base_vqa_capfilt_large)

Captions are written to ``{video_frames_folder}Captions/{frame:06d}.txt``
(same convention as the notebook) and looked up piecewise-constant per frame.
"""

import os
from glob import glob
from typing import Any, List, Optional

import torch
from PIL import Image


_blip_cache = {}


def select_keyframes(
    num_frames: int,
    source: str = 'Every n-th frame',
    nth_frame: int = 10,
    user_keyframes: Optional[List[int]] = None,
    diffs: Optional[List[float]] = None,
    diff_thresh: float = 0.33,
) -> Optional[List[int]]:
    """Select 1-based caption keyframes (notebook keyframe_source options).

    Returns None for 'Content-aware scheduling keyframes' without diffs
    (notebook prints an error and skips captioning).
    """
    if source == 'User-defined keyframe list':
        return list(user_keyframes or [])
    if source == 'Content-aware scheduling keyframes':
        if diffs in (None, '', []):
            print('ERROR: Keyframes were not generated. Enable content-aware '
                  'scheduling analysis or choose a different caption keyframe source.')
            return None
        return [1] + [i + 1 for i, o in enumerate(diffs) if o >= diff_thresh]
    # 'Every n-th frame'
    return list(range(1, num_frames, nth_frame))


def remap_keyframes(
    keyframes: List[int],
    offset_mode: str = 'Fixed',
    fixed_offset: int = 0,
) -> List[int]:
    """Remap keyframes per the notebook's offset_mode logic.

    'None' → unchanged; 'Fixed' → each keyframe (except the last) moves by
    +fixed_offset, clamped to the next keyframe; 'Between Keyframes' → each
    keyframe (except the last) moves to the midpoint toward the next.
    """
    if not keyframes or offset_mode == 'None':
        return list(keyframes)

    src = list(keyframes)
    out = list(keyframes)
    last = max(src)
    for i in range(len(src)):
        if src[i] >= last:
            out[i] = src[i]
        elif offset_mode == 'Fixed':
            out[i] = min(src[i] + fixed_offset, src[i + 1])
        elif offset_mode == 'Between Keyframes':
            out[i] = src[i] + int((src[i + 1] - src[i]) / 2)
    return out


def _load_blip(vqa: bool, device: str = 'cuda'):
    """Load (and cache) the transformers BLIP captioning or VQA model."""
    key = 'vqa' if vqa else 'caption'
    if key not in _blip_cache:
        if vqa:
            from transformers import BlipProcessor, BlipForQuestionAnswering
            ckpt = 'Salesforce/blip-vqa-base'
            processor = BlipProcessor.from_pretrained(ckpt)
            model = BlipForQuestionAnswering.from_pretrained(ckpt)
        else:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            ckpt = 'Salesforce/blip-image-captioning-base'
            processor = BlipProcessor.from_pretrained(ckpt)
            model = BlipForConditionalGeneration.from_pretrained(ckpt)
        model = model.to(device if torch.cuda.is_available() else 'cpu').eval()
        _blip_cache[key] = (processor, model)
    return _blip_cache[key]


def clear_caption_model_cache() -> None:
    _blip_cache.clear()


def generate_captions(
    video_frames_folder: str,
    keyframes: List[int],
    blip_question: str = '',
    captions_folder: Optional[str] = None,
    device: str = 'cuda',
) -> str:
    """Caption the given 1-based keyframes; write one .txt per keyframe.

    Clears any existing captions first (notebook behavior). Returns the
    captions folder path.
    """
    input_frames = sorted(glob(os.path.join(video_frames_folder, '*.jpg')))
    if captions_folder is None:
        captions_folder = video_frames_folder + 'Captions'
    os.makedirs(captions_folder, exist_ok=True)
    for f in glob(os.path.join(captions_folder, '*.txt')):
        os.unlink(f)

    vqa = blip_question not in (None, '', [])
    processor, model = _load_blip(vqa, device=device)
    model_device = next(model.parameters()).device

    for kf in keyframes:
        if kf < 1 or kf > len(input_frames):
            print(f"  caption keyframe {kf} out of range, skipping")
            continue
        frame_path = input_frames[kf - 1]
        raw_image = Image.open(frame_path).convert('RGB')

        with torch.no_grad():
            if vqa:
                inputs = processor(raw_image, blip_question, return_tensors='pt').to(model_device)
                out = model.generate(**inputs)
            else:
                inputs = processor(raw_image, return_tensors='pt').to(model_device)
                # Notebook: sample=True, top_p=0.9, max_length=30, min_length=5
                out = model.generate(**inputs, do_sample=True, top_p=0.9,
                                     max_length=30, min_length=5)
        caption = processor.decode(out[0], skip_special_tokens=True)

        base = os.path.basename(frame_path)
        caption_path = os.path.join(captions_folder, os.path.splitext(base)[0] + '.txt')
        with open(caption_path, 'w') as f:
            f.write(caption)

    return captions_folder


def get_caption(frame_num: int, captions_folder: str) -> Optional[str]:
    """Piecewise-constant caption lookup for a 0-based render frame.

    Notebook `get_caption`: captions are keyed by 1-based frame numbers;
    a frame uses the caption of the latest keyframe at or before it
    (frames before the first keyframe use the first caption).
    """
    if not captions_folder or not os.path.isdir(captions_folder):
        return None
    caption_files = sorted(glob(os.path.join(captions_folder, '*.txt')))
    if not caption_files:
        return None
    frame_num1 = frame_num + 1
    frame_numbers = [
        int(os.path.splitext(os.path.basename(o))[0]) for o in caption_files]

    def _load(path):
        with open(path, 'r') as f:
            return f.read()

    if frame_num1 < frame_numbers[0]:
        return _load(caption_files[0])
    if frame_num1 >= frame_numbers[-1]:
        return _load(caption_files[-1])
    for i in range(len(frame_numbers)):
        if frame_numbers[i] <= frame_num1 < frame_numbers[i + 1]:
            return _load(caption_files[i])
    return None


def apply_caption(prompt: str, caption: Optional[str]) -> str:
    """Substitute the {caption} placeholder (no-op without a caption)."""
    if caption and '{caption}' in prompt:
        print('Replacing {caption} with', caption)
        return prompt.replace('{caption}', caption)
    return prompt
