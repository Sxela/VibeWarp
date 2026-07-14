"""Prompt parsing utilities.

Handles three things the notebook does before encoding prompts:
  1. Strip <lora:name:weight> tags (for LORA scheduling)
  2. Strip trailing :weight suffix (for multi-prompt weighting)
  3. Split multi-prompt strings by '|' separator
"""

import re
from typing import Dict, List, Optional, Tuple

# Matches <lora:name:weight> or <lora:name> tags
_LORA_RE = re.compile(r'<lora:([^:>]+)(?::([0-9.]+))?>', re.IGNORECASE)

# Matches trailing :number at end of prompt (multi-prompt weight)
_WEIGHT_RE = re.compile(r':\s*([\d.]+)\s*$')


def parse_prompt(text: str) -> Tuple[str, Dict[str, float]]:
    """Strip LORA tags and trailing weight from a single prompt string.

    Returns:
        (clean_prompt, {lora_name: weight})
    """
    loras: Dict[str, float] = {}

    def _extract_lora(m):
        name = m.group(1).strip()
        weight = float(m.group(2)) if m.group(2) else 1.0
        loras[name] = weight
        return ''

    clean = _LORA_RE.sub(_extract_lora, text).strip()
    return clean, loras


def split_weighted_prompts(text: str) -> Tuple[List[str], List[float]]:
    """Split '|'-separated multi-prompt string, extract per-prompt weights.

    Example:
        "a painting:0.7 | anime style:0.3"
        -> (["a painting", "anime style"], [0.7, 0.3])

    Single prompt with no weight gets weight 1.0.
    """
    parts = [p.strip() for p in text.split('|')]
    prompts: List[str] = []
    weights: List[float] = []
    for part in parts:
        if not part:
            continue
        m = _WEIGHT_RE.search(part)
        if m:
            w = float(m.group(1))
            p = _WEIGHT_RE.sub('', part).strip()
        else:
            w = 1.0
            p = part
        prompts.append(p)
        weights.append(w)
    if not prompts:
        return [''], [1.0]
    return prompts, weights


def split_lora_from_prompts(
    text_prompts: Dict[int, str],
) -> Tuple[Dict[int, str], Dict[str, Dict[int, float]]]:
    """Extract <lora:name:weight> tags from all prompt keyframes.

    Mirrors the notebook's split_lora_from_prompts().

    Args:
        text_prompts: {frame: prompt_string}

    Returns:
        (clean_prompts, lora_schedule)
        where lora_schedule is {lora_name: {frame: weight}}
    """
    clean_prompts: Dict[int, str] = {}
    lora_schedule: Dict[str, Dict[int, float]] = {}

    for frame, prompt in text_prompts.items():
        clean, frame_loras = parse_prompt(prompt)
        clean_prompts[frame] = clean
        for name, weight in frame_loras.items():
            if name not in lora_schedule:
                lora_schedule[name] = {}
            lora_schedule[name][frame] = weight

    return clean_prompts, lora_schedule


def get_prompt_and_loras_for_frame(
    frame_num: int,
    text_prompts: Dict[int, str],
    lora_schedule: Dict[str, Dict[int, float]],
) -> Tuple[str, List[str], List[float]]:
    """Resolve prompt and LORA weights for a given frame.

    Uses the same keyframe lookup as _get_prompt_for_frame:
    the active entry is the one with the highest key <= frame_num.

    Returns:
        (clean_prompt, [lora_name, ...], [weight, ...])
    """
    from vibewarp.utils.scheduling import get_scheduled_arg

    # Resolve prompt
    active_key = 0
    for k in sorted(text_prompts.keys()):
        if k <= frame_num:
            active_key = k
        else:
            break
    prompt = text_prompts.get(active_key, '')

    # Resolve LORA weights for this frame
    names: List[str] = []
    weights: List[float] = []
    for name, sched in lora_schedule.items():
        w = get_scheduled_arg(frame_num, sched, blend_json_schedules=False)
        if isinstance(w, (int, float)) and float(w) != 0.0:
            names.append(name)
            weights.append(float(w))

    return prompt, names, weights
