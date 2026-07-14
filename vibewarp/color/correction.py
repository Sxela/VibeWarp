"""Frame brightness and color correction utilities."""

import numpy as np
from PIL import Image, ImageEnhance, ImageStat


def get_brightness_contrast(image):
    """Get mean brightness and contrast (stddev) of a PIL image.

    Args:
        image: PIL Image

    Returns:
        (brightness, contrast) tuple of floats.
    """
    stat = ImageStat.Stat(image)
    brightness = sum(stat.mean) / len(stat.mean)
    contrast = sum(stat.stddev) / len(stat.stddev)
    return brightness, contrast


def adjust_brightness(image,
                      high_brightness_threshold=180,
                      high_brightness_adjust_ratio=0.97,
                      high_brightness_adjust_fix_amount=2,
                      max_brightness_threshold=254,
                      low_brightness_threshold=40,
                      low_brightness_adjust_ratio=1.03,
                      low_brightness_adjust_fix_amount=2,
                      min_brightness_threshold=1):
    """Adjust image brightness to keep it within acceptable thresholds.

    Implementation based on progrockdiffusion by lowfuel.

    Args:
        image: PIL Image
        high_brightness_threshold: upper brightness trigger
        high_brightness_adjust_ratio: enhancement factor for bright images
        high_brightness_adjust_fix_amount: fixed amount to subtract from bright pixels
        max_brightness_threshold: absolute max brightness clamp
        low_brightness_threshold: lower brightness trigger
        low_brightness_adjust_ratio: enhancement factor for dark images
        low_brightness_adjust_fix_amount: fixed amount to add to dark pixels
        min_brightness_threshold: absolute min brightness clamp

    Returns:
        Adjusted PIL Image.
    """
    brightness, contrast = get_brightness_contrast(image)

    if brightness > high_brightness_threshold:
        filt = ImageEnhance.Brightness(image)
        image = filt.enhance(high_brightness_adjust_ratio)
        arr = np.array(image)
        arr = np.where(arr > high_brightness_threshold,
                       arr - high_brightness_adjust_fix_amount, arr)
        image = Image.fromarray(arr.clip(0, 255).round().astype('uint8'))

    if brightness < low_brightness_threshold:
        filt = ImageEnhance.Brightness(image)
        image = filt.enhance(low_brightness_adjust_ratio)
        arr = np.array(image)
        arr = np.where(arr < low_brightness_threshold,
                       arr + low_brightness_adjust_fix_amount, arr)
        image = Image.fromarray(arr.clip(0, 255).round().astype('uint8'))

    # Final hard clamps
    arr = np.array(image)
    arr = np.where(arr > max_brightness_threshold,
                   arr - high_brightness_adjust_fix_amount, arr)
    arr = np.where(arr < min_brightness_threshold,
                   arr + low_brightness_adjust_fix_amount, arr)
    image = Image.fromarray(arr.clip(0, 255).round().astype('uint8'))

    return image
