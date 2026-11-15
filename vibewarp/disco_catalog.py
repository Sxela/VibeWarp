"""Guidance encoders offered independently of the legacy SD text encoders."""
CLASSIC_ENCODERS = ('ViT-B-32', 'ViT-B-16', 'ViT-L-14', 'RN50', 'RN101')
MODERN_ENCODERS = {
    'siglip2-base': 'google/siglip2-base-patch16-224',
    'siglip2-large': 'google/siglip2-large-patch16-256',
}
