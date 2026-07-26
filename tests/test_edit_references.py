from PIL import Image
import pytest

from vibewarp.config import EditReferenceConfig
from vibewarp.core.edit_references import (
    add_reference_label, reference_prompt, resolve_reference_images)


def _image(color):
    return Image.new('RGB', (12, 8), color)


def test_resolves_ordered_sources_and_stops_at_none():
    refs = resolve_reference_images(
        [
            EditReferenceConfig('raw'),
            EditReferenceConfig('previous'),
            EditReferenceConfig('warped'),
            EditReferenceConfig('none'),
            EditReferenceConfig('raw'),
        ],
        raw_image=_image('green'),
        previous_image=_image('blue'),
        warped_image=_image('red'),
        size=(12, 8),
    )
    assert [ref.getpixel((0, 0)) for ref in refs] == [
        (0, 128, 0), (0, 0, 255), (255, 0, 0)]


def test_required_temporal_source_falls_back_and_optional_source_is_omitted():
    refs = resolve_reference_images(
        [
            EditReferenceConfig('previous'),
            EditReferenceConfig('warped'),
            EditReferenceConfig('none'),
        ],
        raw_image=_image('green'),
        previous_image=None,
        warped_image=None,
        size=(12, 8),
    )
    assert [ref.getpixel((0, 0)) for ref in refs] == [(0, 128, 0)]


def test_missing_optional_temporal_source_compacts_later_references(tmp_path,
                                                                    monkeypatch):
    upload = tmp_path / 'style.png'
    _image('purple').save(upload)
    labels = []
    import vibewarp.core.edit_references as reference_module
    monkeypatch.setattr(
        reference_module, 'add_reference_label',
        lambda image, text, opacity: labels.append(text) or image)

    refs = resolve_reference_images(
        [
            EditReferenceConfig('raw', label=True),
            EditReferenceConfig('previous', label=True),
            EditReferenceConfig('upload', label=True,
                                image_path=str(upload)),
        ],
        raw_image=_image('green'),
        previous_image=None,
        warped_image=None,
        size=(12, 8),
    )

    assert [ref.getpixel((0, 0)) for ref in refs] == [
        (0, 128, 0), (128, 0, 128)]
    assert labels == ['@Image1', '@Image2']


def test_upload_is_loaded_resized_and_can_be_labeled(tmp_path):
    upload = tmp_path / 'style.png'
    _image('purple').resize((4, 4)).save(upload)
    refs = resolve_reference_images(
        [
            EditReferenceConfig('raw'),
            EditReferenceConfig('upload', label=True, image_path=str(upload)),
            EditReferenceConfig('none'),
        ],
        raw_image=_image('green'),
        previous_image=None,
        warped_image=None,
        size=(120, 80),
    )
    assert refs[1].size == (120, 80)
    assert refs[1].getpixel((0, 0)) == (128, 0, 128)
    assert refs[1].getpixel((0, 79)) == (39, 0, 39)


def test_reference_label_opacity_blends_with_source_pixels():
    image = Image.new('RGB', (120, 80), (100, 100, 100))
    labeled = add_reference_label(image, '@Image1', opacity=0.7)
    assert labeled.getpixel((0, 79)) == (30, 30, 30)
    assert add_reference_label(image, '@Image1', opacity=0).getpixel(
        (0, 79)) == (100, 100, 100)


def test_baked_labels_use_short_unique_image_tokens(monkeypatch):
    import vibewarp.core.edit_references as reference_module

    labels = []
    monkeypatch.setattr(
        reference_module, 'add_reference_label',
        lambda image, text, opacity: labels.append((text, opacity)) or image)
    resolve_reference_images(
        [
            EditReferenceConfig('raw', label=True),
            EditReferenceConfig('warped', label=True),
            EditReferenceConfig('none'),
        ],
        raw_image=_image('green'),
        previous_image=None,
        warped_image=_image('red'),
        size=(12, 8),
    )
    assert labels == [('@Image1', 0.7), ('@Image2', 0.7)]


def test_first_reference_cannot_be_upload():
    with pytest.raises(ValueError, match='cannot be an upload'):
        resolve_reference_images(
            [EditReferenceConfig('upload', image_path='missing.png')],
            raw_image=_image('green'), previous_image=None, warped_image=None,
            size=(12, 8))


def test_multi_reference_instruction_is_added_once():
    instruction = 'Use ref 1 for content and the others for style.'
    assert reference_prompt('paint it', instruction, 1) == 'paint it'
    assert reference_prompt('paint it', instruction, 2) == \
        f'{instruction}\n\npaint it'
