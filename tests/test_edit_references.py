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
        lambda image, text, opacity, **kwargs: labels.append(text) or image)

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
        lambda image, text, opacity, **kwargs: labels.append((text, opacity)) or image)
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


def test_missing_system_fonts_still_produce_a_large_scalable_tag(monkeypatch):
    import numpy as np
    from PIL import ImageFont
    original = ImageFont.truetype
    def missing_named_font(font, size, *args, **kwargs):
        if isinstance(font, str): raise OSError('No installed font')
        return original(font, size, *args, **kwargs)
    monkeypatch.setattr(ImageFont, 'truetype', missing_named_font)
    output = add_reference_label(Image.new('RGB',(720,1280)), '@Image2', opacity=1)
    y,x = np.where(np.asarray(output)[...,0] > 200)
    assert y.max()-y.min() >= 35
    assert x.max()-x.min() >= 150
    assert output.size == (720,1280)


def test_tag_size_and_badge_height_change_actual_input_pixels():
    import numpy as np
    source = Image.new('RGB',(720,1280),(100,100,100))
    small = add_reference_label(source,'@Image2',opacity=1,font_scale=1,badge_height=.12)
    large = add_reference_label(source,'@Image2',opacity=1,font_scale=2,badge_height=.24)
    small_y,small_x=np.where(np.asarray(small)[...,0]>200)
    large_y,large_x=np.where(np.asarray(large)[...,0]>200)
    assert large_y.max()-large_y.min() > 1.6*(small_y.max()-small_y.min())
    assert large_x.max()-large_x.min() > 1.6*(small_x.max()-small_x.min())
    assert small.getpixel((0,1000)) == (100,100,100)
    assert large.getpixel((0,1000)) == (0,0,0)
    assert source.getpixel((0,1279)) == (100,100,100)


def test_oversized_tag_is_fitted_inside_the_badge():
    import numpy as np
    output = add_reference_label(Image.new('RGB',(320,640)), '@Image10', opacity=1,
                                 font_scale=4,badge_height=.04)
    y,x = np.where(np.asarray(output)[...,0]>200)
    assert x.min()>0 and x.max()<319
    assert y.min()>=608 and y.max()<639


def test_reference_tag_controls_reach_the_baking_helper(monkeypatch):
    import vibewarp.core.edit_references as module
    captured=[]
    monkeypatch.setattr(module, 'add_reference_label', lambda image,text,**kwargs: captured.append(kwargs) or image)
    resolve_reference_images([EditReferenceConfig('raw',label=True)],raw_image=_image('green'),
        previous_image=None,warped_image=None,size=(120,80),label_font_scale=2.5,label_badge_height=.2)
    assert captured == [{'opacity':.7,'font_scale':2.5,'badge_height':.2}]


@pytest.mark.parametrize('field,value',[('reference_label_font_scale',.4),('reference_label_font_scale',float('inf')),
    ('reference_label_badge_height',.31),('reference_label_badge_height',float('nan'))])
def test_invalid_label_geometry_is_rejected_for_comfy_models(field,value):
    from vibewarp.config import RunConfig
    from vibewarp.config_io import validate_config
    config=RunConfig(model_version='flux3_klein')
    setattr(config,field,value)
    assert any(error['path']==field for error in validate_config(config,require_inputs=False,check_paths=False))


def test_reference_label_geometry_roundtrips_with_project_settings(tmp_path):
    import json
    from vibewarp.config import RunConfig
    from vibewarp.config_io import config_from_settings
    from vibewarp.settings import settings_to_dict
    config=RunConfig(model_version='flux3_klein')
    config.reference_label_font_scale=2.5
    config.reference_label_badge_height=.2
    path=tmp_path/'settings.txt'
    path.write_text(json.dumps(settings_to_dict(config)))
    restored=config_from_settings(path)
    assert restored.reference_label_font_scale==2.5
    assert restored.reference_label_badge_height==.2


def test_multi_reference_instruction_is_added_once():
    instruction = 'Use ref 1 for content and the others for style.'
    assert reference_prompt('paint it', instruction, 1) == 'paint it'
    assert reference_prompt('paint it', instruction, 2) == \
        f'{instruction}\n\npaint it'
