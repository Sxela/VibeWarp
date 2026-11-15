from vibewarp.config import RunConfig
from vibewarp.config_io import config_from_dict, config_schema


def fields(schema):
    for prop in schema['properties'].values():
        yield from prop['properties'].values() if prop['type'] == 'dataclass' else [prop]


def test_every_editable_field_has_a_workspace_location():
    schema = config_schema()
    workflows = {item['id'] for item in schema['workflows']}
    for prop in fields(schema):
        if prop.get('tier') == 'hidden':
            assert 'workflows' not in prop
        else:
            assert prop['workflows'] and set(prop['workflows']) <= workflows


def test_workspace_model_files_keep_their_machine_local_tier():
    schema = config_schema()
    prop = schema['properties']['flux']['properties']['comfy_unet_name']
    assert prop['workflows'] == ['model', 'system']
    assert prop['tier'] == 'system'
    motion = schema['properties']['warp']['properties']['flow_blend_schedule']
    assert motion['workflows'] == ['consistency']
    assert not motion['workspace_advanced']


def test_mask_workspace_is_rvm_only_and_functional_masks_have_one_home():
    schema=config_schema();props=schema['properties']
    tab=next(item for item in schema['workflows'] if item['id']=='background')
    assert tab['label']=='Mask & tint' and tab['groups']==['Background masking','RVM files']
    for section,group in props.items():
        for name,field in group.get('properties',{}).items():
            if 'background' in field.get('workflows',[]):
                assert section=='mask' and not name.startswith('raw_color_')
    cn=props['controlnet']['properties']['qr_cn_mask_grayscale']
    assert cn['editor']=='controlnet' and cn['workflows']==['conditioning']
    assert props['video_assembly']['properties']['use_background_mask_video']['workflows']==['export']
    assert props['flow']['properties']['check_consistency']['group']=='Flow & Consistency'
    assert props['flow']['properties']['check_consistency']['workflows']==['consistency']
    for name in ('consistency_soften','disable_cc'):
        assert props['turbo']['properties'][name]['workflows']==['consistency']
    assert props['diffusion']['properties']['temporal_guidance_cc_trusted_only']['workflows']==['clip']
    for name in ('rvm_model_path','rvm_cache_dir'):
        field=props['mask']['properties'][name]
        assert field['tier']=='system' and field['workflows']==['background','system']
    assert props['mask']['properties']['apply_mask_after_warp']['tier']=='hidden'
    for name in ('inpaint','inpainting_stop','inpaint_mask_blur','inpaint_mask_sigma','inpaint_mask_threshold'):
        assert props['disco']['properties'][name]['workflows']==['prompts']
        assert props['disco']['properties'][name]['group']=='Disco inpainting masks'
    for name in ('raw_color_enabled','raw_color_method','raw_color_strength','raw_color_reference'):
        field=props['mask']['properties'][name]
        assert field['workflows']==['color'] and field['preview']=='raw_video'


def test_consistency_schedules_use_the_actual_hidden_weight_fallbacks():
    props=config_schema()['properties']['flow']['properties']
    for prefix in ('missed','overshoot','edges'):
        assert props[prefix+'_consistency_schedule']['schedule_fallback']==prefix+'_consistency_weight'
        assert props[prefix+'_consistency_weight']['tier']=='hidden'


def test_fixed_seed_policies_share_the_seed_group():
    props = config_schema()['properties']
    for family in ('diffusion', 'disco', 'flux', 'hidream', 'qwen', 'mage'):
        fixed = props[family]['properties']['fixed_seed']
        assert fixed['group'] == 'Seed'
        assert fixed['workflows'] == ['prompts']
        assert not fixed['workspace_advanced']


def test_disco_clip_controls_and_cutouts_have_an_ordered_workspace():
    schema = config_schema()
    clip = next(tab for tab in schema['workflows'] if tab['id'] == 'clip')
    assert clip['label']=='Guidance'
    assert clip['groups'] == ['General', 'CLIP guidance', 'Cutouts', 'Advanced guidance']
    props = schema['properties']['disco']['properties']
    for name in ('encoders', 'clip_guidance_scale', 'image_prompts'):
        assert props[name]['workflows'] == ['clip']
        assert props[name]['group'] == 'CLIP guidance'
    for name in ('use_secondary_model','init_scale_schedule','image_scale_schedule','mask_frame_guidance'):
        assert props[name]['group']=='General' and props[name]['workflows']==['clip']
    for name in ('cutn_batches', 'cut_max_size', 'cut_overview', 'cut_innercut', 'cut_icgray_schedule',
                 'cut_ic_pow', 'skip_augs', 'cut_batch_group', 'clip_divide_cuts',
                 'lock_cutouts_across_frames'):
        assert props[name]['workflows'] == ['clip']
        assert props[name]['group'] == 'Cutouts'
        assert not props[name]['workspace_advanced']
    for name in ('clip_grad_steps', 'guidance_every', 'balance_guidance',
                 'clamp_max', 'guidance_noise_cap_percent', 'tv_scale', 'range_scale', 'sat_scale'):
        assert props[name]['workflows'] == ['clip']
        assert props[name]['group'] == 'Advanced guidance'
    assert props['sampler']['workflows'] == ['prompts']
    assert props['cut_icgray_p']['tier'] == 'hidden'


def test_model_picker_omits_disco_256_without_breaking_legacy_settings():
    schema = config_schema()
    options = schema['model_options']
    assert len(options) == 10
    assert 'disco_256' not in {option['value'] for option in options}
    assert schema['properties']['model_version']['choice_labels']['control_multi_v15'] == 'Stable Diffusion 1.5'
    assert config_from_dict({'model_version': 'disco_256'}).model_version == 'disco_256'
    assert RunConfig().model_version in {option['value'] for option in options}


def test_custom_editors_are_declared_by_the_schema():
    props = config_schema()['properties']
    assert props['video']['properties']['video_init_path']['editor'] == 'video'
    assert props['controlnet']['properties']['models']['editor'] == 'controlnet'
    assert props['qwen']['properties']['references']['editor'] == 'references'
    assert 'editor' not in props['controlnet']['properties']['model_dir']


def test_frame_range_has_no_independent_frame_limit():
    assert 'max_frames' not in config_schema()['properties']['video']['properties']


def test_model_picker_groups_native_and_comfy_managed_models():
    schema = config_schema()
    assert schema['model_groups'] == [
        {'id': 'native', 'label': 'Native models'},
        {'id': 'comfy', 'label': 'ComfyUI-managed models'},
    ]
    native = {option['value'] for option in schema['model_options'] if option['group'] == 'native'}
    assert native == {'control_multi_v15', 'control_multi_sdxl', 'disco_512'}
    assert len([option for option in schema['model_options'] if option['group'] == 'comfy']) == 7
