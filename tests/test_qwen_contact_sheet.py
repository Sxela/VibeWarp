"""Qwen contact-sheet geometry, planning, and render-loop tests."""

import json
from pathlib import Path

import pytest
from PIL import Image

from vibewarp.config import EditReferenceConfig, RunConfig, VideoConfig
from vibewarp.config_io import validate_config
from vibewarp.core.diffusion import (
    RenderContext, _contact_sheet_groups, _contact_sheet_instruction,
    run_frames)
from vibewarp.core.contact_sheet import (
    SheetCell, assemble_contact_sheet, choose_sheet_layout, sheet_manifest,
    split_contact_sheet)


def test_adaptive_layout_uses_column_for_landscape_frames():
    layout = choose_sheet_layout((1280, 720), 3, 'adaptive', 4)

    assert (layout.rows, layout.columns) == (3, 1)
    assert layout.canvas_size == (800, 1328)
    assert layout.cell_size == (800, 440)
    assert len(layout.boxes) == 3


def test_adaptive_layout_uses_row_for_portrait_frames():
    layout = choose_sheet_layout((720, 1280), 3, 'adaptive', 4)

    assert (layout.rows, layout.columns) == (1, 3)
    assert layout.canvas_size == (1328, 800)
    assert layout.cell_size == (440, 800)


def test_adaptive_layout_uses_grid_for_square_frames():
    layout = choose_sheet_layout((512, 512), 3, 'adaptive', 4)

    assert (layout.rows, layout.columns) == (2, 2)
    assert layout.canvas_size == (1024, 1024)
    assert layout.cell_size == (510, 510)
    assert len(layout.boxes) == 4


def test_assemble_and_split_preserve_cell_order_and_manifest():
    layout = choose_sheet_layout((1280, 720), 3, 'adaptive', 4)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    images = [Image.new('RGB', (32, 18), color) for color in colors]
    sheet = assemble_contact_sheet(images, layout)
    tiles = split_contact_sheet(sheet, layout, 3)
    manifest = sheet_manifest(layout, [
        SheetCell(2, 'anchor', False),
        SheetCell(3, 'raw', True),
        SheetCell(4, 'raw', True),
    ])

    assert sheet.size == layout.canvas_size
    assert [tile.getpixel((0, 0)) for tile in tiles] == colors
    assert [cell['frame_num'] for cell in manifest['cells']] == [2, 3, 4]
    assert manifest['cells'][0]['accept'] is False


def test_split_rejects_a_comfy_resize():
    layout = choose_sheet_layout((1280, 720), 3, 'adaptive', 4)

    with pytest.raises(ValueError, match='changed contact-sheet size'):
        split_contact_sheet(Image.new('RGB', (64, 64)), layout, 3)


def test_reinject_instruction_uses_two_anchors_and_protects_padding():
    layout = choose_sheet_layout((512, 512), 3, 'adaptive', 4)
    cells = [
        SheetCell(1, 'anchor', False),
        SheetCell(2, 'anchor', False),
        SheetCell(3, 'raw', True),
    ]

    instruction = _contact_sheet_instruction(cells, layout)

    assert 'panel 1 (top-left)' in instruction
    assert 'Panels 1 and 2 are two already-stylized' in instruction
    assert 'Panel 3 is the different raw next video frame' in instruction
    assert 'Never paste or repeat either reference frame' in instruction
    assert 'panel 4 (bottom-right)' in instruction
    assert 'leave it flat gray' in instruction


def test_custom_instruction_keeps_mandatory_grid_guard():
    layout = choose_sheet_layout((512, 512), 3, 'adaptive', 4)
    cells = [SheetCell(index, 'raw', True) for index in range(3)]

    instruction = _contact_sheet_instruction(
        cells, layout, 'Render all targets as ink drawings.')

    assert instruction.startswith('Edit this 2x2 contact sheet in place.')
    assert 'Never duplicate, replace, merge, extend, or rearrange' in instruction
    assert instruction.endswith('Render all targets as ink drawings.')


def test_raw_groups_split_at_prompt_changes():
    config = RunConfig(model_version='qwen_image_edit_2511')
    config.contact_sheet.mode = 'raw_triplets'
    config.text_prompts = {0: 'style a', 2: 'style b'}
    ctx = RenderContext(config=config)

    groups = _contact_sheet_groups(ctx, 0, 5)

    assert [[cell.frame_num for cell in group] for group in groups] == [
        [0, 1], [2, 3, 4]]


def test_reinject_groups_roll_two_stylized_anchors_for_one_raw_target():
    config = RunConfig(model_version='qwen_image_edit_2511')
    config.contact_sheet.mode = 'reinject'
    ctx = RenderContext(config=config)

    groups = _contact_sheet_groups(ctx, 0, 7)

    assert [[(cell.frame_num, cell.role, cell.accept) for cell in group]
            for group in groups] == [
        [(0, 'raw', True), (1, 'raw', True), (2, 'raw', True)],
        [(1, 'anchor', False), (2, 'anchor', False), (3, 'raw', True)],
        [(2, 'anchor', False), (3, 'anchor', False), (4, 'raw', True)],
        [(3, 'anchor', False), (4, 'anchor', False), (5, 'raw', True)],
        [(4, 'anchor', False), (5, 'anchor', False), (6, 'raw', True)],
    ]


class _EchoQwenBackend:
    def __init__(self):
        self.calls = []

    def render(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs['edit_image'].copy()


def _make_context(tmp_path: Path, mode: str, frame_count: int):
    video_dir = tmp_path / 'video_frames'
    video_dir.mkdir()
    for frame_num in range(frame_count):
        Image.new(
            'RGB', (64, 36),
            ((frame_num + 1) * 20, 10, 10),
        ).save(video_dir / f'{frame_num + 1:06d}.jpg')
    batch_dir = tmp_path / 'batch'
    batch_dir.mkdir()
    config = RunConfig(
        model_version='qwen_image_edit_2511',
        batch_name='sheet',
        video=VideoConfig(width=64, height=36),
    )
    config.contact_sheet.mode = mode
    config.contact_sheet.save_debug = True
    backend = _EchoQwenBackend()
    return RenderContext(
        config=config,
        batch_folder=str(batch_dir),
        video_frames_folder=str(video_dir),
        edit_backend=backend,
    ), backend


def test_raw_contact_sheet_renders_three_then_two_frames(tmp_path):
    ctx, backend = _make_context(tmp_path, 'raw_triplets', 5)
    progress = []
    ctx.progress_fn = lambda current, total: progress.append((current, total))

    paths = run_frames(ctx, frame_range=[0, 4])

    assert len(paths) == 5
    assert len(backend.calls) == 2
    assert all(len(call['reference_images']) == 1 for call in backend.calls)
    assert progress == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]
    assert all(Path(path).is_file() for path in paths)


def test_reinject_contact_sheet_uses_two_anchors_and_discards_echoes(tmp_path):
    ctx, backend = _make_context(tmp_path, 'reinject', 7)

    paths = run_frames(ctx, frame_range=[0, 6])

    assert len(paths) == 7
    assert len(backend.calls) == 5
    debug = Path(ctx.batch_folder) / 'debug'
    manifests = sorted(debug.glob('qwen_sheet_manifest_*.json'))
    assert len(manifests) == 5
    second = json.loads(manifests[1].read_text(encoding='utf-8'))
    assert [(cell['frame_num'], cell['role'], cell['accept'])
            for cell in second['cells']] == [
        (1, 'anchor', False), (2, 'anchor', False), (3, 'raw', True)]
    second_prompt = backend.calls[1]['prompt']
    assert 'Never paste or repeat either reference frame' in second_prompt
    assert 'Requested visual transformation for the target panel only' in second_prompt
    assert (debug / 'qwen_sheet_echo_000001.png').is_file()
    assert (debug / 'qwen_sheet_echo_000002.png').is_file()
    assert len(list(Path(ctx.batch_folder).glob('sheet(0)_*.png'))) == 7


def test_contact_sheet_resume_reconstructs_group_and_only_saves_missing_frames(
    tmp_path,
):
    ctx, _ = _make_context(tmp_path, 'reinject', 5)
    run_frames(ctx, frame_range=[0, 3])
    resumed_backend = _EchoQwenBackend()
    resumed = RenderContext(
        config=ctx.config,
        batch_folder=ctx.batch_folder,
        video_frames_folder=ctx.video_frames_folder,
        edit_backend=resumed_backend,
    )

    paths = run_frames(resumed, frame_range=[0, 4], resume_from=4)

    assert [Path(path).name for path in paths] == ['sheet(0)_000004.png']
    assert len(resumed_backend.calls) == 1


@pytest.mark.parametrize('model_version', [
    'flux2_klein_edit',
    'hidream_o1_edit',
    'qwen_image_edit_2511',
    'mage_flow_edit',
])
def test_shared_contact_sheet_path_dispatches_all_comfy_edit_families(
    tmp_path, model_version,
):
    ctx, backend = _make_context(tmp_path, 'raw_triplets', 1)
    ctx.config.model_version = model_version

    paths = run_frames(ctx, frame_range=[0, 0])

    assert len(paths) == 1
    assert len(backend.calls) == 1


def test_validation_rejects_optional_references_in_contact_sheet_mode():
    config = RunConfig(model_version='qwen_image_edit_2511')
    config.contact_sheet.mode = 'raw_triplets'
    config.qwen.references = [
        EditReferenceConfig(source='raw'),
        EditReferenceConfig(source='upload', image_path='style.png'),
    ]

    errors = validate_config(config, check_paths=False)

    assert any(error['path'] == 'qwen.references'
               and 'contact-sheet mode' in error['message'].lower()
               for error in errors)
