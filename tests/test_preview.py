"""Preview/debug tab: run discovery, layer discovery, frame alignment."""

import pytest
from fastapi.testclient import TestClient

from vibewarp.preview import (describe_run, image_path, list_runs, resume_status,
                              set_run_label, video_file)
from vibewarp.web import create_app
from vibewarp.web_jobs import JobManager


def build_run(tmp_path, run_id="0", batch="warpfusion", frames=3):
    """A run laid out exactly as the renderer writes it."""
    run = tmp_path / batch / run_id
    (run / "video_frames").mkdir(parents=True)
    (run / "debug").mkdir()
    for n in range(frames):
        # Extracted video frames are 1-BASED on disk (render frame N -> N+1).
        (run / "video_frames" / f"{n + 1:06d}.jpg").write_bytes(b"init")
        (run / f"{batch}({run_id})_{n:06d}.png").write_bytes(b"out")
        (run / "debug" / f"control_sd15_depth_source_{n:06d}.jpg").write_bytes(b"src")
        (run / "debug" / f"control_sd15_depth_detected_{n:06d}.jpg").write_bytes(b"det")
        (run / "debug" / f"diffusion_input_{n:06d}.jpg").write_bytes(b"dif")
        (run / "debug" / f"hidream_reference_1_{n:06d}.png").write_bytes(b"ref1")
        (run / "debug" / f"flux_reference_1_{n:06d}.png").write_bytes(b"flux1")
        if n > 0:
            (run / "debug" / f"hidream_reference_2_{n:06d}.png").write_bytes(b"ref2")
            (run / "debug" / f"flux_reference_2_{n:06d}.png").write_bytes(b"flux2")
        if n > 0:  # frame 0 has no previous frame to warp
            (run / f"_warped_{n:06d}.png").write_bytes(b"warp")
            (run / "debug" / f"consistency_mask_{n:06d}.png").write_bytes(b"mask")
    return str(tmp_path)


def client():
    return TestClient(create_app(JobManager(runner=lambda *_a, **_k: [])))


class TestRunDiscovery:
    def test_lists_runs_newest_first(self, tmp_path):
        build_run(tmp_path, "1")
        build_run(tmp_path, "2")
        runs = list_runs(str(tmp_path), "warpfusion")
        assert {r["id"] for r in runs} == {"1", "2"}
        assert all(r["frames"] == 3 for r in runs)

    def test_numeric_runs_do_not_sort_lexicographically(self, tmp_path):
        import os
        import time
        for run_id in ("2", "10"):
            build_run(tmp_path, run_id)
        # Same mtime -> the numeric tiebreak must put 10 above 2.
        now = time.time()
        for run_id in ("2", "10"):
            os.utime(tmp_path / "warpfusion" / run_id, (now, now))
        runs = list_runs(str(tmp_path), "warpfusion")
        assert [r["id"] for r in runs][0] == "10"

    def test_missing_root_is_empty_not_an_error(self, tmp_path):
        assert list_runs(str(tmp_path / "nope"), "warpfusion") == []

    def test_run_carries_thumbnail_frame_and_prompt(self, tmp_path):
        """The gallery needs a thumbnail (last output frame) and something that
        makes an old run identifiable."""
        import json
        build_run(tmp_path, "3", frames=4)
        settings = tmp_path / "warpfusion" / "3" / "settings"
        settings.mkdir()
        (settings / "warpfusion(3)_settings.txt").write_text(
            json.dumps({"text_prompts": {"0": "a watercolor painting"}}))

        run = list_runs(str(tmp_path), "warpfusion")[0]
        assert run["last_frame"] == 3
        assert run["prompt"] == "a watercolor painting"

    def test_empty_run_has_no_thumbnail_frame(self, tmp_path):
        (tmp_path / "warpfusion" / "9").mkdir(parents=True)
        run = list_runs(str(tmp_path), "warpfusion")[0]
        assert run["frames"] == 0
        assert run["last_frame"] is None
        assert run["prompt"] == ""

    def test_unreadable_settings_do_not_break_the_listing(self, tmp_path):
        build_run(tmp_path, "3")
        settings = tmp_path / "warpfusion" / "3" / "settings"
        settings.mkdir()
        (settings / "broken.txt").write_text("{not json")
        run = list_runs(str(tmp_path), "warpfusion")[0]
        assert run["prompt"] == ""
        assert run["frames"] == 3

    def test_display_label_persists_without_renaming_or_reordering(self, tmp_path):
        build_run(tmp_path, "1")
        build_run(tmp_path, "2")
        before = [run["id"] for run in list_runs(str(tmp_path), "warpfusion")]

        label = set_run_label(str(tmp_path), "warpfusion", "1", "  Blue   forest  ")
        runs = list_runs(str(tmp_path), "warpfusion")

        assert label == "Blue forest"
        assert [run["id"] for run in runs] == before
        assert {run["id"]: run["label"] for run in runs}["1"] == "Blue forest"
        assert (tmp_path / "warpfusion" / "1").is_dir()

    def test_empty_display_label_restores_numeric_fallback(self, tmp_path):
        build_run(tmp_path, "1")
        set_run_label(str(tmp_path), "warpfusion", "1", "Favorite")
        assert set_run_label(str(tmp_path), "warpfusion", "1", "  ") == ""
        assert list_runs(str(tmp_path), "warpfusion")[0]["label"] == ""

    def test_video_and_resume_state_are_exposed(self, tmp_path):
        import json
        build_run(tmp_path, "1", frames=3)
        run = tmp_path / "warpfusion" / "1"
        settings = run / "settings"
        settings.mkdir()
        (settings / "warpfusion(0)_settings.txt").write_text(json.dumps({
            "frame_range": [0, 4],
            "video": {},
        }))
        (run / "warpfusion.mp4").write_bytes(b"video")

        listed = list_runs(str(tmp_path), "warpfusion")[0]
        assert listed["video_available"] is True
        assert listed["resume_available"] is True
        assert listed["resume_from"] == 3
        assert resume_status(str(run), "warpfusion", "1") == 3
        assert video_file(str(run), "warpfusion").endswith("warpfusion.mp4")


class TestLayers:
    def test_finds_every_layer(self, tmp_path):
        root = build_run(tmp_path)
        described = describe_run(root, "warpfusion", "0")
        ids = {layer["id"] for layer in described["layers"]}
        assert ids == {
            "init", "warped", "output",
            "debug:control_sd15_depth_source",
            "debug:control_sd15_depth_detected",
            "debug:diffusion_input",
            "debug:consistency_mask",
            "debug:hidream_reference_1",
            "debug:hidream_reference_2",
            "debug:flux_reference_1",
            "debug:flux_reference_2",
        }

    def test_controlnet_layers_get_catalog_labels(self, tmp_path):
        root = build_run(tmp_path)
        layers = {l["id"]: l for l in describe_run(root, "warpfusion", "0")["layers"]}
        assert layers["debug:control_sd15_depth_detected"]["label"] == "Depth — detected map"
        assert layers["debug:control_sd15_depth_source"]["group"] == "ControlNet"
        assert layers["debug:diffusion_input"]["label"] == "Diffusion input"
        assert layers["debug:consistency_mask"]["label"] == \
            "Consistency mask (white = keep warp)"
        assert layers["debug:consistency_mask"]["group"] == "Optical flow"
        assert layers["debug:hidream_reference_1"]["label"] == "HiDream reference 1"
        assert layers["debug:hidream_reference_2"]["label"] == \
            "HiDream reference 2 — style"
        assert layers["debug:hidream_reference_2"]["group"] == "Edit inputs"
        assert layers["debug:flux_reference_1"]["label"] == "FLUX.2 reference 1"
        assert layers["debug:flux_reference_2"]["label"] == "FLUX.2 reference 2"
        assert layers["debug:flux_reference_2"]["group"] == "Edit inputs"

    def test_warped_layer_has_no_frame_zero(self, tmp_path):
        root = build_run(tmp_path)
        layers = {l["id"]: l for l in describe_run(root, "warpfusion", "0")["layers"]}
        assert layers["warped"]["frames"] == [1, 2]
        assert layers["output"]["frames"] == [0, 1, 2]

    def test_unknown_run_is_none(self, tmp_path):
        assert describe_run(str(tmp_path), "warpfusion", "nope") is None

    def test_run_description_lists_each_layer_directory_only_once(self, tmp_path, monkeypatch):
        root = build_run(tmp_path, frames=20)
        import os
        real_listdir = os.listdir
        calls = []

        def counted(path):
            calls.append(os.path.normpath(str(path)))
            return real_listdir(path)

        monkeypatch.setattr("vibewarp.preview.os.listdir", counted)
        describe_run(root, "warpfusion", "0")

        run = os.path.normpath(str(tmp_path / "warpfusion" / "0"))
        assert calls.count(run) == 1
        assert calls.count(os.path.join(run, "video_frames")) == 1
        assert calls.count(os.path.join(run, "debug")) == 1


class TestFrameAlignment:
    """Video frames are 1-based on disk, everything else is 0-based. Getting this
    wrong shows the wrong init frame next to the output and silently misleads."""

    def test_init_frame_zero_maps_to_file_one(self, tmp_path):
        root = build_run(tmp_path)
        path = image_path(root, "warpfusion", "0", "init", 0)
        assert path.endswith("000001.jpg")

    def test_init_and_output_stay_in_step(self, tmp_path):
        root = build_run(tmp_path)
        for frame in (0, 1, 2):
            init = image_path(root, "warpfusion", "0", "init", frame)
            out = image_path(root, "warpfusion", "0", "output", frame)
            assert init.endswith(f"{frame + 1:06d}.jpg")
            assert out.endswith(f"{frame:06d}.png")

    def test_init_layer_frames_are_render_numbers(self, tmp_path):
        root = build_run(tmp_path)
        layers = {l["id"]: l for l in describe_run(root, "warpfusion", "0")["layers"]}
        # 3 files on disk (000001..000003) -> render frames 0..2.
        assert layers["init"]["frames"] == [0, 1, 2]

    def test_missing_frame_returns_none(self, tmp_path):
        root = build_run(tmp_path)
        assert image_path(root, "warpfusion", "0", "warped", 0) is None
        assert image_path(root, "warpfusion", "0", "output", 99) is None

    def test_seeking_known_frames_does_not_rescan_the_run(self, tmp_path, monkeypatch):
        """Image requests are the scrubber hot path; directory scans make it O(frames)."""
        root = build_run(tmp_path, run_id="7")

        def no_listing(_path):
            raise AssertionError("seeking a known frame must not list the run directory")

        monkeypatch.setattr("vibewarp.preview.os.listdir", no_listing)
        assert image_path(root, "warpfusion", "7", "output", 2).endswith(
            "warpfusion(7)_000002.png")
        assert image_path(root, "warpfusion", "7", "init", 2).replace('\\', '/').endswith(
            "video_frames/000003.jpg")

    def test_current_zero_prefix_is_direct_even_in_later_run_dir(self, tmp_path, monkeypatch):
        run = tmp_path / "warpfusion" / "19"
        run.mkdir(parents=True)
        (run / "warpfusion(0)_000123.png").write_bytes(b"out")
        monkeypatch.setattr(
            "vibewarp.preview.os.listdir",
            lambda _path: (_ for _ in ()).throw(AssertionError("unexpected scan")))
        assert image_path(str(tmp_path), "warpfusion", "19", "output", 123) \
            == str(run / "warpfusion(0)_000123.png")


class TestPrefixCollisions:
    def test_similar_net_names_do_not_bleed(self, tmp_path):
        """'control_sd15_depth_' is a prefix of 'control_sd15_depth_anything_'."""
        root = build_run(tmp_path)
        debug = tmp_path / "warpfusion" / "0" / "debug"
        (debug / "control_sd15_depth_anything_detected_000000.jpg").write_bytes(b"x")

        layers = {l["id"]: l for l in describe_run(root, "warpfusion", "0")["layers"]}
        assert "debug:control_sd15_depth_anything_detected" in layers
        # depth_detected must still have only its own 3 frames, not 4.
        assert layers["debug:control_sd15_depth_detected"]["frames"] == [0, 1, 2]


class TestSandboxing:
    @pytest.mark.parametrize("run_id", ["../..", "../../etc", "a/b", "..\\..", ""])
    def test_rejects_paths_outside_the_runs_root(self, tmp_path, run_id):
        build_run(tmp_path)
        assert describe_run(str(tmp_path), "warpfusion", run_id) is None
        assert image_path(str(tmp_path), "warpfusion", run_id, "output", 0) is None

    def test_rejects_unknown_layer(self, tmp_path):
        root = build_run(tmp_path)
        assert image_path(root, "warpfusion", "0", "settings", 0) is None
        assert image_path(root, "warpfusion", "0", "debug:../../secret", 0) is None


class TestPreviewEndpoints:
    def test_runs_endpoint(self, tmp_path):
        build_run(tmp_path, "7")
        body = client().get("/api/preview/runs", params={
            "output_dir": str(tmp_path), "batch_name": "warpfusion"}).json()
        assert [r["id"] for r in body["runs"]] == ["7"]

    def test_run_detail_endpoint(self, tmp_path):
        build_run(tmp_path, "7")
        body = client().get("/api/preview/runs/7", params={
            "output_dir": str(tmp_path), "batch_name": "warpfusion"}).json()
        assert body["frames"] == [0, 1, 2]
        assert any(l["id"] == "output" for l in body["layers"])

    def test_run_label_endpoint_updates_listing(self, tmp_path):
        build_run(tmp_path, "7")
        api = client()
        response = api.put("/api/preview/runs/7/label", params={
            "output_dir": str(tmp_path), "batch_name": "warpfusion"},
            json={"label": "Night city"})

        assert response.status_code == 200
        assert response.json()["label"] == "Night city"
        runs = api.get("/api/preview/runs", params={
            "output_dir": str(tmp_path), "batch_name": "warpfusion"}).json()["runs"]
        assert runs[0]["label"] == "Night city"

    def test_run_label_endpoint_validates_length_and_run(self, tmp_path):
        build_run(tmp_path, "7")
        api = client()
        params = {"output_dir": str(tmp_path), "batch_name": "warpfusion"}
        assert api.put("/api/preview/runs/7/label", params=params,
                       json={"label": "x" * 81}).status_code == 422
        assert api.put("/api/preview/runs/nope/label", params=params,
                       json={"label": "name"}).status_code == 404

    def test_image_endpoint_serves_the_right_file(self, tmp_path):
        build_run(tmp_path, "7")
        response = client().get("/api/preview/runs/7/image", params={
            "output_dir": str(tmp_path), "batch_name": "warpfusion",
            "layer": "init", "frame": 0})
        assert response.status_code == 200
        assert response.content == b"init"

    def test_image_endpoint_404s_on_missing_frame(self, tmp_path):
        build_run(tmp_path, "7")
        response = client().get("/api/preview/runs/7/image", params={
            "output_dir": str(tmp_path), "batch_name": "warpfusion",
            "layer": "warped", "frame": 0})
        assert response.status_code == 404

    def test_unknown_run_404s(self, tmp_path):
        response = client().get("/api/preview/runs/nope", params={
            "output_dir": str(tmp_path), "batch_name": "warpfusion"})
        assert response.status_code == 404

    def test_video_endpoint_serves_an_assembled_run(self, tmp_path):
        build_run(tmp_path, "7")
        video = tmp_path / "warpfusion" / "7" / "warpfusion.mp4"
        video.write_bytes(b"mp4")
        response = client().get("/api/preview/runs/7/video", params={
            "output_dir": str(tmp_path), "batch_name": "warpfusion"})
        assert response.status_code == 200
        assert response.content == b"mp4"
        assert response.headers["content-type"] == "video/mp4"

    def test_resume_endpoint_queues_the_first_missing_frame(self, tmp_path, monkeypatch):
        import json
        import time
        monkeypatch.setenv("VIBEWARP_HOME", str(tmp_path / "home"))
        build_run(tmp_path, "7", frames=3)
        settings = tmp_path / "warpfusion" / "7" / "settings"
        settings.mkdir()
        (settings / "warpfusion(0)_settings.txt").write_text(json.dumps({
            "frame_range": [0, 4], "video": {},
        }))
        calls = []
        manager = JobManager(runner=lambda config, **kwargs: calls.append(kwargs) or [])
        api = TestClient(create_app(manager))

        response = api.post("/api/preview/runs/7/resume", params={
            "output_dir": str(tmp_path), "batch_name": "warpfusion"})

        assert response.status_code == 202
        assert response.json()["operation"] == "resume"
        deadline = time.time() + 1
        while not calls and time.time() < deadline:
            time.sleep(.01)
        assert calls[0]["resume_from"] == 3
        assert calls[0]["existing_run_dir"] == str(
            (tmp_path / "warpfusion" / "7").resolve())
