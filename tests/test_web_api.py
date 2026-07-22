import os
from dataclasses import asdict

import pytest
from fastapi.testclient import TestClient

from vibewarp.config import RunConfig
from vibewarp.web import create_app
from vibewarp.web_jobs import JobManager


def valid_config(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    video = tmp_path / "input.mp4"
    checkpoint.write_bytes(b"model")
    video.write_bytes(b"video")
    config = RunConfig(sd_checkpoint_path=str(checkpoint))
    config.video.video_init_path = str(video)
    return asdict(config)


def test_config_endpoint_exposes_complete_defaults_and_schema():
    client = TestClient(create_app(JobManager(runner=lambda *_args, **_kwargs: [])))
    response = client.get("/api/config")
    assert response.status_code == 200
    body = response.json()
    assert body["defaults"]["vae"]["tile_size"] == 128
    assert body["defaults"]["video"]["max_size"] == 512
    assert body["prefilled"] is False
    assert "video_assembly" in body["schema"]["properties"]


def test_config_endpoint_uses_startup_settings():
    initial = RunConfig(batch_name="prefilled", sd_checkpoint_path="model.safetensors")
    client = TestClient(create_app(JobManager(runner=lambda *_args, **_kwargs: []), initial_config=initial))
    body = client.get("/api/config").json()
    assert body["defaults"]["batch_name"] == "prefilled"
    assert body["prefilled"] is True


def test_import_structured_settings_file():
    client = TestClient(create_app(JobManager(runner=lambda *_args, **_kwargs: [])))
    response = client.post("/api/config/import", json={
        "filename": "render.json",
        "content": '{"batch_name":"imported","diffusion":{"steps":31}}',
    })
    assert response.status_code == 200
    assert response.json()["config"]["batch_name"] == "imported"
    assert response.json()["config"]["diffusion"]["steps"] == 31


def test_import_rejects_unsupported_file_type():
    client = TestClient(create_app(JobManager(runner=lambda *_args, **_kwargs: [])))
    response = client.post("/api/config/import", json={"filename": "render.yaml", "content": "x: 1"})
    assert response.status_code == 422


def test_submit_and_read_job(tmp_path):
    client = TestClient(create_app(JobManager(runner=lambda config, progress_fn: [])))
    response = client.post("/api/jobs", json=valid_config(tmp_path))
    assert response.status_code == 202
    job_id = response.json()["id"]
    assert client.get(f"/api/jobs/{job_id}").status_code == 200


def test_submit_returns_field_errors():
    client = TestClient(create_app(JobManager(runner=lambda *_args, **_kwargs: [])))
    response = client.post("/api/jobs", json=asdict(RunConfig()))
    assert response.status_code == 422
    paths = {error["path"] for error in response.json()["detail"]}
    assert {"sd_checkpoint_path", "video.video_init_path"} <= paths


def test_unknown_job_and_artifact_are_not_exposed(tmp_path):
    client = TestClient(create_app(JobManager(runner=lambda *_args, **_kwargs: [])))
    assert client.get("/api/jobs/nope").status_code == 404
    response = client.post("/api/jobs", json=valid_config(tmp_path))
    assert client.get(f"/api/jobs/{response.json()['id']}/artifacts/0").status_code == 404


def test_submit_rejects_missing_input_files_before_queueing(tmp_path):
    payload = valid_config(tmp_path)
    payload["video"]["video_init_path"] = str(tmp_path / "missing.mp4")
    client = TestClient(create_app(JobManager(runner=lambda *_args, **_kwargs: [])))
    response = client.post("/api/jobs", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"] == [{
        "path": "video.video_init_path",
        "message": f"Input video does not exist: {tmp_path / 'missing.mp4'}",
    }]


def test_preview_endpoint_serves_current_job_preview(tmp_path):
    preview = tmp_path / "preview.png"
    preview.write_bytes(b"current frame")
    manager = JobManager(runner=lambda *_args, **_kwargs: [])
    job = manager.submit(RunConfig())
    job.preview_path = str(preview)
    client = TestClient(create_app(manager))
    response = client.get(f"/api/jobs/{job.id}/preview")
    assert response.status_code == 200
    assert response.content == b"current frame"
    assert response.headers["cache-control"] == "no-store"


def test_path_exists_endpoint_reports_missing_files(tmp_path):
    """A typo'd checkpoint path used to surface only mid-render, minutes in."""
    client = TestClient(create_app(JobManager(runner=lambda *_a, **_k: [])))
    real = tmp_path / "model.safetensors"
    real.write_bytes(b"model")

    found = client.get("/api/fs/exists", params={"path": str(real)}).json()
    assert found["checked"] and found["exists"] and not found["is_dir"]

    gone = client.get("/api/fs/exists", params={"path": str(tmp_path / "nope.ckpt")}).json()
    assert gone["checked"] and not gone["exists"]

    directory = client.get("/api/fs/exists", params={"path": str(tmp_path)}).json()
    assert directory["exists"] and directory["is_dir"]

    # Quotes pasted in from Explorer must not make a real path look missing.
    quoted = client.get("/api/fs/exists", params={"path": f'"{real}"'}).json()
    assert quoted["exists"]

    # Empty is not "missing" — many path fields are legitimately optional.
    assert client.get("/api/fs/exists", params={"path": ""}).json()["checked"] is False


def test_comfy_status_reports_reachable_and_unavailable_ports(monkeypatch):
    import vibewarp.web as web

    class Connection:
        def __enter__(self): return self
        def __exit__(self, *_args): pass

    calls = []
    monkeypatch.setattr(web.socket, 'create_connection',
                        lambda address, timeout: calls.append((address, timeout)) or Connection())
    client = TestClient(create_app(JobManager(runner=lambda *_a, **_k: [])))
    available = client.get('/api/comfy/status',
                           params={'url': 'http://localhost:8188'}).json()
    assert available['available'] is True
    assert available['host'] == 'localhost' and available['port'] == 8188
    assert calls == [(('localhost', 8188), 0.5)]

    def refused(*_args, **_kwargs):
        raise ConnectionRefusedError('port closed')
    monkeypatch.setattr(web.socket, 'create_connection', refused)
    unavailable = client.get('/api/comfy/status',
                             params={'url': 'http://127.0.0.1:9191'}).json()
    assert unavailable['available'] is False
    assert unavailable['port'] == 9191


def test_comfy_status_rejects_invalid_server_urls():
    client = TestClient(create_app(JobManager(runner=lambda *_a, **_k: [])))
    status = client.get('/api/comfy/status', params={'url': 'localhost:8188'}).json()
    assert status['available'] is False
    assert status['message'].startswith('Invalid ComfyUI URL:')


def test_system_settings_survive_a_settings_import(tmp_path, monkeypatch):
    """Importing someone else's settings must not repoint YOUR model paths."""
    import json
    monkeypatch.setenv("VIBEWARP_HOME", str(tmp_path))
    client = TestClient(create_app(JobManager(runner=lambda *_a, **_k: [])))

    mine = valid_config(tmp_path)
    mine["controlnet"]["model_dir"] = "C:/my/models"
    assert client.put("/api/system", json=mine).status_code == 200

    theirs = valid_config(tmp_path)
    theirs["controlnet"]["model_dir"] = "/home/them/models"
    theirs["text_prompts"] = {"0": "their prompt"}
    response = client.post("/api/config/import", json={
        "filename": "theirs.json", "content": json.dumps(theirs)})

    imported = response.json()["config"]
    assert imported["controlnet"]["model_dir"] == "C:/my/models"     # mine wins
    assert imported["text_prompts"] == {"0": "their prompt"}          # theirs is the point


def _run_with_settings(tmp_path, settings):
    """Lay out a run the way the pipeline does: <out>/<batch>/<id>/settings/*.txt"""
    import json
    run = tmp_path / "images_out" / "warpfusion" / "7"
    (run / "settings").mkdir(parents=True)
    (run / "settings" / "warpfusion(0)_settings.txt").write_text(json.dumps(settings))
    return str(tmp_path / "images_out")


def test_load_settings_from_a_run(tmp_path, monkeypatch):
    """The point of keeping history is iterating on a run you liked, not just looking at it."""
    monkeypatch.setenv("VIBEWARP_HOME", str(tmp_path))
    client = TestClient(create_app(JobManager(runner=lambda *_a, **_k: [])))

    base = valid_config(tmp_path)
    base["text_prompts"] = {"0": "a cat on a bike"}
    base["diffusion"]["seed"] = 4242
    out = _run_with_settings(tmp_path, base)

    response = client.post("/api/preview/runs/7/settings",
                           params={"output_dir": out, "batch_name": "warpfusion"})
    assert response.status_code == 200
    config = response.json()["config"]
    assert config["text_prompts"] == {"0": "a cat on a bike"}
    assert config["diffusion"]["seed"] == 4242


def test_loading_a_run_cannot_repoint_this_machines_paths(tmp_path, monkeypatch):
    """Same rule as a settings import: the system tier is MINE."""
    monkeypatch.setenv("VIBEWARP_HOME", str(tmp_path))
    client = TestClient(create_app(JobManager(runner=lambda *_a, **_k: [])))

    mine = valid_config(tmp_path)
    mine["controlnet"]["model_dir"] = "C:/my/models"
    assert client.put("/api/system", json=mine).status_code == 200

    theirs = valid_config(tmp_path)
    theirs["controlnet"]["model_dir"] = "/somewhere/else"
    out = _run_with_settings(tmp_path, theirs)

    config = client.post("/api/preview/runs/7/settings",
                         params={"output_dir": out, "batch_name": "warpfusion"}).json()["config"]
    assert config["controlnet"]["model_dir"] == "C:/my/models"


def test_a_run_without_settings_says_so(tmp_path):
    client = TestClient(create_app(JobManager(runner=lambda *_a, **_k: [])))
    run = tmp_path / "images_out" / "warpfusion" / "7"
    run.mkdir(parents=True)          # no settings/ dir at all

    response = client.post("/api/preview/runs/7/settings",
                           params={"output_dir": str(tmp_path / "images_out"),
                                   "batch_name": "warpfusion"})
    assert response.status_code == 404
    assert "saved no settings" in response.json()["detail"][0]["message"]

    # ...and the listing flags it, so the UI can grey the button out instead of
    # offering one that cannot work.
    runs = client.get("/api/preview/runs",
                      params={"output_dir": str(tmp_path / "images_out"),
                              "batch_name": "warpfusion"}).json()["runs"]
    assert runs[0]["has_settings"] is False


VIDEO = "refs/video_20260418_095626.mp4"


@pytest.mark.skipif(not os.path.isfile(VIDEO), reason="sample video not present")
def test_video_probe_reports_the_real_render_size_and_frame_ceiling():
    """What the UI shows must be what the render produces, not an approximation.

    The size comes from the same `fit_dimensions` the pipeline uses, and the frame
    ceiling accounts for extract_nth_frame -- previously you could ask for frames that
    did not exist and only find out mid-render.
    """
    client = TestClient(create_app(JobManager(runner=lambda *_a, **_k: [])))

    body = client.get("/api/video/probe",
                      params={"path": VIDEO, "max_size": 768}).json()
    assert body["source"]["width"] == 1280 and body["source"]["height"] == 720
    assert body["source"]["frames"] > 0 and body["source"]["fps"] > 0
    # 1280x720 capped at 768 on the long edge, rounded to /8
    assert (body["render"]["width"], body["render"]["height"]) == (768, 432)

    from vibewarp.video.input import fit_dimensions
    assert (body["render"]["width"], body["render"]["height"]) == \
        fit_dimensions(1280, 720, 768)

    # frame_range is inclusive, so the last valid index is n-1.
    assert body["max_frame"] == body["renderable_frames"] - 1

    # Keeping every 4th frame quarters what there is to render.
    quartered = client.get("/api/video/probe",
                           params={"path": VIDEO, "extract_nth_frame": 4}).json()
    assert quartered["renderable_frames"] == -(-body["source"]["frames"] // 4)


@pytest.mark.skipif(not os.path.isfile(VIDEO), reason="sample video not present")
def test_video_probe_strips_quotes_and_reports_a_missing_file():
    client = TestClient(create_app(JobManager(runner=lambda *_a, **_k: [])))

    quoted = client.get("/api/video/probe", params={"path": f'"{VIDEO}"'})
    assert quoted.status_code == 200

    gone = client.get("/api/video/probe", params={"path": "C:/nope.mp4"})
    assert gone.status_code == 404
    assert "No video" in gone.json()["detail"][0]["message"]


@pytest.mark.skipif(not os.path.isfile(VIDEO), reason="sample video not present")
def test_video_thumbnail_returns_a_visible_frame():
    """A clip that fades in would otherwise show a black thumbnail."""
    from io import BytesIO

    import numpy as np
    from PIL import Image

    client = TestClient(create_app(JobManager(runner=lambda *_a, **_k: [])))
    response = client.get("/api/video/thumbnail", params={"path": VIDEO})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    image = Image.open(BytesIO(response.content))
    assert max(image.size) <= 480
    assert np.asarray(image.convert("L")).mean() >= 8.0     # not black
