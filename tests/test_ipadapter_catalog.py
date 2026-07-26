"""IP-Adapter catalog and web selector contract."""

from fastapi.testclient import TestClient

from vibewarp.config import IPAdapterEntry, RunConfig
from vibewarp.config_io import validate_config
from vibewarp.ipadapter_catalog import (
    IPADAPTER_CATALOG,
    resolve_ipadapter_path,
    specs_for_version,
)
from vibewarp.web import create_app
from vibewarp.web_jobs import JobManager


def client():
    return TestClient(create_app(JobManager(runner=lambda *_args, **_kwargs: [])))


def test_catalog_contains_every_regular_and_plus_variant():
    assert set(IPADAPTER_CATALOG) == {
        "ipadapter_sd15",
        "ipadapter_sd15_light",
        "ipadapter_sd15_plus",
        "ipadapter_sd15_plus_face",
        "ipadapter_sd15_full_face",
        "ipadapter_sd15_vit_G",
        "ipadapter_sdxl",
        "ipadapter_sdxl_vit_h",
        "ipadapter_sdxl_plus_vit_h",
        "ipadapter_sdxl_plus_face_vit_h",
    }
    assert all(spec.filename and spec.url for spec in IPADAPTER_CATALOG.values())


def test_catalog_filters_adapters_by_base_model():
    assert {spec.model_version for spec in specs_for_version(
        "control_multi_v15")} == {"sd15"}
    assert {spec.model_version for spec in specs_for_version(
        "control_multi_sdxl")} == {"sdxl"}


def test_endpoint_discovers_only_ipadapter_checkpoints(tmp_path):
    adapter = tmp_path / "ip-adapter-plus_sd15.safetensors"
    adapter.write_bytes(b"x")
    (tmp_path / "control_v11p_sd15_canny.pth").write_bytes(b"x")

    body = client().get("/api/ipadapter/catalog", params={
        "model_version": "control_multi_v15",
        "model_dir": str(tmp_path),
    }).json()

    assert body["files"] == [str(adapter)]
    plus = next(item for item in body["adapters"]
                if item["key"] == "ipadapter_sd15_plus")
    assert plus["resolved_path"] == str(adapter)
    assert {item["model_version"] for item in body["adapters"]} == {"sd15"}


def test_validation_rejects_adapter_from_the_wrong_base_model():
    config = RunConfig(model_version="control_multi_sdxl")
    config.ipadapter.enabled = True
    config.ipadapter.models = {
        "ipadapter_sd15_plus": IPAdapterEntry(path="ip.bin"),
    }

    errors = validate_config(config, require_inputs=False, check_paths=False)

    assert any(error["path"] == "ipadapter.models.ipadapter_sd15_plus"
               and "sd15" in error["message"] for error in errors)


def test_resolve_uses_catalog_filename_in_controlnet_directory(tmp_path):
    assert resolve_ipadapter_path(
        "ipadapter_sd15", "", str(tmp_path)
    ) == str(tmp_path / "ip-adapter_sd15.safetensors")


def test_validation_rejects_missing_enabled_adapter(tmp_path):
    config = RunConfig(model_version="control_multi_v15")
    config.controlnet.model_dir = str(tmp_path)
    config.ipadapter.enabled = True
    config.ipadapter.models = {
        "ipadapter_sd15": IPAdapterEntry(),
    }

    errors = validate_config(config, require_inputs=False, check_paths=True)

    assert any(
        error["path"] == "ipadapter.models.ipadapter_sd15.path"
        and "does not exist" in error["message"]
        for error in errors)


def test_validation_rejects_enabled_ipadapter_without_models():
    config = RunConfig(model_version="control_multi_v15")
    config.ipadapter.enabled = True

    errors = validate_config(config, require_inputs=False, check_paths=False)

    assert any(
        error["path"] == "ipadapter.models"
        and "no adapter models" in error["message"]
        for error in errors)


def _mock_sd_loader(monkeypatch):
    from vibewarp import pipeline
    import vibewarp.core.model_loader as model_loader

    monkeypatch.setattr(pipeline, "load_sd_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        pipeline, "wrap_model_for_kdiffusion",
        lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(
        model_loader, "configure_clip_skip", lambda *args, **kwargs: None)
    return pipeline


def test_loader_resolves_catalog_checkpoint_and_reports_it(
        tmp_path, monkeypatch, capsys):
    pipeline = _mock_sd_loader(monkeypatch)
    checkpoint = tmp_path / "ip-adapter_sd15.safetensors"
    checkpoint.write_bytes(b"weights")
    captured = {}

    import vibewarp.core.ipadapter as ipadapter
    def load_adapter(path):
        captured["path"] = path
        return {"image_proj": {}, "ip_adapter": {}}
    monkeypatch.setattr(ipadapter, "load_ipadapter_model", load_adapter)

    config = RunConfig(model_version="control_multi_v15")
    config.controlnet.model_dir = str(tmp_path)
    config.ipadapter.enabled = True
    config.ipadapter.models = {
        "ipadapter_sd15": IPAdapterEntry(),
    }

    result = pipeline.load_models(config, device="cpu")

    assert captured["path"] == str(checkpoint)
    assert result["ipadapters"]["ipadapter_sd15"]["path"] == str(checkpoint)
    assert "Loaded IP-Adapter ipadapter_sd15" in capsys.readouterr().out


def test_loader_raises_instead_of_silently_skipping_missing_checkpoint(
        tmp_path, monkeypatch):
    pipeline = _mock_sd_loader(monkeypatch)
    config = RunConfig(model_version="control_multi_v15")
    config.controlnet.model_dir = str(tmp_path)
    config.ipadapter.enabled = True
    config.ipadapter.models = {
        "ipadapter_sd15": IPAdapterEntry(),
    }

    import pytest
    with pytest.raises(FileNotFoundError, match="ip-adapter_sd15.safetensors"):
        pipeline.load_models(config, device="cpu")


def test_loader_rejects_enabled_ipadapter_without_models(
        monkeypatch):
    pipeline = _mock_sd_loader(monkeypatch)
    config = RunConfig(model_version="control_multi_v15")
    config.ipadapter.enabled = True

    import pytest
    with pytest.raises(ValueError, match="no adapter models"):
        pipeline.load_models(config, device="cpu")
