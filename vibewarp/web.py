"""Local FastAPI server for the VibeWarp Svelte application."""
import argparse, asyncio, copy, io, json, math, mimetypes, os, socket, tempfile, uuid, webbrowser
from dataclasses import asdict
from pathlib import Path
from threading import Timer
from urllib.parse import urlsplit
from PIL import Image
from vibewarp.config import RunConfig, flux_model_files, qwen_model_files
from vibewarp.config_io import ConfigError, apply_path_defaults, config_from_dict, config_from_settings, config_schema, strip_path_quotes, validate_config
from vibewarp.controlnet_catalog import CONTROLNET_MODES, MODE_PRESETS, specs_for_version
from vibewarp import system_settings
from vibewarp.preview import (describe_run, image_path, list_runs, resume_status,
                              run_dir, runs_root, set_run_label, settings_file,
                              video_file)
from vibewarp.video.input import first_visible_frame, fit_dimensions, frame_at, probe_video
from vibewarp.web_jobs import JobManager, TERMINAL_STATES

STATIC_DIR = Path(__file__).with_name("web_static")
CHECKPOINT_SUFFIXES = (".pth", ".safetensors", ".ckpt", ".bin")
manager = JobManager()


def _comfy_port_status(server_url: str, timeout: float = 0.5) -> dict:
    """Check whether the TCP port configured for ComfyUI accepts connections."""
    url = (server_url or '').strip().rstrip('/')
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname:
            raise ValueError('expected an http:// or https:// server URL')
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    except ValueError as exc:
        return {'url': url, 'available': False, 'message': f'Invalid ComfyUI URL: {exc}'}

    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            pass
    except OSError as exc:
        return {
            'url': url, 'host': parsed.hostname, 'port': port,
            'available': False, 'message': str(exc),
        }
    return {
        'url': url, 'host': parsed.hostname, 'port': port,
        'available': True, 'message': 'ComfyUI port is reachable',
    }

def _ui_config(config: RunConfig) -> RunConfig:
    config = copy.deepcopy(config)
    for key, value in flux_model_files(config).items():
        setattr(config.flux, key, value)
    for key, value in qwen_model_files(config).items():
        setattr(config.qwen, key, value)
    if config.video.max_size <= 0:
        config.video.max_size = max(config.video.width, config.video.height)
    return config

def _controlnet_files(model_dir: str) -> list:
    """Checkpoint files present in model_dir (non-recursive), sorted by name."""
    if not model_dir or not os.path.isdir(model_dir):
        return []
    return sorted(
        os.path.join(model_dir, name)
        for name in os.listdir(model_dir)
        if name.lower().endswith(CHECKPOINT_SUFFIXES)
        and os.path.isfile(os.path.join(model_dir, name))
    )

def create_app(job_manager: JobManager | None = None, initial_config: RunConfig | None = None):
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise RuntimeError("The web UI requires `pip install -e .[ui]`") from exc
    jobs = job_manager or manager
    app = FastAPI(title="VibeWarp", version="1")

    def parse(payload: dict, require_inputs: bool = True) -> RunConfig:
        try:
            config = apply_path_defaults(config_from_dict(payload))
        except ConfigError as exc:
            raise HTTPException(422, detail=[{"path": "config", "message": str(exc)}])
        errors = validate_config(config, require_inputs=require_inputs, check_paths=require_inputs)
        if errors:
            raise HTTPException(422, detail=errors)
        return config

    def saved_run_config(run: str) -> RunConfig:
        path = settings_file(run)
        if path is None:
            raise HTTPException(404, detail="This run saved no settings")
        try:
            config = config_from_settings(path)
            # Keep portable run settings, but restore this installation's model
            # paths and machine-specific values exactly as settings loading does.
            return apply_path_defaults(config_from_dict(
                system_settings.overlay(asdict(config))))
        except (ConfigError, ValueError, json.JSONDecodeError, TypeError) as exc:
            raise HTTPException(422, detail=str(exc))

    @app.get("/api/config")
    def get_config_metadata():
        # This machine's paths/VRAM tuning win over the shipped defaults.
        overlaid = config_from_dict(system_settings.overlay(
            asdict(initial_config or RunConfig())))
        defaults = asdict(_ui_config(overlaid))
        return {
            "defaults": defaults,
            "schema": config_schema(),
            "prefilled": initial_config is not None,
        }

    @app.get("/api/fs/exists")
    def path_exists(path: str = ""):
        """Does this path exist on the server?

        A typo'd checkpoint path used to surface only at render time, minutes in. The UI
        checks on blur so you find out immediately.
        """
        cleaned = strip_path_quotes(path or "")
        if not cleaned:
            return {"path": path, "checked": False}
        target = Path(cleaned)
        return {
            "path": cleaned,
            "checked": True,
            "exists": target.exists(),
            "is_dir": target.is_dir(),
        }

    @app.get('/api/comfy/status')
    def comfy_status(url: str = ''):
        """Fast UI preflight for Comfy-backed models; no render is submitted."""
        return _comfy_port_status(url)

    @app.get("/api/video/probe")
    def probe_video_endpoint(path: str = "", max_size: int = 0, extract_nth_frame: int = 1):
        """Source metadata + the size and frame count this render will ACTUALLY produce.

        The render size is computed with the same `fit_dimensions` the pipeline uses, so
        what the UI shows is what you get -- not an approximation of it.
        """
        cleaned = strip_path_quotes(path or "")
        if not cleaned or not os.path.isfile(cleaned):
            raise HTTPException(404, detail=[{
                "path": "video.video_init_path", "message": "No video at that path"}])
        try:
            info = probe_video(cleaned)
        except Exception as exc:
            raise HTTPException(422, detail=[{
                "path": "video.video_init_path",
                "message": f"Could not read that video: {exc}"}])

        width, height = info["width"], info["height"]
        if max_size > 0:
            width, height = fit_dimensions(info["width"], info["height"], max_size)

        nth = max(1, int(extract_nth_frame or 1))
        # Extraction keeps every nth frame, so this is the real ceiling on frame_range.
        renderable = math.ceil(info["frames"] / nth) if info["frames"] else 0

        return {
            "source": info,
            "render": {"width": width, "height": height},
            "extract_nth_frame": nth,
            "renderable_frames": renderable,
            # frame_range is INCLUSIVE at both ends, so the last valid index is n-1.
            "max_frame": max(0, renderable - 1),
        }

    @app.get("/api/video/thumbnail")
    def video_thumbnail(path: str = "", frame: int | None = None):
        """The first frame that is not essentially black.

        Clips that open on a fade-in would otherwise show a black thumbnail, which tells
        you nothing about the video you just picked.
        """
        cleaned = strip_path_quotes(path or "")
        if not cleaned or not os.path.isfile(cleaned):
            raise HTTPException(404, detail="No video at that path")
        try:
            index, pixels = (
                frame_at(cleaned, frame) if frame is not None
                else first_visible_frame(cleaned))
        except Exception as exc:
            raise HTTPException(422, detail=str(exc))

        from io import BytesIO
        image = Image.fromarray(pixels)
        image.thumbnail((480, 480))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=82)
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="image/jpeg",
                                 headers={"X-Frame-Index": str(index)})

    @app.post('/api/references/upload')
    async def upload_reference(request: Request, filename: str = 'reference.png'):
        """Store a browser-uploaded style reference in the portable app cache."""
        payload = await request.body()
        if not payload or len(payload) > 50 * 1024 * 1024:
            raise HTTPException(422, detail='Reference image must be between 1 B and 50 MB')
        try:
            image = Image.open(io.BytesIO(payload))
            image.load()
            image = image.convert('RGB')
        except Exception as exc:
            raise HTTPException(422, detail='The uploaded file is not a readable image') from exc
        root = system_settings.settings_path().parent / '.vibewarp_cache' / 'reference_uploads'
        root.mkdir(parents=True, exist_ok=True)
        stem = ''.join(
            char if char.isalnum() or char in '-_' else '_'
            for char in Path(filename).stem)[:60] or 'reference'
        target = root / f'{stem}-{uuid.uuid4().hex[:12]}.png'
        image.save(target, format='PNG')
        return {'path': str(target.resolve()), 'width': image.width, 'height': image.height}

    @app.get('/api/references/thumbnail')
    def reference_thumbnail(path: str = ''):
        cleaned = strip_path_quotes(path or '')
        if not cleaned or not os.path.isfile(cleaned):
            raise HTTPException(404, detail='Reference image does not exist')
        try:
            image = Image.open(cleaned)
            image.load()
            image = image.convert('RGB')
        except Exception as exc:
            raise HTTPException(422, detail='Could not read the reference image') from exc
        image.thumbnail((480, 480))
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=86)
        buffer.seek(0)
        return StreamingResponse(buffer, media_type='image/jpeg',
                                 headers={'Cache-Control': 'no-store'})

    @app.put("/api/system")
    async def put_system_settings(request: Request):
        """Persist the system tier next to the install.

        Model paths, VRAM tuning and thread counts describe the MACHINE, not the render,
        so they live in one file and are re-applied after every settings import.
        """
        return {"system": system_settings.save(await request.json())}

    @app.get("/api/controlnet/catalog")
    def controlnet_catalog(model_version: str = "", model_dir: str = ""):
        """Nets the engine can drive, plus which checkpoints are actually on disk.

        The UI used to hardcode its own list of 15 SD1.5 nets, so SDXL and the
        third-party nets were unreachable and a missing checkpoint only surfaced
        as a mid-render crash.
        """
        found = _controlnet_files(model_dir)
        by_name = {os.path.basename(path): path for path in found}
        nets = []
        for spec in specs_for_version(model_version):
            resolved = by_name.get(spec.filename) if spec.filename else None
            nets.append({
                "key": spec.key,
                "label": spec.label,
                "model_version": spec.model_version,
                "annotator": spec.annotator,
                "filename": spec.filename,
                "detectors": list(spec.detectors),
                "downloadable": bool(spec.url),
                "resolved_path": resolved,
            })
        return {"nets": nets, "files": found, "modes": list(CONTROLNET_MODES),
                "mode_presets": {name: {"layer_weights": weights, "zero_uncond": zero}
                                 for name, (weights, zero) in MODE_PRESETS.items()}}

    @app.get("/api/preview/runs")
    def preview_runs(output_dir: str = "images_out", batch_name: str = "warpfusion"):
        return {"runs": list_runs(output_dir, batch_name),
                "root": runs_root(output_dir, batch_name)}

    @app.get("/api/preview/runs/{run_id}")
    def preview_run(run_id: str, output_dir: str = "images_out",
                    batch_name: str = "warpfusion"):
        described = describe_run(output_dir, batch_name, run_id)
        if described is None:
            raise HTTPException(404, detail="Unknown run")
        return described

    @app.put("/api/preview/runs/{run_id}/label")
    async def preview_run_label(request: Request, run_id: str,
                                output_dir: str = "images_out",
                                batch_name: str = "warpfusion"):
        payload = await request.json()
        try:
            label = set_run_label(
                output_dir, batch_name, run_id, payload.get('label'))
        except FileNotFoundError:
            raise HTTPException(404, detail="Unknown run")
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc))
        return {'id': run_id, 'label': label}

    @app.post("/api/preview/runs/{run_id}/settings")
    def preview_run_settings(run_id: str, output_dir: str = "images_out",
                             batch_name: str = "warpfusion"):
        """Load an old run's saved settings back into the form.

        Goes through the same loader as an imported settings file, so a run's config comes
        back exactly as if you had exported and re-imported it -- including the system-tier
        overlay, so pulling in an old run cannot repoint this machine's model paths.
        """
        run = run_dir(output_dir, batch_name, run_id)
        if run is None:
            raise HTTPException(404, detail="Unknown run")
        path = settings_file(run)
        if path is None:
            raise HTTPException(404, detail=[{
                "path": "run", "message": f"Run {run_id} saved no settings"}])
        try:
            config = config_from_settings(path)
        except (ConfigError, ValueError, json.JSONDecodeError, TypeError) as exc:
            raise HTTPException(422, detail=[{"path": "run", "message": str(exc)}])
        return {"config": system_settings.overlay(asdict(_ui_config(config))),
                "source": os.path.basename(path)}

    @app.get("/api/preview/runs/{run_id}/image")
    def preview_image(run_id: str, layer: str, frame: int,
                      output_dir: str = "images_out", batch_name: str = "warpfusion"):
        path = image_path(output_dir, batch_name, run_id, layer, frame)
        if path is None:
            raise HTTPException(404, detail="No image for that layer/frame")
        return FileResponse(path, media_type=mimetypes.guess_type(path)[0])

    @app.get("/api/preview/runs/{run_id}/video")
    def preview_video(run_id: str, output_dir: str = "images_out",
                      batch_name: str = "warpfusion"):
        run = run_dir(output_dir, batch_name, run_id)
        path = video_file(run, batch_name) if run else None
        if path is None:
            raise HTTPException(404, detail="This run has no assembled video")
        return FileResponse(path, media_type="video/mp4")

    @app.post("/api/preview/runs/{run_id}/video", status_code=202)
    def build_preview_video(run_id: str, output_dir: str = "images_out",
                            batch_name: str = "warpfusion"):
        run = run_dir(output_dir, batch_name, run_id)
        if run is None:
            raise HTTPException(404, detail="Unknown run")
        described = describe_run(output_dir, batch_name, run_id)
        output = next((layer for layer in described['layers']
                       if layer['id'] == 'output'), None)
        if not output or not output['frames']:
            raise HTTPException(422, detail="This run has no completed frames")
        return jobs.submit_video(saved_run_config(run), run).snapshot(include_config=True)

    @app.post("/api/preview/runs/{run_id}/resume", status_code=202)
    def resume_preview_run(run_id: str, output_dir: str = "images_out",
                           batch_name: str = "warpfusion"):
        run = run_dir(output_dir, batch_name, run_id)
        if run is None:
            raise HTTPException(404, detail="Unknown run")
        config = saved_run_config(run)
        if config.animatediff.enabled:
            raise HTTPException(
                422, detail="Resuming AnimateDiff batches is not supported yet")
        frame = resume_status(run, batch_name, run_id)
        if frame is None:
            raise HTTPException(409, detail="This run already has every configured frame")
        return jobs.submit_resume(config, run, frame).snapshot(include_config=True)

    @app.post("/api/config/import")
    async def import_config(request: Request):
        payload = await request.json()
        content = payload.get("content")
        filename = os.path.basename(payload.get("filename") or "settings.json")
        if not isinstance(content, str):
            raise HTTPException(422, detail=[{"path": "file", "message": "Settings content is required"}])
        suffix = Path(filename).suffix.lower()
        if suffix not in {".json", ".txt"}:
            raise HTTPException(422, detail=[{"path": "file", "message": "Use a .json or .txt settings file"}])
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=suffix, encoding="utf-8", delete=False) as handle:
                handle.write(content)
                temp_path = handle.name
            config = config_from_settings(temp_path, models_root=payload.get("models_root"))
        except (ConfigError, ValueError, json.JSONDecodeError, TypeError) as exc:
            raise HTTPException(422, detail=[{"path": "file", "message": str(exc)}])
        finally:
            if temp_path:
                try: os.unlink(temp_path)
                except OSError: pass
        # Someone else's settings file must not repoint YOUR checkpoint directory or
        # retune YOUR VRAM. Re-apply this machine's system tier on top of the import.
        return {"config": system_settings.overlay(asdict(_ui_config(config)))}

    @app.post("/api/config/validate")
    async def validate(request: Request):
        return {"valid": True, "config": asdict(parse(await request.json()))}

    @app.post("/api/jobs", status_code=202)
    async def submit(request: Request):
        return jobs.submit(parse(await request.json())).snapshot(include_config=True)

    @app.get("/api/jobs")
    def list_jobs(): return jobs.list()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = jobs.get(job_id)
        if not job: raise HTTPException(404, detail="Unknown job")
        return job.snapshot(include_config=True)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        job = jobs.cancel(job_id)
        if not job: raise HTTPException(404, detail="Unknown job")
        return job.snapshot()

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str):
        job = jobs.get(job_id)
        if not job: raise HTTPException(404, detail="Unknown job")
        async def stream():
            revision = -1
            while True:
                snapshot = await asyncio.to_thread(jobs.wait_for_change, job, revision)
                revision = snapshot["revision"]
                yield f"data: {json.dumps(snapshot)}\n\n"
                if snapshot["state"] in TERMINAL_STATES: break
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/jobs/{job_id}/artifacts/{index}")
    def artifact(job_id: str, index: int):
        job = jobs.get(job_id)
        if not job: raise HTTPException(404, detail="Unknown job")
        if index < 0 or index >= len(job.artifacts): raise HTTPException(404, detail="Unknown artifact")
        path = os.path.abspath(job.artifacts[index])
        if not os.path.isfile(path): raise HTTPException(404, detail="Artifact no longer exists")
        return FileResponse(path, media_type=mimetypes.guess_type(path)[0])

    @app.get("/api/jobs/{job_id}/preview")
    def preview(job_id: str):
        job = jobs.get(job_id)
        if not job: raise HTTPException(404, detail="Unknown job")
        path = job.preview_path
        if not path or not os.path.isfile(path): raise HTTPException(404, detail="Preview is not ready")
        return FileResponse(path, media_type=mimetypes.guess_type(path)[0], headers={"Cache-Control": "no-store"})

    if STATIC_DIR.is_dir():
        if (STATIC_DIR / "assets").is_dir(): app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")
        @app.get("/{path:path}")
        def spa(path: str):
            candidate = (STATIC_DIR / path).resolve()
            if path and candidate.is_relative_to(STATIC_DIR.resolve()) and candidate.is_file(): return FileResponse(candidate)
            return FileResponse(STATIC_DIR / "index.html")
    else:
        @app.get("/")
        def missing_frontend(): return JSONResponse({"error": "Frontend assets are missing; run the web build."}, 503)
    return app

def main(argv=None):
    parser = argparse.ArgumentParser(prog="vibewarp-ui")
    parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=7860); parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--settings", help="JSON, VibeWarp, or WarpFusion settings file to prefill")
    parser.add_argument("--models-root", help="Model root used while translating WarpFusion settings")
    args = parser.parse_args(argv)
    initial_config = None
    if args.settings:
        try: initial_config = config_from_settings(args.settings, models_root=args.models_root)
        except Exception as exc: raise SystemExit(f"Could not load settings: {exc}")
    try: import uvicorn
    except ImportError: raise SystemExit("Install UI dependencies: pip install -e .[ui]")
    if not args.no_browser: Timer(1, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()
    uvicorn.run(create_app(initial_config=initial_config), host=args.host, port=args.port)

if __name__ == "__main__": main()
