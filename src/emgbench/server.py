"""Local replay server; only exported bundles, static assets, and selected videos."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .common import PACKAGE, ROOT, read_json
from .data import parse_record


def create_app(root=ROOT):
    root = Path(root)
    app = FastAPI(title="putEMG · Replay Lab")
    web = PACKAGE / "web"
    app.mount("/assets", StaticFiles(directory=web), name="assets")

    def asset(recording, directory, suffix):
        try:
            parse_record(recording)
        except ValueError as error:
            raise HTTPException(404, "Unknown recording") from error
        base = (root / directory).resolve()
        path = (base / (recording + suffix)).resolve()
        if path.parent != base or not path.is_file():
            raise HTTPException(404, "Artifact unavailable; export replay first")
        return path

    @app.get("/")
    def index():
        return FileResponse(web / "index.html")

    @app.get("/api/recordings")
    def recordings():
        path = root / "artifacts/replay/catalog.json"
        return read_json(path) if path.exists() else []

    @app.get("/api/replay/{recording}")
    def replay(recording: str):
        return FileResponse(asset(recording, "artifacts/replay", ".json"), media_type="application/json")

    @app.get("/api/traces/{recording}")
    def traces(recording: str):
        return FileResponse(asset(recording, "artifacts/replay", ".f32"), media_type="application/octet-stream")

    @app.get("/video/{recording}")
    def video(recording: str):
        asset(recording, "artifacts/replay", ".json")  # Only recordings exported for replay.
        return FileResponse(asset(recording, "data/raw/video-576p", ".mp4"), media_type="video/mp4")

    return app
