from contextlib import contextmanager

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from emgbench.common import atomic_json
from emgbench.data import download_asset, parse_record
from emgbench.replay import segments
from emgbench.server import create_app

STEM = "emg_gestures-08-sequential-2018-04-04-15-54-42-398"


def test_download_resume_and_hash_verification(tmp_path, monkeypatch):
    content = b"0123456789" * 20
    asset = {"url": "https://example.test/data", "size_bytes": len(content), "etag": "original"}
    target = tmp_path / "trial.hdf5"
    target.with_suffix(".hdf5.part").write_bytes(content[:17])
    atomic_json(target.with_suffix(".hdf5.part.json"), asset)

    @contextmanager
    def stream(method, url, headers, **kwargs):
        assert headers == {"Range": "bytes=17-"}
        yield httpx.Response(206, request=httpx.Request(method, url), headers={"Content-Range": "bytes 17-199/200"}, content=content[17:])

    monkeypatch.setattr(httpx, "stream", stream)
    download_asset(asset, target)
    assert target.read_bytes() == content
    download_asset(asset, target)
    target.write_bytes(b"x" * len(content))
    with pytest.raises(ValueError, match="integrity"):
        download_asset(asset, target)


def test_negative_video_time_and_excluded_labels_are_preserved():
    assert segments([-1, -1, 0, 1, 1], np.array([-1.5, -.5, .5, 1.5, 2.5])) == [
        {"start": -1.5, "end": .5, "label": -1},
        {"start": .5, "end": 1.5, "label": 0},
        {"start": 1.5, "end": 3.5, "label": 1},
    ]


def test_video_ranges_and_workspace_boundaries(tmp_path):
    (tmp_path / ".env").write_text("API_TOKEN=must-not-be-served")
    video = tmp_path / "data/raw/video-576p" / (STEM + ".mp4")
    video.parent.mkdir(parents=True)
    video.write_bytes(bytes(range(256)))
    atomic_json(tmp_path / "artifacts/replay" / (STEM + ".json"), {"recording": STEM})
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/video/" + STEM, headers={"Range": "bytes=10-19"})
        assert response.status_code == 206
        assert response.content == bytes(range(10, 20))
        assert response.headers["content-range"] == "bytes 10-19/256"
        assert client.get("/").status_code == 200
        assert client.get("/.env").status_code == 404
        assert client.get("/assets/../.env").status_code == 404
        assert client.get("/api/replay/invalid").status_code == 404
    with pytest.raises(ValueError):
        parse_record("../../.env")
