import json

import pytest

from emgbench import snapshot
from emgbench.common import digest


def fixture_snapshot(tmp_path, monkeypatch):
    source, restored, archive = [tmp_path / name for name in ("source", "restored", "snapshot")]
    monkeypatch.setattr(snapshot, "ROOT", source)
    monkeypatch.setattr(snapshot, "CHUNK_BYTES", 64)
    payload = bytes(range(256)) * 4
    path = source / "artifacts/features/example.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    snapshot.write_snapshot(archive, ["artifacts/features/example.parquet"])
    monkeypatch.setattr(snapshot, "ROOT", restored)
    return archive, restored, payload


def test_round_trip_chunked_binary_without_source(tmp_path, monkeypatch):
    archive, restored, payload = fixture_snapshot(tmp_path, monkeypatch)
    (tmp_path / "source/artifacts/features/example.parquet").unlink()
    manifest = snapshot.restore_snapshot(archive)
    assert len(manifest["files"]["artifacts/features/example.parquet"]["chunks"]) > 1
    assert (restored / "artifacts/features/example.parquet").read_bytes() == payload
    snapshot.restore_snapshot(archive)  # Idempotent without touching verified caches.
    snapshot.restore_snapshot(archive, verify_only=True)


def test_corrupt_blob_fails_before_restoration(tmp_path, monkeypatch):
    archive, restored, _ = fixture_snapshot(tmp_path, monkeypatch)
    blob = next((archive / "blobs").iterdir())
    blob.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="blob integrity"):
        snapshot.restore_snapshot(archive)
    assert not restored.exists()


def test_restore_does_not_overwrite_work(tmp_path, monkeypatch):
    archive, restored, _ = fixture_snapshot(tmp_path, monkeypatch)
    path = restored / "artifacts/features/example.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"new experiment")
    with pytest.raises(ValueError, match="Existing file differs"):
        snapshot.restore_snapshot(archive)
    assert path.read_bytes() == b"new experiment"


@pytest.mark.parametrize("name", ["../outside", "data/../../outside", "src/emgbench/common.py", "data/.env", "/tmp/outside"])
def test_manifest_cannot_write_outside_runtime_paths(tmp_path, monkeypatch, name):
    archive, _, _ = fixture_snapshot(tmp_path, monkeypatch)
    manifest_path = archive / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][name] = manifest["files"].pop("artifacts/features/example.parquet")
    manifest["snapshot_id"] = digest({key: value for key, value in manifest.items() if key not in {"created_at", "snapshot_id"}})
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Invalid snapshot destination"):
        snapshot.restore_snapshot(archive)


def test_restore_rejects_symlink_destination(tmp_path, monkeypatch):
    archive, restored, _ = fixture_snapshot(tmp_path, monkeypatch)
    restored.mkdir()
    (restored / "artifacts").symlink_to(tmp_path / "source/artifacts", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        snapshot.restore_snapshot(archive)
