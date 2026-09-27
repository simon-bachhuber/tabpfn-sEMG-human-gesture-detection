"""Portable, content-addressed caches: ordinary Git blobs, no external asset store."""

import gzip
import hashlib
import os
import re
import tempfile
import zlib
from pathlib import Path, PurePosixPath

from .common import ROOT, atomic_json, digest, read_json, settings, sha256, utc_now

CHUNK_BYTES = 32 * 1024 * 1024
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")


def destination(name):
    """Snapshots restore runtime files only, never source, credentials or symlinks."""
    relative = PurePosixPath(name)
    if (
        relative.is_absolute() or relative.as_posix() != name
        or not relative.parts or relative.parts[0] not in {"artifacts", "data"}
        or any(part in {"..", ".git"} or part.startswith(".env") for part in relative.parts)
    ):
        raise ValueError(f"Invalid snapshot destination: {name}")
    path = ROOT.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(ROOT.resolve()) or any(
        parent.is_symlink() for parent in [path, *path.parents] if parent != ROOT
    ):
        raise ValueError(f"Snapshot destination traverses a symlink: {name}")
    return path


def write_snapshot(directory, names):
    """Preserve original file bytes; split before compression to bound Git blobs."""
    directory = Path(directory)
    blob_directory = directory / "blobs"
    blob_directory.mkdir(parents=True, exist_ok=True)
    files, blobs = {}, {}
    for name in sorted(set(names)):
        path = destination(name)
        chunks, file_hash, size = [], hashlib.sha256(), 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
                file_hash.update(chunk)
                size += len(chunk)
                compressed = gzip.compress(chunk, compresslevel=1, mtime=0)
                identity = hashlib.sha256(compressed).hexdigest()
                target = blob_directory / (identity + ".gz")
                if not target.exists():
                    temporary = target.with_suffix(f".{os.getpid()}.tmp")
                    temporary.write_bytes(compressed)
                    temporary.replace(target)
                elif sha256(target) != identity:
                    raise ValueError(f"Snapshot blob integrity mismatch: {target}")
                blobs[identity] = {"size_bytes": len(compressed), "raw_size_bytes": len(chunk)}
                chunks.append(identity)
        files[name] = {"size_bytes": size, "sha256": file_hash.hexdigest(), "chunks": chunks}
    content = {
        "schema_version": 1, "configuration_sha256": digest(settings()),
        "chunk_bytes": CHUNK_BYTES, "files": files, "blobs": blobs,
        "dataset_license": "CC-BY-NC-4.0",
        "dataset_source": "https://biolab.put.poznan.pl/putemg-dataset/",
        "citation": "Kaczmarek, Mańkowski, Tomczyński (2019), Sensors 19(16), 3548. doi:10.3390/s19163548",
    }
    manifest = {"created_at": utc_now(), "snapshot_id": digest(content), **content}
    atomic_json(directory / "manifest.json", manifest)
    print(f"Snapshot: {len(files)} files; {sum(b['size_bytes'] for b in blobs.values()) / 1e6:.1f} MB compressed; "
          f"{sum(f['size_bytes'] for f in files.values()) / 1e6:.1f} MB restored", flush=True)
    return manifest


def load_manifest(directory):
    manifest = read_json(Path(directory) / "manifest.json")
    content = {key: value for key, value in manifest.items() if key not in {"snapshot_id", "created_at"}}
    if manifest["snapshot_id"] != digest(content) or manifest["schema_version"] != 1:
        raise ValueError("Snapshot manifest identity/schema mismatch")
    if manifest["configuration_sha256"] != digest(settings()):
        raise ValueError("Snapshot does not match the packaged comparison configuration")
    if not 0 < manifest["chunk_bytes"] <= CHUNK_BYTES:
        raise ValueError("Invalid snapshot chunk size")
    referenced = set()
    for name, entry in manifest["files"].items():
        destination(name)
        if not HASH_PATTERN.fullmatch(entry["sha256"]) or entry["size_bytes"] < 0:
            raise ValueError(f"Invalid snapshot file entry: {name}")
        referenced.update(entry["chunks"])
    if referenced != set(manifest["blobs"]):
        raise ValueError("Snapshot blob inventory differs from its file inventory")
    for identity, blob in manifest["blobs"].items():
        if not HASH_PATTERN.fullmatch(identity) or not 0 < blob["raw_size_bytes"] <= manifest["chunk_bytes"]:
            raise ValueError("Invalid snapshot blob entry")
        if not 0 < blob["size_bytes"] <= CHUNK_BYTES + 1024 * 1024:
            raise ValueError("Oversized snapshot blob")
    for name, entry in manifest["files"].items():
        if sum(manifest["blobs"][key]["raw_size_bytes"] for key in entry["chunks"]) != entry["size_bytes"]:
            raise ValueError(f"Snapshot chunk lengths disagree: {name}")
    return manifest


def read_chunk(directory, identity, entry):
    path = Path(directory) / "blobs" / (identity + ".gz")
    if path.stat().st_size != entry["size_bytes"] or sha256(path) != identity:
        raise ValueError(f"Snapshot blob integrity mismatch: {identity}")
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    raw = decoder.decompress(path.read_bytes(), entry["raw_size_bytes"] + 1)
    if len(raw) != entry["raw_size_bytes"] or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError(f"Snapshot decompression/length mismatch: {identity}")
    return raw


def restore_snapshot(directory, verify_only=False):
    manifest = load_manifest(directory)
    # Check the complete committed payload, even when a workspace already has caches.
    for identity, entry in manifest["blobs"].items():
        read_chunk(directory, identity, entry)
    pending = []
    for name, entry in manifest["files"].items():
        path = destination(name)
        if path.exists():
            if path.stat().st_size != entry["size_bytes"] or sha256(path) != entry["sha256"]:
                raise ValueError(f"Existing file differs from the snapshot: {name}; use a fresh workspace")
        elif verify_only:
            raise FileNotFoundError(f"Missing restored file: {name}; run emgbench snapshot restore")
        else:
            pending.append((path, entry))
    for path, entry in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".snapshot-", delete=False) as stream:
                temporary = Path(stream.name)
                file_hash = hashlib.sha256()
                for identity in entry["chunks"]:
                    raw = read_chunk(directory, identity, manifest["blobs"][identity])
                    stream.write(raw)
                    file_hash.update(raw)
            if file_hash.hexdigest() != entry["sha256"]:
                raise ValueError(f"Restored file hash mismatch: {path}")
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print(f"Verified snapshot {manifest['snapshot_id']}: {len(manifest['files'])} files, {len(pending)} restored", flush=True)
    return manifest


def build_snapshot(directory):
    """Explicit allowlist: no raw signals, local overrides, credentials or old runs."""
    from .comparison import compare, paths_for
    from .model_cache import cache_paths
    from .replay import replay_files, validate_replay_cache
    from .training_replay import prepare_training_replay

    if compare(quote_only=True)["pending"]:
        raise ValueError("Complete all three comparisons before building a snapshot")
    prepare_training_replay(quote_only=True)
    catalog = validate_replay_cache()
    paths = set(replay_files(catalog))
    paths.update([ROOT / "artifacts/replay/manifest.json", ROOT / "artifacts/catalog.json"])
    for feature in (ROOT / "artifacts/features").glob("emg_gestures-*.parquet"):
        paths.update([feature, feature.with_suffix(".json"), ROOT / "data/raw/hdf5" / (feature.stem + ".hdf5.metadata.json")])
    for key in settings()["comparisons"]:
        directory_path = paths_for(key)[0]
        if key == "tabpfn-du":
            paths.add(directory_path / "models" / f"{settings()['name']}_Du.json")
        else:
            paths.update(cache_paths(key))
        for name in ("summary.json", "cost_quotes.json"):
            if (directory_path / name).exists():
                paths.add(directory_path / name)
    for name in ("cost_quote.json", "live_verification.json"):
        path = ROOT / "artifacts/training-replay" / name
        if path.exists():
            paths.add(path)
    return write_snapshot(directory, [path.relative_to(ROOT).as_posix() for path in paths])
