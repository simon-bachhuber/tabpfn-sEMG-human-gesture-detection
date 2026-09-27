"""Paths, content identities, and atomic experiment records."""

import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("EMGBENCH_WORKSPACE", Path.cwd())).resolve()
LABELS = [0, 1, 2, 3, 6, 7, 8, 9]
GESTURES = ["Idle", "Fist", "Flexion", "Extension", "Index pinch", "Middle pinch", "Ring pinch", "Small pinch"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def environment():
    names = ("numpy", "pandas", "scipy", "scikit-learn", "tables", "pyarrow", "tabpfn-client")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in names},
    }


def settings():
    return read_json(PACKAGE / "comparison.json")


def profile():
    return settings()["signal"]


def validate_feature_metadata(metadata):
    """Accept existing immutable caches only when their numerical recipe matches."""
    expected = profile()
    for key, value in expected.items():
        if metadata["profile"].get(key) != value:
            raise ValueError(f"Feature recipe mismatch: {key}")
    if metadata["provenance"]["upstream"] != settings()["upstream"]:
        raise ValueError("Feature cache uses a different reference implementation")
    return metadata


def provenance():
    return {
        "profile": profile(),
        "upstream": settings()["upstream"],
        "environment": environment(),
    }
