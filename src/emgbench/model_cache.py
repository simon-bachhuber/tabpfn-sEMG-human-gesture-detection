"""Local fitted classical pipelines, tied to their complete pooled training input."""

import os

import joblib

from .common import (
    ROOT,
    atomic_json,
    digest,
    environment,
    read_json,
    settings,
    sha256,
    utc_now,
)


def training_identity(key, dataset, train):
    spec = settings()["comparisons"][key]
    return digest({
        "dataset_id": dataset.identity, "comparison": spec,
        "train_recordings": dataset.split["train"],
        "window_ids": train[2].window_id.tolist(), "feature_columns": list(train[0]),
    })


def cache_paths(key):
    directory = ROOT / "artifacts/runs" / settings()["comparisons"][key]["run"] / "models"
    return directory / "pooled.joblib", directory / "pooled.metadata.json"


def save_classical(key, pipeline, identity, verification):
    path, metadata_path = cache_paths(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    joblib.dump(pipeline, temporary)
    temporary.replace(path)
    metadata = {
        "created_at": utc_now(), "training_identity": identity, "sha256": sha256(path),
        "environment": environment(), "comparison": settings()["comparisons"][key],
        "fit_parameters": pipeline.steps[-1][1].get_params(deep=False),
        "verification": verification,
    }
    atomic_json(metadata_path, metadata)
    return metadata


def load_classical(key, identity):
    path, metadata_path = cache_paths(key)
    if not path.exists() or not metadata_path.exists():
        raise FileNotFoundError(path)
    metadata = read_json(metadata_path)
    if metadata["training_identity"] != identity or sha256(path) != metadata["sha256"]:
        raise ValueError(f"Fitted classical model identity/integrity mismatch: {path}")
    return joblib.load(path), metadata
