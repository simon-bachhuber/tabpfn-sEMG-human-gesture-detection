"""Replay the three pooled models with explicit training/held-out provenance."""

import os

import numpy as np
import pandas as pd

from .common import (
    GESTURES,
    LABELS,
    ROOT,
    atomic_json,
    digest,
    read_json,
    settings,
    sha256,
    utc_now,
)
from .comparison import load_result, paths_for
from .data import parse_record
from .metrics import score
from .processing import feature_path, probe_video
from .protocol import masked_labels
from .training_replay import load_predictions as load_training_predictions
from .training_replay import prediction_paths, training_subjects


def segments(labels, timestamps):
    values, times = np.asarray(labels, dtype=int), np.asarray(timestamps, dtype=float)
    starts = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    ends = np.r_[starts[1:], len(values)]
    dt = float(np.median(np.diff(times)))
    return [
        {"start": float(times[a]), "end": float(times[b]) if b < len(times) else float(times[-1] + dt), "label": int(values[a])}
        for a, b in zip(starts, ends)
    ]


def replay_files(catalog):
    """Everything needed to serve a replay, plus its feature/prediction inputs."""
    paths = {ROOT / "artifacts/replay/catalog.json"}
    for key in settings()["comparisons"]:
        directory, result, predictions = paths_for(key)
        paths.update([directory / "manifest.json", result, predictions, *prediction_paths(key)])
    for record in catalog:
        stem = record["recording"]
        parse_record(stem)
        paths.update([
            ROOT / "artifacts/replay" / (stem + ".json"),
            ROOT / "artifacts/replay" / (stem + ".f32"),
            ROOT / "data/raw/video-576p" / (stem + ".mp4"),
            ROOT / "data/raw/video-576p" / (stem + ".mp4.metadata.json"),
            feature_path(stem), feature_path(stem).with_suffix(".json"),
        ])
    return sorted(paths)


def validate_replay_cache():
    output = ROOT / "artifacts/replay"
    manifest = read_json(output / "manifest.json")
    if manifest["schema_version"] != 1 or manifest["configuration_sha256"] != digest(settings()):
        raise ValueError("Replay configuration changed; use emgbench demo --rebuild with raw/filter inputs")
    catalog = read_json(output / "catalog.json")
    expected = {"held_out": set(settings()["test_subjects"]), "training": set(training_subjects())}
    if len(catalog) != 8 or any(
        {record["subject"] for record in catalog if record["split_role"] == role} != subjects
        for role, subjects in expected.items()
    ):
        raise ValueError("Replay must cover four held-out and four training participants")
    paths = replay_files(catalog)
    if set(manifest["files"]) != {str(path.relative_to(ROOT)) for path in paths}:
        raise ValueError("Replay manifest has missing or unexpected dependencies")
    for path in paths:
        if not path.is_file() or sha256(path) != manifest["files"][str(path.relative_to(ROOT))]:
            raise ValueError(f"Replay cache integrity mismatch: {path}; restore the snapshot or rebuild")
    return catalog


def export_replay(rebuild=False):
    config = settings()
    output = ROOT / "artifacts/replay"
    if not rebuild and (output / "manifest.json").exists():
        catalog = validate_replay_cache()
        print(f"Verified {len(catalog)} cached replays; raw signals and filtering are not needed", flush=True)
        return catalog
    output.mkdir(parents=True, exist_ok=True)
    results = {key: load_result(key) for key in config["comparisons"]}
    training_results = {key: load_training_predictions(key) for key in config["comparisons"]}
    training_ids = training_subjects()
    training_records = set(next(iter(training_results.values()))[0]["recordings"])
    if any(set(metadata["recordings"]) != training_records for metadata, _ in training_results.values()):
        raise ValueError("Training replay recordings differ across pipelines")
    videos = [
        path for path in sorted((ROOT / "data/raw/video-576p").glob("emg_gestures-*.mp4"))
        if parse_record(path.stem)["subject"] in config["test_subjects"] or path.stem in training_records
    ]
    if {parse_record(path.stem)["subject"] for path in videos} != set(config["test_subjects"] + training_ids):
        raise ValueError("Videos for all four held-out and four training participants are required")
    catalog = []
    for video_path in videos:
        stem = video_path.stem
        info = read_json(feature_path(stem).with_suffix(".json"))
        frame = pd.read_parquet(feature_path(stem))
        raw = pd.read_hdf(ROOT / "data/raw/hdf5" / (stem + ".hdf5"))
        metadata = info["validation"]
        role = "training" if metadata["subject"] in training_ids else "held_out"
        mask = masked_labels(frame.TRAJ_GT.to_numpy())
        expected = frame.loc[mask >= 0]
        positions = {window: i for i, window in enumerate(frame.window_id)}
        models = []
        for key, spec in config["comparisons"].items():
            result, _, all_predictions = results[key]
            prediction_hash = result["prediction_sha256"]
            if role == "training":
                training_metadata, all_predictions = training_results[key]
                prediction_hash = training_metadata["prediction_sha256"]
                if stem not in result["fold"]["train"]:
                    raise ValueError(f"Training video is outside the model's training pool: {stem}")
            elif stem not in result["fold"]["test"]:
                raise ValueError(f"Video is not held out by {key}: {stem}")
            predictions = all_predictions.loc[all_predictions.recording == stem]
            np.testing.assert_array_equal(predictions.window_id, expected.window_id)
            np.testing.assert_array_equal(predictions.TRAJ_GT, expected.TRAJ_GT)
            predicted, probabilities = [None] * len(frame), [None] * len(frame)
            for row in predictions.to_dict(orient="records"):
                index = positions[row["window_id"]]
                predicted[index] = int(row["prediction"])
                if "probability_0" in row:
                    probabilities[index] = [round(float(row[f"probability_{label}"]), 7) for label in LABELS]
            models.append({
                "id": key, "name": spec["model"], "feature_set": spec["representation"],
                "label": spec["label"], "fit_id": result.get("fitted_model_id") or result["experiment_id"],
                "split_role": role,
                "predictions": predicted, "probabilities": probabilities,
                "metrics": score(predictions.TRAJ_GT, predictions.prediction),
                "train_subjects": config["train_subjects"], "test_subjects": config["test_subjects"],
                "test_calibration_rows": 0, "prediction_sha256": prediction_hash,
            })
        bin_samples = 64
        count = len(raw) // bin_samples
        trace = np.empty((count, 49), dtype="<f4")
        trace[:, 0] = raw.VIDEO_STAMP.iloc[np.arange(count) * bin_samples + bin_samples // 2].to_numpy()
        cache = ROOT / info["filter_cache"]
        channel_info = {item["channel"]: item for item in info["channels"]}
        scales = []
        for channel in range(1, 25):
            path = cache / f"EMG_{channel}.npy"
            if sha256(path) != channel_info[f"EMG_{channel}"]["sha256"]:
                raise ValueError(f"Filtered trace integrity mismatch: {path}")
            signal = np.load(path, mmap_mode="r", allow_pickle=False)
            bins = signal[:count * bin_samples].reshape(count, bin_samples)
            trace[:, channel * 2 - 1], trace[:, channel * 2] = bins.min(axis=1), bins.max(axis=1)
            scales.append(float(max(1.0, np.quantile(np.abs(signal[::16]), .995))))
        trace_path = output / (stem + ".f32")
        temporary = trace_path.with_suffix(f".f32.{os.getpid()}.tmp")
        trace.tofile(temporary)
        temporary.replace(trace_path)
        bundle = {
            "recording": stem, "created_at": utc_now(), "subject": metadata["subject"],
            "date": metadata["date"], "trajectory": metadata["trajectory"],
            "split_role": role,
            "labels": LABELS, "gesture_names": GESTURES, "video": probe_video(video_path),
            "observed_sample_rate_hz": metadata["observed_sample_rate_hz"],
            "video_minus_sample_seconds": metadata["video_minus_sample_seconds"],
            "alignment": "VIDEO_STAMP seconds; predictions at window endpoints",
            "window_samples": 2500, "hop_samples": 1250,
            "trace": {"rows": count, "stride": 49, "dtype": "little-endian float32", "bin_samples": bin_samples, "scales": scales, "sha256": sha256(trace_path)},
            "windows": [
                {"time": float(row.VIDEO_STAMP), "start": float(raw.VIDEO_STAMP.iloc[int(row.sample_start)]), "label": int(row.TRAJ_GT), "mask": int(mask[i]), "pure": bool(row.label_pure_window)}
                for i, row in enumerate(frame.itertuples(index=False))
            ],
            "ground_truth": segments(raw.TRAJ_GT, raw.VIDEO_STAMP),
            "prompts": segments(raw.TRAJ_1, raw.VIDEO_STAMP),
            "models": sorted(models, key=lambda model: (model["id"] != "tabpfn-du", model["id"])),
            "dataset_license": "CC BY-NC 4.0",
            "dataset_source": "https://biolab.put.poznan.pl/putemg-dataset/",
        }
        atomic_json(output / (stem + ".json"), bundle)
        catalog.append({"recording": stem, "subject": metadata["subject"], "date": metadata["date"], "trajectory": metadata["trajectory"], "split_role": role})
        print(f"Exported {stem}: three comparison pipelines", flush=True)
    catalog.sort(key=lambda item: (item["split_role"] == "training", item["subject"], item["recording"]))
    atomic_json(output / "catalog.json", catalog)
    atomic_json(output / "manifest.json", {
        "schema_version": 1, "configuration_sha256": digest(config),
        "files": {str(path.relative_to(ROOT)): sha256(path) for path in replay_files(catalog)},
    })
    return catalog
