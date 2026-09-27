"""Validate raw recordings and cache authors-compatible signal features."""

import json
import os
import subprocess
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from threadpoolctl import threadpool_limits

from .common import (
    LABELS,
    ROOT,
    atomic_json,
    digest,
    profile,
    provenance,
    read_json,
    sha256,
    utc_now,
    validate_feature_metadata,
)
from .data import parse_record
from .reference import upstream

FEATURE_ORDER = ["RMS", "WL", "ZC", "SSC", "IAV", "VAR", "WAMP"]


def validate_frame(frame, stem):
    record = parse_record(stem)
    channels = [c for c in frame if c.startswith("EMG_")]
    if channels != [f"EMG_{i}" for i in range(1, 25)]:
        raise ValueError(f"Unexpected channel order in {stem}: {channels}")
    required = {"TRAJ_1", "TRAJ_GT", "VIDEO_STAMP", "type", "subject", "trajectory", "date_time"}
    if not required <= set(frame):
        raise ValueError(f"Missing columns: {required - set(frame)}")
    if len(frame) < profile()["window_samples"]:
        raise ValueError(f"Recording too short: {stem}")
    time = frame.index.to_numpy(dtype=float)
    video = frame.VIDEO_STAMP.to_numpy(dtype=float)
    if not np.isfinite(time).all() or not (np.diff(time) > 0).all():
        raise ValueError(f"Invalid/nonmonotonic sample time: {stem}")
    if not np.isfinite(video).all() or not (np.diff(video) >= 0).all():
        raise ValueError(f"Invalid/nonmonotonic video time: {stem}")
    if not np.isfinite(frame[channels].to_numpy()).all():
        raise ValueError(f"Nonfinite EMG signal: {stem}")
    labels = frame.TRAJ_GT.to_numpy()
    if not np.isin(labels, [-1, *LABELS]).all():
        raise ValueError(f"Unexpected target values: {np.unique(labels)}")
    if set(frame["subject"].astype(int)) != {int(record["subject"])}:
        raise ValueError(f"Subject metadata mismatch: {stem}")
    if set(frame["trajectory"].astype(str)) != {record["trajectory"]}:
        raise ValueError(f"Trajectory metadata mismatch: {stem}")
    if set(pd.to_datetime(frame["date_time"]).dt.strftime("%Y-%m-%d")) != {record["date"]}:
        raise ValueError(f"Session metadata mismatch: {stem}")
    offset = video - time
    counts = {str(int(k)): int(v) for k, v in frame.TRAJ_GT.value_counts().items()}
    return {
        **record, "samples": len(frame), "columns": list(frame),
        "label_sample_counts": counts,
        "duration_seconds": float(time[-1] - time[0]),
        "observed_sample_rate_hz": float(1 / np.median(np.diff(time))),
        "sample_time_start": float(time[0]), "sample_time_end": float(time[-1]),
        "video_stamp_start": float(video[0]), "video_stamp_end": float(video[-1]),
        "video_minus_sample_seconds": float(np.median(offset)),
        "video_offset_range_seconds": float(np.ptp(offset)),
    }


def probe_video(path):
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,time_base,start_time,duration,nb_frames",
        "-show_entries", "format=duration,start_time", "-of", "json", str(path),
    ]
    return json.loads(subprocess.check_output(command, text=True))


def extract_features(frame, window=2500, hop=1250):
    """Legacy formulas, including signed WL/WAMP and the ZC deadband rule."""
    endpoints = np.arange(window - 1, len(frame), hop)
    values = {feature: {} for feature in FEATURE_ORDER}
    for channel in [c for c in frame if c.startswith("EMG_")]:
        suffix = channel.split("_")[1]
        windows = sliding_window_view(frame[channel].to_numpy(), window)[::hop]
        differences = np.diff(windows, axis=1)
        values["RMS"][f"RMS_{suffix}"] = np.sqrt(np.mean(np.square(windows), axis=1))
        values["WL"][f"WL_{suffix}"] = np.sum(differences, axis=1)
        values["ZC"][f"ZC_{suffix}"] = np.asarray([
            np.sum(np.diff(row[np.abs(row) > 30] > 0)) for row in windows
        ])
        values["SSC"][f"SSC_{suffix}"] = np.sum(differences[:, :-1] * differences[:, 1:] <= -16, axis=1)
        values["IAV"][f"IAV_{suffix}"] = np.sum(np.abs(windows), axis=1)
        values["VAR"][f"VAR_{suffix}"] = np.var(windows, axis=1)
        values["WAMP"][f"WAMP_{suffix}"] = np.sum(differences >= 10, axis=1)
    result = pd.DataFrame({k: v for feature in FEATURE_ORDER for k, v in values[feature].items()}, index=frame.index[endpoints])
    metadata = frame.iloc[endpoints].drop(columns=[c for c in frame if c.startswith("EMG_")])
    result = pd.concat([result, metadata], axis=1)
    result["sample_start"] = endpoints - window + 1
    result["sample_end"] = endpoints
    result["sample_time"] = frame.index[endpoints].to_numpy()
    raw_labels = sliding_window_view(frame.TRAJ_GT.to_numpy(), window)[::hop]
    result["label_pure_window"] = np.all(raw_labels == raw_labels[:, -1:], axis=1)
    prompts = frame.TRAJ_1.to_numpy()
    occurrences = np.cumsum(np.r_[True, prompts[1:] != prompts[:-1]])
    result["prompt_occurrence"] = occurrences[endpoints]
    return result


def feature_path(stem):
    return ROOT / "artifacts/features" / (stem + ".parquet")


def filter_identity(input_hash):
    source = provenance()
    return digest({
        "input_sha256": input_hash, "upstream": source["upstream"],
        "environment": source["environment"], "filter_profile": "authors_unmodified_pre_process_v1",
    })


def same_filter_environment(previous, current):
    """Classifier/API upgrades do not change cached signal-filter computations."""
    if previous["upstream"] != current["upstream"]:
        return False
    old, new = previous["environment"], current["environment"]
    return (
        old["python"] == new["python"] and old["platform"] == new["platform"]
        and all(old["packages"][p] == new["packages"][p] for p in ("numpy", "pandas", "scipy"))
    )


def _atomic_npy(path, values):
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    temp.replace(path)


def process_recording(path):
    path = Path(path)
    source_hash = sha256(path)
    input_metadata = read_json(path.with_suffix(".hdf5.metadata.json"))
    if source_hash != input_metadata["sha256"]:
        raise ValueError(f"Input hash mismatch: {path}")
    settings = profile()
    key = filter_identity(source_hash)
    output = feature_path(path.stem)
    metadata_path = output.with_suffix(".json")
    if metadata_path.exists():
        previous = read_json(metadata_path)
        if previous["input_sha256"] == source_hash and same_filter_environment(previous["provenance"], provenance()):
            key = previous["filter_key"]
    feature_key = digest({"filter_key": key, "profile": settings, "extractor_sha256": sha256(Path(__file__))})
    if output.exists() and metadata_path.exists():
        metadata = validate_feature_metadata(read_json(metadata_path))
        if metadata["input_sha256"] == source_hash and sha256(output) == metadata["feature_sha256"]:
            print(f"Cached features {path.stem}", flush=True)
            return metadata

    _, _, filtering = upstream()
    frame = pd.read_hdf(path)
    validation = validate_frame(frame, path.stem)
    cache = ROOT / "artifacts/filtered" / path.stem / key[:16]
    cache.mkdir(parents=True, exist_ok=True)
    channel_reports = []
    with threadpool_limits(limits=1):
        for channel in [c for c in frame if c.startswith("EMG_")]:
            channel_path = cache / (channel + ".npy")
            report_path = cache / (channel + ".json")
            if channel_path.exists() and report_path.exists():
                stats = read_json(report_path)
                if sha256(channel_path) != stats["sha256"]:
                    raise ValueError(f"Filter checkpoint hash mismatch: {channel_path}")
                filtered = np.load(channel_path, allow_pickle=False)
            else:
                stats = {"channel": channel, "optimizer_calls": 0, "optimizer_failures": 0}
                original_minimize = filtering.minimize

                def monitored_minimize(*args, **kwargs):
                    result = original_minimize(*args, **kwargs)
                    stats["optimizer_calls"] += 1
                    stats["optimizer_failures"] += int(not result.success)
                    return result

                start = perf_counter()
                filtering.minimize = monitored_minimize
                try:
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter("always")
                        filtered = filtering.pre_process(frame[channel])
                    stats["warnings"] = sorted({str(w.message) for w in captured})
                finally:
                    filtering.minimize = original_minimize
                stats["seconds"] = perf_counter() - start
                if not np.isfinite(filtered).all():
                    raise ValueError(f"Nonfinite filtered signal: {path.stem}/{channel}")
                _atomic_npy(channel_path, filtered)
                stats["sha256"] = sha256(channel_path)
                atomic_json(report_path, stats)
                print(f"Filtered {path.stem}/{channel}: {stats['seconds']:.1f}s", flush=True)
            if len(filtered) != len(frame):
                raise ValueError(f"Wrong checkpoint length: {channel_path}")
            frame[channel] = filtered
            channel_reports.append(stats)

        start = perf_counter()
        features = extract_features(frame, settings["window_samples"], settings["hop_samples"])
        feature_seconds = perf_counter() - start
    features["recording"] = path.stem
    features["window_id"] = [f"{path.stem}:{a}:{b}" for a, b in zip(features.sample_start, features.sample_end)]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.tmp")
    features.to_parquet(temporary, index=False)
    temporary.replace(output)
    metadata = {
        "recording": path.stem, "created_at": utc_now(), "input_sha256": source_hash,
        "filter_key": key, "feature_key": feature_key, "feature_sha256": sha256(output),
        "profile": settings, "provenance": provenance(), "validation": validation,
        "windows": len(features), "feature_columns": [c for c in features if c.split("_")[0] in FEATURE_ORDER],
        "channels": channel_reports, "filter_seconds": sum(c["seconds"] for c in channel_reports),
        "feature_seconds": feature_seconds, "filter_cache": str(cache.relative_to(ROOT)),
    }
    atomic_json(metadata_path, metadata)
    print(f"Features complete {path.stem}: {len(features)} windows", flush=True)
    return metadata
