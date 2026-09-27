"""The single fixed 40/4 split and identical sample support for all pipelines."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

from .common import ROOT, digest, read_json, settings, sha256, validate_feature_metadata
from .data import TRAJECTORIES, parse_record


def subjects_of(recordings):
    return sorted({parse_record(name)["subject"] for name in recordings})


def split_recordings(recordings):
    config = settings()
    train_ids, test_ids = config["train_subjects"], config["test_subjects"]
    if len(set(train_ids)) != 40 or len(set(test_ids)) != 4 or set(train_ids) & set(test_ids):
        raise ValueError("The configuration must contain 40 training and four disjoint test users")
    recordings = sorted(recordings)
    if len(recordings) != 264 or len(set(recordings)) != 264:
        raise ValueError("All 264 unique recordings are required")
    if set(subjects_of(recordings)) != set(train_ids + test_ids):
        raise ValueError("Recording participants differ from the frozen split")
    groups, days = defaultdict(list), defaultdict(set)
    for name in recordings:
        record = parse_record(name)
        groups[(record["subject"], record["date"])].append(record["trajectory"])
        days[record["subject"]].add(record["date"])
    if any(len(values) != 2 for values in days.values()):
        raise ValueError("Both recording days are required for every participant")
    if any(len(values) != 3 or set(values) != TRAJECTORIES for values in groups.values()):
        raise ValueError("Each day must contain all three trajectories")
    train = [name for name in recordings if parse_record(name)["subject"] in train_ids]
    test = [name for name in recordings if parse_record(name)["subject"] in test_ids]
    if len(train) != 240 or len(test) != 24:
        raise ValueError("Expected 240 training and 24 test recordings")
    return {
        "id": config["name"], "train": train, "test": test,
        "train_subjects": train_ids, "test_subjects": test_ids,
        "test_calibration_rows": 0,
    }


def masked_labels(labels):
    """Original mask, applied separately to each recording before pooling.

    A positive-label change at i rejects i-1 and i. A -1 interval rejects
    itself and four following feature rows. These are the released semantics.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("Expected a nonempty one-dimensional label sequence")
    numeric = labels.astype(float)
    numeric[numeric < 0] = np.nan
    changes = np.r_[0, np.diff(numeric)]
    changes[np.isnan(changes)] = 0
    starts = (changes != 0) & (labels > 0)
    rejected = binary_dilation(starts, structure=[1, 1, 0])
    rejected |= binary_dilation(starts, structure=[1, 0, 0])
    pauses = binary_dilation(labels == -1, structure=[0, 1, 1], iterations=4)
    output = labels.copy()
    output[rejected] = -5
    output[pauses] = -6
    return output


def feature_columns(frame, representation):
    members = settings()["representations"].get(representation)
    if members is None:
        raise ValueError(f"Unsupported representation: {representation}")
    expected = [f"{feature}_{channel}" for feature in members for channel in range(1, 25)]
    if [column for column in frame if column in expected] != expected:
        raise ValueError(f"Missing or misordered {representation} features")
    return expected


def prepare_side(frames, recordings, representation):
    columns = feature_columns(frames[recordings[0]], representation)
    for name in recordings:
        if feature_columns(frames[name], representation) != columns:
            raise ValueError(f"Inconsistent feature schema: {name}")
    combined = pd.concat([frames[name] for name in recordings], ignore_index=True)
    combined["masked_label"] = np.concatenate([
        masked_labels(frames[name].TRAJ_GT.to_numpy()) for name in recordings
    ])
    kept = combined.loc[combined.masked_label >= 0].reset_index(drop=True)
    X = kept[columns].rename(columns={c: f"input_{i}_{c}" for i, c in enumerate(columns)})
    if kept.empty or not np.isfinite(X.to_numpy()).all() or not kept.window_id.is_unique:
        raise ValueError("Invalid or duplicate retained model inputs")
    return X, kept.masked_label.astype(int), kept, len(combined) - len(kept)


@dataclass
class Dataset:
    frames: dict
    metadata: dict
    split: dict
    identity: str

    def tables(self, representation):
        train = prepare_side(self.frames, self.split["train"], representation)
        test = prepare_side(self.frames, self.split["test"], representation)
        if set(train[2].window_id) & set(test[2].window_id):
            raise ValueError("Training and test sample support overlap")
        return train, test


def load_dataset():
    paths = sorted((ROOT / "artifacts/features").glob("emg_gestures-*.parquet"))
    split = split_recordings([path.stem for path in paths])
    frames, metadata = {}, {}
    for path in paths:
        info = validate_feature_metadata(read_json(path.with_suffix(".json")))
        if sha256(path) != info["feature_sha256"]:
            raise ValueError(f"Feature cache integrity mismatch: {path}")
        frame = pd.read_parquet(path)
        if not frame.recording.eq(path.stem).all():
            raise ValueError(f"Wrong recording identity in {path}")
        frames[path.stem], metadata[path.stem] = frame, info
    identity = digest({name: info["feature_sha256"] for name, info in metadata.items()})
    return Dataset(frames, metadata, split, identity)
