import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import emgbench.model_cache as model_cache
import emgbench.training_replay as training_replay
from emgbench.common import atomic_json, digest, settings, sha256


def test_training_replays_select_only_training_users_first_day():
    names = [
        f"emg_gestures-{subject}-{trajectory}-{date}-11-05-00-695"
        for subject in settings()["train_subjects"]
        for date in ("2018-05-11", "2018-06-14")
        for trajectory in ("repeats_long", "repeats_short", "sequential")
    ]
    selected = training_replay.training_recordings(SimpleNamespace(split={"train": names}))
    assert len(selected) == 4
    assert [name.split("-")[1] for name in selected] == ["03", "04", "05", "06"]
    assert all("-sequential-2018-05-11-" in name for name in selected)
    assert set(selected) <= set(names)


def test_held_out_users_cannot_be_labeled_as_training(monkeypatch):
    config = copy.deepcopy(settings())
    config["demo"]["training_subjects"] = ["03", "04", "05", "08"]
    monkeypatch.setattr(training_replay, "settings", lambda: config)
    with pytest.raises(ValueError, match="training-pool"):
        training_replay.training_subjects()


def test_classical_cache_is_bound_to_training_identity_and_file_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(model_cache, "ROOT", tmp_path)
    X = np.random.default_rng(7).normal(size=(20, 3))
    y = np.repeat([0, 1], 10)
    pipeline = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()).fit(X, y)
    model_cache.save_classical("lda-du", pipeline, "training-A", {"method": "test"})
    restored, _ = model_cache.load_classical("lda-du", "training-A")
    np.testing.assert_array_equal(restored.predict(X), pipeline.predict(X))
    with pytest.raises(ValueError, match="identity/integrity"):
        model_cache.load_classical("lda-du", "different-training-data")
    path, _ = model_cache.cache_paths("lda-du")
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="identity/integrity"):
        model_cache.load_classical("lda-du", "training-A")


def test_training_prediction_metadata_rejects_test_recordings(tmp_path, monkeypatch):
    monkeypatch.setattr(training_replay, "ROOT", tmp_path)
    names = [f"emg_gestures-{s}-sequential-2018-05-11-11-05-00-695" for s in ["03", "04", "05", "06"]]
    reference = {"experiment_id": "reference", "prediction_sha256": "heldout-hash", "model": "LDA", "fold": {"train": names}}
    monkeypatch.setattr(training_replay, "load_result", lambda key: (reference, {"dataset_id": "dataset"}, None))
    path, meta = training_replay.prediction_paths("lda-du")
    path.parent.mkdir(parents=True)
    pd.DataFrame({"recording": names, "window_id": names}).to_parquet(path, index=False)
    metadata = {
        "reference_experiment_id": "reference", "reference_prediction_sha256": "heldout-hash",
        "dataset_id": "dataset", "subjects": ["03", "04", "05", "06"],
        "recordings": names, "rows": 4, "split_role": "training", "prediction_sha256": sha256(path),
    }
    atomic_json(meta, metadata)
    assert len(training_replay.load_predictions("lda-du")[1]) == 4
    metadata["recordings"] = ["emg_gestures-08-sequential-2018-05-11-11-05-00-695"]
    atomic_json(meta, metadata)
    with pytest.raises(ValueError, match="outside the training pool"):
        training_replay.load_predictions("lda-du")


def test_cached_training_predictions_do_not_consume_api_tokens(monkeypatch, tmp_path):
    monkeypatch.setattr(training_replay, "ROOT", tmp_path)
    names = [f"emg_gestures-{s}-sequential-2018-05-11-11-05-00-695" for s in ["03", "04", "05", "06"]]
    X = pd.DataFrame({"feature": [1., 2., 3., 4.]})
    rows = pd.DataFrame({"recording": names, "window_id": names})
    side = (X, pd.Series([0, 1, 0, 1]), rows, 0)
    dataset = SimpleNamespace(identity="dataset", tables=lambda name: (side, side))
    monkeypatch.setattr(training_replay, "load_dataset", lambda: dataset)
    monkeypatch.setattr(training_replay, "training_recordings", lambda data: names)
    monkeypatch.setattr(training_replay, "verify_cached_inputs", lambda *args: {"experiment_id": "reference"})
    identity = digest({"reference": "reference", "dataset": "dataset", "window_ids": names, "features": ["feature"]})
    monkeypatch.setattr(training_replay, "load_predictions", lambda key: ({"input_identity": identity}, rows))
    monkeypatch.setattr(training_replay, "fetch_training_videos", lambda recordings: None)
    monkeypatch.setattr(training_replay, "authentication", lambda: pytest.fail("Cached replay must not authenticate"))
    path, meta = training_replay.prediction_paths("tabpfn-du")
    path.parent.mkdir(parents=True)
    path.touch()
    meta.touch()
    assert training_replay.prepare_training_replay(["tabpfn-du"])["completed"] == ["tabpfn-du"]
