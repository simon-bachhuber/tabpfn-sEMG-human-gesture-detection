"""In-sample replay predictions, kept separate from held-out evaluation artifacts."""

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .common import (
    LABELS,
    ROOT,
    atomic_json,
    digest,
    read_json,
    settings,
    sha256,
    utc_now,
)
from .comparison import authentication, load_result, paths_for, verify_cached_inputs
from .data import catalog, download_asset, parse_record
from .metrics import score
from .model_cache import load_classical, save_classical, training_identity
from .protocol import load_dataset, subjects_of


def training_subjects():
    config = settings()
    subjects = config.get("demo", {}).get("training_subjects", [])
    if len(subjects) != 4 or len(set(subjects)) != 4 or not set(subjects) <= set(config["train_subjects"]):
        raise ValueError("The training replay requires four distinct training-pool participants")
    return subjects


def training_recordings(dataset):
    recordings = []
    for subject in training_subjects():
        candidates = [name for name in dataset.split["train"] if parse_record(name)["subject"] == subject]
        date = min(parse_record(name)["date"] for name in candidates)
        chosen = [name for name in candidates if parse_record(name)["date"] == date and parse_record(name)["trajectory"] == "sequential"]
        if len(chosen) != 1:
            raise ValueError(f"Missing first-day sequential recording for training user {subject}")
        recordings.extend(chosen)
    return recordings


def prediction_paths(key):
    directory = ROOT / "artifacts/training-replay"
    return directory / f"{key}.parquet", directory / f"{key}.json"


def fetch_training_videos(recordings):
    paths = [ROOT / "data/raw/video-576p" / (name + ".mp4") for name in recordings]
    if all(path.exists() and path.with_suffix(".mp4.metadata.json").exists() for path in paths):
        for path in paths:
            download_asset(read_json(path.with_suffix(".mp4.metadata.json")), path)
        return
    available = {item["recording"]: item for item in catalog()["recordings"]}
    for name in recordings:
        asset = available[name]["video"]
        if asset is None:
            raise ValueError(f"No 576p video for {name}")
        download_asset(asset, ROOT / "data/raw/video-576p" / asset["name"])


def load_predictions(key):
    path, metadata_path = prediction_paths(key)
    metadata = read_json(metadata_path)
    reference, manifest, _ = load_result(key)
    if metadata["reference_experiment_id"] != reference["experiment_id"] or metadata["dataset_id"] != manifest["dataset_id"]:
        raise ValueError("Training replay and held-out comparison use different pooled models/data")
    if metadata["reference_prediction_sha256"] != reference["prediction_sha256"]:
        raise ValueError("The referenced held-out predictions have changed")
    if reference["model"] == "TabPFN" and metadata["model_provenance"].get("model_id") != reference["fitted_model_id"]:
        raise ValueError("Training replay does not use the original TabPFN context")
    if not set(metadata["recordings"]) <= set(reference["fold"]["train"]):
        raise ValueError("Training replay includes recordings outside the training pool")
    if metadata["subjects"] != training_subjects() or metadata["split_role"] != "training":
        raise ValueError("Training replay participant configuration changed")
    if sha256(path) != metadata["prediction_sha256"]:
        raise ValueError(f"Training prediction integrity mismatch: {path}")
    data = pd.read_parquet(path)
    if len(data) != metadata["rows"] or not data.window_id.is_unique or set(data.recording) != set(metadata["recordings"]):
        raise ValueError("Missing or duplicate training replay predictions")
    return metadata, data


def prepare_training_replay(keys=None, max_estimated_tokens=None, quote_only=False, svm_cache_mb=2048):
    config = settings()
    keys = keys or list(config["comparisons"])
    if not set(keys) <= set(config["comparisons"]) or len(keys) != len(set(keys)):
        raise ValueError("Unsupported or duplicate comparison selection")
    if svm_cache_mb <= 0:
        raise ValueError("SVM kernel cache must be positive")
    dataset = load_dataset()
    recordings = training_recordings(dataset)
    tables = {representation: dataset.tables(representation) for representation in {config["comparisons"][key]["representation"] for key in keys}}
    pending, subsets, references, identities = [], {}, {}, {}
    for key in keys:
        spec = config["comparisons"][key]
        train, test = tables[spec["representation"]]
        references[key] = verify_cached_inputs(key, dataset, (train, test))
        selected = train[2].recording.isin(recordings)
        X, rows = train[0].loc[selected].reset_index(drop=True), train[2].loc[selected].reset_index(drop=True)
        if set(rows.recording) != set(recordings) or not set(rows.window_id) <= set(train[2].window_id):
            raise ValueError("Replay inputs must be exact retained training rows")
        identity = digest({"reference": references[key]["experiment_id"], "dataset": dataset.identity, "window_ids": rows.window_id.tolist(), "features": list(X)})
        identities[key], subsets[key] = identity, (X, rows)
        path, meta = prediction_paths(key)
        if path.exists() and meta.exists():
            metadata, cached = load_predictions(key)
            if metadata["input_identity"] != identity:
                raise ValueError("Training replay sample support changed")
            np.testing.assert_array_equal(cached.window_id, rows.window_id)
            print(f"Verified cached training replay: {spec['label']}", flush=True)
        else:
            pending.append(key)

    quote = None
    if "tabpfn-du" in pending:
        authentication()
        from tabpfn_client import estimate_cost

        params = config["comparisons"]["tabpfn-du"]["parameters"]
        quote = estimate_cost(tables["Du"][0][0], subsets["tabpfn-du"][0], model_version=params["version"], n_estimators=params["n_estimators"]).model_dump(mode="json")
        atomic_json(ROOT / "artifacts/training-replay/cost_quote.json", {"recordings": recordings, "created_at": utc_now(), **quote})
    estimate = quote["estimated_cost"] if quote else 0
    print(f"Training replay TabPFN estimate: {estimate:,} tokens", flush=True)
    if quote_only:
        return {"estimated_tokens": estimate, "recordings": recordings, "rows": len(next(iter(subsets.values()))[1])}
    if quote and (max_estimated_tokens is None or estimate > max_estimated_tokens):
        raise ValueError(f"Training replay quote is {estimate:,} tokens; supply a sufficient --max-estimated-tokens")
    fetch_training_videos(recordings)
    for key in pending:
        spec, reference = config["comparisons"][key], references[key]
        train, test = tables[spec["representation"]]
        X, rows = subsets[key]
        start = perf_counter()
        with threadpool_limits(limits=1):
            if spec["model"] == "TabPFN":
                from tabpfn_client import TabPFNClassifier

                directory = paths_for(key)[0]
                model = TabPFNClassifier.load_model(directory / "models" / f"{config['name']}_Du.json")
                if str(model.model_id_) != reference["fitted_model_id"]:
                    raise ValueError("Wrong saved TabPFN context")
                scaler = StandardScaler().fit(train[0])
                np.testing.assert_allclose(scaler.mean_, reference["scaler_mean"], rtol=1e-12, atol=1e-10)
                np.testing.assert_allclose(scaler.scale_, reference["scaler_scale"], rtol=1e-12, atol=1e-10)
                probabilities = np.asarray(model.predict_proba(pd.DataFrame(scaler.transform(X), columns=X.columns)), dtype=float)
                if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6):
                    raise ValueError("Invalid training replay probabilities")
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                classes = model.classes_
                predictions = np.asarray(classes)[probabilities.argmax(axis=1)]
                provenance = {"method": "loaded original pooled API context", "model_id": str(model.model_id_)}
            else:
                identity = training_identity(key, dataset, train)
                try:
                    pipeline, provenance = load_classical(key, identity)
                except FileNotFoundError:
                    params = dict(spec["parameters"])
                    if spec["model"] == "SVM":
                        params["cache_size"] = svm_cache_mb  # Memory allocation only; C/gamma and all learning settings match.
                    cls = SVC if spec["model"] == "SVM" else LinearDiscriminantAnalysis
                    pipeline = make_pipeline(StandardScaler(), cls(**params))
                    print(f"Reconstructing pooled {spec['label']} on all {len(train[0]):,} training rows", flush=True)
                    pipeline.fit(train[0], train[1])
                    original_predictions = load_result(key)[2]
                    np.testing.assert_array_equal(pipeline.predict(test[0]), original_predictions.prediction)
                    scaler = pipeline.steps[0][1]
                    np.testing.assert_allclose(scaler.mean_, reference["scaler_mean"], rtol=1e-12, atol=1e-10)
                    np.testing.assert_allclose(scaler.scale_, reference["scaler_scale"], rtol=1e-12, atol=1e-10)
                    provenance = save_classical(key, pipeline, identity, {
                        "method": "reconstructed pooled model with identical held-out predictions",
                        "reference_experiment_id": reference["experiment_id"],
                        "verified_held_out_rows": len(test[0]),
                        "reference_prediction_sha256": reference["prediction_sha256"],
                    })
                    print(f"Verified {len(test[0]):,} identical held-out predictions for {spec['label']}", flush=True)
                predictions = pipeline.predict(X)
                classes = pipeline.classes_
                probabilities = pipeline.predict_proba(X) if spec["model"] == "LDA" else None
            saved = rows[["window_id", "recording", "sample_start", "sample_end", "sample_time", "VIDEO_STAMP", "TRAJ_1", "TRAJ_GT", "label_pure_window"]].copy()
            saved["prediction"] = np.asarray(predictions, dtype=int)
            saved["subject"] = saved.recording.map(lambda name: subjects_of([name])[0])
            if probabilities is not None:
                for label in LABELS:
                    saved[f"probability_{label}"] = 0.0
                for index, label in enumerate(classes):
                    saved[f"probability_{int(label)}"] = probabilities[:, index]
                saved["confidence"] = probabilities.max(axis=1)
        path, metadata_path = prediction_paths(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".parquet.tmp")
        saved.to_parquet(temporary, index=False)
        temporary.replace(path)
        metadata = {
            "created_at": utc_now(), "comparison": key, "split_role": "training",
            "subjects": training_subjects(), "recordings": recordings, "rows": len(saved),
            "input_identity": identities[key], "dataset_id": dataset.identity,
            "reference_experiment_id": reference["experiment_id"],
            "reference_prediction_sha256": reference["prediction_sha256"],
            "prediction_sha256": sha256(path), "model_provenance": provenance,
            "elapsed_seconds": perf_counter() - start,
            "per_recording_metrics": {str(name): score(group.TRAJ_GT, group.prediction) for name, group in saved.groupby("recording", sort=True)},
            "cost_quote": quote if key == "tabpfn-du" else None,
            "interpretation": "In-sample replay: these windows and labels were in the pooled training context. Excluded from held-out reports.",
        }
        atomic_json(metadata_path, metadata)
        print(f"Saved {len(saved):,} training-replay predictions for {spec['label']}", flush=True)
    return {"recordings": recordings, "completed": keys}
