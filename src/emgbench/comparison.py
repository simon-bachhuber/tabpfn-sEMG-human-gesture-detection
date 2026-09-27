"""Run or verify exactly SVM/RMS, LDA/Du, and TabPFN/Du on the fixed split."""

import os
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from dotenv import load_dotenv
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
    environment,
    read_json,
    settings,
    sha256,
    utc_now,
)
from .metrics import score
from .model_cache import save_classical, training_identity
from .protocol import load_dataset, subjects_of


def paths_for(key):
    spec = settings()["comparisons"][key]
    directory = ROOT / "artifacts/runs" / spec["run"]
    result = directory / "folds" / settings()["name"] / f"{spec['representation']}_{spec['model']}.json"
    return directory, result, result.with_suffix(".parquet")


def authentication():
    load_dotenv(ROOT / ".env", override=False)
    token = os.environ.get("API_TOKEN")
    if not token:
        raise ValueError("Set API_TOKEN in .env or the environment")
    os.environ["TABPFN_TOKEN"] = token
    os.environ.setdefault("TABPFN_CLIENT_TIMEOUT", "3600")


def load_result(key):
    """Read immutable predictions and their original provenance, without an API call."""
    spec = settings()["comparisons"][key]
    directory, path, prediction_path = paths_for(key)
    if not path.exists() or not prediction_path.exists():
        raise FileNotFoundError(f"Missing {spec['label']} result; run emgbench compare --models {key}")
    result = read_json(path)
    manifest = read_json(directory / "manifest.json")
    if result["experiment_id"] != manifest["experiment_id"]:
        raise ValueError(f"Result/manifest mismatch: {key}")
    if result["model"] != spec["model"] or result["feature_set"] != spec["representation"]:
        raise ValueError(f"Wrong model/representation in {path}")
    for role in ("train_subjects", "test_subjects"):
        if result["fold"][role] != settings()[role]:
            raise ValueError(f"Changed participant split: {key}/{role}")
    if subjects_of(result["fold"]["train"]) != settings()["train_subjects"]:
        raise ValueError("Training recordings disagree with the 40 training users")
    if subjects_of(result["fold"]["test"]) != settings()["test_subjects"]:
        raise ValueError("Test recordings disagree with the four held-out users")
    if "comparison" in manifest:
        actual = manifest["comparison"]["parameters"]
    elif spec["model"] == "TabPFN":
        actual = manifest["evaluation"]["tabpfn"]
    else:
        actual = manifest["classifiers"][spec["model"]]["args"]
    if actual != spec["parameters"]:
        raise ValueError(f"Cached model parameters differ from the frozen configuration: {key}")
    if sha256(prediction_path) != result["prediction_sha256"]:
        raise ValueError(f"Prediction integrity mismatch: {prediction_path}")
    predictions = pd.read_parquet(prediction_path)
    if len(predictions) != result["test_rows"] or not predictions.window_id.is_unique:
        raise ValueError("Missing or duplicate held-out predictions")
    if set(predictions.recording) != set(result["fold"]["test"]):
        raise ValueError("The result does not cover all 24 test recordings")
    if len(result["feature_columns"]) != 24 * len(settings()["representations"][spec["representation"]]):
        raise ValueError("Incorrect feature count")
    return result, manifest, predictions


def verify_cached_inputs(key, dataset, tables):
    result, manifest, predictions = load_result(key)
    train, test = tables
    if manifest["dataset_id"] != dataset.identity:
        raise ValueError(f"Cached feature dataset changed for {key}")
    for role in ("train", "test"):
        if result["fold"][role] != dataset.split[role]:
            raise ValueError("Changed training/test recording order")
    identity = digest({
        "train": train[2].window_id.tolist(), "test": test[2].window_id.tolist(),
        "features": list(train[0]),
    })
    if result["row_identity"] != identity or result["train_rows"] != len(train[0]):
        raise ValueError("Changed training or test sample support")
    for column in ("window_id", "TRAJ_GT", "VIDEO_STAMP"):
        np.testing.assert_array_equal(predictions[column], test[2][column])
    scaler = StandardScaler().fit(train[0])
    np.testing.assert_allclose(scaler.mean_, result["scaler_mean"], rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(scaler.scale_, result["scaler_scale"], rtol=1e-12, atol=1e-10)
    return result


def quote_prediction(train_x, test_x):
    authentication()
    from tabpfn_client import estimate_cost

    params = settings()["comparisons"]["tabpfn-du"]["parameters"]
    quote = estimate_cost(
        train_x, test_x, model_version=params["version"], n_estimators=params["n_estimators"],
    ).model_dump(mode="json")
    atomic_json(ROOT / "artifacts/quotes/tabpfn-du.json", {"quoted_at": utc_now(), **quote})
    return quote


def fit_comparison(key, dataset, tables, quote=None):
    spec = settings()["comparisons"][key]
    directory, result_path, prediction_path = paths_for(key)
    train, test = tables
    train_x, train_y, train_rows, excluded_train = train
    test_x, test_y, test_rows, excluded_test = test
    row_identity = digest({
        "train": train_rows.window_id.tolist(), "test": test_rows.window_id.tolist(),
        "features": list(train_x),
    })
    manifest = {
        "dataset_id": dataset.identity, "configuration": settings(), "comparison": spec,
        "environment": environment(), "split": dataset.split,
        "source_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
    }
    experiment_id = digest(manifest)
    atomic_json(directory / "manifest.json", {"experiment_id": experiment_id, **manifest})
    print(f"Fitting {spec['label']}: {len(train_x):,} train rows, {train_x.shape[1]} features; {len(test_x):,} test rows", flush=True)
    probabilities, server_timings, fitted_model_id = None, None, None
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        start = perf_counter()
        if spec["model"] == "TabPFN":
            from tabpfn_client import TabPFNClassifier

            params = spec["parameters"]
            scaler = StandardScaler().fit(train_x)
            scaled_train = pd.DataFrame(scaler.transform(train_x), columns=train_x.columns)
            scaled_test = pd.DataFrame(scaler.transform(test_x), columns=test_x.columns)
            estimator = TabPFNClassifier.create_default_for_version(
                params["version"], n_estimators=params["n_estimators"], random_state=params["random_state"],
            )
            estimator.fit(scaled_train, train_y)
            fit_seconds = perf_counter() - start
            atomic_json(directory / "models" / f"{settings()['name']}_Du.json", estimator.save_model())
            fitted_model_id = str(estimator.model_id_)
            start = perf_counter()
            probabilities = np.asarray(estimator.predict_proba(scaled_test), dtype=float)
            predict_seconds = perf_counter() - start
            classes = estimator.classes_
            if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1, rtol=1e-5, atol=1e-7):
                raise ValueError("Invalid TabPFN probabilities")
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            predictions = np.asarray(classes)[probabilities.argmax(axis=1)]
            probability_seconds = predict_seconds
            server_timings = estimator.get_timings()
        else:
            cls = SVC if spec["model"] == "SVM" else LinearDiscriminantAnalysis
            pipeline = make_pipeline(StandardScaler(), cls(**spec["parameters"]))
            pipeline.fit(train_x, train_y)
            fit_seconds = perf_counter() - start
            scaler, estimator = pipeline.steps[0][1], pipeline.steps[1][1]
            classes = estimator.classes_
            start = perf_counter()
            predictions = pipeline.predict(test_x)
            predict_seconds = perf_counter() - start
            probability_seconds = None
            if spec["model"] == "LDA":
                start = perf_counter()
                probabilities = pipeline.predict_proba(test_x)
                probability_seconds = perf_counter() - start
    saved = test_rows[["window_id", "recording", "sample_start", "sample_end", "sample_time", "VIDEO_STAMP", "TRAJ_1", "TRAJ_GT", "label_pure_window"]].copy()
    saved["prediction"] = np.asarray(predictions, dtype=int)
    saved["subject"] = saved.recording.map(lambda name: subjects_of([name])[0])
    if probabilities is not None:
        for label in LABELS:
            saved[f"probability_{label}"] = 0.0
        for index, label in enumerate(classes):
            saved[f"probability_{int(label)}"] = probabilities[:, index]
        saved["confidence"] = probabilities.max(axis=1)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = prediction_path.with_suffix(".parquet.tmp")
    saved.to_parquet(temporary, index=False)
    temporary.replace(prediction_path)
    result = {
        "experiment_id": experiment_id, "created_at": utc_now(), "fold": dataset.split,
        "protocol": settings()["name"], "model": spec["model"], "feature_set": spec["representation"],
        "feature_columns": list(train_x), "row_identity": row_identity,
        "sample_identity": digest({"train": train_rows.window_id.tolist(), "test": test_rows.window_id.tolist()}),
        "train_rows": len(train_x), "test_rows": len(test_x),
        "excluded_train_rows": excluded_train, "excluded_test_rows": excluded_test,
        "training_label_counts": {str(k): int(v) for k, v in train_y.value_counts().items()},
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "metrics": score(test_y, predictions),
        "per_subject_metrics": {str(s): score(rows.TRAJ_GT, rows.prediction) for s, rows in saved.groupby("subject", sort=True)},
        "per_recording_metrics": {str(r): score(rows.TRAJ_GT, rows.prediction) for r, rows in saved.groupby("recording", sort=True)},
        "timings": {"fit_seconds": fit_seconds, "predict_seconds": predict_seconds, "probability_seconds": probability_seconds},
        "warnings": sorted({str(w.message) for w in captured}),
        "prediction_file": str(prediction_path.relative_to(ROOT)), "prediction_sha256": sha256(prediction_path),
        "fitted_model_id": fitted_model_id, "server_timings": server_timings, "cost_quote": quote,
    }
    atomic_json(result_path, result)
    if spec["model"] != "TabPFN":
        save_classical(key, pipeline, training_identity(key, dataset, train), {
            "source": "original comparison fit", "experiment_id": experiment_id,
            "held_out_prediction_sha256": result["prediction_sha256"],
        })
    print(f"Completed {spec['label']}: pooled accuracy={result['metrics']['accuracy']:.4f}", flush=True)
    return result


def compare(keys=None, max_estimated_tokens=None, quote_only=False):
    comparisons = settings()["comparisons"]
    keys = keys or list(comparisons)
    if len(keys) != len(set(keys)) or not set(keys) <= set(comparisons):
        raise ValueError("Choose only svm-rms, lda-du, and tabpfn-du, without duplicates")
    dataset = load_dataset()
    tables = {name: dataset.tables(name) for name in {comparisons[key]["representation"] for key in keys}}
    pending, results = [], {}
    for key in keys:
        directory, result_path, prediction_path = paths_for(key)
        if result_path.exists() and prediction_path.exists():
            results[key] = verify_cached_inputs(key, dataset, tables[comparisons[key]["representation"]])
            print(f"Verified cached {comparisons[key]['label']}", flush=True)
        elif directory.exists() and any(directory.iterdir()):
            raise ValueError(f"Incomplete run at {directory}; inspect it before retrying")
        else:
            pending.append(key)
    quote = None
    if "tabpfn-du" in pending:
        train, test = tables["Du"]
        quote = quote_prediction(train[0], test[0])
    estimate = quote["estimated_cost"] if quote else 0
    print(f"Pending TabPFN prediction estimate: {estimate:,} tokens", flush=True)
    if quote_only:
        return {"estimated_tokens": estimate, "pending": pending}
    if quote and (max_estimated_tokens is None or estimate > max_estimated_tokens):
        raise ValueError(f"Set --max-estimated-tokens to a budget covering the {estimate:,}-token quote")
    for key in pending:
        results[key] = fit_comparison(key, dataset, tables[comparisons[key]["representation"]], quote if key == "tabpfn-du" else None)
    return results
