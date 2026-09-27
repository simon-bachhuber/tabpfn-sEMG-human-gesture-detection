"""Verify and publish the three requested pipelines, reusing their predictions."""

import numpy as np

from .common import ROOT, atomic_json, read_json, settings, utc_now
from .comparison import load_result, verify_cached_inputs
from .metrics import METRICS, summarize
from .protocol import load_dataset


def create_report():
    config = settings()
    dataset = load_dataset()
    tables = {name: dataset.tables(name) for name in config["representations"]}
    rows, provenance = [], []
    reference = None
    for key, spec in config["comparisons"].items():
        verified = verify_cached_inputs(key, dataset, tables[spec["representation"]])
        result, manifest, predictions = load_result(key)
        if reference is not None:
            for column in ("window_id", "TRAJ_GT", "VIDEO_STAMP", "subject", "recording"):
                np.testing.assert_array_equal(predictions[column], reference[column])
        else:
            reference = predictions
        metrics = summarize(predictions)
        for subject, values in metrics["participants"].items():
            for name in METRICS:
                if values[name] != result["per_subject_metrics"][subject][name]:
                    raise ValueError(f"Stored metrics disagree with predictions: {key}/{subject}")
        rows.append({
            "id": key, "label": spec["label"], "model": spec["model"],
            "representation": spec["representation"], "features": len(result["feature_columns"]),
            "train_rows": result["train_rows"], "test_rows": result["test_rows"],
            "metrics": metrics, "timings": result["timings"],
            "fitted_at": result["created_at"], "prediction_sha256": result["prediction_sha256"],
        })
        provenance.append({
            "comparison": key, "parameters": spec["parameters"],
            "experiment_id": result["experiment_id"], "dataset_id": manifest["dataset_id"],
            "row_identity": verified["row_identity"], "prediction_sha256": result["prediction_sha256"],
            "feature_columns": result["feature_columns"],
            "scaler_mean": result["scaler_mean"], "scaler_scale": result["scaler_scale"],
            "environment": manifest.get("environment", manifest.get("provenance", {}).get("environment")),
            "fitted_model_id": result.get("fitted_model_id"),
        })
    raw_inputs = []
    for name, info in dataset.metadata.items():
        sidecar = ROOT / "data/raw/hdf5" / (name + ".hdf5.metadata.json")
        source = read_json(sidecar) if sidecar.exists() else {}
        raw_inputs.append({
            "recording": name, "sha256": info["input_sha256"], "url": source.get("url"),
            "feature_sha256": info["feature_sha256"],
        })
    diagnostic = {
        "optimizer_calls": sum(c["optimizer_calls"] for info in dataset.metadata.values() for c in info["channels"]),
        "optimizer_non_success": sum(c["optimizer_failures"] for info in dataset.metadata.values() for c in info["channels"]),
    }
    report = {
        "generated_at": utc_now(), "configuration": config,
        "dataset_id": dataset.identity, "recordings": 264, "fits_per_pipeline": 1,
        "identical_sample_support_verified": True, "training_only_scalers_verified": True,
        "results": rows, "filter_diagnostics": diagnostic,
    }
    output = ROOT / "results"
    atomic_json(output / "comparison.json", report)
    atomic_json(output / "provenance.json", {
        "configuration": config, "split": dataset.split,
        "inputs": raw_inputs, "runs": provenance,
    })
    table = [
        "| Pipeline | Features | Mean accuracy | Macro-F1 | Macro recall | Fit | Predict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        m, t = row["metrics"]["participant_mean"], row["timings"]
        table.append(f"| {row['label']} | {row['features']} | {100*m['accuracy']:.2f}% | {m['macro_f1']:.4f} | {m['macro_recall']:.4f} | {t['fit_seconds']:.2f}s | {t['predict_seconds']:.2f}s |")
    people = [
        "| Held-out user | SVM + RMS accuracy | LDA + Du accuracy | TabPFN + Du accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for subject in config["test_subjects"]:
        values = [row["metrics"]["participants"][subject]["accuracy"] for row in rows]
        people.append(f"| {subject} | " + " | ".join(f"{100*value:.2f}%" for value in values) + " |")
    text = f"""# Three-way putEMG comparison: fixed 40/4 participant split

Generated {report['generated_at']}.

## Protocol

- Train on all retained windows from **40 participants / 80 sessions / 240 recordings**.
- Hold out **{', '.join(config['test_subjects'])}**, including both days: **24 test recordings**.
- **{rows[0]['train_rows']:,} training windows**, **{rows[0]['test_rows']:,} test windows**, zero target-user calibration.
- One pooled fit per pipeline. All three pipelines use identical window IDs, labels and exclusions.
- Primary scores average the four participant-level metrics equally, pooling both days within each person.
- SVM + RMS is the authors' highest-mean-accuracy published pipeline; LDA + Du is their strong Du comparator. Their paper used within-person/day evaluation. These results use the stated unseen-user split.

## Results

{chr(10).join(table)}

Classical models run locally. TabPFN times include API/network overhead. Common signal preprocessing is excluded; extra LDA probability calculation is recorded separately in JSON.

{chr(10).join(people)}

This compares model–representation pipelines. The LDA/TabPFN pair holds Du features fixed; SVM uses RMS. Four held-out people give one fixed evaluation, not a population-wide superiority estimate. Per-gesture metrics, confusion matrices and per-recording scores are in `comparison.json`.

## Reproducibility

The package configuration records participant IDs, all model parameters and upstream source pins. `provenance.json` records source URLs/hashes, fitted scalers, input column order and prediction hashes. Cached results retain their original experiment identity rather than being relabeled as new fits.

The authors-compatible filter uses harmonic subtraction and zero-phase 20–700 Hz filtering. Windows contain 2,500 samples with a 1,250-sample hop. Targets are endpoint `TRAJ_GT`; the original transition mask is applied per recording. The released signed WL/WAMP definitions are retained. Idle is class 0; excluded/unconstrained -1 labels are not learned as idle.

Across preprocessing, {diagnostic['optimizer_non_success']} of {diagnostic['optimizer_calls']:,} harmonic fits reported non-success. Their returned solutions are retained exactly as in the source algorithm; model-input features are finite.

## Demo

The demo shows the four held-out users plus four explicitly labeled training users and exactly these three pipelines. Training replay scores are in-sample and excluded from this report. Score cards describe the selected recording. It replays recorded video with `VIDEO_STAMP` alignment and cached predictions; offline filtering is noncausal. SVM preserves the authors' `probability=False` setting and therefore does not provide probabilities.

Restore the committed cache with `emgbench snapshot restore`, then run `emgbench demo`. See [the reproduction guide](../docs/reproduction.md) for verification, training-replay preparation, and complete regeneration commands.
"""
    (output / "comparison.md").write_text(text)
    print("\n".join(table + [""] + people))
    return report
